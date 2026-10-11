"""RadSpeed Web Server — FastAPI application.

Launch via:
    python RadSpeed.py --web [--host 0.0.0.0] [--port 8765]

Authentication: HTTP Basic Auth.
Password is read from the VOXRAD_WEB_PASSWORD environment variable
(default: "voxrad" — change before any non-localhost deployment).

WARNING: HTTP Basic Auth sends credentials in cleartext over plain HTTP.
Always run behind an HTTPS reverse proxy (e.g. nginx with TLS) in
production. See docs/web-server-setup.md.
"""

import asyncio
import base64
import contextlib
import difflib
import json
import logging
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, OpenAI
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from config.config import config
from config.settings import save_web_settings
from llm.dicom_sr_export import save_dicom_sr_report
from llm.disclosure import append_disclosure
from llm.fhir_export import save_fhir_report
from llm.hl7_export import save_hl7_report
from llm.hl7_import import archive_order, list_inbox
from llm.format import (
    TextModelAuthError,
    TextModelUnavailableError,
    apply_report_feedback,
    reinsert_identifiers,
    format_text,
    join_template,
    number_long_impression_body,
    correct_mri_platform,
    postprocess_report,
    split_template,
    stream_format_text,
)
from llm.fracture_analysis import (
    MAX_FRACTURE_IMAGE_BYTES,
    MAX_FRACTURE_IMAGES,
    FractureImageError,
    analyse_fracture_images,
    mock_fracture_assessment,
    prepare_fracture_images,
)
from llm.chest_open_model import score_chest_images
from llm.fracture_locator import (
    locate_fracture_candidates,
    locate_wrist_fracture_candidates,
)
from llm.strong_fracture_model import score_strong_fracture_images
from llm.impressions import extract_findings, replace_impression, stream_impression
from llm.model_compat import completion_options, reasoning_effort_for_model
from llm.text_client import get_text_client
from config import practice
from llm.worksheet import (
    MAX_WORKSHEET_IMAGE_BYTES,
    MAX_WORKSHEET_IMAGES,
    WorksheetImageError,
    draft_worksheet_report,
    extract_worksheet_findings,
    validate_worksheet_images,
)
from web.auth_oauth import (
    exchange_google_code,
    exchange_microsoft_code,
    get_or_create_user,
    get_user_style,
    google_auth_url,
    google_enabled,
    get_session_user,
    init_db,
    microsoft_auth_url,
    microsoft_enabled,
    oauth_enabled,
    require_oauth_user,
    save_user_style,
    SESSION_SECRET_KEY,
    set_session_user,
    clear_session,
)
from web.audit import (
    get_report,
    init_audit_db,
    list_audit_events,
    list_reports_for_accession,
    list_prior_reports_for_patient,
    list_reports_for_user,
    log_event,
    save_report_version,
    verify_chain,
)
from web.followups import (
    create_followup,
    init_followup_db,
    list_followups,
    suggest_followups,
    update_followup,
)
from web.quality_samples import (
    init_quality_samples,
    save_quality_sample,
    set_quality_retention,
)
from web.fracture_workbench import resolve_workbench_image
from web import retention
from web import adrenal
from web import fleischner
from web import nodule_growth
from web import report_templates as report_library
from web import tirads
from web.qa import run_qa_checks
from web.stt_providers.factory import (
    get_streaming_provider,
    resolve_streaming_provider_name,
)

logger = logging.getLogger(__name__)

_PUBLIC_HEAD_PATHS = {
    "/",
    "/impressions",
    "/radiology-reporting-software",
    "/powerscribe-companion",
    "/ti-rads-calculator",
    "/fleischner-calculator",
    "/adrenal-washout-calculator",
    "/volume-doubling-time-calculator",
    "/report-templates",
    "/robots.txt",
    "/sitemap.xml",
    "/llms.txt",
    "/googleb219a940c12a9cd3.html",
}


class HeadRequestMiddleware:
    """Serve HEAD for public discovery pages without widening API methods."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        is_public_page = path in _PUBLIC_HEAD_PATHS or path.startswith("/report-templates/")
        if scope["type"] != "http" or scope.get("method") != "HEAD" or not is_public_page:
            await self.app(scope, receive, send)
            return

        get_scope = dict(scope)
        get_scope["method"] = "GET"

        async def send_headers_only(message):
            if message["type"] == "http.response.body":
                message = {**message, "body": b""}
            await send(message)

        await self.app(get_scope, receive, send_headers_only)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

async def _retention_loop() -> None:
    """Run the retention purge on a schedule while the app is up."""
    interval = max(300, practice.settings.retention_interval_seconds)
    while True:
        try:
            await asyncio.get_running_loop().run_in_executor(None, retention.run_once)
        except Exception as exc:  # never let a purge failure kill the loop
            logger.warning("[retention] scheduled run failed: %s", exc)
        await asyncio.sleep(interval)


async def _session_cleanup_loop() -> None:
    """Remove expired in-memory transcripts within one minute of expiry."""
    while True:
        _prune_sessions()
        await asyncio.sleep(60)


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Quality samples have a fixed 365-day limit even when the optional
    # patient-report/outbox policy is disabled, so cleanup always runs.
    task = asyncio.create_task(_retention_loop())
    session_task = asyncio.create_task(_session_cleanup_loop())
    if retention.enabled():
        logger.info(
            "[retention] enabled: reports %sd, outboxes %sd, every %ss",
            practice.settings.retention_days,
            practice.settings.outbox_retention_days,
            practice.settings.retention_interval_seconds,
        )
    try:
        yield
    finally:
        task.cancel()
        session_task.cancel()


app = FastAPI(title="RadSpeed Web", docs_url=None, redoc_url=None, lifespan=_lifespan)


def _cookie_secure() -> bool:
    """Mark the session cookie Secure when the site is served over HTTPS.

    Explicit RADSPEED_COOKIE_SECURE wins; otherwise infer from the OAuth
    redirect base URL so a plain-HTTP local install keeps working.
    """
    if practice.settings.cookie_secure is not None:
        return practice.settings.cookie_secure
    base_url = getattr(config, "oauth_redirect_base_url", "") or ""
    return base_url.lower().startswith("https://")


class PrivacyHeadersMiddleware:
    """Cache and origin hygiene for everything that is not a public page.

    * ``Cache-Control: no-store`` on authenticated pages and API responses so
      report text and patient identifiers never sit in a browser or proxy
      cache. Static assets and public marketing pages are left cacheable.
    * With RADSPEED_STRICT_ORIGIN, state-changing requests whose ``Origin``
      header names another site are rejected. Browsers always send Origin on
      cross-site POSTs, so this closes cross-site request forgery for the JSON
      API without adding a token to every form.
    """

    _PUBLIC_PREFIXES = ("/static/", "/report-templates", "/impressions")
    _PUBLIC_EXACT = {
        "/", "/login", "/health", "/favicon.ico", "/robots.txt", "/sitemap.xml",
        "/llms.txt", "/privacy", "/terms", "/radiology-reporting-software", "/powerscribe-companion",
        "/ti-rads-calculator", "/fleischner-calculator", "/adrenal-washout-calculator",
        "/volume-doubling-time-calculator",
    }

    def __init__(self, app):
        self.app = app

    def _is_public(self, path: str) -> bool:
        return path in self._PUBLIC_EXACT or path.startswith(self._PUBLIC_PREFIXES)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}

        if practice.settings.strict_origin and method in ("POST", "PUT", "PATCH", "DELETE"):
            origin = headers.get("origin")
            if origin and not _origin_allowed(origin, headers.get("host", "")):
                response = JSONResponse({"detail": "Cross-site request rejected"}, status_code=403)
                await response(scope, receive, send)
                return

        if self._is_public(path):
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                raw = list(message.get("headers", []))
                if not any(k.lower() == b"cache-control" for k, _ in raw):
                    raw.append((b"cache-control", b"no-store"))
                message = {**message, "headers": raw}
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _origin_allowed(origin: str, host_header: str) -> bool:
    from urllib.parse import urlparse
    origin_host = (urlparse(origin).hostname or "").lower()
    if not origin_host:
        return False
    allowed = {(host_header.split(":")[0] or "").lower()}
    base = (getattr(config, "oauth_redirect_base_url", "") or "").strip()
    if base:
        allowed.add((urlparse(base).hostname or "").lower())
    allowed.discard("")
    return origin_host in allowed


app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY(),
    max_age=practice.settings.session_max_age_seconds,
    https_only=_cookie_secure(),
)
app.add_middleware(HeadRequestMiddleware)
app.add_middleware(PrivacyHeadersMiddleware)
# auto_error=False so we can return a redirect (not a 401) when OAuth is active
security = HTTPBasic(auto_error=False)

# Initialise user + audit databases on startup (noop if already created)
init_db()
init_audit_db()
init_followup_db()
init_quality_samples()


def _validate_deployment_profile() -> None:
    """Check the configured profile is internally consistent.

    A practice deployment stops here rather than start with a hole in it,
    because the whole point of the profile is that a misconfiguration cannot
    quietly route patient data offshore or drop back to a shared password.
    Personal deployments only log the findings.
    """
    if os.environ.get("VOXRAD_MOCK_MODE"):
        return
    try:
        oauth_configured = bool(oauth_enabled())
    except Exception:  # config may be a partial stub under test
        oauth_configured = False
    problems = practice.validate(
        practice.settings,
        oauth_configured=oauth_configured,
        text_base_url=getattr(config, "BASE_URL", None),
        transcription_base_url=getattr(config, "TRANSCRIPTION_BASE_URL", None),
        streaming_provider=resolve_streaming_provider_name(),
        deepgram_key=bool(getattr(config, "DEEPGRAM_API_KEY", None)),
    )
    for problem in problems:
        logger.error("[profile] %s", problem)
    if problems and practice.settings.is_practice:
        raise RuntimeError(
            "RadSpeed practice profile refused to start: " + " | ".join(problems)
        )
    logger.info(
        "[profile] %s profile, data_residency=%s, require_sso=%s, research_features=%s, "
        "retention_days=%s, text_provider=%s, deepgram_region=%s",
        practice.settings.profile, practice.settings.data_residency or "none",
        practice.settings.require_sso, practice.settings.research_features,
        practice.settings.retention_days, practice.settings.text_provider,
        practice.settings.deepgram_region,
    )


_validate_deployment_profile()

_BASE_DIR = os.path.dirname(__file__)
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(_BASE_DIR, "static")),
    name="static",
)
_jinja = Jinja2Templates(directory=os.path.join(_BASE_DIR, "templates"))
_FRACTURE_WORKBENCH_PAGE = Path(_BASE_DIR) / "private" / "fracture_workbench.html"
_fracture_analysis_slot = asyncio.Semaphore(1)

# Cache-busting: use the current git commit hash (or a timestamp fallback)
# so browsers always load fresh JS/CSS after each deploy.
try:
    import subprocess
    _STATIC_VERSION = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=os.path.dirname(_BASE_DIR),
        stderr=subprocess.DEVNULL,
    ).decode().strip()
except Exception:
    _STATIC_VERSION = str(int(time.time()))

# ---------------------------------------------------------------------------
# Authentication — OAuth (primary) or HTTP Basic Auth (fallback)
# ---------------------------------------------------------------------------

_DEFAULT_WEB_PASSWORD = "voxrad"


def _verify_auth(
    request: Request,
    credentials: Optional[HTTPBasicCredentials] = Depends(security),
) -> dict:
    """Unified auth dependency.

    OAuth mode  — reads the session; returns 307 → /login if not signed in.
    Basic Auth mode — validates the shared password; returns 401 if wrong.
    """
    if oauth_enabled():
        return require_oauth_user(request)

    if practice.settings.require_sso:
        # A practice deployment must never fall back to the shared password:
        # it has no per-user identity, so sign-off, ownership checks and the
        # audit trail would all lose their meaning.
        raise HTTPException(
            status_code=503,
            detail="Single sign-on is required by this deployment but no OAuth client is configured.",
        )

    # Basic Auth mode
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )
    client_ip = request.client.host if request.client else "unknown"
    if _basic_auth_locked(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many failed sign-in attempts. Try again in a minute.",
        )
    expected = os.environ.get("VOXRAD_WEB_PASSWORD", _DEFAULT_WEB_PASSWORD)
    ok = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        expected.encode("utf-8"),
    )
    if not ok:
        _basic_auth_record_failure(client_ip)
        raise HTTPException(
            status_code=401,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Basic"},
        )
    _basic_auth_failures.pop(client_ip, None)
    return {"id": None, "email": credentials.username, "name": credentials.username}


# Basic Auth brute-force guard: per-IP failure counter with a short lockout.
_basic_auth_failures: dict[str, list[float]] = {}
_BASIC_AUTH_MAX_FAILURES = 10
_BASIC_AUTH_WINDOW = 60.0


def _basic_auth_record_failure(client_ip: str) -> None:
    now = time.time()
    attempts = [t for t in _basic_auth_failures.get(client_ip, []) if now - t < _BASIC_AUTH_WINDOW]
    attempts.append(now)
    _basic_auth_failures[client_ip] = attempts


def _basic_auth_locked(client_ip: str) -> bool:
    now = time.time()
    attempts = [t for t in _basic_auth_failures.get(client_ip, []) if now - t < _BASIC_AUTH_WINDOW]
    if attempts:
        _basic_auth_failures[client_ip] = attempts
    else:
        _basic_auth_failures.pop(client_ip, None)
    return len(attempts) >= _BASIC_AUTH_MAX_FAILURES


def _require_research_feature() -> None:
    """Gate features that generate clinical content beyond formatting.

    The Impression generator and Fracture Lab produce diagnostic content the
    radiologist did not dictate. Under the TGA's software guidance that puts
    them in medical-device territory, so a practice deployment keeps them off
    (RADSPEED_RESEARCH_FEATURES=false) and the routes answer 404.
    """
    if not practice.settings.research_features:
        raise HTTPException(status_code=404, detail="This feature is not enabled on this deployment.")


def _get_username(user: dict) -> str:
    return user.get("name") or user.get("email") or "user"


def _is_admin(user: dict) -> bool:
    """True when the caller may change server-wide settings.

    In Basic Auth mode the single shared operator is trusted. In OAuth mode
    emails in RADSPEED_ADMIN_EMAILS (comma-separated) may mutate global process
    config — model IDs, base URLs, streaming provider, HL7 paths. When no
    allow-list has been configured, the first registered account is treated as
    the owner so a single-user installation does not silently lock its owner
    out of settings. Every other user can still save per-user style prefs.

    This closes a privilege hole: any logged-in user could otherwise point
    config.BASE_URL at an attacker server and exfiltrate the shared LLM API key
    and PHI on the next format/transcribe call by any user.
    """
    if not oauth_enabled():
        return True
    allow = os.environ.get("RADSPEED_ADMIN_EMAILS", "")
    emails = {e.strip().lower() for e in allow.split(",") if e.strip()}
    email = (user.get("email") or "").lower()
    if emails:
        return bool(email) and email in emails
    return user.get("id") == 1


def _user_style(user: dict) -> Optional[dict]:
    """Return per-user style dict in OAuth mode; None (→ global config) in Basic Auth mode."""
    if oauth_enabled() and user.get("id") is not None:
        return get_user_style(user["id"])
    return None


def _user_fhir_enabled(user: dict) -> bool:
    """Per-user FHIR export toggle in OAuth mode; global config in Basic Auth mode."""
    if oauth_enabled() and user.get("id") is not None:
        return get_user_style(user["id"]).get("fhir_export_enabled", False)
    return config.fhir_export_enabled


# ---------------------------------------------------------------------------
# Session cache  {session_id: {transcription, expires_at}}
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}
_SESSION_TTL = 1800  # 30 minutes


def _create_session(transcription: str) -> str:
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "transcription": transcription,
        "expires_at": time.time() + _SESSION_TTL,
    }
    return sid


def _get_session(sid: str) -> Optional[str]:
    s = _sessions.get(sid)
    if s and s["expires_at"] > time.time():
        return s["transcription"]
    _sessions.pop(sid, None)
    return None


def _prune_sessions() -> None:
    """Remove expired sessions (called lazily on each transcription)."""
    now = time.time()
    expired = [k for k, v in _sessions.items() if v["expires_at"] <= now]
    for k in expired:
        del _sessions[k]


# ---------------------------------------------------------------------------
# Thread lock for format_text() — protects config.global_md_text_content
# mutation from concurrent requests.
# ---------------------------------------------------------------------------

_format_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helper: list available templates
# ---------------------------------------------------------------------------

_BUNDLED_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")


def _list_templates() -> list[str]:
    template_dir = os.path.join(config.save_directory or "", "templates")
    names: set[str] = set()
    for d in (template_dir, _BUNDLED_TEMPLATES_DIR):
        if os.path.isdir(d):
            names.update(
                f for f in os.listdir(d)
                if f.endswith((".txt", ".md")) and not f.startswith("_")
            )
    # Pin Plain_Prose.txt to the top so the unstructured option is discoverable
    # without scrolling past the alphabetical list of structured templates.
    ordered = sorted(names)
    for pinned in ("Plain_Prose.txt",):
        if pinned in ordered:
            ordered.remove(pinned)
            ordered.insert(0, pinned)
    return ordered


def _load_template_content(template_name: str) -> str:
    if not template_name:
        return ""
    # Guard against path traversal / absolute paths. template_name arrives from
    # a free-form request body (FormatRequest.template_name); without this an
    # authenticated user could pass "../../data/session_secret.key" or an
    # absolute path and have its contents loaded into the LLM system prompt.
    if os.path.basename(template_name) != template_name:
        logger.warning("Rejected template name with path separators: %r", template_name)
        return ""
    for d in (
        os.path.join(config.save_directory or "", "templates"),
        _BUNDLED_TEMPLATES_DIR,
    ):
        if not d:
            continue
        base = os.path.realpath(d)
        path = os.path.realpath(os.path.join(base, template_name))
        # Ensure the resolved path stays inside the templates directory.
        if path != base and not path.startswith(base + os.sep):
            continue
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return RedirectResponse("/static/favicon.svg", status_code=308)


@app.get("/googleb219a940c12a9cd3.html", response_class=PlainTextResponse, include_in_schema=False)
def google_site_verification():
    """Keep Search Console ownership verifiable from the public site root."""
    return "google-site-verification: googleb219a940c12a9cd3.html"


@app.get("/robots.txt", response_class=PlainTextResponse, include_in_schema=False)
def robots():
    return """User-agent: *
Allow: /
Disallow: /api/
Disallow: /app
Disallow: /auth/
Disallow: /login
Disallow: /settings

Sitemap: https://radspeed.com.au/sitemap.xml
"""


@app.get("/sitemap.xml", response_class=PlainTextResponse, include_in_schema=False)
def sitemap():
    content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://radspeed.com.au/</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>
  <url><loc>https://radspeed.com.au/radiology-reporting-software</loc><changefreq>weekly</changefreq><priority>0.9</priority></url>
  <url><loc>https://radspeed.com.au/powerscribe-companion</loc><changefreq>weekly</changefreq><priority>0.9</priority></url>
  <url><loc>https://radspeed.com.au/impressions</loc><changefreq>weekly</changefreq><priority>0.9</priority></url>
  <url><loc>https://radspeed.com.au/ti-rads-calculator</loc><changefreq>monthly</changefreq><priority>0.8</priority></url>
  <url><loc>https://radspeed.com.au/fleischner-calculator</loc><changefreq>monthly</changefreq><priority>0.8</priority></url>
  <url><loc>https://radspeed.com.au/adrenal-washout-calculator</loc><changefreq>monthly</changefreq><priority>0.8</priority></url>
  <url><loc>https://radspeed.com.au/volume-doubling-time-calculator</loc><changefreq>monthly</changefreq><priority>0.8</priority></url>
  <url><loc>https://radspeed.com.au/report-templates</loc><changefreq>weekly</changefreq><priority>0.8</priority></url>
  <url><loc>https://radspeed.com.au/privacy</loc><changefreq>yearly</changefreq><priority>0.3</priority></url>
  <url><loc>https://radspeed.com.au/terms</loc><changefreq>yearly</changefreq><priority>0.3</priority></url>
{template_urls}</urlset>
"""
    template_urls = "".join(
        f'  <url><loc>https://radspeed.com.au/report-templates/{slug}</loc>'
        f"<changefreq>monthly</changefreq><priority>0.6</priority></url>\n"
        for slug in report_library.all_slugs()
    )
    content = content.format(template_urls=template_urls)
    return PlainTextResponse(content, media_type="application/xml")


@app.get("/llms.txt", response_class=PlainTextResponse, include_in_schema=False)
def llms_txt():
    return """# RadSpeed

> Radiology reporting software built for radiologists. Public pages use synthetic examples only.

## Public pages

- [RadSpeed](https://radspeed.com.au/): Product overview, workflow, governance, and current limits.
- [Radiology reporting software](https://radspeed.com.au/radiology-reporting-software): Practical selection guide and RadSpeed workflow for AU/NZ reporting rooms.
- [PowerScribe companion](https://radspeed.com.au/powerscribe-companion): Windows companion workflow for drafting and pasting an impression beside an existing reporting system.
- [Impressions](https://radspeed.com.au/impressions): Free radiology impression drafting tool. It is assistive software, not a diagnostic device.
- [TI-RADS calculator](https://radspeed.com.au/ti-rads-calculator): Free ACR TI-RADS 2017 thyroid nodule score with FNA and follow-up thresholds, a prior-study growth check against the ACR enlargement rule, and a paste-ready report line. Decision support, not a diagnostic device.
- [Fleischner calculator](https://radspeed.com.au/fleischner-calculator): Free Fleischner Society 2017 incidental pulmonary nodule follow-up recommendation for solid and subsolid nodules. Averages the long and short axis into the guideline mean diameter with whole-millimetre rounding, compares the prior study against the 2 mm diameter and 25% volume growth thresholds, and returns a paste-ready report line. Decision support, not a diagnostic device.
- [Adrenal washout calculator](https://radspeed.com.au/adrenal-washout-calculator): Free adrenal CT washout calculator for an incidental adrenal nodule — computes absolute and relative percentage washout from the unenhanced, portal-venous and delayed attenuation, with a paste-ready report line. Decision support, not a diagnostic device.
- [Volume doubling time calculator](https://radspeed.com.au/volume-doubling-time-calculator): Free lung nodule volume doubling time calculator — compares two CT studies by volume (or mean diameter), returns the percentage volume change and VDT, and reads the result against the Australian NLCSP growth definitions and the BTS 2015 400- and 600-day thresholds, with a paste-ready report line. Decision support, not a diagnostic device.
- [Report templates](https://radspeed.com.au/report-templates): Free library of structured report templates for CT, MRI, ultrasound, X-ray and nuclear medicine, with synthetic sample impressions.

## Important limits

- Do not infer diagnostic performance or clinical validation.
- Do not submit patient information or real clinical text to automated research tools.
"""


# ---------------------------------------------------------------------------
# Public Impressions wedge tool — free, no auth, IP rate-limited.
# ---------------------------------------------------------------------------

# Simple in-memory sliding-window rate limiter keyed by client IP.
# Process-local; sufficient for the single-instance free tier. Replace with
# Redis or a CDN/edge limiter when we scale beyond one app server.
_IMPRESSIONS_RATE_LIMIT = int(os.environ.get("RADSPEED_IMPRESSIONS_HOURLY_LIMIT", "20"))
_IMPRESSIONS_WINDOW_SEC = 3600
_impressions_hits: dict[str, list[float]] = {}
_impressions_lock = threading.Lock()


def _impressions_client_ip(request: Request) -> str:
    """Best-effort real client IP for rate limiting.

    The leftmost X-Forwarded-For entry is fully client-controlled, so keying
    the limiter on it lets an attacker rotate a fake header per request and
    bypass the limit entirely. Fly's edge sets Fly-Client-IP to the real peer
    address; prefer it. Otherwise take the *rightmost* XFF entry (the hop
    appended by our own trusted proxy), never the leftmost.
    """
    fly_ip = request.headers.get("fly-client-ip", "").strip()
    if fly_ip:
        return fly_ip
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        parts = [p.strip() for p in fwd.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return request.client.host if request.client else "unknown"


def _impressions_rate_check(ip: str) -> tuple[bool, int]:
    """Return (allowed, remaining). Mutates the hit log on allow."""
    now = time.time()
    cutoff = now - _IMPRESSIONS_WINDOW_SEC
    with _impressions_lock:
        hits = _impressions_hits.get(ip, [])
        hits = [t for t in hits if t > cutoff]
        if len(hits) >= _IMPRESSIONS_RATE_LIMIT:
            _impressions_hits[ip] = hits
            return False, 0
        hits.append(now)
        _impressions_hits[ip] = hits
        return True, _IMPRESSIONS_RATE_LIMIT - len(hits)


@app.get("/impressions", include_in_schema=False, dependencies=[Depends(_require_research_feature)])
def impressions_page(request: Request):
    return _jinja.TemplateResponse(
        request,
        "impressions.html",
        {
            "request": request,
            "static_version": _STATIC_VERSION,
        },
    )


class ImpressionsRequest(BaseModel):
    findings: str
    modality: Optional[str] = None
    with_guidelines: bool = False
    style: Optional[dict] = None


class ReportImpressionRequest(BaseModel):
    report: str
    modality: Optional[str] = None
    with_guidelines: bool = True


@app.post("/api/impressions/stream", dependencies=[Depends(_require_research_feature)])
def api_impressions_stream(req: ImpressionsRequest, request: Request):
    """Stream a guideline-aware radiology impression from the supplied findings.

    Public endpoint (no auth) — rate-limited per client IP.
    SSE frames:
      data: {"token": "..."}     — text chunk
      data: {"done": true}       — completion
      data: {"error": "..."}     — error
    """
    findings = (req.findings or "").strip()
    if not findings:
        raise HTTPException(status_code=400, detail="Findings are required.")
    if len(findings) > 8000:
        raise HTTPException(
            status_code=413,
            detail="Findings too long. Please trim to under 8000 characters.",
        )

    ip = _impressions_client_ip(request)
    allowed, remaining = _impressions_rate_check(ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Hourly limit of {_IMPRESSIONS_RATE_LIMIT} impressions reached. "
                "Try again later or sign in to RadSpeed for unlimited use."
            ),
        )

    if not config.TEXT_API_KEY:
        def _err():
            yield 'data: {"error": "Text model is not configured on this server."}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    def _generate():
        full_impression = ""
        try:
            for chunk in stream_impression(
                findings=findings,
                modality=req.modality,
                style=req.style,
                with_guidelines=bool(req.with_guidelines),
            ):
                if chunk:
                    full_impression += chunk
                    yield f'data: {json.dumps({"token": chunk})}\n\n'
        except Exception as e:
            logger.error("Impressions stream error: %s", e, exc_info=True)
            yield f'data: {json.dumps({"error": str(e)})}\n\n'
            return
        corrected = number_long_impression_body(full_impression.strip())
        yield f'data: {json.dumps({"done": True, "report": corrected})}\n\n'

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"X-RadSpeed-Remaining": str(remaining)},
    )


@app.post("/api/impressions/text", response_class=PlainTextResponse, dependencies=[Depends(_require_research_feature)])
def api_impressions_text(req: ImpressionsRequest, request: Request):
    """Non-streaming impression generation — returns plain text.

    Designed for the Windows desktop helper (AutoHotkey / Tauri) where a
    single HTTP POST with a plain-text body is much simpler than parsing
    SSE. Same validation and rate limiting as the streaming endpoint.
    """
    findings = (req.findings or "").strip()
    if not findings:
        raise HTTPException(status_code=400, detail="Findings are required.")
    if len(findings) > 8000:
        raise HTTPException(
            status_code=413,
            detail="Findings too long. Please trim to under 8000 characters.",
        )

    ip = _impressions_client_ip(request)
    allowed, remaining = _impressions_rate_check(ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Hourly limit of {_IMPRESSIONS_RATE_LIMIT} impressions reached. "
                "Try again later or sign in to RadSpeed for unlimited use."
            ),
        )

    if not config.TEXT_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Text model is not configured on this server.",
        )

    try:
        chunks = list(stream_impression(
            findings=findings,
            modality=req.modality,
            style=req.style,
            with_guidelines=bool(req.with_guidelines),
        ))
    except Exception as e:
        logger.error("Impressions text generation error: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}")

    text = number_long_impression_body("".join(chunks).strip())
    return PlainTextResponse(
        text,
        headers={"X-RadSpeed-Remaining": str(remaining)},
    )


@app.post("/api/report/impression", dependencies=[Depends(_require_research_feature)])
def api_report_impression(
    req: ReportImpressionRequest,
    user: dict = Depends(_verify_auth),
):
    """Regenerate only the impression in an existing unsigned report."""
    report = (req.report or "").strip()
    if not report:
        raise HTTPException(status_code=400, detail="Report is required.")
    if len(report) > 30000:
        raise HTTPException(status_code=413, detail="Report is too long.")

    findings = extract_findings(report)
    if not findings:
        raise HTTPException(
            status_code=400,
            detail="The report does not contain a FINDINGS section.",
        )

    if _MOCK_MODE:
        impression = "- Mock guideline-aware impression."
    else:
        if not config.TEXT_API_KEY:
            raise HTTPException(
                status_code=503,
                detail="Text model is not configured on this server.",
            )
        try:
            impression = "".join(stream_impression(
                findings=findings,
                modality=req.modality,
                style=_user_style(user),
                with_guidelines=bool(req.with_guidelines),
            )).strip()
        except Exception as exc:
            logger.error("Report impression generation error: %s", exc, exc_info=True)
            raise HTTPException(status_code=502, detail="Impression generation failed.")

    if not impression:
        raise HTTPException(status_code=502, detail="No impression was generated.")

    impression = number_long_impression_body(impression)
    updated_report = replace_impression(report, impression)
    log_event(
        user_id=user.get("id"),
        event_type="impression_refresh",
        metadata={"guidelines": bool(req.with_guidelines)},
    )
    return {"report": updated_report, "impression": impression}


# ---------------------------------------------------------------------------
# Public ACR TI-RADS calculator — free, no auth, deterministic (no model call,
# no patient data stored). Scoring lives in web/tirads.py.
# ---------------------------------------------------------------------------

class TiradsRequest(BaseModel):
    composition: str
    echogenicity: str
    shape: str
    margin: str
    foci: list[str] = []
    size_mm: Optional[float] = None
    location: Optional[str] = None
    dims_mm: list[Optional[float]] = []
    prior_dims_mm: list[Optional[float]] = []
    prior_level: Optional[str] = None


@app.post("/api/tirads/score")
def api_tirads_score(req: TiradsRequest):
    """Return the ACR TI-RADS points, level and a paste-ready report line.

    Pure arithmetic on ultrasound feature selections — no external model call
    and nothing is persisted, so no rate limiting is needed. A free-text
    location is echoed only into the report line the caller already sees.
    """
    location = (req.location or "").strip()[:80] or None
    try:
        return tirads.score(
            composition=req.composition,
            echogenicity=req.echogenicity,
            shape=req.shape,
            margin=req.margin,
            foci=req.foci,
            size_mm=req.size_mm,
            location=location,
            dims_mm=req.dims_mm,
            prior_dims_mm=req.prior_dims_mm,
            prior_level=req.prior_level or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Public Fleischner 2017 pulmonary nodule calculator — free, no auth,
# deterministic (no model call, no patient data stored). Rules live in
# web/fleischner.py.
# ---------------------------------------------------------------------------

class FleischnerRequest(BaseModel):
    nodule_type: str
    size_mm: Optional[float] = None
    multiple: bool = False
    risk: str = "low"
    solid_component_mm: Optional[float] = None
    location: Optional[str] = None
    # Supplying both axes lets web/fleischner.py apply the guideline mean and
    # rounding, so the browser never reimplements the measurement rule.
    long_axis_mm: Optional[float] = None
    short_axis_mm: Optional[float] = None
    # Optional interval comparison against the prior study.
    prior_size_mm: Optional[float] = None
    interval_months: Optional[float] = None
    volume_mm3: Optional[float] = None
    prior_volume_mm3: Optional[float] = None


@app.post("/api/fleischner/recommend")
def api_fleischner_recommend(req: FleischnerRequest):
    """Return the Fleischner 2017 follow-up recommendation and a report line.

    Pure logic on the nodule descriptors — no external model call and nothing is
    persisted, so no rate limiting is needed. A free-text location is echoed only
    into the report line the caller already sees.
    """
    location = (req.location or "").strip()[:80] or None
    try:
        return fleischner.assess(
            nodule_type=req.nodule_type,
            size_mm=req.size_mm,
            multiple=req.multiple,
            risk=req.risk,
            solid_component_mm=req.solid_component_mm,
            location=location,
            long_axis_mm=req.long_axis_mm,
            short_axis_mm=req.short_axis_mm,
            prior_size_mm=req.prior_size_mm,
            interval_months=req.interval_months,
            volume_mm3=req.volume_mm3,
            prior_volume_mm3=req.prior_volume_mm3,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Public adrenal CT washout calculator — free, no auth, deterministic (no model
# call, no patient data stored). Arithmetic lives in web/adrenal.py.
# ---------------------------------------------------------------------------

class AdrenalRequest(BaseModel):
    enhanced_hu: float
    delayed_hu: float
    unenhanced_hu: Optional[float] = None
    size_mm: Optional[float] = None
    location: Optional[str] = None


@app.post("/api/adrenal/washout")
def api_adrenal_washout(req: AdrenalRequest):
    """Return the adrenal CT washout metrics, category and a report line.

    Pure arithmetic on the region-of-interest attenuation values — no external
    model call and nothing is persisted, so no rate limiting is needed. A
    free-text location is echoed only into the report line the caller sees.
    """
    location = (req.location or "").strip()[:80] or None
    try:
        return adrenal.assess(
            enhanced_hu=req.enhanced_hu,
            delayed_hu=req.delayed_hu,
            unenhanced_hu=req.unenhanced_hu,
            size_mm=req.size_mm,
            location=location,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Public nodule volume doubling time calculator — free, no auth, deterministic
# (no model call, no patient data stored). Arithmetic lives in
# web/nodule_growth.py.
# ---------------------------------------------------------------------------

class NoduleGrowthRequest(BaseModel):
    prior: float
    current: float
    interval_days: float
    method: str = "volume"


@app.post("/api/nodule-growth/vdt")
def api_nodule_growth_vdt(req: NoduleGrowthRequest):
    """Return the volume change, doubling time, framework readings and a report line.

    Pure arithmetic on two measurements and an interval — no external model
    call and nothing is persisted, so no rate limiting is needed. Scan dates
    and the nodule location stay in the browser; only numbers are sent.
    """
    try:
        return nodule_growth.assess(
            prior=req.prior,
            current=req.current,
            interval_days=req.interval_days,
            method=req.method,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# WebSocket auth helpers
# ---------------------------------------------------------------------------

# WS tokens are short-lived signed tokens minted server-side for an already
# authenticated user (the index page is behind _verify_auth). Signing them with
# the session secret means an outsider cannot forge one — the previous scheme
# was base64(user:password) keyed on the shared VOXRAD_WEB_PASSWORD, which in
# OAuth deployments defaults to the public "voxrad", letting anyone open the
# socket and burn STT/LLM credits.
_WS_TOKEN_MAX_AGE = 12 * 3600  # 12 hours


def _ws_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(
        SESSION_SECRET_KEY() or "voxrad-ws-fallback", salt="voxrad-ws-transcribe"
    )


def _make_ws_token(username: str) -> str:
    """Return a signed, time-limited token authorising a WS connection."""
    return _ws_serializer().dumps({"u": username})


def _verify_ws_token(token: str) -> Optional[dict]:
    """Verify a signed WS token; return a minimal user dict, or None on failure."""
    if not token:
        return None
    try:
        data = _ws_serializer().loads(token, max_age=_WS_TOKEN_MAX_AGE)
    except (BadSignature, SignatureExpired, Exception):
        return None
    username = data.get("u") if isinstance(data, dict) else None
    if not username:
        return None
    return {"id": None, "name": username, "email": username}


# ---------------------------------------------------------------------------
# Keyword list builder — used for provider-side vocabulary boosting
# ---------------------------------------------------------------------------

def _build_keyword_list(template_name: Optional[str], user: Optional[dict] = None) -> list:
    """Return a list of medical terms for STT keyword boosting.

    Prefers the [correct spellings] block from the selected template;
    falls back to extracting terms from _RADIOLOGY_PROMPT.
    User-learned vocab terms come first: both streaming providers cap
    keyterms at 100, so terms the user explicitly corrected must not be
    the ones truncated away.
    """
    text = _RADIOLOGY_PROMPT
    if template_name:
        content = _load_template_content(template_name)
        match = re.search(
            r"\[correct spellings\](.*?)\[correct spellings\]", content, re.DOTALL
        )
        if match:
            text = match.group(1).strip()
    parts = re.split(r"[,\n.]+", text)
    template_terms = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
    keywords = []
    if user:
        keywords = list(_load_user_vocab(user))
    if template_name and "mri" in template_name.lower():
        if "3T platform" not in keywords:
            keywords.append("3T platform")
    for term in template_terms:
        if term not in keywords:
            keywords.append(term)
    return keywords[:100]


# Mount mock OpenAI-compatible routes when running without real API keys.
if os.environ.get("VOXRAD_MOCK_MODE"):
    from web.mock_routes import router as _mock_router
    app.include_router(_mock_router)
    logger.info("[mock] Mock API routes mounted at /mock/v1/...")


@app.get("/login")
def login_page(request: Request):
    if oauth_enabled() and get_session_user(request):
        return RedirectResponse("/app")
    return _jinja.TemplateResponse(
        request,
        "login.html",
        {
            "request": request,
            "google_enabled": google_enabled(),
            "microsoft_enabled": microsoft_enabled(),
            "error": request.query_params.get("error"),
            "static_version": _STATIC_VERSION,
        },
    )


@app.get("/auth/google")
def auth_google(request: Request):
    if not google_enabled():
        raise HTTPException(status_code=404)
    state = secrets.token_urlsafe(16)
    request.session["oauth_state"] = state
    return RedirectResponse(google_auth_url(state))


@app.get("/auth/google/callback")
def auth_google_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    if error:
        return RedirectResponse(f"/login?error={error}")
    expected = request.session.pop("oauth_state", None)
    if not expected or state != expected:
        return RedirectResponse("/login?error=invalid_state")
    try:
        info = exchange_google_code(code)
    except Exception as exc:
        logger.error("Google OAuth error: %s", exc)
        return RedirectResponse("/login?error=google_auth_failed")
    if not info.get("email"):
        return RedirectResponse("/login?error=no_email")
    db_user = get_or_create_user(info["email"], info["name"], "google")
    set_session_user(request, db_user)
    log_event(user_id=db_user.get("id"), event_type="login", metadata={"provider": "google"})
    return RedirectResponse("/app")


@app.get("/auth/microsoft")
def auth_microsoft(request: Request):
    if not microsoft_enabled():
        raise HTTPException(status_code=404)
    state = secrets.token_urlsafe(16)
    request.session["oauth_state"] = state
    return RedirectResponse(microsoft_auth_url(state))


@app.get("/auth/microsoft/callback")
def auth_microsoft_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    if error:
        return RedirectResponse(f"/login?error={error}")
    expected = request.session.pop("oauth_state", None)
    if not expected or state != expected:
        return RedirectResponse("/login?error=invalid_state")
    try:
        info = exchange_microsoft_code(code)
    except Exception as exc:
        logger.error("Microsoft OAuth error: %s", exc)
        return RedirectResponse("/login?error=microsoft_auth_failed")
    if not info.get("email"):
        return RedirectResponse("/login?error=no_email")
    db_user = get_or_create_user(info["email"], info["name"], "microsoft")
    set_session_user(request, db_user)
    log_event(user_id=db_user.get("id"), event_type="login", metadata={"provider": "microsoft"})
    return RedirectResponse("/app")


@app.get("/logout")
def logout(request: Request):
    clear_session(request)
    return RedirectResponse("/")


@app.get("/")
def landing_page(request: Request):
    if oauth_enabled() and get_session_user(request):
        return RedirectResponse("/app")
    return _jinja.TemplateResponse(
        request,
        "landing.html",
        {
            "request": request,
            "google_enabled": google_enabled(),
            "microsoft_enabled": microsoft_enabled(),
            "static_version": _STATIC_VERSION,
        },
    )


_LEGAL_DIR = Path(_BASE_DIR) / "templates" / "legal"
_LEGAL_PAGES = {
    "/privacy": (
        "privacy.html", "Privacy Policy",
        "How RadSpeed handles personal and health information under the Australian Privacy Principles.",
    ),
    "/terms": (
        "terms.html", "Terms of Use",
        "Terms for radiology practices and radiologists using RadSpeed.",
    ),
}


def _legal_page(request: Request, path: str):
    """Serve a legal page generated from docs/compliance by tools/build_compliance_pack.py."""
    filename, title, description = _LEGAL_PAGES[path]
    source = _LEGAL_DIR / filename
    if not source.is_file():
        raise HTTPException(status_code=503, detail="Page not built")
    return _jinja.TemplateResponse(
        request,
        "legal.html",
        {
            "request": request,
            "title": title,
            "description": description,
            "path": path,
            "body": source.read_text(encoding="utf-8"),
            "static_version": _STATIC_VERSION,
        },
    )


@app.get("/privacy", include_in_schema=False)
def privacy_page(request: Request):
    return _legal_page(request, "/privacy")


@app.get("/terms", include_in_schema=False)
def terms_page(request: Request):
    return _legal_page(request, "/terms")


@app.get("/radiology-reporting-software", include_in_schema=False)
def radiology_reporting_software_page(request: Request):
    return _jinja.TemplateResponse(
        request,
        "radiology_reporting_software.html",
        {"request": request, "static_version": _STATIC_VERSION},
    )


@app.get("/powerscribe-companion", include_in_schema=False)
def powerscribe_companion_page(request: Request):
    return _jinja.TemplateResponse(
        request,
        "powerscribe_companion.html",
        {"request": request, "static_version": _STATIC_VERSION},
    )


@app.get("/ti-rads-calculator", include_in_schema=False)
def tirads_calculator_page(request: Request):
    return _jinja.TemplateResponse(
        request,
        "tirads_calculator.html",
        {"request": request, "static_version": _STATIC_VERSION},
    )


@app.get("/fleischner-calculator", include_in_schema=False)
def fleischner_calculator_page(request: Request):
    return _jinja.TemplateResponse(
        request,
        "fleischner_calculator.html",
        {"request": request, "static_version": _STATIC_VERSION},
    )


@app.get("/adrenal-washout-calculator", include_in_schema=False)
def adrenal_washout_calculator_page(request: Request):
    return _jinja.TemplateResponse(
        request,
        "adrenal_washout_calculator.html",
        {"request": request, "static_version": _STATIC_VERSION},
    )


@app.get("/volume-doubling-time-calculator", include_in_schema=False)
def volume_doubling_time_calculator_page(request: Request):
    return _jinja.TemplateResponse(
        request,
        "volume_doubling_time_calculator.html",
        {"request": request, "static_version": _STATIC_VERSION},
    )


@app.get("/report-templates", include_in_schema=False)
def report_templates_index(request: Request, q: str = ""):
    return _jinja.TemplateResponse(
        request,
        "report_templates_index.html",
        {
            "request": request,
            "static_version": _STATIC_VERSION,
            "groups": report_library.library_groups(),
            "template_count": report_library.library_count(),
            "initial_query": q.strip()[:80],
            "search_stopwords": " ".join(sorted(report_library.SEARCH_STOPWORDS)),
        },
    )


@app.get("/report-templates/{slug}", include_in_schema=False)
def report_template_detail(slug: str, request: Request):
    entry = report_library.get_entry(slug)
    if not entry:
        raise HTTPException(status_code=404, detail="Report template not found")
    related = report_library.related_entries(slug)
    return _jinja.TemplateResponse(
        request,
        "report_template_detail.html",
        {
            "request": request,
            "static_version": _STATIC_VERSION,
            "t": entry,
            "related": related,
        },
    )


@app.get("/app")
def index(request: Request, user: dict = Depends(_verify_auth)):
    _username = _get_username(user)
    _style = _user_style(user) or {}
    paste_format = _style.get("paste_format", config.style_paste_format) or "rich"
    return _jinja.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "templates": _list_templates(),
            "username": _username,
            "fhir_enabled": _user_fhir_enabled(user),
            "ws_token": _make_ws_token(_username),
            "static_version": _STATIC_VERSION,
            "oauth_mode": oauth_enabled(),
            "paste_format": paste_format,
            "research_features": practice.settings.research_features,
        },
    )


@app.get("/fracture-workbench", dependencies=[Depends(_require_research_feature)])
def fracture_workbench_page(user: dict = Depends(_verify_auth)):
    """Serve the public-data benchmark viewer behind RadSpeed sign-in."""
    if not _FRACTURE_WORKBENCH_PAGE.is_file():
        raise HTTPException(status_code=503, detail="Fracture Lab is not installed")
    return FileResponse(
        _FRACTURE_WORKBENCH_PAGE,
        media_type="text/html",
        headers={
            "Cache-Control": "private, no-store",
            "Content-Security-Policy": (
                "default-src 'self'; img-src 'self' blob:; "
                "style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; "
                "worker-src 'self' blob:; base-uri 'none'; "
                "frame-ancestors 'none'"
            ),
            "X-Robots-Tag": "noindex, nofollow",
        },
    )


@app.post("/api/fracture-analysis", dependencies=[Depends(_require_research_feature)])
async def fracture_analysis(
    images: list[UploadFile] = File(...),
    clinical_context: Optional[str] = Form(None),
    privacy_confirmed: Optional[str] = Form(None),
    user: dict = Depends(_verify_auth),
):
    """Run an ephemeral, experimental multi-view fracture second read."""
    if not images or len(images) > MAX_FRACTURE_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Use between 1 and {MAX_FRACTURE_IMAGES} X-ray views.",
        )
    if privacy_confirmed != "true":
        raise HTTPException(
            status_code=400,
            detail="Check the cleaned previews and confirm privacy before analysis.",
        )
    if not _MOCK_MODE and not config.TEXT_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="The RadSpeed vision model is not configured.",
        )

    try:
        # The small production VM handles one memory-intensive upload at a
        # time. Waiting requests remain in Starlette's bounded upload spool and
        # never reach RadSpeed's persistent data volume.
        async with _fracture_analysis_slot:
            payloads: list[bytes] = []
            try:
                for image in images:
                    payloads.append(await image.read(MAX_FRACTURE_IMAGE_BYTES + 1))
            finally:
                for image in images:
                    await image.close()
            prepared = await asyncio.to_thread(prepare_fracture_images, payloads)
            if _MOCK_MODE:
                assessment = mock_fracture_assessment(
                    len(prepared), study_region="chest_ribs"
                )
                method = "synthetic_test_fixture"
                supporting_models = [
                    {
                        "kind": "classifier",
                        "role": "chest_fracture_classifier",
                        "label": "Open chest/rib classifier",
                        "model": "synthetic_test_fixture",
                        "scope": "Chest and rib radiographs",
                        "probability_kind": (
                            "public_dataset_estimate_not_patient_specific"
                        ),
                        "view_probabilities": [0.42] * len(prepared),
                        "highest_view_probability": 0.42,
                        "validation_threshold": 0.05,
                        "evaluation": {
                            "auc": 0.780,
                            "cases": 553,
                            "dataset": "ChestX-Det untouched public test set",
                            "limitation": (
                                "Performance and calibration may differ on local images."
                            ),
                        },
                    }
                ]
            else:
                assessment, method, supporting_models = await asyncio.to_thread(
                    analyse_fracture_images,
                    prepared,
                    clinical_context=clinical_context,
                    chest_scorer=score_chest_images,
                    general_locator=locate_fracture_candidates,
                    wrist_locator=locate_wrist_fracture_candidates,
                    strong_scorer=score_strong_fracture_images,
                )
    except FractureImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AuthenticationError as exc:
        logger.error("Vision model API key rejected during fracture analysis.")
        raise HTTPException(
            status_code=503,
            detail="The RadSpeed vision model could not authenticate.",
        ) from exc
    except Exception as exc:
        # Do not include provider text, filenames, clinical context or image
        # details in logs: any of them may contain health information.
        logger.error("Fracture analysis failed (%s).", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="The fracture review could not be completed. Try again shortly.",
        ) from exc

    log_event(
        user_id=user.get("id"),
        event_type="fracture_analysis",
        metadata={
            "image_count": len(prepared),
            "assessment": assessment.assessment,
            "confidence_percent": assessment.confidence_percent,
            "method": method,
            "detected_study_type": assessment.study_region,
            "supporting_model_count": len(supporting_models),
        },
    )
    return JSONResponse(
        {
            "assessment": assessment.model_dump(),
            "method": method,
            "image_count": len(prepared),
            "detected_study_type": assessment.study_region,
            "model_confidence_is_calibrated": False,
            "supporting_models": supporting_models,
        },
        headers={"Cache-Control": "private, no-store"},
    )


@app.get("/fracture-workbench/images/{image_path:path}", dependencies=[Depends(_require_research_feature)])
def fracture_workbench_image(image_path: str, user: dict = Depends(_verify_auth)):
    """Serve one benchmark radiograph from the persistent encrypted volume."""
    try:
        image = resolve_workbench_image(image_path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Image not found") from exc
    if not image.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        image,
        headers={
            "Cache-Control": "private, max-age=86400",
            "X-Robots-Tag": "noindex, nofollow",
        },
    )


@app.get("/settings")
def settings_page(request: Request, user: dict = Depends(_verify_auth)):
    return _jinja.TemplateResponse(
        request,
        "settings.html",
        {
            "request": request,
            "username": _get_username(user),
            "static_version": _STATIC_VERSION,
        },
    )


@app.get("/templates")
def list_templates(user: dict = Depends(_verify_auth)):
    user_dir = os.path.join(config.save_directory or "", "templates")
    user_names: set[str] = set()
    if os.path.isdir(user_dir):
        user_names = {f for f in os.listdir(user_dir) if f.endswith((".txt", ".md"))}
    return {
        "templates": [
            {"name": t, "is_custom": t in user_names}
            for t in _list_templates()
        ]
    }


_TEMPLATE_NAME_RE = re.compile(r'^[\w\-. ]+\.(txt|md)$')


@app.get("/api/templates/{name}")
def get_template_content(name: str, user: dict = Depends(_verify_auth)):
    if not _TEMPLATE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid template name")
    user_dir = os.path.join(config.save_directory or "", "templates")
    user_path = os.path.join(user_dir, name)
    bundled_path = os.path.join(_BUNDLED_TEMPLATES_DIR, name)
    is_custom = os.path.exists(user_path)
    path = user_path if is_custom else bundled_path
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Template not found")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    structure, ai_instructions, has_split = split_template(content)
    return {
        "content": content,
        "structure": structure,
        "ai_instructions": ai_instructions,
        "has_split": has_split,
        "is_custom": is_custom,
    }


class _TemplateSaveBody(BaseModel):
    # Either pass `content` (raw, legacy) OR the two-section fields. When the
    # section fields are present they take precedence and are recombined into
    # the on-disk file via join_template().
    content: Optional[str] = None
    structure: Optional[str] = None
    ai_instructions: Optional[str] = None


@app.put("/api/templates/{name}")
def save_template_content(name: str, body: _TemplateSaveBody, user: dict = Depends(_verify_auth)):
    if not _TEMPLATE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid template name")
    if body.structure is not None or body.ai_instructions is not None:
        content = join_template(body.structure or "", body.ai_instructions or "")
    elif body.content is not None:
        content = body.content
    else:
        raise HTTPException(status_code=400, detail="No content provided")
    user_dir = os.path.join(config.save_directory or "", "templates")
    os.makedirs(user_dir, exist_ok=True)
    with open(os.path.join(user_dir, name), "w", encoding="utf-8") as f:
        f.write(content)
    return {"ok": True}


@app.delete("/api/templates/{name}")
def delete_template_content(name: str, user: dict = Depends(_verify_auth)):
    if not _TEMPLATE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid template name")
    user_dir = os.path.join(config.save_directory or "", "templates")
    path = os.path.join(user_dir, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No custom version found")
    os.remove(path)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Whisper hallucination filter
# ---------------------------------------------------------------------------

# Whisper reliably hallucinates these phrases on silent/noisy audio.
# Normalise to lowercase, strip trailing punctuation before comparing.
_HALLUCINATIONS: set[str] = {
    "", "thank you", "thanks", "thank you so much", "thank you very much",
    "thank you for watching", "thanks for watching", "thank you for listening",
    "thanks for listening", "please like and subscribe", "don't forget to subscribe",
    "okay", "ok", "alright", "right", "sure", "yes", "no", "yep", "nope",
    "bye", "goodbye", "see you", "see you next time", "take care",
    "hello", "hi", "hey", "welcome", "welcome back",
    "you", "hmm", "mm", "mm-hmm", "um", "uh", "ah", "oh",
    "of course", "absolutely", "indeed", "certainly",
    "subtitles by", "subtitles", "captions by", "captions",
    "transcribed by", "translated by",
    "john", "omar", "james", "michael", "david",  # common single-name hallucinations
}


def _is_hallucination(text: str, asr_prompt: str = "") -> bool:
    normalised = text.strip().lower().rstrip(".,!?;: ").strip()
    if normalised in _HALLUCINATIONS:
        return True
    # Single short non-medical word
    words = normalised.split()
    if len(words) == 1 and len(normalised) <= 6 and normalised.isalpha():
        return True
    # Repetition loop: any 5-word ngram appearing more than once
    if len(words) >= 10:
        seen: set[tuple] = set()
        for i in range(len(words) - 4):
            ngram = tuple(words[i:i+5])
            if ngram in seen:
                return True
            seen.add(ngram)
    # Prompt-echo detection: discard if the transcription matches any sentence
    # from the ASR prompt (Whisper completes/repeats prompt on silent audio)
    if asr_prompt:
        text_norm = text.strip().lower().rstrip(".,!?;: ")
        for sentence in re.split(r"[.!?]", asr_prompt):
            s = sentence.strip().lower().rstrip(".,!?;: ")
            if len(s) > 15 and (text_norm == s or text_norm in s or s in text_norm):
                return True
    return False


# Valid single-word replacements a radiologist may dictate in voice-edit mode.
# We allow these through the voice-edit hallucination filter; the regular
# dictation filter still rejects them (because standalone they're suspicious).
# Laterality ("right"/"left") and severity/presence short words are the
# most common one-word voice edits, so they must pass through even though
# some (e.g. "right") are in _HALLUCINATIONS for the general filter.
_VOICE_EDIT_ALLOWED_SHORT = {
    # Laterality and orientation — critical for anatomy swaps
    "left", "right", "bilateral", "upper", "lower",
    "anterior", "posterior", "medial", "lateral",
    "superior", "inferior", "proximal", "distal",
    # Presence / absence
    "no", "yes", "none", "absent", "present", "unchanged",
    "normal", "abnormal", "positive", "negative",
    # Severity
    "mild", "moderate", "severe", "minimal", "marked",
    "small", "large", "trace",
    # Character
    "acute", "chronic", "focal", "diffuse",
    "intact", "clear", "stable", "benign", "patent", "occluded",
    # Common punctuated variants the ASR may return
    "mild.", "no.", "yes.", "right.", "left.",
}


def _is_voice_edit_hallucination(text: str, asr_prompt: str = "") -> bool:
    """Hallucination filter for voice-edit mode.

    Same as _is_hallucination but permissive of short valid replacements
    (e.g. "normal", "intact", "no") that a radiologist may genuinely say.
    Still rejects known Whisper garbage phrases, prompt-echo, and repetition
    loops — the ones that fire on silent/noisy audio.
    """
    normalised = text.strip().lower().rstrip(".,!?;: ").strip()
    if normalised in _VOICE_EDIT_ALLOWED_SHORT:
        return False
    if normalised in _HALLUCINATIONS:
        return True
    words = normalised.split()
    # Repetition loop
    if len(words) >= 10:
        seen: set[tuple] = set()
        for i in range(len(words) - 4):
            ngram = tuple(words[i:i+5])
            if ngram in seen:
                return True
            seen.add(ngram)
    # Prompt-echo: Whisper repeats the surrounding-text prompt verbatim on
    # silent audio. Decisive signal — keep this check active.
    if asr_prompt:
        text_norm = text.strip().lower().rstrip(".,!?;: ")
        for sentence in re.split(r"[.!?]", asr_prompt):
            s = sentence.strip().lower().rstrip(".,!?;: ")
            if len(s) > 15 and (text_norm == s or text_norm in s or s in text_norm):
                return True
    return False


# ---------------------------------------------------------------------------
# Radiology vocabulary prompt for Whisper
# Two-part design per OpenAI Whisper docs:
#   1. A short context sentence (tells Whisper this is medical dictation)
#   2. A curated vocabulary list of RadLex-derived terms Whisper commonly
#      confuses — comma-separated so Whisper learns their spelling/casing.
# Template-specific [correct spellings] blocks override this entirely.
# ---------------------------------------------------------------------------
_RADIOLOGY_PROMPT = (
    # Context sentence + RadLex-derived vocabulary list (≤ 896 chars for Groq)
    "Radiology dictation. "
    "oedema, meniscus, menisci, supraspinatus, infraspinatus, subscapularis, "
    "chondromalacia, chondral, subchondral, osteochondral, trabecular, "
    "Baker's cyst, Hoffa's fat pad, trochanteric, ACL, PCL, MCL, LCL, MPFL, "
    "Bankart, Hill-Sachs, SLAP, rotator cuff, "
    "effusion, synovitis, tenosynovitis, enthesopathy, bursitis, "
    "consolidation, atelectasis, bronchiectasis, pneumothorax, "
    "pleural effusion, mediastinum, hilar, parenchyma, "
    "hepatomegaly, splenomegaly, cholelithiasis, nephrolithiasis, "
    "lymphadenopathy, herniation, spondylosis, spondylolisthesis, stenosis, "
    "cauda equina, ligamentum flavum, intraosseous, cortical, cancellous, "
    "T1-weighted, T2-weighted, STIR, gradient echo, Hounsfield, "
    "coronal, sagittal, axial."
)

# ---------------------------------------------------------------------------
# LLM post-correction for raw Whisper output
# A fast, tight prompt that fixes medical ASR errors without rephrasing.
# ---------------------------------------------------------------------------
_CORRECTION_SYSTEM = """\
You are a medical transcription corrector for radiology dictation.
Fix ONLY obvious speech recognition errors: misspelled medical terms, \
mangled anatomy, and garbled drug or procedure names.
Do NOT rephrase, reorder, summarise, or add any words not present in the input.
Return the corrected text only — no explanation, no prefix, no punctuation changes \
beyond fixing the erroneous word itself.\
"""


def _correct_asr_text(raw: str) -> str:
    """Pass raw Whisper output through a fast LLM to fix medical terminology errors.

    Uses the configured text LLM. Skips correction if
    the text LLM is not configured or if the text is very short (≤ 3 words).
    """
    if not config.TEXT_API_KEY:
        return raw
    words = raw.split()
    if len(words) <= 3:
        return raw  # too short to bother; unlikely to have complex errors
    try:
        client = get_text_client(OpenAI)
        resp = client.chat.completions.create(
            model=config.SELECTED_MODEL,
            messages=[
                {"role": "system", "content": _CORRECTION_SYSTEM},
                {"role": "user", "content": raw},
            ],
            **completion_options(
                config.SELECTED_MODEL,
                temperature=0.0,
                max_tokens=len(words) + 30,  # corrected text won't be longer
            ),
        )
        corrected = resp.choices[0].message.content.strip()
        # Safety: if the LLM somehow returns nothing or drastically expands the
        # text, fall back to the original Whisper output.
        if not corrected or len(corrected) > len(raw) * 2:
            return raw
        return corrected
    except Exception as exc:
        logger.warning("ASR correction skipped: %s", exc)
        return raw


# ---------------------------------------------------------------------------
# Per-user vocabulary — "remember my corrections"
# ---------------------------------------------------------------------------
# When the user voice-edits a highlighted word and the replacement sounds
# like a correction of a misheard word (rather than a content change), we
# suggest adding the new spelling to their vocab. Accepted terms are then
# injected into the Whisper ASR prompt on subsequent dictations, so the
# same mistake is less likely to recur.
#
# Phonetic check is intentionally heuristic — the toast is dismissable, the
# user has the final say.

def _vocab_dir() -> str:
    return os.path.join(config.save_directory or "/tmp", "vocab")


def _vocab_path(user: dict) -> str:
    os.makedirs(_vocab_dir(), exist_ok=True)
    uid = user.get("id")
    # OAuth → per-user file; Basic Auth → one shared file.
    key = f"user_{uid}" if uid is not None else "shared"
    return os.path.join(_vocab_dir(), f"{key}.txt")


def _load_user_vocab(user: dict) -> list:
    path = _vocab_path(user)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except OSError:
        return []


def _add_to_user_vocab(user: dict, term: str) -> bool:
    term = term.strip()
    if not term:
        return False
    existing = _load_user_vocab(user)
    if term.lower() in [t.lower() for t in existing]:
        return False
    path = _vocab_path(user)
    with open(path, "a", encoding="utf-8") as f:
        f.write(term + "\n")
    logger.info("Added vocab term %r for %s", term, _get_username(user))
    return True


def _soundalike(word: str) -> str:
    """Crude phonetic reduction. Not full metaphone, zero deps.

    Collapses homophonic digraphs, drops interior vowels, removes doubles —
    enough to make 'supraspinatus'↔'super spin atrice' look close without
    making 'mild'↔'moderate' look close.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    w = w.replace("ph", "f").replace("ck", "k").replace("qu", "kw")
    w = w.replace("ch", "k").replace("sh", "s")
    head, tail = w[0], re.sub(r"[aeiouy]", "", w[1:])
    collapsed = head + tail
    out = []
    for ch in collapsed:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def _phonetic_similar(old: str, new: str) -> bool:
    """True when `new` looks like an ASR correction of `old`, not a content edit."""
    import difflib
    a = re.sub(r"[^\w\s]", "", old.strip().lower())
    b = re.sub(r"[^\w\s]", "", new.strip().lower())
    if not a or not b or a == b:
        return False
    if difflib.SequenceMatcher(None, a, b).ratio() >= 0.65:
        return True
    sa, sb = _soundalike(a), _soundalike(b)
    if sa and sb and difflib.SequenceMatcher(None, sa, sb).ratio() >= 0.7:
        return True
    return False


def _should_suggest_vocab(user: dict, old: str, new: str) -> Optional[dict]:
    """Return {old, new} if this edit looks like a correction worth remembering.

    Single-word edits only — multi-word diffs are too error-prone to
    auto-suggest. Skips terms already in the user's vocab.
    """
    a = (old or "").strip()
    b = (new or "").strip()
    if not a or not b:
        return None
    if len(a.split()) != 1 or len(b.split()) != 1:
        return None
    a_key = a.lower().rstrip(".,!?;:")
    b_key = b.lower().rstrip(".,!?;:")
    if a_key == b_key:
        return None
    if not _phonetic_similar(a, b):
        return None
    if b_key in {t.lower().rstrip(".,!?;:") for t in _load_user_vocab(user)}:
        return None
    return {"old": a.rstrip(".,!?;:"), "new": b.rstrip(".,!?;:")}


# ---------------------------------------------------------------------------
# Style drift detection — proactive setting suggestions
# ---------------------------------------------------------------------------
# Detects when the user repeatedly corrects the LLM's output in a way that
# signals a style-setting preference. After 2 similar edits, returns a
# suggest_setting dict so the client can offer to save the preference.
#
# Patterns checked via voice-edit old/new pairs (/transcribe):
#   1. Spelling — British↔American word substitutions
#   3. Laterality — full ("left") ↔ abbreviated ("L")
#   5. Negation phrasing — "no evidence of" / "no X identified" / "X absent"
#   N. Numerals — same-combo rule: "grade VI"→"grade 6" must appear ≥2×
#      for the SAME preceding-word+numeral combo to count
#
# Patterns checked via full-text report diff (/api/track-report-edit):
#   4. Impression style — bullets ↔ numbered ↔ prose
#   6. Date format — format inferred from the dates that appear in the report

# British → American spelling pairs frequent in radiology.
_BRIT_TO_AMER: dict = {
    "colour": "color",        "colours": "colors",
    "tumour": "tumor",        "tumours": "tumors",
    "oedema": "edema",        "oedemas": "edemas",     "oedematous": "edematous",
    "haemorrhage": "hemorrhage",  "haemorrhagic": "hemorrhagic",
    "haematoma": "hematoma",  "haematomas": "hematomas",
    "haematological": "hematological",
    "anaesthesia": "anesthesia",  "anaesthetic": "anesthetic",
    "paediatric": "pediatric",    "paediatrics": "pediatrics",
    "localised": "localized",     "generalised": "generalized",
    "organised": "organized",     "visualised": "visualized",
    "recognised": "recognized",   "characterised": "characterized",
    "analysed": "analyzed",
    "oesophagus": "esophagus",    "oesophageal": "esophageal",
    "orthopaedic": "orthopedic",  "orthopaedics": "orthopedics",
    "gynaecology": "gynecology",  "gynaecological": "gynecological",
    "diarrhoea": "diarrhea",
    "foetal": "fetal",            "foetus": "fetus",
    "aetiology": "etiology",      "aetiological": "etiological",
    "grey": "gray",               "greys": "grays",
    "artefact": "artifact",       "artefacts": "artifacts",
    "caecum": "cecum",            "caecal": "cecal",
    "oestrogen": "estrogen",      "haemoglobin": "hemoglobin",
    "fibre": "fiber",             "fibres": "fibers",
    "centre": "center",           "centres": "centers",
}
_AMER_TO_BRIT: dict = {v: k for k, v in _BRIT_TO_AMER.items()}

# Roman ↔ Arabic numerals (I–XII covers all clinical grades/segments/stages).
_ROMAN_TO_INT: dict = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6,
    "vii": 7, "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12,
}
_INT_TO_ROMAN: dict = {v: k.upper() for k, v in _ROMAN_TO_INT.items()}

# Per-user in-memory edit counters — {user_key: {pattern_key: count}}.
# Resets on server restart (deploy). Two edits within a session is enough.
_STYLE_COUNTS: dict = {}
_STYLE_THRESHOLD = 2


def _sty_key(user: dict) -> str:
    return str(user.get("id") or user.get("email") or "anon")


def _sty_count(user: dict, pk: str) -> int:
    """Increment the counter for pattern_key and return the new count."""
    k = _sty_key(user)
    if k not in _STYLE_COUNTS:
        _STYLE_COUNTS[k] = {}
    _STYLE_COUNTS[k][pk] = _STYLE_COUNTS[k].get(pk, 0) + 1
    return _STYLE_COUNTS[k][pk]


def _dismissed_path(user: dict) -> str:
    os.makedirs(_vocab_dir(), exist_ok=True)
    return os.path.join(_vocab_dir(), f"style_dismissed_{_sty_key(user)}.json")


def _load_dismissed(user: dict) -> set:
    path = _dismissed_path(user)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return set(json.load(f))
    except (OSError, ValueError):
        pass
    return set()


def _save_dismissed(user: dict, pattern_key: str) -> None:
    current = _load_dismissed(user)
    current.add(pattern_key)
    with open(_dismissed_path(user), "w", encoding="utf-8") as f:
        json.dump(list(current), f)


def _numeral_prefs_path(user: dict) -> str:
    os.makedirs(_vocab_dir(), exist_ok=True)
    return os.path.join(_vocab_dir(), f"numeral_prefs_{_sty_key(user)}.json")


def _load_numeral_prefs(user: dict) -> list:
    """List of {pattern, replacement} dicts, e.g. [{"pattern": "grade VI", "replacement": "grade 6"}]."""
    path = _numeral_prefs_path(user)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (OSError, ValueError):
        pass
    return []


def _add_numeral_pref(user: dict, pattern: str, replacement: str) -> None:
    prefs = [p for p in _load_numeral_prefs(user) if p.get("pattern", "").lower() != pattern.lower()]
    prefs.append({"pattern": pattern, "replacement": replacement})
    with open(_numeral_prefs_path(user), "w", encoding="utf-8") as f:
        json.dump(prefs, f)
    logger.info("Saved numeral pref %r→%r for %s", pattern, replacement, _get_username(user))


def _make_suggest(setting: str, value: str, message: str, pattern_key: str, **extra) -> dict:
    d = {"setting": setting, "value": value, "message": message, "pattern_key": pattern_key}
    d.update(extra)
    return d


def _check_and_suggest(user: dict, pk: str, make_fn) -> Optional[dict]:
    """Increment counter for pk; if >= threshold and not dismissed, return make_fn()."""
    if pk in _load_dismissed(user):
        return None
    if _sty_count(user, pk) >= _STYLE_THRESHOLD:
        return make_fn()
    return None


def _detect_style_drift(
    user: dict, old: str, new: str, pre_context: str = ""
) -> Optional[dict]:
    """Detect style-setting drift from a voice-edit old→new pair.

    Checks spelling, laterality, negation phrasing, and numerals (same-combo rule).
    Returns a suggest_setting dict when the threshold is reached, else None.
    """
    if not old or not new:
        return None
    o_clean = re.sub(r"[^\w\s]", "", old.lower()).strip()
    n_clean = re.sub(r"[^\w\s]", "", new.lower()).strip()
    if not o_clean or not n_clean or o_clean == n_clean:
        return None

    o_set = set(o_clean.split())
    n_set = set(n_clean.split())

    # 1. Spelling drift (per-word scan handles multi-word selections)
    for ow in o_set:
        if ow in _BRIT_TO_AMER and _BRIT_TO_AMER[ow] in n_set:
            amer = _BRIT_TO_AMER[ow]
            pk = "spelling:brit→amer"
            return _check_and_suggest(user, pk, lambda ow=ow, amer=amer: _make_suggest(
                "style_spelling", "american",
                f"You keep using American spelling (e.g. <b>{amer}</b> instead of <b>{ow}</b>) — set American English as default?",
                pk,
            ))
        if ow in _AMER_TO_BRIT and _AMER_TO_BRIT[ow] in n_set:
            brit = _AMER_TO_BRIT[ow]
            pk = "spelling:amer→brit"
            return _check_and_suggest(user, pk, lambda ow=ow, brit=brit: _make_suggest(
                "style_spelling", "british",
                f"You keep using British spelling (e.g. <b>{brit}</b> instead of <b>{ow}</b>) — set British English as default?",
                pk,
            ))

    # 3. Laterality drift — full ↔ abbreviated
    _lat_f2a = {"left": "l", "right": "r", "bilateral": "b"}
    _lat_a2f = {v: k for k, v in _lat_f2a.items()}
    for ow in o_set:
        if ow in _lat_f2a and _lat_f2a[ow] in n_set:
            abbrev = _lat_f2a[ow].upper()
            pk = "laterality:full→abbrev"
            return _check_and_suggest(user, pk, lambda ow=ow, abbrev=abbrev: _make_suggest(
                "style_laterality", "abbrev",
                f"You keep abbreviating laterality (e.g. <b>{ow}</b> → <b>{abbrev}</b>) — use abbreviated style by default?",
                pk,
            ))
        if ow in _lat_a2f and _lat_a2f[ow] in n_set:
            full = _lat_a2f[ow]
            pk = "laterality:abbrev→full"
            return _check_and_suggest(user, pk, lambda ow=ow, full=full: _make_suggest(
                "style_laterality", "full",
                f"You keep writing out laterality in full (e.g. <b>{ow.upper()}</b> → <b>{full}</b>) — use full style by default?",
                pk,
            ))

    # 5. Negation phrasing drift — compare phrase structure
    def _neg_style(text: str) -> Optional[str]:
        t = text.lower()
        if "no evidence of" in t:
            return "no_evidence_of"
        if re.search(r'\bno\b.{1,40}\bidentified\b', t):
            return "no_x_identified"
        if re.search(r'\b\w+\b.{0,30}\babsent\b', t):
            return "x_absent"
        return None

    old_neg = _neg_style(old)
    new_neg = _neg_style(new)
    if old_neg and new_neg and old_neg != new_neg:
        pk = f"negation:{old_neg}→{new_neg}"
        label = new_neg.replace("_", " ")
        sug = _check_and_suggest(user, pk, lambda label=label, new_neg=new_neg: _make_suggest(
            "style_negation_phrasing", new_neg,
            f"You keep rephrasing negations to <b>{label}</b> style — set as default?",
            pk,
        ))
        if sug:
            return sug

    # Numerals — single-word only; same-combo rule uses preceding word from pre_context
    o_words = o_clean.split()
    n_words = n_clean.split()
    if len(o_words) == 1 and len(n_words) == 1:
        ow, nw = o_words[0], n_words[0]
        roman_val = _ROMAN_TO_INT.get(ow)
        if roman_val is not None and nw.isdigit() and int(nw) == roman_val:
            ctx = (pre_context or "").strip()
            preceding = re.sub(r"[^\w]", "", ctx.split()[-1]).lower() if ctx else ""
            pk = f"numeral:roman→arabic:{preceding}:{ow}"
            lbl = f"{preceding} {old.strip().upper()}".strip()
            repl = f"{preceding} {new.strip()}".strip()
            return _check_and_suggest(user, pk, lambda lbl=lbl, repl=repl, pk=pk: _make_suggest(
                "numeral_correction", "",
                f"You keep changing <b>{lbl}</b> to <b>{repl}</b> — remember this preference?",
                pk,
                numeral_pattern=lbl, numeral_replacement=repl,
            ))
        arabic_val = int(ow) if ow.isdigit() else None
        if arabic_val is not None:
            roman = _INT_TO_ROMAN.get(arabic_val, "").lower()
            if roman and roman == nw:
                ctx = (pre_context or "").strip()
                preceding = re.sub(r"[^\w]", "", ctx.split()[-1]).lower() if ctx else ""
                pk = f"numeral:arabic→roman:{preceding}:{ow}"
                lbl = f"{preceding} {old.strip()}".strip()
                repl = f"{preceding} {new.strip().upper()}".strip()
                return _check_and_suggest(user, pk, lambda lbl=lbl, repl=repl, pk=pk: _make_suggest(
                    "numeral_correction", "",
                    f"You keep changing <b>{lbl}</b> to <b>{repl}</b> — remember this preference?",
                    pk,
                    numeral_pattern=lbl, numeral_replacement=repl,
                ))

    return None


def _detect_report_drift(
    user: dict, original_report: str, edited_report: str
) -> Optional[dict]:
    """Detect style drift by diffing the LLM-generated report against the user's final edit.

    Checks impression style and date format — patterns not visible from single word edits.
    """
    if not original_report or not edited_report:
        return None
    # Safety cap
    orig = original_report[:20_000]
    edit = edited_report[:20_000]

    # 4. Impression style
    def _extract_impression(text: str) -> str:
        m = re.search(
            r'(?:^|\n)[^\n]*(?:IMPRESSION|Impression)\s*:?\s*\n(.*?)(?:\n\n|\Z)',
            text, re.DOTALL,
        )
        return m.group(1).strip() if m else ""

    def _imp_style(imp: str) -> Optional[str]:
        lines = [ln.strip() for ln in imp.split("\n") if ln.strip()]
        if len(lines) < 2:
            return None
        n_bullet = sum(1 for ln in lines if re.match(r"^[-*•]\s", ln))
        n_num = sum(1 for ln in lines if re.match(r"^\d+[.)]\s", ln))
        if n_bullet >= len(lines) // 2:
            return "bulleted"
        if n_num >= len(lines) // 2:
            return "numbered"
        return "prose"

    old_imp = _imp_style(_extract_impression(orig))
    new_imp = _imp_style(_extract_impression(edit))
    if old_imp and new_imp and old_imp != new_imp:
        pk = f"impression:{old_imp}→{new_imp}"
        sug = _check_and_suggest(user, pk, lambda new_imp=new_imp: _make_suggest(
            "style_impression_style", new_imp,
            f"You keep reformatting your Impression to <b>{new_imp}</b> style — set as default?",
            pk,
        ))
        if sug:
            return sug

    # 6. Date format
    def _date_fmt(text: str) -> Optional[str]:
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b", text):
            return "yyyy_mm_dd"
        hits = re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
        if hits:
            if any(int(h[0]) > 12 for h in hits):
                return "dd_mm_yyyy"
            if any(int(h[1]) > 12 for h in hits):
                return "mm_dd_yyyy"
            return "dd_mm_yyyy"
        return None

    old_fmt = _date_fmt(orig)
    new_fmt = _date_fmt(edit)
    if old_fmt and new_fmt and old_fmt != new_fmt:
        pk = f"date:{old_fmt}→{new_fmt}"
        friendly = new_fmt.replace("_", "/")
        sug = _check_and_suggest(user, pk, lambda friendly=friendly, new_fmt=new_fmt: _make_suggest(
            "style_date_format", new_fmt,
            f"You keep reformatting dates to <b>{friendly}</b> — set as default?",
            pk,
        ))
        if sug:
            return sug

    return None


_MOCK_TRANSCRIPTION = (
    "CT chest with contrast. "
    "The lungs are clear. No focal consolidation, pleural effusion, or pneumothorax. "
    "The heart size is normal. The mediastinum is unremarkable. "
    "No axillary, mediastinal, or hilar lymphadenopathy. "
    "Impression: No acute cardiopulmonary abnormality."
)

_MOCK_REPORT = """\
CT CHEST WITH CONTRAST

TECHNIQUE: Axial CT images of the chest were obtained with IV contrast.

FINDINGS:

Lungs: Clear bilaterally. No focal consolidation, mass, nodule, or pleural effusion.
       No pneumothorax.

Heart: Normal in size. No pericardial effusion.

Mediastinum: Normal width. No lymphadenopathy.

IMPRESSION:
1. No acute cardiopulmonary abnormality.
"""

_MOCK_MODE = bool(os.environ.get("VOXRAD_MOCK_MODE"))


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    template_name: Optional[str] = Form(None),
    whisper_prompt: Optional[str] = Form(None),
    selected_text: Optional[str] = Form(None),
    user: dict = Depends(_verify_auth),
):
    """Accept a WebM audio blob, transcribe via Whisper-compatible API."""
    if _MOCK_MODE:
        _ = await audio.read()
        _prune_sessions()
        session_id = _create_session(_MOCK_TRANSCRIPTION)
        logger.info("[mock] Returning canned transcription for session %s", session_id)
        return {"transcription": _MOCK_TRANSCRIPTION, "session_id": session_id}

    if not config.TRANSCRIPTION_API_KEY:
        raise HTTPException(
            status_code=503, detail="Transcription API key not loaded on server."
        )

    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    if len(audio_bytes) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio exceeds 100 MB limit.")


    tmp_path = None
    try:
        # Save to temp file — suffix preserves format for Whisper
        suffix = ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Build ASR prompt.
        # whisper_prompt (from voice-edit mode) overrides everything — it is the
        # surrounding transcript text, which is what Whisper's prompt parameter
        # is actually designed for.  Using the vocabulary list as the prompt for
        # short voice-edit clips causes Whisper to hallucinate completions of it.
        user_vocab = _load_user_vocab(user)
        if whisper_prompt is not None:
            asr_prompt = whisper_prompt[:896]
        else:
            asr_prompt = _RADIOLOGY_PROMPT
            if template_name:
                content = _load_template_content(template_name)
                match = re.search(
                    r"\[correct spellings\](.*?)\[correct spellings\]", content, re.DOTALL
                )
                if match:
                    asr_prompt = match.group(1).strip()[:896]  # Groq hard limit
            # Append user-learned vocab, truncating to the 896-char Groq limit.
            if user_vocab:
                addition = ", " + ", ".join(user_vocab)
                asr_prompt = (asr_prompt + addition)[:896]

        client = OpenAI(
            api_key=config.TRANSCRIPTION_API_KEY,
            base_url=config.TRANSCRIPTION_BASE_URL,
        )
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(os.path.basename(tmp_path), f.read()),
                model=config.SELECTED_TRANSCRIPTION_MODEL,
                prompt=asr_prompt,
                language="en",
                temperature=0.0,
            )

        text = result.text.strip()
        text = correct_mri_platform(text, template_name or "")
        # Hallucination filtering. Voice-edit mode uses a permissive variant
        # that allows short valid replacements ("normal", "intact", "no") but
        # still rejects known Whisper garbage phrases (e.g. "thank you for
        # watching") that fire on silent or noisy audio.
        is_voice_edit = whisper_prompt is not None
        hallucinated = (
            _is_voice_edit_hallucination(text, asr_prompt)
            if is_voice_edit
            else _is_hallucination(text, asr_prompt)
        )
        if hallucinated:
            logger.debug("Discarded hallucination (voice_edit=%s): %r", is_voice_edit, text)
            return {"transcription": "", "session_id": ""}

        _prune_sessions()
        session_id = _create_session(text)
        logger.info("Transcription complete for session %s (%d chars)", session_id, len(text))

        response = {"transcription": text, "session_id": session_id}
        if is_voice_edit and selected_text:
            # Vocab suggestion: phonetically similar replacement → offer to remember spelling
            vocab_sug = _should_suggest_vocab(user, selected_text, text)
            if vocab_sug:
                response["suggest_vocab"] = vocab_sug
            # Style drift: repeated correction pattern → offer to change a style setting
            style_sug = _detect_style_drift(user, selected_text, text, whisper_prompt or "")
            if style_sug:
                response["suggest_setting"] = style_sug
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Transcription failed: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Transcription failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


class VocabAddRequest(BaseModel):
    term: str


@app.post("/vocab/add")
def vocab_add(req: VocabAddRequest, user: dict = Depends(_verify_auth)):
    """Append a term to the signed-in user's custom vocabulary."""
    term = (req.term or "").strip()
    if not term:
        raise HTTPException(status_code=400, detail="Empty term.")
    if len(term) > 64:
        raise HTTPException(status_code=400, detail="Term too long (max 64 chars).")
    added = _add_to_user_vocab(user, term)
    return {"added": added, "term": term}


@app.get("/vocab")
def vocab_list(user: dict = Depends(_verify_auth)):
    """Return the signed-in user's custom vocabulary."""
    return {"terms": _load_user_vocab(user)}


# ---------------------------------------------------------------------------
# Style drift endpoints
# ---------------------------------------------------------------------------

class TrackReportEditRequest(BaseModel):
    original_report: str
    edited_report: str


@app.post("/api/track-report-edit")
def api_track_report_edit(req: TrackReportEditRequest, user: dict = Depends(_verify_auth)):
    """Diff the LLM-generated report against the user's edited version.

    Called (fire-and-forget) when the user copies a report. Detects
    impression-style and date-format drift and returns a suggest_setting.
    """
    suggestion = _detect_report_drift(user, req.original_report, req.edited_report)
    return {"suggest_setting": suggestion}


class StyleSuggestionRequest(BaseModel):
    pattern_key: str
    setting: Optional[str] = None
    value: Optional[str] = None
    numeral_pattern: Optional[str] = None
    numeral_replacement: Optional[str] = None


@app.post("/api/style-suggestion/apply")
def api_style_suggestion_apply(req: StyleSuggestionRequest, user: dict = Depends(_verify_auth)):
    """Apply a suggested style change and mark the pattern as permanently dismissed."""
    _save_dismissed(user, req.pattern_key)

    if req.setting == "numeral_correction":
        if req.numeral_pattern and req.numeral_replacement:
            _add_numeral_pref(user, req.numeral_pattern, req.numeral_replacement)
        return {"ok": True}

    setting = req.setting or ""
    style_key = _STYLE_FIELD_MAP.get(setting)
    if not style_key:
        raise HTTPException(status_code=400, detail=f"Unknown setting: {setting!r}")
    allowed = _STYLE_ALLOWED.get(setting, set())
    if req.value not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid value {req.value!r} for {setting}")

    if oauth_enabled() and user.get("id") is not None:
        existing = get_user_style(user["id"])
        existing[style_key] = req.value
        save_user_style(user["id"], existing)
    else:
        setattr(config, setting, req.value)
        save_web_settings()

    logger.info("Applied style %s=%r for %s", setting, req.value, _get_username(user))
    return {"ok": True}


@app.post("/api/style-suggestion/dismiss")
def api_style_suggestion_dismiss(req: StyleSuggestionRequest, user: dict = Depends(_verify_auth)):
    """Permanently dismiss a style suggestion without applying it."""
    _save_dismissed(user, req.pattern_key)
    return {"ok": True}


class KeyboardEditRequest(BaseModel):
    old_text: str
    new_text: str


@app.post("/api/check-edit-suggestion")
def api_check_edit_suggestion(req: KeyboardEditRequest, user: dict = Depends(_verify_auth)):
    """Check whether a keyboard edit of the transcript warrants a vocab or style suggestion.

    Word-diffs the old and new transcript, then runs the same vocab + style-drift
    checks as the voice-edit pipeline. Called on textarea blur when text changed.
    """
    old_words = req.old_text.split()
    new_words = req.new_text.split()
    if old_words == new_words:
        return {"suggest_vocab": None, "suggest_setting": None}

    matcher = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)
    suggest_vocab = None
    suggest_setting = None

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        old_phrase = " ".join(old_words[i1:i2])
        new_phrase = " ".join(new_words[j1:j2])
        pre_context = " ".join(old_words[:i1])

        # Vocab: try the full block first, then leading/trailing word pairs.
        # The leading-pair check catches "edema" -> "oedema with no consolidation"
        # where the user's correction is followed by STT extension.
        if not suggest_vocab:
            candidates = []
            if i2 - i1 == 1 and j2 - j1 == 1:
                candidates.append((old_phrase, new_phrase))
            else:
                if i2 - i1 >= 1 and j2 - j1 >= 1:
                    candidates.append((old_words[i1], new_words[j1]))     # leading pair
                    candidates.append((old_words[i2 - 1], new_words[j2 - 1]))  # trailing pair
            for ow, nw in candidates:
                sv = _should_suggest_vocab(user, ow, nw)
                if sv:
                    suggest_vocab = sv
                    break

        # Style drift: works on any phrase length
        if not suggest_setting:
            ss = _detect_style_drift(user, old_phrase, new_phrase, pre_context)
            if ss:
                suggest_setting = ss

    return {"suggest_vocab": suggest_vocab, "suggest_setting": suggest_setting}


class FormatRequest(BaseModel):
    transcription: str
    template_name: Optional[str] = None
    session_id: Optional[str] = None
    source_kind: Literal["dictation", "worksheet"] = "dictation"
    # Patient context fields
    patient_name: Optional[str] = None
    patient_dob: Optional[str] = None
    patient_id: Optional[str] = None
    accession: Optional[str] = None
    modality: Optional[str] = None
    body_part: Optional[str] = None
    referring_physician: Optional[str] = None
    radiologist: Optional[str] = None
    # A prior is included only after explicit radiologist selection in the UI.
    comparison_report: Optional[str] = None
    comparison_date: Optional[str] = None
    cap_layout: Optional[Literal["components", "regions"]] = None
    # Comparison drafts must not create duplicate clinical exports. The chosen
    # draft can still be signed off through the normal audited workflow.
    comparison_draft: bool = False


_CT_CAP_TEMPLATE = "CT_CAP_Staging.txt"
_CT_CAP_REGIONS_TEMPLATE = "_CT_CAP_Staging_Regions.txt"


def _format_template_content(req: "FormatRequest") -> str:
    """Resolve the visible CT CAP template to its selected report layout."""
    name = req.template_name
    if name == _CT_CAP_TEMPLATE and req.cap_layout == "regions":
        name = _CT_CAP_REGIONS_TEMPLATE
    return _load_template_content(name) if name else ""


def _mock_report_for_request(req: "FormatRequest") -> str:
    if req.template_name != _CT_CAP_TEMPLATE:
        return _MOCK_REPORT
    if req.cap_layout == "regions":
        return (
            "**EXAM:** CT chest, abdomen and pelvis\n\n"
            "**FINDINGS:**\n\n**Chest:** No thoracic metastatic disease.\n\n"
            "**Abdomen and Pelvis:** No abdominopelvic metastatic disease.\n\n"
            "**IMPRESSION:**\n1. No metastatic disease in the chest, abdomen or pelvis."
        )
    return (
        "**EXAM:** CT chest, abdomen and pelvis\n\n"
        "**FINDINGS:**\n\n**Lungs and Airways:** No pulmonary metastasis.\n\n"
        "**Liver and Biliary System:** No hepatic metastasis.\n\n"
        "**Lymph Nodes:** No lymphadenopathy.\n\n"
        "**IMPRESSION:**\n1. No metastatic disease in the chest, abdomen or pelvis."
    )


def _hl7_outbox_dir() -> Optional[str]:
    """Resolve the HL7 outbox directory, falling back to {save_directory}/hl7_outbox."""
    if not config.hl7_export_enabled:
        return None
    if config.hl7_outbox_path:
        return config.hl7_outbox_path
    if config.save_directory:
        return os.path.join(config.save_directory, "hl7_outbox")
    return None


def _hl7_inbox_dir() -> Optional[str]:
    """Resolve the HL7 order inbox directory used by the worklist."""
    if config.hl7_inbox_path:
        return config.hl7_inbox_path
    if config.save_directory:
        return os.path.join(config.save_directory, "hl7_inbox")
    return None


def _sr_outbox_dir() -> Optional[str]:
    """Resolve the DICOM SR outbox directory, or None when export is disabled."""
    if not config.dicom_sr_export_enabled:
        return None
    if config.dicom_sr_outbox_path:
        return config.dicom_sr_outbox_path
    if config.save_directory:
        return os.path.join(config.save_directory, "sr_outbox")
    return None


def _patient_context(req: "FormatRequest") -> Optional[dict]:
    """Build a patient context dict from a FormatRequest, returning None if all fields empty."""
    ctx = {k: v for k, v in {
        "patient_name": req.patient_name,
        "patient_dob": req.patient_dob,
        "patient_id": req.patient_id,
        "accession": req.accession,
        "modality": req.modality,
        "body_part": req.body_part,
        "referring_physician": req.referring_physician,
        "radiologist": req.radiologist,
        "comparison_report": req.comparison_report,
        "comparison_date": req.comparison_date,
    }.items() if v}
    return ctx or None


def _retain_quality_pair(
    *,
    user: dict,
    style: Optional[dict],
    kind: str,
    source_text: str,
    output_text: str,
    patient_context: Optional[dict],
    feedback_text: Optional[str] = None,
    template_name: Optional[str] = None,
    source_kind: Optional[str] = None,
) -> None:
    """Best-effort opt-in retention. A storage fault must not block reporting."""
    if source_kind == "worksheet":
        return
    try:
        save_quality_sample(
            user_id=user.get("id"),
            enabled=bool((style or {}).get("retain_deidentified_samples", False)),
            kind=kind,
            source_text=source_text,
            output_text=output_text,
            feedback_text=feedback_text,
            patient_context=patient_context,
            template_name=template_name,
            model_name=config.SELECTED_MODEL or None,
            source_kind=source_kind,
        )
    except Exception as exc:
        logger.warning("Could not retain de-identified quality sample (%s).", type(exc).__name__)


@app.post("/api/worksheet/extract")
async def extract_worksheet(
    images: list[UploadFile] = File(...),
    modality: Optional[str] = Form(None),
    body_part: Optional[str] = Form(None),
    user: dict = Depends(_verify_auth),
):
    """Extract entered observations from pasted sonographer worksheet images.

    Image bytes are kept in memory only for this request, sent to the configured
    text/vision model, and never written to RadSpeed's data volume.
    """
    if not images or len(images) > MAX_WORKSHEET_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Use between 1 and {MAX_WORKSHEET_IMAGES} worksheet screenshots.",
        )

    if _MOCK_MODE:
        for image in images:
            await image.read(MAX_WORKSHEET_IMAGE_BYTES + 1)
            await image.close()
        return {
            "findings": (
                "Study / renal ultrasound: Worksheet observations.\n"
                "Right kidney: 10.8 cm; no pelvicaliectasis documented.\n"
                "Left kidney: 10.2 cm; mild pelvicaliectasis documented.\n"
                "Bladder / post-void residual: 24 mL.\n"
                "Ureteric jets: Bilateral jets documented."
            ),
            "image_count": len(images),
        }

    if not config.TEXT_API_KEY:
        raise HTTPException(
            status_code=503, detail="Text/vision model API key not loaded on server."
        )

    payloads: list[bytes] = []
    try:
        for image in images:
            # Bound each in-memory read even if a caller bypasses the browser
            # limit. Starlette may spool larger uploads to disk, but RadSpeed
            # never needs to load more than the accepted limit into memory.
            payloads.append(await image.read(MAX_WORKSHEET_IMAGE_BYTES + 1))
    finally:
        for image in images:
            await image.close()

    try:
        validated = validate_worksheet_images(payloads)
        findings = extract_worksheet_findings(
            validated,
            modality=modality,
            body_part=body_part,
        )
    except WorksheetImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AuthenticationError as exc:
        logger.error("Text model API key rejected during worksheet extraction.")
        raise HTTPException(
            status_code=503,
            detail="Text/vision model API key was rejected. Update it in Settings.",
        ) from exc
    except APIStatusError as exc:
        # Provider error text can repeat image content. Expose only stable,
        # non-clinical status fields so failures can be diagnosed safely.
        code = getattr(exc, "code", None)
        safe_code = code if isinstance(code, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", code) else "unknown"
        logger.error("Worksheet model request failed: HTTP %d, code %s.", exc.status_code, safe_code)
        detail = (
            "OpenAI API credit is exhausted. Add credit in OpenAI billing and try again."
            if safe_code in {"credit_balance_exhausted", "insufficient_quota"}
            else f"Worksheet model request failed (provider HTTP {exc.status_code}, code {safe_code})."
        )
        raise HTTPException(
            status_code=503,
            detail=detail,
        ) from exc
    except APITimeoutError as exc:
        logger.error("Worksheet model request timed out.")
        raise HTTPException(status_code=504, detail="Worksheet reading timed out. Try again.") from exc
    except APIConnectionError as exc:
        logger.error("Worksheet model connection failed.")
        raise HTTPException(status_code=503, detail="Worksheet model connection failed. Try again.") from exc
    except Exception as exc:
        # Do not log provider error text: some providers may echo part of the
        # request, and worksheet screenshots can contain health information.
        logger.error("Worksheet extraction failed (%s).", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail=f"Worksheet reading failed ({type(exc).__name__}).",
        ) from exc

    if findings == "NO_EXTRACTABLE_FINDINGS":
        raise HTTPException(
            status_code=422,
            detail=(
                "No entered worksheet findings were detected. Include the marked "
                "findings or measurement area and try again."
            ),
        )

    log_event(
        user_id=user.get("id"),
        event_type="worksheet_extract",
        metadata={"image_count": len(validated), "chars": len(findings)},
    )
    return {"findings": findings, "image_count": len(validated)}


@app.post("/api/worksheet/draft")
async def draft_worksheet(
    images: list[UploadFile] = File(...),
    modality: Optional[str] = Form(None),
    body_part: Optional[str] = Form(None),
    reasoning_effort: str = Form("high"),
    user: dict = Depends(_verify_auth),
):
    """Experimental one-request worksheet reading and report drafting."""
    if not images or len(images) > MAX_WORKSHEET_IMAGES:
        raise HTTPException(status_code=400, detail=f"Use between 1 and {MAX_WORKSHEET_IMAGES} worksheet screenshots.")
    if reasoning_effort not in {"low", "high"}:
        raise HTTPException(status_code=400, detail="Choose low or high worksheet reasoning.")
    if not _MOCK_MODE and not config.TEXT_API_KEY:
        raise HTTPException(status_code=503, detail="Text/vision model API key not loaded on server.")
    payloads: list[bytes] = []
    try:
        for image in images:
            payloads.append(await image.read(MAX_WORKSHEET_IMAGE_BYTES + 1))
    finally:
        for image in images:
            await image.close()
    try:
        validated = validate_worksheet_images(payloads)
        if _MOCK_MODE:
            notes = "Right kidney: 10.8 cm.\nLeft kidney: 10.2 cm; mild pelvicaliectasis."
            report = "**FINDINGS:**\nRight kidney: 10.8 cm. Left kidney: 10.2 cm; mild pelvicaliectasis.\n\n**IMPRESSION:**\nMild left pelvicaliectasis."
        else:
            draft = draft_worksheet_report(
                validated,
                template_content=_load_template_content("Ultrasound_Worksheet.txt"),
                modality=modality,
                body_part=body_part,
                style=_user_style(user),
                reasoning_effort=reasoning_effort,
            )
            notes, report = draft.source_notes, draft.report
    except WorksheetImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AuthenticationError as exc:
        logger.error("Text model API key rejected during worksheet draft.")
        raise HTTPException(status_code=503, detail="Text/vision model API key was rejected. Update it in Settings.") from exc
    except APIStatusError as exc:
        code = getattr(exc, "code", None)
        safe_code = code if isinstance(code, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", code) else "unknown"
        logger.error("Worksheet draft model request failed: HTTP %d, code %s.", exc.status_code, safe_code)
        detail = "OpenAI API credit is exhausted. Add credit in OpenAI billing and try again." if safe_code in {"credit_balance_exhausted", "insufficient_quota"} else f"Worksheet model request failed (provider HTTP {exc.status_code}, code {safe_code})."
        raise HTTPException(status_code=503, detail=detail) from exc
    except APITimeoutError as exc:
        logger.error("Worksheet draft model request timed out.")
        raise HTTPException(status_code=504, detail="Worksheet reading timed out. Try again.") from exc
    except APIConnectionError as exc:
        logger.error("Worksheet draft model connection failed.")
        raise HTTPException(status_code=503, detail="Worksheet model connection failed. Try again.") from exc
    except Exception as exc:
        logger.error("Worksheet draft failed (%s).", type(exc).__name__)
        raise HTTPException(status_code=503, detail=f"Worksheet reading failed ({type(exc).__name__}).") from exc
    if notes == "NO_EXTRACTABLE_FINDINGS":
        raise HTTPException(status_code=422, detail="No entered worksheet findings were detected.")
    log_event(user_id=user.get("id"), event_type="worksheet_draft", metadata={"image_count": len(validated), "chars": len(report)})
    return {"findings": notes, "report": report, "image_count": len(validated)}


@app.post("/format")
def format_report(req: FormatRequest, user: dict = Depends(_verify_auth)):
    """Format a transcription into a structured radiology report."""
    if _MOCK_MODE:
        logger.info("[mock] Returning canned report")
        return {"report": _mock_report_for_request(req), "fhir_saved": False}

    if not config.TEXT_API_KEY:
        raise HTTPException(
            status_code=503, detail="Text model API key not loaded on server."
        )

    _style = dict(_user_style(user) or {})
    numeral_prefs = _load_numeral_prefs(user)
    if numeral_prefs:
        _style["numeral_corrections"] = numeral_prefs
    with _format_lock:
        old_template = config.global_md_text_content
        config.global_md_text_content = _format_template_content(req)
        try:
            report = format_text(
                req.transcription,
                patient_context=_patient_context(req),
                style=_style,
                source_kind=req.source_kind,
            )
        finally:
            config.global_md_text_content = old_template

    if report is None:
        raise HTTPException(status_code=503, detail="Report generation failed.")

    fhir_saved = False
    if not req.comparison_draft and _user_fhir_enabled(user):
        path = save_fhir_report(
            report_text=report,
            template_name=req.template_name,
            patient_id=req.patient_id or None,
            accession=req.accession or None,
            radiologist=req.radiologist or None,
        )
        fhir_saved = path is not None

    hl7_saved = False
    _hl7_dir = _hl7_outbox_dir()
    if not req.comparison_draft and _hl7_dir:
        try:
            path = save_hl7_report(
                report_text=report,
                outbox_path=_hl7_dir,
                patient_context=_patient_context(req),
                template_name=req.template_name,
                sending_facility=config.hl7_sending_facility or "VOXRAD",
                receiving_facility=config.hl7_receiving_facility or "",
            )
            hl7_saved = path is not None
        except Exception as e:
            logger.warning("HL7 save failed: %s", e)

    sr_saved = False
    _sr_dir = _sr_outbox_dir()
    if not req.comparison_draft and _sr_dir:
        try:
            path = save_dicom_sr_report(
                report_text=report,
                outbox_path=_sr_dir,
                patient_context=_patient_context(req),
                template_name=req.template_name,
                institution_name=config.dicom_sr_institution_name or "VOXRAD",
            )
            sr_saved = path is not None
        except Exception as e:
            logger.warning("DICOM SR save failed: %s", e)

    logger.info(
        "Report formatted (%d chars), fhir_saved=%s, hl7_saved=%s, sr_saved=%s",
        len(report), fhir_saved, hl7_saved, sr_saved,
    )
    log_event(
        user_id=user.get("id"),
        event_type="format",
        accession=req.accession or None,
        metadata={
            "template": req.template_name,
            "chars": len(report),
            "fhir_saved": fhir_saved,
            "hl7_saved": hl7_saved,
            "sr_saved": sr_saved,
            "cap_layout": req.cap_layout,
            "comparison_draft": req.comparison_draft,
        },
    )
    _retain_quality_pair(
        user=user,
        style=_style,
        kind="format",
        source_text=req.transcription,
        output_text=report,
        patient_context=_patient_context(req),
        template_name=req.template_name,
        source_kind=req.source_kind,
    )
    return {
        "report": report,
        "fhir_saved": fhir_saved,
        "hl7_saved": hl7_saved,
        "sr_saved": sr_saved,
    }


@app.post("/format/stream")
def format_report_stream(req: FormatRequest, user: dict = Depends(_verify_auth)):
    """Stream a structured radiology report as Server-Sent Events.

    Each SSE event is one of:
      data: {"token": "..."}       — a text chunk from the LLM
      data: {"done": true, "fhir_saved": bool}  — signals completion
      data: {"error": "..."}       — an error occurred
    """
    if _MOCK_MODE:
        def _mock_stream():
            import time
            for word in _mock_report_for_request(req).split(" "):
                yield f'data: {{"token": {json.dumps(word + " ")}}}\n\n'
                time.sleep(0.03)
            report = _mock_report_for_request(req)
            yield f'data: {json.dumps({"done": True, "fhir_saved": False, "report": report})}\n\n'
        return StreamingResponse(_mock_stream(), media_type="text/event-stream")

    if not config.TEXT_API_KEY:
        def _err():
            yield 'data: {"error": "Text model API key not loaded on server."}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    _ctx = _patient_context(req)
    _style = dict(_user_style(user) or {})
    numeral_prefs = _load_numeral_prefs(user)
    if numeral_prefs:
        _style["numeral_corrections"] = numeral_prefs

    # Resolve the template once, per-request, and pass it explicitly into the
    # generator. Previously this mutated the process-global
    # config.global_md_text_content and released the lock before streaming, so
    # a concurrent request could swap the template mid-stream and a report
    # would be structured against the wrong template.
    _template_snapshot = _format_template_content(req)

    def _generate():
        try:
            full_report = ""
            for chunk in stream_format_text(
                req.transcription,
                patient_context=_ctx,
                style=_style,
                template_content=_template_snapshot,
                source_kind=req.source_kind,
            ):
                if chunk:
                    full_report += chunk
                    yield f'data: {json.dumps({"token": chunk})}\n\n'
        except (TextModelAuthError, TextModelUnavailableError) as exc:
            logger.error("Streaming format failed (%s).", type(exc).__name__)
            yield f'data: {json.dumps({"error": str(exc)})}\n\n'
            return
        except Exception as exc:
            logger.error("Streaming format failed (%s).", type(exc).__name__)
            yield 'data: {"error": "Report generation failed. Try again."}\n\n'
            return

        # Apply post-processing (capitalise after colons) to the full report.
        # Send corrected version so the client can replace the streamed text.
        corrected_report = postprocess_report(
            reinsert_identifiers(full_report, _ctx), _style
        )

        fhir_saved = False
        if not req.comparison_draft and _user_fhir_enabled(user) and corrected_report:
            try:
                path = save_fhir_report(
                    report_text=corrected_report,
                    template_name=req.template_name,
                    patient_id=req.patient_id or None,
                    accession=req.accession or None,
                    radiologist=req.radiologist or None,
                )
                fhir_saved = path is not None
            except Exception as e:
                logger.warning("FHIR save failed during streaming: %s", e)

        hl7_saved = False
        _hl7_dir = _hl7_outbox_dir()
        if not req.comparison_draft and _hl7_dir and corrected_report:
            try:
                path = save_hl7_report(
                    report_text=corrected_report,
                    outbox_path=_hl7_dir,
                    patient_context=_ctx,
                    template_name=req.template_name,
                    sending_facility=config.hl7_sending_facility or "VOXRAD",
                    receiving_facility=config.hl7_receiving_facility or "",
                )
                hl7_saved = path is not None
            except Exception as e:
                logger.warning("HL7 save failed during streaming: %s", e)

        sr_saved = False
        _sr_dir = _sr_outbox_dir()
        if not req.comparison_draft and _sr_dir and corrected_report:
            try:
                path = save_dicom_sr_report(
                    report_text=corrected_report,
                    outbox_path=_sr_dir,
                    patient_context=_ctx,
                    template_name=req.template_name,
                    institution_name=config.dicom_sr_institution_name or "VOXRAD",
                )
                sr_saved = path is not None
            except Exception as e:
                logger.warning("DICOM SR save failed during streaming: %s", e)

        log_event(
            user_id=user.get("id"),
            event_type="format",
            accession=req.accession or None,
            metadata={
                "template": req.template_name,
                "chars": len(corrected_report),
                "fhir_saved": fhir_saved,
                "hl7_saved": hl7_saved,
                "sr_saved": sr_saved,
                "stream": True,
                "cap_layout": req.cap_layout,
                "comparison_draft": req.comparison_draft,
            },
        )
        _retain_quality_pair(
            user=user,
            style=_style,
            kind="format",
            source_text=req.transcription,
            output_text=corrected_report,
            patient_context=_ctx,
            template_name=req.template_name,
            source_kind=req.source_kind,
        )
        yield f'data: {json.dumps({"done": True, "fhir_saved": fhir_saved, "hl7_saved": hl7_saved, "sr_saved": sr_saved, "report": corrected_report})}\n\n'

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Voice-feedback report refinement
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    report: str
    feedback: str
    selected_text: str = ""
    # Optional patient fields so identifier minimisation can scrub the report
    # before it is sent to the model and restore the values afterwards.
    patient_name: Optional[str] = None
    patient_dob: Optional[str] = None
    patient_id: Optional[str] = None
    accession: Optional[str] = None
    referring_physician: Optional[str] = None


@app.post("/format/feedback")
def format_feedback(req: FeedbackRequest, user: dict = Depends(_verify_auth)):
    """Apply radiologist verbal feedback to refine an already-generated report.

    Accepts a full report, a feedback transcription, and an optional selected
    passage.  If selected_text is provided, only that passage is revised.
    Returns {"report": "<corrected markdown>"}.
    """
    if _MOCK_MODE:
        tag = f" [targeted: {req.selected_text[:40]}…]" if req.selected_text else ""
        return {"report": req.report + f"\n\n*[Feedback applied{tag}: {req.feedback}]*"}

    if not config.TEXT_API_KEY:
        raise HTTPException(
            status_code=503, detail="Text model API key not loaded on server."
        )

    feedback_ctx = {
        k: v for k, v in {
            "patient_name": req.patient_name,
            "patient_dob": req.patient_dob,
            "patient_id": req.patient_id,
            "accession": req.accession,
            "referring_physician": req.referring_physician,
        }.items() if v
    } or None
    try:
        corrected = apply_report_feedback(
            req.report, req.feedback, req.selected_text, patient_context=feedback_ctx
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Report refinement failed: {e}")

    corrected = postprocess_report(corrected, _user_style(user))

    _retain_quality_pair(
        user=user,
        style=_user_style(user),
        kind="refine",
        source_text=req.report,
        output_text=corrected,
        feedback_text=req.feedback,
        patient_context=feedback_ctx,
        source_kind="dictation",
    )

    return {"report": corrected}


# ---------------------------------------------------------------------------
# Phase 1: Sign-off, amendments, audit log
# ---------------------------------------------------------------------------

class SignOffRequest(BaseModel):
    report_text: str
    template_name: Optional[str] = None
    patient_name: Optional[str] = None
    patient_dob: Optional[str] = None
    patient_id: Optional[str] = None
    accession: Optional[str] = None
    modality: Optional[str] = None
    body_part: Optional[str] = None
    referring_physician: Optional[str] = None
    radiologist: Optional[str] = None


class AmendmentRequest(BaseModel):
    prior_report_id: int
    report_text: str
    amendment_reason: str


def _exports_for_signed_report(
    *,
    report_text: str,
    template_name: Optional[str],
    patient_context: Optional[dict],
    user: dict,
    user_id_for_audit: Optional[int],
    accession: Optional[str],
    report_id: int,
    is_amendment: bool,
) -> dict:
    """Run the HL7 / DICOM SR / FHIR export pipeline and audit each result."""
    exports = {"hl7_saved": False, "sr_saved": False, "fhir_saved": False}

    if _user_fhir_enabled(user):
        try:
            path = save_fhir_report(
                report_text=report_text,
                template_name=template_name,
                patient_id=(patient_context or {}).get("patient_id"),
                accession=accession,
                radiologist=(patient_context or {}).get("radiologist"),
            )
            exports["fhir_saved"] = path is not None
            log_event(
                user_id=user_id_for_audit,
                event_type="export_fhir",
                accession=accession,
                report_id=report_id,
                metadata={"path": path, "amendment": is_amendment},
            )
        except Exception as e:
            logger.warning("FHIR export at sign-off failed: %s", e)

    _hl7_dir = _hl7_outbox_dir()
    if _hl7_dir:
        try:
            path = save_hl7_report(
                report_text=report_text,
                outbox_path=_hl7_dir,
                patient_context=patient_context,
                template_name=template_name,
                sending_facility=config.hl7_sending_facility or "VOXRAD",
                receiving_facility=config.hl7_receiving_facility or "",
            )
            exports["hl7_saved"] = path is not None
            log_event(
                user_id=user_id_for_audit,
                event_type="export_hl7",
                accession=accession,
                report_id=report_id,
                metadata={"path": path, "amendment": is_amendment},
            )
        except Exception as e:
            logger.warning("HL7 export at sign-off failed: %s", e)

    _sr_dir = _sr_outbox_dir()
    if _sr_dir:
        try:
            path = save_dicom_sr_report(
                report_text=report_text,
                outbox_path=_sr_dir,
                patient_context=patient_context,
                template_name=template_name,
                institution_name=config.dicom_sr_institution_name or "VOXRAD",
            )
            exports["sr_saved"] = path is not None
            log_event(
                user_id=user_id_for_audit,
                event_type="export_sr",
                accession=accession,
                report_id=report_id,
                metadata={"path": path, "amendment": is_amendment},
            )
        except Exception as e:
            logger.warning("DICOM SR export at sign-off failed: %s", e)

    return exports


@app.post("/api/reports/sign-off")
def api_sign_off(req: SignOffRequest, user: dict = Depends(_verify_auth)):
    """Lock a report as FINAL.

    Persists the signed version, runs the HL7 / SR / FHIR export pipeline,
    and writes audit events for each step. The user-supplied report text is
    treated as the canonical FINAL — no further LLM processing is applied.
    """
    if not req.report_text or not req.report_text.strip():
        raise HTTPException(status_code=400, detail="report_text is required")

    user_id = user.get("id") if user.get("id") is not None else None
    if user_id is None:
        # In Basic Auth mode there is no users.id row to FK to. Sign-off
        # requires per-user accountability, so insist on OAuth here.
        raise HTTPException(
            status_code=403,
            detail="Sign-off requires an authenticated user (OAuth login).",
        )

    signed_text = append_disclosure(req.report_text, req.radiologist or user.get("name"))
    req.report_text = signed_text

    saved = save_report_version(
        user_id=user_id,
        report_text=signed_text,
        status="final",
        accession=req.accession or None,
        patient_id=req.patient_id or None,
        patient_name=req.patient_name or None,
        patient_dob=req.patient_dob or None,
        modality=req.modality or None,
        body_part=req.body_part or None,
        referring=req.referring_physician or None,
        radiologist=req.radiologist or None,
        template_name=req.template_name or None,
    )

    log_event(
        user_id=user_id,
        event_type="sign_off",
        accession=req.accession or None,
        report_id=saved["id"],
        metadata={
            "version": saved["version"],
            "report_hash": saved["report_hash"],
            "template": req.template_name,
            "chars": len(req.report_text),
        },
    )

    patient_context = {
        k: v for k, v in {
            "patient_name": req.patient_name,
            "patient_dob": req.patient_dob,
            "patient_id": req.patient_id,
            "accession": req.accession,
            "modality": req.modality,
            "body_part": req.body_part,
            "referring_physician": req.referring_physician,
            "radiologist": req.radiologist,
        }.items() if v
    } or None

    exports = _exports_for_signed_report(
        report_text=req.report_text,
        template_name=req.template_name,
        patient_context=patient_context,
        user=user,
        user_id_for_audit=user_id,
        accession=req.accession or None,
        report_id=saved["id"],
        is_amendment=False,
    )

    return {"report": saved, **exports}


@app.post("/api/reports/amend")
def api_amend(req: AmendmentRequest, user: dict = Depends(_verify_auth)):
    """Create an amendment (versioned successor) of a previously signed report."""
    if not req.report_text or not req.report_text.strip():
        raise HTTPException(status_code=400, detail="report_text is required")
    if not req.amendment_reason or not req.amendment_reason.strip():
        raise HTTPException(status_code=400, detail="amendment_reason is required")

    user_id = user.get("id") if user.get("id") is not None else None
    if user_id is None:
        raise HTTPException(
            status_code=403,
            detail="Amendments require an authenticated user (OAuth login).",
        )

    prior = get_report(req.prior_report_id, user_id=user_id)
    if not prior:
        raise HTTPException(status_code=404, detail="Prior report not found")
    if prior["status"] not in ("final", "amended"):
        raise HTTPException(
            status_code=400,
            detail="Only signed reports (final / amended) can be amended.",
        )

    amended_text = append_disclosure(req.report_text, prior.get("radiologist") or user.get("name"))
    req.report_text = amended_text

    saved = save_report_version(
        user_id=user_id,
        report_text=amended_text,
        status="amended",
        accession=prior["accession"],
        patient_id=prior["patient_id"],
        patient_name=prior["patient_name"],
        patient_dob=prior["patient_dob"],
        modality=prior["modality"],
        body_part=prior["body_part"],
        referring=prior["referring"],
        radiologist=prior["radiologist"],
        template_name=prior["template_name"],
        prior_version_id=prior["id"],
        amendment_reason=req.amendment_reason,
    )

    log_event(
        user_id=user_id,
        event_type="amend",
        accession=prior["accession"],
        report_id=saved["id"],
        metadata={
            "prior_report_id": prior["id"],
            "prior_version": prior["version"],
            "version": saved["version"],
            "report_hash": saved["report_hash"],
            "reason": req.amendment_reason,
            "chars": len(req.report_text),
        },
    )

    patient_context = {
        k: v for k, v in {
            "patient_name": prior["patient_name"],
            "patient_dob": prior["patient_dob"],
            "patient_id": prior["patient_id"],
            "accession": prior["accession"],
            "modality": prior["modality"],
            "body_part": prior["body_part"],
            "referring_physician": prior["referring"],
            "radiologist": prior["radiologist"],
        }.items() if v
    } or None

    exports = _exports_for_signed_report(
        report_text=req.report_text,
        template_name=prior["template_name"],
        patient_context=patient_context,
        user=user,
        user_id_for_audit=user_id,
        accession=prior["accession"],
        report_id=saved["id"],
        is_amendment=True,
    )

    return {"report": saved, **exports}


@app.get("/api/reports")
def api_list_reports(
    accession: Optional[str] = None,
    user: dict = Depends(_verify_auth),
):
    """List versions for an accession, or recent reports for the current user."""
    user_id = user.get("id")
    if accession:
        # Scope to the caller's own reports so an accession lookup can't
        # disclose another radiologist's report text or patient identifiers.
        return {"reports": list_reports_for_accession(accession, user_id=user_id)}
    if user_id is None:
        return {"reports": []}
    return {"reports": list_reports_for_user(user_id, limit=100)}


@app.get("/api/reports/priors")
def api_list_prior_reports(
    patient_id: str,
    exclude_accession: Optional[str] = None,
    user: dict = Depends(_verify_auth),
):
    """List this radiologist's signed local reports for the same MRN."""
    user_id = user.get("id")
    if user_id is None:
        return {"reports": []}
    if not patient_id.strip():
        return {"reports": []}
    return {
        "reports": list_prior_reports_for_patient(
            user_id=user_id,
            patient_id=patient_id,
            exclude_accession=exclude_accession,
            limit=20,
        )
    }


@app.get("/api/reports/{report_id}")
def api_get_report(report_id: int, user: dict = Depends(_verify_auth)):
    rep = get_report(report_id, user_id=user.get("id"))
    if not rep:
        raise HTTPException(status_code=404, detail="Report not found")
    return rep


class FollowupSuggestRequest(BaseModel):
    report_text: str


class FollowupCreateRequest(BaseModel):
    recommendation: str
    report_id: Optional[int] = None
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    accession: Optional[str] = None
    finding: Optional[str] = None
    due_date: Optional[str] = None
    notes: Optional[str] = None


class FollowupUpdateRequest(BaseModel):
    status: Optional[str] = None
    due_date: Optional[str] = None
    notes: Optional[str] = None


def _followup_user_id(user: dict) -> int:
    user_id = user.get("id")
    if user_id is None:
        raise HTTPException(
            status_code=403,
            detail="Follow-up tracking requires an authenticated user (OAuth login).",
        )
    return int(user_id)


def _validate_due_date(value: Optional[str]) -> Optional[str]:
    clean = (value or "").strip()
    if clean:
        try:
            date.fromisoformat(clean)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="due_date must be a valid YYYY-MM-DD date") from exc
    return clean or None


@app.post("/api/followups/suggest")
def api_suggest_followups(req: FollowupSuggestRequest, user: dict = Depends(_verify_auth)):
    """Find explicit follow-up language; never creates a task automatically."""
    return {"suggestions": suggest_followups(req.report_text)}


@app.get("/api/followups")
def api_list_followups(
    status: Optional[str] = "open",
    user: dict = Depends(_verify_auth),
):
    user_id = _followup_user_id(user)
    try:
        return {"followups": list_followups(user_id, status=status or None)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/followups")
def api_create_followup(req: FollowupCreateRequest, user: dict = Depends(_verify_auth)):
    user_id = _followup_user_id(user)
    if req.report_id is not None and not get_report(req.report_id, user_id=user_id):
        raise HTTPException(status_code=404, detail="Report not found")
    try:
        saved = create_followup(
            user_id=user_id,
            recommendation=req.recommendation,
            report_id=req.report_id,
            patient_id=req.patient_id or None,
            patient_name=req.patient_name or None,
            accession=req.accession or None,
            finding=req.finding or None,
            due_date=_validate_due_date(req.due_date),
            notes=req.notes or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log_event(
        user_id=user_id,
        event_type="followup_create",
        accession=req.accession or None,
        report_id=req.report_id,
        metadata={"followup_id": saved["id"], "due_date": saved["due_date"]},
    )
    return saved


@app.patch("/api/followups/{followup_id}")
def api_update_followup(
    followup_id: int,
    req: FollowupUpdateRequest,
    user: dict = Depends(_verify_auth),
):
    user_id = _followup_user_id(user)
    try:
        saved = update_followup(
            followup_id,
            user_id,
            status=req.status,
            due_date=_validate_due_date(req.due_date) if req.due_date is not None else None,
            notes=req.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not saved:
        raise HTTPException(status_code=404, detail="Follow-up not found")
    log_event(
        user_id=user_id,
        event_type="followup_update",
        accession=saved.get("accession"),
        report_id=saved.get("report_id"),
        metadata={"followup_id": followup_id, "status": saved["status"]},
    )
    return saved


@app.get("/api/audit-log")
def api_audit_log(
    accession: Optional[str] = None,
    report_id: Optional[int] = None,
    limit: int = 200,
    user: dict = Depends(_verify_auth),
):
    """Return audit-log entries. Filter by accession or report_id when given."""
    user_id = user.get("id")
    # In Basic Auth mode (id None) we still allow filtering by accession,
    # but block unrestricted "all events" reads.
    if accession is None and report_id is None and user_id is None:
        raise HTTPException(
            status_code=403,
            detail="Specify accession or report_id when not signed in via OAuth.",
        )
    return {
        "events": list_audit_events(
            # Scope to the caller's own events so an accession/report_id can't
            # be used to read another radiologist's audit trail (which embeds
            # patient identifiers and report hashes).
            user_id=user_id,
            accession=accession,
            report_id=report_id,
            limit=min(int(limit), 1000),
        )
    }


@app.get("/api/audit-log/verify")
def api_audit_verify(user: dict = Depends(_verify_auth)):
    return verify_chain()


@app.get("/api/retention")
def api_retention_status(user: dict = Depends(_verify_auth)):
    """Report the retention policy in force (admin only)."""
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="Admin only")
    return {
        "enabled": retention.enabled(),
        "retention_days": practice.settings.retention_days,
        "outbox_retention_days": practice.settings.outbox_retention_days,
        "interval_seconds": practice.settings.retention_interval_seconds,
    }


@app.post("/api/retention/run")
def api_retention_run(user: dict = Depends(_verify_auth)):
    """Run the retention purge now (admin only). Audited as retention_purge."""
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="Admin only")
    if not retention.enabled():
        raise HTTPException(status_code=400, detail="Retention is not enabled on this deployment.")
    return retention.run_once(user_id=user.get("id"))


# ---------------------------------------------------------------------------
# Phase 2: NLP QA layer
# ---------------------------------------------------------------------------

class QACheckRequest(BaseModel):
    report_text: str
    source_text: Optional[str] = None
    source_kind: Literal["dictation", "worksheet"] = "dictation"
    accession: Optional[str] = None
    patient_gender: Optional[str] = None  # "M" / "F" / "male" / "female"
    body_part: Optional[str] = None
    ordered_side: Optional[str] = None    # "left" / "right" / "bilateral"


@app.post("/api/qa-check")
def api_qa_check(req: QACheckRequest, user: dict = Depends(_verify_auth)):
    """Run deterministic NLP QA checks against a report. Flag-only — no rewrite."""
    if not req.report_text or not req.report_text.strip():
        return {"flags": []}
    flags = run_qa_checks(
        report_text=req.report_text,
        patient_gender=req.patient_gender,
        ordered_side=req.ordered_side,
        body_part=req.body_part,
        source_text=req.source_text,
        source_kind=req.source_kind,
    )
    log_event(
        user_id=user.get("id"),
        event_type="qa_check",
        accession=req.accession,
        metadata={
            "flag_count": len(flags),
            "flag_types": sorted({f.get("type", "?") for f in flags}),
            "chars": len(req.report_text),
        },
    )
    return {"flags": flags}


# ---------------------------------------------------------------------------
# FHIR RIS patient lookup
# ---------------------------------------------------------------------------

@app.get("/patient/{accession}")
async def lookup_patient(accession: str, user: dict = Depends(_verify_auth)):
    """Look up patient data from a configured FHIR R4 server by accession number.

    Requires FHIR_BASE_URL env var (e.g. https://fhir.hospital.org/r4).
    Optional FHIR_BEARER_TOKEN env var for Bearer auth.

    Returns a JSON object with any of:
      patient_name, patient_dob, patient_id, accession,
      modality, body_part, referring_physician
    """
    import httpx

    fhir_base = os.environ.get("FHIR_BASE_URL", "").rstrip("/")
    if not fhir_base:
        raise HTTPException(
            status_code=503,
            detail="FHIR_BASE_URL is not configured on this server.",
        )

    headers: dict = {"Accept": "application/fhir+json"}
    token = os.environ.get("FHIR_BEARER_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Search ImagingStudy by identifier, requesting inline Patient
            resp = await client.get(
                f"{fhir_base}/ImagingStudy",
                params={"identifier": accession, "_include": "ImagingStudy:patient"},
                headers=headers,
            )
            resp.raise_for_status()
            bundle = resp.json()

        result: dict = {"accession": accession}
        patient_ref: Optional[str] = None

        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            rt = resource.get("resourceType", "")

            if rt == "ImagingStudy":
                series = resource.get("series", [])
                if series:
                    s = series[0]
                    mod = s.get("modality", {})
                    if mod:
                        result["modality"] = mod.get("code", "")
                    site = s.get("bodySite", {})
                    if site:
                        result["body_part"] = site.get("display", "")
                subject = resource.get("subject", {})
                patient_ref = subject.get("reference") or patient_ref

            elif rt == "Patient":
                _extract_patient_fields(resource, result)

        # If patient wasn't bundled via _include, fetch separately
        if patient_ref and "patient_name" not in result:
            url = patient_ref if patient_ref.startswith("http") else f"{fhir_base}/{patient_ref}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                p_resp = await client.get(url, headers=headers)
            if p_resp.status_code == 200:
                _extract_patient_fields(p_resp.json(), result)

        if len(result) <= 1:  # only "accession" key — nothing found
            raise HTTPException(
                status_code=404,
                detail=f"No imaging study found for accession: {accession}",
            )

        return result

    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Accession not found: {accession}")
        raise HTTPException(status_code=503, detail=f"FHIR server error: {exc.response.status_code}")
    except Exception as exc:
        logger.error("FHIR lookup failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"FHIR lookup failed: {exc}")


# ---------------------------------------------------------------------------
# Settings API
# ---------------------------------------------------------------------------

class SettingsRequest(BaseModel):
    streaming_stt_provider: Optional[str] = None
    transcription_base_url: Optional[str] = None
    transcription_model: Optional[str] = None
    text_base_url: Optional[str] = None
    text_model: Optional[str] = None
    fhir_export_enabled: bool = False
    retain_deidentified_samples: Optional[bool] = None
    # Reporting style preferences
    style_spelling: Optional[str] = None
    style_numerals: Optional[str] = None
    style_measurement_unit: Optional[str] = None
    style_measurement_separator: Optional[str] = None
    style_decimal_precision: Optional[int] = None
    style_laterality: Optional[str] = None
    style_impression_style: Optional[str] = None
    style_negation_phrasing: Optional[str] = None
    style_date_format: Optional[str] = None
    style_paste_format: Optional[str] = None
    # HL7 v2.4 ORU^R01 export (global, server-admin controlled)
    hl7_export_enabled: Optional[bool] = None
    hl7_outbox_path: Optional[str] = None
    hl7_sending_facility: Optional[str] = None
    hl7_receiving_facility: Optional[str] = None


_STYLE_ALLOWED = {
    "style_spelling": {"american", "british"},
    "style_numerals": {"roman", "arabic"},
    "style_measurement_unit": {"mm", "cm", "auto"},
    "style_measurement_separator": {"x", "times", "by"},
    "style_decimal_precision": {0, 1, 2},
    "style_laterality": {"full", "abbrev"},
    "style_impression_style": {"bulleted", "numbered", "prose"},
    "style_negation_phrasing": {"no_evidence_of", "no_x_identified", "x_absent"},
    "style_date_format": {"dd_mm_yyyy", "mm_dd_yyyy", "yyyy_mm_dd"},
    "style_paste_format": {"rich", "plain", "markdown"},
}

_STREAMING_STT_PROVIDERS = {"auto", "groq", "deepgram", "assemblyai"}


@app.get("/api/capabilities")
def api_capabilities():
    """Return streaming STT availability — no auth required."""
    provider = get_streaming_provider()
    return {
        "streaming_stt": provider is not None,
        "provider": resolve_streaming_provider_name(),
        "profile": practice.settings.profile,
        "data_residency": practice.settings.data_residency or None,
        "research_features": practice.settings.research_features,
        "ai_disclosure_footer": practice.settings.ai_disclosure_footer,
    }


@app.get("/api/hl7/worklist")
def api_hl7_worklist(user: dict = Depends(_verify_auth)):
    """Return pending radiology orders parsed from the HL7 inbox directory.

    The inbox is configured via VOXRAD_HL7_INBOX or the [HL7] InboxPath
    setting, and defaults to {save_directory}/hl7_inbox. Each order includes
    an order_id so the client can dismiss it after dictating.
    """
    inbox = _hl7_inbox_dir()
    if not inbox:
        return {"enabled": False, "orders": []}
    try:
        orders = list_inbox(inbox)
    except Exception as e:
        logger.warning("HL7 worklist scan failed: %s", e)
        return {"enabled": True, "orders": [], "error": str(e)}
    return {"enabled": True, "orders": orders, "inbox": inbox}


@app.post("/api/hl7/worklist/{order_id}/archive")
def api_hl7_worklist_archive(order_id: str, user: dict = Depends(_verify_auth)):
    """Move a processed order out of the inbox into the 'processed' subfolder."""
    inbox = _hl7_inbox_dir()
    if not inbox:
        raise HTTPException(status_code=404, detail="HL7 inbox not configured")
    # Reject path traversal — order_id must be a simple filename stem.
    if "/" in order_id or "\\" in order_id or order_id.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid order_id")
    ok = archive_order(inbox, order_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Order not found in inbox")
    return {"ok": True}


# ───────────────────────────────────────────────────────────────────────────────────────
# MWL bridge agent — accepts pushed DICOM Modality Worklist orders
# ───────────────────────────────────────────────────────────────────────────────────────
# The clinic's on-prem agent runs C-FIND SCU against their PACS/broker and
# POSTs each parsed order here. We write one JSON file per order into the
# HL7 inbox so the existing worklist UI, filter, and archive flow pick it up
# with no further changes. Shared-secret auth only — not a user session.

_MWL_FIELD_WHITELIST = {
    "patient_name", "patient_dob", "patient_id",
    "accession", "modality", "body_part", "procedure",
    "referring_physician", "scheduled_datetime",
}


@app.post("/api/worklist/push")
def api_worklist_push(
    payload: dict = Body(...),
    x_voxrad_agent_token: str | None = Header(None),
):
    """Accept a batch of MWL orders from an on-prem bridge agent.

    Authentication is via the X-VoxRad-Agent-Token header matching the
    VOXRAD_MWL_AGENT_TOKEN env var. When the token is unset, the endpoint
    is disabled. The agent POSTs `{"orders": [{...}, ...]}` — each order
    uses the same field names as the HL7 worklist output.
    """
    expected = (config.mwl_agent_token or "").strip()
    if not expected:
        raise HTTPException(status_code=404, detail="MWL push not configured")
    if not x_voxrad_agent_token or not secrets.compare_digest(
        x_voxrad_agent_token.strip(), expected
    ):
        raise HTTPException(status_code=401, detail="Invalid agent token")

    inbox = _hl7_inbox_dir()
    if not inbox:
        raise HTTPException(status_code=500, detail="HL7 inbox not configured on server")
    try:
        os.makedirs(inbox, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create MWL inbox %s: %s", inbox, exc)
        raise HTTPException(status_code=500, detail="Could not initialise worklist inbox") from exc

    orders = payload.get("orders") if isinstance(payload, dict) else None
    if not isinstance(orders, list):
        raise HTTPException(status_code=400, detail="Expected {'orders': [...]}")

    written = 0
    skipped = 0
    for order in orders:
        if not isinstance(order, dict):
            skipped += 1
            continue
        clean = {k: v for k, v in order.items() if k in _MWL_FIELD_WHITELIST and v}
        if not clean.get("accession") and not clean.get("patient_id"):
            # Need at least one stable identifier to build the filename
            skipped += 1
            continue
        clean["source"] = "mwl"
        # Deterministic order_id so re-pushed orders overwrite rather than
        # duplicating — keyed on accession (preferred) or patient+scheduled.
        key = clean.get("accession") or f"{clean.get('patient_id','')}_{clean.get('scheduled_datetime','')}"
        # Sanitise for filesystem
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:80] or "order"
        fname = f"mwl_{safe}.json"
        fpath = os.path.join(inbox, fname)
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(clean, f, ensure_ascii=False, indent=2)
            written += 1
        except OSError as e:
            logger.warning("Could not write MWL order %s: %s", fpath, e)
            skipped += 1

    return {"ok": True, "written": written, "skipped": skipped, "inbox": inbox}


@app.get("/api/settings")
def api_get_settings(user: dict = Depends(_verify_auth)):
    """Return current (non-sensitive) configuration state.

    In OAuth mode, style settings are per-user; in Basic Auth mode they come
    from the global config / settings.ini.
    """
    style = _user_style(user)
    if style is None:
        # Basic Auth mode — read from global config
        style = {
            "spelling":              config.style_spelling,
            "numerals":              config.style_numerals,
            "measurement_unit":      config.style_measurement_unit,
            "measurement_separator": config.style_measurement_separator,
            "decimal_precision":     config.style_decimal_precision,
            "laterality":            config.style_laterality,
            "impression_style":      config.style_impression_style,
            "negation_phrasing":     config.style_negation_phrasing,
            "date_format":           config.style_date_format,
            "paste_format":          config.style_paste_format,
            "fhir_export_enabled":   config.fhir_export_enabled,
        }
    return {
        "streaming_stt_provider": config.STREAMING_STT_PROVIDER or "auto",
        "effective_streaming_stt_provider": resolve_streaming_provider_name(),
        "can_manage_global_settings": _is_admin(user),
        "transcription_base_url": config.TRANSCRIPTION_BASE_URL or "",
        "transcription_model":    config.SELECTED_TRANSCRIPTION_MODEL or "",
        "text_base_url":          config.BASE_URL or "",
        "text_model":             config.SELECTED_MODEL or "",
        "text_reasoning_effort":  reasoning_effort_for_model(config.SELECTED_MODEL) or "model default",
        "fhir_export_enabled":    style.get("fhir_export_enabled", config.fhir_export_enabled),
        "retain_deidentified_samples": bool(style.get("retain_deidentified_samples", False)),
        "quality_retention_available": bool(oauth_enabled() and user.get("id") is not None),
        "style":                  {k: v for k, v in style.items() if k not in {"fhir_export_enabled", "retain_deidentified_samples"}},
        "oauth_mode":             oauth_enabled(),
        "hl7": {
            "export_enabled":     config.hl7_export_enabled,
            "outbox_path":        config.hl7_outbox_path or "",
            "sending_facility":   config.hl7_sending_facility or "VOXRAD",
            "receiving_facility": config.hl7_receiving_facility or "",
        },
        "keys": {
            "transcription": bool(config.TRANSCRIPTION_API_KEY),
            "text":          bool(config.TEXT_API_KEY),
            "deepgram":      bool(config.DEEPGRAM_API_KEY),
            "assemblyai":    bool(config.ASSEMBLYAI_API_KEY),
        },
    }


_STYLE_FIELD_MAP = {
    "style_spelling":             "spelling",
    "style_numerals":             "numerals",
    "style_measurement_unit":     "measurement_unit",
    "style_measurement_separator":"measurement_separator",
    "style_decimal_precision":    "decimal_precision",
    "style_laterality":           "laterality",
    "style_impression_style":     "impression_style",
    "style_negation_phrasing":    "negation_phrasing",
    "style_date_format":          "date_format",
    "style_paste_format":         "paste_format",
}


@app.post("/api/settings")
def api_save_settings(req: SettingsRequest, user: dict = Depends(_verify_auth)):
    """Persist non-sensitive settings.

    Global settings (model URLs, streaming provider) → settings.ini.
    Style preferences → per-user SQLite row (OAuth mode) or settings.ini (Basic Auth mode).
    """
    current_provider = (config.STREAMING_STT_PROVIDER or "auto").strip().lower()
    requested_provider = (
        req.streaming_stt_provider.strip().lower()
        if req.streaming_stt_provider is not None
        else current_provider
    )
    if requested_provider not in _STREAMING_STT_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid streaming transcription provider: {requested_provider!r}",
        )

    # Server-wide settings (model IDs, base URLs, streaming provider) mutate
    # process-global config and affect every user, so only admins may change
    # them. A non-admin change now fails visibly instead of reporting a false
    # success and reverting on the next page load.
    _admin = _is_admin(user)
    global_change_requested = any((
        requested_provider != current_provider,
        bool(req.transcription_base_url and req.transcription_base_url != config.TRANSCRIPTION_BASE_URL),
        bool(req.transcription_model and req.transcription_model != config.SELECTED_TRANSCRIPTION_MODEL),
        bool(req.text_base_url and req.text_base_url != config.BASE_URL),
        bool(req.text_model and req.text_model != config.SELECTED_MODEL),
    ))
    if global_change_requested and not _admin:
        raise HTTPException(
            status_code=403,
            detail="Only a RadSpeed administrator can change shared AI model settings.",
        )

    if _admin:
        config.STREAMING_STT_PROVIDER = requested_provider
        if req.transcription_base_url:
            config.TRANSCRIPTION_BASE_URL = req.transcription_base_url
        if req.transcription_model:
            config.SELECTED_TRANSCRIPTION_MODEL = req.transcription_model
        if req.text_base_url:
            config.BASE_URL = req.text_base_url
        if req.text_model:
            config.SELECTED_MODEL = req.text_model

    # Validate style fields
    style_update: dict = {}
    for req_field, style_key in _STYLE_FIELD_MAP.items():
        val = getattr(req, req_field)
        if val is None:
            continue
        allowed = _STYLE_ALLOWED[req_field]
        if val not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {req_field}: {val!r} (allowed: {sorted(allowed)})",
            )
        style_update[style_key] = val

    # HL7 settings are always global (server-admin controlled) — admin only.
    hl7_touched = False
    if _admin:
        if req.hl7_export_enabled is not None:
            config.hl7_export_enabled = bool(req.hl7_export_enabled)
            hl7_touched = True
        if req.hl7_outbox_path is not None:
            config.hl7_outbox_path = req.hl7_outbox_path.strip()
            hl7_touched = True
        if req.hl7_sending_facility is not None:
            config.hl7_sending_facility = req.hl7_sending_facility.strip() or "VOXRAD"
            hl7_touched = True
        if req.hl7_receiving_facility is not None:
            config.hl7_receiving_facility = req.hl7_receiving_facility.strip()
            hl7_touched = True

    if oauth_enabled() and user.get("id") is not None:
        # Per-user: merge with existing preferences and save to SQLite
        existing = get_user_style(user["id"])
        was_retaining = bool(existing.get("retain_deidentified_samples", False))
        existing.update(style_update)
        existing["fhir_export_enabled"] = req.fhir_export_enabled
        # Keep the old consent value for this general settings write. The
        # dedicated transaction below changes consent and deletes samples
        # atomically when the user withdraws it.
        existing["retain_deidentified_samples"] = was_retaining
        save_user_style(user["id"], existing)
        if req.retain_deidentified_samples is not None:
            set_quality_retention(user["id"], bool(req.retain_deidentified_samples))
        if _admin and (global_change_requested or hl7_touched):
            # Persist global changes in OAuth mode as well as Basic Auth mode.
            save_web_settings()
    else:
        # Basic Auth / global mode: write to config + settings.ini
        config.fhir_export_enabled = req.fhir_export_enabled
        for style_key, val in style_update.items():
            setattr(config, f"style_{style_key}", val)
        save_web_settings()

    logger.info("Settings saved by %s", _get_username(user))
    return {"ok": True}


# ---------------------------------------------------------------------------
# WebSocket: real-time streaming STT proxy
# ---------------------------------------------------------------------------

@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket, token: str = ""):
    """Proxy PCM audio from browser to Deepgram/AssemblyAI; stream transcripts back.

    Protocol:
      1. Client connects with ?token=<base64(user:pass)>
      2. Client sends JSON: {"template_name": "..."}
      3. Client sends binary frames: raw PCM 16kHz mono int16
      4. Server sends JSON: {"type":"interim","text":"..."} or {"type":"final","text":"..."}
      5. Client sends JSON: {"type":"stop"} when done recording
      6. Server sends JSON: {"type":"session_complete","transcription":"...","session_id":"..."}
    """
    ws_user = _verify_ws_token(token)
    if not ws_user:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    provider = None
    try:
        # Step 1: receive config message
        config_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        template_name = config_msg.get("template_name")
        keywords = _build_keyword_list(template_name, ws_user)

        provider = get_streaming_provider()
        if not provider:
            await websocket.send_json({
                "type": "error",
                "message": "No streaming STT provider configured. Configure one in Settings.",
            })
            return

        provider_name = resolve_streaming_provider_name()
        if provider_name == "deepgram":
            api_key = config.DEEPGRAM_API_KEY or ""
        elif provider_name == "assemblyai":
            api_key = config.ASSEMBLYAI_API_KEY or ""
        else:
            api_key = ""

        await provider.connect(api_key, 16000, keywords)

        finals: list[str] = []
        stop_event = asyncio.Event()
        # Tracks the most recent interim, so a stop pressed before the
        # provider's endpointing window fires (e.g. a short voice-edit like
        # "oedema") can fall back to committing the last interim as final.
        last_interim_state = {"text": "", "committed": True}

        async def receive_loop():
            """Read from browser: binary frames are audio, text frames are control."""
            try:
                while not stop_event.is_set():
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        stop_event.set()
                        break
                    raw_bytes = msg.get("bytes")
                    if raw_bytes:
                        await provider.send_audio(raw_bytes)
                    raw_text = msg.get("text")
                    if raw_text:
                        try:
                            ctrl = json.loads(raw_text)
                            if ctrl.get("type") == "stop":
                                stop_event.set()
                        except (json.JSONDecodeError, KeyError):
                            pass
            except Exception:
                stop_event.set()

        async def results_loop():
            """Forward transcript events from provider to browser.

            Keeps running after stop_event so finals emitted in response to
            provider.finalize() can still be captured during the drain phase.
            """
            try:
                async for event in provider.receive_results():
                    if not _is_hallucination(event.text):
                        # Strip em dash artifacts (STT inserts — during pauses)
                        clean = re.sub(r'\s*—\s*', ' ', event.text).strip()
                        clean = correct_mri_platform(clean, template_name or "")
                        if not clean:
                            continue
                        if event.is_final:
                            finals.append(clean)
                            last_interim_state["text"] = ""
                            last_interim_state["committed"] = True
                            try:
                                await websocket.send_json({"type": "final", "text": clean})
                            except Exception:
                                pass
                        else:
                            last_interim_state["text"] = clean
                            last_interim_state["committed"] = False
                            try:
                                await websocket.send_json({"type": "interim", "text": clean})
                            except Exception:
                                pass
            except Exception as e:
                logger.warning("Results loop error: %s", e)

        receive_task = asyncio.create_task(receive_loop())
        results_task = asyncio.create_task(results_loop())

        await stop_event.wait()

        # Stop accepting more audio from the client first.
        receive_task.cancel()
        try:
            await receive_task
        except (asyncio.CancelledError, Exception):
            pass

        # Ask the provider to flush any pending speech as a final event,
        # then give it a brief window to deliver that final before teardown.
        try:
            await provider.finalize()
        except Exception:
            pass
        # Drain finals emitted in response to finalize() before teardown.
        await asyncio.sleep(1.5)

        results_task.cancel()
        try:
            await results_task
        except (asyncio.CancelledError, Exception):
            pass

        # Fallback for the short-utterance edge case: if the provider never
        # finalised the last interim (e.g. user stopped inside the endpointing
        # window and finalize didn't take effect), commit the interim text so
        # the voice edit isn't silently dropped.
        recovered_text = ""
        if last_interim_state["text"] and not last_interim_state["committed"]:
            logger.info(
                "[stt] no final received after stop — committing last interim (%d chars)",
                len(last_interim_state["text"]),
            )
            recovered_text = last_interim_state["text"]
            finals.append(recovered_text)

        try:
            await asyncio.wait_for(provider.close(), timeout=4.0)
        except asyncio.TimeoutError:
            logger.warning("provider.close() timed out — forcing teardown")
        provider = None

        full_text = correct_mri_platform(" ".join(finals).strip(), template_name or "")
        # Run synchronous LLM call in a thread so it doesn't block the event loop.
        # Cap at 8 s — if the LLM is slow/unavailable, send the raw streaming
        # transcript rather than hanging.
        if full_text:
            try:
                corrected = await asyncio.wait_for(
                    asyncio.to_thread(_correct_asr_text, full_text),
                    timeout=8.0,
                )
            except asyncio.TimeoutError:
                logger.warning("_correct_asr_text timed out — using raw text")
                corrected = full_text
        else:
            corrected = ""
        corrected = correct_mri_platform(corrected, template_name or "")

        _prune_sessions()
        session_id = _create_session(corrected) if corrected else ""

        await websocket.send_json({
            "type": "session_complete",
            "transcription": corrected,
            "raw_transcription": full_text,
            "recovered_text": recovered_text,
            "session_id": session_id,
        })

    except WebSocketDisconnect:
        logger.info("WS client disconnected during streaming")
    except Exception as e:
        logger.error("WS transcribe error: %s", e, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if provider:
            try:
                await provider.close()
            except Exception:
                pass


def _extract_patient_fields(resource: dict, result: dict) -> None:
    """Pull name, DOB, and MRN from a FHIR Patient resource into result dict."""
    names = resource.get("name", [])
    if names:
        n = names[0]
        given = " ".join(n.get("given", []))
        family = n.get("family", "")
        full = f"{given} {family}".strip()
        if full:
            result["patient_name"] = full
    dob = resource.get("birthDate", "")
    if dob:
        result["patient_dob"] = dob
    for ident in resource.get("identifier", []):
        code = (ident.get("type", {}).get("coding") or [{}])[0].get("code", "")
        if code == "MR":
            result["patient_id"] = ident.get("value", "")
            break
