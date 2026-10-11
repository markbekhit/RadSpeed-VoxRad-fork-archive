import logging
import os
import re
import tempfile
from datetime import datetime

from cryptography.fernet import Fernet, InvalidToken
from openai import OpenAI, AuthenticationError, APIError
import google.generativeai as genai

from config.config import config
from ui.utils import update_status
from utils.file_handling import strip_markdown
from llm.format import correct_mri_platform, format_text

logger = logging.getLogger(__name__)

_GEMINI_MAX_BYTES = 20 * 1024 * 1024  # 20 MB hard limit for Gemini upload

_GEMINI_KEY_REJECTED_MSG = (
    "Multimodal Model API key rejected (invalid or expired). "
    "Please update your key in Settings → Multimodal Model."
)


def is_gemini_auth_error(exc):
    """True if a Gemini exception indicates a rejected/invalid API key.

    google-generativeai surfaces key problems as several exception types
    (PermissionDenied, Unauthenticated, InvalidArgument with API_KEY_INVALID),
    so match on the message rather than the class.
    """
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("api key not valid", "api_key_invalid", "unauthenticated", "permission denied", "401", "403")
    )


def transcribe_audio(encrypted_mp3_path, decryption_key):
    """Transcribes the audio from an encrypted MP3 file using OpenAI's API."""
    if not encrypted_mp3_path or not decryption_key:
        logger.error("Encrypted MP3 path or decryption key missing.")
        update_status("Error: Could not process audio.")
        return

    cipher_suite = Fernet(decryption_key)
    decrypted_mp3_path = None
    try:
        with open(encrypted_mp3_path, "rb") as encrypted_file:
            encrypted_data = encrypted_file.read()

        try:
            decrypted_data = cipher_suite.decrypt(encrypted_data)
        except InvalidToken:
            logger.error("Audio decryption failed — stale or invalid encryption key.")
            update_status(
                "Could not decrypt audio. The recording key may be stale. Please re-record."
            )
            return

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_decrypted_file:
            tmp_decrypted_file.write(decrypted_data)
            decrypted_mp3_path = tmp_decrypted_file.name

        client = OpenAI(api_key=config.TRANSCRIPTION_API_KEY, base_url=config.TRANSCRIPTION_BASE_URL)

        if hasattr(config, 'global_md_text_content') and config.global_md_text_content:
            content = config.global_md_text_content
        else:
            content = " "

        spellings_match = re.search(
            r'\[correct spellings\](.*?)\[correct spellings\]',
            content,
            re.DOTALL,
        )
        prompt_spellings = spellings_match.group(1).strip() if spellings_match else " "

        with open(decrypted_mp3_path, "rb") as decrypted_file:
            update_status("Transcribing...📝")
            try:
                transcription_result = client.audio.transcriptions.create(
                    file=(decrypted_mp3_path, decrypted_file.read()),
                    model=config.SELECTED_TRANSCRIPTION_MODEL,
                    prompt=prompt_spellings,
                    language="en",
                    temperature=0.0,
                )
            except AuthenticationError:
                logger.error("Transcription API key is invalid or expired.")
                update_status(
                    "Transcription API key rejected. Please update your key in Settings → Transcription Model."
                )
                return
            except APIError as e:
                logger.error("Transcription API error: %s", e)
                update_status(f"Transcription API error: {e}")
                return

            transcription = correct_mri_platform(transcription_result.text, content)
            update_status("Performing AI analysis.🤖")
            formatted_text = format_text(transcription)
            if formatted_text is None:
                # format_text already surfaced the specific model/template
                # failure.  Do not replace it with a generic NoneType error,
                # and keep the last successfully generated report available.
                logger.error("Report formatting failed; preserving previous report.")
                return
            stripped_text = strip_markdown(formatted_text)

            # Encrypt the report and store it in config
            report_key = Fernet.generate_key()
            report_cipher = Fernet(report_key)
            encrypted_report = report_cipher.encrypt(stripped_text.encode()).decode()
            config.current_encrypted_report = encrypted_report
            config.current_report_encryption_key = report_key.decode()

            update_status(
                f"Report generated. Use {config.secure_paste_shortcut} to securely paste.✨"
            )

    except Exception as e:
        logger.error("Error decrypting or transcribing audio: %s", e)
        update_status(f"Error decrypting or transcribing audio: {e}")
    finally:
        if decrypted_mp3_path and os.path.exists(decrypted_mp3_path):
            os.remove(decrypted_mp3_path)


def mm_gemini(encrypted_mp3_path, decryption_key):
    """Generates text from an encrypted audio file using the multimodal Gemini model."""
    if not encrypted_mp3_path or not decryption_key:
        logger.error("Encrypted MP3 path or decryption key missing.")
        update_status("Error: Could not process audio.")
        return

    cipher_suite = Fernet(decryption_key)
    decrypted_mp3_path = None
    try:
        with open(encrypted_mp3_path, "rb") as encrypted_file:
            encrypted_data = encrypted_file.read()

        try:
            decrypted_data = cipher_suite.decrypt(encrypted_data)
        except InvalidToken:
            logger.error("Audio decryption failed — stale or invalid encryption key.")
            update_status(
                "Could not decrypt audio. The recording key may be stale. Please re-record."
            )
            return

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_decrypted_file:
            tmp_decrypted_file.write(decrypted_data)
            decrypted_mp3_path = tmp_decrypted_file.name

        # Guard against Gemini's 20 MB upload limit
        file_size = os.path.getsize(decrypted_mp3_path)
        if file_size > _GEMINI_MAX_BYTES:
            logger.error(
                "Audio file too large for Gemini upload: %.1f MB (limit 20 MB).", file_size / 1e6
            )
            update_status(
                f"Audio too large for Gemini ({file_size / 1e6:.1f} MB). "
                "Record a shorter dictation or switch to the standard transcription model."
            )
            return

        update_status("Performing AI analysis.🤖")
        genai.configure(api_key=config.MM_API_KEY)

        try:
            audio_file = genai.upload_file(path=decrypted_mp3_path)
        except Exception as e:
            logger.error("Gemini file upload failed: %s", e)
            if is_gemini_auth_error(e):
                update_status(_GEMINI_KEY_REJECTED_MSG)
            else:
                update_status(f"Gemini upload failed: {e}")
            return

        model = genai.GenerativeModel(model_name=config.multimodal_model)
        prompt = (
            "The provided audio is as dictated by a radiologist regarding a report of radiological study. "
            "Format is according to a standard radiological report.\n"
            f"This is the report template format as chosen by the user:\n{config.global_md_text_content}"
        )

        try:
            response = model.generate_content([prompt, audio_file])
        except Exception as e:
            logger.error("Gemini generate_content failed: %s", e)
            if is_gemini_auth_error(e):
                update_status(_GEMINI_KEY_REJECTED_MSG)
            else:
                update_status(f"Gemini error: {e}")
            return

        if response.text:
            stripped_text = strip_markdown(response.text)

            report_key = Fernet.generate_key()
            report_cipher = Fernet(report_key)
            encrypted_report = report_cipher.encrypt(stripped_text.encode()).decode()
            config.current_encrypted_report = encrypted_report
            config.current_report_encryption_key = report_key.decode()

            update_status(
                f"Report generated. Use {config.secure_paste_shortcut} to securely paste.✨"
            )
            return stripped_text
        else:
            update_status("No text returned by the multimodal model.")
            return None

    except Exception as e:
        logger.error("Failed to generate summary: %s", e)
        update_status(f"Failed to generate summary. Error: {e}")
        return None
    finally:
        if decrypted_mp3_path and os.path.exists(decrypted_mp3_path):
            os.remove(decrypted_mp3_path)


def save_report(report_text):
    """Saves the transcribed report to a file in the reports directory."""
    reports_dir = os.path.join(config.save_directory, "reports")
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"report_{timestamp}.txt"
    report_filepath = os.path.join(reports_dir, report_filename)

    with open(report_filepath, "w") as report_file:
        report_file.write(report_text)

    logger.info("Report saved to %s", report_filepath)
