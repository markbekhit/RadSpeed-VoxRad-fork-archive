"""Integration-style tests for the desktop transcription pipeline.

The external speech and text APIs are mocked, but encryption, temporary-file
handling, prompt extraction, report cleanup, and report encryption are real.
"""
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import httpx
from cryptography.fernet import Fernet
from openai import AuthenticationError


# ``utils.file_handling`` imports Tk directly.  The pipeline does not need a
# display, so provide small import stubs when this file is run on headless CI.
sys.modules.setdefault("tkinter", MagicMock())
sys.modules.setdefault("tkinter.ttk", MagicMock())
sys.modules.setdefault("tkinter.filedialog", MagicMock())
sys.modules.setdefault("tkinter.messagebox", MagicMock())

import audio.transcriber as transcriber  # noqa: E402
from config.config import config  # noqa: E402


def _authentication_error():
    request = httpx.Request("POST", "http://test/v1/audio/transcriptions")
    response = httpx.Response(401, request=request, json={"error": {"message": "bad key"}})
    return AuthenticationError("Invalid API key", response=response, body=None)


class TranscriptionPipelineTests(unittest.TestCase):
    _CONFIG_DEFAULTS = {
        "TRANSCRIPTION_API_KEY": "speech-key",
        "TRANSCRIPTION_BASE_URL": "https://speech.example/v1",
        "SELECTED_TRANSCRIPTION_MODEL": "whisper-test",
        "global_md_text_content": "",
        "secure_paste_shortcut": "ctrl+shift+v",
        "current_encrypted_report": None,
        "current_report_encryption_key": None,
    }

    def setUp(self):
        self._old_config = {
            name: getattr(config, name, None) for name in self._CONFIG_DEFAULTS
        }
        for name, value in self._CONFIG_DEFAULTS.items():
            setattr(config, name, value)

    def tearDown(self):
        for name, value in self._old_config.items():
            setattr(config, name, value)

    @staticmethod
    def _encrypted_audio(payload=b"synthetic mp3 bytes"):
        key = Fernet.generate_key()
        handle = tempfile.NamedTemporaryFile(suffix=".mp3.enc", delete=False)
        try:
            handle.write(Fernet(key).encrypt(payload))
            return handle.name, key
        finally:
            handle.close()

    def test_end_to_end_transcription_formats_encrypts_and_cleans_up(self):
        encrypted_path, audio_key = self._encrypted_audio()
        config.global_md_text_content = (
            "### Exam:\nMRI shoulder\n"
            "[correct spellings]\nsupraspinatus, glenoid\n[correct spellings]\n"
            "FINDINGS:\n"
        )
        statuses = []
        client = MagicMock()
        client.audio.transcriptions.create.return_value = types.SimpleNamespace(
            text="Imaged on a 3D platform. Supraspinatus tear. 3D reconstruction performed."
        )

        try:
            with patch.object(transcriber, "OpenAI", return_value=client) as openai, \
                 patch.object(transcriber, "format_text", return_value="**FINDINGS:** Tear") as format_text, \
                 patch.object(transcriber, "update_status", statuses.append), \
                 patch.object(transcriber.os, "remove", wraps=os.remove) as remove:
                result = transcriber.transcribe_audio(encrypted_path, audio_key)

            self.assertIsNone(result)
            format_text.assert_called_once_with(
                "Imaged on a 3T platform. Supraspinatus tear. 3D reconstruction performed."
            )
            openai.assert_called_once_with(
                api_key="speech-key", base_url="https://speech.example/v1"
            )
            request = client.audio.transcriptions.create.call_args.kwargs
            self.assertEqual(request["file"][1], b"synthetic mp3 bytes")
            self.assertEqual(request["model"], "whisper-test")
            self.assertEqual(request["prompt"], "supraspinatus, glenoid")
            self.assertEqual(request["language"], "en")
            self.assertEqual(request["temperature"], 0.0)

            report = Fernet(config.current_report_encryption_key.encode()).decrypt(
                config.current_encrypted_report.encode()
            )
            self.assertEqual(report.decode(), "FINDINGS: Tear")
            decrypted_temp = remove.call_args.args[0]
            self.assertFalse(os.path.exists(decrypted_temp))
            self.assertEqual(statuses[:2], ["Transcribing...📝", "Performing AI analysis.🤖"])
            self.assertIn("Report generated", statuses[-1])
            # Exercise the Windows/web batch route, not only the helper.
            import asyncio
            import importlib
            from io import BytesIO
            from fastapi import UploadFile
            web_app = importlib.import_module("web.app")
            raw = "Imaged on a 3D platform. 3D reconstruction performed."
            client.audio.transcriptions.create.return_value = types.SimpleNamespace(text=raw)
            with patch.object(web_app, "_MOCK_MODE", False), \
                 patch.object(web_app, "OpenAI", return_value=client):
                for template, expected in (
                    ("MRI_Knee.txt", raw.replace("3D platform", "3T platform")),
                    ("CT_Sinuses.txt", raw),
                ):
                    response = asyncio.run(web_app.transcribe(
                        audio=UploadFile(file=BytesIO(b"synthetic audio"), filename="test.webm"),
                        template_name=template, whisper_prompt=None, selected_text=None,
                        user={"id": None, "name": "synthetic", "email": ""},
                    ))
                    self.assertEqual(response["transcription"], expected)
                    self.assertEqual(web_app._sessions.pop(response["session_id"])["transcription"], expected)
            self.assertIn("3T platform", web_app._build_keyword_list("MRI_Knee.txt"))
        finally:
            os.remove(encrypted_path)

    def test_transcription_auth_failure_preserves_previous_report_and_cleans_up(self):
        encrypted_path, audio_key = self._encrypted_audio()
        config.current_encrypted_report = "previous-report"
        config.current_report_encryption_key = "previous-key"
        statuses = []
        client = MagicMock()
        client.audio.transcriptions.create.side_effect = _authentication_error()

        try:
            with patch.object(transcriber, "OpenAI", return_value=client), \
                 patch.object(transcriber, "format_text") as format_text, \
                 patch.object(transcriber, "update_status", statuses.append), \
                 patch.object(transcriber.os, "remove", wraps=os.remove) as remove:
                transcriber.transcribe_audio(encrypted_path, audio_key)

            format_text.assert_not_called()
            self.assertEqual(config.current_encrypted_report, "previous-report")
            self.assertEqual(config.current_report_encryption_key, "previous-key")
            self.assertIn("API key rejected", statuses[-1])
            self.assertFalse(os.path.exists(remove.call_args.args[0]))
        finally:
            os.remove(encrypted_path)

    def test_formatting_failure_does_not_replace_last_good_report(self):
        encrypted_path, audio_key = self._encrypted_audio()
        config.current_encrypted_report = "previous-report"
        config.current_report_encryption_key = "previous-key"
        statuses = []
        client = MagicMock()
        client.audio.transcriptions.create.return_value = types.SimpleNamespace(text="dictation")

        try:
            with patch.object(transcriber, "OpenAI", return_value=client), \
                 patch.object(transcriber, "format_text", return_value=None), \
                 patch.object(transcriber, "update_status", statuses.append):
                transcriber.transcribe_audio(encrypted_path, audio_key)

            self.assertEqual(config.current_encrypted_report, "previous-report")
            self.assertEqual(config.current_report_encryption_key, "previous-key")
            self.assertEqual(statuses[-1], "Performing AI analysis.🤖")
            self.assertFalse(any("NoneType" in s for s in statuses))
        finally:
            os.remove(encrypted_path)

    def test_missing_input_stops_before_decryption_or_api_call(self):
        statuses = []
        with patch.object(transcriber, "OpenAI") as openai, \
             patch.object(transcriber, "update_status", statuses.append):
            transcriber.transcribe_audio(None, None)
        openai.assert_not_called()
        self.assertEqual(statuses, ["Error: Could not process audio."])


if __name__ == "__main__":
    unittest.main()
