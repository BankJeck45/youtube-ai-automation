"""Tests for pipeline/assemble.py — audio duration parsing."""

from unittest.mock import patch, MagicMock
from pathlib import Path

from verticals.assemble import _choose_final_encoding, get_audio_duration


class TestGetAudioDuration:
    @patch("verticals.assemble.run_cmd")
    def test_parses_duration(self, mock_cmd):
        mock_result = MagicMock()
        mock_result.stdout = "65.432000\n"
        mock_cmd.return_value = mock_result

        duration = get_audio_duration(Path("/tmp/test.mp3"))
        assert abs(duration - 65.432) < 0.001

    @patch("verticals.assemble.run_cmd")
    def test_parses_short_duration(self, mock_cmd):
        mock_result = MagicMock()
        mock_result.stdout = "3.5\n"
        mock_cmd.return_value = mock_result

        duration = get_audio_duration(Path("/tmp/test.mp3"))
        assert abs(duration - 3.5) < 0.001

    @patch("verticals.assemble.run_cmd")
    def test_calls_ffprobe(self, mock_cmd):
        mock_result = MagicMock()
        mock_result.stdout = "10.0\n"
        mock_cmd.return_value = mock_result

        get_audio_duration(Path("/tmp/audio.mp3"))
        args = mock_cmd.call_args[0][0]
        assert "ffprobe" in args
        assert "/tmp/audio.mp3" in args


class TestChooseFinalEncoding:
    @patch("verticals.assemble._ffmpeg_encoders")
    def test_prefers_libx264_mp4(self, mock_encoders):
        mock_encoders.return_value = {"libx264", "aac"}

        encoding = _choose_final_encoding()

        assert encoding.extension == "mp4"
        assert "libx264" in encoding.video_args
        assert encoding.browser_safe is True

    @patch("verticals.assemble._ffmpeg_encoders")
    def test_uses_webm_when_h264_unavailable(self, mock_encoders):
        mock_encoders.return_value = {"libvpx-vp9", "libopus"}

        encoding = _choose_final_encoding()

        assert encoding.extension == "webm"
        assert "libvpx-vp9" in encoding.video_args
        assert "libopus" in encoding.audio_args
        assert encoding.browser_safe is True

    @patch("verticals.assemble._ffmpeg_encoders")
    def test_marks_mpeg4_fallback_as_not_browser_safe(self, mock_encoders):
        mock_encoders.return_value = set()

        encoding = _choose_final_encoding()

        assert encoding.extension == "mp4"
        assert "mpeg4" in encoding.video_args
        assert encoding.browser_safe is False
