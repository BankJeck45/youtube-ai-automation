"""Tests for video assembly."""

from unittest.mock import MagicMock, patch
from pathlib import Path

from PIL import Image

from verticals.assemble import _build_music_audio_filter, _caption_overlay_pngs, _choose_final_encoding, _parse_srt, get_audio_duration


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
    @patch("verticals.assemble._encoding_works", return_value=True)
    @patch("verticals.assemble._ffmpeg_encoders")
    def test_prefers_libx264_mp4(self, mock_encoders, mock_encoding_works):
        mock_encoders.return_value = {"libx264", "aac"}

        encoding = _choose_final_encoding()

        assert encoding.extension == "mp4"
        assert "libx264" in encoding.video_args
        assert encoding.browser_safe is True
        mock_encoding_works.assert_called_once()

    @patch("verticals.assemble._encoding_works", return_value=True)
    @patch("verticals.assemble._ffmpeg_encoders")
    def test_uses_webm_when_h264_unavailable(self, mock_encoders, mock_encoding_works):
        mock_encoders.return_value = {"libvpx-vp9", "libopus"}

        encoding = _choose_final_encoding()

        assert encoding.extension == "webm"
        assert "libvpx-vp9" in encoding.video_args
        assert "libopus" in encoding.audio_args
        assert encoding.browser_safe is True
        mock_encoding_works.assert_called_once()

    @patch("verticals.assemble._encoding_works", return_value=True)
    @patch("verticals.assemble._ffmpeg_encoders")
    def test_prefers_fast_vp8_before_vp9(self, mock_encoders, mock_encoding_works):
        mock_encoders.return_value = {"libvpx", "libvorbis", "libvpx-vp9", "libopus"}

        encoding = _choose_final_encoding()

        assert encoding.extension == "webm"
        assert "libvpx" in encoding.video_args
        assert "libvpx-vp9" not in encoding.video_args
        assert "realtime" in encoding.video_args
        mock_encoding_works.assert_called_once()

    @patch("verticals.assemble._ffmpeg_encoders")
    def test_skips_broken_libopenh264_for_webm(self, mock_encoders):
        mock_encoders.return_value = {"libopenh264", "aac", "libvpx-vp9", "libopus"}

        def works(encoding):
            return encoding.name != "H.264/libopenh264"

        with patch("verticals.assemble._encoding_works", side_effect=works) as mock_encoding_works:
            encoding = _choose_final_encoding()

        assert encoding.extension == "webm"
        assert "libvpx-vp9" in encoding.video_args
        assert mock_encoding_works.call_count == 2

    @patch("verticals.assemble._ffmpeg_encoders")
    def test_marks_mpeg4_fallback_as_not_browser_safe(self, mock_encoders):
        mock_encoders.return_value = set()

        encoding = _choose_final_encoding()

        assert encoding.extension == "mp4"
        assert "mpeg4" in encoding.video_args
        assert encoding.browser_safe is False


def test_parse_srt_cues(tmp_path):
    srt = tmp_path / "captions.srt"
    srt.write_text(
        "1\n00:00:00,500 --> 00:00:02,000\nThe corridor answers\n\n"
        "2\n00:00:02,200 --> 00:00:03,400\nDo not turn around\n",
        encoding="utf-8",
    )

    cues = _parse_srt(srt)

    assert cues == [
        (0.5, 2.0, "The corridor answers"),
        (2.2, 3.4, "Do not turn around"),
    ]


def test_caption_overlay_pngs_are_transparent_with_visible_text(tmp_path):
    overlays = _caption_overlay_pngs(
        [(0.5, 2.0, "The corridor answers")],
        tmp_path,
        {"font_size": 48, "position": "lower_left", "text_color": "#D8D8D8"},
    )

    assert len(overlays) == 1
    assert overlays[0][0] == 0.5
    assert overlays[0][1] == 2.0
    img = Image.open(overlays[0][2]).convert("RGBA")
    assert img.size == (1080, 1920)
    assert img.getbbox() is not None


def test_music_audio_filter_avoids_aloop_scientific_size():
    filt = _build_music_audio_filter(74.832, "volume=if(between(t\\,0.00\\,75.13)\\,0.08\\,0.2):eval=frame")

    assert "aloop" not in filt
    assert "2e+09" not in filt
    assert "atrim=0:74.832" in filt
    assert "asetpts=PTS-STARTPTS" in filt
    assert "amix=inputs=2" in filt
