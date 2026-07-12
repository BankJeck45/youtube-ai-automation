from pathlib import Path

import pytest

from verticals import tts


def _disable_paid_tts(monkeypatch):
    monkeypatch.setattr(tts, "get_minimax_key", lambda: "")
    monkeypatch.setattr(tts, "get_elevenlabs_key", lambda: "")
    monkeypatch.setattr(tts, "get_60db_key", lambda: "")


def test_say_request_falls_back_to_edge_when_say_is_unavailable(monkeypatch):
    _disable_paid_tts(monkeypatch)
    monkeypatch.setattr(tts, "_say_available", lambda: False)
    monkeypatch.setattr(tts, "_edge_tts_available", lambda: True)

    assert tts.get_tts_provider("say") == "edge"


def test_generate_voiceover_does_not_call_missing_say(monkeypatch, tmp_path):
    _disable_paid_tts(monkeypatch)
    monkeypatch.setattr(tts, "_say_available", lambda: False)
    monkeypatch.setattr(tts, "_edge_tts_available", lambda: True)
    monkeypatch.setattr(tts, "_generate_say", lambda *_: pytest.fail("say should not be called"))

    out = tmp_path / "voiceover_en.mp3"

    def fake_edge(script, out_dir, lang, voice_override="", settings=None):
        out.write_bytes(b"mp3")
        return out

    monkeypatch.setattr(tts, "_generate_edge_tts", fake_edge)

    assert tts.generate_voiceover("hello", tmp_path, provider="say") == out


def test_generate_voiceover_reports_clear_error_without_say(monkeypatch, tmp_path):
    _disable_paid_tts(monkeypatch)
    monkeypatch.setattr(tts, "_say_available", lambda: False)
    monkeypatch.setattr(tts, "_edge_tts_available", lambda: True)
    monkeypatch.setattr(
        tts,
        "_generate_edge_tts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("edge down")),
    )

    with pytest.raises(RuntimeError, match="No TTS provider could generate voiceover"):
        tts.generate_voiceover("hello", Path(tmp_path), provider="edge")
