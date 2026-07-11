"""Tests for b-roll generation fallbacks."""

from io import BytesIO

from PIL import Image, ImageStat

from verticals.broll import (
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    _extract_image_bytes,
    _fallback_frame,
    _generate_image_pollinations,
)


def test_extracts_interactions_output_image():
    payload = {"output_image": {"data": "aGVsbG8="}}

    assert _extract_image_bytes(payload) == b"hello"


def test_extracts_interactions_step_image():
    payload = {
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "image", "data": "aGVsbG8="}],
            }
        ]
    }

    assert _extract_image_bytes(payload) == b"hello"


def test_fallback_frame_is_not_plain_solid_color(tmp_path):
    path = _fallback_frame(0, tmp_path, "cinematic UFO lights over a desert")

    img = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(img)

    assert img.size == (VIDEO_WIDTH, VIDEO_HEIGHT)
    assert max(stat.stddev) > 10


def test_pollinations_fallback_saves_portrait_image(monkeypatch, tmp_path):
    source = Image.new("RGB", (540, 960), (20, 90, 180))
    buf = BytesIO()
    source.save(buf, format="JPEG")

    class Response:
        status_code = 200
        content = buf.getvalue()
        text = ""
        headers = {"Content-Type": "image/jpeg"}

    def fake_get(*args, **kwargs):
        return Response()

    monkeypatch.setattr("verticals.broll.requests.get", fake_get)

    out = tmp_path / "frame.png"
    _generate_image_pollinations("cinematic authentication screen", out)

    img = Image.open(out).convert("RGB")
    assert img.size == (VIDEO_WIDTH, VIDEO_HEIGHT)
