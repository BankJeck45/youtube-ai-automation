"""Tests for thumbnail fallbacks."""

from PIL import Image, ImageStat

from verticals.thumbnail import THUMB_HEIGHT, THUMB_WIDTH, _generate_local_dark_thumb


def test_local_dark_thumbnail_is_not_blank(tmp_path):
    path = tmp_path / "thumb.png"

    _generate_local_dark_thumb("endless corridor voice", path)

    img = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(img)
    assert img.size == (THUMB_WIDTH, THUMB_HEIGHT)
    assert max(stat.stddev) > 5
