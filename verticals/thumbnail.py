"""Thumbnail generation — Gemini Imagen (16:9) + Pillow text overlay."""

import base64
import hashlib
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageFont

from .config import get_gemini_key
from .log import log
from .niche import get_thumbnail_config, load_niche
from .retry import with_retry

THUMB_WIDTH = 1280
THUMB_HEIGHT = 720


@with_retry(max_retries=3, base_delay=2.0)
def _generate_thumb_image(prompt: str, output_path: Path, api_key: str):
    """Generate a 16:9 thumbnail via Gemini native image generation."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta"
        "/models/gemini-2.0-flash-exp-image-generation:generateContent"
    )
    body = {
        "contents": [{"parts": [{"text": f"Generate a 16:9 landscape image: {prompt}"}]}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }
    r = requests.post(
        url, json=body, timeout=90,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    if r.status_code != 200:
        try:
            detail = r.json().get("error", {}).get("message", r.text[:200])
        except Exception:
            detail = r.text[:200]
        hint = ""
        if r.status_code == 403:
            hint = (
                " — check that GEMINI_API_KEY is set in this environment and is "
                "an AI Studio key (https://aistudio.google.com/apikey), not a "
                "Vertex AI / service-account credential"
            )
        raise RuntimeError(f"Gemini API {r.status_code}: {detail}{hint}")

    data = r.json()
    for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        if "inlineData" in part:
            img_b64 = part["inlineData"]["data"]
            output_path.write_bytes(base64.b64decode(img_b64))
            return
    raise RuntimeError("No image in Gemini response")


@with_retry(max_retries=1, base_delay=2.0)
def _generate_thumb_pollinations(prompt: str, output_path: Path):
    """Generate a 16:9 thumbnail base image through a no-key fallback."""
    seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
    image_prompt = (
        f"{prompt}. 16:9 YouTube thumbnail, dark cinematic scene, high contrast, "
        "no text, no logo, no watermark, dramatic lighting."
    )
    url = f"https://image.pollinations.ai/prompt/{quote(image_prompt, safe='')}"
    params = {
        "width": "1280",
        "height": "720",
        "nologo": "true",
        "private": "true",
        "seed": str(seed),
    }
    r = requests.get(url, params=params, timeout=150, headers={"Accept": "image/*"})
    content_type = r.headers.get("Content-Type", "")
    if r.status_code != 200 or not content_type.startswith("image/"):
        detail = r.text[:200] if getattr(r, "text", "") else content_type
        raise RuntimeError(f"Pollinations thumbnail {r.status_code}: {detail}")

    img = Image.open(BytesIO(r.content)).convert("RGB")
    img = img.resize((THUMB_WIDTH, THUMB_HEIGHT), Image.LANCZOS)
    img.save(output_path)


def _fit_thumb_background(image_path: Path) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    scale = max(THUMB_WIDTH / orig_w, THUMB_HEIGHT / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - THUMB_WIDTH) // 2
    top = (new_h - THUMB_HEIGHT) // 2
    return img.crop((left, top, left + THUMB_WIDTH, top + THUMB_HEIGHT))


def _short_thumbnail_text(title: str, max_words: int) -> str:
    words = [w.strip(".,:;!?\"'()[]").upper() for w in title.split()]
    words = [w for w in words if w]
    return " ".join(words[:max_words]) or "WATCH THIS"


def _load_title_font(size: int, style: str = ""):
    serif = "serif" in (style or "").lower()
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if serif else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if serif else "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Georgia Bold.ttf" if serif else "/Library/Fonts/Arial Bold.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _overlay_vignette(img: Image.Image) -> Image.Image:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rectangle((0, 0, THUMB_WIDTH, 210), fill=(0, 0, 0, 130))
    draw.rectangle((0, 0, 260, THUMB_HEIGHT), fill=(0, 0, 0, 95))
    draw.rectangle((THUMB_WIDTH - 260, 0, THUMB_WIDTH, THUMB_HEIGHT), fill=(0, 0, 0, 105))
    draw.rectangle((0, THUMB_HEIGHT - 170, THUMB_WIDTH, THUMB_HEIGHT), fill=(0, 0, 0, 100))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _generate_local_dark_thumb(prompt: str, output_path: Path):
    """Create a no-network dark archive thumbnail background."""
    seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
    img = Image.new("RGB", (THUMB_WIDTH, THUMB_HEIGHT), (4, 6, 7))
    draw = ImageDraw.Draw(img, "RGBA")

    # Corridor walls and floor perspective.
    vanish_x = THUMB_WIDTH // 2 + ((seed % 121) - 60)
    vanish_y = 250 + ((seed >> 8) % 90)
    draw.rectangle((0, 0, THUMB_WIDTH, THUMB_HEIGHT), fill=(3, 5, 6, 255))
    for i in range(12):
        t = i / 11
        shade = int(12 + t * 28)
        x_left = int((1 - t) * 0 + t * vanish_x)
        x_right = int((1 - t) * THUMB_WIDTH + t * vanish_x)
        y = int((1 - t) * THUMB_HEIGHT + t * vanish_y)
        draw.line((x_left, y, vanish_x, vanish_y), fill=(shade, shade + 4, shade + 8, 170), width=3)
        draw.line((x_right, y, vanish_x, vanish_y), fill=(shade, shade + 4, shade + 8, 170), width=3)

    # Dim ceiling lamps down the corridor.
    for i in range(6):
        t = i / 5
        x = int(vanish_x + (seed % 31 - 15) * (1 - t))
        y = int(vanish_y + 24 + t * 300)
        radius = int(24 - t * 13)
        alpha = int(120 - t * 55)
        draw.ellipse((x - radius, y - radius // 2, x + radius, y + radius // 2), fill=(230, 221, 180, alpha))
        draw.ellipse((x - radius * 4, y - radius * 2, x + radius * 4, y + radius * 2), fill=(180, 175, 145, 18))

    # Thin smoke/fog.
    for i in range(18):
        x = (seed * (i + 7) * 41) % THUMB_WIDTH
        y = 120 + ((seed * (i + 3) * 23) % 520)
        w = 220 + ((seed >> (i % 8)) % 360)
        h = 34 + ((seed >> (i % 6)) % 90)
        draw.ellipse((x - w, y - h, x + w, y + h), fill=(155, 168, 160, 16))

    # A small silhouette anchors the horror story without turning into a face.
    sx = vanish_x + ((seed >> 12) % 70 - 35)
    sy = 465
    draw.ellipse((sx - 18, sy - 82, sx + 18, sy - 46), fill=(2, 2, 3, 210))
    draw.rounded_rectangle((sx - 22, sy - 47, sx + 22, sy + 55), radius=13, fill=(2, 2, 3, 220))

    img = _overlay_vignette(img)
    img.save(output_path)


def _overlay_title(image_path: Path, title: str, output_path: Path, config: dict | None = None):
    """Overlay bold title text with the dark archive reference style."""
    config = config or {}
    img = _overlay_vignette(_fit_thumb_background(image_path))
    draw = ImageDraw.Draw(img)

    max_words = int(config.get("max_words", 6))
    title_text = _short_thumbnail_text(title, max_words)
    text_color = config.get("text_color", "#F3E8C8")
    rgb = tuple(int(text_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) if text_color.startswith("#") and len(text_color) == 7 else (243, 232, 200)

    # Word wrap the title
    font_size = int(config.get("font_size", 68))
    font = _load_title_font(font_size, config.get("font_style", "bold serif"))
    max_width = int(THUMB_WIDTH * 0.56)
    lines = _wrap_text(draw, title_text, font, max_width)
    while len(lines) > 4 and font_size > 42:
        font_size -= 6
        font = _load_title_font(font_size, config.get("font_style", "bold serif"))
        lines = _wrap_text(draw, title_text, font, max_width)
    text_block = "\n".join(lines)

    bbox = draw.multiline_textbbox((0, 0), text_block, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (THUMB_WIDTH - text_w) // 2
    y = int(config.get("text_top", 44))

    shadow_offset = 4
    draw.multiline_text(
        (x + shadow_offset, y + shadow_offset),
        text_block, fill=(0, 0, 0), font=font, align="center", spacing=0,
    )
    draw.multiline_text(
        (x, y), text_block, fill=rgb, font=font, align="center", spacing=0,
    )

    img.save(output_path)


def _wrap_text(draw: ImageDraw.Draw, text: str, font, max_width: int) -> list[str]:
    """Simple word-wrap for Pillow text rendering."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def generate_thumbnail(draft: dict, out_dir: Path) -> Path:
    """Generate a YouTube thumbnail with Gemini + text overlay.

    Uses the thumbnail_prompt from the draft, overlays the video title.
    Returns path to the final thumbnail PNG.
    """
    api_key = get_gemini_key()
    prompt = draft.get("thumbnail_prompt", "Cinematic YouTube thumbnail")
    title = draft.get("youtube_title", draft.get("news", ""))
    job_id = draft.get("job_id", "unknown")
    profile = load_niche(draft.get("niche", "general"))
    thumb_config = get_thumbnail_config(profile)

    raw_path = out_dir / f"thumb_raw_{job_id}.png"
    final_path = out_dir / f"thumb_{job_id}.png"

    if api_key:
        try:
            log("Generating thumbnail via Gemini Imagen...")
            _generate_thumb_image(prompt, raw_path, api_key)
        except Exception as e:
            log(f"Gemini thumbnail failed: {e} — trying no-key visual fallback")
            try:
                _generate_thumb_pollinations(prompt, raw_path)
            except Exception as fallback_error:
                log(f"Pollinations thumbnail failed: {fallback_error} — using local dark archive thumbnail")
                _generate_local_dark_thumb(prompt, raw_path)
    else:
        log("GEMINI_API_KEY not set — generating thumbnail via no-key visual fallback")
        try:
            _generate_thumb_pollinations(prompt, raw_path)
        except Exception as e:
            log(f"Pollinations thumbnail failed: {e} — using local dark archive thumbnail")
            _generate_local_dark_thumb(prompt, raw_path)

    log("Adding title overlay...")
    _overlay_title(raw_path, title, final_path, thumb_config)

    log(f"Thumbnail saved: {final_path.name}")
    return final_path
