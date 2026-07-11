"""Gemini image b-roll generation + Ken Burns animation."""

import base64
import hashlib
import os
import textwrap
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from .config import VIDEO_WIDTH, VIDEO_HEIGHT, get_gemini_key, run_cmd
from .log import log
from .retry import with_retry


def _extract_image_bytes(data: dict) -> bytes | None:
    """Extract base64 image bytes from both current and legacy Gemini responses."""
    output_image = data.get("output_image") or data.get("outputImage")
    if isinstance(output_image, dict) and output_image.get("data"):
        return base64.b64decode(output_image["data"])

    for step in data.get("steps", []):
        if step.get("type") != "model_output":
            continue
        for block in step.get("content", []):
            if block.get("type") == "image" and block.get("data"):
                return base64.b64decode(block["data"])

    # Legacy generateContent response shape.
    for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            return base64.b64decode(inline["data"])

    # Imagen predict response shape.
    for prediction in data.get("predictions", []):
        image_data = prediction.get("bytesBase64Encoded") or prediction.get("image", {}).get("bytesBase64Encoded")
        if image_data:
            return base64.b64decode(image_data)

    return None


@with_retry(max_retries=1, base_delay=2.0)
def _generate_image_gemini(prompt: str, output_path: Path, api_key: str):
    """Generate a portrait b-roll frame via Gemini native image generation."""
    url = "https://generativelanguage.googleapis.com/v1beta/interactions"
    model = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-lite-image")
    image_prompt = (
        f"{prompt}. Vertical 9:16 YouTube Shorts b-roll frame, cinematic, "
        "photorealistic, visually clear, no captions, no text overlays."
    )
    body = {
        "model": model,
        "input": image_prompt,
        "response_format": {
            "type": "image",
            "aspect_ratio": "9:16",
        },
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
    image_bytes = _extract_image_bytes(data)
    if image_bytes:
        output_path.write_bytes(image_bytes)
        return
    raise RuntimeError("No image in Gemini response")


def _load_font(size: int, bold: bool = False):
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.replace("\n", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if not lines:
        lines = textwrap.wrap(text, width=26) or ["Untitled visual"]
    return lines[:7]


def _fallback_frame(i: int, out_dir: Path, prompt: str = "") -> Path:
    """Readable graphic fallback if Gemini fails or is not configured."""
    seed = int(hashlib.sha256(f"{prompt}:{i}".encode()).hexdigest()[:8], 16)
    palettes = [
        ((12, 18, 42), (37, 99, 235), (245, 158, 11)),
        ((18, 24, 38), (16, 185, 129), (244, 63, 94)),
        ((24, 16, 44), (168, 85, 247), (34, 211, 238)),
    ]
    bg, primary, accent = palettes[seed % len(palettes)]
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), bg)
    draw = ImageDraw.Draw(img, "RGBA")

    for y in range(VIDEO_HEIGHT):
        t = y / VIDEO_HEIGHT
        r = int(bg[0] * (1 - t) + max(primary[0] - 40, 0) * t)
        g = int(bg[1] * (1 - t) + max(primary[1] - 40, 0) * t)
        b = int(bg[2] * (1 - t) + max(primary[2] - 40, 0) * t)
        draw.line([(0, y), (VIDEO_WIDTH, y)], fill=(r, g, b, 255))

    for n in range(9):
        x = (seed * (n + 3) * 37) % VIDEO_WIDTH
        y = (seed * (n + 5) * 53) % VIDEO_HEIGHT
        radius = 120 + ((seed >> (n % 8)) % 260)
        color = primary if n % 2 == 0 else accent
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 28))

    margin = 96
    card_top = 430
    card_bottom = 1450
    draw.rounded_rectangle(
        (margin, card_top, VIDEO_WIDTH - margin, card_bottom),
        radius=44,
        fill=(5, 8, 18, 178),
        outline=(*accent, 180),
        width=3,
    )

    label_font = _load_font(34, bold=True)
    title_font = _load_font(70, bold=True)
    small_font = _load_font(34)
    draw.text((margin + 52, card_top + 58), "B-ROLL VISUAL", font=label_font, fill=(*accent, 255))

    clean_prompt = prompt.strip() or "Cinematic visual scene"
    clean_prompt = clean_prompt.replace("photorealistic, cinematic lighting, high quality, 4K", "")
    lines = _wrap(draw, clean_prompt, title_font, VIDEO_WIDTH - (margin + 52) * 2)
    y = card_top + 140
    for line in lines:
        draw.text((margin + 52, y), line, font=title_font, fill=(245, 247, 255, 255))
        y += 86

    draw.rectangle((margin + 52, card_bottom - 170, VIDEO_WIDTH - margin - 52, card_bottom - 164), fill=(*accent, 210))
    draw.text(
        (margin + 52, card_bottom - 118),
        f"SCENE {i + 1:02d}",
        font=small_font,
        fill=(207, 213, 225, 255),
    )

    path = out_dir / f"broll_{i}.png"
    img.save(path)
    return path


def generate_broll(prompts: list, out_dir: Path) -> list[Path]:
    """Generate 3 b-roll frames via Gemini Imagen, with fallback."""
    api_key = get_gemini_key()
    if not api_key:
        log(
            "GEMINI_API_KEY not set — using solid-color fallback frames. "
            "Get an AI Studio key at https://aistudio.google.com/apikey "
            "(must be an AI Studio key; Vertex AI / service-account credentials "
            "are rejected with a 403 'unregistered callers' error)."
        )
        return [_fallback_frame(i, out_dir, prompts[i % len(prompts)] if prompts else "") for i in range(min(3, max(len(prompts), 1)))]
    frames = []

    for i, prompt in enumerate(prompts[:3]):
        out_path = out_dir / f"broll_{i}.png"
        log(f"Generating b-roll frame {i+1}/3 via Gemini Imagen...")

        try:
            _generate_image_gemini(prompt, out_path, api_key)

            # Resize/crop to 9:16 portrait
            img = Image.open(out_path).convert("RGB")
            target_w, target_h = VIDEO_WIDTH, VIDEO_HEIGHT
            orig_w, orig_h = img.size
            scale = max(target_w / orig_w, target_h / orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            img = img.crop((left, top, left + target_w, top + target_h))
            img.save(out_path)
            frames.append(out_path)

        except Exception as e:
            log(f"Frame {i+1} failed: {e} — using fallback")
            frames.append(_fallback_frame(i, out_dir, prompt))

    return frames


def animate_frame(img_path: Path, out_path: Path, duration: float, effect: str = "zoom_in"):
    """Ken Burns animation on a single frame."""
    fps = 30
    frames = int(duration * fps)
    w, h = VIDEO_WIDTH, VIDEO_HEIGHT

    if effect == "zoom_in":
        vf = (
            f"scale={int(w * 1.12)}:{int(h * 1.12)},"
            f"zoompan=z='1.12-0.12*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={w}x{h}:fps={fps}"
        )
    elif effect == "pan_right":
        vf = (
            f"scale={int(w * 1.15)}:{int(h * 1.15)},"
            f"zoompan=z=1.15:x='0.15*iw*on/{frames}':y='ih*0.075'"
            f":d={frames}:s={w}x{h}:fps={fps}"
        )
    else:  # zoom_out
        vf = (
            f"scale={int(w * 1.12)}:{int(h * 1.12)},"
            f"zoompan=z='1.0+0.12*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={w}x{h}:fps={fps}"
        )

    run_cmd([
        "ffmpeg", "-loop", "1", "-i", str(img_path),
        "-vf", vf, "-t", str(duration), "-r", str(fps),
        "-pix_fmt", "yuv420p", str(out_path), "-y", "-loglevel", "quiet",
    ])
