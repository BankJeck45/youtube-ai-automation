"""Gemini image b-roll generation + Ken Burns animation."""

import base64
import hashlib
from io import BytesIO
import math
import os
import re
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageOps

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


def _fit_to_portrait(image: Image.Image, output_path: Path):
    """Resize/crop any image into the configured 9:16 video frame."""
    img = ImageOps.exif_transpose(image).convert("RGB")
    target_w, target_h = VIDEO_WIDTH, VIDEO_HEIGHT
    orig_w, orig_h = img.size
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))
    img.save(output_path)


@with_retry(max_retries=1, base_delay=2.0)
def _generate_image_pollinations(prompt: str, output_path: Path):
    """Generate a portrait b-roll frame through Pollinations as a no-key fallback."""
    enabled = os.environ.get("POLLINATIONS_ENABLED", "true").lower()
    if enabled in {"0", "false", "no", "off"}:
        raise RuntimeError("Pollinations fallback disabled")

    seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
    image_prompt = (
        f"{prompt}. Vertical 9:16 YouTube Shorts b-roll frame, cinematic, "
        "photorealistic, visually clear, no captions, no text overlays."
    )
    url = f"https://image.pollinations.ai/prompt/{quote(image_prompt, safe='')}"
    params = {
        "width": os.environ.get("POLLINATIONS_WIDTH", "720"),
        "height": os.environ.get("POLLINATIONS_HEIGHT", "1280"),
        "nologo": "true",
        "private": "true",
        "seed": str(seed),
    }
    model = os.environ.get("POLLINATIONS_IMAGE_MODEL", "").strip()
    if model:
        params["model"] = model

    r = requests.get(url, params=params, timeout=150, headers={"Accept": "image/*"})
    content_type = r.headers.get("Content-Type", "")
    if r.status_code != 200 or not content_type.startswith("image/"):
        detail = r.text[:200] if getattr(r, "text", "") else content_type
        raise RuntimeError(f"Pollinations image {r.status_code}: {detail}")

    img = Image.open(BytesIO(r.content))
    _fit_to_portrait(img, output_path)


def estimate_visual_count(duration: float, target_seconds: float = 4.5, min_count: int = 3, max_count: int = 24) -> int:
    """Return a b-roll count that changes visuals about every 3-6 seconds."""
    if duration <= 0:
        return min_count
    count = math.ceil(duration / max(target_seconds, 3.0))
    return max(min_count, min(max_count, count))


def _split_script_beats(script: str, target_count: int) -> list[str]:
    """Split narration into roughly equal visual beats."""
    cleaned = re.sub(r"\s+", " ", (script or "").strip())
    if not cleaned:
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if not sentences:
        sentences = [cleaned]

    if len(sentences) >= target_count:
        return sentences[:target_count]

    words = cleaned.split()
    if not words:
        return sentences

    per_group = max(6, math.ceil(len(words) / max(target_count, 1)))
    beats = []
    for i in range(0, len(words), per_group):
        beats.append(" ".join(words[i:i + per_group]))
    return beats[:target_count]


def build_timed_broll_prompts(
    script: str,
    base_prompts: list[str],
    target_count: int,
    prompt_suffix: str = "",
) -> list[str]:
    """Create narration-aligned image prompts for each 3-6 second segment."""
    base = [str(p).strip() for p in (base_prompts or []) if str(p).strip()]
    if not base:
        base = ["cinematic scene matching the narration"]

    beats = _split_script_beats(script, target_count)
    prompts = []
    for i in range(target_count):
        beat = beats[i] if i < len(beats) else beats[-1] if beats else ""
        anchor = base[i % len(base)]
        parts = [
            anchor,
            f"narration beat {i + 1}: {beat}" if beat else "",
            "show the scene described by this beat, no text, no subtitles, no logos, no UI words",
            "thin atmospheric smoke or fog, subtle film grain, slow cinematic motion feel",
            prompt_suffix,
        ]
        prompts.append(". ".join(p.strip().strip(".") for p in parts if p).strip() + ".")
    return prompts


def _fallback_frame(i: int, out_dir: Path, prompt: str = "") -> Path:
    """Text-free graphic fallback if all image providers fail."""
    seed = int(hashlib.sha256(f"{prompt}:{i}".encode()).hexdigest()[:8], 16)
    palettes = [
        ((16, 24, 52), (37, 99, 235), (245, 158, 11)),
        ((16, 42, 46), (20, 184, 166), (244, 63, 94)),
        ((34, 24, 64), (168, 85, 247), (34, 211, 238)),
        ((38, 34, 28), (234, 88, 12), (132, 204, 22)),
    ]
    bg, primary, accent = palettes[seed % len(palettes)]
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), bg)
    draw = ImageDraw.Draw(img, "RGBA")

    for y in range(VIDEO_HEIGHT):
        t = y / VIDEO_HEIGHT
        r = int(bg[0] * (1 - t) + primary[0] * t)
        g = int(bg[1] * (1 - t) + primary[1] * t)
        b = int(bg[2] * (1 - t) + primary[2] * t)
        draw.line([(0, y), (VIDEO_WIDTH, y)], fill=(r, g, b, 255))

    # Soft depth and motion. Keep it visual-only: no prompt text, no labels.
    for n in range(16):
        x = (seed * (n + 3) * 37) % VIDEO_WIDTH
        y = (seed * (n + 5) * 53) % VIDEO_HEIGHT
        radius = 100 + ((seed >> (n % 8)) % 300)
        color = primary if n % 2 == 0 else accent
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 24))

    # Build a cinematic "scene" from silhouettes, panels, and light trails.
    horizon = 1180 + (seed % 180)
    draw.rectangle((0, horizon, VIDEO_WIDTH, VIDEO_HEIGHT), fill=(0, 0, 0, 42))

    for n in range(11):
        x = 90 + ((seed * (n + 11) * 19) % 900)
        width = 28 + ((seed >> (n % 10)) % 95)
        height = 220 + ((seed * (n + 7)) % 720)
        top = max(260, horizon - height)
        color = primary if n % 3 else accent
        draw.rounded_rectangle(
            (x, top, x + width, horizon + 80),
            radius=10,
            fill=(*color, 32 + (n % 4) * 10),
            outline=(*color, 80),
            width=2,
        )
        for m in range(3):
            yy = top + 34 + m * 74
            if yy < horizon:
                draw.line((x + 10, yy, x + width - 10, yy), fill=(*accent, 92), width=3)

    for n in range(18):
        y = 260 + ((seed * (n + 13) * 17) % 1200)
        x1 = -80 + ((seed * (n + 3)) % 340)
        x2 = VIDEO_WIDTH + 80 - ((seed * (n + 9)) % 320)
        color = accent if n % 2 else primary
        draw.line((x1, y, x2, y + ((n % 5) - 2) * 42), fill=(*color, 42), width=2 + (n % 4))

    center_x = VIDEO_WIDTH // 2 + ((seed % 161) - 80)
    center_y = 730 + ((seed >> 8) % 260)
    for radius, alpha in [(360, 28), (260, 46), (165, 72), (72, 150)]:
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            outline=(*accent, alpha),
            width=4,
        )

    draw.polygon(
        [
            (center_x, center_y - 220),
            (center_x + 230, center_y + 180),
            (center_x - 230, center_y + 180),
        ],
        fill=(*primary, 36),
        outline=(*accent, 90),
    )

    path = out_dir / f"broll_{i}.png"
    img.save(path)
    return path


def generate_broll(prompts: list, out_dir: Path) -> list[Path]:
    """Generate b-roll frames, preferring image providers before graphic fallback."""
    api_key = get_gemini_key()
    if not api_key:
        log(
            "GEMINI_API_KEY not set — trying no-key Pollinations image fallback. "
            "Get an AI Studio key at https://aistudio.google.com/apikey "
            "(must be an AI Studio key; Vertex AI / service-account credentials "
            "are rejected with a 403 'unregistered callers' error)."
        )
    frames = []
    selected_prompts = [str(p) for p in prompts if str(p).strip()] or ["Cinematic visual scene"]

    gemini_available = bool(api_key)

    for i, prompt in enumerate(selected_prompts):
        out_path = out_dir / f"broll_{i}.png"

        if gemini_available:
            log(f"Generating b-roll frame {i+1}/{len(selected_prompts)} via Gemini Imagen...")
            try:
                _generate_image_gemini(prompt, out_path, api_key)
                _fit_to_portrait(Image.open(out_path), out_path)
                frames.append(out_path)
                continue
            except Exception as e:
                log(f"Gemini frame {i+1} failed: {e} — trying Pollinations fallback")
                if "429" in str(e) or "quota" in str(e).lower():
                    gemini_available = False
                    log("Gemini quota/rate limit detected — using Pollinations for remaining b-roll frames")

        try:
            log(f"Generating b-roll frame {i+1}/{len(selected_prompts)} via Pollinations fallback...")
            _generate_image_pollinations(prompt, out_path)
            frames.append(out_path)
            continue

        except Exception as e:
            log(f"Image fallback frame {i+1} failed: {e} — using readable graphic fallback")
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
