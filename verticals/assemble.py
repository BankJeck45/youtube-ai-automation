"""ffmpeg video assembly — frames + voiceover + music + captions."""

from dataclasses import dataclass
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont

from .broll import animate_frame
from .config import MEDIA_DIR, VIDEO_HEIGHT, VIDEO_WIDTH, run_cmd
from .log import log


def _ffmpeg_has_libass() -> bool:
    """Check whether this ffmpeg build ships the `ass` filter (libass).

    Some builds (e.g. minimal/static ones) omit libass; burning captions in
    would fail with `No such filter: 'ass'`, so we skip burn-in instead.
    """
    try:
        r = run_cmd(["ffmpeg", "-hide_banner", "-filters"], capture=True)
        return any(line.split()[1:2] == ["ass"] for line in r.stdout.splitlines())
    except Exception:
        return False


def get_audio_duration(path: Path) -> float:
    """Get duration of an audio file in seconds."""
    r = run_cmd(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture=True,
    )
    return float(r.stdout.strip())


def _srt_timestamp_to_seconds(value: str) -> float:
    h, m, rest = value.strip().split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _parse_srt(path: str | Path | None) -> list[tuple[float, float, str]]:
    if not path or not Path(path).exists():
        return []

    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text.strip())
    cues: list[tuple[float, float, str]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        timing_line = next((line for line in lines if "-->" in line), "")
        if not timing_line:
            continue
        start_s, end_s = [p.strip().split()[0] for p in timing_line.split("-->", 1)]
        caption_lines = lines[lines.index(timing_line) + 1:]
        caption = " ".join(caption_lines).strip()
        if caption:
            cues.append((_srt_timestamp_to_seconds(start_s), _srt_timestamp_to_seconds(end_s), caption))
    return cues


def _load_caption_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _hex_to_rgb(value: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    raw = (value or "").lstrip("#")
    if len(raw) != 6:
        return fallback
    try:
        return tuple(int(raw[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return fallback


def _wrap_caption(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
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
    return lines[:3]


def _caption_overlay_pngs(
    cues: list[tuple[float, float, str]],
    out_dir: Path,
    style: dict | None = None,
) -> list[tuple[float, float, Path]]:
    """Render SRT cues as transparent PNG overlays for ffmpeg overlay fallback."""
    style = style or {}
    font_size = int(style.get("font_size", 52))
    position = str(style.get("position", "lower_left")).lower()
    text_color = _hex_to_rgb(str(style.get("text_color", "#D8D8D8")), (216, 216, 216))
    font = _load_caption_font(font_size)
    overlays: list[tuple[float, float, Path]] = []
    overlay_dir = out_dir / "caption_overlays"
    overlay_dir.mkdir(exist_ok=True)

    for i, (start, end, text) in enumerate(cues):
        img = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        max_width = int(VIDEO_WIDTH * (0.74 if "left" in position else 0.82))
        lines = _wrap_caption(draw, text, font, max_width)
        if not lines:
            continue
        line_height = int(font_size * 1.25)
        block_height = line_height * len(lines)
        margin_x = 72
        margin_bottom = int(VIDEO_HEIGHT * 0.14)
        y = VIDEO_HEIGHT - margin_bottom - block_height
        if "third" in position and "left" not in position:
            y = int(VIDEO_HEIGHT * 0.62)

        if "left" in position:
            x = margin_x
            align = "left"
        else:
            widest = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
            x = (VIDEO_WIDTH - widest) // 2
            align = "center"

        # Soft dark backing, close to the reference but not a heavy caption box.
        backing_w = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines) + 40
        backing_h = block_height + 28
        backing_x = max(20, x - 20)
        backing_y = max(20, y - 14)
        draw.rounded_rectangle(
            (backing_x, backing_y, min(VIDEO_WIDTH - 20, backing_x + backing_w), backing_y + backing_h),
            radius=14,
            fill=(0, 0, 0, 74),
        )

        for n, line in enumerate(lines):
            yy = y + n * line_height
            if align == "center":
                bbox = draw.textbbox((0, 0), line, font=font)
                xx = (VIDEO_WIDTH - (bbox[2] - bbox[0])) // 2
            else:
                xx = x
            for ox, oy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                draw.text((xx + ox, yy + oy), line, font=font, fill=(0, 0, 0, 205))
            draw.text((xx, yy), line, font=font, fill=(*text_color, 255))

        path = overlay_dir / f"caption_{i:03d}.png"
        img.save(path)
        overlays.append((start, end, path))

    return overlays


def _burn_captions_with_overlays(
    video_path: Path,
    srt_path: str | Path | None,
    out_dir: Path,
    caption_style: dict | None = None,
) -> Path | None:
    cues = _parse_srt(srt_path)
    if not cues:
        return None

    overlays = _caption_overlay_pngs(cues, out_dir, caption_style)
    if not overlays:
        return None

    captioned = out_dir / "merged_video_captioned.mp4"
    cmd = ["ffmpeg", "-i", str(video_path)]
    for _, _, png in overlays:
        cmd += ["-loop", "1", "-i", str(png)]

    chain = []
    current = "[0:v]"
    for idx, (start, end, _) in enumerate(overlays, 1):
        out_label = f"[cv{idx}]"
        enable = f"between(t\\,{start:.3f}\\,{end:.3f})"
        chain.append(f"{current}[{idx}:v]overlay=0:0:enable='{enable}'{out_label}")
        current = out_label

    run_cmd([
        *cmd,
        "-filter_complex", ";".join(chain),
        "-map", current,
        "-an",
        "-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p",
        str(captioned), "-y", "-loglevel", "quiet",
    ])
    log(f"Captions burned with PNG overlay fallback: {captioned.name}")
    return captioned


@dataclass(frozen=True)
class FinalEncoding:
    name: str
    extension: str
    video_args: list[str]
    audio_args: list[str]
    browser_safe: bool = True


def _ffmpeg_encoders() -> set[str]:
    try:
        r = run_cmd(["ffmpeg", "-hide_banner", "-encoders"], capture=True)
    except Exception:
        return set()

    encoders: set[str] = set()
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and (parts[0].startswith("V") or parts[0].startswith("A")):
            encoders.add(parts[1])
    return encoders


def _choose_final_encoding() -> FinalEncoding:
    encoders = _ffmpeg_encoders()

    if "libx264" in encoders:
        return FinalEncoding(
            name="H.264/libx264",
            extension="mp4",
            video_args=["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            audio_args=["-c:a", "aac", "-b:a", "128k"],
        )

    if "libopenh264" in encoders:
        return FinalEncoding(
            name="H.264/libopenh264",
            extension="mp4",
            video_args=["-c:v", "libopenh264", "-b:v", "2600k", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            audio_args=["-c:a", "aac", "-b:a", "128k"],
        )

    if "h264_videotoolbox" in encoders:
        return FinalEncoding(
            name="H.264/videotoolbox",
            extension="mp4",
            video_args=["-c:v", "h264_videotoolbox", "-b:v", "2600k", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            audio_args=["-c:a", "aac", "-b:a", "128k"],
        )

    if "libvpx-vp9" in encoders and "libopus" in encoders:
        return FinalEncoding(
            name="WebM/VP9",
            extension="webm",
            video_args=["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-pix_fmt", "yuv420p"],
            audio_args=["-c:a", "libopus", "-b:a", "128k"],
        )

    if "libvpx" in encoders and "libvorbis" in encoders:
        return FinalEncoding(
            name="WebM/VP8",
            extension="webm",
            video_args=["-c:v", "libvpx", "-b:v", "2600k", "-pix_fmt", "yuv420p"],
            audio_args=["-c:a", "libvorbis", "-q:a", "4"],
        )

    log(
        "WARNING: no browser-safe ffmpeg video encoder found. Falling back to "
        "mpeg4-in-mp4, which may play as audio-only in some browsers. Install "
        "ffmpeg with libx264 or libvpx/libopus on the backend."
    )
    return FinalEncoding(
        name="MPEG-4 Part 2 fallback",
        extension="mp4",
        video_args=["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"],
        audio_args=["-c:a", "aac", "-b:a", "128k"],
        browser_safe=False,
    )


def assemble_video(
    frames: list[Path],
    voiceover: Path,
    out_dir: Path,
    job_id: str,
    lang: str = "en",
    ass_path: str | None = None,
    srt_path: str | None = None,
    caption_style: dict | None = None,
    music_path: str | None = None,
    duck_filter: str | None = None,
) -> Path:
    """Assemble final video from frames, voiceover, captions, and music."""
    log("Assembling video...")
    duration = get_audio_duration(voiceover)
    per_frame = duration / len(frames)
    effects = ["zoom_in", "pan_right", "zoom_out"]

    # Animate each frame with Ken Burns effect
    animated = []
    for i, frame in enumerate(frames):
        anim = out_dir / f"anim_{i}.mp4"
        animate_frame(frame, anim, per_frame + 0.1, effects[i % len(effects)])
        animated.append(anim)

    # Concat animated segments (escape single quotes for ffmpeg concat demuxer)
    concat_file = out_dir / "concat.txt"
    def _esc(p):
        return str(p).replace("'", "'\\''" )
    concat_file.write_text("\n".join(f"file '{_esc(p)}'" for p in animated))

    merged_video = out_dir / "merged_video.mp4"
    run_cmd([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p",
        str(merged_video), "-y", "-loglevel", "quiet",
    ])

    # Build the final ffmpeg command with optional captions + music
    encoding = _choose_final_encoding()
    log(f"Final video encoding: {encoding.name}")
    out_path = MEDIA_DIR / f"verticals_{job_id}_{lang}.{encoding.extension}"

    # Determine caption path. ASS is best when libass exists; otherwise burn
    # SRT captions through PNG overlays so the final video still has text.
    vf_parts = []
    video_input = merged_video
    has_ass = bool(ass_path and Path(ass_path).exists())
    has_srt = bool(srt_path and Path(srt_path).exists())
    if has_ass:
        if _ffmpeg_has_libass():
            # Escape special chars in path for ffmpeg filter
            escaped_ass = str(ass_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            vf_parts.append(f"ass={escaped_ass}")
        else:
            log(
                "WARNING: this ffmpeg build has no libass — using PNG overlay "
                "fallback for burned-in captions."
            )
            try:
                captioned_video = _burn_captions_with_overlays(
                    merged_video, srt_path, out_dir, caption_style,
                )
            except Exception as e:
                captioned_video = None
                log(f"WARNING: caption overlay fallback failed: {e}")
            if captioned_video:
                video_input = captioned_video
            else:
                log("WARNING: final video will have no burned-in captions")
    elif has_srt:
        log("ASS captions unavailable — using PNG overlay fallback for burned-in SRT captions.")
        try:
            captioned_video = _burn_captions_with_overlays(
                merged_video, srt_path, out_dir, caption_style,
            )
        except Exception as e:
            captioned_video = None
            log(f"WARNING: caption overlay fallback failed: {e}")
        if captioned_video:
            video_input = captioned_video
        else:
            log("WARNING: final video will have no burned-in captions")
    vf = ",".join(vf_parts) if vf_parts else None

    if music_path and Path(music_path).exists():
        # Three inputs: video, voiceover, music
        cmd = ["ffmpeg", "-i", str(video_input), "-i", str(voiceover)]

        # Loop music to match video duration, apply ducking
        music_filter = f"[2:a]aloop=loop=-1:size=2e+09,atrim=0:{duration}"
        if duck_filter:
            music_filter += f",{duck_filter}"
        music_filter += "[music]"

        # Mix voiceover + ducked music
        audio_filter = f"{music_filter};[1:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"

        cmd += [
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex", audio_filter,
        ]

        if vf:
            cmd += ["-vf", vf]

        cmd += [
            "-map", "0:v", "-map", "[aout]",
            *encoding.video_args,
            *encoding.audio_args,
            "-shortest",
            str(out_path), "-y", "-loglevel", "quiet",
        ]
    else:
        # Two inputs: video + voiceover (no music)
        cmd = ["ffmpeg", "-i", str(video_input), "-i", str(voiceover)]

        if vf:
            cmd += ["-vf", vf]

        cmd += [
            "-map", "0:v:0", "-map", "1:a:0",
            *encoding.video_args,
            *encoding.audio_args,
            "-shortest",
            str(out_path), "-y", "-loglevel", "quiet",
        ]

    run_cmd(cmd)
    log(f"Video assembled: {out_path}")
    return out_path
