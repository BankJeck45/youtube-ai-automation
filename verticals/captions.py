"""Whisper word-level timestamps + ASS subtitle generation + Pillow fallback."""

from pathlib import Path
import re

from .log import log


def _has_ass_filter() -> bool:
    """Check if ffmpeg has libass (for ASS subtitle burn-in)."""
    import subprocess
    try:
        r = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True, text=True, timeout=5,
        )
        return "ass" in r.stdout
    except Exception:
        return False


def _whisper_word_timestamps(audio_path: Path, lang: str = "en") -> list[dict]:
    """Get word-level timestamps from Whisper.

    Returns list of {"word": str, "start": float, "end": float}.
    """
    try:
        import whisper
    except ImportError:
        log("Whisper not installed — skipping word timestamps")
        return []

    log("Running Whisper for word-level timestamps...")
    model = whisper.load_model("base")
    result = model.transcribe(
        str(audio_path),
        language=lang[:2],
        word_timestamps=True,
    )

    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": w["start"],
                "end": w["end"],
            })

    log(f"Got {len(words)} word timestamps.")
    return words


def _group_words(words: list[dict], group_size: int = 4) -> list[list[dict]]:
    groups = []
    for i in range(0, len(words), group_size):
        groups.append(words[i:i + group_size])
    return groups


def _hex_to_ass_color(color: str, default: str = "&HFFFFFF&", alpha: str = "") -> str:
    """Convert #RRGGBB to ASS BGR color."""
    value = (color or "").lstrip("#")
    if len(value) != 6:
        return default
    return f"&H{alpha}{value[4:6]}{value[2:4]}{value[0:2]}&"


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _estimated_word_timestamps(script_text: str, duration: float) -> list[dict]:
    """Estimate word timings from script text when Whisper is unavailable."""
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s]", script_text or "")
    words = [t for t in tokens if re.search(r"[A-Za-z0-9]", t)]
    if not words or duration <= 0:
        return []

    usable_duration = max(1.0, duration - 0.6)
    weights = [max(1.0, min(8.0, len(w) / 3)) for w in words]
    total = sum(weights)
    t = 0.3
    result = []
    for word, weight in zip(words, weights):
        span = max(0.18, usable_duration * (weight / total))
        start = t
        end = min(duration, start + span)
        result.append({"word": word, "start": start, "end": end})
        t = end
    return result


def _format_ass_time(seconds: float) -> str:
    """Format seconds to ASS timestamp: H:MM:SS.cc (centiseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _generate_ass(
    words: list[dict],
    output_path: Path,
    video_width: int = 1080,
    video_height: int = 1920,
    highlight_color: str = "#FFFF00",
    text_color: str = "#FFFFFF",
    group_size: int = 4,
    font_family: str = "Arial",
    font_size: int = 72,
    highlight_words: bool = True,
    position: str = "lower_third",
    outline: int = 3,
    shadow: int = 0,
):
    """Generate ASS subtitle file with word-by-word color highlighting.

    White text for inactive words, highlight color for current word.
    Semi-transparent background, positioned at lower third (~70% down).

    The font_family is taken from the niche profile (captions.font_family) so
    non-Latin scripts (Korean, Japanese, Chinese, Arabic, etc.) can render
    correctly. The default "Arial" preserves the original behavior for English.
    """
    # ASS header
    position_key = (position or "").lower()
    alignment = 1 if position_key in {"lower_left", "left_lower", "reference"} else 2
    margin_l = 72 if alignment == 1 else 40
    margin_r = 72 if alignment == 1 else 40
    margin_v = int(video_height * (0.14 if alignment == 1 else 0.25))
    primary_color = _hex_to_ass_color(text_color, "&H00FFFFFF&", alpha="00")
    header = f"""[Script Info]
Title: Pipeline Captions
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_family},{font_size},{primary_color},&H000000FF,&H00000000,&H70000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},{margin_l},{margin_r},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Convert hex color to ASS BGR format (e.g. #00FF88 -> 88FF00).
    # Override tags use &HBBGGRR& without the alpha byte.
    hc = highlight_color.lstrip("#")
    if len(hc) == 6:
        ass_highlight = f"&H{hc[4:6]}{hc[2:4]}{hc[0:2]}&"
    else:
        ass_highlight = "&H00FFFF&"  # fallback yellow

    groups = _group_words(words, group_size=group_size)
    events = []

    for group in groups:
        if not group:
            continue

        group_start = group[0]["start"]
        group_end = group[-1]["end"]
        if not highlight_words:
            text = " ".join(_escape_ass_text(w["word"]) for w in group)
            events.append(
                f"Dialogue: 0,{_format_ass_time(group_start)},{_format_ass_time(group_end)},Default,,0,0,0,,{text}"
            )
            continue

        # For each word in the group being active, emit one dialogue line
        for active_idx, active_word in enumerate(group):
            start = active_word["start"]
            end = active_word["end"]

            # Build text with override tags: highlight color for active, white for rest
            parts = []
            for j, w in enumerate(group):
                if j == active_idx:
                    parts.append(f"{{\\c{ass_highlight}\\b1\\fs{font_size + 8}}}{_escape_ass_text(w['word'])}{{\\r}}")
                else:
                    parts.append(_escape_ass_text(w["word"]))

            text = " ".join(parts)
            events.append(
                f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},Default,,0,0,0,,{text}"
            )

    output_path.write_text(header + "\n".join(events), encoding="utf-8")
    log(f"ASS captions saved: {output_path.name}")
    return output_path


def _generate_srt(words: list[dict], output_path: Path, group_size: int = 4) -> Path:
    """Generate standard SRT file from word timestamps."""
    groups = _group_words(words, group_size=group_size)
    lines = []

    for i, group in enumerate(groups, 1):
        if not group:
            continue
        start = group[0]["start"]
        end = group[-1]["end"]
        text = " ".join(w["word"] for w in group)

        start_ts = _srt_time(start)
        end_ts = _srt_time(end)
        lines.append(f"{i}\n{start_ts} --> {end_ts}\n{text}\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"SRT captions saved: {output_path.name}")
    return output_path


def _srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp: HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_captions(
    audio_path: Path,
    work_dir: Path,
    lang: str = "en",
    highlight_color: str = "#FFFF00",
    words_per_group: int = 4,
    font_family: str = "Arial",
    font_size: int = 72,
    script_text: str = "",
    audio_duration: float = 0.0,
    text_color: str = "#FFFFFF",
    highlight_words: bool = True,
    position: str = "lower_third",
    outline: int = 3,
    shadow: int = 0,
) -> dict:
    """Generate captions: ASS (for burn-in) + SRT (for YouTube upload).

    Args:
        font_family: ASS Style font name. Use a CJK-capable font (e.g.
            "Noto Sans CJK KR", "Noto Sans CJK JP") for non-Latin languages,
            otherwise glyphs render as boxes. Pulled from the niche profile's
            captions.font_family field.
        font_size: ASS Style font size. Pulled from the niche profile's
            captions.font_size field.

    Returns dict with keys: srt_path, ass_path, words (for music ducking).
    """
    words = _whisper_word_timestamps(audio_path, lang)
    if not words and script_text:
        log("No Whisper word timestamps — estimating captions from script + audio duration")
        words = _estimated_word_timestamps(script_text, audio_duration)

    result = {"words": words}

    if not words:
        log("No word timestamps — skipping caption generation")
        # Fallback: run whisper CLI for SRT only
        try:
            from .config import run_cmd
            run_cmd([
                "whisper", str(audio_path),
                "--model", "base",
                "--language", lang[:2],
                "--output_format", "srt",
                "--output_dir", str(work_dir),
            ], capture=True)
            candidates = list(work_dir.glob("*.srt"))
            if candidates:
                srt = candidates[0]
                final = audio_path.with_suffix(".srt")
                srt.rename(final)
                result["srt_path"] = str(final)
        except Exception as e:
            log(f"Whisper CLI fallback failed: {e}")
        return result

    # Generate SRT
    srt_path = work_dir / f"captions_{lang}.srt"
    _generate_srt(words, srt_path, group_size=words_per_group)
    result["srt_path"] = str(srt_path)

    # Generate ASS for burn-in (niche-aware highlight color)
    ass_path = work_dir / f"captions_{lang}.ass"
    _generate_ass(
        words, ass_path,
        highlight_color=highlight_color,
        text_color=text_color,
        group_size=words_per_group,
        font_family=font_family,
        font_size=font_size,
        highlight_words=highlight_words,
        position=position,
        outline=outline,
        shadow=shadow,
    )
    result["ass_path"] = str(ass_path)

    return result
