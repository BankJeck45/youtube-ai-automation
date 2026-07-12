"""CLI entry point — python -m verticals."""

import argparse
import sys
import time
from pathlib import Path

from .config import CONFIG_FILE, DRAFTS_DIR, MEDIA_DIR, run_setup
from .log import log, set_verbose
from .niche import list_niches


def maybe_run_setup(args):
    """Run first-run setup only for commands that need creator credentials.

    Help, niche listing, topic discovery, and local/free-provider paths should
    not block on an interactive setup wizard.
    """
    if CONFIG_FILE.exists() or args.cmd not in {"draft", "run"}:
        return

    provider = getattr(args, "provider", None)
    if provider in {"ollama", "gemini", "openai"}:
        return

    print("  First run detected. Running setup...")
    run_setup()


def cmd_draft(args):
    from .draft import generate_draft
    from .state import PipelineState
    import json

    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = str(int(time.time()))

    niche = getattr(args, "niche", "general") or "general"
    platform = getattr(args, "platform", "shorts") or "shorts"
    provider = getattr(args, "provider", None)

    print(f"\n  Drafting: {args.news} [niche: {niche}, platform: {platform}]\n")
    draft = generate_draft(
        args.news,
        getattr(args, "context", ""),
        niche=niche,
        platform=platform,
        provider=provider,
    )
    draft["job_id"] = job_id

    out_path = DRAFTS_DIR / f"{job_id}.json"
    state = PipelineState(draft)
    state.complete_stage("research")
    state.complete_stage("draft")
    state.save(out_path)

    print(f"\n  Draft saved: {out_path}")
    print(f"\n  Script:\n{draft['script']}")
    print(f"\n  Title: {draft.get('youtube_title', '')}")
    print(f"\n  B-roll prompts:")
    for i, p in enumerate(draft.get("broll_prompts", [])):
        print(f"  {i+1}. {p}")

    return out_path


def cmd_produce(args):
    from .broll import build_timed_broll_prompts, estimate_visual_count, generate_broll
    from .tts import generate_voiceover
    from .captions import generate_captions
    from .music import select_and_prepare_music
    from .assemble import assemble_video, get_audio_duration
    from .thumbnail import generate_thumbnail
    from .niche import (
        load_niche,
        get_voice_config,
        get_caption_config,
        get_music_config,
        get_visual_prompt_suffix,
    )
    from .state import PipelineState
    import json
    import shutil

    draft_path = Path(args.draft)
    draft = json.loads(draft_path.read_text())
    job_id = draft["job_id"]
    lang = args.lang
    state = PipelineState(draft)

    # Load niche profile for voice/caption/music config
    niche_name = draft.get("niche", "general")
    profile = load_niche(niche_name)

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = MEDIA_DIR / f"work_{job_id}_{lang}"
    work_dir.mkdir(exist_ok=True)

    force = getattr(args, "force", False)
    tts_provider = getattr(args, "voice", None)
    script = getattr(args, "script", None) or (
        draft.get("script_hi") if lang == "hi" else draft.get("script")
    )

    print(f"\n  Producing {lang.upper()} video for job {job_id} [niche: {niche_name}]")

    draft["_producing"] = True
    draft.pop("_production_error", None)
    current_stage = "produce"
    previous_excepthook = sys.excepthook

    def record_uncaught_error(exc_type, exc, tb):
        if exc_type is not SystemExit and draft.get("_producing"):
            draft["_producing"] = False
            draft["_production_error"] = str(exc)
            state.fail_stage(current_stage, str(exc), percent=state._current_percent())
            state.save(draft_path)
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = record_uncaught_error

    def progress(stage: str, percent: int, message: str, artifacts: dict | None = None):
        nonlocal current_stage
        current_stage = stage
        state.update_progress(stage, percent, message, artifacts)
        state.save(draft_path)
        log(f"Progress {percent}% — {message}")

    def complete(stage: str, percent: int, artifacts: dict | None = None):
        state.complete_stage(stage, artifacts, percent=percent)
        state.save(draft_path)

    progress("produce", 3, "Produksi video dimulai")

    # Voiceover (niche-aware voice selection)
    if force or not state.is_done("voiceover"):
        progress("voiceover", 8, "Membuat voiceover TTS")
        voice_config = get_voice_config(
            profile,
            provider=tts_provider or "edge_tts",
            lang=lang,
        )
        vo_path = generate_voiceover(
            script, work_dir, lang,
            provider=tts_provider,
            voice_config=voice_config,
        )
        complete("voiceover", 18, {"path": str(vo_path)})
    else:
        log("Skipping voiceover (already done)")
        vo_path = Path(state.get_artifact("voiceover", "path"))
        progress("voiceover", 18, "Voiceover sudah ada, lanjut ke tahap berikutnya", {"path": str(vo_path)})

    try:
        progress("voiceover", 19, "Membaca durasi voiceover")
        voiceover_duration = get_audio_duration(vo_path)
    except Exception as e:
        log(f"Could not read voiceover duration: {e} — estimating from script length")
        voiceover_duration = max(12.0, len(script.split()) / 2.5)
    progress("voiceover", 20, f"Durasi voiceover sekitar {voiceover_duration:.1f} detik")

    # B-roll: one visual every ~3-6 seconds, aligned to narration beats.
    if force or not state.is_done("broll"):
        visual_cfg = profile.get("visuals", {})
        target_seconds = float(visual_cfg.get("segment_seconds", 4.5))
        max_visuals = int(visual_cfg.get("max_segments", 24))
        visual_count = estimate_visual_count(
            voiceover_duration,
            target_seconds=target_seconds,
            max_count=max_visuals,
        )
        progress("broll", 22, f"Menyiapkan {visual_count} visual b-roll")
        timed_prompts = build_timed_broll_prompts(
            script,
            draft.get("broll_prompts", ["Cinematic landscape"]),
            visual_count,
            get_visual_prompt_suffix(profile),
            visual_cfg,
        )
        draft["broll_prompts_timed"] = timed_prompts
        log(f"Generating {visual_count} narration-aligned b-roll frames (~{target_seconds:.1f}s each)...")

        def broll_progress(done: int, total: int, message: str):
            pct = 25 + int((max(0, done) / max(total, 1)) * 28)
            progress("broll", pct, message, {"done": done, "total": total})

        frames = generate_broll(timed_prompts, work_dir, progress_callback=broll_progress)
        complete("broll", 55, {"frames": [str(f) for f in frames], "prompts": timed_prompts})
    else:
        log("Skipping b-roll (already done)")
        frames = [Path(f) for f in state.get_artifact("broll", "frames", [])]
        progress("broll", 55, f"Visual b-roll sudah ada ({len(frames)} frame)")

    # Whisper + Captions (niche-aware styling)
    caption_config = get_caption_config(profile)
    if force or not state.is_done("captions"):
        progress("captions", 58, "Membuat subtitle/captions")
        captions_result = generate_captions(
            vo_path, work_dir, lang,
            highlight_color=caption_config.get("highlight_color", "#FFFF00"),
            words_per_group=caption_config.get("words_per_group", 4),
            font_family=caption_config.get("font_family", "Arial"),
            font_size=int(caption_config.get("font_size", 72)),
            script_text=script,
            audio_duration=voiceover_duration,
            text_color=caption_config.get("text_color", "#FFFFFF"),
            highlight_words=bool(caption_config.get("highlight_words", True)),
            position=caption_config.get("position", "lower_third"),
            outline=int(caption_config.get("outline", 3)),
            shadow=int(caption_config.get("shadow", 0)),
        )
        complete("captions", 66, {
            "srt_path": str(captions_result.get("srt_path", "")),
            "ass_path": str(captions_result.get("ass_path", "")),
        })
    else:
        log("Skipping captions (already done)")
        captions_result = {
            "srt_path": state.get_artifact("captions", "srt_path", ""),
            "ass_path": state.get_artifact("captions", "ass_path", ""),
        }
        progress("captions", 66, "Subtitle/captions sudah ada")

    # Music (niche-aware mood/ducking)
    music_config = get_music_config(profile)
    if force or not state.is_done("music"):
        progress("music", 68, "Menyiapkan backsound dan ducking suara")
        music_result = select_and_prepare_music(
            vo_path, work_dir,
            duck_speech=music_config.get("duck_volume_speech", 0.12),
            duck_gap=music_config.get("duck_volume_gap", 0.25),
        )
        complete("music", 72, {
            "track_path": str(music_result.get("track_path", "")),
            "duck_filter": music_result.get("duck_filter", ""),
        })
    else:
        log("Skipping music (already done)")
        music_result = {
            "track_path": state.get_artifact("music", "track_path", ""),
            "duck_filter": state.get_artifact("music", "duck_filter", ""),
        }
        progress("music", 72, "Backsound sudah ada")

    # Assemble
    if force or not state.is_done("assemble"):
        progress("assemble", 76, "Merender video final dengan visual, voiceover, subtitle, dan musik")
        video_path = assemble_video(
            frames=frames,
            voiceover=vo_path,
            out_dir=work_dir,
            job_id=job_id,
            lang=lang,
            ass_path=captions_result.get("ass_path"),
            srt_path=captions_result.get("srt_path"),
            caption_style=caption_config,
            music_path=music_result.get("track_path"),
            duck_filter=music_result.get("duck_filter"),
        )
        complete("assemble", 90, {"video_path": str(video_path)})
    else:
        log("Skipping assembly (already done)")
        video_path = Path(state.get_artifact("assemble", "video_path"))
        progress("assemble", 90, "Video final sudah dirender", {"video_path": str(video_path)})

    # Save SRT to media dir
    srt_path = captions_result.get("srt_path")
    if srt_path and Path(srt_path).exists():
        final_srt = MEDIA_DIR / f"verticals_{job_id}_{lang}.srt"
        shutil.copy(srt_path, final_srt)
        draft[f"srt_{lang}"] = str(final_srt)

    draft[f"video_{lang}"] = str(video_path)

    if force or not state.is_done("thumbnail"):
        try:
            progress("thumbnail", 92, "Membuat thumbnail")
            thumb_path = generate_thumbnail(draft, MEDIA_DIR)
            draft["thumbnail"] = str(thumb_path)
            complete("thumbnail", 98, {"path": str(thumb_path)})
        except Exception as e:
            log(f"Thumbnail generation failed: {e}")
            state.fail_stage("thumbnail", str(e), percent=98)
            state.save(draft_path)
    else:
        thumb_path = state.get_artifact("thumbnail", "path", "")
        if thumb_path:
            draft["thumbnail"] = thumb_path
        progress("thumbnail", 98, "Thumbnail sudah ada")

    draft["_producing"] = False
    draft.pop("_production_error", None)
    state.add_event("produce", "done", "Video selesai dibuat", 100)

    state.save(draft_path)

    print(f"\n  Video: {video_path}")
    return video_path


def cmd_upload(args):
    from .upload import upload_to_youtube
    from .thumbnail import generate_thumbnail
    from .state import PipelineState
    import json

    draft_path = Path(args.draft)
    draft = json.loads(draft_path.read_text())
    lang = args.lang
    state = PipelineState(draft)
    force = getattr(args, "force", False)

    video_path = Path(draft.get(f"video_{lang}", ""))
    srt_path_str = draft.get(f"srt_{lang}")
    srt_path = Path(srt_path_str) if srt_path_str else None

    if not video_path.exists():
        print(f"  No produced video found for lang={lang}. Run produce first.")
        sys.exit(1)

    # Thumbnail
    thumb_path = None
    if force or not state.is_done("thumbnail"):
        try:
            thumb_path = generate_thumbnail(draft, MEDIA_DIR)
            state.complete_stage("thumbnail", {"path": str(thumb_path)})
        except Exception as e:
            log(f"Thumbnail generation failed: {e} — uploading without thumbnail")
    else:
        thumb_p = state.get_artifact("thumbnail", "path", "")
        if thumb_p and Path(thumb_p).exists():
            thumb_path = Path(thumb_p)

    # Upload
    if force or not state.is_done("upload"):
        url = upload_to_youtube(video_path, draft, srt_path, lang, thumb_path)
        state.complete_stage("upload", {"url": url})
    else:
        url = state.get_artifact("upload", "url", "")
        log(f"Skipping upload (already done): {url}")

    draft[f"youtube_url_{lang}"] = url
    state.save(draft_path)
    print(f"\n  Live: {url}")
    return url


def cmd_run(args):
    draft_path = cmd_draft(args)
    if args.dry_run:
        print("  Dry run — skipping produce + upload")
        return

    class ProduceArgs:
        draft = str(draft_path)
        lang = args.lang
        script = None
        force = False
        voice = getattr(args, "voice", None)

    video_path = cmd_produce(ProduceArgs())

    class UploadArgs:
        draft = str(draft_path)
        lang = args.lang
        force = False

    url = cmd_upload(UploadArgs())
    print(f"\n  Done! {url}")


def cmd_topics(args):
    from .topics import TopicEngine

    niche = getattr(args, "niche", "general") or "general"
    engine = TopicEngine(niche=niche)
    candidates = engine.discover(limit=getattr(args, "limit", 15))

    if not candidates:
        print("  No topics found from enabled sources.")
        return

    print(f"\n  Trending topics for [{niche}] ({len(candidates)} found):\n")
    for i, topic in enumerate(candidates, 1):
        print(f"  {i:2d}. [{topic.source}] {topic.title}")
        if topic.summary:
            print(f"      {topic.summary[:100]}")


def cmd_niches(args):
    """List all available niche profiles."""
    niches = list_niches()
    print(f"\n  Available niches ({len(niches)}):\n")
    for n in niches:
        from .niche import load_niche
        profile = load_niche(n)
        display = profile.get("display_name", n)
        desc = profile.get("description", "")[:80]
        print(f"    {n:20s}  {display}")
        if desc:
            print(f"    {' ':20s}  {desc}")


def cmd_voices(args):
    """List voices available for a TTS provider.

    Currently only 60db is supported — it exposes GET /myvoices and
    GET /default-voices. Edge TTS voices are language-coded strings (see
    EDGE_VOICES in tts.py); ElevenLabs voice IDs come from the ElevenLabs
    dashboard.
    """
    provider = (args.provider or "").lower()
    if provider not in ("60db", "sixtydb"):
        print("  Error: --provider 60db is the only listing currently supported.")
        print("  Edge voices: see EDGE_VOICES in verticals/tts.py.")
        print("  ElevenLabs voices: https://elevenlabs.io/app/voice-library")
        sys.exit(1)

    import requests
    from .config import get_60db_key

    api_key = get_60db_key()
    if not api_key:
        print("  Error: SIXTYDB_API_KEY not set. Run setup or export the env var.")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {api_key}"}
    endpoints = [
        ("Default voices", "https://api.60db.ai/default-voices"),
        ("My voices",      "https://api.60db.ai/myvoices"),
    ]

    def _print_voice_row(v: dict):
        labels = v.get("labels") or {}
        lang = labels.get("language_name") or labels.get("language") or "?"
        gender = labels.get("gender") or "?"
        accent = labels.get("accent") or "?"
        model = v.get("model") or "?"
        category = v.get("category") or "?"
        name = v.get("name") or "?"
        vid = v.get("voice_id") or "?"
        print(f"    {vid}  {name:18.18}  {lang:10.10}  {gender:6.6}  {accent:10.10}  {model:14.14}  {category}")

    for title, url in endpoints:
        try:
            r = requests.get(url, headers=headers, timeout=30)
        except Exception as exc:
            print(f"\n  {title}: request failed — {exc}")
            continue
        if r.status_code != 200:
            print(f"\n  {title}: HTTP {r.status_code} — {r.text[:120]}")
            continue
        body = r.json()
        items = body.get("data") or []
        print(f"\n  {title} ({len(items)}):")
        if not items:
            print("    (none)")
            continue
        print(f"    {'voice_id':36}  {'name':18}  {'language':10}  {'gender':6}  {'accent':10}  {'model':14}  category")
        print(f"    {'-' * 36}  {'-' * 18}  {'-' * 10}  {'-' * 6}  {'-' * 10}  {'-' * 14}  {'-' * 8}")
        for v in items:
            _print_voice_row(v)


def main():
    parser = argparse.ArgumentParser(
        description="Verticals v3 — AI-Native Vertical Video Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Docs: https://github.com/rushindrasinha/verticals\n"
               "Product: https://verticals.gg",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="cmd")

    # Shared niche/provider args
    niche_help = f"Content niche ({', '.join(list_niches()[:8])}...)"

    # draft
    p_draft = sub.add_parser("draft", help="Generate script + metadata")
    p_draft.add_argument("--topic", "--news", dest="news", required=False, help="Topic/news headline")
    p_draft.add_argument("--context", default="", help="Channel context")
    p_draft.add_argument("--niche", default="general", help=niche_help)
    p_draft.add_argument("--platform", default="shorts", choices=["shorts", "reels", "tiktok", "all"])
    p_draft.add_argument("--provider", default=None, help="LLM: claude, gemini, openai, openrouter, ollama, litellm")
    p_draft.add_argument("--discover", action="store_true", help="Use topic engine")
    p_draft.add_argument("--auto-pick", action="store_true", help="Let LLM pick the best topic")
    p_draft.add_argument("--dry-run", action="store_true", help="Draft only")

    # produce
    p_produce = sub.add_parser("produce", help="Generate video from draft")
    p_produce.add_argument("--draft", required=True)
    p_produce.add_argument("--lang", default="en", choices=["en", "hi", "es", "pt", "de", "fr", "ja", "ko"])
    p_produce.add_argument("--voice", default=None, help="TTS: edge, elevenlabs, 60db, say")
    p_produce.add_argument("--script", default=None, help="Override script text")
    p_produce.add_argument("--force", action="store_true", help="Redo all stages")

    # upload
    p_upload = sub.add_parser("upload", help="Upload to YouTube")
    p_upload.add_argument("--draft", required=True)
    p_upload.add_argument("--lang", default="en", choices=["en", "hi", "es", "pt", "de", "fr", "ja", "ko"])
    p_upload.add_argument("--force", action="store_true", help="Re-upload even if done")

    # run (full pipeline)
    p_run = sub.add_parser("run", help="Full pipeline: draft -> produce -> upload")
    p_run.add_argument("--topic", "--news", dest="news", required=False, help="Topic/news headline")
    p_run.add_argument("--niche", default="general", help=niche_help)
    p_run.add_argument("--platform", default="shorts", choices=["shorts", "reels", "tiktok", "all"])
    p_run.add_argument("--provider", default=None, help="LLM: claude, gemini, openai, openrouter, ollama, litellm")
    p_run.add_argument("--voice", default=None, help="TTS: edge, elevenlabs, 60db, say")
    p_run.add_argument("--lang", default="en", choices=["en", "hi", "es", "pt", "de", "fr", "ja", "ko"])
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--context", default="")
    p_run.add_argument("--discover", action="store_true")
    p_run.add_argument("--auto-pick", action="store_true")

    # topics
    p_topics = sub.add_parser("topics", help="Discover trending topics")
    p_topics.add_argument("--niche", default="general", help=niche_help)
    p_topics.add_argument("--limit", type=int, default=15, help="Max topics to show")

    # niches
    sub.add_parser("niches", help="List available niche profiles")

    # voices
    p_voices = sub.add_parser("voices", help="List TTS voices (currently: 60db)")
    p_voices.add_argument("--provider", default="60db", help="TTS provider (only '60db' supported)")

    args = parser.parse_args()

    if args.verbose:
        set_verbose(True)

    if not args.cmd:
        parser.print_help()
        return

    # Handle utility commands that don't need first-run setup
    if args.cmd == "niches":
        cmd_niches(args)
        return
    if args.cmd == "voices":
        cmd_voices(args)
        return

    maybe_run_setup(args)

    # Handle --discover flag for draft/run
    if args.cmd in ("draft", "run") and getattr(args, "discover", False):
        from .topics import TopicEngine
        niche = getattr(args, "niche", "general") or "general"
        engine = TopicEngine(niche=niche)
        candidates = engine.discover(limit=15)
        if not candidates:
            print("  No trending topics found. Use --topic instead.")
            sys.exit(1)

        if getattr(args, "auto_pick", False):
            args.news = engine.auto_pick(candidates)
            print(f"  Auto-picked: {args.news}")
        else:
            print("\n  Trending topics:\n")
            for i, t in enumerate(candidates, 1):
                print(f"  {i:2d}. [{t.source}] {t.title}")
            choice = input("\n  Pick a number (or enter custom topic): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                args.news = candidates[int(choice) - 1].title
            else:
                args.news = choice
    elif args.cmd in ("draft", "run") and not getattr(args, "news", None):
        print("  Error: --topic or --discover required")
        sys.exit(1)

    if args.cmd == "draft":
        cmd_draft(args)
    elif args.cmd == "produce":
        cmd_produce(args)
    elif args.cmd == "upload":
        cmd_upload(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "topics":
        cmd_topics(args)


if __name__ == "__main__":
    main()
