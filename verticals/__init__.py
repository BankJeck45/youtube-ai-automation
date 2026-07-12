"""Verticals — AI-Native Content Engine for vertical video."""

__version__ = "3.0.0"


def _fix_legacy_ffmpeg_filter_arg(arg: str) -> str:
    """Normalize older final-render filters for stricter ffmpeg builds."""
    if "aloop=loop=-1:size=2e+09," in arg:
        arg = arg.replace("[2:a]aloop=loop=-1:size=2e+09,atrim=", "[2:a]atrim=")

    if "volume='if(" not in arg:
        return arg

    import re

    arg = arg.replace("volume='if(", "volume=if(").replace(")':eval=frame", "):eval=frame")
    arg = re.sub(r"between\(t,([0-9.]+),([0-9.]+)\)", r"between(t\\,\1\\,\2)", arg)
    arg = re.sub(r"\),\s*([0-9.]+),\s*([0-9.]+)\):eval=frame", r")\\,\1\\,\2):eval=frame", arg)
    return arg


def _install_ffmpeg_render_hotfix() -> None:
    """Patch config.run_cmd before submodules import it."""
    try:
        from . import config
    except Exception:
        return

    original_run_cmd = config.run_cmd
    if getattr(original_run_cmd, "_verticals_ffmpeg_render_hotfix", False):
        return

    def run_cmd_with_ffmpeg_hotfix(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] == "ffmpeg":
            cmd = [_fix_legacy_ffmpeg_filter_arg(part) if isinstance(part, str) else part for part in cmd]
        return original_run_cmd(cmd, *args, **kwargs)

    run_cmd_with_ffmpeg_hotfix._verticals_ffmpeg_render_hotfix = True
    config.run_cmd = run_cmd_with_ffmpeg_hotfix


_install_ffmpeg_render_hotfix()


def _install_tts_provider_hotfix() -> None:
    """Avoid macOS-only say TTS on Linux production hosts."""
    try:
        import shutil

        from . import tts
    except Exception:
        return

    original_get_tts_provider = tts.get_tts_provider
    original_generate_voiceover = tts.generate_voiceover
    original_generate_say = tts._generate_say

    if getattr(original_generate_voiceover, "_verticals_tts_say_hotfix", False):
        return

    def say_available() -> bool:
        return shutil.which("say") is not None

    def safe_get_tts_provider(name=None):
        requested = (name or "").strip().lower().replace("-", "_")
        provider = original_get_tts_provider(name)
        if (requested == "say" or provider == "say") and not say_available():
            return original_get_tts_provider("edge")
        return provider

    def safe_generate_say(script, out_dir):
        if say_available():
            return original_generate_say(script, out_dir)
        try:
            return tts._generate_edge_tts(script, out_dir, "en")
        except Exception as exc:
            raise RuntimeError(f"macOS say TTS is unavailable and Edge TTS fallback failed: {exc}") from exc

    def safe_generate_voiceover(script, out_dir, lang="en", provider=None, voice_config=None):
        requested = (provider or "").strip().lower().replace("-", "_")
        if requested == "say" and not say_available():
            provider = "edge"
        return original_generate_voiceover(script, out_dir, lang, provider, voice_config)

    safe_generate_voiceover._verticals_tts_say_hotfix = True
    tts.get_tts_provider = safe_get_tts_provider
    tts._generate_say = safe_generate_say
    tts.generate_voiceover = safe_generate_voiceover


_install_tts_provider_hotfix()
