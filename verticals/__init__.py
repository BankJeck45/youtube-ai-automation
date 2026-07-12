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
