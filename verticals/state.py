"""Draft JSON state machine for pipeline resume capability."""

import json
from datetime import datetime, timezone
from pathlib import Path

# Ordered pipeline stages
STAGES = [
    "research", "draft", "broll", "voiceover", "whisper",
    "captions", "music", "assemble", "thumbnail", "upload",
]


class PipelineState:
    """Tracks completion per stage in the draft JSON.

    Each stage records: status (done/failed), timestamp, artifact paths.
    Re-running `produce` skips completed stages automatically.
    """

    def __init__(self, draft: dict):
        self.draft = draft
        if "_pipeline_state" not in self.draft:
            self.draft["_pipeline_state"] = {}

    @property
    def state(self) -> dict:
        return self.draft["_pipeline_state"]

    def is_done(self, stage: str) -> bool:
        """Check if a stage completed successfully."""
        entry = self.state.get(stage, {})
        return entry.get("status") == "done"

    def is_failed(self, stage: str) -> bool:
        entry = self.state.get(stage, {})
        return entry.get("status") == "failed"

    def start_stage(
        self,
        stage: str,
        message: str = "",
        artifacts: dict | None = None,
        percent: int | None = None,
    ):
        """Mark a stage as actively running so external UIs can show progress."""
        self.state[stage] = {
            "status": "running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if message:
            self.state[stage]["message"] = message
        if artifacts:
            self.state[stage]["artifacts"] = artifacts
        if percent is not None:
            self.state[stage]["percent"] = self._clamp_percent(percent)
        self.add_event(stage, "running", message or f"{stage} started", percent)

    def complete_stage(self, stage: str, artifacts: dict | None = None, percent: int | None = None):
        """Mark a stage as completed with optional artifact metadata."""
        self.state[stage] = {
            "status": "done",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if artifacts:
            self.state[stage]["artifacts"] = artifacts
        if percent is not None:
            self.state[stage]["percent"] = self._clamp_percent(percent)
        self.add_event(stage, "done", f"{stage} completed", percent)

    def fail_stage(self, stage: str, error: str = "", percent: int | None = None):
        """Mark a stage as failed."""
        self.state[stage] = {
            "status": "failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }
        if percent is not None:
            self.state[stage]["percent"] = self._clamp_percent(percent)
        self.add_event(stage, "failed", error or f"{stage} failed", percent)

    def update_progress(
        self,
        stage: str,
        percent: int,
        message: str = "",
        artifacts: dict | None = None,
    ):
        """Update a running stage with a new percent and optional message."""
        entry = self.state.setdefault(stage, {})
        entry["status"] = entry.get("status", "running")
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        entry["percent"] = self._clamp_percent(percent)
        if message:
            entry["message"] = message
        if artifacts:
            entry.setdefault("artifacts", {}).update(artifacts)
        self.add_event(stage, entry["status"], message or f"{stage} progress", percent)

    def add_event(self, stage: str, status: str, message: str = "", percent: int | None = None):
        """Append a compact progress event to the draft JSON."""
        events = self.draft.setdefault("_production_events", [])
        pct = self._clamp_percent(percent) if percent is not None else self._current_percent()
        event = {
            "stage": stage,
            "status": status,
            "message": message,
            "percent": pct,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        events.append(event)
        del events[:-40]
        self.draft["_production_progress"] = event
        self.draft["_production_percent"] = pct

    def _current_percent(self) -> int:
        value = self.draft.get("_production_percent", 0)
        try:
            return self._clamp_percent(int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _clamp_percent(value: int | float) -> int:
        return max(0, min(100, int(round(value))))

    def get_artifact(self, stage: str, key: str, default=None):
        """Get an artifact value from a completed stage."""
        entry = self.state.get(stage, {})
        artifacts = entry.get("artifacts", {})
        return artifacts.get(key, default)

    def reset(self):
        """Clear all pipeline state (for --force)."""
        self.draft["_pipeline_state"] = {}

    def summary(self) -> str:
        """Human-readable status of all stages."""
        lines = []
        for stage in STAGES:
            entry = self.state.get(stage, {})
            status = entry.get("status", "pending")
            marker = {"done": "+", "failed": "!", "running": "~", "pending": " "}.get(status, "?")
            lines.append(f"  [{marker}] {stage}")
        return "\n".join(lines)

    def save(self, path: Path):
        """Write the draft (with embedded state) to disk."""
        path.write_text(json.dumps(self.draft, indent=2, ensure_ascii=False))
