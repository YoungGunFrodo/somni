"""Versioned checkpoints. Every sleep makes a candidate; only morning review (or an
autonomy level that allows it) adopts one. Rollback is one call."""
from __future__ import annotations

import json
import time
from pathlib import Path


class CheckpointRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = self.root / "registry.json"
        self.data = json.loads(self.file.read_text(encoding="utf-8")) if self.file.exists() else {
            "base": None, "active": None, "history": [], "candidates": {}, "next": 0}

    def _save(self) -> None:
        self.file.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def new_id(self) -> str:
        cid = f"ckpt-{self.data['next']:04d}"
        self.data["next"] += 1
        self._save()
        return cid

    def path(self, cid: str) -> Path:
        return self.root / cid

    def register_base(self, cid: str) -> None:
        self.data["base"] = cid
        self.data["active"] = cid
        self.data["history"].append({"id": cid, "adopted_at": time.time(), "note": "base"})
        self._save()

    def add_candidate(self, cid: str, meta: dict) -> None:
        self.data["candidates"][cid] = {"created_at": time.time(), **meta}
        self._save()

    def adopt(self, cid: str, note: str = "") -> None:
        if cid not in self.data["candidates"]:
            raise ValueError(f"{cid} is not a candidate")
        self.data["active"] = cid
        self.data["history"].append({"id": cid, "adopted_at": time.time(), "note": note})
        self.data["candidates"][cid]["status"] = "adopted"
        self._save()

    def reject(self, cid: str, reason: str = "") -> None:
        if cid in self.data["candidates"]:
            self.data["candidates"][cid].update(status="rejected", reason=reason)
            self._save()

    def rollback(self) -> str:
        """Return to the previously adopted checkpoint."""
        hist = self.data["history"]
        if len(hist) < 2:
            raise ValueError("nothing to roll back to")
        dropped = hist.pop()
        self.data["active"] = hist[-1]["id"]
        if dropped["id"] in self.data["candidates"]:
            self.data["candidates"][dropped["id"]]["status"] = "rolled-back"
        self._save()
        return self.data["active"]

    @property
    def active(self) -> str | None:
        return self.data["active"]

    @property
    def base(self) -> str | None:
        return self.data["base"]

    def adopted_count(self) -> int:
        return len(self.data["history"]) - 1
