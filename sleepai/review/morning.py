"""Morning review: the human decides what the sleep proposed.

Autonomy levels (see the design doc):
  0  approval needed to adopt any checkpoint                (default)
  1  as 0; reserved for auto-accepting verified memories once audited
  2  checkpoints that pass every gate by a clear margin are adopted automatically
  3  everything except evaluation code, gates, and autonomy settings
Any rollback drops the level by one. Raising the level is always a human decision.
"""
from __future__ import annotations


def pending(system) -> list[str]:
    return [cid for cid, meta in system.registry.data["candidates"].items()
            if meta.get("status") == "awaiting review"]


def approve(system, cid: str | None = None, note: str = "approved at morning review") -> str:
    cid = cid or (pending(system) or [None])[-1]
    if cid is None:
        raise ValueError("no candidate awaiting review")
    system.registry.adopt(cid, note)
    meta = system.registry.data["candidates"][cid]
    system.store.mark_consolidated(meta.get("memories", []), cid)
    update_reference(system, meta.get("probe_results"))
    # Older candidates waiting in the queue are now stale: they were trained from an older base.
    for other in pending(system):
        system.registry.reject(other, f"superseded by {cid}")
    return cid


def update_reference(system, probe_file) -> None:
    """Keep, per probe, the best result any ADOPTED checkpoint had (for the retention floor).
    Only morning review writes this file; the sleep pipeline cannot."""
    import json
    from pathlib import Path
    if not probe_file or not Path(probe_file).exists():
        return
    path = system.home / "retention_reference.json"
    ref = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for k, ok in json.loads(Path(probe_file).read_text(encoding="utf-8")).items():
        ref[k] = bool(ref.get(k)) or bool(ok)
    path.write_text(json.dumps(ref), encoding="utf-8")


def reject(system, cid: str | None = None, reason: str = "rejected at morning review") -> str:
    cid = cid or (pending(system) or [None])[-1]
    if cid is None:
        raise ValueError("no candidate awaiting review")
    system.registry.reject(cid, reason)
    return cid


def reject_memories(system, memory_ids: list[str], reason: str = "wrong") -> None:
    """Mark memories as wrong. They stay in the store as disputed (so the same claim is not
    proposed again), are never consolidated, and are tested by the disputed-not-asserted gate.
    Pending candidates that contain them are rejected; the next sleep re-runs without them."""
    for mid in memory_ids:
        system.store.update_fields(mid, status="disputed", never_consolidate=True,
                                   rejection_reason=f"rejected at review: {reason}")
    for cid in pending(system):
        if set(memory_ids) & set(system.registry.data["candidates"][cid].get("memories", [])):
            system.registry.reject(cid, "contains rejected memories")


def resolve_dispute(system, memory_id: str, accept: bool, reason: str = "") -> str | None:
    """Decide a disputed memory (a weaker claim that conflicts with an active one).

    accept=True : the disputed claim is right. It becomes the active memory and the old one is
                  superseded (archived), so the next sleep trains the correction.
    accept=False: the claim is wrong. It stays disputed with a review rejection, so the same
                  claim is not proposed again and the disputed-not-asserted gate tests it.
    Returns the id of the memory now active for the key (None if there is none)."""
    m = system.store.get(memory_id)
    if m is None or m.status != "disputed":
        raise ValueError(f"{memory_id} is not a disputed memory")
    if not accept:
        system.store.update_fields(memory_id, never_consolidate=True,
                                   rejection_reason=f"rejected at review: {reason or 'dispute rejected'}")
        active = system.store.active_by_key(m.key)
        return active.id if active else None
    old = system.store.active_by_key(m.key)
    if old is not None:
        system.store.update_fields(old.id, status="superseded", superseded_by=memory_id)
    system.store.update_fields(memory_id, status="active", supersedes=old.id if old else "",
                               rejection_reason="", sources=m.sources + ["resolved at review"])
    return memory_id


def rollback(system) -> str:
    dropped = system.registry.data["history"][-1]["id"]
    active = system.registry.rollback()
    for mid in system.registry.data["candidates"].get(dropped, {}).get("memories", []):
        m = system.store.get(mid)
        if m and dropped in m.consolidated_in:
            system.store.update_fields(mid, consolidated_in=[c for c in m.consolidated_in if c != dropped])
    if system.autonomy > 0:
        system.set_autonomy(system.autonomy - 1, "automatic drop after rollback")
    return active
