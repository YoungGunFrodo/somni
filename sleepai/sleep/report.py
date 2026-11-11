"""Sleep stage 5, the dream report: everything morning review needs, in one page."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path


def _pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def write_report(folder: Path, data: dict, store) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    f, c, d, e = data["formation"], data["consolidation"], data.get("dreaming"), data["evaluation"]
    lines = [f"# Dream report — sleep {data['sleep_no']} (day {data['day']})", "",
             f"**Recommendation: {data['recommendation']}**  ",
             f"Candidate: `{data.get('candidate') or 'none'}` · status: {data['status']}", "",
             "## Memory formation",
             f"{f['episodes']} episodes → {len(f['created'])} new, {len(f['updated'])} updated, "
             f"{len(f['superseded'])} superseded, {len(f['disputed'])} disputed, "
             f"{len(f['rejected'])} rejected by lint, {f['resolved_questions']} open questions resolved.", ""]
    if f["created"]:
        lines += ["New memories (reject any that are wrong before approving):", ""]
        for mid in f["created"][:40]:
            m = store.get(mid)
            lines.append(f"- `{m.key}` — {m.summary} _(sources: {', '.join(m.sources)})_")
        if len(f["created"]) > 40:
            lines.append(f"- … and {len(f['created']) - 40} more (see report.json)")
        lines.append("")
    if f["superseded"] or f["disputed"]:
        lines += ["Changes needing attention:", ""]
        for old, new in f["superseded"]:
            o, n = store.get(old), store.get(new)
            lines.append(f"- SUPERSEDED `{o.key}`: {o.value!r} → {n.value!r} ({n.strength})")
        for mid in f["disputed"]:
            m = store.get(mid)
            lines.append(f"- DISPUTED `{m.key}` = {m.value!r}: {m.rejection_reason}")
        lines.append("")
    if f["rejected"]:
        lines += ["Rejected by lint: " + "; ".join(f"`{k}` ({', '.join(p)})" for k, p in f["rejected"][:10]), ""]
    lines += ["## Consolidation",
              f"{len(c['new_memories'])} memories, {c['new_examples']} new examples "
              f"({c['corrections']} corrections), {c['replay_examples']} replay examples, "
              f"{c['dropped_by_hygiene']} dropped by hygiene. Rebuild from base: {c['rebuild']}.", "",
              f"Forgetting policy: {c['protected']} older memories protected (full replay, retention "
              f"gate), {c['condensed']} condensed (light trace, {c['condensed_examples']} examples this "
              f"sleep), {c['faded']} faded from the weights. All stay in the memory store and are "
              "used by retrieval.", ""]
    if data.get("rehearsal_mix"):
        mix = ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(data["rehearsal_mix"].items(), key=lambda x: -x[1]))
        lines += [f"Rehearsal mix (follows your recent episodes; every skill keeps a floor): {mix}.", ""]
    t2 = data.get("second_take") or {}
    if t2.get("tried"):
        lines += ["## Second take",
                  f"The first take failed {', '.join(t2['failed_first'])}. Retrained once: left out "
                  f"{len(t2['left_out'])} memories that failed their probes (they stay in the memory store "
                  f"and are retried next sleep), rehearsal x{t2['rehearsal_boost']:g}. "
                  f"Second take {'passed' if t2['passed'] else 'failed'}.", ""]
    if d:
        lines += ["## Dreaming",
                  f"{d['seeds']} seeds, {d['attempts']} practice attempts, pass rate by round: "
                  f"{', '.join(_pct(r) for r in d['rounds'])}.", ""]
    lines += ["## Evaluation gates (no retrieval; paired vs current checkpoint)", "",
              "| Gate | Result | n | Current | Candidate | Diff (95% CI) |", "|---|---|---|---|---|---|"]
    for g in e["gates"]:
        ci = f"{g['diff']:+.3f} ({g['ci'][0]:+.3f}, {g['ci'][1]:+.3f})" if g["ci"] else "—"
        flag = (" ⚠ underpowered" if g["underpowered"] else "") + (f" ({g['note']})" if g["note"] else "")
        lines.append(f"| {g['name']} | {'PASS' if g['passed'] else 'FAIL'}{flag} | {g['n']} | "
                     f"{_pct(g['current'])} | {_pct(g['candidate'])} | {ci} |")
    cr = (e.get("info") or {}).get("condensed_recall")
    if cr:
        lines += ["", f"Condensed memories (informational, not a gate): recall without retrieval "
                      f"{_pct(cr['current'])} → {_pct(cr['candidate'])} (n={cr['n']})."]
    lines += ["", "Approve: `python -m sleepai.cli review approve`  ·  Reject: `... review reject`  ·  "
              "Reject specific memories: `... review reject-memories <id> ...`"]
    path = folder / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
