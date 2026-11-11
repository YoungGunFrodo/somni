"""The memory system: an append-only episode log plus a curated memory store.

Episode log  = the hippocampus: everything that happened, raw.
Memory store = curated, keyed, sourced memories that outlive any model.

Rules enforced here:
  * Two ACTIVE memories never share a key (partial unique index).
  * Superseded memories are kept (archive) so consolidation can train corrections.
  * Search is full-text (FTS5) over key, summary and value, plus exact key lookup.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

STRENGTH_RANK = {"verified": 3, "user-confirmed": 2, "inferred": 1}

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    day INTEGER NOT NULL DEFAULT 0,
    kind TEXT NOT NULL,                 -- conversation | task | tool_call | correction | feedback
    transcript TEXT NOT NULL,           -- JSON list of {role, content}
    outcome TEXT NOT NULL DEFAULT 'unknown',   -- success | failure | abandoned | unknown
    outcome_evidence TEXT DEFAULT '',
    user_signal TEXT DEFAULT '',
    never_consolidate INTEGER NOT NULL DEFAULT 0,
    formed INTEGER NOT NULL DEFAULT 0,  -- processed by memory formation?
    formed_memories TEXT DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    type TEXT NOT NULL,
    summary TEXT NOT NULL,
    value TEXT DEFAULT '',
    body TEXT DEFAULT '',
    sources TEXT NOT NULL,              -- JSON list
    strength TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    supersedes TEXT DEFAULT '',
    superseded_by TEXT DEFAULT '',
    created_at REAL NOT NULL,
    created_day INTEGER NOT NULL DEFAULT 0,
    last_verified_day INTEGER NOT NULL DEFAULT 0,
    never_consolidate INTEGER NOT NULL DEFAULT 0,
    consolidated_in TEXT DEFAULT '[]',  -- JSON list of checkpoint ids
    last_probe_result TEXT DEFAULT '',
    retrieval_count INTEGER NOT NULL DEFAULT 0,  -- uses: answers that actually used it
    rejection_reason TEXT DEFAULT '',
    last_used_day INTEGER NOT NULL DEFAULT -1   -- -1 = never used (salience uses created_day)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_per_key ON memories(key) WHERE status = 'active';
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(id UNINDEXED, key, summary, value);
-- Memory files (cold storage, user's idea): the FULL text of documents and tool outputs, searched
-- next to the memories, so details that never reached (or faded from) the weights stay findable.
CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(episode_id UNINDEXED, title, text);
CREATE TABLE IF NOT EXISTS open_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    key_hint TEXT DEFAULT '',
    created_day INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open', -- open | resolved | dropped
    resolved_by TEXT DEFAULT ''
);
"""


@dataclass
class Memory:
    key: str
    type: str
    summary: str
    value: str = ""
    body: str = ""
    sources: list = field(default_factory=list)
    strength: str = "inferred"
    status: str = "active"
    id: str = ""
    supersedes: str = ""
    superseded_by: str = ""
    created_day: int = 0
    last_verified_day: int = 0
    never_consolidate: bool = False
    consolidated_in: list = field(default_factory=list)
    last_probe_result: str = ""
    retrieval_count: int = 0      # uses (see MemoryStore.touch), not search hits
    rejection_reason: str = ""
    last_used_day: int = -1

    @property
    def strength_rank(self) -> int:
        return STRENGTH_RANK.get(self.strength, 0)

    @property
    def eligible(self) -> bool:
        """Only active, well-sourced memories may become weights."""
        return (self.status == "active" and not self.never_consolidate
                and self.strength in ("verified", "user-confirmed"))


@dataclass
class Episode:
    id: int
    day: int
    kind: str
    transcript: list
    outcome: str
    outcome_evidence: str
    user_signal: str
    never_consolidate: bool


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


class MemoryStore:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._migrate()
        self.db.commit()

    def _migrate(self) -> None:
        """Bring databases created by older versions up to the current schema."""
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(memories)")}
        if "last_used_day" not in cols:
            self.db.execute("ALTER TABLE memories ADD COLUMN last_used_day INTEGER NOT NULL DEFAULT -1")
        # Memory files: index documents and tool outputs logged before the index existed.
        if not self.db.execute("SELECT 1 FROM files_fts LIMIT 1").fetchone():
            for r in self.db.execute("SELECT id, kind, transcript FROM episodes").fetchall():
                self._index_file(r["id"], r["kind"], json.loads(r["transcript"]))

    # ------------------------------------------------------------ episodes
    def log_episode(self, kind: str, transcript: list, outcome: str = "unknown",
                    evidence: str = "", user_signal: str = "", day: int = 0,
                    never_consolidate: bool = False) -> int:
        cur = self.db.execute(
            "INSERT INTO episodes(created_at, day, kind, transcript, outcome, outcome_evidence,"
            " user_signal, never_consolidate) VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), day, kind, json.dumps(transcript), outcome, evidence, user_signal,
             int(never_consolidate)))
        if not never_consolidate:
            self._index_file(cur.lastrowid, kind, transcript)
        self.db.commit()
        return cur.lastrowid

    # ------------------------------------------------------------ memory files (cold storage)
    FILE_MIN_CHARS = 200

    def _index_file(self, episode_id: int, kind: str, transcript: list) -> None:
        """Documents and tool outputs become memory files. The assistant's own answers never do:
        an old wrong answer must not come back as evidence."""
        parts = [str(t.get("content", "")) for t in transcript if t.get("role") == "tool"]
        text = "\n".join(parts).strip()
        if len(text) < self.FILE_MIN_CHARS:
            return
        m = re.match(r"\[document: ([^\]]+)\]", text)
        title = m.group(1) if m else f"{kind} {episode_id}"
        self.db.execute("INSERT INTO files_fts(episode_id, title, text) VALUES (?,?,?)", (episode_id, title, text))

    def search_files(self, query: str, k: int = 2, window: int = 700) -> list[dict]:
        """Best-matching memory files with the passage around the matched words (not the whole
        file, to keep the context small). Snippets that fail lint are dropped."""
        from .lint import scan_text
        terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2]
        if not terms:
            return []
        rows = self.db.execute(
            "SELECT episode_id, title, text FROM files_fts WHERE files_fts MATCH ? "
            "ORDER BY bm25(files_fts) LIMIT ?", (" OR ".join(f'"{t}"' for t in terms), k)).fetchall()
        out = []
        for r in rows:
            text, low = r["text"], r["text"].lower()
            hits = [low.find(t) for t in terms if low.find(t) >= 0]
            centre = sorted(hits)[len(hits) // 2] if hits else 0
            start = max(0, centre - window // 2)
            snippet = text[start:start + window]
            if not scan_text(snippet):
                out.append({"episode_id": r["episode_id"], "title": r["title"], "snippet": snippet})
        return out

    def unformed_episodes(self) -> list[Episode]:
        """Episodes not yet processed by memory formation, in the order they happened.
        (Time order matters: a later statement must win over an earlier one. Corrections get
        priority through higher training weight, not by being processed first.)"""
        rows = self.db.execute("SELECT * FROM episodes WHERE formed = 0 ORDER BY id").fetchall()
        return [Episode(r["id"], r["day"], r["kind"], json.loads(r["transcript"]), r["outcome"],
                        r["outcome_evidence"], r["user_signal"], bool(r["never_consolidate"]))
                for r in rows]

    def recent_texts(self, since_day: int, limit: int = 2000) -> list[str]:
        """Plain text of recent episodes (for measuring what the user actually does)."""
        rows = self.db.execute("SELECT transcript FROM episodes WHERE day >= ? ORDER BY id DESC LIMIT ?",
                               (since_day, limit)).fetchall()
        out = []
        for r in rows:
            try:
                out.append(" ".join(str(m.get("content", "")) for m in json.loads(r["transcript"])))
            except (ValueError, AttributeError):
                continue
        return out

    def count_unformed(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM episodes WHERE formed = 0").fetchone()[0]

    def mark_formed(self, episode_id: int, memory_ids: list[str]) -> None:
        self.db.execute("UPDATE episodes SET formed = 1, formed_memories = ? WHERE id = ?",
                        (json.dumps(memory_ids), episode_id))
        self.db.commit()

    # ------------------------------------------------------------ memories
    def _row_to_memory(self, r) -> Memory:
        return Memory(
            id=r["id"], key=r["key"], type=r["type"], summary=r["summary"], value=r["value"],
            body=r["body"], sources=json.loads(r["sources"]), strength=r["strength"],
            status=r["status"], supersedes=r["supersedes"], superseded_by=r["superseded_by"],
            created_day=r["created_day"], last_verified_day=r["last_verified_day"],
            never_consolidate=bool(r["never_consolidate"]),
            consolidated_in=json.loads(r["consolidated_in"]),
            last_probe_result=r["last_probe_result"], retrieval_count=r["retrieval_count"],
            rejection_reason=r["rejection_reason"], last_used_day=r["last_used_day"])

    def get(self, memory_id: str) -> Memory | None:
        r = self.db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_memory(r) if r else None

    def active_by_key(self, key: str) -> Memory | None:
        r = self.db.execute("SELECT * FROM memories WHERE key = ? AND status = 'active'",
                            (key,)).fetchone()
        return self._row_to_memory(r) if r else None

    def by_key_any(self, key: str) -> list[Memory]:
        rows = self.db.execute("SELECT * FROM memories WHERE key = ? ORDER BY created_at",
                               (key,)).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def keys(self, prefix: str = "") -> list[str]:
        rows = self.db.execute("SELECT DISTINCT key FROM memories WHERE status='active' "
                               "AND key LIKE ? ORDER BY key", (prefix + "%",)).fetchall()
        return [r[0] for r in rows]

    def insert(self, mem: Memory) -> Memory:
        if not mem.id:
            n = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            mem.id = f"m{n + 1:06d}-{_slug(mem.key)}"
        self.db.execute(
            "INSERT INTO memories(id, key, type, summary, value, body, sources, strength, status,"
            " supersedes, superseded_by, created_at, created_day, last_verified_day,"
            " never_consolidate, consolidated_in, rejection_reason, last_used_day)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mem.id, mem.key, mem.type, mem.summary, mem.value, mem.body,
             json.dumps(mem.sources), mem.strength, mem.status, mem.supersedes,
             mem.superseded_by, time.time(), mem.created_day, mem.last_verified_day,
             int(mem.never_consolidate), json.dumps(mem.consolidated_in), mem.rejection_reason,
             mem.last_used_day))
        self.db.execute("INSERT INTO memories_fts(id, key, summary, value) VALUES (?,?,?,?)",
                        (mem.id, mem.key.replace("/", " "), mem.summary, mem.value))
        self.db.commit()
        return mem

    def update_fields(self, memory_id: str, **fields) -> None:
        for k in ("sources", "consolidated_in"):
            if k in fields:
                fields[k] = json.dumps(fields[k])
        if "never_consolidate" in fields:
            fields["never_consolidate"] = int(fields["never_consolidate"])
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE memories SET {cols} WHERE id = ?", (*fields.values(), memory_id))
        self.db.commit()

    def supersede(self, old_id: str, new: Memory) -> Memory:
        """Retire the old memory (kept in the archive) and insert the new one."""
        self.update_fields(old_id, status="superseded", superseded_by="(pending)")
        new.supersedes = old_id
        new = self.insert(new)
        self.update_fields(old_id, superseded_by=new.id)
        return new

    def dispute(self, memory_id: str, reason: str = "") -> None:
        self.update_fields(memory_id, status="disputed", rejection_reason=reason)

    def all(self, status: str | None = None) -> list[Memory]:
        if status:
            rows = self.db.execute("SELECT * FROM memories WHERE status = ? ORDER BY created_at",
                                   (status,)).fetchall()
        else:
            rows = self.db.execute("SELECT * FROM memories ORDER BY created_at").fetchall()
        return [self._row_to_memory(r) for r in rows]

    def eligible_for_consolidation(self) -> list[Memory]:
        return [m for m in self.all("active") if m.eligible]

    def mark_consolidated(self, memory_ids: list[str], checkpoint_id: str) -> None:
        for mid in memory_ids:
            m = self.get(mid)
            if m and checkpoint_id not in m.consolidated_in:
                self.update_fields(mid, consolidated_in=m.consolidated_in + [checkpoint_id])

    # ------------------------------------------------------------ search
    def search(self, query: str, k: int = 5, include_disputed: bool = True) -> list[Memory]:
        """Full-text search. Returns active (and optionally disputed) memories, best first."""
        terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in terms)
        statuses = ("active", "disputed") if include_disputed else ("active",)
        rows = self.db.execute(
            "SELECT m.* FROM memories_fts f JOIN memories m ON m.id = f.id "
            f"WHERE memories_fts MATCH ? AND m.status IN ({','.join('?' * len(statuses))}) "
            "ORDER BY bm25(memories_fts) LIMIT ?", (fts_query, *statuses, k)).fetchall()
        # Search does NOT count as use: scoring, gates and multi-phrasing recall all search.
        # Only an answer that actually used a memory strengthens it (touch).
        return [self._row_to_memory(r) for r in rows]

    def touch(self, memory_ids, day: int) -> None:
        """A wake answer used these memories: one more use, last used today. Feeds salience."""
        for mid in memory_ids:
            self.db.execute("UPDATE memories SET retrieval_count = retrieval_count + 1, "
                            "last_used_day = MAX(last_used_day, ?) WHERE id = ?", (day, mid))
        self.db.commit()

    # ------------------------------------------------------------ open questions
    def add_open_question(self, question: str, key_hint: str = "", day: int = 0) -> int:
        existing = self.db.execute(
            "SELECT id FROM open_questions WHERE question = ? AND status = 'open'",
            (question,)).fetchone()
        if existing:
            return existing[0]
        cur = self.db.execute(
            "INSERT INTO open_questions(question, key_hint, created_day) VALUES (?,?,?)",
            (question, key_hint, day))
        self.db.commit()
        return cur.lastrowid

    def open_questions(self) -> list[dict]:
        rows = self.db.execute("SELECT * FROM open_questions WHERE status='open' ORDER BY id")
        return [dict(r) for r in rows.fetchall()]

    def resolve_open_questions_for_key(self, key: str, memory_id: str) -> int:
        cur = self.db.execute(
            "UPDATE open_questions SET status='resolved', resolved_by=? "
            "WHERE status='open' AND key_hint = ?", (memory_id, key))
        self.db.commit()
        return cur.rowcount

    # ------------------------------------------------------------ privacy
    def purge_episodes(self, episode_ids: list[int]) -> None:
        """Delete episodes (privacy). Memories formed from them are marked never_consolidate;
        the next adapter rebuild from base removes them from weights."""
        for eid in episode_ids:
            r = self.db.execute("SELECT formed_memories FROM episodes WHERE id=?", (eid,)).fetchone()
            if r:
                for mid in json.loads(r[0]):
                    self.update_fields(mid, never_consolidate=True, status="disputed",
                                       rejection_reason="source episode purged")
            self.db.execute("DELETE FROM episodes WHERE id=?", (eid,))
            self.db.execute("DELETE FROM files_fts WHERE episode_id=?", (eid,))
        self.db.commit()

    def stats(self) -> dict:
        q = lambda sql: self.db.execute(sql).fetchone()[0]
        return {
            "episodes": q("SELECT COUNT(*) FROM episodes"),
            "unformed_episodes": q("SELECT COUNT(*) FROM episodes WHERE formed=0"),
            "memories_active": q("SELECT COUNT(*) FROM memories WHERE status='active'"),
            "memories_disputed": q("SELECT COUNT(*) FROM memories WHERE status='disputed'"),
            "memories_superseded": q("SELECT COUNT(*) FROM memories WHERE status='superseded'"),
            "open_questions": q("SELECT COUNT(*) FROM open_questions WHERE status='open'"),
        }
