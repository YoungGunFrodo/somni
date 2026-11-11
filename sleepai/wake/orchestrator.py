"""Wake: every request goes through the orchestrator, which reads memory first and
logs every episode. The model never writes to memory itself."""
from __future__ import annotations

import re

from ..memory.store import MemoryStore
from ..models.base import ABSTAIN, Answer


_STOP = set("what which when where who whom whose how why does did was were the and for with that "
            "this from are is of in to a an by on at as it its be".split())


def match_strength(question: str, context) -> float:
    """Share of the question's content words found in the best retrieved memory or memory file
    (0..1). Used to route: a strong match is read by the base model with the experts off."""
    terms = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2 and w not in _STOP}
    if not terms or not context:
        return 0.0
    return max(sum(w in (c.summary + " " + c.key).lower() for w in terms) / len(terms) for c in context)


class Orchestrator:
    def __init__(self, store: MemoryStore, student, use_memory: bool = True):
        self.store = store
        self.student = student
        self.use_memory = use_memory

    def recall(self, question: str, k: int = 5, files: int = 2) -> list:
        """Search memory with a couple of phrasings, merged, best first; then the best-matching
        memory files (cold storage: the full source text around the matched words)."""
        phrasings = [question, " ".join(re.findall(r"[A-Za-z0-9]{3,}", question)[-4:])]
        seen, merged = set(), []
        for q in phrasings:
            for m in self.store.search(q, k=k):
                if m.id not in seen:
                    seen.add(m.id)
                    merged.append(m)
        merged = merged[:k]
        if files and hasattr(self.store, "search_files"):
            from ..memory.store import Memory
            for f in self.store.search_files(question, k=files):
                merged.append(Memory(key=f"file/{f['episode_id']}", type="file", summary=f["snippet"],
                                     strength="verified", status="active", id=f"file:{f['episode_id']}"))
        return merged

    def ask(self, question: str, day: int = 0, use_memory: bool | None = None) -> tuple[Answer, int]:
        use_memory = self.use_memory if use_memory is None else use_memory
        context = self.recall(question) if use_memory else []
        ans = self.student.answer(question, context)
        transcript = [{"role": "user", "content": question}]
        if context:
            transcript.append({"role": "memory", "content": ", ".join(m.id for m in context)})
        transcript.append({"role": "assistant", "content": ans.text})
        eid = self.store.log_episode("conversation", transcript, outcome="unknown", day=day)
        if ans.used_memory:   # only the memory the answer actually used is strengthened
            self.store.touch([ans.used_memory], day)
        if ans.value == ABSTAIN and not ans.used_memory:
            self.store.add_open_question(question, day=day)
        return ans, eid

    def tell(self, text: str, day: int = 0, never_consolidate: bool = False) -> int:
        """The user states something."""
        return self.store.log_episode("conversation", [{"role": "user", "content": text}],
                                      day=day, never_consolidate=never_consolidate)

    def tool_result(self, text: str, day: int = 0) -> int:
        """A tool or system of record reported something (strongest evidence)."""
        return self.store.log_episode("tool_call", [{"role": "tool", "content": text}],
                                      outcome="success", evidence="tool output", day=day)

    def feedback(self, episode_id: int, correct: bool, correction: str = "", day: int = 0) -> int:
        """Outcome evidence from the user. A correction is high priority for memory formation."""
        if correct:
            return self.store.log_episode("feedback", [{"role": "user", "content": f"(episode {episode_id} was correct)"}],
                                          outcome="success", evidence=f"user confirmed episode {episode_id}",
                                          user_signal="approve", day=day)
        return self.store.log_episode("correction", [{"role": "user", "content": correction}],
                                      outcome="failure", evidence=f"user corrected episode {episode_id}",
                                      user_signal="correction", day=day)
