"""The toy brain: a small numpy model for self-tests and the synthetic benchmark.

It is NOT a language model. It is a capacity-limited associative learner that
shows the behaviors the sleep pipeline must handle, in seconds on a CPU:

  * it learns facts into its weights through gradient descent (consolidation),
  * it forgets older facts when trained only on new ones (interference),
  * replay protects old facts,
  * it can read retrieved memories from its context (wake-time memory use),
  * it abstains when unsure (calibration).

Questions are turned into hashed features: content words plus all pairs of
content words (so entity x attribute conjunctions can be memorized). The
feature space is deliberately small, so facts collide and interfere, as in a
small real model.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np

from .base import ABSTAIN, Answer, Probe, ProposedMemory, TrainingExample

# Filler words of the TRAINING templates. Probe-only filler words are left out
# on purpose so probes are genuinely held out (they add unfamiliar features).
STOPWORDS = set("""what is the of tell me do you know for remind which does have quick
question i need its it s a an whats to about please""".split())

TRAIN_TEMPLATES = [
    "What is the {attr} of {entity}?",
    "Tell me the {attr} of {entity}.",
    "{entity}: what's its {attr}?",
    "Do you know the {attr} for {entity}?",
    "Remind me of {entity}'s {attr}.",
    "Which {attr} does {entity} have?",
    "Quick question, {entity} {attr}?",
    "I need the {attr} of {entity}.",
]
PROBE_TEMPLATES = [               # held out: never used for training
    "What {attr} is set for {entity}?",
    "Can you recall {entity}'s {attr}?",
    "For {entity}, the {attr} is what?",
]
DREAM_TEMPLATES = TRAIN_TEMPLATES[::-1]
# Filler words used to vary training phrasings so the model learns to ignore unfamiliar
# words. Disjoint from probe wording, so probes stay held out. (Self-test finding: without
# this, facts were learned for the training phrasings only and held-out probes failed.)
FILLER = """hey so um okay well actually honestly basically again sorry real quick btw listen
right now today maybe just also urgent friendly reminder hmm yeah sure""".split()

VALUE = r"(?P<value>[a-z0-9:-]+(?:\.[a-z0-9]+)*)"
FACT_RE = re.compile(r"(?:no, )?the (?P<attr>[a-z ]+?) of (?P<entity>[a-z0-9 ]+?) is " + VALUE + r"\.?$", re.I)
TOOL_RE = re.compile(r"^\[tool:(?P<tool>[a-z]+)\] (?P<entity>[a-z0-9 ]+?) \| (?P<attr>[a-z ]+?) = " + VALUE + "$", re.I)
GUESS_RE = re.compile(r"i think the (?P<attr>[a-z ]+?) of (?P<entity>[a-z0-9 ]+?) might be " + VALUE, re.I)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def content_tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("'s", ""))
    return [w for w in words if w not in STOPWORDS]


def _h(s: str, d: int) -> tuple[int, float]:
    digest = hashlib.blake2b(s.encode(), digest_size=8).digest()
    n = int.from_bytes(digest, "little")
    return n % d, (1.0 if (n >> 63) & 1 else -1.0)


def featurize(text: str, d: int) -> np.ndarray:
    toks = sorted(set(content_tokens(text)))
    x = np.zeros(d, dtype=np.float32)
    for t in toks:
        i, s = _h("u:" + t, d)
        x[i] += s
    for a, b in itertools.combinations(toks, 2):
        i, s = _h(f"p:{a}|{b}", d)
        x[i] += s
    n = np.linalg.norm(x)
    return x / n if n else x


def parse_key(key: str) -> tuple[str, str]:
    entity, attr = key.split("/")[:2]
    return entity.replace("-", " "), attr.replace("-", " ")


class ToyStudent:
    """Linear softmax over answers, on hashed conjunction features."""

    def __init__(self, dim: int = 1024, abstain_below: float = 0.35, seed: int = 0):
        self.dim = dim
        self.abstain_below = abstain_below
        self.rng = np.random.default_rng(seed)
        self.vocab: list[str] = []
        self.index: dict[str, int] = {}
        self.W = np.zeros((0, dim), dtype=np.float32)
        self.b = np.zeros(0, dtype=np.float32)

    # -------------------------------------------------- vocabulary
    def _ensure(self, answers) -> None:
        new = [a for a in dict.fromkeys(answers) if a not in self.index]
        if not new:
            return
        for a in new:
            self.index[a] = len(self.vocab)
            self.vocab.append(a)
        self.W = np.vstack([self.W, np.zeros((len(new), self.dim), dtype=np.float32)])
        self.b = np.concatenate([self.b, np.zeros(len(new), dtype=np.float32)])

    # -------------------------------------------------- inference
    def predict_weights(self, question: str) -> tuple[str, float]:
        if not self.vocab:
            return ABSTAIN, 0.0
        logits = self.W @ featurize(question, self.dim) + self.b
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        i = int(p.argmax())
        return self.vocab[i], float(p[i])

    def read_context(self, question: str, context) -> object | None:
        """Wake-time memory use: pick the retrieved memory whose key words are all in the
        question. Stands in for a real model reading memories in its prompt."""
        q = set(content_tokens(question))
        best, best_len = None, 0
        # active memories first, so a disputed claim never shadows the current fact
        for m in sorted(context or [], key=lambda m: m.status != "active"):
            need = set(re.findall(r"[a-z0-9]+", m.key.replace("-", " ").replace("/", " ")))
            if need and need <= q and len(need) > best_len:
                best, best_len = m, len(need)
        return best

    def answer(self, question: str, context=None) -> Answer:
        m = self.read_context(question, context)
        if m is not None:  # rule: retrieved memory overrides weights
            caveat = m.status == "disputed"
            prefix = "I'm not certain, but " if caveat else ""
            return Answer(f"{prefix}{m.summary}", m.value, 0.99, used_memory=m.id, caveat=caveat)
        value, conf = self.predict_weights(question)
        if conf < self.abstain_below:
            return Answer("I'm not sure.", ABSTAIN, conf)
        return Answer(value, value, conf)

    # -------------------------------------------------- training
    def fit(self, examples: list[TrainingExample], epochs: int = 12, lr: float = 5.0,
            batch: int = 32, l2: float = 1e-4) -> float:
        if not examples:
            return 0.0
        self._ensure(e.answer for e in examples)
        X = np.stack([featurize(e.question, self.dim) for e in examples])
        y = np.array([self.index[e.answer] for e in examples])
        w = np.array([e.weight for e in examples], dtype=np.float32)
        loss = 0.0
        for _ in range(epochs):
            order = self.rng.permutation(len(examples))
            loss = 0.0
            for s in range(0, len(order), batch):
                idx = order[s:s + batch]
                xb, yb, wb = X[idx], y[idx], w[idx]
                logits = xb @ self.W.T + self.b
                logits -= logits.max(axis=1, keepdims=True)
                p = np.exp(logits)
                p /= p.sum(axis=1, keepdims=True)
                loss += float(-(wb * np.log(p[np.arange(len(yb)), yb] + 1e-9)).sum())
                g = p
                g[np.arange(len(yb)), yb] -= 1.0
                g *= wb[:, None] / len(yb)
                self.W -= lr * (g.T @ xb + l2 * self.W)
                self.b -= lr * g.sum(axis=0) * 0.1
            loss /= len(examples)
        return loss

    # -------------------------------------------------- checkpoints
    def save_checkpoint(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.savez(p / "weights.npz", W=self.W, b=self.b)
        (p / "vocab.json").write_text(json.dumps({"vocab": self.vocab, "dim": self.dim,
                                                  "abstain_below": self.abstain_below}), encoding="utf-8")

    def load_checkpoint(self, path: str) -> None:
        p = Path(path)
        meta = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
        arr = np.load(p / "weights.npz")
        self.dim, self.abstain_below = meta["dim"], meta["abstain_below"]
        self.vocab = meta["vocab"]
        self.index = {a: i for i, a in enumerate(self.vocab)}
        self.W, self.b = arr["W"].copy(), arr["b"].copy()


class ToyTrainer:
    """Consolidation for the toy brain: gradient descent on its weights."""

    def __init__(self, base_checkpoint: str, lr: float = 5.0):
        self.base_checkpoint = base_checkpoint
        self.lr = lr

    def train(self, student: ToyStudent, examples, epochs: int = 8, from_base: bool = False):
        if from_base:
            student.load_checkpoint(self.base_checkpoint)
        loss = student.fit(examples, epochs=epochs, lr=self.lr)
        return {"loss": round(loss, 4), "examples": len(examples), "from_base": from_base}


class ToyTeacher:
    """Rule-based teacher for the synthetic world. A real teacher is an LLM (models/llm.py)."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def form_memories(self, episode) -> list[ProposedMemory]:
        found = []
        for turn in episode.transcript:
            text = turn["content"].strip()
            role = turn["role"]
            if role == "tool":
                m = TOOL_RE.match(text)
                if m:
                    found.append(self._fact(m, "verified", [f"episode:{episode.id}", f"tool:{m['tool']}"], text))
            elif role == "user":
                for sentence in re.split(r"(?<=\.)\s+", text):
                    m = FACT_RE.search(sentence)
                    if m:
                        found.append(self._fact(m, "user-confirmed",
                                                [f"episode:{episode.id}", "user-confirmed"], text))
            elif role == "assistant":
                m = GUESS_RE.search(text)
                if m:
                    found.append(self._fact(m, "inferred", [f"episode:{episode.id}", "model-inferred"], text))
        return found

    @staticmethod
    def _fact(m, strength, sources, raw) -> ProposedMemory:
        entity, attr, value = m["entity"].strip().lower(), m["attr"].strip().lower(), m["value"].lower()
        return ProposedMemory(key=f"{slug(entity)}/{slug(attr)}", type="fact",
                              summary=f"The {attr} of {entity} is {value}.", value=value,
                              strength=strength, sources=sources, body=raw)

    def _vary(self, q: str) -> str:
        if self.rng.random() < 0.6:
            words = " ".join(self.rng.choice(FILLER, size=self.rng.integers(1, 3), replace=False))
            q = f"{words} {q}" if self.rng.random() < 0.5 else f"{q} {words}"
        return q

    def training_questions(self, memory, n: int) -> list[str]:
        entity, attr = parse_key(memory.key)
        picks = self.rng.choice(len(TRAIN_TEMPLATES), size=min(n, len(TRAIN_TEMPLATES)), replace=False)
        return [self._vary(TRAIN_TEMPLATES[i].format(entity=entity, attr=attr)) for i in picks]

    def probes(self, memory) -> list[Probe]:
        entity, attr = parse_key(memory.key)
        return [Probe(memory.id, t.format(entity=entity, attr=attr), memory.value) for t in PROBE_TEMPLATES]

    def dream_questions(self, memory, n: int) -> list[str]:
        entity, attr = parse_key(memory.key)
        return [self._vary(t.format(entity=entity, attr=attr)) for t in DREAM_TEMPLATES[:n]]

    def condensed_examples(self, memories, weight: float):
        """The toy brain answers one question with one value, so it cannot lump several facts
        into one example; its condensed trace is one plain question per memory at low weight."""
        return [TrainingExample(q, m.value, weight, "condensed", m.id)
                for m in memories for q in self.training_questions(m, 1)]

    _SYLLABLES = "bri dex fon gul hask jor kev lum mip nax plo quen rix sov tuv wex yar zel".split()

    def abstain_questions(self, memories, n: int) -> list[str]:
        """Questions shaped like real memories but about invented subjects, which have no
        answer. The invented name always carries a digit, so it can never be a real entity."""
        memories = list(memories)
        if not memories or n <= 0:
            return []
        out = []
        for _ in range(n):
            m = memories[self.rng.integers(len(memories))]
            entity, attr = parse_key(m.key)
            kind = entity.split()[0]
            name = "".join(self.rng.choice(self._SYLLABLES, size=2)) + str(self.rng.integers(10, 99))
            t = TRAIN_TEMPLATES[self.rng.integers(len(TRAIN_TEMPLATES))]
            out.append(self._vary(t.format(entity=f"{kind} {name}", attr=attr)))
        return out
