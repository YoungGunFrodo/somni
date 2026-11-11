"""Interfaces for the student, the teacher, and the trainer.

Two implementations exist:
  * toy  (sleepai.models.toy)  - a small numpy brain used for self-tests and the
    synthetic benchmark. Runs anywhere, in seconds.
  * real (sleepai.models.llm, sleepai.models.peft_trainer) - llama.cpp server for
    wake, an OpenAI-compatible teacher, and PEFT LoRA training on a GPU.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

ABSTAIN = "i'm not sure"


@dataclass
class Answer:
    text: str
    value: str                 # the extracted answer, normalized, or ABSTAIN
    confidence: float
    used_memory: str = ""      # memory id used from context, if any
    caveat: bool = False       # answered from a disputed memory


@dataclass
class TrainingExample:
    question: str
    answer: str
    weight: float = 1.0
    origin: str = "new"        # new | replay | correction | dream | abstain
    memory_id: str = ""
    context: list | None = None  # retrieved memories/files shown in the prompt (lookup practice)


@dataclass
class Probe:
    memory_id: str
    question: str
    answer: str
    kind: str = "recall"       # recall | use | reverse


@dataclass
class ProposedMemory:
    key: str
    type: str
    summary: str
    value: str
    strength: str
    sources: list = field(default_factory=list)
    body: str = ""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 .:-]", " ", text.lower())).strip(" .")


def answers_match(predicted: str, expected: str) -> bool:
    p, e = normalize(predicted), normalize(expected)
    return bool(e) and (p == e or re.search(rf"(^|\s){re.escape(e)}($|\s|\.)", p) is not None)


class Student(Protocol):
    def answer(self, question: str, context: list | None = None) -> Answer: ...
    def load_checkpoint(self, path: str) -> None: ...
    def save_checkpoint(self, path: str) -> None: ...


class Teacher(Protocol):
    def form_memories(self, episode) -> list[ProposedMemory]: ...
    def training_questions(self, memory, n: int) -> list[str]: ...
    def probes(self, memory) -> list[Probe]: ...
    def dream_questions(self, memory, n: int) -> list[str]: ...
    def abstain_questions(self, memories, n: int) -> list[str]: ...


class Trainer(Protocol):
    def train(self, student, examples: list[TrainingExample], epochs: int,
              from_base: bool = False) -> dict: ...
