"""Wires the parts together: memory store, checkpoints, student, teacher, trainer."""
from __future__ import annotations

import json
from pathlib import Path

from .checkpoints import CheckpointRegistry
from .config import Config
from .memory.store import MemoryStore
from .sleep.evaluation import ProbeCache
from .wake.orchestrator import Orchestrator


class System:
    def __init__(self, config: Config, teacher, trainer, student_factory,
                 base_probes=None, unknown_probes=None, base_replay=None):
        self.config = config
        home = config.home_path
        home.mkdir(parents=True, exist_ok=True)
        self.home = home
        self.store = MemoryStore(home / "memory.db")
        self.registry = CheckpointRegistry(home / "checkpoints")
        self.probe_cache = ProbeCache(home / "probes.json")
        self.sleeps_dir = home / "sleeps"
        self.sleeps_dir.mkdir(exist_ok=True)
        self.teacher, self.trainer, self.student_factory = teacher, trainer, student_factory
        self.base_probes = base_probes or []       # general-ability probes (dicts)
        self.unknown_probes = unknown_probes or [] # questions it should NOT know (calibration)
        self.base_replay = base_replay or []       # TrainingExamples protecting base knowledge
        self.autonomy_file = home / "autonomy.json"
        if not self.autonomy_file.exists():
            self.autonomy_file.write_text(json.dumps({"level": config.autonomy_level, "log": []}), encoding="utf-8")

    def load_student(self, cid: str):
        s = self.student_factory()
        s.load_checkpoint(str(self.registry.path(cid)))
        return s

    def orchestrator(self, use_memory: bool = True) -> Orchestrator:
        factory = getattr(self, "wake_factory", None) or self.student_factory
        s = factory()
        s.load_checkpoint(str(self.registry.path(self.registry.active)))
        return Orchestrator(self.store, s, use_memory)

    @property
    def autonomy(self) -> int:
        return json.loads(self.autonomy_file.read_text(encoding="utf-8"))["level"]

    def set_autonomy(self, level: int, reason: str) -> None:
        data = json.loads(self.autonomy_file.read_text(encoding="utf-8"))
        data["log"].append({"from": data["level"], "to": level, "reason": reason})
        data["level"] = max(0, min(3, level))
        self.autonomy_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_toy_system(config: Config, base_examples, base_probes, unknown_probes, dim: int = 2048):
    """A complete system on the toy brain. Trains the base checkpoint on first use."""
    from .models.toy import ToyStudent, ToyTeacher, ToyTrainer

    factory = lambda: ToyStudent(dim=dim, seed=config.seed)
    home = config.home_path
    base_path = home / "checkpoints" / "ckpt-0000"
    system = System(config, ToyTeacher(seed=config.seed), ToyTrainer(str(base_path), lr=config.sleep.learning_rate), factory,
                    base_probes, unknown_probes, base_examples)
    if system.registry.base is None:
        cid = system.registry.new_id()
        s = factory()
        s.fit(base_examples, epochs=30)
        s.save_checkpoint(str(system.registry.path(cid)))
        system.registry.register_base(cid)
    return system


def build_real_system(config: Config):
    """The system for your PC.
    Sleep: in-process student (Hugging Face base model + the checkpoint's LoRA adapter), PEFT
    trainer, and a teacher: a server (LM Studio / llama-server) when `teacher_url` is set,
    otherwise in-process (`teacher_model`, default the student's own base model).
    Wake: in-process too by default (`wake: "hf"`); `wake: "server"` uses llama-server at
    `llamacpp_url` with the approved adapter (GGUF)."""
    from . import evalsets
    from .models.hf import HFStudent, HFTeacher
    from .models.llm import LLMStudent, LLMTeacher
    from .models.peft_trainer import PeftTrainer

    if config.teacher_url:
        teacher = LLMTeacher(config.teacher_url, config.teacher_model)
    else:
        teacher = HFTeacher(config.teacher_model or config.base_model)
    if config.trainer == "experts":
        # Stacked experts (models/experts.py): each sleep adds a new expert, older ones frozen,
        # so a later night cannot overwrite an earlier one (multi-night lab: one retrained
        # adapter lost night-2 facts 97% -> 80%; stacked kept 97%). Rehearsal keeps old skills.
        # The learned gate is optional (expert_gate): it gave no gain in the lab. Wake must be
        # in-process ("hf"): llama-server cannot run these experts.
        from .models.experts import ExpertTrainer
        trainer = ExpertTrainer(config.base_model, str(config.home_path / "adapters"),
                                gated=config.expert_gate, route=config.expert_route)
    else:
        trainer = PeftTrainer(config.base_model, str(config.home_path / "adapters"))
    system = System(config, teacher, trainer, lambda: HFStudent(config.base_model),
                    base_probes=evalsets.general_probes(), unknown_probes=evalsets.unknown_probes())
    # Skill packs (config.skills): each enabled pack adds rehearsal and, if it has probes, a gate.
    from .skills import get
    packs = get(config.skills)
    system.coding_probes = evalsets.coding_probes() if "coding" in config.skills else []
    system.skill_probes = {p.name: (p.probes(), p.probe_mode) for p in packs
                           if p.probes and p.name not in ("general", "coding")}
    # Rehearsal: the base model's own answers for every enabled pack, replayed every sleep
    # (built on first use; one-time cost of a few minutes).
    from .rehearsal import build_rehearsal
    system.base_replay = build_rehearsal(config.home_path, config.base_model, list(config.skills))
    if config.wake == "server":
        system.wake_factory = lambda: LLMStudent(config.llamacpp_url, config.student_model)
    if system.registry.base is None:
        cid = system.registry.new_id()
        system.registry.path(cid).mkdir(parents=True, exist_ok=True)
        system.registry.register_base(cid)
    return system
