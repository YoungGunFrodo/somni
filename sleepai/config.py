"""Configuration. JSON on disk, dataclasses in code."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class GateConfig:
    learning_min: float = 0.80          # probe pass rate on newly consolidated memories
    retention_tolerance: float = 0.03   # max allowed drop on replay probes (vs current checkpoint)
    retention_floor_tolerance: float | None = 0.10  # max CUMULATIVE drop of protected probes vs
                                        # the best they ever did in an adopted checkpoint
    general_tolerance: float = 0.03     # max allowed drop on base knowledge vs current checkpoint
    drift_tolerance: float = 0.08       # max CUMULATIVE drop on base knowledge vs the base model
    calibration_tolerance: float = 0.05
    bootstrap_samples: int = 2000
    min_probes_for_power: int = 30      # below this the dream report warns "underpowered"


@dataclass
class SleepConfig:
    idle_minutes: int = 30
    nightly_start_hour: int = 1
    nightly_end_hour: int = 7
    min_new_episodes: int = 5
    replay: bool = True
    replay_ratio: float = 1.0           # extra replay examples per new example (importance-sampled)
    replay_coverage: bool = True        # every protected memory gets one replay example each sleep
    base_replay_ratio: float = 1.0      # base-knowledge examples per training example
    rehearsal_min_share: float = 0.2    # rehearsal repeated up to this share when the pool is small
    paraphrases_per_memory: int = 6
    max_new_per_sleep: int = 16         # sleep budget; the rest waits for the next sleep
    rebuild_every: int = 10             # full rebuild from base every N sleeps
    rebuild_after_rejections: int = 2   # stuck detector: rebuild after N gate rejections in a row
    rebuild_epochs: int = 30            # a rebuild retrains everything from base: a bigger job
    dreaming: bool = True
    dream_rounds: int = 2
    repair: bool = True                 # dreaming re-practices protected memories that slipped
    repair_max: int = 64                # at most this many per sleep, most salient first
    epochs: int = 12
    # Calibration: "I'm not sure" examples per training example, about INVENTED subjects (so
    # they never touch a real key or a calibration probe). Without them the student's only way
    # to abstain is low confidence, which training on many facts erodes.
    abstain_ratio: float = 0.15
    learning_rate: float = 5.0          # toy trainer only; the PEFT trainer has its own
    # Forgetting policy (memory/salience.py): unused memories are first condensed (a light,
    # low-weight trace in the weights), and only fade after months without use. Every memory
    # stays in the memory store. The retention gate covers PROTECTED memories.
    forgetting: bool = True
    half_life_days: float = 30.0        # salience halves after this many days without use
    protect_min_salience: float = 1.5   # protected at or above this salience
    forget_below_salience: float = 0.1  # below this a memory fades (never used: ~100 days)
    grace_days: int = 7                 # new memories are protected this long regardless (14 made
                                        # every candidate guard 100+ memories at once and stalled)
    condensed_replay_ratio: float = 0.5 # condensed examples per new example, each sleep
    condensed_weight: float = 0.5       # training weight of a condensed example (less detail)
    condensed_rebuild_paraphrases: int = 3  # a rebuild relearns condensed memories (vs 6 for protected)
    # Second take (user's request: keep the good parts of a night instead of all-or-nothing).
    # If gates fail: learning -> retrain WITHOUT the memories that failed their probes (they stay
    # in the store and are retried next night); a skill gate -> retrain with 2x rehearsal. Then
    # evaluate again. Off by default (the toy self-test measures single takes).
    second_take: bool = False
    # Lookup practice: per new memory, this many training questions carry what wake's recall
    # actually retrieves (memories + memory files, distractors included), so the trained model
    # keeps reading its memory. Without it, SQuAD lookup fell from 75.0% to 43.6% after training.
    lookup_practice: int = 4   # candidates per memory; only base-verified ones are kept (about half)


@dataclass
class Config:
    home: str = "~/.sleep-ai"
    backend: str = "toy"                # toy | llamacpp (any OpenAI-compatible server, LM Studio too)
    trainer: str = "toy"                # toy | peft | experts (stacked experts, recommended for real models)
    expert_gate: bool = False           # experts: learned gate per expert (multi-night lab: no gain)
    llamacpp_url: str = "http://127.0.0.1:8080"   # student server; LM Studio: http://127.0.0.1:1234
    teacher_url: str = ""               # optional separate teacher server (OpenAI-compatible)
    student_model: str = ""             # model id sent to the server (LM Studio needs it; `lms ls`)
    teacher_model: str = ""             # server model id, or HF model for the in-process teacher
    wake: str = "hf"                    # hf (in-process, no server) | server (llama-server + GGUF)
    base_model: str = "Qwen/Qwen3-1.7B"
    # Skill packs the model must keep while learning (sleepai/skills.py): all protected; how much
    # each is rehearsed follows the user's own episodes, so there is nothing to set here.
    skills: list = field(default_factory=lambda: ["general", "reasoning", "writing", "coding", "reading"])
    autonomy_level: int = 0
    seed: int = 0
    sleep: SleepConfig = field(default_factory=SleepConfig)
    gates: GateConfig = field(default_factory=GateConfig)

    @property
    def home_path(self) -> Path:
        return Path(self.home).expanduser()

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str | None = None, **overrides) -> "Config":
        data: dict = {}
        if path and Path(path).exists():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.update(overrides)
        sleep = data.pop("sleep", {})
        gates = data.pop("gates", {})
        sleep = sleep if isinstance(sleep, SleepConfig) else SleepConfig(**sleep)
        gates = gates if isinstance(gates, GateConfig) else GateConfig(**gates)
        return cls(sleep=sleep, gates=gates, **data)
