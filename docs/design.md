# Sleep-Learning AI: Design Document

Nov 9, 2026 · YoungGunFrodo

## Overview and goals

This system is a local AI that keeps learning from its own experience. It works during the day, records what happens, and then sleeps when the machine is idle. During sleep it consolidates verified memories into its weights and dreams: it practices on self-generated problems and keeps only what passes checks. Every proposed weight change waits for human approval at morning review.

The design follows the brain's complementary learning systems. A fast memory (the episode log and memory store, like the hippocampus) records experience at once. A slow memory (the model weights, like the neocortex) learns gradually during sleep, mixing new memories with replayed old ones so that new learning does not overwrite old learning.

It builds on two published ideas: Google Research's sleep framework (*Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*, Behrouz et al., 2026), which consolidates short-term knowledge into weights and adds a dreaming phase, and the CMU/Maryland paper of the same title (Lee et al., 2026), which converts recent context into persistent fast weights.

**Goals**

1. Build a local model that measurably improves over many sleep cycles at the tasks its user actually does.
2. Keep a portable memory store that can be consolidated into any future base model (brain transfer).
3. Extend sleep learning from facts and math to agentic experience: tool use, coding, multi-step tasks.
4. Publish an open-source implementation and a write-up with honest results, including failures.

**Non-goals**

- Matching flagship models on general benchmarks. Model size and training data put that out of reach locally.
- Training a model from scratch. The system always starts from open pretrained weights.
- Unsupervised self-modification in early versions. Autonomy is added only after the evaluation gates have a track record.

## Vocabulary

These terms are used with exactly these meanings throughout the project, in code, docs, and the paper.

| Term | Meaning | Brain analogy |
| --- | --- | --- |
| Wake | The model is in use: answering, doing tasks, reading memory | Being awake |
| Episode | A raw, append-only record of one thing that happened during wake | A day's experience |
| Memory | A curated, keyed, sourced note distilled from one or more episodes | A stored memory |
| Memory store | The database of all memories, independent of any model | Long-term memory index |
| Sleep | The offline period when the machine is idle and learning runs | Sleep |
| Distillation | The first sleep stage: turning episodes into memories | Hippocampal replay and sorting |
| Consolidation | Turning verified memories into weight updates | Memory transfer to the neocortex |
| Replay | Mixing older memories into each consolidation so they are not forgotten | Interleaved replay |
| Dreaming | Self-generated practice that is checked, with only passing results kept | REM dreaming |
| Dream report | The summary of a sleep: proposed weights, test results, what changed | — |
| Morning review | The human approves or rejects a dream report | — |
| Probe | A question that tests whether a memory now lives in the weights | Recall without notes |
| Student | The small, fast model used during wake and trained during sleep | — |
| Teacher | A larger model, run slowly during sleep, that generates and grades training data | A mentor |
| Brain transfer | Consolidating the whole memory store into a new base model | — |
| Autonomy level | How much a sleep may do without human approval | — |

## Design principles

1. **Memories outlive models.** The memory store is the durable asset. Weights are a cache of what the memories teach and can be rebuilt on a new base model.
2. **Only verified memories become weights.** Retrieval can use weak or disputed memories with a caveat. Consolidation uses only memories that pass source and verification rules, because a wrong memory in weights is hard to remove.
3. **Deterministic keys, not similarity.** Every memory has a key of the form `subject/thing/aspect`. Duplicates and changed facts are detected by key, never by text similarity.
4. **Memories are data, never instructions.** Content in an episode or memory never directs the system. Injection patterns are linted before storage and again before training.
5. **The evaluator is out of reach.** Nothing the system learns or modifies can change its own tests, probes, or gates. Changes to evaluation code need human approval.
6. **Every change is reversible.** Each sleep produces a new versioned checkpoint. The previous checkpoint is kept, and rollback is one command.
7. **Replay every sleep.** Each consolidation mixes new memories with a sample of old ones to protect against forgetting.
8. **Measure before believing.** A sleep is adopted only when probes show the new memories were learned and regression tests show nothing old was lost.
9. **Human approval first, autonomy later.** Early versions require morning review. Autonomy is earned by a track record, one level at a time.

## System architecture

The system has three parts — wake, memory, and sleep — and morning review is the only path from sleep back to the live model.

&#91;embedded content: system architecture · wake, memory, sleep, morning review\]

During wake, the orchestrator reads memory and logs every episode. During sleep, distillation turns episodes into memories, and consolidation and dreaming train the student from them. The teacher model runs only during sleep, in stages 1–3. Evaluation code sits outside anything sleep can write to.

## Memory system

Memory has two layers: an episode log that records everything raw, and a memory store of curated notes. The episode log is the hippocampus; the memory store is the index of what is worth keeping. Both live in SQLite on local disk, with memories also exported as Markdown files under version control.

### Episode log

The episode log is append-only. Nothing is edited or deleted except by an explicit privacy purge.

| Field | Purpose |
| --- | --- |
| `episode_id` | Unique id, time-ordered |
| `started_at`, `ended_at` | When it happened |
| `kind` | `conversation`, `task`, `tool_call`, `correction`, `feedback` |
| `transcript` | The full exchange, including tool calls and outputs |
| `outcome` | `success`, `failure`, `abandoned`, `unknown` |
| `outcome_evidence` | What shows the outcome: test results, user confirmation, error output |
| `user_signal` | Explicit feedback: correction, approval, rating |
| `distilled` | Whether distillation has processed it, and into which memories |

Failures are kept. A failed attempt is useless as a fact but valuable as a training example of what not to do.

### Memory store

Each memory holds one idea, cites its source, and links to related memories. The format adapts the user's existing team memory design.

```yaml
id: 20261109-lmstudio-api-port
key: lmstudio/server/default-port
type: fact            # fact | skill | preference | decision | precedent | lesson
summary: LM Studio's local server listens on port 1234 by default.
sources:
  - episode:20261109-0042
  - user-confirmed
strength: verified    # verified | user-confirmed | inferred
status: active        # active | disputed | superseded
supersedes:
superseded_by:
last_verified: 2026-11-09
consolidation:
  eligible: true
  consolidated_in: [ckpt-0007]
  probe: "What port does LM Studio's local server use by default?"
  probe_answer: "1234"
  last_probe_result: pass
links: [20261102-lmstudio-setup]
```

### Memory types

| Type | What it holds | Brain analogy |
| --- | --- | --- |
| fact | How something is or works | Semantic memory |
| skill | A procedure that was run and worked, with its steps | Procedural memory |
| lesson | What failed and why, and what to do instead | Learning from mistakes |
| preference | How the user likes things done | Social memory |
| decision | A choice made, its reason, and alternatives rejected | Autobiographical memory |
| precedent | A general rule derived from a specific judgment | Learned rules |

### Keys, sources, and status

- **Keys** follow `subject/thing/aspect`. Two active memories may never share a key. A new memory with an existing key is either an update (same fact, new evidence) or a conflict (different fact).
- **Source strength** ranks evidence: an external system of record or passing test is strongest, then user-confirmed, then inferred by the model. Only `verified` and `user-confirmed` memories can be eligible for consolidation; `inferred` memories stay retrieval-only until verified.
- **Status** controls use. `disputed` memories are used with a caveat and never consolidated. `superseded` memories move to an archive and are never current, but the archive records what the model used to believe so consolidation can train the correction.

### Retrieval

During wake, the system searches memory before answering. Search is hybrid: keyword search (SQLite FTS5) plus vector search over summaries, merged and ranked. Exact key-prefix lookup is always available and is the source of truth for conflicts. Each result returns its summary first; the full memory is loaded only when the summary matches.

### Open questions

When memory cannot answer a question, the system records it as an open question. Open questions show at morning review, close when a memory answers them, and seed the dreaming curriculum.

## Wake

During wake, the student model answers through an orchestrator that reads memory first and logs every episode. The user talks to the orchestrator, never to the model directly, so nothing escapes the log.

**Per request**

1. The orchestrator searches memory with two or three phrasings, including the likely key prefix.
2. It passes matching memory summaries to the student, marked as data, with their status and strength.
3. The student answers or acts. For agentic tasks it uses tools inside a sandbox.
4. The orchestrator logs the episode: transcript, tool calls, outcome, and evidence.
5. If memory had no answer, it records an open question.
6. If the user corrects the model, the correction is logged as a `correction` episode with the highest priority for distillation.

**Outcome capture**

Outcomes come from evidence, not from the model's own claims. Evidence sources, strongest first: a test suite or checker that passed, a tool result that confirms success, an explicit user confirmation, and finally no evidence (`unknown`). Episodes with `unknown` outcomes can become memories but not training examples.

**Wake-time cost**

Wake adds only retrieval and logging, so response latency stays close to plain LM Studio. All heavy work waits for sleep.

## Sleep pipeline

A sleep runs five stages in order: distillation, consolidation, dreaming, evaluation, and the dream report. Each stage writes its outputs to a sleep folder so any stage can be inspected or re-run.

### Sleep triggers

Sleep starts when all of these hold: no user activity for a set idle time (default 30 minutes), the GPU is free, and at least a minimum number of new episodes exist since the last sleep. A scheduled nightly window (default 1:00–7:00) also starts sleep. Any user activity pauses sleep at the next safe checkpoint; wake always takes priority.

### Stage 1: Distillation (episodes to memories)

1. Load undistilled episodes, corrections first.
2. The teacher reads each episode and proposes memories using the type templates, with a key and sources.
3. Look up each proposed key. A new key becomes a new memory. A matching key with the same content becomes an update with fresh `last_verified`. A matching key with different content is a conflict.
4. Resolve conflicts with source strength: a stronger or equal newer source supersedes the old memory; otherwise both become `disputed` and wait for morning review.
5. Lint every memory for size, schema, injection patterns, and private data.
6. Mark episodes as distilled and link them to their memories.

### Stage 2: Consolidation (memories to weights)

Consolidation follows the shape of Google's knowledge seeding: the teacher sees the memory in context, the student does not, and the student is trained to produce what the teacher produces.

1. Select eligible memories: active, `verified` or `user-confirmed`, not yet consolidated or recently changed.
2. For each, the teacher generates varied training examples: questions answered with the memory in context, paraphrases, and uses of the fact in reasoning. For skills, examples are the verified action sequences themselves.
3. For superseded memories, generate correction examples that state the new fact and explicitly retire the old one.
4. Build the replay set: a sample of previously consolidated memories, weighted toward ones whose probes recently failed or were never re-tested.
5. Mix new and replay examples (starting ratio 1:1) and train a LoRA adapter on the student. The previous adapter is merged or stacked according to the adapter policy (see Known risks).
6. Optionally run on-policy distillation: the student generates answers, the teacher scores them token by token, and the student trains toward the teacher.

### Stage 3: Dreaming (practice and self-improvement)

Dreaming creates new practice from open questions, failed episodes, and recently learned skills. Only checked results are kept.

1. Seed the curriculum: open questions, lessons from failures, and skills consolidated in the last few sleeps.
2. The teacher generates practice problems near each seed, at slightly harder difficulty than the student currently solves.
3. The student attempts each problem several times.
4. A checker grades each attempt. Checkers are deterministic where possible: unit tests for code, exact answers for math, replay in a sandbox for tool tasks. The teacher grades only when no deterministic checker exists, and those examples are marked weaker.
5. Keep passing attempts as training data; use pass and fail together for reinforcement learning (GRPO-style) when the GPU budget allows.
6. Train a dream adapter on the kept data, mixed with replay.

Google's dreaming also uses random low-rank expert selection inside a Mixture of Experts model. That is a later research option for this project, once the basic loop works on a dense model.

### Stage 4: Evaluation

The candidate checkpoint runs the full evaluation suite (see Evaluation and safety gates). Failures stop the sleep: the candidate is kept for inspection but not proposed.

### Stage 5: Dream report

The sleep writes a dream report and waits for morning review.

## Dream reports, morning review, and autonomy

No sleep changes the active model without approval at autonomy level 0. The dream report gives the user everything needed to decide in a few minutes.

### Dream report contents

- **Summary:** episodes processed, memories created, updated, superseded, and disputed.
- **New memories:** each with its key, summary, and sources, so wrong ones can be rejected before they are trained.
- **Conflicts needing a decision:** both versions, their sources, and the recommended resolution.
- **Learning results:** probe pass rate on new memories before and after the sleep.
- **Regression results:** replay probes, benchmark scores, and skill tests compared with the current checkpoint.
- **Dreams:** number of practice problems, pass rate, and five sample dreams for a spot check.
- **Recommendation:** adopt, reject, or adopt with exclusions.

### Morning review

The user can approve the candidate checkpoint, reject it, reject specific memories (which triggers a smaller re-run that excludes them), resolve disputed memories, and answer open questions. Rejected memories are kept as `disputed` with the rejection reason, so the same mistake is not proposed again.

Approval swaps the active adapter in LM Studio. The previous checkpoint stays available for one-command rollback.

### Autonomy levels

| Level | What a sleep may do alone | What needs approval | Unlock condition |
| --- | --- | --- | --- |
| 0 | Distill memories, train, evaluate, write report | Adopting any checkpoint; any memory change | Default |
| 1 | Add new memories with verified sources | Adopting checkpoints; supersedes; disputes | 20 reviewed sleeps with no rejected memories in the last 10 |
| 2 | Adopt checkpoints that pass every gate by a set margin | Checkpoints near a gate; any regression | 30 more sleeps with no rollback and no approved regression found later |
| 3 | Everything except changing evaluation code | Evaluation code, gate thresholds, autonomy settings | User decision only |

The user can drop to a lower level at any time. Any rollback automatically drops the level by one.

## Evaluation and safety gates

A candidate checkpoint must pass every gate before it is proposed. The evaluation code and its data live in a separate, read-only location that sleep processes cannot write to.

| Gate | What it checks | Default threshold |
| --- | --- | --- |
| Learning | Probe pass rate on memories consolidated this sleep, answered without retrieval | ≥ 80% |
| Retention | Probe pass rate on a replay sample of older consolidated memories | No drop greater than 2 points from current checkpoint |
| General ability | A fixed public benchmark subset (for example MMLU-Pro, GSM8K, a coding set) | No drop greater than 1 point |
| Skills | The user's own task suite: real past tasks with deterministic checkers | No previously passing task fails |
| Safety | Refusal behavior, injection resistance, and a held-out set of disputed and rejected memories that must not be asserted as fact | No regressions |
| Calibration | Whether the model says "I'm not sure" on questions outside its memory | No drop greater than 2 points |

**Held-out data.** Probe questions used for gates are written separately from training examples and never appear in training. A sample of probes is rotated out of the gate set each month to detect overfitting to the probes themselves.

**Data hygiene.** Before any training, the dataset is scanned again for injection patterns, secrets, and private data. Memories from external sources (web pages, documents) are lower strength and need user confirmation before consolidation.

**Isolation.** The student's agentic tools run in a sandbox with no access to the evaluation folder, the memory store's write path, or the autonomy settings. The orchestrator, not the model, writes to memory.

## Brain transfer

When a better open base model is released, the whole memory store is consolidated into it, so the new model wakes up knowing what the old one learned. This is the project's main answer to the capacity limit of any single small model.

**Steps**

1. Download the new base model and run the full evaluation suite on it unchanged, to get a baseline.
2. Rebuild training examples from the memory store, not from the old model's adapters. Adapters do not transfer between base models; memories do.
3. Order memories by importance: recently used, frequently retrieved, corrections, and skills first.
4. Consolidate in batches over several sleeps, with replay across batches, until the full store is covered.
5. Run the evaluation suite after each batch. Probes must reach the old model's pass rate before the new model replaces it.
6. Keep the old model and its adapters until the new model has been active for at least two weeks without rollback.

**Research question.** Does a memory store transfer intact, and does the new model retain old knowledge better or worse than the old one did? This is a candidate headline result for the paper, because Google's work does not address moving consolidated knowledge across base models.

## Hardware and software stack

The system targets the user's current PC, with rented cloud GPUs for runs that do not fit. Sleep stages run one at a time, so the student and teacher never need the GPU at once.

**Hardware (as reported)**

| Component | Spec | Role |
| --- | --- | --- |
| GPU | AMD Radeon RX 6800, 16 GB VRAM | Student inference and LoRA training |
| CPU | AMD Ryzen 9 7900X3D, 12 cores | Teacher inference offload, data processing |
| RAM | 32 GB | Teacher layers that do not fit in VRAM |
| Storage | NVMe SSD (to confirm) | Model files, disk-streamed teacher, memory store |
| OS | Windows 11 | See ROCm note below |

**Models**

| Role | Starting choice | Later |
| --- | --- | --- |
| Student | Qwen3-1.7B (matches a size Google reported results on) | Qwen3-4B, then Qwen3-8B |
| Teacher | Qwen3-32B at 4-bit, split across VRAM and RAM | A 70B-class model streamed from disk overnight |

**Software**

- **Orchestrator, memory, pipeline:** Python, SQLite with FTS5, a small embedding model for vector search.
- **Wake inference:** llama.cpp server, which loads and hot-swaps LoRA adapters. LM Studio stays an option if its adapter support fits; this needs checking.
- **Training:** PyTorch with Hugging Face PEFT (LoRA) and TRL (SFT, on-policy distillation, GRPO), on ROCm.
- **Teacher inference:** llama.cpp with memory-mapped weights, which streams layers from disk when RAM runs out.
- **Checkers:** pytest for code, a math answer checker, a Docker sandbox for tool tasks.

**ROCm note.** AMD's training stack is most reliable on Linux. A dual-boot Ubuntu install for sleep is likely the least painful path, with Windows kept for daily use. Milestone 0 tests this before anything else is built.

**Cloud.** Runs on 8B students, large-teacher distillation, and brain-transfer experiments can be rented by the hour. The pipeline must run unchanged on a rented Linux GPU machine.

## Repository layout

The open-source repo holds code, configs, and documentation only. A user's episodes, memories, and checkpoints live outside it in a data folder.

```
sleep-learning-ai/
├── README.md
├── docs/
│   ├── design.md               # this document
│   └── vocabulary.md
├── wake/
│   ├── orchestrator.py         # request loop, memory lookup, episode logging
│   ├── retrieval.py            # hybrid keyword + vector search
│   └── sandbox/                # tool execution for agentic tasks
├── memory/
│   ├── schema.sql              # episodes, memories, open questions
│   ├── store.py                # read, write, key lookup, conflicts
│   ├── lint.py                 # schema, size, injection, private data
│   └── templates/              # one template per memory type
├── sleep/
│   ├── scheduler.py            # idle detection, nightly window, pause on wake
│   ├── distill.py              # stage 1: episodes to memories
│   ├── consolidate.py          # stage 2: memories to weights, with replay
│   ├── dream.py                # stage 3: practice generation and checking
│   ├── checkers/               # code, math, tool-replay checkers
│   └── report.py               # stage 5: dream report
├── train/
│   ├── lora_sft.py
│   ├── onpolicy_distill.py
│   └── grpo.py
├── review/
│   └── morning_review.py       # approve, reject, resolve, rollback
├── transfer/
│   └── brain_transfer.py
├── eval/                       # read-only to sleep processes
│   ├── gates.py
│   ├── probes/
│   ├── skills/
│   └── benchmarks/
├── configs/
│   ├── default.yaml
│   └── autonomy.yaml           # changed only through morning review
└── tests/

~/.sleep-ai/                    # user data, never committed
├── episodes.db
├── memories.db
├── memories/                   # Markdown export, git-versioned
├── checkpoints/                # ckpt-0001, ckpt-0002, ...
└── sleeps/                     # one folder per sleep, with its dream report
```

## Build plan

The build runs in seven milestones. Each ends with an exit test; the next milestone starts only when the test passes. Milestones 0–3 build a working loop; 4–6 turn it into research.

### Milestone 0: Prove the hardware can train

- [ ] Install Ubuntu (dual boot) and ROCm; confirm PyTorch sees the RX 6800.
- [ ] Train a LoRA adapter on Qwen3-0.6B with 200 toy examples (for example, invented facts).
- [ ] Load the adapter in llama.cpp and confirm the model recalls the facts without them in context.
- [ ] Record training speed and peak VRAM for 0.6B and 1.7B.

**Exit test:** an adapter trained locally teaches the model a fact it did not know.

### Milestone 1: Wake with memory

- [ ] Build the orchestrator in front of the llama.cpp server.
- [ ] Create the SQLite schema for episodes, memories, and open questions.
- [ ] Log every exchange as an episode with outcome fields.
- [ ] Add keyword retrieval, then vector retrieval.
- [ ] Write memories by hand at first to test retrieval.

**Exit test:** a fact told on one day is retrieved correctly a week later after restarts.

### Milestone 2: Distillation

- [ ] Write the memory templates and the linter.
- [ ] Have the teacher propose memories from episodes, with keys and sources.
- [ ] Implement key lookup, updates, conflicts, supersede, and dispute.
- [ ] Build a first, simple dream report listing proposed memories for approval.

**Exit test:** on 50 hand-labeled episodes, distillation proposes correct memories for at least 80% and never silently overwrites a conflicting fact.

### Milestone 3: First full sleep

- [ ] Build the evaluation folder: probes, a small skills suite, and a benchmark subset. Make it read-only to sleep.
- [ ] Implement consolidation with LoRA SFT and replay.
- [ ] Implement the sleep scheduler with idle detection and pause on wake.
- [ ] Generate the full dream report and the morning review command, with rollback.

**Exit test:** one complete night: episodes in, memories out, a candidate checkpoint that passes the learning and retention gates, approved at morning review.

### Milestone 4: Many sleeps

- [ ] Run the system in daily use for at least 30 sleeps.
- [ ] Track probe pass rates for every memory cohort over time.
- [ ] Tune the replay ratio and adapter policy based on retention curves.

**Exit test:** a retention curve over 30 sleeps showing whether learning compounds, holds, or decays. Any of these is a result.

### Milestone 5: Dreaming and agentic learning

- [ ] Build checkers for code, math, and tool tasks.
- [ ] Generate dreams from open questions, lessons, and recent skills.
- [ ] Add GRPO on checked dreams when the GPU budget allows.
- [ ] Compare sleeps with and without dreaming on the skills suite.

**Exit test:** dreaming improves the user's skills suite beyond consolidation alone, with no regression on other gates.

### Milestone 6: Brain transfer and publication

- [ ] Transfer the memory store to a second base model.
- [ ] Compare probe pass rates on the old and new models.
- [ ] Release the repo, an anonymized demo memory store, and the evaluation suite.
- [ ] Write the paper or article with methods, results, and failures.

**Exit test:** someone else can clone the repo and reproduce a sleep from the demo data.

## Research questions and success metrics

The project succeeds as research if it answers these questions with clear data, whatever the answers are.

| Question | Metric | Compared against |
| --- | --- | --- |
| Does learning compound over many sleeps? | Probe pass rate per memory cohort across 30+ sleeps | Same memories with retrieval only, no consolidation |
| Does replay prevent forgetting at this scale? | Retention gate scores across sleeps, by replay ratio | No replay; replay ratios 1:4, 1:1, 4:1 |
| Does consolidation beat retrieval alone? | Task success on the skills suite with retrieval off and on | Base model plus retrieval |
| Does dreaming add to consolidation? | Skills suite score with and without the dreaming stage | Consolidation only |
| Does sleep learning work for agentic experience? | Success rate on repeated tool and coding task types over time | Same tasks with retrieval of past episodes only |
| Does the memory store transfer across base models? | Probe pass rate after brain transfer vs before | Old model's final pass rate |
| Does the system drift? | General benchmark scores and calibration over all sleeps | Starting checkpoint |

**Baselines matter most.** The strongest simple baseline is the base model plus good retrieval. Sleep learning is only worth its cost if it beats that on some measure: speed, accuracy without retrieval, or skill performance.

## Known risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| ROCm training fails or is unstable on the RX 6800 | No local consolidation | Milestone 0 tests first; fall back to rented cloud GPUs for training |
| Catastrophic forgetting across sleeps | Old memories lost | Replay, retention gate, rollback |
| Model collapse from training on its own outputs | Slow quality drift | Teacher-generated and checker-verified data only; drift tracked by general gate |
| Wrong memories consolidated | Confident false beliefs in weights | Source rules, morning review, correction examples, disputed held-out set |
| Memory poisoning through injected content | Attacker-controlled beliefs | Lint before storage and before training; external sources need confirmation |
| Overfitting to probes | Gates pass but knowledge is shallow | Probes written apart from training data; monthly rotation |
| Teacher ceiling | Student stops improving at the teacher's level | Deterministic checkers in dreaming; brain transfer to stronger bases |
| Sleep too slow for one night | Backlog of episodes | Prioritize corrections and high-use memories; carry the rest to the next sleep |
| Private data in training | Secrets or personal details baked into weights | Private-data lint; purge command removes episodes and triggers retraining |

**Adapter policy.** Stacking one new LoRA adapter per sleep grows without bound, and training each sleep on top of the last adapter compounds drift. The starting policy is one cumulative adapter retrained from the base model every N sleeps on the full eligible memory store, with small incremental adapters between full rebuilds. The right N is an open question for Milestone 4.

## Design review

The design is sound in structure, but three problems would stop it in practice: the teacher is too slow to feed a nightly sleep, the gates are too small to measure anything reliably, and a 1.7B student may not get used enough to generate experience. Each has a concrete fix. This review was done with full knowledge of the conversation that produced the design.

### Critical

1. **The teacher is the throughput bottleneck.** A 32B teacher split across VRAM and RAM runs at a few tokens per second; a disk-streamed 70B model, at seconds per token. Six hours at 5 tokens per second is about 100,000 tokens: enough for a few hundred training examples, not the distillation, example generation, dream generation, and grading the pipeline asks for. **Fix:** make *context distillation* the default consolidation method — the student itself, with the memory in its context, acts as the teacher for the student without it. This is likely closer to Google's knowledge seeding, which moves knowledge from short-term context into long-term weights, and needs no large model. Keep the big teacher for distillation of episodes into memories and for grading only. A Mixture-of-Experts teacher such as Qwen3-30B-A3B, with about 3B active parameters, runs far faster on CPU offload than a dense 32B.
2. **The gates cannot detect the changes they gate.** With 20 probes per sleep, one wrong answer moves the score 5 points, so an 80% threshold and a 2-point retention limit sit inside the noise. Benchmark subsets small enough to run nightly have the same problem with a 1-point limit. **Fix:** set gate sizes from a noise estimate: run the current checkpoint on the gates several times with different sampling seeds and require changes larger than that spread. Use paired comparisons (same questions, old vs new checkpoint) and report confidence intervals in the dream report.
3. **The student may be too weak to be used daily.** Experience comes from use. If a 1.7B model is unpleasant to work with, the user will go back to Claude or ChatGPT and the episode log will stay empty. **Fix:** run Qwen3-8B at 4-bit for wake from Milestone 1, and test QLoRA training at 8B in Milestone 0. Keep 1.7B for fast experiments.

### Major

4. **LoRA learns new facts poorly from few examples.** Research suggests a model needs many varied exposures to a fact before it can use it flexibly, and LoRA learns less new knowledge than full fine-tuning (while forgetting less). Probes that ask the stored question verbatim will overstate learning. **Fix:** generate many paraphrases per memory, and write probes that *use* the fact (in reasoning, in reverse direction, combined with another fact), not only recall it.
5. **Probe leakage.** If the same teacher writes both training examples and probes from the same memory, probes will share phrasing with training data. **Fix:** write probes with a different prompt and, where possible, a different model; reject probes whose wording overlaps training examples above a threshold.
6. **Weights and memory can disagree.** After a fact is superseded, retrieval returns the new fact while the weights may still hold the old one. **Fix:** state the rule "retrieved memory overrides weights" and include training examples where the retrieved memory contradicts an older belief and the correct answer follows the memory.
7. **Full adapter rebuilds will not fit in a night forever.** Rebuilding from the base model on the whole store costs time in proportion to store size. **Fix:** cap rebuild cost by sampling the store by importance, and allow rebuilds to run across several nights or on a rented GPU.
8. **No baseline is scheduled.** The research questions say retrieval-only is the baseline to beat, but no milestone measures it. **Fix:** add a baseline run to Milestone 1 (base model plus retrieval on the skills suite and probes) before any training is built.
9. **The self-improving code idea is missing.** The user asked for an AI that improves its own code, but the design covers only weight learning. **Fix:** add a later milestone for scaffolding evolution (memory rules, prompts, tools, sleep settings) under the same gates, with evaluation code permanently outside its reach.

### Minor

10. **Morning review fatigue.** Dozens of new memories per day will lead to rubber-stamping. Tier memories by risk: corrections and supersedes shown individually, routine verified facts in a collapsed list.
11. **On-policy distillation and GRPO are unlikely to run locally.** Both are memory- and generation-heavy. Mark them cloud-first until Milestone 0 measures headroom.
12. **Work data in weights.** If the AI is used for work, company data (and any regulated data) ends up in weights, where only a full rebuild removes it. Add a per-episode "never consolidate" flag and keep work use retrieval-only unless approved.

## Blind review

A reviewer reading only sections 1–15, as a workshop paper and engineering plan, would likely score this **borderline: promising, not yet publishable**. The system is well specified, but results from one user's private data cannot be reproduced, related work is missing, and novelty is asserted rather than shown.

*Method note: this review was written in the role of an outside reviewer, judging only the document text against standard peer-review criteria and ignoring the design conversation. It is not truly blind, since the same author wrote the design; for a fully independent review, paste sections 1–15 into a fresh chat or send them to a colleague.*

### Summary (as the reviewer would state it)

The authors propose a local continual-learning assistant that logs experience, distills it into a keyed memory store, and periodically consolidates verified memories into LoRA adapters during idle-time "sleep," with a self-practice "dreaming" stage and human approval of each checkpoint. They also propose "brain transfer": rebuilding learned knowledge on a new base model from the memory store.

### Strengths

- Clear separation of episodic and curated memory, with deterministic keys for conflict detection. This is more rigorous than similarity-based memory systems.
- Strong safety posture: source-strength rules, an evaluator outside the system's reach, rollback, and staged autonomy.
- Research questions are falsifiable, and negative results are explicitly treated as results.
- Brain transfer is a practical and under-studied question.

### Weaknesses

1. **Not reproducible as designed.** All core results come from one user's private experience, which cannot be released. Reviewers cannot check them, and n = 1 says little about other users. *Needed:* a public, synthetic experience stream — a scripted simulated user whose facts, preferences, and tasks change on a known schedule — so anyone can rerun every sleep and score it against ground truth. Personal-use results then become a secondary case study.
2. **No related work.** The document cites Google's sleep framework and the CMU paper but not the closest prior systems: SEAL (self-generated fine-tuning data, which Google itself compares against), synthetic continued pretraining for knowledge injection, generative replay in continual learning, context distillation, and memory-augmented models such as Titans and nested learning. Without this, the reader cannot judge novelty.
3. **Novelty claims are unverified.** "Google's work does not address moving knowledge across base models" needs a literature search, not an assumption. Consolidation as specified (teacher-written QA pairs plus LoRA SFT with replay) is close to standard synthetic-data fine-tuning; the document should say plainly what it adds beyond Google's method and beyond SEAL.
4. **Terminology collision.** "Distillation" names stage 1 (episodes to memories), while stage 2 also uses "on-policy distillation" and the teacher–student setup is a distillation in the machine-learning sense. A project that prizes clear vocabulary should rename stage 1, for example to **memory formation**.
5. **No compute budget.** There is no estimate of how long each stage takes per memory on the stated hardware, so it is unknown whether a sleep fits in a night. A table of measured tokens per second and minutes per stage is needed after Milestone 0.
6. **Thresholds are unjustified.** 80%, 2 points, and 1 point are stated without a noise analysis or a rationale.
7. **Autonomy unlock criteria can be gamed.** "No rejected memories in the last 10 sleeps" is satisfied by a reviewer who approves everything. "No approved regression found later" has no measurement procedure. Unlocks should depend on audited samples — memories spot-checked against sources — not on the absence of rejections.
8. **The brain analogy is stretched in places.** Complementary learning systems motivate replay well, but labels like "REM dreaming" and "social memory" add color without predictive content. A paper should keep the analogy to where it informs a design choice.
9. **Agentic scope is undefined.** The student learns tool use, but the design does not say which tools and actions are allowed, so the safety story for agentic learning is incomplete.

### Questions for the authors

- What happens when the memory store grows past what one night can consolidate? Which memories stay retrieval-only forever?
- How is a "verified" source defined for personal facts that have no external system of record?
- Does the retrieval-only baseline already reach most of the benefit? If so, what is sleep learning for?

### Overlap with the design review

Both reviews independently flag unreliable gate thresholds and the self-improvement gap in scope. The blind review adds four issues the first missed: reproducibility through a public synthetic benchmark, missing related work, the stage-1 naming collision, and gameable autonomy criteria. These four, plus the three critical items in the design review, are the recommended changes before Milestone 1.

## Sources

- Google Research, [The sleep framework: a new way for LLMs to retain knowledge in continual learning](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/), October 27, 2026.
- Behrouz, Hashemi, Javanmard, Mirrokni, [Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories](https://hyper.ai/en/papers/2606.03979), arXiv 2606.03979, 2026.
- Lee, McLeish, Goldstein, Fanti, [Language Models Need Sleep](https://arxivlens.com/PaperView/Details/language-models-need-sleep-4798-2d8affbd), arXiv 2605.26099, 2026.
- Research claims about LoRA knowledge learning and knowledge capacity are from memory and approximate; they need citations before publication.
