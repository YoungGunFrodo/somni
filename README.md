# SOMNI

**Sleep-Optimized Memory for Neural Intelligence (SOMNI)**: a framework that gives local LLMs a human memory process.

A local AI that keeps learning from its own experience. During the day (**wake**) it answers
through a memory system that remembers what happened. While the computer is idle it **sleeps**:
it turns verified memories into new weights, practises on problems it can check (**dreaming**),
and tests itself before anything changes. You approve each new version at **morning review**, and
any change can be rolled back. Memories outlive models, so **brain transfer** can move everything
learned onto a newer base model.

It runs on stock open models (Qwen3) on a single consumer GPU: any NVIDIA or AMD card that PyTorch
supports, on Windows or Linux. A 1.7B model trains in about 8 GB of VRAM. Without a GPU, the toy
self-test still runs on any computer.

> **Status: research prototype.** The full pipeline runs end to end, on a toy model with a
> ground-truth benchmark and on real models (Qwen3 0.6B / 1.7B). Results below are single runs
> unless stated.

---

## The idea, and why it is built this way

Language models stop learning the day training ends. The two usual fixes each have a weakness:

- **Retrieval** (look things up in a database at answer time) learns instantly and never forgets.
  But the model never really *knows* anything new: every answer depends on the search finding
  the right text, and nothing improves the model's skills.
- **Fine-tuning** (change the weights) makes knowledge part of the model, but it is slow and
  needs care. Worst of all, new training overwrites old abilities: **catastrophic forgetting**.

People do both. We remember the day in short-term memory, and during sleep the brain replays
those memories and builds them into long-term memory, without erasing what we already knew. This
project copies that division of labour:

1. **Wake = memory first.** Everything goes into a memory store immediately, so it is usable
   right away through lookup.
2. **Sleep = consolidation.** Overnight, verified memories are trained into the weights, with
   safeguards against forgetting.
3. **Never trust a change blindly.** Every sleep ends with tests (gates). A night that makes the
   model worse at anything it could already do is rejected, and a human has the final say.

Two principles came out of building it and shaped most decisions:

- **Memories are data, never instructions.** Anything stored is checked (lint) for injected
  instructions, secrets and private data, both before storage and again before training. A
  memory cannot tell the model to do something.
- **Measure, don't assume.** Every fix in this repo came from a test that failed and an
  investigation into *why*. Gates are never loosened to make a run pass.

## How it works

```
 wake:   question ──► recall (memories + memory files) ──► answer ──► episode log
                                                                         │
 sleep:  memory formation ◄──────────────────────────────────────────────┘
           │   (episodes -> structured memories: subject / attribute / value, with source)
           ▼
         consolidation ── train a new frozen-base expert on: new memories, replay of old ones,
           │               rehearsal of the base model's own skills, lookup practice
           ▼
         dreaming ── practise only what failed; checked answers only
           ▼
         evaluation gates ──► dream report ──► morning review ──► adopt or roll back
```

### 1. Memory store (wake)

Every conversation, document and tool result is logged as an **episode**. During sleep, **memory
formation** turns episodes into structured memories:

- Each memory is a **subject**, an **attribute** and a **value** ("the port of billing-api is
  8443"). The key is built in code from the subject and attribute, never left to the model, so
  the same fact always lands in the same place.
- Each memory has a **source and a strength**: verified by a tool, confirmed by the user, or only
  inferred. Conflicts are settled by key and source strength, not by text similarity. A weaker
  claim never silently overwrites a stronger one; it is stored as *disputed* for you to resolve.
- Only verified or user-confirmed memories are ever trained into weights. The model's own guesses
  stay lookup-only.

### 2. Memory files: cold storage

A memory is a single short fact, and a document holds far more than its extracted facts. So the
full text of every document and tool output is also kept as a **memory file**, searchable next to
the memories. At answer time, recall returns the best matching memories *and* the passage around
the matched words in the best matching files.

This works like human memory: the weights hold the gist, the memory store holds the facts, and
the files are the notebook you can always go back to. Details that never made it into the weights,
or that faded from them, can still be found. The model's own past answers are never stored as
files, so an old mistake can't come back as evidence.

On the SQuAD benchmark (below), adding memory files raised lookup accuracy from **45.2% to 76.3%**
with no training at all. Lookup is cheap: about 3 ms to search 1,000 memories and 200 documents,
and 2-6% extra time per answer to read the results.

### 3. Consolidation: learning without forgetting

The central problem. On the first real night, plain LoRA fine-tuning on a batch of facts cut
coding ability from **95% to 10%** and general knowledge from **92.5% to 50%** (the gates rejected
it). The adapter had been trained only on short fact answers, so it changed the model's habits on
*every* prompt.

What fixed it, each measured in `scripts/forgetting_lab.py` and `scripts/forgetting_nights.py`:

- **Rehearsal.** Mixed into every night's training: the *base model's own answers* to general,
  reasoning, writing and coding prompts. Training is then pushed to keep those behaviours, not just
  to learn facts. The mix follows what *you* actually use (detected from the last 30 days of
  episodes): half is spread evenly across skills, half weighted by your usage. No settings needed,
  and no skill is ever dropped.
- **A new expert per night, older ones frozen.** Each sleep adds a small low-rank expert to every
  layer. The base model and all previous nights' experts stay frozen, so a later night cannot
  overwrite an earlier one. Over three nights this kept earlier facts at 95-98%, where one
  retrained adapter fell to 80%.
- **Combine practice.** Facts used inside small tasks (a port number used in code), so new
  knowledge connects to existing skills instead of sitting in isolation.
- **Varied phrasings.** Each fact is trained under many phrasings, including ones the teacher model
  did not write, so recall doesn't depend on asking the question one exact way.
- **Lookup practice.** Some training questions include exactly what recall would return (relevant
  memories, files *and* unrelated ones), and the answer cites the memory it used. This keeps reading
  memory a trained skill. Without it, the trained model started ignoring its lookup results (see
  SQuAD below).
- **Routing.** Each night's expert also learns to tell its own questions apart from everything
  else, and switches on only for those. Other questions get exactly the base model, so a big
  night can't shift general habits (SQuAD below: general knowledge 75% without routing, 90% with,
  base 92.5%).

### 4. Dreaming

After training, the model practises on questions about what it just learned and about older
protected memories, never on the gate's test questions. Only answers that fail are practised again,
and only answers that can be checked count. Practising on material already mastered was found to
erode other knowledge, so dreaming targets weaknesses only.

### 5. Gates and morning review

A candidate checkpoint must pass paired, bootstrapped tests before adoption:

| Gate | Question it answers |
|---|---|
| learning | Did it learn the new memories? |
| learning-outside | ...even when asked in phrasings never used in training? |
| retention / retention-floor | Did it keep old memories, tonight and against its best ever? |
| general, drift-vs-base | Did general knowledge stay at the base model's level? |
| skill: reasoning, coding | Do reasoning and coding still work? (coding runs real unit tests) |
| stale beliefs, disputed | Does it drop outdated facts and flag uncertain ones? |
| calibration | Does it still say "I'm not sure" when it doesn't know? |

If a night fails only because of a few memories, a **second take** retrains without them, so the
good part of the night is kept. You review a plain-language **dream report** and approve, reject
or roll back.

### 6. Forgetting, on purpose

Memories you use stay sharp in the weights. Unused ones are first **condensed** (a light,
low-detail trace, lumped by subject) and **fade** from the weights only after months. Nothing ever
leaves the memory store or memory files, so a faded memory can always be looked up.

### 7. Running on one consumer GPU

- **Turn-taking:** the teacher (a llama-server process) and the student (PyTorch) never hold the
  GPU at the same time. Each is paused while the other works.
- **Resource governor:** sleep reads free RAM and VRAM, picks batch sizes to fit, pauses when the
  PC is busy, shrinks batches when memory runs low (results identical), and saves progress so an
  interrupted night resumes.

---

## Comparison with Google's "sleep" framework

Google Research's *Language Models Need Sleep* (Behrouz, Hashemi, Javanmard, Mirrokni, 2026)
inspired SOMNI. Both add new low-rank experts during sleep, distill new knowledge into
them, and use dreaming. The differences:

| | Google Sleep | SOMNI |
|---|---|---|
| Model | Custom multi-rate MoE ("Hope") architecture | Stock open models (Qwen3), single consumer GPU |
| Short-term memory | Fast temporary parameters, pruned after sleep | An explicit memory store + memory files, kept forever |
| What gets learned | Everything in short-term memory | Only verified / user-confirmed memories |
| Conflicts | Implicit in training | Keyed memories, source strength, disputes for the user |
| Dream acceptance | Automatic, if performance improves | Gates on many skills, then human morning review |
| Forgetting | Not reported over many cycles | Protected → condensed → faded, measured; lookup never forgets |
| Answering | From weights | From weights *and* memory lookup |

### SQuAD knowledge incorporation (the benchmark Google and SEAL use)

The model reads 200 Wikipedia passages, then must answer 1,014 questions about them **without**
the passage in front of it.

| Method | Model | Base | After | Gain |
|---|---|---|---|---|
| SEAL (Zweiger et al., 2025) | Qwen2.5-7B | 31.9 | 43.2 | +11.3 |
| Google Sleep | Qwen2.5-7B | 31.9 | 46.2 | +14.3 |
| **SOMNI**, weights only | Qwen3-1.7B | 10.7 | 26.3 | **+15.7** |

**Read this as "roughly on par", not "better":**
- Our model is about 4× smaller, and our grading is stricter: an answer must contain the exact
  answer text, while theirs uses an AI judge that also accepts correct rewordings.
- The bases differ. Measured as the share of the remaining gap closed, we close 17% and Google
  closes 21%.
- It is a single run.

**What SOMNI adds: learning without forgetting.** The same run, with the model's other skills
measured before and after:

| Qwen3-1.7B, 200 passages | SQuAD, no context | SQuAD + memory lookup | General knowledge | Reading |
|---|---|---|---|---|
| Base model | 10.7% | 76.3% | 92.5% | 97.5% |
| Trained, no routing | 26.7% | 76.2% | **75.0%** | 100% |
| **Trained, with routing** | **26.3%** | **75.3%** | **90.0%** | **97.5%** |

Without routing, the night's expert learned the facts but cost 17.5 points of general knowledge;
the general gate would have rejected it. **Routing** (WISE-style, per question) trains each
night's expert to switch itself on only for questions about what it learned. A question routed
off gets exactly the base model. In this run the expert switched on for 96.5% of SQuAD questions,
2.5% of general questions and 0% of reading questions, and general knowledge stayed within one
question of the base model.

Memory lookup is where a memory-store design differs most from weights-only methods:

| Setting (Qwen3-1.7B) | Score |
|---|---|
| Base model, no context | 10.7% |
| Memory lookup, memories only | 45.2% |
| Memory lookup + memory files | **76.3%** |
| Trained, no context | 26.3% |
| Trained + memory lookup | 75.3% |

An early version of training taught the model to ignore what lookup handed it (trained + lookup
43.6%). Lookup practice and a reading skill pack in rehearsal fixed this: the trained model now
reads its lookup results about as well as the base model.

**Tried, no gain at this scale (honest negatives):**
- *Implications* (SEAL/Google-style: the teacher rewrites each document as restatements and
  consequences, trained as extra examples) and *dreaming* (two rounds of self-quizzing on the
  night's memories, failures re-practised): +3 points on a 30-passage run, none on 200 passages.
  Likely cause: dream questions copy the memory's own wording ("What is the label of ...?"), so
  the model improved on its quizzes (83% -> 92%) but not on how people actually ask.
- *Subject tags* (each memory tagged field / topic / subject, Zettelkasten-style, so recall can
  find memories by tag combinations and link related ones): no change on SQuAD, because plain
  search already returns the answer for 94-98% of questions. Tags are aimed at large stores and
  reworded questions, which SQuAD does not test.

## Other results

**Forgetting lab** (Qwen3-1.7B, 200 new facts in one night):

| Recipe | New facts | General | Coding | Combine |
|---|---|---|---|---|
| Base model | 0% | 92.5% | 95% | 0% |
| Plain LoRA, real night 1 | - | 50.0% | 10% | - |
| Expert + rehearsal + combine practice | 100% | 97.5% | 100% | 88% |

**Full real night** (Qwen3-1.7B, 20 facts from conversations, the whole pipeline): learning
100%, learning-outside 96.9%, general 92.5 → 97.5, coding 95 → 100, adopted. On phrasings
written independently of the pipeline, every one of the 16 memories trained that night was
recalled (48/48).

**Toy benchmark** (numpy model with a simulated user and ground truth; shows pipeline behaviour,
not LLM performance): see [`docs/selftest-latest.md`](docs/selftest-latest.md).

## Quick start

Requirements: Python 3.10+. The self-test needs only numpy (and matplotlib for charts) and runs
on any computer.

```
python -m venv .venv
.venv\Scripts\pip install numpy matplotlib         # Linux/macOS: .venv/bin/pip
python -m sleepai.cli selftest                      # ~6 minutes, writes results/results.md
python -m unittest discover -s tests -t .           # unit tests
```

### Real models (GPU)

Install the PyTorch build for your GPU from [pytorch.org](https://pytorch.org/get-started/locally/)
(NVIDIA: CUDA; AMD: ROCm, or AMD's Windows wheels), then the rest:

```
python -m venv .venv-gpu
.venv-gpu\Scripts\pip install torch            # use the command pytorch.org gives for your GPU
.venv-gpu\Scripts\pip install numpy transformers peft accelerate huggingface_hub safetensors
.venv-gpu\Scripts\python scripts\milestone0.py --model Qwen/Qwen3-0.6B    # can the GPU teach a fact?
```

About 8 GB of VRAM is enough to train a 1.7B model; the resource governor sizes batches to the card
it finds and leaves room for other programs. Development and all results so far used an AMD
Radeon RX 6800 (16 GB) on Windows 11, with AMD's PyTorch wheels
(`pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ "torch[device-gfx1030]"`).
Other GPUs have not been tested yet; reports are welcome.

See [`docs/usage.md`](docs/usage.md) for chatting with your model, teaching it, running sleeps,
and morning review.

```
python -m sleepai.cli [--home DIR] chat | tell | sleep | review | status | selftest | doctor | daemon
```

## Project layout

```
sleepai/        wake, memory store + memory files, sleep stages, gates, review, brain transfer,
                models (toy + real), experts, skill packs, resource governor
scripts/        experiments: milestone 0, forgetting lab, multi-night, real night, SQuAD
tests/          unit tests (stdlib unittest)
docs/           design document, results, related work, usage
```

## Documents

- [`docs/design.md`](docs/design.md): design, design review and blind review
- [`docs/related-work.md`](docs/related-work.md): Google's results, similar projects, how we compare
- [`docs/usage.md`](docs/usage.md): user guide
- [`CLAUDE.md`](CLAUDE.md): working notes, every experiment and finding, next tasks

## License

MIT. See [`LICENSE`](LICENSE).
