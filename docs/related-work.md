# Related work and how this project compares

Researched 2026-11-09 (session 4). Sources at the bottom. Numbers from secondary sources are marked.

## Google's sleep framework (Behrouz, Hashemi, Javanmard, Mirrokni; arXiv 2606.03979)

**What it is.** Wake stores new information in fast, temporary parameters; sleep (1) expands the
model with new low-rank Mixture-of-Experts experts in slow memory blocks, (2) "knowledge seeding":
distills the short-term memory into them with on-policy distillation plus "learning to imitate"
(RL with semantic/Levenshtein rewards), (3) prunes the temporary parameters; then dreaming:
random expert routing to generate synthetic data, gradient-based selection of dreams, SFT on the
best, kept only if downstream performance improves. Needs their MoE/"Hope" multi-rate
architecture; the paper's multi-rate memory has no public code (per a community reproduction).

**Headline result for facts** (SQuAD knowledge incorporation, setup of SEAL/Zweiger et al. 2025:
answer questions about passages WITHOUT the passage in context; accuracy %):

| Method | 1 passage | 200 passages (continued pretraining) |
|---|---|---|
| Base model | 31.9 | 31.9 |
| Fine-tuned, no dreaming | 33.4 | 32.0 |
| SEAL | 46.7 | 43.2 |
| Sleep (Transformer) | 48.1 | 44.3 |
| Sleep (Transformer + four-level memory) | 48.9 | 46.2 |
| Sleep without dreaming | 35.7 | 36.2 |

Other results: math (AIME 24 79.2%, AIME 25 69.0%, HMMT 25 46.1% for a sleep-enhanced model;
Qwen3-1.7B/8B); "near-zero drop" in continual translation (figures only); class-incremental
learning and BABILong (figures only). No numeric retention over many sleep cycles in the text we
could read; no external memory store, no human review; dream acceptance is automatic.

**How we compare today: we cannot yet.** Our only numbers are (a) the toy brain, which is not an
LLM and is not comparable, and (b) Milestone 0: Qwen3-0.6B learns 200 invented facts, 0% → 99.7%
recall on held-out phrasings. That is recall of short facts, a much easier task than SQuAD
passage QA, so it is NOT comparable to 46.2%.

**How to compare honestly.** Run the same SQuAD knowledge-incorporation benchmark (the SEAL setup,
which Google follows) through our pipeline: each passage becomes memories → consolidation (LoRA +
replay) → dreaming → evaluated without context. Report against base 31.9, SFT 32.0, SEAL 43.2,
Sleep 46.2. Plain fine-tuning there gets ~32-36, so the bar is high: what matters is how many
varied, checked training examples the teacher makes per passage (SEAL's self-edits and Google's
dreams both are exactly that). Our wins are likely elsewhere (below), and a negative result on
SQuAD is still a result.

**Differences that matter.**
- Google changes the architecture (MoE expansion, multi-rate memory); we use stock open models +
  LoRA, so it runs on any GGUF/HF model and on a 16 GB AMD card under Windows.
- Google's knowledge lives only in parameters; ours lives in a memory store first, and weights are
  a cache of it. That enables retrieval at wake (100% recall in every toy run), brain transfer to a
  new base model, auditing, and deletion.
- Google gates dreams automatically by downstream improvement; we gate every checkpoint with paired,
  bootstrapped tests (learning, retention, general ability, drift vs base, stale beliefs, disputed
  claims, calibration) and a human morning review, with rollback.

## Closest open-source projects (checked 2026-11-09)

| Project | Weights? | Memory store | Gates / review | Forgetting | Transfer | Evidence |
|---|---|---|---|---|---|---|
| **This project** | LoRA + replay + dreaming | keyed, sourced, conflicts by key, disputes | paired bootstrap gates + human morning review + rollback | protected → condensed → faded by salience | yes, from the store | toy benchmark with ground truth; M0 on real model |
| Hypnos (Erazor48) | LoRA nightly, 24-card replay | Q/A cards, dedup | pre-registered decision rules, no human review | none | no | LoCoMo, 1 conversation: RAG top-10 33.2 F1 beat LoRA 12.6-18.6; adapter added nothing significant over RAG |
| sleeping-llm (vbario) | MEMIT edits → LoRA fused at sleep | none (facts only in weights) | recall audit, perplexity rollback | re-inject facts below 0.5 recall; 50-edit cap | no | ≤60 synthetic facts, single runs |
| llm-sleep (everrete) | implements Google's paradigm on HF models | none | none | none | no | Qwen2.5-0.5B smoke test; CLINC150: sleep did not beat no-sleep |
| sleep (aydinnyunus) | **no** (external memory only) | 3 tiers, sources, supersede, validity windows | human promote/refuse | flags stale, never auto-drops | n/a | no accuracy benchmark |
| SCM (arXiv 2604.20943) | no (external) | memory graph | none | "algorithmic forgetting" | n/a | paper |

**Is the combination unique?** As far as these searches show, yes: no project found combines a
keyed, sourced memory store with conflict rules + weight consolidation with replay and dreaming +
statistical gates + human approval and rollback + salience-based condensing/fading + brain
transfer + a ground-truth synthetic benchmark. Each piece has precedent. This was a few web
searches, not a systematic review; a paper needs a proper literature search (SEAL, synthetic
continued pretraining, generative replay, context distillation, MemoryLLM, Titans/nested learning,
Cross-LoRA/TransLoRA for transfer).

**Is the memory system "better than most"?** It is more rigorous than the common pattern
(embedding store + similarity search, or Q/A cards): one active memory per key, conflicts decided
by source strength instead of text similarity, superseded beliefs archived and trained as
corrections, disputes reviewed, lint against injection/secrets. But it is not unique on its own:
`aydinnyunus/sleep` has a comparable approval/supersede/source design (without weight training).
And it has limits: keys must be extracted correctly by the teacher, which is unproven on real
conversations; vector search is not built yet (FTS5 only).

**Warning from Hypnos.** On real conversational QA, retrieval beat the LoRA adapter by a wide
margin. The design doc already says the strongest baseline is base model + good retrieval. Our
retrieval-only condition must stay in every real experiment.

## Sources

- Google Research blog: https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/
- Paper: https://arxiv.org/html/2606.03979v1
- Lee, McLeish, Goldstein, Fanti (offline recurrence into fast weights, a different "sleep"): https://www.alphaxiv.org/abs/2605.26099v3
- Hypnos: https://github.com/Erazor48/hypnos
- sleeping-llm: https://github.com/vbario/sleeping-llm
- llm-sleep: https://github.com/everrete/llm-sleep
- sleep (external memory): https://github.com/aydinnyunus/sleep
- community reproduction note: https://github.com/workkiran01-lab/llm-sleep-consolidation
- SCM: https://arxiv.org/html/2604.20943v1
- Cross-model transfer: https://arxiv.org/pdf/2508.05232 (Cross-LoRA), https://www.opentrain.ai/papers/cross-model-memory-transfer-via-target-side-reader-adaptation--arxiv-2608.17050/
