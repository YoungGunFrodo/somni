# Latest self-test results

Toy brain, not an LLM: these show pipeline dynamics, not real-model performance.

Run on 2026-11-09 (session 3), Windows 11, Python 3.13, `python -m sleepai.cli selftest`
(dim 4096, transfer to 16384, seed 0). Charts: `selftest-latest.png` (30 days),
`selftest-60days.png` (60 days, two conditions).

Forgetting policy in this run: protected → condensed → faded (see `sleepai/memory/salience.py`):
half-life 30 days, grace 7 days, a never-used memory condenses after 7 days and fades after
~100 days; every protected memory is replayed every sleep (`replay_coverage`).

## 30 days, all conditions

30 simulated days, 240 facts told, scored against ground truth after each night. "Weights" columns answer WITHOUT retrieval (what was consolidated); "With memory" uses retrieval. Questions follow a Zipf-like popularity: "frequent" = the 10% most-asked facts, "rare" = the least-asked half.

| Condition | Weights: all facts | Weights: facts ≥7 days old | Weights: frequent | Weights: rare | With memory | Stale beliefs | Base knowledge | Abstains on unknown | Sleeps adopted |
|---|---|---|---|---|---|---|---|---|---|
| retrieval-only | 1.5% | 1.8% | 2.4% | 1.5% | 100.0% | 1.8% | 100.0% | 92.5% | 0/0 |
| no-replay (ungated) | 51.4% | 43.5% | 51.2% | 53.3% | 100.0% | 1.8% | 41.7% | 95.0% | 30/30 |
| no-replay (gated) | 25.0% | 32.4% | 34.5% | 21.5% | 100.0% | 24.6% | 100.0% | 92.5% | 4/30 |
| replay | 34.3% | 44.7% | 44.0% | 29.1% | 100.0% | 24.6% | 97.5% | 100.0% | 10/30 |
| replay + dreaming | 25.0% | 32.6% | 35.7% | 19.7% | 100.0% | 24.6% | 95.8% | 97.5% | 8/30 |
| replay + dreaming + forgetting | 59.3% | 51.1% | 89.3% | 57.0% | 100.0% | 0.0% | 96.7% | 95.0% | 26/30 |

Adoption by night (A = adopted, . = rejected or nothing to learn):

- `AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA` no-replay (ungated)
- `AA..A..A......................` no-replay (gated)
- `AAAAAAAA.A.A..................` replay
- `AAAAAAAA......................` replay + dreaming
- `AAAAAAAAAAA.A.AAAAAAAAA..AAAAA` replay + dreaming + forgetting

Gate failures (number of nights each gate failed):

- no-replay (ungated): general 14, calibration 1, retention 20, drift-vs-base 21
- no-replay (gated): general 4, retention 26, drift-vs-base 1
- replay: learning 1, calibration 3, retention 20
- replay + dreaming: calibration 1, retention 22
- replay + dreaming + forgetting: retention 4

**Note on the baselines:** replay coverage (every protected memory replayed every sleep) is on
for all replay conditions. Without forgetting, every memory stays protected, so coverage means
replaying the whole store every night; that made "replay + dreaming" worse than in session 2
(25.0% vs 43.1%, 8 vs 10 adopted). Coverage helps only together with forgetting
(26/30 with it, 15/30 without it, forgetting on).

## 60 days

| replay + dreaming + forgetting | day 10 | day 20 | day 30 | day 40 | day 50 | day 60 |
|---|---|---|---|---|---|---|
| Weights: all facts | 96% | 71% | 59% | 49% | 44% | 46% |
| Weights: facts ≥7 days old | 96% | 62% | 51% | 42% | 39% | 46% |
| Weights: frequent | 100% | 88% | 89% | 71% | 72% | 82% |
| Weights: rare | 95% | 68% | 57% | 47% | 41% | 42% |

At day 60: 49/60 sleeps adopted, 111 memories protected, 288 condensed, 0 faded (fading needs
~100 days without use), retrieval 100%, stale beliefs 0%, base knowledge 96.7%. Abstains on unknown
fell to 80.0% (calibration drifts slowly; each night passes the relative gate).

Same 60 days without forgetting ("replay + dreaming"): 8/60 adopted (none after night 8),
weights 15.2% all facts, 25.0% frequent.

## Brain transfer (toy brain dim 4096 → 16384)

Source: the "replay + dreaming + forgetting" run; the transfer rebuilds the whole store in full
(condensed memories included). The new brain's candidate is scored whether or not adopted.

| Measure | Old brain | New brain candidate after one transfer sleep |
|---|---|---|
| Facts in weights (all) | 59.3% | 80.1% |
| Facts ≥7 days old | 51.1% | 80.6% |
| Base knowledge | 96.7% | 100.0% |
| Stale beliefs | 0.0% | 0.0% |
| Abstains on unknown | 95.0% | 87.5% |

Transfer adopted by gates: True. Failed gates: none (calibration −7.5 points passes as "drop not
significant" with 40 probes, as in session 2).

## Where the remaining loss is (toy brain)

Protected memories hold ~85% in the weights; condensed memories ~30–40% (they stay 100%
retrievable). Capacity is NOT the limit: trained all at once, the toy brain holds 240 facts at
84%. The loss comes from incremental sleeps: each night's new facts interfere with old ones by a
few points, and the retention gate allows 3 points per night against the current checkpoint, so
protected recall ratchets down (1.00 → ~0.85). A cumulative retention floor (a gate change, the
user's decision) is the candidate fix.

## History

| Session / step | Change | Effect |
|---|---|---|
| 2 / 0 | Reproduce on Windows (UTF-8 fix) | Identical to session 1 Run B. |
| 2 / 1a | Search no longer counts as use; Zipf questions; frequent/rare metrics | Same stall after night 10; exposed the dispute bug. |
| 2 / 1a′ | `resolve_dispute` at morning review | Retrieval back to 100%. |
| 2 / 1b | Forgetting (protected vs fading, half-life 10) | 20/30 adopted; frequent 92.9%; but most rare facts left the weights within a month. |
| 2 / 2 | Score the transfer candidate; abstain examples | Transfer adopted. |
| 3 | Slow forgetting with condensation: half-life 30, grace 7, condensed tier, replay coverage, rebuilds relearn condensed memories, transfer in full | 26/30 adopted; old facts 43% → 51%, rare 50% → 57% at day 30; 0 faded; 49/60 adopted over 60 days. |

Session 2 results: kept in git history once the repo is initialized (not yet).
