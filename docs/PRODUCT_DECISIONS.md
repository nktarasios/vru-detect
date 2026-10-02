# Product Decisions

This note records the product calls behind VRU-Detect. The goal is not a
leaderboard number. The goal is a reproducible perception evaluation that a
product or perception teammate can audit: what problem mattered, what
was cut, what tradeoff was chosen, and what remains unsafe to claim.

## 1. Problem framing

**Decision:** Frame the project around *vulnerable road users under adverse
conditions*, not generic object detection.

**Why:** NHTSA Standing General Order (SGO) reporting already treats crashes
involving pedestrians/cyclists as specially reportable. That makes VRU
detection a named safety-relevant task, not just another COCO class.

**What this changes in the work:** Evaluation must be stratified by lighting,
weather, scene, and object size. Aggregate precision/recall is a summary, not
the product truth.

## 2. Scope cuts (non-goals)

| Cut | Why cut | What we shipped instead |
| --- | --- | --- |
| Live video / streaming | Scope explosion; hides data and threshold work behind infra | Single-image / small-batch demo |
| Lidar + camera fusion | Multi-modal systems need different data contracts | Camera-only BDD subset |
| Manufacturer comparisons | Unprovable and irresponsible without their stacks | Open YOLO26 baseline vs fine-tune |
| Full BDD / SOTA chase | Compute and license constraints; vanity metric risk | Honest subsample + explicit limitations |

These cuts are the product instinct signal: prefer a defensible slice over an
overbuilt demo.

## 3. Taxonomy and class mapping

**Decision:** Normalize to `person`, `rider`, `bike`.

**Why:** Those three classes cover the VRU risk story while staying close to
both BDD labels and COCO pretrained behavior.

**Tradeoff accepted:** Baseline rider evaluation is approximate because
COCO-pretrained models mostly emit `person` / `bicycle`. That mapping is
documented in `configs/classes.yaml` rather than hidden.

## 4. Data strategy

**Decision:** Prefer official ETH BDD100K; fall back to Hugging Face
`dgural/bdd100k` when ETH DNS is unavailable; do not redistribute raw data.

**Why:** License compliance beats convenience. Shipping code + reproduction
instructions is the open-source posture when weights/data may be encumbered.

**Tradeoff accepted:** Local split is a VRU-preferring 1200/300 subsample of
the validation mirror. That is enough to validate methodology; it is not enough
to claim full-BDD performance.

## 5. Model and training choices

**Decision:** Fine-tune YOLO26n with VRU oversampling (×3), not a larger model
first.

**Why:** Nano keeps iteration cheap and matches an edge/robotics deployment
story. Oversampling addresses class imbalance without inventing synthetic
labels.

**Tradeoff accepted:** Completed reference run is CPU, 10 epochs, modest
mAP50 (~0.33). Absolute scores are intentionally secondary to the evaluation
story.

## 6. Metric productization

**Decision:** Report class × condition precision/recall, then choose one
operating threshold per class with an explicit cost story.

**Why:** Perception systems are operated at a threshold. Publishing only mAP
hides the product decision.

| class | chosen threshold | product rationale |
| --- | ---: | --- |
| person | 0.05 | Missed pedestrian is worse than an extra false trigger → recall-first |
| rider | 0.05 | Same miss-cost logic as person |
| bike | 0.15 | Best F1; bike false positives were less worth extreme recall chasing |

## 7. Portfolio continuity with the public SGO archive

**Decision:** Compare the detector's recall by lighting with pedestrian and
cyclist crashes in the public NHTSA SGO 2021-01 archive. Report the overlap
with the counts and the query, whether the alignment rule is met or unmet.

**Why:** An SGO-Audit export from another repo was never available here. The
public archive is the file set that actually carries Lighting and Roadway Type.

**Current state:** `python3 -m src.crossref_sgo` writes
`results/sgo_crossref/finding.md`. On the 2026-10-02 retrieval, the combined
pedestrian-plus-cyclist check leaves the alignment rule unmet: daylight is the
crash mode and night is the lowest fine-tuned recall. The pedestrian slice
meets the rule on a small count, with unknown lighting close to that mode.
The cyclist slice leaves the rule unmet. The finding holds the counts, the
query, and the limits.

## 8. What "done" means for this portfolio piece

Done means a reviewer can answer all of these in under five minutes:

1. What problem is being optimized, and why it matters beyond CV homework?
2. What was deliberately *not* built?
3. Where does the baseline fail in safety-relevant conditions?
4. What improved after fine-tuning, and what did not?
5. Why were the operating thresholds chosen?
6. What claims are explicitly disallowed?

If those answers are clear, the piece demonstrates product instinct and
technical execution even when absolute metrics remain modest.

See also: [`PORTFOLIO_ROADMAP.md`](PORTFOLIO_ROADMAP.md) for the remaining
ceiling-raises (GPU scale-up, local demos, SGO cross-ref, public writeup).
