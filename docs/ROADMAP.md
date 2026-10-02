# Roadmap: from benchmark to launch gates

**Updated:** 2026-10-02

Current state: a CPU-trained YOLO26n run (10 epochs, 1,200/300 BDD100K subsample) improved
precision 0.396 → 0.685 and recall 0.248 → 0.335, with night and small-object recall still the
gaps. The SGO cross-reference (Phase 4 of the PRD) was recorded as unavailable.

**Goal:** answer the question a perception team actually faces: *which conditions is this
detector fit to ship in, and which need a gate?*

## Rules

1. Report class × condition metrics, never aggregate mAP alone.
2. Never claim deployment readiness or safety certification (MODEL_CARD).
3. Do not redistribute BDD100K data or restricted weights.
4. One phase per pull request.

---

## Phase A: GPU scale-up

- Add a GPU training config (`configs/train_gpu.yaml`): full BDD100K VRU-relevant split where
  license and disk allow, YOLO26s and YOLO26n, 50+ epochs, same oversampling.
- Re-run the stratified evaluation on the same 300-image holdout **and** on the full validation
  split. Keep the CPU run as the baseline row.
- `results/gpu/comparison.md`: CPU vs GPU, nano vs small, per class × condition.

**Acceptance:** results regenerate with `scripts/reproduce.sh --gpu`. Hardware and wall-clock
training time recorded.

## Phase B: The SGO cross-reference, for real

**Status:** implemented. See `results/sgo_crossref/finding.md`.

- Using the public NHTSA SGO archive directly (no dependency on another repo), extract crashes
  involving pedestrians or cyclists and their lighting and roadway fields.
- Compare the lighting distribution of real VRU crashes with the detector's recall by lighting.
- `results/sgo_crossref/finding.md`: report the overlap honestly, whether it lines up or not.

**Acceptance:** a real finding or a real non-finding, with the counts and the query.

## Phase C: Launch gates

- `results/launch_gates.md`: for each class × condition, a pass / gate / block call against
  explicit recall floors (for example person-at-night recall ≥ a stated floor), with the miss-cost
  reasoning from `docs/PRODUCT_DECISIONS.md`.
- One chart: the condition grid colored by gate status.

**Acceptance:** every gate decision traces to a metric in `results/`.

## Phase D: README refresh

- Lead the README with the launch-gate grid and the cross-reference finding.
