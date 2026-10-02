# Product Requirements Document
## Project: VRU-Detect, Vulnerable Road User Detection Model

**Owner:** Nabil Khoury
**Status:** Phases 0–5 implemented; portfolio packaging in progress
**License intent:** MIT (code); dataset terms respected separately (see Section 7)
**Doc type:** PRD, pairs with `VRU_TECHNICAL_IMPLEMENTATION.md`
**Series:** Second of three related projects (SGO-Audit → VRU-Detect → Agentic SDLC Toolkit)
**Companion docs:** `docs/PRODUCT_DECISIONS.md`, `results/portfolio/FINDINGS.md`, `MODEL_CARD.md`

---

## 1. Summary

NHTSA's own SGO reporting criteria single out crashes involving a "vulnerable road user" (pedestrian, cyclist) as reportable regardless of other severity thresholds, meaning detection quality for these specific road users has direct, named real-world safety relevance, not just generic computer-vision benchmark value. Off-the-shelf, general-purpose object detectors are trained on broad datasets (COCO) that under-represent the specific conditions where VRU detection is hardest: small, distant, or partially-occluded pedestrians and cyclists, often at night.

**VRU-Detect** fine-tunes a modern detector (Ultralytics YOLO26, released January 2026, purpose-built for edge/robotics deployment with small-object-aware label assignment) on an open driving dataset, measures exactly how much a general-purpose baseline underperforms on VRU classes under adverse conditions, and quantifies how much fine-tuning plus deliberate confidence-threshold selection closes that gap, using the same precision/recall tradeoff methodology already applied in SGO-Audit.

---

## 2. Problem Statement

**Primary problem:** General-purpose object detectors are not optimized for the specific, safety-relevant task of reliably detecting vulnerable road users under the conditions where real crashes are most likely to occur (low light, small/distant objects, partial occlusion). There is limited public, reproducible work quantifying exactly how much performance is lost under these conditions and what a deliberately-chosen operating threshold looks like once you account for that tradeoff.

**Secondary, integrative problem (stretch goal, Phase 4):** Does the model's weakest operating conditions match the conditions where SGO-Audit's own crash-context data shows VRU-involved incidents actually cluster? If a detector's blind spot is low-light small-object detection, and the real-world crash data shows VRU crashes clustering in exactly those conditions, that's a genuine, non-obvious finding connecting the two projects, not a coincidence to force, but worth checking for.

**Why this matters beyond a portfolio exercise:** it's a real, reproducible benchmark of a named safety-relevant detection problem, using a current-generation model, with methodology (stratified evaluation, explicit threshold justification) that's directly transferable to how a real perception team would evaluate and tune a production system.

---

## 3. Goals & Non-Goals

### Goals
- Reproducible fine-tuning pipeline for VRU (pedestrian/cyclist/rider) detection on an open dataset.
- Stratified evaluation, performance broken out by lighting condition, weather, and object size/distance bucket, not just an aggregate mAP number.
- One explicit, justified confidence-threshold recommendation per class, using the same tradeoff-reasoning approach as SGO-Audit's classifier (state the cost being traded, not just the number).
- If time allows: a cross-reference of the model's weak conditions against SGO-Audit's crash-context findings (Phase 4).

### Non-goals
- Not a production-ready or deployment-certified perception system. The README and model card must state this plainly.
- Not a real-time video pipeline in v1, a single-image/small-batch inference demo is sufficient; live video is an explicit stretch, not a requirement.
- Not a multi-modal (lidar+camera) fusion project, camera-only, to keep scope tight and match the dataset most readily available.
- Not a claim about any specific manufacturer's actual perception system performance.

---

## 4. Users & Use Cases

| User | Use case |
|---|---|
| Me (primary) | A worked example of perception evaluation that connects directly to SGO-Audit |
| Open-source contributors | Researchers/engineers benchmarking small-object detection for AV safety use cases |
| Robotics/AV teams | A reference for end-to-end applied CV work: dataset handling, fine-tuning, stratified evaluation, and threshold tradeoff reasoning |

---

## 5. Scope, Phased

### Phase 0, Environment & Dataset Setup
- Select an open driving dataset with pedestrian/cyclist/rider labels and condition metadata (lighting, weather) already annotated, BDD100K is the leading candidate given its labeled diversity of conditions; nuImages is a reasonable alternative. **Confirm current access process, registration requirements, and license terms at build time**: both typically require free registration for research use, and their terms may restrict redistribution of derived model weights; this must be checked before anything is published (see Section 7).
- Convert annotations to YOLO label format; extract condition metadata (time-of-day, weather tags) as a separate stratification key used throughout evaluation.

### Phase 1, Baseline Evaluation
- Run the COCO-pretrained YOLO26 checkpoint, unmodified, against the chosen dataset's validation split.
- Measure precision/recall specifically for the VRU-relevant classes (person, bicycle/rider, mapped to the dataset's own taxonomy), stratified by lighting/weather/object-size bucket.
- This establishes the "off-the-shelf" baseline and is expected to underperform, that gap is the headline number Phase 2 will close.

### Phase 2, Fine-Tuning
- Fine-tune YOLO26 (nano or small variant, sized for practical training time on a single RTX 5080) on the dataset's training split, with class weighting or sampling strategy to counter the typical imbalance between VRU classes and dominant vehicle classes.
- Track training metrics; checkpoint the best-performing weights by validation mAP.

### Phase 3, Stratified Post-Fine-Tune Evaluation & Threshold Selection
- Repeat Phase 1's stratified evaluation on the fine-tuned model.
- Produce per-class, per-condition precision/recall curves across a confidence-threshold sweep.
- Choose and explicitly justify one recommended operating threshold per class, stating the tradeoff in the same language as prior work: "chosen to catch X% of pedestrian instances while keeping false-positive rate under Y%, given that a missed pedestrian detection is a materially worse failure mode than an extra false trigger."

### Phase 4, Cross-Reference to SGO-Audit (stretch, does not block shipping Phases 0–3 as a complete deliverable)
- Pull the condition fields (lighting, road type) already available from SGO-Audit's analysis of VRU-involved incidents.
- Compare against this project's own stratified weak-performance buckets.
- Report as a short, honest note, if the conditions line up, say so and show it; if they don't, say that too. This is a genuine check, not a result to force.

### Phase 5, Packaging
- Single-image/small-batch inference demo script (deliberately not a live-video pipeline, see non-goals).
- README with example detection outputs (images with bounding boxes).
- A model card documenting intended use, known limitations, dataset provenance, and the explicit "not production-certified" disclaimer.

### Explicitly out of scope for all phases
- Live video/streaming inference pipeline.
- Multi-modal sensor fusion.
- Any claim of exceeding published state-of-the-art benchmarks, this project's value is rigor and transparency, not a leaderboard result.

---

## 6. Success Criteria

1. Phase 0–1 reproduce a clear, quantified baseline gap on VRU classes under adverse conditions using an unmodified pretrained model.
2. Phase 2–3 show a measurable, honestly-reported improvement from fine-tuning, with one justified operating threshold per class, not just a raw accuracy number.
3. The repository is reproducible by an outside contributor who has separately obtained dataset access (the dataset itself is not redistributed if terms disallow it).
4. Phase 4, if completed, is reported with the same honesty standard as the rest of the project, a real finding or a real non-finding, not a forced narrative.

---

## 7. Responsible Use, Licensing & Disclaimers

- **Not a safety-certified system.** Every public-facing description (README, model card, any writeup) must state this is a research/portfolio benchmark, not a production or safety-certified perception system.
- **Dataset license compliance.** Confirm the exact redistribution terms of whichever dataset is chosen before publishing anything derived from it. If terms restrict redistribution of the data itself or of weights trained on it for certain uses, the repo should ship code and reproduction instructions rather than the weights or data directly, resolve this explicitly before the repo goes public, not after.
- **No manufacturer comparisons.** This project evaluates a general open-weight model, not any company's proprietary perception stack. Nothing in the repo should imply otherwise.
- **Model card required**, following standard responsible-AI model card conventions (intended use, out-of-scope use, training data provenance, known limitations, evaluation methodology).

---

## 8. Open Source Plan

- **License:** MIT for all original code. Dataset and any pretrained-weight redistribution governed separately by the source dataset's terms (Section 7).
- **Repo name suggestion:** `vru-detect`.
- **README must include:** plain-language problem statement, quickstart, dataset access instructions (pointing to the official source, not a redistributed copy, if terms require), the stratified results (baseline vs. fine-tuned), the chosen thresholds and their justification, and the model card.
- **Attribution:** clear credit to the dataset providers and to Ultralytics for the base YOLO26 architecture.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Dataset registration/access friction or licensing restricting redistribution | Confirm terms in Phase 0 before any public commitment; ship code + instructions, not restricted data/weights, if needed |
| Class imbalance suppressing VRU-class performance | Explicit class weighting/sampling strategy in Phase 2, reported openly |
| Overclaiming model readiness or safety relevance | Model card + README disclaimers (Section 7), non-negotiable |
| Phase 4 cross-reference feels forced or cherry-picked | Report honestly regardless of outcome; a clean non-finding is still a legitimate, defensible result |
| Training time on local hardware underestimated | Start with the nano/small YOLO26 variant and a data subset to validate the pipeline before committing to a full training run |

---

## 10. Timeline (effort-sized)

- Phase 0: a weekend (mostly dataset access/registration + format conversion)
- Phase 1: a few hours
- Phase 2: a weekend (training time itself should be modest on an RTX 5080 for nano/small variants, but validate on a subset first)
- Phase 3: a few hours
- Phase 4 (stretch): a few focused hours, only if time allows
- Phase 5 + writeup: a few hours

Total: roughly 2 weekends for a complete Phase 0–3+5 deliverable, consistent with your existing runway.

---

## 11. References

- Ultralytics YOLO26 documentation and release notes (confirm current version/API at build time)
- BDD100K / nuImages official dataset pages (confirm current access process and license terms at build time)
- SGO-Audit repository (for Phase 4 cross-reference and portfolio continuity)
