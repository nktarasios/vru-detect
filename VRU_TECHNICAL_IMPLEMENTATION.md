# Technical Implementation Doc
## Project: VRU-Detect
**Pairs with:** `VRU_PRD.md`: read that first for problem statement, scope, and non-goals.
**Audience:** Cursor / Claude Code + human review.

---

## 1. Repo Structure

```
vru-detect/
├── data/
│   ├── raw/                      # gitignored, dataset not committed (license terms)
│   └── processed/                # YOLO-format labels + condition metadata
├── src/
│   ├── dataset.py                  # Phase 0: annotation conversion + metadata extraction
│   ├── baseline_eval.py             # Phase 1: pretrained model, stratified eval
│   ├── train.py                     # Phase 2: fine-tuning
│   ├── evaluate.py                  # Phase 3: stratified post-fine-tune eval + threshold selection
│   ├── crossref_sgo.py              # Phase 4: public NHTSA SGO lighting cross-reference
│   └── infer_demo.py                # Phase 5: single-image/small-batch inference demo
├── configs/
│   ├── classes.yaml                 # VRU class list + dataset taxonomy mapping
│   └── train_config.yaml            # training hyperparameters
├── notebooks/
│   ├── 01_baseline_exploration.ipynb
│   └── 02_stratified_results.ipynb
├── models/                        # gitignored or Git LFS, trained checkpoints not committed directly to git history
├── tests/
│   └── test_dataset.py, test_evaluate.py, ...
├── MODEL_CARD.md
├── README.md
├── CONTRIBUTING.md
├── LICENSE                        # MIT
└── requirements.txt
```

Design principle: same as SGO-Audit, each phase is a standalone script callable on its own, not one monolithic pipeline. This makes partial reproduction (e.g., someone just wanting the baseline eval) possible without running the full project.

---

## 2. Tech Stack

- **Python 3.11+**
- **ultralytics** package (YOLO26): confirm current package name/API at build time; Ultralytics' interface has historically been stable across YOLO versions (`from ultralytics import YOLO`), but verify against current docs before writing training code.
- **torch** with CUDA support for the RTX 5080
- **pandas**: metadata/condition stratification bookkeeping
- **matplotlib**: precision/recall curves, example detection visualizations
- **pytest**

No web framework, no deployment infra, no live video handling.

---

## 3. Phase 0, Dataset Setup (`dataset.py`)

- Manually register for and download the chosen dataset (BDD100K or nuImages) per its official current instructions, this is a one-time manual step, not something to automate/scrape.
- Confirm the exact current class taxonomy and condition-metadata field names against the dataset's own documentation before writing conversion logic, do not assume field names from this doc are exact; datasets update their schemas over time.
- Write a conversion script from the dataset's native annotation format to YOLO's expected label format (class index + normalized bounding box per line, one file per image).
- Extract condition metadata (time-of-day, weather) into a separate lookup keyed by image ID, this is what powers every stratified evaluation later, so get this right early and test it.
- Output: a `processed/` directory with YOLO-format labels plus a `conditions.csv` mapping image ID → lighting/weather/object-size-bucket tags.

---

## 4. Phase 1, Baseline Evaluation (`baseline_eval.py`)

- Load the COCO-pretrained YOLO26 checkpoint, unmodified.
- Run inference on the dataset's validation split.
- Map the dataset's own VRU-relevant classes to the closest COCO-equivalent classes (e.g., "pedestrian"/"person", "cyclist"/"rider" to COCO's "person" and "bicycle"): document this mapping explicitly in `configs/classes.yaml`, since it's a methodological choice that affects every downstream number.
- Compute precision/recall for VRU classes, stratified by the condition metadata from Phase 0 (lighting, weather, and an object-size bucket derived from bounding-box pixel area, e.g., small/medium/large terciles).
- Output: a stratified results table (condition × class × precision/recall) and one summary chart showing where the baseline is weakest.

---

## 5. Phase 2, Fine-Tuning (`train.py`)

- Fine-tune the YOLO26 nano or small variant (choose based on training-time tradeoffs observed on the RTX 5080, start small, scale up only if time/results justify it) on the dataset's training split.
- Address class imbalance explicitly: either class-weighted loss (if supported by the current Ultralytics API) or oversampling of VRU-class-containing images in the training set. Document which approach was used and why.
- Track training curves; checkpoint on best validation mAP for the VRU classes specifically, not just overall mAP (overall mAP can look fine while VRU-class performance lags, since vehicle classes typically dominate).
- Validate the pipeline on a small data subset first before committing to a full training run, to catch data/config issues cheaply.

---

## 6. Phase 3, Stratified Evaluation & Threshold Selection (`evaluate.py`)

- Repeat Phase 1's exact stratified evaluation methodology on the fine-tuned model, so baseline and fine-tuned results are directly comparable (same strata, same metrics).
- For each VRU class, sweep the confidence threshold and plot a precision/recall curve per condition stratum.
- Select one recommended operating threshold per class. The justification must name the specific tradeoff being made (e.g., "prioritizing recall over precision for the pedestrian class, since a missed detection is a worse failure mode than an extra false trigger"): this is the section that should read almost identically in structure to the threshold-selection reasoning in SGO-Audit's Phase 1, since it's deliberately the same methodology applied to a new problem.
- Output: final stratified results table (baseline vs. fine-tuned, side by side), the chosen thresholds with justification, and the precision/recall curve charts.

---

## 7. Phase 4, SGO cross-reference (`crossref_sgo.py`)

- Read the public NHTSA SGO 2021-01 CSVs directly. Do not depend on another repo.
- Keep pedestrian and cyclist incidents (`Crash With` of `Non-Motorist: Pedestrian` or `Non-Motorist: Cyclist`) after latest-version and `Same Incident ID` reduction. Extract Lighting and Roadway Type.
- Use the pre-third-amendment archive for lighting. The third-amendment files drop that column; count those crashes and leave them out of the lighting table.
- Compare the lighting distribution with fine-tuned recall by time of day from `results/finetuned/stratified.csv` (confidence 0.25). The alignment rule is documented in the generated note.
- Write `results/sgo_crossref/finding.md` with the counts and the query. A met rule and an unmet rule are both valid outcomes.

---

## 8. Phase 5, Packaging (`infer_demo.py`, `MODEL_CARD.md`, `README.md`)

- A simple script: load the fine-tuned weights, run inference on a small folder of sample images, save annotated output images with bounding boxes and confidence scores at the chosen thresholds.
- `MODEL_CARD.md`: intended use, out-of-scope use (explicitly: not a certified safety system), training data provenance and license, evaluation methodology summary, known limitations (dataset bias, class-mapping caveat from Phase 1, small validation sample within some condition strata if applicable).
- `README.md`: problem statement in plain language, quickstart, the headline baseline-vs-fine-tuned comparison, links to the dataset's official source and to SGO-Audit.

---

## 9. Testing Strategy

- Unit tests for the annotation-conversion logic (Phase 0): this is the highest-risk-of-silent-bug component, since a subtle labeling error would quietly corrupt every downstream metric.
- Unit tests for the stratification bucketing logic (condition/size tercile assignment).
- A small synthetic fixture (a handful of hand-constructed annotated images) for fast testing without requiring the full dataset present.

---

## 10. Open Source Packaging Checklist

- [x] `LICENSE`: MIT (code only; note dataset/weight redistribution terms separately in README)
- [x] `README.md` per Section 8 (portfolio case-study rewrite)
- [x] `MODEL_CARD.md` per Section 8
- [x] `CONTRIBUTING.md`
- [x] `requirements.txt` with working lower bounds
- [x] Confirm dataset redistribution terms before publishing, ship code + reproduction instructions, not restricted data/weights
- [x] No dataset files committed to git history (gitignore `data/raw/`)
- [x] Portfolio decision log + findings charts (`docs/PRODUCT_DECISIONS.md`, `results/portfolio/`)
- [ ] Optional: attach local `infer_demo` gallery for interviews (gitignored outputs)
- [x] Phase 4 SGO cross-ref against the public NHTSA archive (`results/sgo_crossref/finding.md`)

---

## 11. Build Sequence & Prompts for Cursor / Claude Code

Run these in order, one at a time, confirm each phase works before moving to the next.

**Prompt 1, scaffold + Phase 0:**
> Read VRU_PRD.md and VRU_TECHNICAL_IMPLEMENTATION.md fully first. Scaffold the repo structure from Section 1 of the technical doc. Then implement Phase 0 (`src/dataset.py`): annotation conversion to YOLO format and condition-metadata extraction, per Section 3. I will handle the manual dataset download/registration myself, write the script assuming the raw files already exist at `data/raw/` in the dataset's native format, and confirm the actual current field names against the real downloaded files rather than assuming this doc's field names are exact. Add unit tests per Section 9.

**Prompt 2, Phase 1:**
> Implement `src/baseline_eval.py` per Section 4: load the pretrained YOLO26 checkpoint, run it on the validation split, map classes per the documented COCO-equivalence, and produce the stratified precision/recall results and summary chart. Document the class mapping explicitly in `configs/classes.yaml`.

**Prompt 3, Phase 2:**
> Implement `src/train.py` per Section 5: fine-tune the YOLO26 nano or small variant on the training split, with explicit class-imbalance handling. Validate on a small data subset first before running a full training job, and tell me the expected training time estimate before starting the full run.

**Prompt 4, Phase 3:**
> Implement `src/evaluate.py` per Section 6: repeat the Phase 1 stratified evaluation on the fine-tuned model using identical strata and metrics, sweep confidence thresholds, and produce one recommended operating threshold per class with explicit tradeoff justification in the output.

**Prompt 5, Phase 4:**
> Implement `src/crossref_sgo.py` per Section 7: read the public NHTSA SGO archive, extract pedestrian and cyclist crashes with lighting and roadway, and compare that lighting distribution with stratified recall. Report the result plainly whether or not it shows alignment.

**Prompt 6, Phase 5:**
> Implement `src/infer_demo.py` and write `MODEL_CARD.md` and `README.md` per Section 8. Include the baseline-vs-fine-tuned comparison table and the chosen thresholds with justification directly in the README.

**Standing instruction to include at the top of the session:**
> Do not add functionality beyond what's scoped in VRU_PRD.md Section 5 without checking with me first, in particular, do not build a live-video pipeline or multi-modal fusion even if it seems like a natural next step.

---

## 12. Open Items to Confirm Before/During Build

1. Final dataset choice (BDD100K vs. nuImages) and its current registration process and redistribution terms.
2. Current Ultralytics YOLO26 package API specifics (import paths, config format): confirm against current docs, not assumed from this doc.
3. Actual achievable training time for the chosen model size on the RTX 5080, validated on a subset before committing to a full run.
