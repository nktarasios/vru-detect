# Threshold justification

Fine-tuned thresholds are selected from the validation confidence sweep in
`threshold_sweep.csv`. Selection is a product decision, not a pure F1 chase.

Cost model used here:

- For `person` and `rider`, a **miss is worse than a false positive**. A missed
  vulnerable road user is the failure mode this project exists to reduce.
- For `bike`, extreme recall-first points produced noisier precision relative
  to the recall gain, so the chosen point is **best F1**.

## person

- Recommended threshold: `0.05`
- Precision: `0.291`
- Recall: `0.605`
- F1: `0.393`
- Selection rule: among thresholds within 95% of max recall, pick highest
  precision.
- Product rationale: prioritize catching pedestrians; accept more false boxes
  over silent misses.

## rider

- Recommended threshold: `0.05`
- Precision: `0.219`
- Recall: `0.432`
- F1: `0.291`
- Selection rule: same recall-first band as person.
- Product rationale: riders inherit the same miss-cost logic as pedestrians.
  Baseline rider precision was near zero due to COCO person/bicycle confusion;
  fine-tuning fixed class identity, then thresholding sets the operating point.

## bike

- Recommended threshold: `0.15`
- Precision: `0.424`
- Recall: `0.209`
- F1: `0.280`
- Selection rule: best F1, with recall/precision as tie-breakers.
- Product rationale: bike remains data-sparse and low-recall after fine-tuning.
  Pushing to `0.05` buys recall, but the precision collapse was judged too
  expensive for the current subsample. Treat this as an open risk, not a solved
  class.

## How to reread these numbers

At conf=`0.25` (comparison table in README), the model looks more "balanced."
At the recommended operating points above, person/rider intentionally move
left on the precision-recall curve. Publishing both views is deliberate:

1. `0.25` for apples-to-apples baseline vs fine-tune comparison
2. per-class recommended thresholds for the product operating point the demo uses
