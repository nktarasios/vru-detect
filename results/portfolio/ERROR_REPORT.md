# Error / residual-risk report

This report concentrates on **false negatives (misses)**: the failure mode
that matters most for VRU risk analysis.

## Class-level misses (conf=0.25)

| class | baseline FN | fine-tuned FN | FN delta | baseline recall | fine-tuned recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| person | 710 | 622 | -88 | 0.257 | 0.349 |
| rider | 24 | 25 | +1 | 0.351 | 0.324 |
| bike | 62 | 57 | -5 | 0.075 | 0.149 |
| all | 796 | 704 | -92 | 0.248 | 0.335 |

## Where misses still concentrate after fine-tuning

Highest-FN condition buckets (excluding size; support ≥ 10):

| stratum | value | FN | recall | support |
| --- | --- | ---: | ---: | ---: |
| scene | city street | 623 | 0.342 | 947 |
| timeofday | daytime | 489 | 0.355 | 758 |
| weather | clear | 249 | 0.318 | 365 |
| weather | undefined | 223 | 0.314 | 325 |
| timeofday | night | 151 | 0.241 | 199 |
| weather | overcast | 133 | 0.351 | 205 |
| scene | residential | 64 | 0.264 | 87 |
| timeofday | dawn/dusk | 62 | 0.367 | 98 |

## Size is the sharpest remaining knife-edge

| size | baseline recall | fine-tuned recall | baseline FN | fine-tuned FN |
| --- | ---: | ---: | ---: | ---: |
| small | 0.003 | 0.079 | 329 | 304 |
| medium | 0.185 | 0.335 | 282 | 230 |
| large | 0.517 | 0.559 | 185 | 169 |

## Spotlight: night pedestrians

- Baseline night-person recall: **0.145** (FN=147)
- Fine-tuned night-person recall: **0.262** (FN=127)
- Absolute gain: **+0.116**

Interpretation: fine-tuning helps, but night pedestrians remain a first-class
product risk. Any claim about this model should lead with this residual gap, not hide
it behind aggregate precision.

## Baseline miss geography (for contrast)

| stratum | value | FN | recall | support |
| --- | --- | ---: | ---: | ---: |
| scene | city street | 718 | 0.242 | 947 |
| timeofday | daytime | 554 | 0.269 | 758 |
| weather | clear | 283 | 0.225 | 365 |
| weather | undefined | 240 | 0.262 | 325 |
| timeofday | night | 170 | 0.146 | 199 |
| weather | overcast | 150 | 0.268 | 205 |
| timeofday | dawn/dusk | 71 | 0.276 | 98 |
| weather | snowy | 59 | 0.234 | 77 |

## Next experiments this report implies

1. Night-heavy resampling or night-only fine-tune pass
2. Higher resolution / SAHI-style tiling for small objects
3. Rider/bike co-occurrence aware augmentation
4. Keep publishing stratified FN tables, do not regress to mAP-only reporting
