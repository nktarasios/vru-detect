# SGO cross-reference

This note compares lighting on pedestrian and cyclist crashes in the public
NHTSA Standing General Order 2021-01 archive with VRU-Detect recall by lighting.
It is a research check of published codes against a stratified evaluation.
It does not establish deployment readiness or safety certification, and it
does not describe any manufacturer's perception system.

## Result

Primary check, pedestrian and cyclist incidents together versus all-VRU recall: the
alignment rule is unmet. Pedestrian incidents versus person recall: the alignment rule
is met. Cyclist incidents versus rider and bike recall: the alignment rule is unmet.

Combined pedestrian and cyclist incidents: 64 incidents, 55 with a mapped lighting value
(38 daytime, 3 dawn/dusk, 14 night), 9 unknown, 0 other. The alignment rule is unmet.
The known-lighting mode is daytime. The lowest fine-tuned all-VRU recall is night
(recall 0.241, support 199).

Pedestrian incidents, paired with person recall: 21 incidents, 14 with a mapped lighting
value (5 daytime, 1 dawn/dusk, 8 night), 7 unknown, 0 other. The alignment rule is met.
The known-lighting mode and the lowest fine-tuned person recall are both night (recall
0.262, support 172). Unknown lighting is 7 of 21 incidents, one below the night count of
8.

Cyclist incidents, paired with rider and bike recall: 43 incidents, 41 with a mapped
lighting value (33 daytime, 2 dawn/dusk, 6 night), 2 unknown, 0 other. The alignment
rule is unmet. The known-lighting mode is daytime. The lowest rider recall is night
(recall 0.167, support 12). The lowest bike recall is night (recall 0.067, support 15).

Night is the sum of these dark SGO codes:

- `Dark - Lighted`: 14
- `Dark - Not Lighted`: 0
- `Dark - Unknown Lighting`: 0

BDD's `night` label does not separate lit and unlit roads, so a night match
is a match to that coarser bucket.

## Query

Retrieved 2026-10-02 from the public CSVs linked on
[NHTSA's SGO crash-reporting page](https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting).
No other repository is used.

Lighting population: the archive published for reports under the General
Order before the third amendment (incident reports through 15 June 2025).
Those files have a Lighting column. The third-amendment files (reports from
16 June 2025 onward) are a separate count. They enter the lighting table
only when a Lighting column with values is present.

Steps, in order:

1. Read the ADS, Level 2 ADAS, and Other CSVs. Other holds reports whose
   automation type was not classified as ADS or Level 2 ADAS.
2. Keep the highest `Report Version` for each `Report ID`. Ties break toward
   the later `Report Submission Date` (`MON-YYYY`, parsed as a date) and then
   the lower `Report ID`.
3. Drop rows whose `Report Type` is `No New or Updated Incident Reports`.
4. Collapse rows that share a non-empty `Same Incident ID`, with the same
   version and date rule. A blank `Same Incident ID` stays its own incident.
5. Keep rows whose `Crash With` is `Non-Motorist: Pedestrian` or
   `Non-Motorist: Cyclist` after case and whitespace normalization.
   `Non-Motorist: Other` is excluded. Narrative text is not searched.
6. Map Lighting onto the detector buckets: `Daylight` → daytime,
   `Dawn / Dusk` → dawn/dusk, `Dark - Lighted`, `Dark - Not Lighted`, and
   `Dark - Unknown Lighting` → night, `Unknown` or blank → unknown,
   `Other, see Narrative` and any unrecognized code → other.

Alignment rule, fixed before reading the outcome: among daytime, dawn/dusk,
and night, the unique mode of incidents with a mapped lighting value is the
same bucket as the unique minimum of the paired fine-tuned recall series.
Unknown lighting is reported and left out of the mode. A tie on either side
leaves the rule unmet. The combined check uses all-VRU recall. The
pedestrian check uses person recall. The cyclist check uses rider recall and
bike recall, and it meets the rule only when both class minima are that same
single bucket.

Detector recall is the committed stratified evaluation at confidence 0.25:
`results/finetuned/stratified.csv` and `results/baseline/stratified.csv`.
Support is true positives plus false negatives. The recall-first operating
thresholds in `results/finetuned/threshold_recommendations.csv` (person and
rider 0.05, bike 0.15) are a different operating point. Stratified recall at
those thresholds is not in the committed tables, so this note uses 0.25.

### Archive files

| system | url | bytes | sha256 | last-modified |
| --- | --- | ---: | --- | --- |
| ADS | `https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/Archive-2021-2025/SGO-2021-01_Incident_Reports_ADS.csv` | 3943732 | `cb2b38a21e2ce5c2337dfa0cffe8d6fce5e422cf93fe4c78942eec5cf72f41ba` | Thu, 26 Mar 2026 20:12:34 GMT |
| ADAS | `https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/Archive-2021-2025/SGO-2021-01_Incident_Reports_ADAS.csv` | 4092534 | `99579d4c9add8f2fd0adfcff9199210fee8aaf9cbaf64347fd7adbb4446b6f5e` | Thu, 26 Mar 2026 20:12:33 GMT |
| OTHER | `https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/Archive-2021-2025/SGO-2021-01_Incident_Reports_OTHER.csv` | 1155114 | `2464171f22671aa52047ed2e32a3442801ac85d4409f692acea3bf8c6c250899` | Tue, 12 Aug 2025 13:28:43 GMT |

### Reduction counts (archive)

- Rows read: 9920 (2295 ADS, 4027 ADAS, 3598 OTHER).
- Unique report IDs: 8160.
- After keeping the latest version: 8160.
- After dropping filings with no new incident: 4641.
- After collapsing `Same Incident ID`: 4218 (423 rows removed; 40 incidents had a blank id).
- Groups whose member reports disagreed on `Crash With`: 6.
- Groups that coded both a pedestrian and a cyclist: 0.
- VRU groups whose lighting codes disagreed: 0.
- Incidents where a pedestrian or cyclist code was dropped because the kept row used a different `Crash With`: 0.
- Pedestrian or cyclist incidents kept from the archive: 64 (21 pedestrian, 43 cyclist; 44 ADS, 19 ADAS, 1 OTHER).

### Third-amendment files, excluded from the lighting table

- ADS: `https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADS.csv` (2581860 bytes, sha256 `f856d0b9cedc5f4447515c200eeacdff5d4cabf63003dcac207385eb466ff7f5`, last-modified Tue, 15 Sep 2026 12:30:25 GMT).
- ADAS: `https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADAS.csv` (2219159 bytes, sha256 `1b15d76e4b0c8bf211494ff98d31d6b3068f7addb160421bd174b79c498bc752`, last-modified Tue, 15 Sep 2026 12:30:24 GMT).
- OTHER: `https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_OTHER.csv` (28570 bytes, sha256 `542622998ba8d8bd322c27f50d2f8e3901a3db323642b5af1d252269b3c10d84`, last-modified Tue, 15 Sep 2026 12:30:25 GMT).
- Lighting column usable: false.
- Same reduction, then pedestrian or cyclist `Crash With`: 30 incidents (9 pedestrian,
  21 cyclist) from 3817 rows. They stay out of the lighting distribution because the
  file has no usable Lighting field. The same-incident collapse left out 1 pedestrian or
  cyclist code because the kept row of that group used a different `Crash With`.

## Lighting codes

| SGO Lighting | incidents | pedestrian | cyclist |
| --- | ---: | ---: | ---: |
| Daylight | 38 | 5 | 33 |
| Dark - Lighted | 14 | 8 | 6 |
| Unknown | 9 | 7 | 2 |
| Dawn / Dusk | 3 | 1 | 2 |

## Lighting buckets and detector recall

Share of known uses daytime + dawn/dusk + night as the denominator.
Recall is fine-tuned, confidence 0.25. Baseline all-VRU recall is the last column.

| bucket | SGO | pedestrian | cyclist | share of known | recall all | support | person recall | person support | baseline recall all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| daytime | 38 | 5 | 33 | 69.1% | 0.355 | 758 | 0.365 | 691 | 0.269 |
| dawn/dusk | 3 | 1 | 2 | 5.5% | 0.367 | 98 | 0.386 | 88 | 0.276 |
| night | 14 | 8 | 6 | 25.5% | 0.241 | 199 | 0.262 | 172 | 0.146 |
| unknown | 9 | 7 | 2 | — | — | — | — | — | — |
| other | 0 | 0 | 0 | — | — | — | — | — | — |

Rider and bike recall, the classes paired with the cyclist count:

| bucket | rider recall | rider support | bike recall | bike support |
| --- | ---: | ---: | ---: | ---: |
| daytime | 0.450 | 20 | 0.170 | 47 |
| dawn/dusk | 0.200 | 5 | 0.200 | 5 |
| night | 0.167 | 12 | 0.067 | 15 |

## Roadway

Roadway Type is extracted for the same pedestrian and cyclist incidents.
It is not compared with BDD scene recall. SGO `Street` and `Intersection`
are not BDD `city street` and `residential`, and forcing that join would
invent a match the labels do not support.

| Roadway Type | incidents | pedestrian | cyclist |
| --- | ---: | ---: | ---: |
| Street | 30 | 6 | 24 |
| Intersection | 24 | 6 | 18 |
| Highway / Freeway | 5 | 5 | 0 |
| Unknown | 3 | 3 | 0 |
| Parking Lot | 2 | 1 | 1 |

`Street` is the most common roadway code (30 of 64). Street and Intersection together
are 54 of 64. All 5 Highway / Freeway incidents in this set are pedestrians.

## Limits

- The archive is crashes that named reporting entities submitted under
  SGO 2021-01. NHTSA states that reporting thresholds differ for ADS and
  Level 2 ADAS, that entities only report crashes they know about, and
  that the files are not normalized by vehicles or miles traveled. The
  system mix in the counts above is a reporting mix.
- `Crash With` is one category in the published CSV. A crash coded as a
  passenger car is outside this set even if a person was nearby.
- The same crash can remain split when `Same Incident ID` is blank or wrong.
  NHTSA documents that limitation.
- Several lighting and roadway cells are small. The pedestrian known-lighting
  total is especially thin, and unknown lighting is a large share of the
  pedestrian rows.
- Recall is from the 300-image BDD100K subsample at confidence 0.25, not
  from the crashes themselves. A low-recall bucket can be safety-relevant
  on its own; this note only asks whether that bucket is also where these
  reported crashes sit.
