# DRAMA baseline vs Full FLASH: paired bootstrap

26 games, 102 matched training-seed pairs; 95% percentile CI, 50,000 replicates, seed 2027.

| Metric | DRAMA | Full FLASH | FLASH−DRAMA | Difference CI |
|---|---:|---:|---:|---|
| Mean | 0.947464 | 1.026867 | +0.079403 | [0.007575, 0.156531] |
| Median | 0.374209 | 0.394068 | +0.019859 | [-0.061557, 0.086899] |
| IQM | 0.374136 | 0.390282 | +0.016147 | [-0.028504, 0.063560] |
| Optimality Gap | 0.547371 | 0.539690 | -0.007681 | [-0.032313, 0.017506] |

Difference CIs are paired; games stay fixed. Negative Optimality Gap differences favor FLASH.
These intervals are conditional on the selected paper seeds and benchmark games.
Freeway uses seeds 2030 and 2710 (two pairs); all other games have four pairs.
Krull uses seeds 1710, 2000, 3710 and 6000; the six Freeway/Krull scores missing from the workbook come from frozen W&B summaries.
185 selected scores matched W&B evaluate/score; 19 have only Excel/CSV score verification.
Freeway seed 2030, rather than the identical zero-score seed 2040, is inferred from the paper commit and run timestamps.
