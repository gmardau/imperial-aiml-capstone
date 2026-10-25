# Model card: per-function Bayesian optimisation with a trust region

## Overview

- **Name:** per-function Bayesian optimisation with an adaptive trust region.
- **Type:** a sequential, model-based black-box optimiser: a Gaussian-process (GP) surrogate with an acquisition function, plus a least-squares physical model for function 1.
- **Version:** week 10, in [implementation/week10.py](implementation/week10.py). Every earlier week's version is kept in `implementation/`.

## Intended use

- **Suitable for:** maximising expensive black-box functions with few inputs (2–8), scaled to [0, 1], and a very small budget of tens of evaluations, one at a time. Examples are tuning a few hyperparameters or experimental settings, where each query has to count.
- **Avoid:**
  - *Many inputs or many observations.* Above about 10 inputs, the exact GP gets slow, and the trust region settings weren't tested there.
  - *Discrete or categorical inputs.*
  - *Settings where a bad query has a real cost.* The strategy takes risky queries when the downside is zero, as it is in this capstone.
  - *Other functions as they are.* Function 1's source model and the per-function scales are tailored to this data.

## Details

How the approach evolved over the ten rounds:

| Rounds | Surrogate | Acquisition | Search |
|---|---|---|---|
| 1–2 | scikit-learn GP, RBF kernel, hand-set length scale | UCB, kappa per function | grid search, plus manual choices from plots |
| 3 | GP with one length scale per input, normalised outputs | UCB | multi-start L-BFGS-B |
| 4–8 | BoTorch GP, Matérn kernel, hand-set length-scale bounds | UCB, kappa up to 5 | BoTorch's `optimize_acqf` |
| 9 | BoTorch's default GP (priors instead of bounds); a source model for function 1 | Log Expected Improvement | ±0.25 box around the best point |
| 10 | as week 9 | as week 9 | box that adapts each week (TuRBO-style) |

How it decides now:

1. **Fit a GP to every observation, on the function's own scale.** That's log(y) for function 5, which is positive and spans 0.1 to 8,662, and −log(−y) for function 3, which can't exceed 0.
2. **Search only a box around the best point so far.** Its side started at 0.5 in week 9. It doubles after a week whose query improved the best value and halves after one that didn't, and it's longer along inputs the GP finds less important.
3. **Pick the point in the box with the highest Log Expected Improvement**, measured against the GP's best mean at the observed points. The search starts from 20,000 random points plus small changes to the five best observations, because BoTorch's own starting points missed the best region for two functions. Fixed random seeds make every query reproducible.
4. **Function 1:** fit log10|reading| as a quadratic bowl around a radiation source, by least squares on the 13 readings nearest the strongest one, and query the bowl's centre.
5. **Function 7:** it was identified as the Hartmann-6 benchmark, but it's deliberately treated as a black box, because using the known answer would defeat the purpose of the project.

## Performance

**Metrics**
- **Primary:** the best value found for each function, compared with the best of the initial data. This is what the capstone scores.
- **Supporting:**
  - leave-one-out rank correlation: how well the surrogate ranks observations it hasn't seen;
  - stability: how far the query moves when one observation is left out;
  - simulations of the final rounds on test functions with known maxima.

**Results through week 8** (week 9's results are pending). Earlier rounds used earlier versions of the approach, so this reflects the whole campaign:

| Function | Best initial | Best so far (week) |
|---|---|---|
| 1 | 7.7e-16 | 7.7e-16 (no weekly improvement) |
| 2 | 0.611 | 0.624 (4) |
| 3 | −0.0348 | −0.0063 (4) |
| 4 | −4.03 | −0.54 (3) |
| 5 | 1,089 | 8,662 (5) |
| 6 | −0.714 | −0.247 (3) |
| 7 | 1.365 | 2.964 (7) |
| 8 | 9.598 | 9.885 (3) |

- **Improvement:** seven of the eight functions beat their initial data. Function 1 hasn't returned a real signal yet.
- **Gap to the maximum:** function 7 is the only one with a known maximum (3.32237), so it's the only one where this can be measured. It's 0.36.
- **Surrogate quality:** the current GP's leave-one-out rank correlation is between 0.78 and 1.00 for functions 2, 4, 5, 6 and 8, but only 0.66 for function 3 and 0.65 for function 7.
- **Simulations:** over three final rounds, LogEI with the box closed 41% of the gap to the maximum on average, against 17% for the UCB kappa of 5 used in earlier weeks.

## Assumptions and limitations

**Assumptions**
- **Smoothness:** the GP assumes each function is equally smooth everywhere, with one length scale per input.
- **Function 1:** the sign of a reading is noise, and its size depends only on the distance to a single source.
- **Ranges from the brief:** function 3 stays below 0, and function 5 stays above 0.
- **Scoring:** only the best value counts, so a risky query costs nothing, and few rounds remain, so exploitation is favoured.

**Limitations and failure modes**
- **Few points:** every decision rests on fewer than 50 observations, and the GP can be confidently wrong about narrow peaks, as with function 7.
- **Local maxima:** the trust region favours the best region found so far, so it can miss a better peak elsewhere.
- **Reacting to one result:** the box resizes after a single week, so one noisy reading (function 2) can shrink it.
- **Function 1:** if the source reads negative at its centre, the query is wasted. The fallback is the weaker source near (0.42, 0.42).
- **Hard stops:** the script stops with an error if function 1's source model loses its peak, or if function 3 returns exactly 0.

## Ethical considerations

- **Reproducibility:** every choice is in one script, with a per-function settings table, comments, fixed seeds and a pinned environment. With the data in this repository, anyone can regenerate each week's query exactly.
- **Real-world adaptation:** the assumptions are written down so they can be checked before the approach is reused. In real settings a bad query rarely costs nothing (a failed experiment, a harmful compound, a wasted batch), so the risk-taking parts would need to change.
- **Honesty about shortcuts:** function 7 matches a published benchmark. Using its known maximum would have scored well but said nothing about the method, so it's treated as a black box, and this card says so.

## Is this card detailed enough?

Yes, for its purpose. It explains how the approach decides, why, and where it can fail, and it points to the details instead of repeating them:
- the code for the exact settings;
- `weekly_data/` for every query;
- the [week 9 report](assignments/week9_report.md) for the evidence behind each choice.

More detail would mostly duplicate those. The one addition worth making is the week 9 and 10 results, once they're back.