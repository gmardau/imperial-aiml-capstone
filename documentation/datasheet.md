# Datasheet: BBO capstone functions and data

This datasheet documents the data, decisions and learning for the capstone's eight black-box functions. It was created by the repository's author for the Imperial AI/ML capstone; the initial data and every function output come from the course. One datasheet covers all eight functions, with per-function answers in tables where they differ. The method itself is described in the [model card](model_card.md), and its week-by-week changes in the [technical approach](technical_approach.md).

## Function overview

### 1. Which function does this datasheet describe?

All eight, functions 1 to 8. The briefs are transcribed in [function_descriptions.md](../function_descriptions.md).

### 2–5. Scenario, input dimensionality, initial data points and output

| Function | Real-world scenario (from the brief) | Inputs | Initial points | Output |
|---|---|---|---|---|
| 1 | Finding contamination sources, such as radiation, in a 2D area | 2D | 10 | Detector reading, non-zero only near a source |
| 2 | Tuning a "mystery" ML model | 2D | 10 | Noisy log-likelihood score |
| 3 | Drug discovery with three compounds | 3D | 15 | Negative of the adverse reactions, so 0 is best |
| 4 | Placing products across warehouses, with a fast ML approximation | 4D | 30 | Difference from an expensive baseline |
| 5 | A chemical process in a factory | 4D | 20 | Yield |
| 6 | A cake recipe with five ingredients | 5D | 20 | Taster's score, negative by design |
| 7 | Tuning six hyperparameters of an ML model | 6D | 30 | Performance score, such as accuracy |
| 8 | Tuning eight hyperparameters of an ML model | 8D | 40 | Validation score |

## Nature of the data

### 1. Describe the structure of the initial dataset.

For each function, `initial_data/function_N/` holds two NumPy arrays: `initial_inputs.npy`, with shape (n, d), and `initial_outputs.npy`, with shape (n,). The shapes are (10, 2), (10, 2), (15, 3), (30, 4), (20, 4), (20, 5), (30, 6) and (40, 8). Every input is in [0, 1]. The sampling design isn't documented, but it looks uniformly random.

### 2. How does the dataset evolve as you add new queries weekly?

- **New points:** one query and one result per function per week. Over 13 weeks that's 104 new points, for 279 in total (23 to 53 per function).
- **Format:** `weekly_data/weekK_inputs.txt` and `weekK_outputs.txt` have one line per function, function 1 first. Inputs are six decimals, comma-separated in most weeks; week 12 uses the portal's hyphen format, which the week 13 script also reads.
- **Exploration pattern:**
  - *Weeks 1–8:* UCB with kappas up to 5 often pushed queries to the edges. 23% of all weekly input values are exactly 0 or 1, and 77% for function 5.
  - *From week 9:* queries concentrate around each function's best point, inside a trust region that grows or shrinks with each result.
  - *Final weeks:* very local. Function 1's last four queries are within 0.002 of each other.

### 3. Does the function include noise or randomness?

| Function | Evidence | Effect on strategy |
|---|---|---|
| 1 | Readings 0.0002 apart changed smoothly and consistently, so it's effectively deterministic near the source. The sign of the faint readings far away looks random. | The sign is ignored; near-peak readings were used to locate the peak. |
| 2 | The brief says it's noisy. The best reading, 0.718, has readings of 0.51 and 0.47 within 0.017 on either side. | The GP learns a noise level (about ±0.06), and "best so far" uses its mean, not the lucky reading. |
| 3 | No repeats. | Learned noise. |
| 4 | Two readings within 0.013–0.04 of the best (0.571) gave 0.43. That's steeper than the function's curvature, so it's probably somewhat noisy. | Treated as noise-free from week 10, which in hindsight was wrong. |
| 5 | The (1, 1, 1, 1) corner was queried three times (weeks 5, 12 and 13) and returned exactly 8662.4825 each time. Deterministic. | The repeat confirmed the maximum. |
| 6–8 | No repeats. | Learned noise. |

### 4. Based on observations, does the function appear unimodal, multimodal, noisy, or smooth?

| Function | Shape | Reasoning |
|---|---|---|
| 1 | One narrow peak | Readings fall off with distance from a source near (0.628, 0.629) and are about 0 elsewhere, as the brief describes. |
| 2 | Noisy, with a narrow ridge | The good points all have x1 ≈ 0.67–0.70, whatever x2 is. The GP's x1 length scale is 0.04, and x2's is about 2. |
| 3 | Smooth overall, with a sharp ridge | The best points have x1 = 1 and x3 ≈ 0.450. Moving x3 by 0.012, or x2 by 0.021, made the result 10–40 times worse. |
| 4 | A smooth bowl, sharp or noisy at the top | A second-degree fit ranks all held-out points almost perfectly (0.99), around (0.40, 0.36, 0.37, 0.42). |
| 5 | Unimodal and smooth | Every direction away from the (1, 1, 1, 1) corner is lower, as the brief's single peak suggests. |
| 6 | Smooth, one good region | The good points are near (0.42–0.47, 0.35, 0.64–0.72, 0.82–0.84, 0). |
| 7 | Smooth but multimodal | It matches the Hartmann-6 benchmark, which has several local maxima. x5 matters most (length scale 0.22). |
| 8 | Smooth and very flat near the top | The last five results are within 0.04 of each other, and the GP's x8 length scale is about 15, so x8 barely matters. |

## Your optimisation strategy

### 1. Which optimisation method(s) did you use?

- **Bayesian optimisation with a GP surrogate:**
  - *Weeks 1–8:* UCB, with hand-tuned kappas and length-scale bounds.
  - *From week 9:* Log Expected Improvement (LogEI) inside a TuRBO-style trust region.
- **A least-squares radiation-source model** for function 1.
- **Manual reasoning:** judgement from plots in weeks 1–2, and eight hand-set queries in weeks 11–13, each listed in the [technical approach](technical_approach.md).

### 2. Why did you choose this method for this particular function?

| Function | Method | Why |
|---|---|---|
| 1 | Source model | Almost every reading is about 0, which a GP can't learn from. The brief describes a source, and the log of a reading's size is a bowl around it. |
| 2 | GP, learned noise | The brief says the outputs are noisy. |
| 3 | GP on −log(−y) | The output can't exceed 0, and the log stretches the region just below 0, where the best values are. |
| 4 | GP, noise-free (from week 10) | Its week 9 result jumped well above the GP's prediction, and with learned noise the GP had treated that jump as noise. |
| 5 | GP on log(y) | The yield is positive and spans 0.1 to 8,662. |
| 6, 8 | Default GP | No special structure was needed. |
| 7 | Default GP | It matches Hartmann-6, but it's treated as a black box: using the known maximum would defeat the purpose of the project. |

The GP suits all of them because there are only 2–8 inputs and under 60 points, and it gives the uncertainty that the acquisition needs.

### 3. How did you balance exploration and exploitation?

- **Weeks 1–8:** UCB, whose kappa sets the balance by hand. High kappas (up to 5) explored too much and sent queries to edges and corners, where results were among the worst.
- **From week 9:**
  - *LogEI:* it weighs a high mean against high uncertainty with no kappa to tune.
  - *Trust region:* it limits the search to a box around the best point. The box doubles after a week that improved the best value and halves after one that didn't, so it exploits more where queries stop paying off.
- **Final round:** with no future round to learn from, queries aimed for the best chance of beating the best value with small, safe steps.

### 4. Did your strategy change over the weeks? Why?

Yes. The [technical approach](technical_approach.md) has the full history. The main changes were these:
- **Weeks 4–8:** a move to BoTorch, with tuned kappas and length-scale bounds.
- **Week 9:** the hand tuning was replaced with tested defaults and a method per function. The high kappas had wasted queries, the hand-set bounds rather than the data were deciding the length scales, and a GP couldn't learn function 1.
- **Week 10:** an adaptive trust region, and a noise-free GP for function 4.
- **Weeks 11–13:** hand-set queries where local evidence contradicted the model.

## Data handling and preprocessing

### 1. Did you rescale or normalise inputs? Why or why not?

No. The inputs are already in [0, 1], and the GP's length scales, one per input, take care of each input's importance. Outputs are standardised inside the GP, and some are transformed (see question 3).

### 2. Did you train any surrogate models?

- **GPs:** scikit-learn in weeks 1–3, BoTorch from week 4.
- **Least-squares quadratic models:** the source model for function 1. In the final weeks, local second-degree fits and curves through nearby readings also guided hand-set queries for functions 1, 4 and 6.
- **Tried but not adopted:**
  - *SVRs and neural networks:* they predicted held-out points worse than the GP.
  - *Matérn kernels:* about as good as RBF, so the simpler choice stayed.

### 3. If yes, what preprocessing did the surrogate require?

- **Kernel:** RBF with one length scale per input (Matérn in weeks 4–8).
- **Hyperparameters:** priors on the length scales and the noise, which replaced hand-set bounds in week 9. The values are fitted by maximising the marginal likelihood.
- **Noise:** learned, except for function 4 from week 10, where it's fixed near zero.
- **Output transforms** at modelling time only, never applied to the stored data:
  - log(y) for function 5;
  - −log(−y) for function 3;
  - log10|y| for function 1, floored at 1e-320 so that an exact zero still has a logarithm.

### 4. Did you handle outliers or unusual data points?

No data was removed. Unusual points were handled in the model instead:
- **Function 1:**
  - *Negative readings:* its seven negative readings show no spatial pattern, so the sign is ignored and only their size is used.
  - *Distant readings:* the source model uses only readings within 0.4 of the strongest one, because farther readings may come from a second, weaker source.
- **Function 2:** "best so far" is the GP's highest mean, so one lucky reading doesn't set the bar.

## Weekly iteration and learning

### 1. How did new data points change your understanding of the function landscape?

- **Function 1:** week 9's reading of 1.99, the first real signal after 18 near-zero readings, confirmed the source model. Readings near the peak then mapped it finely.
- **Function 5:** results kept rising towards the (1, 1, 1, 1) corner, and tests in every direction around it confirmed it as the maximum.
- **Functions 3 and 4:** queries close to the best revealed how sharp their tops are.
- **Function 8:** the recent results showed a plateau.
- **Function 7:** the GP overestimated it twice (weeks 9 and 10) before it improved again.

### 2. Did you encounter local optima? How did you detect them?

Probably for function 2, whose best hasn't improved since week 10, and possibly for function 7, which as Hartmann-6 has several local maxima. I detected them from plateaus in the best value, repeated misses around the same point, and the trust region shrinking as a result. I didn't search other regions systematically in the final weeks, so I can't rule out better peaks elsewhere.

### 3. Which queried inputs were most informative and why?

- **Function 1, week 9:** the first non-zero reading, which validated the whole source model.
- **Function 5, around the corner:** the four one-direction checks (weeks 4, 9, 10 and 11) and the repeat (week 12). Together they confirmed the maximum and showed the function is deterministic.
- **High-gradient points near the best:** function 4 in week 11 and function 3 in weeks 12–13. They showed how sharp those peaks are.
- **Boundary points from weeks 6–8:** they showed that the edges were poor for most functions.

### 4. If you restarted, what would you do differently?

- **Measure noise early,** with one repeated query per function: the noise assumption mattered more than any other setting.
- **Start with LogEI and a trust region,** and skip the high kappas.
- **Check calibration every week:** compare each result with the model's 68% range, and widen the uncertainty when it misses.
- **Use local fits near the peak sooner** for function 1.
- **Keep one or two queries per function** to look for other peaks before narrowing down.

## Performance and results

### 1–2. What is the best output value you achieved, and which input vector produced it?

| Function | Best value | Input vector | Week |
|---|---|---|---|
| 1 | 1.99938 | 0.628295-0.628804 | 13 |
| 2 | 0.717634 | 0.681118-1.000000 | 10 |
| 3 | −0.000704 | 1.000000-0.545968-0.450318 | 11 |
| 4 | 0.570867 | 0.397544-0.362300-0.374132-0.421624 | 10 |
| 5 | 8662.48 | 1.000000-1.000000-1.000000-1.000000 | 5 |
| 6 | −0.152900 | 0.420012-0.349150-0.641525-0.817656-0.000000 | 13 |
| 7 | 3.18182 | 0.215508-0.098979-0.421262-0.319386-0.335008-0.645428 | 13 |
| 8 | 9.98912 | 0.141742-0.150552-0.167304-0.164967-0.793873-0.496895-0.230659-0.704508 | 13 |

### 3. How confident are you that this is near the global maximum? Why?

| Function | Confidence | Why |
|---|---|---|
| 1 | High | Four readings map the peak along one line. The other direction is untested, but there's little room left. |
| 2 | Low | Its outputs are noisy, the best is probably a lucky reading of a ridge around 0.6, and the brief warns of local optima. |
| 3 | High in value | Outputs can't exceed 0, and the best is within 0.0007 of it, even though the exact location of the ridge is uncertain. |
| 4 | Medium | The peak of the bowl is well located, but the top is sharp or noisy, and 0.571 may be a high reading. |
| 5 | Very high | The function is deterministic, every direction from the corner is lower, and the brief says there's one peak. |
| 6 | Medium | It improved by a third in the final round, so the maximum may still be higher. |
| 7 | Medium | It's 96% of Hartmann-6's known maximum (3.32), which is the only direct check available. |
| 8 | Medium-high | Its last five results are within 0.04, but 53 points sparsely cover eight dimensions. |

### 4. Did your results align with expectations for this function?

Mostly, yes:
- **Function 1** behaved like a source that only reads non-zero nearby.
- **Function 2** was noisy.
- **Functions 3 and 6** stayed negative and approached 0.
- **Function 5** had a single peak.
- **Function 8** reached a strong local maximum.
- **Function 4's** brief warns of local optima, but its data looked like a single bowl.
- **Function 7** turned out to be a standard benchmark rather than a real ML model.

## Ethical, practical and general considerations

### 1. How does this black-box optimisation task relate to real-world applications?

It mirrors any problem where each evaluation is expensive and gradients aren't available: hyperparameter tuning, drug dosing, chemical yields, recipe design and sensor search. The same trade-off between exploring and exploiting under a small budget applies.

### 2. What limitations arise from the synthetic nature of the function?

- **No real costs:** a bad query costs nothing, and there are no constraints or side effects. That makes risky queries look cheaper than they would be in reality.
- **Illustrative labels:** the scenarios in the briefs are examples. Function 3's outputs, for instance, aren't real counts of adverse reactions.
- **Known identity:** function 7 matches a published benchmark, which a real problem wouldn't.
- **Scoring:** only the best value is scored, not the cost of reaching it.

### 3. Would your strategy scale to more serious or more expensive problems? Why or why not?

- **Similar problems:** it scales to similar ones, up to about 10 inputs and a few hundred points, where the exact GP is still fast.
- **Larger problems:** they would need sparse GPs, batch queries for parallel evaluations, or several trust regions.
- **Real stakes:** where a bad query has a real cost, the strategy would need safety constraints and better-calibrated uncertainty, since its 68% ranges held only a quarter of the results.

### 4. What risks or pitfalls should a future user be aware of when analysing this function?

- **Biased sampling:** the data is concentrated near the best points and at the edges. Don't use it to estimate global properties, like how often values are high, or to train a general surrogate without accounting for that.
- **Noise assumptions:** check them before trusting a model, as functions 4 and 5 show.
- **The model's ranges are too narrow:** treat its predictions as optimistic.
- **Function 1's sign** isn't meaningful far from the source.
- **Hand-set queries** in weeks 11–13 aren't produced by the code; they're listed in the [technical approach](technical_approach.md).

## Access and maintenance

- **Where:** in this repository, in `initial_data/` and `weekly_data/`. Changes are tracked in the repository's history.
- **Terms of use:** no licence has been set. The initial data and the function outputs come from the course, so reuse outside it should follow the course's terms.
- **Status:** complete. All 13 rounds are recorded, and the data is stored as received, with no cleaning or filtering.
