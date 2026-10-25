# Datasheet: BBO capstone data set

## Motivation

- **Why it was created:** to record every observation of the capstone's eight black-box functions. Each week's query is chosen from everything seen so far, and the record lets anyone review and reproduce those choices.
- **Task it supports:** sequential black-box maximisation, choosing the next input to query for each function, with one query per function per week.
- **Who created it:** the repository's author, for the Imperial AI/ML capstone. The initial data and every function output come from the course.

## Composition

There are eight functions. Each takes inputs in the unit hypercube [0, 1]^d and returns one number; [function_descriptions.md](../function_descriptions.md) has their descriptions from the brief.

| Function | Inputs | Initial points | Weekly results | Total | Output range so far |
|---|---|---|---|---|---|
| 1 | 2 | 10 | 8 | 18 | −0.0036 to 7.7e-16 |
| 2 | 2 | 10 | 8 | 18 | −0.066 to 0.624 |
| 3 | 3 | 15 | 8 | 23 | −0.399 to −0.0063 |
| 4 | 4 | 30 | 8 | 38 | −35.8 to −0.54 |
| 5 | 4 | 20 | 8 | 28 | 0.11 to 8,662 |
| 6 | 5 | 20 | 8 | 28 | −3.04 to −0.25 |
| 7 | 6 | 30 | 8 | 38 | 0.0027 to 2.96 |
| 8 | 8 | 40 | 8 | 48 | 5.59 to 9.89 |

That's 239 observations through week 8.

**Format**
- `initial_data/function_N/initial_inputs.npy` (n × d) and `initial_outputs.npy` (n): NumPy arrays, as provided by the course.
- `weekly_data/weekK_inputs.txt` and `weekK_outputs.txt`: one line per function, function 1 first. Inputs are comma-separated, with six decimals.

**Gaps**
- **Pending results:** week 9's results haven't come back yet (its inputs are recorded), and week 10 hasn't been submitted.
- **Clustered queries:** recent rounds concentrate near each function's best point, while earlier rounds with a high UCB kappa pushed queries to the edges of the space. About a quarter of all weekly input values are exactly 0 or 1, and 72% for function 5.
- **Sparse coverage in higher dimensions:** fewer than 50 points cover the 6D and 8D spaces.
- **Function 1:** its outputs are almost all effectively zero (as small as 1e-291), and seven are negative. The region of a possible second, weaker source, near (0.42, 0.42), is barely sampled.
- **No repeats:** no point has been queried twice, so noise can't be measured directly, even though the brief describes function 2 as noisy.

## Collection process

- **Initial data:** provided by the course at the start. Its sampling design isn't documented; it looks uniformly random.
- **Weekly queries:** one input per function per week, submitted to the course portal as six decimals joined by hyphens. The output came back before the next round.
- **How the queries were generated:**
  - Weeks 1–2: a GP with an RBF kernel and UCB, combined with manual judgement from plots and input-output correlations.
  - Week 3: normalised outputs, one length scale per input, and a gradient-based search of the acquisition function.
  - Weeks 4–8: a BoTorch GP (Matérn kernel) with hand-set length-scale bounds and a UCB kappa per function, up to 5.
  - Week 9: a method per function. Function 1 used a radiation-source model; the others used BoTorch's default GP with Log Expected Improvement, searched within ±0.25 of the best point, with log scales for functions 3 and 5.
  - Week 10: the same, with a search box that adapts each week (TuRBO-style).
- **Time frame:** ten weekly rounds, from late July to late October 2026.

## Preprocessing and uses

- **Stored as received:** there's no cleaning or filtering. The weekly inputs are the six-decimal values that were submitted.
- **Transformations at modelling time only**, never to the stored data: log(y) for function 5, −log(−y) for function 3, log10|y| for function 1, and standardisation inside the GP.
- **Intended uses:**
  - choosing the capstone's queries;
  - reproducing and reviewing the weekly decisions;
  - comparing optimisation strategies on these eight functions.
- **Inappropriate uses:**
  - *Treating it as a representative sample of the functions.* The sampling is biased towards good regions and edges. Don't use it to estimate global properties, like how often values are high, or to train a general surrogate without accounting for that bias.
  - *Reading real-world meaning into it.* The brief's "sample applications" are illustrative: function 3's outputs, for example, aren't real counts of adverse reactions.

## Distribution and maintenance

- **Where:** in this repository, in `initial_data/` and `weekly_data/`.
- **Terms of use:** no licence has been set. The initial data and the function outputs come from the course, so reuse outside it should follow the course's terms.
- **Maintenance:** the repository's author adds each week's query and result until the capstone ends. Changes are tracked in the repository's history.
