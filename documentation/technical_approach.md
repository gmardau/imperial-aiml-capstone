# Technical approach, week by week

My first approach used a Gaussian Process with an RBF kernel and an Upper Confidence Bound acquisition function. I started by manually inspecting each function, plotting the observed points, and using simple heuristics to decide whether a function looked more exploratory or exploitative. For lower-dimensional functions, visual inspection and feature/output correlation were useful; for higher-dimensional ones, I relied more on the GP model itself.

Each week's code is in [implementation/](../implementation/) (`week1.py` to `week13.py`), and every query and result is in [weekly_data/](../weekly_data/).

## Weeks 1 to 10: the method

* Week 1 used a fixed GP setup with a simple RBF kernel and a dense grid search over the acquisition function.
* Week 2 incorporated the previous week’s data, kept tuning the length scale and UCB kappa per function, and used the accumulated observations to refine the next query.
* Week 3 added output normalization, an ARD-style GP with a constant kernel multiplied by an RBF kernel, and gradient-based optimization of the acquisition function using multi-start L-BFGS-B.
* Week 4 is still using a GP surrogate, but I've modified the model to the BoTorch one, and further customised the kernel and UCB parameters per function.
* Week 5 is identical to Week 4.
* Week 6 lowered most length-scale upper bounds to 0.5, switched function 1 to a rougher Matérn kernel (ν = 0.5), raised the UCB kappa for most functions (up to 5), and used more restarts when optimising the acquisition function.
* Week 7 narrowed the length-scale bounds of functions 1, 2, 3 and 7 (function 1 to 0.02–0.03), and raised kappa to 5 for functions 4 and 7.
* Week 8 is identical to Week 7.
* Week 9 replaced the hand-tuned setup with a method chosen per function, based on the brief and on tests with the data so far (see [week9_report.md](../assignments/week9_report.md)):
  * Function 7 turned out to be the Hartmann-6 benchmark (it matches every observation exactly), but I'm not using its known maximum: that would defeat the purpose of the project.
  * Function 1 uses a radiation-source model: the log of each reading's size is a quadratic bowl around the source, so least squares locates it.
  * The other functions use BoTorch's default GP (priors instead of hand-set length-scale bounds) with Log Expected Improvement, searched within ±0.25 of the best point so far. Function 3 is modelled on −log(−y), since it can't go above 0, and function 5 on log(y).
* Week 10 made the search box a TuRBO-style trust region: its side doubles after a week that improved the best value and halves after one that didn't. Function 4's GP became noise-free, after its week 9 result jumped well above the GP's prediction.

## Weeks 11 to 13: the same code, with some queries set by hand

`week11.py` to `week13.py` are copies of week 10's code. I set a few queries by hand where the code's choice looked weak:

* Week 11, function 5: `1.0-0.95-1.0-1.0`, to test the one direction from the (1, 1, 1, 1) corner that hadn't been tried. It was lower, so the corner is the maximum.
* Week 12, function 1: `0.628850-0.629320`, a step up-right of the week 9 point. Readings at the source had dropped slightly each week as the source model's centre drifted down-left. It gave a new best.
* Week 12, function 4: `0.359765-0.376818-0.370648-0.391546`, the peak of a second-degree fit to the points near the best. It's a shorter step than the noise-free GP's, which had overshot in week 11. It scored 0.425, close to the fit's prediction of 0.454 but below the best.
* Week 13, function 1: `0.628295-0.628804`, the peak of a curve through the four readings near the source. It gave a new best, 1.99938, almost exactly the curve's prediction.
* Week 13, function 3: `1.000000-0.525000-0.450318`, the best point with only x2 moved, in the direction that had improved. A change of 0.012 in x3 had made the result ten times worse. It scored −0.028, 40 times worse than the best: the function falls steeply below x2 = 0.546.
* Week 13, function 4: `0.393456-0.365479-0.376885-0.409609`, a small step (0.013) from the best in the GP's uphill direction, since its larger steps had overshot three weeks in a row. It scored 0.427, below the best.
* Week 13, function 6: `0.420012-0.349150-0.641525-0.817656-0.000000`, the peak of a curve through the three best points in that region, just short of the best one. The code's query sat between two readings of −0.26 and −0.28. It gave a new best, −0.153, far better than the curve's −0.229.
* Week 13, function 7: `0.215508-0.098979-0.421262-0.319386-0.335008-0.645428`, the GP's highest predicted mean. It had about the same chance of a new best as the code's query, at a third of the distance from the best point. It gave a new best, 3.182.

Week 12's query for function 5 was the code's own: a repeat of the corner. It returned exactly the same value, 8662.4825, so function 5 is noise-free.

## How the hand-set queries did

Seven of the eight hand-set queries aimed to beat the best value, and four did (functions 1 twice, 6 and 7). The ones that worked stayed between or right next to points already measured. The misses extrapolated beyond them (function 3), or relied on function 4's models, which kept predicting its peak wrongly.
