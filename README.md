# Imperial AI/ML Capstone Project

## 1. Project Overview
This capstone is a black-box optimization project over eight hidden objective functions. The goal is to decide which new point to query next using only previously observed inputs and outputs, without access to gradients or the true function form. That makes it a good stand-in for real ML problems such as hyperparameter tuning, experiment design, and any setting where evaluations are expensive or limited.

The project is useful and relevant because it teaches how to make good decisions under uncertainty. It combines modeling noisy data, estimating uncertainty, and balancing exploration against exploitation. Those are the same ideas that matter when choosing experiments, tuning models, or prioritizing data collection in a future ML role.

## 2. Inputs and outputs
Each function takes a normalized input vector in the unit hypercube (i.e. between 0 and 1), with dimensionality ranging from 2D to 8D depending on the function. The output of all functions is a single scalar value. Some functions do not have any output bounds, while others are only negative or positive. The available data starts with an initial sample set and then grows by one weekly query and response per function.

For submission purposes, each query must follow a single hyphen-separated string with each value rounded to six decimal places.

**Sample input:** 0.1859092-0.27476673
**Sample output:** -31.490847736812487

## 3. Challenge Objectives
The main objective is to identify the next query point that best improves the hidden objective for each function; in this case, maximisation for all functions. Because the functions are black boxes, the challenge is not just to chase high responses, but to decide when to explore uncertain parts of the space and when to exploit a promising region.

The main constraints are the small number of observations, the lack of derivatives, the varying dimensionality, and the fact that the functions can be smooth, multimodal, spiky, or nearly flat. In some cases the data suggests a clear peak; in others, the safest choice is to keep exploring until the shape of the function becomes clearer.

## 4. Technical Approach
My first approach used a Gaussian Process with an RBF kernel and an Upper Confidence Bound acquisition function. I started by manually inspecting each function, plotting the observed points, and using simple heuristics to decide whether a function looked more exploratory or exploitative. For lower-dimensional functions, visual inspection and feature/output correlation were useful; for higher-dimensional ones, I relied more on the GP model itself.

Across the submissions, the approach has changed:
* Week 1 used a fixed GP setup with a simple RBF kernel and a dense grid search over the acquisition function.
* Week 2 incorporated the previous week’s data, kept tuning the length scale and UCB kappa per function, and used the accumulated observations to refine the next query.
* Week 3 added output normalization, an ARD-style GP with a constant kernel multiplied by an RBF kernel, and gradient-based optimization of the acquisition function using multi-start L-BFGS-B.
* Week 4 is still using a GP surrogate, but I've modified the model to the BoTorch one, and further customised the kernel and UCB parameters per function.
* Week 5 is identical to Week 4.
* Week 6 lowered most length-scale upper bounds to 0.5, switched function 1 to a rougher Matérn kernel (ν = 0.5), raised the UCB kappa for most functions (up to 5), and used more restarts when optimising the acquisition function.
* Week 7 narrowed the length-scale bounds of functions 1, 2, 3 and 7 (function 1 to 0.02–0.03), and raised kappa to 5 for functions 4 and 7.
* Week 8 is identical to Week 7.
* Week 9 replaced the hand-tuned setup with a method chosen per function, based on the brief and on tests with the data so far (see [assignments/week9_report.md](assignments/week9_report.md)):
  * Function 7 turned out to be the Hartmann-6 benchmark (it matches every observation exactly), but I'm not using its known maximum: that would defeat the purpose of the project.
  * Function 1 uses a radiation-source model: the log of each reading's size is a quadratic bowl around the source, so least squares locates it.
  * The other functions use BoTorch's default GP (priors instead of hand-set length-scale bounds) with Log Expected Improvement, searched within ±0.25 of the best point so far. Function 3 is modelled on −log(−y), since it can't go above 0, and function 5 on log(y).

## 5. Repository structure
* `implementation/` has each week's script (`week1.py` to `week10.py`), and the graphs they save in `figures/`.
* `assignments/` has each week's answers to the course questions (`week1.md` to `week10.md`), plus `week9_report.md` on the week 9 approach.
* `initial_data/` and `weekly_data/` hold the starting data and each week's queries and results.
* [datasheet.md](datasheet.md) is the datasheet for the data set: what it contains, how it was collected, and its gaps and intended uses.
* [model_card.md](model_card.md) is the model card for the optimisation approach: how it decides, how it performed, and its assumptions and limitations.
* One uv environment at the root covers every week. Run a week from `implementation/`, e.g. `cd implementation && uv run python week10.py`.
