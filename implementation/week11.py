# Week 11 onwards: propose the next query for each of the eight black-box functions.
#
#   Functions 2-8: Bayesian optimisation. A Gaussian process (GP) is fitted to every
#     observation so far, and the next query is the point with the highest Log Expected
#     Improvement inside an adaptive trust region around the best one so far (TuRBO).
#   Function 1: a radiation-source model fitted by least squares, because a GP can't learn
#     from readings that are almost all ~0.
#
# ../assignments/week9_report.md explains what changed and why.
#
# Usage (from the implementation folder):
#   uv run python week11.py                # all eight functions, no graphs
#   uv run python week11.py 3              # function 3 only
#   uv run python week11.py --plots show   # show the graphs in windows
#   uv run python week11.py --plots save   # save the graphs to figures/weekN/

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent  # the implementation folder
ROOT = HERE.parent  # the repository, with initial_data/ and weekly_data/

# The round to propose: the one after the last week whose results are in weekly_data/, so
# the same script works every week from week 11 on (and never proposes a week before 11)
LAST_RESULTS = max(week for week in range(1, 100) if (ROOT / f"weekly_data/week{week}_outputs.txt").exists())
WEEK_NO = max(11, LAST_RESULTS + 1)

# Output transforms as (forward, inverse, label): the GP is fitted to forward(y), its
# predictions are mapped back to the original units with inverse(), and the label names
# the transformed scale in the graphs.
TRANSFORMS = {
    None: (lambda y: y, lambda t: t, "y"),
    # For positive outputs that span several orders of magnitude.
    "log": (np.log, np.exp, "log(y)"),
    # For negative outputs with a ceiling at 0. It stretches the region just below 0,
    # where the best values are, and every prediction maps back to a negative value,
    # so the model can't promise an impossible y > 0.
    "neg_log": (lambda y: -np.log(-y), lambda t: -np.exp(-t), "−log(−y)"),
}

# Per-function settings:
#   method:    "gp"        -> Gaussian process + LogEI (Bayesian optimisation)
#              "source"    -> fit a radiation-source model (function 1)
#              "hartmann6" -> query the known maximum of Hartmann-6 (function 7)
#   transform: fit the GP to a transformed output (a key of TRANSFORMS)
#   noise_free: fix the GP's noise near zero, so it follows every observation exactly
SETTINGS = {
    1: dict(method="source"),
    2: dict(method="gp"),
    # The negative of side effects, so it can't exceed 0; the best is already -0.006.
    # Fitted to y directly, the GP predicted impossible values above 0, and its query
    # jumped around when any single observation was left out.
    3: dict(method="gp", transform="neg_log"),
    # In week 9 it jumped to +0.48, far above the GP's prediction. With learned noise, the GP
    # treated that as a lucky reading and proposed a near-repeat of it; noise-free, it follows
    # the jump and steps further uphill.
    4: dict(method="gp", noise_free=True),
    # Yield is positive and spans 0.1 to 8,662, so log(y) is much easier to model.
    5: dict(method="gp", transform="log"),
    6: dict(method="gp"),
    # It is the Hartmann-6 benchmark, but querying its known maximum ("hartmann6") would
    # defeat the purpose of the project, so it is treated as a black box like the others.
    7: dict(method="gp"),
    8: dict(method="gp"),
}

# The GP functions are only searched inside a box around their best point so far: TuRBO's
# "trust region" (Eriksson et al., 2019). With a few rounds left, improving the best region
# is worth more than exploring far away, where the GP extrapolates least reliably.
# - Its size adapts: the side length doubles after a week whose query improved the best
#   value, and halves after a week whose query didn't. TuRBO waits for several successes or
#   failures in a row first; with only a few rounds left, this reacts every week.
# - Its shape follows the GP: longer along inputs with long length scales (which matter
#   less), and shorter along inputs that matter more.
# It started in week 9 as a box with sides of 0.5 (+/- 0.25), and stays within TuRBO's limits.
TRUST_REGION_START_WEEK = 9
START_LENGTH, MIN_LENGTH, MAX_LENGTH = 0.5, 0.5 ** 7, 1.6


# --- Data ---

# All observations of one function: the initial data, then one row per week. Also returns
# how many of them are initial data, which the graphs colour differently.
def load_data(f, weeks):
    X = np.load(ROOT / f"initial_data/function_{f}/initial_inputs.npy")
    y = np.load(ROOT / f"initial_data/function_{f}/initial_outputs.npy")
    n_initial = len(y)
    for week in weeks:
        x_line = (ROOT / f"weekly_data/week{week}_inputs.txt").read_text().splitlines()[f - 1]
        y_line = (ROOT / f"weekly_data/week{week}_outputs.txt").read_text().splitlines()[f - 1]
        X = np.vstack([X, np.array(x_line.split(","), dtype=float)])
        y = np.append(y, float(y_line))
    return X, y, n_initial


# The portal's format: six decimals, joined by hyphens. Adding 0.0 turns -0.0 into 0.0,
# which would otherwise print as "-0.000000".
def format_query(x):
    return "-".join(f"{v:.6f}" for v in np.clip(x, 0, 1) + 0.0)


# --- Bayesian optimisation (functions 2-8) ---

# BoTorch's default GP: an RBF kernel with one length scale per input, and priors on the
# length scales and the noise level instead of hand-set bounds (the old bounds, not the
# data, were deciding the length scales). Outputs are standardised inside the model.
# With noise_free, the noise is fixed near zero instead of learned, so the GP passes
# through every observation.
def fit_gp(X, t, noise_free=False):
    Y = torch.tensor(t).unsqueeze(-1)
    Yvar = torch.full_like(Y, 1e-6 * t.var()) if noise_free else None
    model = SingleTaskGP(torch.tensor(X), Y, Yvar, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


# Find the point in the box [lower, upper] where the acquisition function is highest.
# Expected improvement is often ~0 everywhere except a small region next to the best
# observations, which BoTorch's own random starting points can miss (they did for
# functions 2 and 4). So the starting points are the best 32 of: 20,000 random points,
# plus 2,000 small random nudges (by 0.01 or 0.05) of the five best observations.
# L-BFGS-B then climbs from each of them, and the best result wins.
def maximise_acquisition(acq, X, t, lower, upper):
    rng = np.random.default_rng(0)  # fixed seed: the same data always gives the same query
    best = np.repeat(X[np.argsort(-t)[:5]], 400, axis=0)
    nudged = best + rng.normal(size=best.shape) * rng.choice([0.01, 0.05], size=(len(best), 1))
    random_points = rng.uniform(lower, upper, size=(20_000, X.shape[1]))
    candidates = torch.tensor(np.vstack([random_points, np.clip(nudged, lower, upper)]))
    with torch.no_grad():
        scores = acq(candidates.unsqueeze(1))
    starts = candidates[scores.argsort(descending=True)[:32]].unsqueeze(1)
    x, _ = optimize_acqf(acq, bounds=torch.tensor(np.array([lower, upper])), q=1, num_restarts=32,
                         batch_initial_conditions=starts)
    return x.numpy().ravel()


# The trust region's side length this week. It starts at START_LENGTH, then doubles or
# halves for each week since TRUST_REGION_START_WEEK, depending on whether that week's
# query improved the best value so far (by more than 0.1%, as in TuRBO).
def trust_region_length(y, n_initial, weeks):
    length = START_LENGTH
    for i, week in enumerate(weeks):
        if week >= TRUST_REGION_START_WEEK:
            best_before = y[:n_initial + i].max()
            improved = y[n_initial + i] > best_before + 1e-3 * abs(best_before)
            length = min(2 * length, MAX_LENGTH) if improved else max(length / 2, MIN_LENGTH)
    return length


# One Bayesian optimisation step: fit the GP on the function's scale, then query the point
# in the trust region with the highest Log Expected Improvement (LogEI).
# Expected improvement is how much a point is expected to beat the best value so far. It
# weighs a high mean against high uncertainty with no kappa to tune, and LogEI is its
# numerically stable form (Ament et al., 2023). "Best so far" is the GP's highest mean at
# the observed points, so that one lucky noisy reading doesn't set the bar; for a
# noise-free function it's simply the best observed value.
def propose_gp(X, y, transform=None, length=START_LENGTH, noise_free=False):
    forward, inverse, label = TRANSFORMS[transform]
    t = forward(y)
    model = fit_gp(X, t, noise_free)

    # The GP's mean and standard deviation at the inputs X_new
    def predict(X_new):
        with torch.no_grad():
            posterior = model.posterior(torch.tensor(X_new))
        return posterior.mean.squeeze(-1).numpy(), posterior.variance.sqrt().squeeze(-1).numpy()

    acq = LogExpectedImprovement(model, best_f=t.max() if noise_free else predict(X)[0].max())

    # The trust region: centred on the best point, with its sides scaled by the GP's length
    # scales (normalised so that their product is 1, as in TuRBO), and clipped to [0, 1]
    lengthscales = model.covar_module.lengthscale.detach().numpy().ravel()
    half_sides = length / 2 * lengthscales / np.prod(lengthscales) ** (1 / len(lengthscales))
    x_best = X[np.argmax(t)]
    x_next = maximise_acquisition(acq, X, t, np.clip(x_best - half_sides, 0, 1), np.clip(x_best + half_sides, 0, 1))

    mean, std = predict(x_next[None])
    print(f"  trust region side length: {length:.3g}")
    print(f"  GP predicts y ≈ {inverse(mean[0]):.4g} here "
          f"(68% range {inverse(mean[0] - std[0]):.4g} to {inverse(mean[0] + std[0]):.4g})")
    print(f"  length scales (smaller = input matters more): {lengthscales.round(3)}")
    return x_next, t, label, predict


# --- Function 1: radiation-source model ---

# log10 of the size of a reading, ignoring its sign. Floored at 1e-320 so that an exact zero
# (an underflow, i.e. very far from any source) still has a logarithm.
def log_magnitude(y):
    return np.log10(np.maximum(np.abs(y), 1e-320))


# The features [1, x1, x2, x1^2 + x2^2], which make the source model linear (see below)
def source_features(X):
    return np.column_stack([np.ones(len(X)), X, (X ** 2).sum(axis=1)])


# Find the radiation source from the size of the readings, and query its centre.
# A reading falls off like a Gaussian with distance from the source's centre c:
#     |y| = A * exp(-||x - c||^2 / 2s^2)   =>   log10|y| = a - k * ||x - c||^2
# Expanding the square makes this linear in source_features(x), so ordinary least squares
# fits it, and the centre is where the fitted bowl peaks.
# - Only readings within 0.4 of the strongest one are used: the far ones are too strong for
#   this source, so they probably come from a second, weaker one.
# - The sign is ignored: the negative readings show no spatial pattern, so only their size
#   says how close they were.
def propose_source(X, y):
    t = log_magnitude(y)
    near = np.linalg.norm(X - X[np.argmax(t)], axis=1) < 0.4
    coef = np.linalg.lstsq(source_features(X[near]), t[near], rcond=None)[0]
    _, b1, b2, c = coef
    if c >= 0:
        raise ValueError("Function 1's source model no longer has a peak; it needs another look.")
    centre = np.clip([b1 / (-2 * c), b2 / (-2 * c)], 0, 1)
    error = np.std(t[near] - source_features(X[near]) @ coef)

    # The model's log10|y| at the inputs X_new, with its typical error
    def predict(X_new):
        return source_features(X_new) @ coef, np.full(len(X_new), error)

    print(f"  source model fitted to {near.sum()} readings (typical error: {error:.1f} orders of magnitude)")
    print(f"  predicted |y| at the source centre ≈ {10 ** predict(centre[None])[0][0]:.2g}")
    return centre, t, "log10 |y|", predict


# --- Function 7: Hartmann-6 ---

# The Hartmann-6 benchmark in its positive form (so it is maximised), on [0, 1]^6
def hartmann6(X):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])
    X = np.atleast_2d(X)
    return (alpha * np.exp(-(A * (X[:, None, :] - P) ** 2).sum(axis=-1))).sum(axis=-1)


# Where Hartmann-6's global maximum (3.32237) is, from the benchmark literature
HARTMANN6_MAX_AT = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])


# Query Hartmann-6's maximum, after checking that every observation still matches it
def propose_hartmann6(X, y):
    if not np.allclose(hartmann6(X), y, rtol=0, atol=1e-9):
        raise ValueError('Function 7 no longer matches Hartmann-6: set its method to "gp" in SETTINGS.')
    print(f"  all {len(y)} observations match Hartmann-6; its maximum is {hartmann6(HARTMANN6_MAX_AT)[0]:.5f}")

    # The true function, with no uncertainty
    def predict(X_new):
        return hartmann6(X_new), np.zeros(len(X_new))

    return HARTMANN6_MAX_AT, y, "y", predict


# --- Graphs ---

# Graph style: charcoal for the initial data, orange for the weekly queries, a black ring
# around the best observation, green for the next query, and blue for the model
INITIAL, WEEKLY, NEXT, MODEL, INK = "#3c3c3c", "#e8710a", "#1baf7a", "#1565c0", "#111111"
# The 2D maps use the lighter 80% of matplotlib's "Blues", so the markers on them stay visible
MAP_COLOURS = ListedColormap(plt.colormaps["Blues"](np.linspace(0, 0.8, 256)))
BEST = dict(marker="o", s=200, facecolors="none", edgecolors=INK, linewidths=1.8, zorder=5)
PLOT_STYLE = {
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titlecolor": "#222222", "axes.labelcolor": "#222222",
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6, "xtick.color": "#333333", "ytick.color": "#333333",
}
LEGEND = [
    Line2D([], [], ls="", marker="o", markersize=6, color=INITIAL, label="initial data"),
    Line2D([], [], ls="", marker="o", markersize=6, color=WEEKLY, label="weekly queries"),
    Line2D([], [], ls="", marker="o", markersize=11, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.8,
           label="best observation"),
    Line2D([], [], color=INK, lw=1.2, label="best so far"),
    Line2D([], [], color=MODEL, lw=2, label="model mean ± 1 sd"),
    Line2D([], [], color=NEXT, lw=2, label="next query"),
]


# One figure per function, on the scale the model works on (t, e.g. log(y)):
# - History: every observation in order, the best so far, and the model's prediction for
#   the next query.
# - Model mean (2D functions only): a map of the model's mean over the whole input space.
# - One slice per input through the next query, varying that input alone. The dots are
#   all the observations projected onto that input, so they don't lie on the slice.
# The best observation (highest y) is marked in every panel.
def plot_function(f, X, y, n_initial, x_next, t, label, predict):
    d = X.shape[1]
    best = np.argmax(y)
    colours = [INITIAL] * n_initial + [WEEKLY] * (len(t) - n_initial)
    mean_next, std_next = (v[0] for v in predict(x_next[None]))

    # Function 1's graphs show the size of each reading, so its negative readings (which can
    # look strong, but are below every positive one) are drawn hollow
    negative = (y < 0) & (label == "log10 |y|")
    faces = ["white" if neg else colour for colour, neg in zip(colours, negative)]
    edges = [colour if neg else "white" for colour, neg in zip(colours, negative)]
    widths = [1.4 if neg else 0.6 for neg in negative]

    def plot_points(ax, u, v):
        ax.scatter(u, v, s=34, facecolors=faces, edgecolors=edges, linewidths=widths, zorder=3)

    # Every panel shows the range of the observations and of the prediction
    low, high = t.min(), max(t.max(), mean_next + std_next)
    y_range = (low - 0.08 * (high - low), high + 0.08 * (high - low))

    n_panels = 1 + (d == 2) + d
    ncols = 2 if n_panels <= 4 else 3
    nrows = -(-n_panels // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), layout="constrained", squeeze=False)
    axes = list(axes.ravel())
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    ax = axes.pop(0)
    n = np.arange(1, len(t) + 1)
    plot_points(ax, n, t)
    ax.scatter(n[best], t[best], **BEST)
    # The best y so far, drawn on this scale (for function 1 the strongest reading isn't
    # the best one, because it was negative)
    best_so_far = [t[np.argmax(y[:k])] for k in n]
    ax.step(n, best_so_far, where="post", color=INK, lw=1.2)
    ax.errorbar(len(t) + 1, mean_next, yerr=std_next, fmt="o", ms=6, capsize=3, color=NEXT, lw=1.6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(title="History", xlabel="observation", ylabel=label, ylim=y_range)

    if d == 2:
        ax = axes.pop(0)
        g = np.linspace(0, 1, 101)
        x1, x2 = np.meshgrid(g, g)
        mean = predict(np.column_stack([x1.ravel(), x2.ravel()]))[0].reshape(x1.shape)
        image = ax.imshow(mean, origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap=MAP_COLOURS,
                          vmin=y_range[0], vmax=y_range[1])
        fig.colorbar(image, ax=ax, label=f"model mean, {label}")
        plot_points(ax, X[:, 0], X[:, 1])
        ax.scatter(*X[best], **BEST)
        ax.scatter(*x_next, marker="+", s=260, color=NEXT, linewidths=2.4, zorder=6)
        ax.grid(False)
        ax.set(title="Model mean", xlabel="x1", ylabel="x2")

    g = np.linspace(0, 1, 201)
    for i, ax in enumerate(axes[:d]):
        X_slice = np.tile(x_next, (len(g), 1))
        X_slice[:, i] = g
        mean, std = predict(X_slice)
        ax.fill_between(g, mean - std, mean + std, color=MODEL, alpha=0.18, lw=0)
        ax.plot(g, mean, color=MODEL, lw=2)
        plot_points(ax, X[:, i], t)
        ax.scatter(X[best, i], t[best], **BEST)
        ax.axvline(x_next[i], color=NEXT, lw=1.6)
        # The x-axis goes a little past 0 and 1, so that a query on the boundary stays visible
        ax.set(title=f"Slice along x{i + 1}", xlabel=f"x{i + 1}", ylabel=label, xlim=(-0.02, 1.02), ylim=y_range)

    fig.suptitle(f"Function {f}: next query {format_query(x_next)}", fontsize=12)
    legend = list(LEGEND)
    if negative.any():
        legend.append(Line2D([], [], ls="", marker="o", markersize=6, color=INITIAL, markerfacecolor="white",
                             label="negative reading"))
    fig.legend(handles=legend, loc="outside lower center", ncol=4 if ncols == 2 else len(legend), frameon=False)
    return fig


# One figure for all eight functions: the best result so far after each week (week 0 is the
# best of the initial data), with an orange dot for every week that found a new best
def plot_progress(weeks):
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), layout="constrained")
    for f, ax in zip(range(1, 9), axes.ravel()):
        _, y, n_initial = load_data(f, weeks)
        best = np.maximum.accumulate(np.concatenate([[y[:n_initial].max()], y[n_initial:]]))
        week = np.array([0, *weeks])
        new = np.flatnonzero(np.diff(best) > 0) + 1
        ax.step(week, best, where="post", color=MODEL, lw=2)
        ax.scatter(week[0], best[0], s=34, color=INITIAL, zorder=3)
        ax.scatter(week[new], best[new], s=34, color=WEEKLY, zorder=3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(title=f"Function {f}", xlabel="week")
    fig.suptitle("Best result so far, by week (orange: a new best)", fontsize=12)
    return fig


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description=f"Propose the week {WEEK_NO} queries.")
    parser.add_argument("function", nargs="?", type=int, choices=range(1, 9), help="only this function (default: all)")
    parser.add_argument("--plots", choices=["none", "show", "save"], default="none",
                        help=f"no graphs (default), show them in windows, or save them to figures/week{WEEK_NO}/")
    args = parser.parse_args()

    # L-BFGS-B warns when it stops early on a flat surface, which is expected here
    warnings.filterwarnings("ignore", category=OptimizationWarning)
    torch.manual_seed(0)
    plt.switch_backend("TkAgg" if args.plots == "show" else "Agg")
    plt.rcParams.update(PLOT_STYLE)
    figures = HERE / f"figures/week{WEEK_NO}"
    if args.plots == "save":
        figures.mkdir(parents=True, exist_ok=True)

    weeks = [week for week in range(1, WEEK_NO) if (ROOT / f"weekly_data/week{week}_outputs.txt").exists()]
    if len(weeks) < WEEK_NO - 1:
        print(f"Warning: weekly_data/ only has results for weeks {weeks}")

    functions = [args.function] if args.function else range(1, 9)
    queries = {}
    for f in functions:
        X, y, n_initial = load_data(f, weeks)
        method = SETTINGS[f]["method"]
        print(f"\nFunction {f} ({X.shape[1]}D, {len(y)} observations, method: {method})")
        print(f"  best so far: {y.max():.6g} at {format_query(X[np.argmax(y)])}")

        if method == "source":
            x_next, t, label, predict = propose_source(X, y)
        elif method == "hartmann6":
            x_next, t, label, predict = propose_hartmann6(X, y)
        else:
            length = trust_region_length(y, n_initial, weeks)
            x_next, t, label, predict = propose_gp(X, y, SETTINGS[f].get("transform"), length,
                                                   SETTINGS[f].get("noise_free", False))
        print(f"  next query: {format_query(x_next)}")
        queries[f] = x_next

        if args.plots != "none":
            fig = plot_function(f, X, y, n_initial, x_next, t, label, predict)
            if args.plots == "save":
                fig.savefig(figures / f"function_{f}.png", dpi=110)
                plt.close(fig)

    if args.plots != "none":
        fig = plot_progress(weeks)
        if args.plots == "save":
            fig.savefig(figures / "progress.png", dpi=110)
            plt.close(fig)

    print(f"\n--- Week {WEEK_NO} queries ---")
    for f, x in queries.items():
        print(f"Function {f}: {format_query(x)}")
    if args.plots == "show":
        plt.show()


if __name__ == "__main__":
    main()
