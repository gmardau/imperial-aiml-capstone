### Initial codebase: how did you choose your starting codebase? Was it drawn from a public library, taken from previous work or built from scratch? Why did you select it? Please include a link to your codebase if it is publicly available so that your peers can explore your work directly.

I built it from scratch on public libraries. In week 1 I used scikit-learn's Gaussian process (GP) with an RBF kernel and UCB, the simplest way to get a model with uncertainty working on so few points. In week 4 I moved to BoTorch, which is built for Bayesian optimisation and has better GP fitting, acquisition functions and optimisers. Every week's script is kept, with a notebook walking through the final method.

### Code modification: what changes did you make week by week? Why did you make those adjustments, and how did they affect your results? Which changes had the most significant impact?

- **Weeks 1–3:** I tuned the length scale and UCB kappa per function by hand, then added output normalisation, one length scale per input and gradient-based search.
- **Weeks 4–8:** BoTorch, a Matérn kernel, hand-set length-scale bounds and kappas up to 5. Weeks 3 to 5 brought most of my early gains, but the high kappas from week 6 sent queries to edges and corners, and weeks 6 to 8 barely improved anything.
- **Week 9:** I rewrote everything. BoTorch's default priors replaced my bounds, which the fitted values kept hitting. Log Expected Improvement replaced UCB, and the search was limited to a box around each best point. Function 1 got a radiation-source model, since its readings were almost all zero, and functions 3 and 5 were modelled on log scales.
- **Week 10:** the box became an adaptive trust region, doubling after an improvement and halving after a miss.
- **Weeks 11–13:** the same code, plus eight queries I set by hand where nearby readings contradicted the model.

The week 9 rewrite had the biggest impact. Function 1 went from no signal to 1.99 in one query, and that week gave four new bests after three weeks with almost none.

### Final result: how did your score evolve in the final weeks? If you had more time or a fresh start, what would you do differently?

The last five weeks gave 14 new bests: four in week 9, two in week 10, three in week 11, one in week 12 and four in the final round. Every function beat its initial data:

- **Function 1:** ≈ 0 → 2.00
- **Function 2:** 0.611 → 0.718
- **Function 3:** −0.035 → −0.0007
- **Function 4:** −4.03 → 0.571
- **Function 5:** 1,089 → 8,662
- **Function 6:** −0.714 → −0.153
- **Function 7:** 1.365 → 3.182
- **Function 8:** 9.598 → 9.989

With a fresh start, I'd repeat one query per function early to measure noise, and start with Log Expected Improvement and a trust region instead of hand-tuned UCB. I'd also check every week whether results fell inside the model's predicted range, and keep a query or two per function to look for other peaks.

### Trade-offs and decisions: what were the most significant trade-offs you faced? How did you balance exploration and exploitation or short-term improvements versus long-term strategies?

Exploration versus exploitation was the main one, and I got it wrong at first: high kappas explored edges that were rarely good. Log Expected Improvement inside a trust region balanced it better, exploiting near the best point and widening only after a success. In the final round I only exploited, with small steps close to points already measured.

Short-term versus long-term showed up in queries that couldn't raise my score directly. For function 5, I tested each direction away from the (1, 1, 1, 1) corner and repeated it. That confirmed the corner as the maximum and showed the function is noise-free, so no later query was wasted there.

Function 7 matches the Hartmann-6 benchmark, and using its known maximum would have scored better. But it would have defeated the purpose, so I treated it as a black box. Setting queries by hand was a trade-off too. It broke my rule that the same data always gives the same query, but it improved four functions, so I kept it and documented each one.

### Learning and application: what was the most important lesson you learned from the BBO capstone project? How might you apply these lessons to future competitions or to real-world ML projects in your field?

Knowing how much to trust each result mattered more than which model I used. The GP's 68% ranges held only about a quarter of my results, function 4's noise-free model overshot three weeks running, and function 5 turned out fully deterministic. Swapping the kernel barely changed anything; the noise assumption changed a lot.

In real projects, like tuning hyperparameters or running experiments, I'd measure noise early with repeats and check that a model's uncertainty is calibrated, not just its accuracy. I'd also prefer tested defaults to hand-tuning, and keep every decision reproducible, including the manual ones.

### Looking back, what surprised you most about your process, your results or your fellow participants' strategies?

How much my hand-tuning hurt: replacing it with tested defaults improved more in one week than the three weeks before.

Johan's repository (https://github.com/jhnlandman-sudo/CapStone) reached similar conclusions by a different route. It classifies each function as solved, noisy, still improving or converged, and its early repeats showed functions 2, 3 and 6 were noisy. We both ended up fitting local curves near the peak, and both found function 5's corner and function 1's source. On best values (Johan's repository has results up to round 12), Johan's functions 2, 4 and 6 beat mine, and mine beat Johan's on 3, 7 and 8, so no single strategy won everywhere.

---

### Which optimisation strategies led to your strongest results, and why were they effective? How did these strategies influence your decisions as the challenge progressed?

Three strategies did most of the work:
- **Modelling each function's structure.** Function 1's radiation-source model took it from no signal to 2.00. Modelling function 3 on −log(−y) brought it within 0.0007 of its ceiling at 0.
- **Log Expected Improvement inside an adaptive trust region.** It gave 14 new bests in the last five weeks, by focusing queries near each best point and shrinking where they stopped paying off.
- **Curves through readings near the peak,** in the final weeks. Function 1's curve predicted 1.99937 and returned 1.99938, and function 6's gave a 34% jump.

They worked because they used what I knew from the brief and the data, instead of exploring blindly. The late fits also interpolated between measured points rather than extrapolating. After week 9 I stopped hand-tuning and let the trust region show where to keep pushing. I only overrode the code where nearby readings clearly contradicted it.

### In your view, what qualities define a ‘successful’ strategy – is it outcomes alone or also adaptability, reasoning or efficiency?

Outcomes matter, but with 13 noisy queries they're partly luck: my best on function 2 was probably a lucky reading. A successful strategy also needs:
- **Reasoning** you can explain and reproduce;
- **Adaptability:** changing course when the evidence says so, as my week 9 rewrite did;
- **Efficiency:** every query either improves the result or settles a question.

A strategy that wins by luck can't be repeated, but one that's reasoned and documented can be improved.

### How could the strategies you identified be applied or adapted to professional ML/AI projects beyond the BBO capstone project?

- **Hyperparameter tuning:** Bayesian optimisation with tested defaults and a trust region suits a few hyperparameters on a tight compute budget.
- **Experiments:** measuring noise with repeats first, and checking predicted ranges against results, applies to A/B tests and lab work.
- **Domain knowledge:** building known structure into the model, like bounds or physics, beats a generic model, as function 1's source model did.
- **Reproducibility:** seeded scripts and documented manual decisions make the work auditable.

In real projects a bad query has a cost, so I'd add constraints, and batch queries when evaluations can run in parallel.

### What successful strategies did you notice in your peers’ approaches, and what made them effective? Do you see overlap with your own strategy? Explain your reasoning.

Johan's approach treats each query as an experiment. Every round, each function is classified as solved, noisy, still improving or converged, and every query comes with a written prediction. Early repeats revealed three noisy functions, which stopped curve fits to noise. Single-coordinate moves closed with a quadratic made each result attributable and accurate (function 4: 0.7431 predicted, 0.7440 returned).

There's a lot of overlap. We both used GPs, moved from exploration to local exploitation, tested function 5's corner and fitted curves near the peaks at the end. The main difference is timing: Johan measured noise early, while I only learned function 5 was deterministic in week 12. Johan's one-coordinate moves were more reliable, but my joint trust-region steps gained more in six and eight dimensions (3.18 against 2.39 on function 7).

### What suggestions or perspectives could strengthen your peers’ strategies, and how do their reflections broaden your view of what success means in optimisation?

- **Function 3:** it can't exceed 0, so modelling −log(−y) might reveal a sharper peak than "flat and noisy". Mine reached −0.0007, against −0.025.
- **Functions 7 and 8:** joint steps inside a small trust region could move faster than one coordinate at a time, without the extrapolation misses.
- **The policy:** turning the classification rules into code would make it reproducible without judgement.

Johan's reflections also broadened my view of success. On function 2, Johan's repeats averaged 0.608, well below the best single reading, and my 0.718 is probably just as lucky. A best value you can trust and reproduce says more about a strategy than a higher one you can't.
