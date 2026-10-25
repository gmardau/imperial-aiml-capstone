### What reasoning guided your submission for this tenth round? Explain your strategy. How did patterns from the previous rounds influence your decisions for each function?

I kept the method I set up in week 9 and made one change: the search box around each function's best point now adapts, as in TuRBO. It doubles after a week whose query improved the best value, halves after one that didn't, and is stretched along the inputs the GP thinks matter less. Earlier rounds shaped each function's set-up: function 1's readings fall off with distance from a source, so I query the source's centre; function 3 is approaching its ceiling of 0, so it's modelled on −log(−y); and function 5 keeps rising towards a corner, so it's modelled on log(y). Since most of my best values came in weeks 3 to 5, every function now leans towards exploitation.

### How transparent is your decision-making process? If another researcher reviewed your notes and data, could they follow your logic and reproduce your strategy? What information would they need to fully understand your approach?

Quite transparent now. Every choice is in one script, with a table of per-function settings and comments explaining each one, and fixed random seeds, so the same data always gives the same query. A reviewer would need the initial data and weekly results, the function descriptions, the script and the pinned environment, all of which are in the repository. The weakest part is the early rounds: weeks 1 and 2 relied partly on judgement from plots, which is documented but can't be reproduced exactly.

### What assumptions are you making in your search/optimisation strategy? Identify at least one key assumption related to the functions or the optimisation process. How might this assumption shape or limit your results?

The key one is about function 1: that the sign of a reading is noise, and its size depends only on the distance to a single source. That's what lets me locate the source from readings that are almost all zero, but if the strongest reading actually comes from a negative bump, the query lands in the wrong place.

### Where do you see gaps or potential biases in your data set? Consider the distribution of your queries, unexplored areas or patterns in how you sampled the search space.

My queries are heavily clustered. Recent rounds concentrate near each function's best point, while earlier rounds with a high kappa pushed many queries to the edges of the search space (values of 0 or 1). Large regions in between are barely sampled, especially in 6 and 8 dimensions, where fewer than 50 points cover a tiny fraction of the space. There is another gap in function 1: weaker source near (0.42, 0.42) is barely explored.

### What is one significant limitation of your approach? Consider factors such as computational constraints, sampling biases or assumptions about function behaviour that might affect the validity or generalisability of your results.

With one query per week, every decision rests on a model fitted to fewer than 50 points. The GP predicts well on smooth functions like 4, 6 and 8, but poorly on function 7, so it can be confidently wrong where the function has narrow peaks. The approach is also tailored to these eight functions, like function 1's source model, so it wouldn't generalise as it is.
