### How has your optimisation strategy evolved since your first few rounds of queries? Which elements now feel more structured or systematic?

In the first rounds I tuned kappa and the length scales by hand every week, and mixed in judgement from plots. Now everything is in one script: a table of per-function settings, the GP's default priors instead of hand-set bounds, Log Expected Improvement instead of UCB, and a trust region that grows or shrinks on its own. The decisions I used to make by hand each week are now rules, so the same data always gives the same query.

### If you think of your current data set as a ‘high-dimensional’ space, which variables or behaviours seem to drive the largest variation in your results – similar to principal components in PCA?

Within each function, the GP's length scales play the role of principal components: a few inputs drive most of the variation, like x1 in function 2, x3 in function 3 and x5 in function 7, while others barely matter, like x8 in function 8. Across all functions, the biggest driver of my results has been how close a query was to the best region found so far. Queries far from it, at high kappa or on the edges, produced most of my worst values.

### How do you decide which aspects of your strategy to keep exploring versus which to reduce or simplify, as PCA reduces dimensions while retaining essential information?

I keep what has evidence behind it and drop what doesn't. I tested each choice on my data, by how well the model predicted points it hadn't seen and how stable its queries were, and removed what added complexity without helping: the hand-tuned bounds and kappa. I didn't adopt alternatives like SVRs or neural networks either, because they predicted worse. The trust region applies the same idea to the search itself, shrinking where queries stop improving.

### How might this round of optimisation influence your next and final round of query submission in Module 24, especially when balancing exploration and exploitation?

In the final round there's no future round to benefit from exploring, so every query should have a real chance of beating the best value. Since I'm keeping the same code, the trust region already moves in that direction: functions that keep failing get smaller boxes, so their queries become local refinements. That makes this round the last one where a deliberate risk is worth taking, like function 1's query at the source.

### Reflect briefly on how insights from PCA, such as focusing on variance and removing redundancy, might apply to how you interpret your BBO results.

Focusing on variance means reading my results along the inputs that matter: the length scales tell me which slices of my graphs are worth looking at. Redundancy matters too: many of my later queries sit close together near the best points, so they add less new information than their number suggests, much like correlated features that PCA would merge. And when a function's results barely vary any more, like function 8's recent plateau, the remaining gains are probably small.
