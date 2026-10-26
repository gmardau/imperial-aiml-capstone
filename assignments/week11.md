### How have patterns in your past queries influenced your latest choices?

The clearest pattern is diminishing returns. Most of my best values came in weeks 3 to 5, and of the high-kappa queries since then, many of them pushed to the edges of the space, rarely improved anything. That's why my latest choices focus on each function's best region, inside a trust region that grows when a query improves and shrinks when it doesn't. Past results also shaped individual functions: function 1's readings fall off with distance from one point, so I query that point, and function 5 kept improving towards a corner, so I search near it.

### Have you identified any ‘clusters’ or recurring regions in your search space that seem promising? (Explain your reasoning – whether you have identified such regions or not.)

Yes. For functions 1, 4, 6, 7 and 8, the five best observations sit very close together. Functions 2 and 3 are different: their best points form a band rather than a blob, so only one input really matters there. Part of this clustering comes from my own sampling, though: I've queried near the best points so often that they were bound to cluster.

### Which strategies or parameter choices have proven less effective, and how are you adjusting for them?

High UCB kappas, up to 5, were the least effective. They sent queries to corners and edges, where the results were often among the worst I've seen. Hand-set length-scale bounds were another: the fitted values kept hitting them, so the bounds rather than the data shaped the models. I've replaced both with Log Expected Improvement, which needs no kappa, the GP's default priors, and a trust region that adapts to each week's result.

### In what ways do your refinements parallel how clustering algorithms separate meaningful patterns from noise?

The trust region works a bit like clustering: it focuses on the dense region of good points and treats isolated results elsewhere as not worth chasing. Function 1 is the most direct case: I treat the sign of each reading as noise and keep only its size, which is the part that follows a pattern.

### If your query results were plotted, what trends or groupings might appear? How could these inform your next iteration?

My graphs already show this. The weekly queries cluster around each function's best point, while the initial data is spread out, and the history shows steep gains until week 5 followed by a plateau. Function 1 has the clearest grouping. If a function's trust region keeps shrinking without improving, that's a sign its region is exhausted, and the next step would be to cluster the good points to find a second promising region to try.
