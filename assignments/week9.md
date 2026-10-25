### How do scaling laws influence your current query choices? Do you see diminishing returns or steady improvements?

Mostly diminishing returns. Scaling laws say a model improves as a power law of its data, so each extra point helps less than the one before, and I see the same here: most of my best values came in weeks 3 to 5, and since then only function 7 has kept improving. So, with few rounds left, I've moved towards exploitation, searching only near each function's best point. I expect bigger gains from better models than from more points, like the radiation-source model for function 1.

### Where might emergent behaviours alter your expectations, and how are you preparing for them?

Function 1 is the clearest case. Every reading so far is close to zero, but a query near the source could jump by many orders of magnitude at once, the way an LLM capability appears suddenly at a certain scale. If that happens, the readings so far stop being a good guide: one query would take its best from about zero to a real signal, and the next queries would refine around the source instead of looking for it. To prepare, I model the readings on a log scale, so a jump like that doesn't break the model, and I have a plan B if the reading comes back negative: the second, weaker source the data points to.

### What trade-offs between cost, robustness and performance are shaping your strategy now?

The cost that matters is queries, not compute: the script runs in seconds, so I can afford to test my choices properly before spending a query. Robustness now matters more than squeezing out the last bit of predicted performance. I checked that no query depends on a single observation, and for function 8 I traded a slightly higher predicted gain further away for a query near the best point, where the model is more reliable. I also replaced my hand-tuned length-scale bounds with the GP's default settings, because the bounds, not the data, were deciding the model.

### How do you balance predictable optimisation with the risk of sudden but uneven emergent capabilities?

I decide per function. Where the model predicts well and the gains are small, like functions 4, 6 and 8, I optimise steadily near the best point. Where a sudden jump is possible and a miss costs nothing, I take the bolder query: function 1's query goes straight to the predicted centre of the source, because its best so far is effectively zero, so there's nothing to lose. With one query per function per week, no gradients, and no way to test a point before submitting it, this keeps the risk where it's cheap and the steady gains where the model is reliable.
