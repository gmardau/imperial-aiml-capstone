### Which hyperparameters did you choose to tune, and why did you prioritise them?

In my model, the hyperparameters of the BP model I tuned were the kernel length scale parameters and the UCB acquisition function Kappa. These give me control over the shape of the function based on the description and also how explorative or exploitative I may want to be on a function based on its shape description.

### How has hyperparameter tuning changed your query strategy compared to earlier rounds?

The strategy itself hasn't really changed much. The general idea is to always adjust Kappa to be more exploitative as rounds go by. However, new data from the queries does change one's perspective of the function, its shape, how it behaves over the entire search space, and dependencies between dimensions. Furthermore, on such a low amount of points, BP models can easily overfit if length scale values are not properly bound, so that is also a consideration to take every week.

### Which tuning method(s) did you apply (manual adjustment, grid search, random search, Bayesian optimisation, Hyperband), and what trade-offs did you notice?

I've applied a BO method, with a hint of manual adjustment. Grid search and Random Search are not feasible given that we can only submit one query per week and have a small amount of queries possible. I would really like to try Hyperband on this problem, but again, we don't have availability of new queries. The BO method is the base, and whenever I feel like the query is not good (e.g. the suggested query includes values of 0 or 1, the edges of the search space), or when even a high Kappa is not explorative enough (e.g. function 5, where I think I already found the maximum).

### As your data set grows to 16 points, what limitations of your model become clearer through tuning (e.g. overfitting, irrelevant features, diminishing returns)?

I think the one I still see a lot, and is difficult to adjust is the overfitting. When allowing the model to find the best Length Scale values, I think it overfits rather quickly, and determines that one variable is much more defining of the function shape than another. I found that in function 2. One of the dimensions has more influence over the shape of the function locally, but in previous rounds, the final length scale value was too low, leading to a jagged function shape. I manually adjusted the length scale bounds and the output of the query confirmed it. Now, the BO model with the initial length scale bounds doesn't overfit as much.
I've also noticed diminishing returns on several functions. On function 5, I think I reached the maximum value, but on others, I think the BO model is a bit too stuck into exploitation, even with high Kappas. Because of the number of dimensions, it is difficult for a human to understand whether the model is doing precisely that.

### How might you apply hyperparameter tuning techniques to larger data sets in future rounds of the BBO capstone project submissions or more complex models in future ML/AI projects?

I would definitely like to try ensembling techniques and/or Hyperband techniques. The first might still be possible to implement in future rounds, but Hyperband is unlikely. Maybe in other projects beyond this course.

### How does tuning in this black-box set-up prepare you to think like a professional ML/AI practitioner in real-world contexts with incomplete information?

This capstone project is especially good because it limits the amounts of queries that I can do. This forces me to think outside the box and experiment with other types of surrogate models and techniques. It also forces me to understand the description of the function and read into the source of the data, allowing me to have a broader understanding of other fields and subject matters (e.g. how radiation works for function 1, or cooking for function 2). However, I also feel it is too limited, and doesn't allow one to experiment with more complex models that require more datapoints to start.