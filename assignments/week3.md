This week I've changed 3 major things in the approach.

### Normalisation
The first is normalisation. When running the models for some functions I've noticed that the acquisition function's proposed point was always an exploration one, no matter the value of Kappa I would set for UCB. I then noticed that this would happen for functions that had negative values. Some are minimisation problems whose values are negated to convert them into a maximisation problems. Because the GP model would assume the value 0 in unexplored regions, both the mean and the standard deviation were highest in those regions, leading to no exploitation whatsoever.
Once I started normalizing the y when constructing the model, then query proposal were much better and controllable.

### Gaussian Processor Kernels
The second thing I've changed was the kernels used in the GP model, and their parameters.
For one, I've started using a constant/scalar kernel alongside the RBF kernel I was already using.
For the RBF kernel, I started using one value per dimension, which allows one to controll the significance of each dimension on the output, which previously wasn't happening.
Finally, I've allowed the model to adapt the length scale parameters, rather than let it as fixed, which also helps the model adapt to the input datapoints. I've still set bounds to the length scale parameter, since the model can very easily overfit some dimensions and underfit others given the low amount of input data points.
I've also allowed the model to perform several starts, reusing the kernel parameters from previous runs.

### Finding the maximum of the acquisition function
Previously, I was still evaluating the acquisition function based on a uniform grid.
This had a problem. With more dimensions, the grid would become really sparse. E.g.: because I was only evaluating around 1M points, evenly distributed across dimensions, it meant that the last function was only being evaluated on 3 points per dimension: 0, 0.5, and 1. In previous weeks, I assumed this had to do with a higher value of UCB, which was tending towards exploration, but I've only realised now this wasn't the case.
Instead, I've used a gradient descent method to find the maximum of the UCB function.
I start with 500 uniformly distributed samples, which somewhat equates to exploration. From those I choose the 50 best and run the gradient descent method from there.
Because the UCB function is quick to evaluate and much smoother than the actual function, given the low number of input datapoints, grandient descent is very effective without much work.
This change has resulted in much better next-query proposals.

---

### How has your query strategy changed from earlier rounds? Do you rely more on model predictions, or are you still exploring new regions? Do you tune hyperparameters or rely on heuristics?
For this week I'm relying much more on the model predictions, especially since I've changed how the model and acquisition function works, as mentioned above.
I've still controlling the Kappa of the UCB acquisition function and see how that affects the predictions, but I'm not manually analysing the datapoints.

### How do you balance exploration against exploitation? Do you focus more on areas known to perform well, or are you still sampling from untested regions?
On some functions I'm still leaning a bit more towards exploration, like function 3, because there is no clear area of a global maximum yet, but overall this week has been more focused on exploitation.

### How would SVMs change your approach? Could you use a soft-margin SVM to classify high vs low performance regions? Would a kernel SVM help if the response surface is non-linear?
SVMs wouldn't change the approach much, as they were not built for the task at hand. Like logistic regression, it could be usefull to determine a boundary through space that would separate noise, or low values, from potential maxima, but it wouldn't be sufficient on its own.
Given the highly non-linear functions that were are provided with, the most likely kernel used would be the RBF one (already used in the GP model), which could provide us with meaningful results.

### What limitations of your current model become apparent as data grows? Is it overfitting? Do any features or dimensions emerge as irrelevant?
This week I noticed that the model did have some limitations, as mentioned before. The normalization part was just plain wrong, not related with the model. The acquisition function maximisation is also not related to the model, although it did affect the proposed points. However, the change in kernels were very useful. By controlling the length scale per dimension, I've managed to prevent both underfitting and overfitting, depending on the function and dimension. So far, I haven't seen a dimension become irrelevant, as opposed to my gut feeling last week, but that might become aparent, with more samples.

### How does this black-box set-up prepare you to think like a data scientist when faced with incomplete knowledge in other projects?
This set-up is really really useful for when access to data is difficult. During lessons and tutorials we often work with problems where we can evaluate the unknown function quickly, and therefore we gain a lot of knowledge about it very quickly. However, in industry, this may not be the case, or even if it is, it comes with high infrastructure costs. When optimisating neural networks for example, changing hyperparameters and re-training the model is very expensive. Thus, exercises like this Capstone one is very useful to think about ways to rely less on samples from the actual function, or extract as much information as possible from what's avaiable.