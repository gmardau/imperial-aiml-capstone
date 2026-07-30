## Function 1
This function is still the most difficult to analyse. The point from the previous week, somewhat demonstrated that there's no maximum near 1.0-1.0. And that the current maximum is bounded by all sides.
Therefore, I will go for more exploration this time.

**Query: 0.05-0.95**

## Function 2
The point from the previous week also delimited the high value region close to the existing maximum. Given that there's already a concentration of samples there, I've decided that best approach would be to choose a point closer to the other potential region, but still within an area not that well explored. GP validated that choice as well. If the value is high, I will explore it further. If not, I will explore the area of low x0 and high x1.

**Query: 0.7-0.3**

## Function 3
I've continued with the same approach as last week and went for more exploration, since the new point did not present an improvement.

**Query: 0.95-0.05-0.75**

## Function 4
The last query resulted in a very low output, so, I will explore the function space further. One thing that I noticed however, is that there's a linear trend between x4 and y. The lower x4 is, the higher y is. I saw this by plotting every dimension of X against y. Therefore, I will explore further in that area. Fortunately, GP agreed with me.

**Query: 0.5-1.0-1.0-0.0**

## Function 5
The last query returned a new maximum, and because the description mentions the function is unimodel, I will continue with the exploitation approach, going even further from the last query.

**Query: 0.25-1.0-1.0-1.0**

## Function 6
Similarly to function 4, the result of the query was low. And also similarly to function 4, there seems to be a linear trend between x4 and y, and between x5 and y. Similarly, I will explore the function further, with an empashis on a high value of x4, and low value of x5.

**Query: 0.0-0.0-0.0-1.0-0.2**

## Function 7
The query from the previous week resulted in a very high value, therefore, I would opt for continuing exploitation. Using the same length scale and Kappa from last week, BP also resulting in a new query close to the maximum.

**Query: 0.0-0.5-0.5-0.5-0.5-0.5**

## Function 8
Similarly to functions 4 and 6, I managed to see a linear relationship between x1 and x3 and y, where higher values of x1 and x3 result in lower values of y. It just so happens that the query from last week included a very low value for x1 and a high y value. I will continue exploring the function using BP as a guide, by overriding the suggested value of x1 from 1 to 0.05.

**Query: 1.0-0.0-0.0-1.0-0.0-1.0-1.0-1.0**

---

### What was the main change in your strategy in this week compared to the last week? What prompted this change? Was it the model predictions, acquisition function behaviour or something else?
The observed outputs have definitely changed the strategy, specially in function 2, where I would expect a global maximum near the current maximum, but that ended up not being the case. I shifted to exploration instead.
Another thing that I changed was to tinker with the values of length scale and UCK Kappa depending on the function and its description, which would drastically change the acquisition function.

### Did you focus more on exploration (sampling uncertain areas) or exploitation (targeting promising areas)? Why? What trade-offs did you weigh?
It depended on the function. Some present clear areas that need more exploitation, such as function 5, the unimodal one. Other seem to be just noise, or have a lot of local maxima, needing more exploration.

### Have any participant strategies, class discussions or recent outputs influenced how you approached this week's submission?
Yes, some participants decided to perform a more manual analysis, while others relied heavily on acquisition functions. I decided to strike a balance between the two. Rely more on instinct in lower dimension functions, and more on the results of the acquisition function for higher dimension functions.

### If you were to fit a simple linear or logistic regression model to your current data for one of the functions, which assumptions would you most likely violate? Consider aspects such as the shape of the response surface, noise levels or number of features.
The assumption that would very easily be violated is that of linearity. The presented functions are clearly non-linear, some of them even described as black-boxes, with many local maxima and minima, with varying degrees of gradients.
Homoscedasticity would also likely be violeted, since some functions, especially function 1, seem to assume values with very different orders of magnitude, and noise is likely not the same accross the function space.

### Are there any regions where the output appears roughly linear or where a decision boundary might form? How might a logistic regression classifier perform on this function, particularly in binary or threshold-based scenarios?
Some regions of functions do appear linear, yes. Either on flat areas or steep gradients. However, this is only locally. Logistic regression could be usefull to determine a boundary through the function space that would separate noise, or low values, from potential maxima, but it wouldn't be sufficient on its own, since we want to find the concrete maximum.

### Interpretability is a key advantage of linear and logistic regression. Did you find it useful to consider individual feature effects before deciding on your query point?
Yes. Because of this question, I started to look into correlation between each of the dimensions and the output value. On three occasions, functions 4, 6, and 8, this helped me choose the next query points. For now, I used only plots, but in the future I will try and use better mathematically models (p-value) to determine correlations.