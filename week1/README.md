## Function 1

This function is really difficult to analyse, because points are orders of magnitude apart from each other. Also, from the datapoints and the description, one could assume the function is very spiky, i.e. it has significant, elevated values at very specific points but it quickly decays, but that assumption might be wrong. More info in the description would be helpful.

With the above assumption in mind, the best approach would be exploration. However, to confirm the function is indeed spiky, not smooth, I choose a point close to the highest and towards an area where there are no other points with lower values. If the value is also high, then I would assume there's a region with elevated values, which I will exploit further. If the value is low, then it confirms the function is spiky, and that the current highest value is surrounded on all sides by low values. Thus, exploration would be the way.

**Query: 0.8, 0.8**

## Function 2
This function wasn't very challenging. It seems to have two regions of high values, with two other potential regions, which haven't been explored much. The function seems to be smooth enough. There are two options: exploit close to the existing maximum, or explore the other potential regions. I've used GP with a high Kappa value of 3, and even then it preferred to do exploitation, so I'll go with the value it has provided. I think that regardless of the output value of the next query, I would explore those two regions next.

**Query: 0.84848485, 0.97979798**

## Function 3
From this point onwards, dimensions are higher than 3, and with the low number of points provided, we could say that there's not enough data to analyse the functions, and exploration is much more desirable than exploitation.

The function has a point with a very low value, but apart from it, the rest is very uniform, which makes it difficult to analyse.
If I knew nothing about the function, I would believe it's just noise or the flat part of the function. However, since we're talking about side-effects in medication, one could assume the function by itself is going to have very little variance. Whatever the case, the best approach is exploration. I went for a zone in space that hasn't been explored much. The next queries would likely be explorative, unless the output of this query would be singificantly higher than the rest.

**Query: 0.5, 0.95, 0.95**

## Function 4
This function's description mentions that is has many local maxima, making it difficult to analyse. Thus, exploration seems to be the best approach for now. The next queries would likely be explorative, unless the output of this query would be significantly higher than the rest. I chose a point in a region that hasn't been explored yet.

**Query: 0, 0.33, 0.95, 0.95**

## Function 5
The function description mentions that the function is likely unimodal, i.e. has a single peak, and thus there should be no local maxima. Therefore, the best approach should be exploitation. We can also see from the values that there is clearly a stand-out region. I've chosen a point close to the existing maximum, and compared that to the result of GP with a reasonable Kappa of 2. The next queries would also be exploitative.

**Query: 0.22222222, 0.88888889, 0.88888889, 0.88888889**

## Function 6
This function is very similar to function 4 from the output values. Since it's baking, one would expect a significant high value at a point where all the ingredients have the correct ratio, but the highest value is already only 4 away from the potential maximum, indicating that it also has many local maxima, likely from differences in taste, which can create noise if there are multiple tasters providing training data. Thus, I'll use exploration for the next few queries.

**Query: 1.0, 0.0, 1.0, 0.0, 1.0**

## Function 7
This function is also quite challenging, not just from the fact that there are 6 dimensions and very few datapoints to work with, but because there's already a point with a value much higher than the rest, so exploration vs exploitation is difficult. By varying Kappa in UCB between 2 and 3, one gets two different points. With Kappa=2, the acquisition function would go very close to one the points from the second layer of maxima, making me think that the value for Kappa was still too small. With 3, the chosen point would be closer to the current maximum, but still exploring a bit. I decided with the second. Why? I suspected the RBF length scale was too small. It was set at 0.1, making the predicted function very spiky, with a high gradient near the point previously chosen for Kappa=2. When I changed the length scale to 0.2, then both Kappa=2 and Kappa=3 gave the same result, the one that previously only Kappa=3 would give. So, in the end, the approach used was exploitation. The next step would heavily depend on the result of the next query, between exploration and exploitation. If a high value is still observed, then exploitation, otherwise, exploration.

**Query: 0.0, 0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.66666667**

## Function 8
This function is very similar to functions 4 and 6, with many local maximima. Thus exploration was the chosen approach, with a datapoint in a previously unexplored area.

**Query: 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5**

---
### Which function(s) were most challenging to query, and why?
From function 5 onwards analysis was difficult because of dimensionality. Not only one has to choose Kappa carefully, but also length scale. It's also more difficult to have an intuition of what to do next, and I'll have to rely more on GP. ## Function 1 was also very difficult because the shape of the function is almost arbitrary; it likely uses proximity to datapoints that an actual function.


