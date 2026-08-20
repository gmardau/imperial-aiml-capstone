### How did the ideas of hierarchical feature learning influence the way you thought about structuring or refining your optimisation strategy this round?
My setup and strategy doesn't have layers at the moment, one because I didn't use neural networks, and because I haven't implemented other models.
However, if I were to think in layers, I could think of 1) an initial model that finds the correlation between inputs and drops redundant ones, 2) a second layer using an SVM that restricts the search space to promissing regions, and 3) the current GP surrogate.

### You saw how breakthroughs such as AlexNet and ImageNet classification reshaped expectations in AI. What parallels do you see between those leaps in performance and the incremental improvements you make in your capstone submissions?
What I saw in AlexNet breakthroughs is that we don't always need completely new models to achieve big changes. Existing models can still work, but they may have restrictions that prevent them for having a good performance. In the case of AlexNet, it was mostly a lack of data. Also, existing models sometimes just need some slight adaptations to perform better. In the case of AlexNet, this was the dropout feature to prevent overfitting, or replacing the sigmoid activation function with ReLU. In my case, the could be using a different kernel, or different length scales bounds, or different acquisition functions.

### When training neural networks, people often weigh trade-offs between depth, complexity and training efficiency. Did you encounter similar trade-offs in deciding whether to explore widely or exploit known promising regions in your queries?
Yes, this is a good parallel between neural networks and the GP surrogate that I'm using. Depending on what data scients perceive of the data, they will adjust the configuration of the neural network. More depth or capacity provides a better fit (e.g.: more exploration of the function space) at the cost of efficiency and risk of overfitting with limited data. I face the same dilema with length scale bounds and the UCK kappa. With such a small amount of datapoints and limited queries, do I focus more on exploration or exploitation, depending on the description of the data.

### Reflecting on the building blocks of neural networks (inputs, activations, loss, gradients, weight updates), which of these concepts helped you think differently about how your model learns from the data you’ve accumulated so far?
Some of the building blocks directly relate to GP surrogates. In the end, both models are trying to do the same. They use new training data to adjust weights in neural networks, or the predicted model and uncertainty in GP surrogates. GP surrogates also require it's own form of loss function, when trying to maximime the acquisition function. The differences are on the type of data that can be used to train these models, how complex the function are thought to be, and how difficult it is to test a new query.

### Module 16 also introduced PyTorch and TensorFlow as different frameworks for building and scaling models. If you were to frame your current optimisation approach in terms of a ‘framework’, would it be closer to rapid prototyping and flexibility or to structured, production-ready design? Why?
I started with more of a rapid prototyping approach, testing new models, activation functions, length scale bounds, kernels, etc.. The more I learned, the more I settled on a model that I think suits the problem best, and I organised the code in a way that resembles more a production-ready design. From now on, I would only make significant changes to the codebase if I were to use a new model altogether, not just slight changes to the existing one.

### In the guest interview, Giovanni Liotta discussed industry applications of deep learning in sport. How might reflecting on real-world deep learning use cases inform the way you benchmark success in your own capstone challenge?
The interview with Giovanni Liotta demonstrated that in the real world, the success of a product (or model) is not defined by success in perfect conditions, with well treated data, but by its robustness to poor or incomplete data. More often that not it's better to not bad with any kind of data, even if on average the model is worse, that to be really good with some specific kinds of data. In my case, the model that I use should be reasonably good for all functions, and not computationally heavy, rather than excelling at one or two functions that are not super complex, or have less noise.

---

### Repository structure
#### How have you organised your repository so far (e.g. data, notebooks, queries, results)?
In my repository I have the main README with a description about the project and the strategies I have adopted.
I have the provided initial data in a folder.
I then have another folder with a file for each week's inputs and another for the outputs.
Finally, I have a folder per week, each with the week's python script, so that I can easily compare the versions throughout the weeks, without having to rely on git history, and a README file with the answers to the capstone weekly questions.

#### What changes will you make to improve clarity, navigability and reproducibility?
I don't see myself changing the structure too much. It may be a structure that induces some duplication, but that is often not a problem.

### Coding libraries and packages
#### Which libraries or frameworks (e.g. PyTorch, TensorFlow, scikit-learn) are central to your approach?
As of week 5 I'm using two GP surrogate implementations, one from scikit-learn and another from PyTorch (BoTorch). I also use these libraries to optimise the acquisition function, using gradient descent methods. Going forward I might drop the scikit-learn GP surrogate altogether, since the PyTorch one appears to provide better results. I might also use these libraries to complement the GP surrogate with an SVM, or even a neural network, if it ever becomes viable.

#### Why are these choices appropriate for your problem, and what trade-offs did you consider?
Initially I used scikit-learn. It was the one provided in the course for GP surrogates. It was simple and easy to understand.
As I was learning more about these models and kernels, and acquisition function optimisation, I moved over to PyTorch, which although slightly more complex to setup and understand, it provides better results, enough to justify the move.

### Documentation
#### How do your README and other documents currently describe the purpose, inputs, outputs and objectives of your BBO capstone project?
The main README file explains the project and its objectives, as well as a brief description of the inputs, outputs, and the strategies I used throughout.
Then, each week's folder includes another README with more in-depth explanations for the choices I've made, including the answers to the Capstone assignment questions from the course.

#### What updates do you need to align the documentation with your most recent strategy and results?
The only thing I haven't documented much is the results of the weekly queries, and my initial reaction to them.
Apart from that. I've been keeping documentation up to date with my strategy, both in README files and code comments.