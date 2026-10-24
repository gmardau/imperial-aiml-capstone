### CNNs build up features from edges and textures to full objects. How did this idea of progressive feature extraction influence the way you thought about refining your BBO strategy?
CNNs build from coarse (edges) to fine (objects) as data passes through layers. My GP does something similar over time rather than depth: with few points, the posterior can only capture the general structure of the function, and only as weekly data accumulates does it resolve finer local detail. That shaped how much I trust fine-grained posterior structure early on versus later in the campaign. Although, in some functions, the number of datapoints is still small enough to only see general structures.

### LeNet and later CNNs redefined what is possible in computer vision. What parallels do you see between those breakthroughs and the incremental improvements you make in your BBO capstone project?
LeNet/AlexNet were step-changes because they introduced a new architecture (convolutions) that redefined what was searchable at all. In my case, I'm not changing the underlying GP/UCB architecture, just squeezing more value out of a fixed search space as data grows. So the parallel is really about two complementary phases: architectural breakthroughs open a new space, and steady optimisation like mine exploits it.

### Training CNNs often involves balancing depth, computational costs and overfitting risks. Did you face similar trade-offs when choosing whether to explore widely or exploit promising regions in your queries?
Yes, Kappa is my direct analogue to that trade-off: higher values push toward exploration (uncertain regions), lower values exploit known good ones, just like more depth/capacity trades off against overfitting risk. I hit the overfitting side concretely too: my length-scale bounds explicitly avoid values that would let the GP overfit given how few data points I have, the same logic behind constraining a CNN's capacity when training data is limited.

### Convolutions, pooling, activations and loss functions influence how CNNs learn from data. Which of these concepts helped you think differently about how your optimisation model learns from your accumulated data?
Pooling is the most influencial. My acquisition search does the same thing: I sample 512 candidates and keep only the best 32 as optimization starting points for the grandient descent, discarding the rest, much like pooling keeps only the strongest activations.

### The interview with Andrea Dunbar highlighted the trade-offs of deploying CNNs in edge AI systems. How might reflecting on real-world deployment challenges help you decide how to benchmark success in your own BBO capstone project?
Edge AI forces accuracy to be balanced against hard resource constraints, not judged in isolation. My setting has an analogous constraint: one query per function per week, no option to sampling more, so success shouldn't just be how much it has improved but how much information I'm getting per query spent. That reframes benchmarking around efficiency-under-constraint rather than raw performance, the same standard real deployments impose on models that look good only in the lab.

---

### What is the main technical justification for your current BBO approach? Which aspect of prior research or established methods supports your choice?
My choice was a Gaussian Process surrogate with a UCB acquisition function, tuned per-dimension (ARD) and per-function (Matérn smoothness, length-scale bounds, Kappa). This is justified by the extreme query budget (one point per week). GPs are the standard surrogate for this regime because they give calibrated uncertainty from very few points, and UCB gives an explicit, tunable exploration-exploitation knob via a single scalar. I've explored other models, such as SVMs, NNs, but none are quite as good as GP surrogates.

### Which academic papers have you used to guide your design? Which ideas or techniques from the literature are most relevant, and how do they strengthen your project?
I've used two. Srinivas et al.'s GP-UCB paper (2010) is the additional read I've made for the UCB acquisition formula and its regret guarantees. Lastly, Balandat et al.'s BoTorch paper (2020) for the BoTorch implementation choice over SciKit.

### Which third-party libraries or frameworks (e.g. PyTorch, TensorFlow, scikit-learn) are central to your approach? Why were these the right choices compared with possible alternatives?
I've used BoTorch/GPyTorch as the primary framework because it optimizes acquisition functions via autodiff (optimize_acqf) rather than naive random-restart optimization, and natively supports output standardization and constrained kernels. SciKit GP is still kept as a simpler, more transparent baseline for sanity-checkin.

### How do you plan to document and present these justifications in your GitHub repository so that peers, facilitators and future employers can clearly understand your reasoning?
I've been adding the reasoning behind my choices on the main repository README file as well as individual week README files, with more details, so that the iterative process is visible, not just the final result.

### Looking ahead, what additional sources (research, benchmarks, software) might you consult to continue refining your strategy?
The next step would be to do some additional reading on ARD/length-scale parameters for a more principled way to set bounds other than manual tuning, which is what I've been doing so far.