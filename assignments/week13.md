### How has your understanding of the exploration–exploitation trade-off evolved with increasing data? How did you balance taking risks by exploring new strategies versus exploiting those that had proven effective?

At the start, with only the initial points, I explored a lot, choosing per function from the descriptions and setting UCB's kappa by hand. I later learnt that exploring only pays when the model's uncertainty points somewhere useful: in weeks 6 to 8, high kappas sent queries to edges and corners and rarely helped. As the data grew, I moved to expected improvement inside a trust region, which exploits near the best point and only widens after a success. I still take risks, but only where a miss costs nothing, like function 1's query at the predicted source.

### As your data set expanded, how did the nature of feedback influence your optimisation process? Can you relate this to how RL agents adjust their reward expectations or update their Q-values?

Each week's result works like a reward that updates two things, much as an RL agent updates its Q-values. The GP updates its estimate of the function, which changes the expected value of every candidate point, and the trust region updates its size: it doubles after an improvement and halves after a miss, reinforcing what paid off. Early feedback brought big improvements and surprises. Later rewards were mostly small, so the model's predictions changed less each week, like Q-values settling as an agent's estimates converge.

### Think about the AlphaGo Zero case study. How does your iterative improvement reflect the idea of self-play or autonomous learning? Did your process resemble model-free learning (trial and error) or model-based planning (anticipating future outcomes)?

Mostly model-based planning. The GP is a model of each function, and every query is chosen by predicting its outcome and uncertainty before spending it, which is closer to AlphaGo Zero's model-guided search than to pure trial and error. The early rounds had more trial and error, when I adjusted settings by hand and judged the results.

### How could RL strategies be applied to enhance exploration design, efficiency or convergence speed in real-world optimisation tasks?

RL could learn the exploration policy itself instead of me fixing it by hand. For example, an agent could decide each round how much to explore, or how big the trust region should be, after training on many simulated functions. In real tasks like lab experiments or hyperparameter tuning, that could reach good results in fewer expensive evaluations, as long as the simulated training problems resemble the real one.
