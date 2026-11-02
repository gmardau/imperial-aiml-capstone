Repository: [github.com/gmardau/imperial-aiml-capstone](https://github.com/gmardau/imperial-aiml-capstone)

### Code presentation

The final method is one script, [implementation/week13.py](../implementation/week13.py). It has a table of per-function settings, a comment above each function and fixed seeds, so the same data always gives the same query, and one uv environment at the root runs everything.

The notebook [implementation/method_and_results.ipynb](../implementation/method_and_results.ipynb) imports that script rather than copying it, and walks through:
- the data and each function's settings;
- two worked examples, a GP function and the radiation-source function;
- a check that rerunning the code on the data up to week 12 regenerates the final-round queries;
- the final results.

Every earlier week's script is kept in `implementation/`.

### Datasheet and model card

Both are in [documentation/](../documentation/) and build on the versions from Module 21, updated with the final results.

- **[Datasheet](../documentation/datasheet.md):** it follows the course template, answering each question for all eight functions. It covers the functions, the data and its noise, the strategy, preprocessing, what I learned each week, the results, and wider considerations.
- **[Model card](../documentation/model_card.md):** it follows the template's sections (description, performance, limitations and trade-offs), plus intended use and ethical considerations.

The main additions since Module 21:
- what repeated queries showed about noise;
- how overconfident the GP's uncertainty was;
- the queries I set by hand in the final weeks, and how they did.

### Non-technical write-up

The [README](../README.md) opens with a 100-word explanation for a general audience. Its other sections follow the course template: data, model, hyperparameter optimisation and results, including the final progress graph. The week-by-week technical approach moved to [documentation/technical_approach.md](../documentation/technical_approach.md), so the README stays short.
