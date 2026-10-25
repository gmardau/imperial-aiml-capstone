### Which prompt patterns (zero-shot, few-shot, etc.) did you use, and why? What changed when you simplified vs structured the prompt?

I didn't use an LLM to choose this week's queries: they came from the same GP and UCB set-up as before. The query decision depends on calibrated uncertainty from very few points, which a GP gives and an LLM doesn't.
If I were to prompt one, I'd use a few-shot pattern: the observed (input, output) pairs as the examples, plus the function's description, asking for one new input in the portal's format.

### What temperature, top-p, top-k and max-tokens settings did you choose? How did they trade off coherence vs diversity? How did they affect your chosen query?

None, since no LLM chose the queries. Temperature plays a similar role to Kappa in UCB, though: a low temperature (or a small top-p/top-k) sticks to the most likely answer, which is like exploiting, and a high one gives more varied answers, which is like exploring, but with no control over whether the variety is useful.
For a query, I'd keep the temperature low for a coherent, well-formatted answer, and get diversity from the prompt instead, e.g. by asking for several candidates and checking each one against the GP. A small max-tokens is enough for a single query string.

### Did token boundaries or unusual input strings affect the model’s behaviour? When did you notice token count limits or truncation influencing the outputs? If no such cases were observed, explain how you checked for those cases.

I didn't see any such cases, since no LLM was involved. This data would be the risky part, though: numbers with lots of decimals, and especially function 1's values in scientific notation are split into several tokens, so an LLM doesn't really read them as numbers, and simple comparisons like "which output is the largest" can go wrong. The checks are cheap: count the prompt's tokens with the model's tokenizer before sending it, so nothing gets truncated, and validate the output before submitting it (the right number of values, all within [0, 1], with six decimals).

### With 17 data points, what limitations did you encounter, such as prompt overfitting, attention focusing on irrelevant context or diminishing returns from longer inputs?

The main limitation is the same one the GP has: 17 points, which is what functions 1 and 2 had by this week, are too few to see the shape of most functions, and the higher-dimensional ones are even sparser. An LLM would add its own problems on top: it would probably anchor on the few best points and propose something close to them: overfitting. The GP, in comparison, uses every point in a consistent way.

### Which strategies did you try to reduce hallucinations? For example, did you use tighter instructions, retrieval of prior relevant information or constrain the output format?

My main protection is that the queries come from a model fitted to the data, not from generated text. With an LLM, I'd constrain the output format to the query string alone, give it the actual data every time instead of relying on earlier rounds, and never submit its answer without checking it: the right number of inputs, all within [0, 1], and a sensible prediction from the GP at that point.

### In future rounds, how would you scale your prompting and decoding strategies when working with larger data sets or more complex LLMs?

With more data I wouldn't paste the raw numbers into a prompt. I'd give the LLM a summary instead (the best points, the GP's length scales, where the model is most uncertain), or let it run code and read the GP's output rather than do the arithmetic itself. That keeps the LLM on what it's good at while the numerical decision stays with the model. Decoding would stay at a low temperature for anything that produces a query.

### How did these design choices for prompts and decoding help you think like a practitioner balancing exploration, risk and computational constraints in a black-box setting with incomplete information?

They made me think about where an LLM fits in the workflow. With one query per function per week, a bad query is expensive, so each one should come from a model whose assumptions I can check: the GP's uncertainty is explicit, while an LLM's isn't. Decoding settings are another exploration-versus-exploitation knob, like Kappa, but a less principled one.
So I'd use an LLM where mistakes are cheap and easy to catch, and keep the expensive decision with the method that quantifies its uncertainty.
