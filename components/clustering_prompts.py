jacob_prompt = """I will provide a series of data for you to remember. Subsequently, I will ask you some
questions to test your performance! Here are some descriptions for you to memorize.

{text}

I’m trying to understand the behavior of different language models. The above are some descriptions of model outputs, and I'd like to cluster them into different axes of variation. Using these specific examples, are there any general patterns, clusters, or variations you notice in the descriptions? Try to give patterns that are specific enough that someone could reliably produce new examples that fit the rule, and that they could understand what it means to be higher or lower on that specific axis. Please try to give as
many general patterns as possible. Please focus on patterns that are important for understanding the behavior of a language model, as these will later be used to help debug an important system. Please explain clearly why the pattern would be important for understanding the behavior of such a system. Please summarize as many as you can and stick to the examples."""

jacob_prompt_pt2 = """Thank you. Now I would like you to use these axes to categorize the specific examples I gave you. To start with, let's consider the first axis you chose:
{cluster}

I want you to consider each of the specific descriptions from before, and determine whether they are relevant to this axis. If they are, say how they score along this axis on a scale of -5 to 5, where -5 means they are strongly towards the low end of the axis, and 5 means they are strongly towards the high end. Provide your output as a list of descriptions followed by the score, each on one line, such as
{{first description}}: {{score from -5 to 5}}
{{second description}}: {{score from -5 to 5}}
Include only the descriptions that are relevant to the axis. As a reminder, here are the descriptions from before:

{text}
"""