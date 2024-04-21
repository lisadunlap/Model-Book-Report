OZ_PROMPT = """
    The following are the result of asking two different language models to generate an answer for the same questions:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two LLM outputs so I can better compare the behavior of these models. Are there any general patterns, clusters, or variations you notice in the outputs? 

   Please output a list differences between the two outputs with relation to specific axes of variation. Try to give axes that a human could easily interpret and they could understand what it means to be higher or lower on that specific axis. Please ensure that the concepts used to explain what is high and low on the axis are distinct and mutually exclusive such that given any pair of text outputs, a human could easily and reliably determine which model is higher or lower on that axis.
   
   Here are some axes of variation to consider:

   {axes}

   This list is not exhaustive, please add new axes in your response even if it does not fit under one of these categories. If the outputs are roughly the same along one of the provided axes do not include it. 

   The format of response should be a bulleted list of differences, one bullet for each axis. The format should be
   - {{axis_1}}: {{difference}}
   - {{axis_2}}: {{difference}}
    
    Please output differences which have a possibility of showing up in future unseen data and which would be useful for a human to know about when deciding with LLM to use. For each axis, define clearly and succinctly what constitutes a high or low score, ensuring these definitions are mutually exclusive. For each axis, also provide an explanation of what in the text proves this difference exists. Please order your response in terms of the most prominent differences between the two outputs. If the outputs are nearly identical, please write "No differences found."
"""

DEFAULT_PROMPT = """
    The following are the result of asking two different language models to generate an answer for the same questions:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two LLM outputs so I can better compare the behavior of these models.

    Please output a list differences between the two outputs with relation to specific axes of variation. Are there any general patterns, clusters, or variations you notice in the outputs? Try to give patterns that are specific enough that someone could reliably produce new examples that fit the rule, and that they could understand what it means to be higher or lower on that specific axis.

   The format of response should be a bulleted list of differences, one bullet for each axis. The format should be
   - {{axis_1}}: {{difference}}
   - {{axis_2}}: {{difference}}
    
    Please output differences which have a possibility of showing up in future unseen data and which would be useful for a human to know about when deciding with LLM to use. Please describe the difference in each axis clearly and concisely, along with an explanation of what in the text proves this difference exists. Please order your response in terms of the most prominent differences between the two outputs. If the outputs are nearly identical, please write "No differences found."
"""


# DEFAULT_PROMPT = """
#     The following are the result of asking two different language models to generate an answer for the same questions:

#     {text}

#    I am a machine learning researcher trying to figure out the major qualitative differences between these two groups so I can correctly identify which model generated which response for unseen questions. This is a very small portion of the data, so I want the differences to be general.

#     Please output a list differences between the two outputs with relation to specific axes of variation. Are there any general patterns, clusters, or variations you notice in the outputs? Try to give patterns that are specific enough that someone could reliably produce new examples that fit the rule, and that they could understand what it means to be higher or lower on that specific axis.

#    The format of response should be a bulleted list of differences, one bullet for each axis. The format should be
#    - {{axis_1}}: {{difference}}
#    - {{axis_2}}: {{difference}}
    
#    Please output 
# """

AXIS_CONVERSION = """The following are the axes of variation that you can consider when comparing the two outputs along with a description of how two models (A and B) vary along that axis:

    {axes}

    I want to formalize exactly what it means to be high and low on each axis. For each axis, I want you to provide a description of what it means to be high and low on that axis, as well as a score of where the two models fall on that axis. The score for a given model could be ("low", "high").This will help me understand the differences between the two models in a more structured way. Your output should be in this format:

    - {{axis_1}}:
        High: {{description of high}}
        Low: {{description of low}}
        Model A Score: {{score for Model A}}
        Model B Score: {{score for Model B}}

    - {{axis_2}}:
        High: {{description of high}}
        Low: {{description of low}}
        Model A Score: {{score for Model A}}
        Model B Score: {{score for Model B}}

    Please ensure that the description what is high and low on the axis are distinct and mutually exclusive such that given any unseen pair of text outputs, a human could easily and reliably determine which model is higher or lower on that axis. Please the axis name and descriptions of what is high and low are less than 5 words each, and ensure the scores are accurate representations of the outputs of the two models.
"""