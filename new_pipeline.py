import pandas as pd
import os
from PIL import Image
import json

from serve.utils_llm import get_llm_output
import ast
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb 
import re

systems_prompt = "Given a dataset of text outputs from two different large language models (LLMs), your task is to analyze and summarize the data based on specific characteristics. The goal is to identify and cluster similar behaviors or traits within the outputs, summarizing these into a concise list of commonly observed behaviors for each model. This analysis will help in understanding the general behaviors of these models for auditing, error discovery, and comparison purposes. Your outputs adhere to the format given by the user."
smaller_systems_prompt = "You are a helpful assistant. Your outputs adhere to the format given by the user."

cluster_axes_descriptions_prompt = ["""The following are the axes of variation that you can consider when comparing the two model outputs along with a description of how two models (A and B) vary along that axis. Each axis has a name as well as a description of what it means to be low and high on this axis. Many of these axes of variations could be named incorrectly or redundant with other axes. I want to cluster these axes so that I can better understand the general patterns seen in these models without having to look through so many axes. Please cluster this large list of axes into a minimal set of parent axes that cover the entire axis list. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Your new set of axes should be distinct so each of the above axes fit under exactly one of your new axes.
                                    
    Here are the axes of varaiation:
    {axes}

    Again I want to cluster these axes into a minimal set of parent axes that cover the entire axis list. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Your new set of axes should be distinct so each of the above axes fit under exactly one of your new axes. Please ensure each axis and parent axis contains an axis name and descriptions of what it means to score high or low on that axis.  Please ensure the descriptions of what is considered high and low on each axis is clear, concise, under 10 words. Please focus on patterns that are important for understanding the behavior of a language model, as these will later be used to help debug an important system""", 
                                     
    """thanks! Now can you please categorize each of the original axes under you new list of axes? Remember that each original axis should only belong to one of the axes you described. Here are the original axes again for reference:
    {axes}

    Please structure your response as:

    {{new axis}}:  High: {{new axis high description}} Low: {{new axis low description}}
    - {{original axis 1}}:  High: {{original axis high description}} Low: {{original axis low description}}
    - {{original axis 2}}:  High: {{original axis high description}} Low: {{original axis low description}}
    
    Please ensure that all the original axes above are categorized under the new axes you provide. Please ensure each of original axes listed above should only belong to one of the axes you described. If there are any axes that do not fit under any of the new axes you provided, please list them under a new axis. If there are any new axes that fit the same original axes, please merge them together."""]

OZ_PROMPT = """
    The following are the result of asking two different language models to generate an answer for the same questions:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two LLM outputs so I can better compare the behavior of these models.

   Please output a list differences between the two outputs with relation to specific axes of variation. Are there any general patterns, clusters, or variations you notice in the outputs? Try to give patterns that are specific enough that someone could reliably produce new examples that fit the rule, and that they could understand what it means to be higher or lower on that specific axis.
   
   Here are some axes of variation to consider:

   {axes}

   This list is not exhaustive, please add new axes in your response even if it does not fit under one of these categories. If the outputs are roughly the same along one of the provided axes do not include it. 

   The format of response should be a bulleted list of differences, one bullet for each axis. The format should be
   - {{axis_1}}: {{difference}}
   - {{axis_2}}: {{difference}}
    
    Please output differences which have a possibility of showing up in future unseen data and which would be useful for a human to know about when deciding with LLM to use. Please describe the difference in each axis clearly and concisely, along with an explanation of what in the text proves this difference exists. Please order your response in terms of the most prominent differences between the two outputs. If the outputs are nearly identical, please write "No differences found."
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

AXIS_CONVERSION = """The following are the axes of variation that you can consider when comparing the two outputs along with a description of how two models (A and B) vary along that axis:

    {axes}

    I want to formalize exactly what it means to be high and low on each axis. For each axis, I want you to provide a description of what it means to be high and low on that axis, as well as a score of where the two models fall on that axis. The score for a given model could be ("very low", "low", "high", "very high").This will help me understand the differences between the two models in a more structured way. Your output should be in this format:

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

    Please ensure the description of what it means to be high and low on the axis is clear and concise and under 5 words, and the scores are accurate representations of the outputs of the two models.
"""

def get_cluster_axes(cluster):
    cluster_axes_descriptions_prompt = ["""The following are the axes of variation that you can consider when comparing the two model outputs along with a description of how two models (A and B) vary along that axis. Each axis has a name as well as a description of what it means to be low and high on this axis. Many of these axes of variations could be named incorrectly or redundant with other axes. I want to cluster these axes so that I can better understand the general patterns seen in these models without having to look through so many axes. Please cluster this large list of axes into a minimal set of parent axes that cover the entire axis list. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Your new set of axes should be distinct so each of the above axes fit under exactly one of your new axes.
                        
    Here are the axes of varaiation (note each axis is formatted {{axis name}}: High: {{high description}} Low: {{low description}}):
    {axes}

    Again I want to cluster these axes into a minimal set of parent axes that cover the entire axis list. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Your new set of axes should be distinct so each of the above axes fit under exactly one of your new axes. Please ensure each axis and parent axis contains an axis name and descriptions of what it means to score high or low on that axis in the same format as the provided axes.  Please ensure the descriptions of what is considered high and low on each axis is clear, concise, under 10 words. Please focus on patterns that are important for understanding the behavior of a language model, as these will later be used to help debug an important system""", 
                                     
    """thanks! Now can you please categorize each of the original axes under you new list of axes? Remember that each original axis should only belong to one of the axes you described. Here are the original axes again for reference:
    {axes}

    Please structure your response as a numbered list that adheres to the following format:

    1. {{new axis name}}:  High: {{new axis high description}} Low: {{new axis low description}}
    - {{original axis 1}}:  High: {{original axis high description}} Low: {{original axis low description}}
    - {{original axis 2}}:  High: {{original axis high description}} Low: {{original axis low description}}
    
    Please ensure that all the original axes above are categorized under the new axes you provide. Please ensure each of original axes listed above should only belong to one of the axes you described. If there are any axes that do not fit under any of the new axes you provided, please list them under a new axis. If there are any new axes that fit the same original axes, please merge them together."""]
    smaller_systems_prompt = "You are a helpful assistant. Your outputs adhere to the format given by the user."

    prompt_1 = cluster_axes_descriptions_prompt[0].format(axes="\n".join(cluster))
    cluster_1_reduced_axes = get_llm_output(prompt_1, model="gpt-4-0125-preview", system_prompt=smaller_systems_prompt)

    history = [{"role": "user", "content": prompt_1}, {"role": "assistant", "content": cluster_1_reduced_axes}]
    prompt_2 = cluster_axes_descriptions_prompt[1].format(axes="\n".join(cluster))
    cluster_1_reduced_axes_categorized = get_llm_output(prompt_2, model="gpt-4-0125-preview", system_prompt=smaller_systems_prompt, history=history)

    return prompt_1, prompt_2, cluster_1_reduced_axes, cluster_1_reduced_axes_categorized

def convert_axes_clusters_to_df(llm_output):
    conversion_prompt = """Below is a numbered list of axes of variation with their high and low descriptions, along with the original axes categorized under them. Please convert this list into a JSON format and return it.

    {axes}

    Please format you JSON response such that the keys are the axes of varation along with their high and low descriptions and the values are a list of the original axes with their high and low descriptions categorized under them. 

    An example JSON format is shown below:
    "{{new_axis_1_with_high_low}}" : ["{{original_axis_1_with_high_low}}", "{{original_axis_2_with_high_low}}"]

    Please ensure that the axes that make up the keys are copied verbatim from your previous output, including the high low descriptions. Please make sure that the original axes that make up the values are copied verbatim from the list above, including the high low descriptions. I should be able to take this response directly and convert it into a Python object with ast.literal_eval().
    """
    for i in range(3):
        try:
            converted_list = get_llm_output(conversion_prompt.format(axes=llm_output), model="gpt-3.5-turbo", system_prompt=smaller_systems_prompt)
            cluster_1_converted_list = ast.literal_eval(converted_list)
            # make sure high and low are mentioned in the keys
            for key in cluster_1_converted_list.keys():
                if "high" not in key.lower() and "low" not in key.lower():
                    print(f"ERROR in key: {key}")
                    raise ValueError("High or low not mentioned in the key")
            # Creating lists to store the processed data
            sub_axes_list = []
            axis_list = []

            # Looping through the dictionary to fill the lists
            for axis, sub_axes in cluster_1_converted_list.items():
                for sub_axis in sub_axes:
                    axis_list.append(axis)
                    sub_axes_list.append(sub_axis)

            # Creating a DataFrame from the lists
            return converted_list, cluster_1_converted_list, pd.DataFrame({
                'sub_axes': sub_axes_list,
                'axis': axis_list
            })
        except:
            print(f"Attempt {i + 1} failed. Retrying...")
            converted_list = get_llm_output(conversion_prompt.format(axes=llm_output), model="gpt-3.5-turbo", system_prompt=smaller_systems_prompt, cache=False)


# Regular expression to match entities: Capitalized words or phrases followed by a colon
regex_pattern = r'-\s*(?:\*\*)?([A-Za-z ]+?)(?:\*\*)?:'

def extract_entities(text):
    matches = re.findall(regex_pattern, text)
    return [m for m in matches if m not in ["Model A", "Model B"]]

def extract_axis_descriptions(text):

    lines = text.strip().split('\n')

    # Initialize variables to construct the sections
    sections = []
    current_section = []

    # Process each line, building sections while excluding model scores
    for line in lines:
        # Check if the line starts a new section or is part of the current one
        if line.startswith('- ') and current_section:  # If starting a new section and there is a current section
            # Join the current section lines and add to sections
            sections.append('\n'.join(current_section).strip().replace("- ", "").replace("\n", ""))
            current_section = [line]  # Start a new section
        elif "Model A Score" not in line and "Model B Score" not in line:
            # If the line is not a model score, add it to the current section
            current_section.append(line)

    # Don't forget to add the last section
    if current_section:
        sections.append('\n'.join(current_section).strip().replace("- ", "").replace("\n", ""))
    return sections

wandb.init(project="llm_eval_refactor", entity="lisadunlap")
df = pd.read_csv('data/all.csv')
# randomly sample 10 rows, set random seed for reproducibility
random.seed(42)
df = df.sample(10)
model_columns = ["human_answers", "chatgpt_answers"]
model_names = ["Human", "ChatGPT"]
oz_axes = ["Tone", "Format", "Level of Detail", "Ability to answer", "Approach", "Creativity", "Fluency and crammatical correctness", "Adherence to prompt"]

# get per question differences
results = {"prompt": [], "response": [], "axes": [], "axis_response": []}
for i, row in df.iterrows():
    texts = f"{row['question']}\nModel A: {row[model_columns[0]]}\nModel B: {row[model_columns[1]]}\n"
    prompt = OZ_PROMPT.format(text=texts, axes="\n".join([f"* {axis}" for axis in oz_axes]))
    # prompt = DEFAULT_PROMPT.format(text=texts)
    response = get_llm_output(prompt, model="gpt-3.5-turbo", system_prompt=systems_prompt, trace_name="per question differences").replace("**", "")
    results["prompt"].append(texts)
    results["response"].append(response)
    results["axes"].append(extract_entities(response))
    axis_prompt = AXIS_CONVERSION.format(axes=response)
    axis_response = get_llm_output(axis_prompt, model="gpt-3.5-turbo", system_prompt=smaller_systems_prompt, trace_name="convert per question axes")
    results["axis_response"].append(axis_response)
# save results
results = pd.DataFrame(results)

# cluster per axis differences
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

results['axis_description'] = results['axis_response'].apply(extract_axis_descriptions)
all_axis_descriptions = list(set([e for entities in results['axis_description'] for e in entities]))

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each description
embeddings = model.encode(all_axis_descriptions)

# Cluster the embeddings
n_clusters = 3  # Adjust based on your preference or use methods like Elbow to find the optimal number
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(embeddings)

# Assign each description to a cluster
clusters = kmeans.labels_

# Group axes by cluster
grouped_axes = {i: [] for i in range(n_clusters)}
for axis, cluster in zip(all_axis_descriptions, clusters):
    grouped_axes[cluster].append(axis)


all_cluster_axes, all_df_cluster, llm_logs = [], [], {}
for cluster, axes in grouped_axes.items():
    # if cluster > 1:
    #     break
    prompt_1, prompt_2, cluster_axes_1, cluster_axes = get_cluster_axes(axes)
    output_3, parsed_output3, df_cluster = convert_axes_clusters_to_df(cluster_axes)
    llm_logs[cluster] = {"prompt_1": prompt_1, "prompt_2": prompt_2, "output_1": cluster_axes_1, "output_2": cluster_axes, "output_3": output_3, "parsed_output3": parsed_output3}
    df_cluster['cluster'] = cluster + 1
    all_cluster_axes.append(cluster_axes)
    all_df_cluster.append(df_cluster)
    print(f"Cluster {cluster + 1} (length = {len(axes)}) (df length = {len(df_cluster)}):")
    print("")  # New line for readability between clusters

df_cluster = pd.concat(all_df_cluster)
df_cluster.to_csv("testing_clustering.csv", index=False) 
llm_outputs = pd.DataFrame(llm_logs).T
llm_outputs.to_csv("llm_outputs.csv", index=False)

wandb.log({"results": wandb.Table(dataframe=results), "df": wandb.Table(dataframe=df), "df_cluster": wandb.Table(dataframe=df_cluster), "llm_outputs": wandb.Table(dataframe=llm_outputs)})


# save results to csv
results.to_csv("results_oz.csv", index=False)

