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

from wandb.integration.openai import autolog

systems_prompt = "Given a dataset of text outputs from two different large language models (LLMs), your task is to analyze and summarize the data based on specific characteristics. The goal is to identify and cluster similar behaviors or traits within the outputs, summarizing these into a concise list of commonly observed behaviors for each model. This analysis will help in understanding the general behaviors of these models for auditing, error discovery, and comparison purposes.\n\n1. Clustering: Identify and group text outputs that share common themes, behaviors, or characteristics. Use the clustering to reveal insights about the models' tendencies in generating text.\n\n2. Summarization: For each cluster identified, provide a summary that encapsulates the key behaviors or traits observed. This should offer a clear, concise overview of the commonalities within the cluster.\n\n3. Comparative Analysis: Compare the summarized behaviors between the two LLMs. Highlight differences and similarities in their output characteristics, providing insights into their unique or shared behaviors.\n\n4. Recommendations: Based on the analysis, suggest potential areas for model improvement or further investigation.\n\nYour analysis should take into account the context that these text outputs are generated by LLMs, focusing on the qualitative aspects that distinguish one model from another. Provide your findings in a structured and easily understandable format."

# systems_prompt = "You are an expert data scientist with the goal of discovering differences in the behavior of LLM's."

jacob_prompt = """I will provide a series of data for you to remember. Subsequently, I will ask you some
questions to test your performance! Here are some descriptions for you to memorize.
[
{text}
]
I’m trying to understand the behavior of different language models. The above are some descriptions of model outputs, and I'd like to cluster them into different axes of variation. Using these specific examples, are there any general patterns, clusters, or variations you notice in the descriptions? Try to give patterns that are specific enough that someone could reliably produce new examples that fit the rule, and that they could understand what it means to be higher or lower on that specific axis. Please make the axes such that no two axes will provide roughly the same takeaways about the data. Please try to give as
many general patterns as possible. Please focus on patterns that are important for understanding the behavior of a language model, as these will later be used to help debug an important system. Please explain clearly why the pattern would be important for understanding the behavior of such a system. Please summarize as many as you can and stick to the examples. Please output your response as a numbered list."""

jacob_prompt_pt2 = """Thank you. Now I would like you to use these axes to categorize the specific examples I gave you. To start with, let's consider this axis you chose:
{cluster}

I want you to provide a description of each end of this axis then I want you to consider each of the specific descriptions from before, and determine whether they are relevant to this axis. If they are, say how they score along this axis on a scale of -5 to 5, where -5 means they are strongly towards the low end of the axis, and 5 means they are strongly towards the high end. Provide your output as a tuple of each end of the axis and then list of descriptions followed by the score, each on one line, such as
Axis descriptions: (\"{{description of low end}}\", \"{{description of high end}}\")
1. \"{{first description}}\": {{score from -5 to 5}}
2. \"{{second description}}\": {{score from -5 to 5}}
Include only the descriptions that are relevant to the axis. As a reminder, here are the descriptions from before:
[
{text}
]
Please output your response as a tuple of axis descriptions on the first lines, and the following lines as a numbered list of the decriptions and their score.
"""

def plot_distributions(group_1_scores, group_2_scores, hypothesis="", axes_descriptions=["", ""], group_names=["Group A", "Group B"]):
    """
    Plots the distributions of cos sim to hypothesis for each group.
    """
    group_1_scores = np.array(group_1_scores).ravel()
    group_2_scores = np.array(group_2_scores).ravel()
    #remove any -100 scores
    group_1_scores = [x for x in group_1_scores if x != -100]
    group_2_scores = [x for x in group_2_scores if x != -100]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(group_1_scores, bins=range(-5, 6), alpha=0.5, label=group_names[0], density=True)
    ax.hist(group_2_scores, bins=range(-5, 6), alpha=0.5, label=group_names[1], density=True)
    ax.set_title(f"Histogram of Axis Scores to {hypothesis} \n Low: {axes_descriptions[0]} \n High: {axes_descriptions[1]}")
    ax.set_ylabel("Density")
    ax.legend()

    return fig

def response_to_dict(response):
    # Convert the list to a dictionary
    converted_dict = {}
    flag = False
    for item in response.split("\n"):
        if "axis descriptions" in item.lower():
            axes = ast.literal_eval(item.lower().replace("axis descriptions: ", ""))
            flag = True
        else:
            try:
                # Split each string by ": "
                parts = item.split(": ")
                # The key is everything after the first ". " and before the ": "
                key = parts[0].split(". ", 1)[1]
                # The value is the part after the ": " converted to an integer
                value = int(parts[1])
                converted_dict[key] = value
            except:
                # print(f"did not parse {item}")
                pass
    if not flag:
        print(response)
        raise ValueError("No axis descriptions found in response")
    return axes, converted_dict

def get_clusters(texts, model_name = "gpt-4", batch=150):
    clustering_outputs = {"input": [], "prompt": [], "output": []}
    texts = list(set([t.lower() for t in texts]))
    texts = random.sample(texts, batch)
    response = get_llm_output(jacob_prompt.format(text='\n'.join(texts)), model_name)
    print(response)
    print("............................................\n............................................")
    clustering_outputs["input"].append("Axes description")
    clustering_outputs["prompt"].append(jacob_prompt.format(text='\n'.join(texts)))
    clustering_outputs["output"].append(response)
    # parse the numbered list string into a list of strings
    parsed_response = response.split("\n")
    # only keep items which first begin with a number
    parsed_response = [r for r in parsed_response if len(r) > 0 and r[0].isdigit()]
    clusters = {}
    for i, res in enumerate(parsed_response):
        cluster_name = res.split(f"{i+1}. ")[1].split(":")[0].replace("**", "")
        response2 = get_llm_output(jacob_prompt_pt2.format(cluster=res, text='\n'.join(texts)), model_name)
        clustering_outputs["input"].append(cluster_name)
        clustering_outputs["prompt"].append(jacob_prompt_pt2.format(cluster=res, text='\n'.join(texts)))
        clustering_outputs["output"].append(response2)
        print("--------------------------")
        print(f"Cluster: {cluster_name}")
        print(response2)
        print("--------------------------")
        clusters[cluster_name] = {}
        (clusters[cluster_name]["axes_description_low"], clusters[cluster_name]["axes_description_high"]), clusters[cluster_name]["description_scores"] = response_to_dict(response2)
    return clusters, clustering_outputs

# Adjust the function to handle both case-insensitivity and punctuation removal
def extract_scores_clean(differences, scores_dict):
    scores = []
    for diff in differences:
        # Clean and convert to lowercase
        diff_clean_lower = diff.lower()
        # Add quotes for matching
        diff_quoted = f'"{diff_clean_lower}"'
        if diff_quoted in scores_dict or diff_clean_lower in scores_dict:
            scores.append({"score": scores_dict[diff_quoted], "difference": diff_clean_lower})
    return scores

# only want 1 score per question per group, so round up or down
def round_scores(scores):
    difference = [x["difference"] for x in scores]
    scores = [x["score"] for x in scores]

    if len(scores) > 0 and np.mean(scores) > 0:
        index, value = max(enumerate(scores), key=lambda x: x[1])
        return [{"score": max(scores), "difference": difference[index]}]
    elif len(scores) > 0 and np.mean(scores) < 0:
        index, value = min(enumerate(scores), key=lambda x: x[1])
        return [{"score": min(scores), "difference": difference[index]}]
    return [{"score": -100, "difference": ""}]

def get_cluster_counts(cluster_descriptions, group_names = ["Group A", "Group B"]):

    plots = []
    for axis_name in cluster_descriptions:
        description_scores = cluster_descriptions[axis_name]["description_scores"]
        # Adjust the dictionary to remove punctuation and be case-insensitive
        description_scores_clean = {k.lower(): v for k, v in description_scores.items()}

        # Example row of differences
        group_1_scores, group_2_scores = [], []
        for i, row in questions.iterrows():
            group_1_differences = row['group_1_hypotheses']
            group_2_differences = row['group_2_hypotheses']
            # Extract scores for both groups with adjustments   
            group_1_scores += round_scores(extract_scores_clean(group_1_differences, description_scores_clean))
            group_2_scores += round_scores(extract_scores_clean(group_2_differences, description_scores_clean))
        group_1_differences, group_2_differences = [x['difference'] for x in group_1_scores], [x['difference'] for x in group_2_scores]
        group_1_scores, group_2_scores = [x['score'] for x in group_1_scores], [x['score'] for x in group_2_scores]
        cluster_descriptions[axis_name]["group_1_scores"] = group_1_scores
        cluster_descriptions[axis_name]["group_2_scores"] = group_2_scores
        cluster_descriptions[axis_name]["group_1_score_diffs"] = group_1_differences
        cluster_descriptions[axis_name]["group_2_score_diffs"] = group_2_differences

        count = set([i for i, x in enumerate(group_1_scores) if x != -100] + [i for i, x in enumerate(group_2_scores) if x != -100])
        cluster_descriptions[axis_name]["count"] = len(count)
        # get value counts for group 1 and group 2 scores as a dictionary sorted by key
        cluster_descriptions[axis_name]["group_1_counts"] = {i: len([x for x in group_1_scores if x == i]) for i in range(-5, 6)}
        cluster_descriptions[axis_name]["group_2_counts"] = {i: len([x for x in group_2_scores if x == i]) for i in range(-5, 6)}
        cluster_descriptions[axis_name]["group_1_avg"] = np.mean([x for x in group_1_scores if x != -100])
        cluster_descriptions[axis_name]["group_2_avg"] = np.mean([x for x in group_2_scores if x != -100])
        cluster_descriptions[axis_name]["difference_score"] = abs(cluster_descriptions[axis_name]["group_1_avg"] - cluster_descriptions[axis_name]["group_2_avg"])

        # Plotting histograms
        axes_descriptions = (cluster_descriptions[axis_name]["axes_description_low"], cluster_descriptions[axis_name]["axes_description_high"])
        plot = plot_distributions(group_1_scores, group_2_scores, hypothesis=axis_name, axes_descriptions=axes_descriptions, group_names=group_names)
        
        plots.append(plot)

    return plots

def summarize(results, threshold=3):
    """
    Summarizes the results of the clustering.
    """
    results_str = []
    for axis in results:
        if results[axis]["difference_score"] > threshold:
            results_str.append(f"{axis}: {results[axis]['axes_description_low']} vs {results[axis]['axes_description_high']}")

if __name__ == "__main__":
    run = wandb.init(project="jacob_clustering", entity="lisadunlap", name="testing")

    # set random seed
    random.seed(42)
    questions = pd.read_csv('./all_outputs.csv')
    model_a = "human_answers"
    model_b = "chatgpt_answers"

    # # explode question, group_1_answers, and group_2_answers at the same time
    # for col in questions.keys():
    #     questions[col] = questions[col].apply(ast.literal_eval)
    # questions = questions.explode(['question', 'group_1_answers', 'group_2_answers', 'group_1_hypotheses', 'group_2_hypotheses'])
    questions['group_1_hypotheses'] = questions['group_1_hypotheses'].apply(ast.literal_eval)
    questions['group_2_hypotheses'] = questions['group_2_hypotheses'].apply(ast.literal_eval)

    group_a_hypotheses = [item for sublist in questions["group_1_hypotheses"] for item in sublist]
    group_b_hypotheses = [item for sublist in questions["group_2_hypotheses"] for item in sublist]
    # mix the hypotheses and keep a lsit of which group they belong to
    mixed_hypotheses = group_a_hypotheses + group_b_hypotheses
    random.shuffle(mixed_hypotheses)
    cluster_descriptions, llm_logs = get_clusters(mixed_hypotheses)
    wandb.log({"llm_logs": wandb.Table(dataframe=pd.DataFrame(llm_logs))})
    plots = get_cluster_counts(cluster_descriptions)
    # save cluster descriptions to a json file
    with open('cluster_descriptions_test.json', 'w') as f:
        json.dump(cluster_descriptions, f)

    # log plots to wandb
    for i, plot in enumerate(plots):
        wandb.log({f"{list(cluster_descriptions.keys())[i]}": wandb.Image(plot)})

    # save the cluster descriptions to wandb
    df = pd.DataFrame(cluster_descriptions).transpose()
    df['axis'] = df.index
    # sort by difference score
    df = df.sort_values(by='difference_score', ascending=False)
    wandb_dataframe = df[["axis", "axes_description_low", "axes_description_high", "group_1_avg", "group_2_avg", "difference_score"]].copy(deep=True)
    wandb.log({"cluster_descriptions": wandb.Table(dataframe=wandb_dataframe)})
    df.to_csv("cluster_descriptions.csv")
    # wandb.save("cluster_descriptions.csv")