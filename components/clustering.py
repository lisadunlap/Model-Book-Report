from collections import defaultdict
import re
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd  
import wandb
import json
from serve.utils_llm import get_llm_output

# Assuming get_embedding is a pre-defined function
from openai import OpenAI
client = OpenAI()

def summarize_text(text, model="gpt-3.5-turbo", max_tokens=150):
    """
    Summarize the given text using OpenAI's GPT-3.5-turbo API.
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"All the following strings are from the same cluster:\n\n{text}\n\nOutput a concept which best describes this cluster in less than 5 words. Answer in the format \"Summary = \" and do not include any other text."}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
summary = summarize_text('\n'.join(["cat", "dog", "sheep", "human", "elephant"]))


def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def cluster_dbscan(embeddings):
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(embeddings)
    # print out the size of each cluster
    print({i: np.sum(clustering.labels_ == i) for i in np.unique(clustering.labels_)})
    return clustering.labels_

def cluster_spectral(embeddings, n_clusters=5):
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=0).fit(embeddings)
    unique_labels = np.unique(clustering.labels_)
    print({i: np.sum(clustering.labels_ == i) for i in unique_labels})
    return clustering.labels_

def cluster_hierarchical(embeddings, n_clusters=5):
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings)
    unique_labels = np.unique(clustering.labels_)
    print({i: np.sum(clustering.labels_ == i) for i in unique_labels})
    return clustering.labels_

def cluster_kmeans(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    print({i: np.sum(kmeans.labels_ == i) for i in range(n_clusters)})
    return kmeans.labels_

def summarize_cluster(cluster, embeddings, texts, K):
    # Calculate the centroid of the cluster
    cluster_embeddings = embeddings[cluster]
    # cluster_texts = [texts[i] for i in cluster]
    # print(cluster_texts)
    centroid = np.mean(cluster_embeddings, axis=0)

    # Find the K texts closest to the centroid
    distances = cdist([centroid], cluster_embeddings, metric='euclidean')[0]
    closest_indices = np.argsort(distances)[:K]

    # Summarize these texts
    selected_texts = [texts[i] for i in closest_indices]
    print(selected_texts)
    summarized_text = summarize_text(" ".join(selected_texts))
    return summarized_text

hardcoded_cluster_prompt = """I have a list of text descriptions of qualities of LLM outputs:

{text}

I want to group these into one of the following categories:
    "Format", "Tone", "Level of detail", "Ability to answer", "Clarity", "Question approach"

There will also be an "Other" option for any descriptions that do not fit into the above categories. Any descriptions that talk about how concise or detailed the output is should be put in "Level of detail". 

Please add the following descriptions to the appropriate clusters. Format your response as a dictionary where the keys are the cluster descriptions and the values are the descriptions that fall under that cluster. The format should look like this:
{{"Format": [item 1, item 2, ...], "Tone": [item 1, item 2, ...]}}

If one of the provided clusters doesnt have a description, please set the cluster value to an emplty list (e.g. "Approach": []). Each desription should only belong to one cluster.

Only respond with a dictionary that I can parse with ast.literal_eval."""


cluster_prompt_dict = """I have a list of text descriptions of qualities of LLM outputs that I want to condense. These descriptions should be summarized into a shorter list of the most common qualities found. Here is the list of descriptions:

{text}

Format your response as a dictionary where the keys are the summarized qualities, and the values are the descriptions that fall under that quality. The format should look like this:
{{'quality 1': ['item 1 in quality 1', 'item 2 in quality 1', ...], 'quality 2': ['item 1 in quality 2', 'item 2 in quality 2', ...]}}

These qualities should be discriminatory, specific, and easy to understand. For instance, "tone" would not be a suitable summary quality because there is no concept of having little or no tone; instead there could be two seperate qualities of "formal tone" and "informal tone". Each cluster should not contain descriptions that mean opposites. For instance, "formal tone" and "informal tone" should not be in the same cluster.

There should be fewer than 5 summarized qualities, and the description of each summarized quality should be less than 10 words. Only respond with a dictionary that I can parse with ast.literal_eval."""



continuing_cluster_prompt = """I have a list of text descriptions of qualities of LLM outputs that I want to condense. These descriptions should be summarized into a shorter list of the most common qualities found. Here is the list of descriptions:

{text}

I already have the following summary qualities from a list of text descriptions of the difference between two model outputs:
{clusters}

Please add the following descriptions to the appropriate qualities or add a new quality if necessary.

Format your response as a dictionary where the keys are the summarized qualities, and the values are the descriptions that fall under that quality. The format should look like this:
{{'quality 1': ['item 1 in quality 1', 'item 2 in quality 1', ...], 'quality 2': ['item 1 in quality 2', 'item 2 in quality 2', ...]}}

These qualities should be directional, meaning there should be a notion of "more X" and "less X" or "X" and "not X". For instance, "tone" would not be a suitable summary quality because there is no concept of having little or no tone, but "formal tone" would be appropriate because it is possible to have a less formal tone.

There should be fewer than 5 summarized qualities, and the description of each summarized quality should be less than 10 words. Only respond with a dictionary that I can parse with ast.literal_eval."""

fix_format = """I tried to parse the following string into a dictionary with ast.literal_eval but it failed ({error}). Please fix the format and try again.

{text}

Only respond with the corrected string."""


import ast
def parse_response(response):
    print(response)
    print("------------------")
    try:
        # First try to parse as JSON
        return ast.literal_eval(response)
    except Exception as inst:
        try:
            response = get_llm_output(fix_format.format(error=inst, text=response), "gpt-4", cache = False)
            return ast.literal_eval(response)
        except ValueError:
            # Log the problematic response and raise a formatting error
            raise ValueError("The response from the model is not in the expected format")

def cluster_with_gpt(texts, model="gpt-4", cluster_prompt_dict=None):
    text = '\n'.join(texts)
    responses = get_llm_output(cluster_prompt_dict.format(text=text), model)
    parsed_responses = parse_response(responses)
    counts = [{"hypothesis": k, "count": len(v), "examples": v} for k, v in parsed_responses.items()]
    return responses, counts

def batch_cluster_with_gpt(texts, model="gpt-4", batch_size=50, hardcode=False, cache = True):
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    results = []
    response_results = []
    existing_clusters = set()
    text = '\n'.join(texts)

    for i, batch in enumerate(batches):
        text = '\n'.join(batch)
        if hardcode:
            response = get_llm_output(hardcoded_cluster_prompt.format(text=text), model, cache = cache)
        else:
            if i == 0:
                prompt = cluster_prompt_dict.format(text=text)
            else:
                prompt = continuing_cluster_prompt.format(text=text, clusters='\n'.join(existing_clusters))
            response = get_llm_output(prompt, model, cache = cache)

        response = response[response.find("{"):response.rfind("}")+1]
        # response = response.replace(".", "")
        parsed_responses = parse_response(response)
        print(parsed_responses)
        print("++++++++++++++++++++++++++++++++++++")
        counts = [{"hypothesis": k, "count": len(v), "examples": v} for k, v in parsed_responses.items()]


        # Update existing clusters and results
        for hyp in counts:
            existing_clusters.add(hyp["hypothesis"])
            match = next((res for res in results if res["hypothesis"] == hyp["hypothesis"]), None)
            if match:
                match["count"] += hyp["count"]
                match["examples"] += hyp["examples"]
            else:
                print(f"Adding new cluster {hyp['hypothesis']}")
                results.append(hyp)

        response_results.append(response)

    for res in results:
        print(res['hypothesis'], res['count'])

    return response_results, results

def log_clusters(clusters):
    table = wandb.Table(dataframe=pd.DataFrame(clusters))
    # log clusters as table to wandb
    wandb.log({"clusters": table})
    # log a bar plot of counts
    # wandb.log({"counts" : wandb.plot.bar(table, "hypothesis", "count",
    #                         title="Cluster Counts")})
    # log each cluster as a separate table with columns count and examples and title as cluster name
    # for cluster in clusters:
    #     wandb.log({cluster["hypothesis"]: wandb.Table(dataframe=pd.DataFrame(cluster["examples"], columns=["examples"]))})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cluster a list of texts")
    parser.add_argument("--wandb_link", type=str, help="The wandb link to the run", default="clipinvariance/LLMEval/as0rl6ca")
    args = parser.parse_args()
    # Example usage
    wandb.init(project="explainable-clustering", entity="clipinvariance", name = args.wandb_link)
    def get_wandb_results(run_id):
        api = wandb.Api()
        dfs = []
        runs = [run_id]
        for r in runs:
            run = api.run(r)
            for f in run.files():
                if "hypotheses" in f.name:
                    file_name = f.name
                    f.download(replace=True) 
            # Load the logs
            with open(file_name) as f:
                logs = json.load(f)

            df = pd.DataFrame(logs['data'], columns=logs['columns'])
            df['group'] = run.group
            dfs.append(df)
        return pd.concat(dfs)
    df = get_wandb_results(args.wandb_link)
    # response, clusters = cluster_with_gpt(df['hypothesis'].tolist())
    response, clusters = batch_cluster_with_gpt(df['hypothesis'][:30].tolist(), batch_size=10, hardcode=True)

    log_clusters(clusters)