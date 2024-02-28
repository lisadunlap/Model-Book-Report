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


cluster_prompt_dict = """Please group the following list of descriptions into summary clusters, with an explanation of what each cluster means:

{text}

Format your response as a dictionary where the keys are the cluster descriptions and the values are the descriptions that fall under that cluster. The format should look like this:
{{summary of cluster 1: [item 1 in cluster 1, item 2 in cluster 1, ...], summary of cluster 2: [item 1 in cluster 2, item 2 in cluster 2, ...]}}
    
Each cluster summary should be less than 10 words. Only respond with a dictionary that I can parse with ast.literal_eval."""

continuing_cluster_prompt = """I am clustering the following list of descriptions into clusters, with an explanation of what each cluster means:

{text}

I already have the following summary clusters from a list of text descriptions of the difference between two model outputs:
{clusters}

Please add the following descriptions to the appropriate clusters. If you would like to add a new cluster, please provide a new explanation of what the cluster means.

Format your response as a dictionary where the keys are the cluster descriptions and the values are the descriptions that fall under that cluster. The format should look like this:
{{explanation of cluster 1: [item 1 in cluster 1, item 2 in cluster 1, ...], explanation of cluster 2: [item 1 in cluster 2, item 2 in cluster 2, ...]}}

Each cluster explanation should be less than 10 words. Only respond with a dictionary that I can parse with ast.literal_eval."""

# def cluster_with_gpt(texts, model="gpt-4"):
#     text = '\n'.join(texts)
#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": f"Please group the following list of descriptions into clusters, with an explanation of what each cluster means:\n\n{text}\n\nFormat your response as a bulletted list (seperated by *). The format should look like this:\n* explanation of cluster 1\n\t-- item 1 in cluster 1\n\t-- item 2 in cluster 1\n\nEach cluster explanation should be less than 5 words. Only respond with the bulleted list of each cluster."}
#         ],
#     )
#     responces = response.choices[0].message.content.strip()

#     # Split the text into sections based on the asterisk pattern
#     sections = re.split(r'\*\s', responces)

#     # Initialize a dictionary to hold the counts
#     counts = []

#     # Process each section
#     for section in sections[1:]:  # Skip the first split as it's empty
#         # Extract the heading using re.search
#         match = re.search(r'(.+)', section)
#         if match:
#             heading = match.group(1).strip()
#             # Count the bullet points
#             examples = re.findall(r'--\s([^-\n]+)', section)
#             bullet_points = len(examples)
#             # Add to the dictionary
#             counts.append({"hypothesis": heading, "count": bullet_points, "examples": examples})

#     return responces, counts

import ast
def cluster_with_gpt(texts, model="gpt-4"):
    text = '\n'.join(texts)
    responces = get_llm_output(cluster_prompt_dict.format(text=text), model)
    try:
        res = ast.literal_eval(responces)
    except:
        print(responces)
        raise ValueError("The response from the model is not in the expected format")
    counts = [{"hypothesis": k, "count": len(v), "examples": v} for k, v in res.items()]

    return responces, counts

def batch_cluster_with_gpt(texts, model="gpt-4", batch_size=50):
    # Split the texts into batches
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    # Initialize a list to hold the results
    results = {}
    response_results = []
    existing_clusters = ()
    # Process each batch
    for i, batch in enumerate(batches):
        # Cluster the batch
        if i == 0:
            response, counts = cluster_with_gpt(batch, model)
            existing_clusters = {c["hypothesis"]for c in counts}
            results = counts
        else:
            print(continuing_cluster_prompt.format(text='\n'.join(batch), clusters='\n'.join(existing_clusters)))
            responces = get_llm_output(continuing_cluster_prompt.format(text='\n'.join(batch), clusters='\n'.join(existing_clusters)), model)
            # remove any text before and after { } 
            responces = responces[responces.find("{") : responces.rfind("}")+1]
            try:
                res = ast.literal_eval(responces)
            except:
                print(responces)
                raise ValueError("The response from the model is not in the expected format")
            counts = [{"hypothesis": k, "count": len(v), "examples": v} for k, v in res.items()]
            existing_clusters = existing_clusters.union({c["hypothesis"]for c in counts})
            for hyp in counts:
                flag = False
                for res in results:
                    if hyp["hypothesis"] == res["hypothesis"]:
                        res["count"] += hyp["count"]
                        res["examples"] += hyp["examples"]
                        flag = True
                        break
                if not flag:
                    print(f"Adding new cluster {hyp['hypothesis']}")
                    results.append(hyp)
                
        # Add the results to the list
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
    for cluster in clusters:
        wandb.log({cluster["hypothesis"]: wandb.Table(dataframe=pd.DataFrame(cluster["examples"], columns=["examples"]))})

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
    response, clusters = batch_cluster_with_gpt(df['hypothesis'][:30].tolist(), batch_size=10)

    log_clusters(clusters)