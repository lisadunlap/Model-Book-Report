# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# import numpy as np
# import pandas as pd
# import wandb
# import os
# from tqdm import tqdm
# import clip
# import torchvision.transforms as transforms
# from PIL import Image
# import re
# import logging

# from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering
# from scipy.spatial.distance import cdist
# from InstructorEmbedding import INSTRUCTOR

# logging.basicConfig(level=logging.INFO)
# import torch
# import torchvision

# from serve.utils_llm import get_llm_output, get_llm_embedding

# # example_generation_prompt = """Given the following example prompts, generate a new prompt that is from a similar domain.

# # Examples:
# # {prompts}

# # Please generate a new prompt that is similar to the examples provided and that is liekeley to lead to an answer which is under 50 words.

# # PROMPT:"""
    
    
# def get_example_prompt(prompts):
#     """
#     Given a list of prompts, generate a new prompt which is similar to the examples.
#     """
#     example_generation_prompt = """Given the following example prompts, generate a new prompt that is from a similar domain.

#     Examples:
#     {prompts}

#     Please generate a new prompt that is similar to the examples provided and that is likely to lead to an answer which is under 20 words.

#     PROMPT:"""

#     prompt = example_generation_prompt.format(prompts="\n".join(prompts[:5]))
#     # prompt = example_generation_prompt.format(prompts="/n".join(prompts)s)
#     response = get_llm_output(prompt, model="gpt-4")
#     return {"example_generation_prompt": prompt, "response": response}
    

# class Sampler:

#     def __init__(self, args):
#         self.args = args
    
#     def sample(self, df):
#         print(df.columns)
#         if self.args.group_column:
#             samples, question_generation_logs = [], []
#             return df[self.args.group_column], {}
#         else:
#             return ["all"] * len(df), {}
        
# def cluster_hierarchical(embeddings, n_clusters=5):
#     print(embeddings.shape)
#     clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings)
#     unique_labels = np.unique(clustering.labels_)
#     print({i: np.sum(clustering.labels_ == i) for i in unique_labels})
#     return clustering.labels_

# class ClusterSampler(Sampler):

#     def __init__(self, args):
#         super().__init__(args)

#     def summarize_group_text(self, texts, model="gpt-4-0125-preview"):
#         """
#         Summarize the given text using OpenAI's GPT-3.5-turbo API.
#         """
#         prompt = f"""I have clustered a larget set of question by their text embedding and I would like to summarize each cluster. Below are examplars from each cluster:
        
#         {texts}
        
#         Output a concept which best describes each cluster in less than 5 words. Please think of a cluster concepts that are mutually exclusive and distinct. Given an unseen text, a human should be able to easily and reliably determine which cluster it belongs to. 

#         Please output your response as:

#         Cluster {{cluster name}}: {{cluster description}}
#         """
#         return get_llm_output(prompt, model=model)
    
#     def parse_clusters(self, cluster_strings):
#         cluster_dict = {}
#         # Pattern includes optional numeric prefixes (e.g., "1. "), optional symbols (-, *), and the word 'Cluster'
#         pattern = re.compile(r'(?:\d+\.\s*)?(?:[-*]\s*)?Cluster (\w+):\s*(.*)')
        
#         for line in cluster_strings.splitlines():
#             match = pattern.match(line)
#             if match:
#                 cluster_name, summary = match.groups()
#                 cluster_dict[cluster_name] = summary
        
#         return cluster_dict

#     def summarize_cluster(self, labels, num_clusters, embeddings, texts, K, prompt=None):
#         all_texts = []
#         # create list of nonsense labels to anonymize the cluster
#         anon_labels = ["fds", "nhg", "jkl", "qwe", "rty", "uio", "zxc", "vbn", "mnb", "poi"]
#         centroids = []
#         for custer_idx in range(num_clusters):
#             cluster = np.where(labels == custer_idx)[0]
#             sub_texts = np.array(texts)[np.where(labels == custer_idx)[0]]
#             # Calculate the centroid of the cluster
#             cluster_embeddings = embeddings[cluster]
#             centroid = np.mean(cluster_embeddings, axis=0)
#             centroids.append(centroid)

#             # Find the K texts closest to the centroid
#             distances = cdist([centroid], cluster_embeddings, metric='cosine')[0]
#             closest_indices = np.argsort(distances)[:50]
#             # randomly sample 5 texts
#             np.random.shuffle(closest_indices)
#             closest_indices = closest_indices[:K]

#             # Summarize these texts
#             selected_texts = [sub_texts[i] for i in closest_indices]
#             all_texts += [f"Cluster {anon_labels[custer_idx]}:\n" + "\n".join(selected_texts)]
#         # # shuffle all_texts
#         # np.random.shuffle(all_texts)
#         summarized_text = self.summarize_group_text("\n".join(all_texts))
#         print(summarized_text)
#         return centroids, summarized_text, self.parse_clusters(summarized_text)

#     def sample(self, df):
#         print(f"------------- Cluster Sampling -------------")
#         # df["embedding"] =  df["question"].apply(lambda x: get_llm_embedding(x, model=self.args.embedding_model))
#         model = INSTRUCTOR('hkunlp/instructor-xl')
#         instruction = "Cluster based on question topics and summarize the cluster."
#         df["embedding"] = df["question"].apply(lambda x: model.encode([[instruction,x]])[0])
#         embeddings = np.stack(df["embedding"].values)
#         n_clusters = self.args.num_topic_clusters
#         labels = cluster_hierarchical(embeddings, n_clusters=n_clusters)
#         anon_labels = ["fds", "nhg", "jkl", "qwe", "rty", "uio", "zxc", "vbn", "mnb", "poi"][:n_clusters]
#         centroid, summary, summary_dict = self.summarize_cluster(labels, n_clusters, embeddings, df["question"].tolist(), K=10)
#         print(summary_dict)
#         return [summary_dict[anon_labels[l]] for l in labels], {"centroids": centroid, "summary": list(summary_dict.values())}

# def match_set_to_centroids(df, topic_centroids={}):
#     if not topic_centroids:
#         return ["all"] * len(df)
#     # match each item in the set to the closest centroid
#     model = INSTRUCTOR('hkunlp/instructor-xl')
#     instruction = "Cluster based on question topics and summarize the cluster."
#     df["embedding"] = df["question"].apply(lambda x: model.encode([[instruction,x]])[0])
#     embeddings = np.stack(df["embedding"].values)
#     names, centroids = topic_centroids['summary'], np.stack(topic_centroids['centroids'])
#     print(f"{embeddings.shape} {centroids.shape}")
#     distances = cdist(embeddings, centroids, metric='euclidean')
#     # get euclidean distance
#     # distances = cdist(embeddings, centroids, metric='euclidean')
#     idxs = np.argmin(distances, axis=1)
#     return [names[i] for i in idxs]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import wandb
import os
from tqdm import tqdm
import clip
import torchvision.transforms as transforms
from PIL import Image
import random
import re
import logging

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering
from scipy.spatial.distance import cdist
from InstructorEmbedding import INSTRUCTOR

logging.basicConfig(level=logging.INFO)
import torch
import torchvision

from serve.utils_llm import get_llm_output, get_llm_embedding

def get_example_prompt(prompts):
    """
    Given a list of prompts, generate a new prompt which is similar to the examples.
    """
    example_generation_prompt = """Given the following example prompts, generate a new prompt that is from a similar domain.

    Examples:
    {prompts}

    Please generate a new prompt that is similar to the examples provided and that is likely to lead to an answer which is under 20 words.

    PROMPT:"""

    prompt = example_generation_prompt.format(prompts="\n".join(prompts[:5]))
    response = get_llm_output(prompt, model="gpt-4")
    return {"example_generation_prompt": prompt, "response": response}
    

class Sampler:

    def __init__(self, args):
        self.args = args
    
    def sample(self, df):
        print(df.columns)
        if self.args.group_column:
            samples, question_generation_logs = [], []
            df["topic"] = df[self.args.group_column]
            df["topic_label"] = df[self.args.group_column]
            df["embedding"] = [0] * len(df)
            return df, {}
        else:
            df["topic"] = ["all"] * len(df)
            df["topic_label"] = [0] * len(df)
            df["embedding"] = [0] * len(df)
            return df, {}
        
def cluster_hierarchical(embeddings, n_clusters=5):
    print(embeddings.shape)
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings)
    unique_labels = np.unique(clustering.labels_)
    print({i: np.sum(clustering.labels_ == i) for i in unique_labels})
    return clustering.labels_

class ClusterSampler(Sampler):

    def __init__(self, args):
        super().__init__(args)

    def summarize_group_text(self, texts, model="gpt-4-0125-preview"):
        """
        Summarize the given text using OpenAI's GPT-3.5-turbo API.
        """
        prompt = f"""I have clustered a large set of questions by their text embeddings and I would like to summarize each cluster. Below are examples from each cluster:
        
        {texts}
        
        Output a concept which best describes each cluster in less than 5 words. Please think of cluster concepts that are mutually exclusive and distinct. Given an unseen text, a human should be able to easily and reliably determine which cluster it belongs to. 

        Please output your response as:

        Cluster {{cluster name}}: {{cluster description}}
        """
        return get_llm_output(prompt, model=model)
    
    def parse_clusters(self, cluster_strings):
        cluster_dict = {}
        # Pattern includes optional numeric prefixes (e.g., "1. "), optional symbols (-, *), and the word 'Cluster'
        pattern = re.compile(r'(?:\d+\.\s*)?(?:[-*]\s*)?Cluster (\w+):\s*(.*)')
        
        for line in cluster_strings.splitlines():
            match = pattern.match(line)
            if match:
                cluster_name, summary = match.groups()
                cluster_dict[cluster_name] = summary
        
        return cluster_dict

    def summarize_cluster(self, labels, num_clusters, embeddings, texts, K, prompt=None):
        all_texts = []
        # create list of nonsense labels to anonymize the cluster
        anon_labels = ["fds", "nhg", "jkl", "qwe", "rty", "uio", "zxc", "vbn", "mnb", "poi"]
        centroids = []
        for custer_idx in range(num_clusters):
            cluster = np.where(labels == custer_idx)[0]
            sub_texts = np.array(texts)[np.where(labels == custer_idx)[0]]
            # Calculate the centroid of the cluster
            cluster_embeddings = embeddings[cluster]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroids.append(centroid)

            # Find the K texts closest to the centroid
            distances = cdist([centroid], cluster_embeddings, metric='cosine')[0]
            closest_indices = np.argsort(distances)[:50]
            # randomly sample 5 texts
            np.random.shuffle(closest_indices)
            closest_indices = closest_indices[:K]

            # Summarize these texts
            selected_texts = [sub_texts[i] for i in closest_indices]
            all_texts += [f"Cluster {anon_labels[custer_idx]}:\n" + "\n".join(selected_texts)]
        # # shuffle all_texts
        # np.random.shuffle(all_texts)
        summarized_text = self.summarize_group_text("\n".join(all_texts))
        print(summarized_text)
        return centroids, summarized_text, self.parse_clusters(summarized_text)

    def sample(self, df):
        print(f"------------- Cluster Sampling -------------")
        model = INSTRUCTOR('hkunlp/instructor-xl')
        instruction = "Cluster based on question topics and summarize the cluster."
        df["embedding"] = df["question"].apply(lambda x: model.encode([[instruction,x]])[0])
        embeddings = np.stack(df["embedding"].values)
        n_clusters = self.args.num_topic_clusters
        labels = cluster_hierarchical(embeddings, n_clusters=n_clusters)
        anon_labels = ["fds", "nhg", "jkl", "qwe", "rty", "uio", "zxc", "vbn", "mnb", "poi"][:n_clusters]
        centroids, summary, summary_dict = self.summarize_cluster(labels, n_clusters, embeddings, df["question"].tolist(), K=10)
        print(summary_dict)
        topic_centroids = {"centroids": centroids, "summary": list(summary_dict.values())}
        df['topic'] = [summary_dict[anon_labels[l]] for l in labels]
        df["topic_label"] = labels
        return df, topic_centroids

def match_set_to_centroids(df, topic_centroids={}, labels=None, embeddings=None):
    if not topic_centroids:
        return ["all"] * len(df)
    
    print(f"------------- Matching to Centroids -------------")
    print(f"Topic Centroids: {topic_centroids}")
    print(f"Labels: {labels}")
    print(f"Embeddings: {embeddings.shape}")    
    # Initialize the embedding model
    model = INSTRUCTOR('hkunlp/instructor-xl')
    instruction = "Cluster based on question topics and summarize the cluster."
    
    # Embed the questions
    df["embedding"] = df["question"].apply(lambda x: model.encode([[instruction, x]])[0])
    new_embeddings = np.stack(df["embedding"].values)
    print(f"Emeddings: {new_embeddings.shape} {embeddings.shape}")
    
    # Names of clusters from topic_centroids
    names = topic_centroids['summary']

    # Match each new embedding to the cluster by calculating distances to all points in each cluster
    assignments = []
    for new_emb in new_embeddings:
        min_avg_distance = float('inf')
        assigned_cluster = None
        for cluster_idx in range(len(names)):
            cluster_points = embeddings[labels == cluster_idx]
            distances = cdist([new_emb], cluster_points, metric='euclidean')
            avg_distance = np.mean(distances)
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                assigned_cluster = names[cluster_idx]
        assignments.append(assigned_cluster)
    
    print(assignments)
    return assignments

import re
import json
import ast
def extract_final_list(text):
    # Use regular expression to find the list in the text
    pattern = re.compile(r'\[\s*([^\]]+)\s*\]', re.DOTALL)
    match = pattern.search(text)
    
    if match:
        list_str = match.group(1)
        
        # Split the list elements and strip any extra whitespace
        parsed_list = [item.strip() for item in list_str.split(',')]
        
        return parsed_list
    else:
        print("No list found in the text.")
        print(text)
        return None

def classify_centroids(df, topics):
    if not topics:
        return ["all"] * len(df)
    prompt = """I would like to categorize the following questions into the topic clusters that were generated by the clustering algorithm. Below are the questions to be categorized:

    {batch}

    Please assign each question topic to the appropriate topic. The topics are as follows:

    {topics}

    Please output your response as a list of topic assignments for each question that I can parse into a python dictionary . Please only respond with the list of topic assignments and nothing else. Please only assign one topic to each question and format you response as:

    ["topic1", "topic2", "topic3", ...]
        """
    
    all_questions = df['question'].tolist()
    batch_size = 10
    assignments = []
    # shuffle topics w seed 1
    shuffle_topics = topics.copy()
    for i in range(0, len(all_questions), batch_size):
        batch = all_questions[i:i+batch_size]
        print(f"Batch: {batch}")
        random.shuffle(shuffle_topics)
        response = get_llm_output(prompt.format(batch=batch, topics=shuffle_topics), model="gpt-4o")
        print(response)
        assignments.extend(extract_final_list(response))
    
    return assignments
