import pandas as pd
import numpy as np
import ast
import random
import wandb
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from serve.utils_llm import get_llm_output, get_llm_embedding

systems_prompt = "Given a dataset of text outputs from two different large language models (LLMs), your task is to analyze and summarize the data based on specific characteristics. The goal is to identify and cluster similar behaviors or traits within the outputs, summarizing these into a concise list of commonly observed behaviors for each model. This analysis will help in understanding the general behaviors of these models for auditing, error discovery, and comparison purposes. Your outputs adhere to the format given by the user."
smaller_systems_prompt = "You are a helpful assistant. Your outputs adhere to the format given by the user."

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

class Reducer:
    """Given a set of text, reduce the set to a smaller set of hypotheses"""
    def __init__(self, args):
        self.args = args
        random.seed(args.seed)

    @staticmethod
    def reduce_texts(texts, n_clusters=5, cluster_method="dbscan", kwargs={}):
        """
        Given a list of hypotheses, reduce them to a smaller set of hypotheses
        Return a list of reduced hypotheses along with a list of (text, parent_text) pairs and any tables we should be logging
        """
        return texts, [(t, t) for t in texts], []
    
    def reduce(self, hypotheses):
        return self.reduce_texts(hypotheses, n_clusters=self.args.k, cluster_method=self.args.cluster_method)
    
#####################################
### Clustering and Axis Reduction ###
#####################################

def get_clusters(texts, embedding_model, n_clusters=5, cluster_method="dbscan", kwargs={}):
    """
    Clusters text into k clusters using hierarchical clustering 
    Reutrns a dictionary of clusters where the key is the cluster number and the value is a list of texts in that cluster.
    """
    print(texts)
    if embedding_model == 'all-MiniLM-L6-v2':
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts)
    else:
        print(f"Using LLM embeddings ({embedding_model}) for clustering")
        embeddings = np.stack([get_llm_embedding(d, embedding_model) for d in texts])

    clusters = cluster_hierarchical(embeddings, n_clusters=n_clusters)

    # Group axes by cluster
    grouped_axes = {i: [] for i in range(n_clusters)}
    for axis, cluster in zip(texts, clusters):
        grouped_axes[cluster].append(axis)
    return grouped_axes

def get_cluster_axes(cluster, model='gpt-4', batch = 50):
    cluster_axes_descriptions_prompt = ["""The following are the axes of variation that you can consider when comparing the two model outputs along with a description of how two models (A and B) vary along that axis. Each axis has a name as well as a description of what it means to be low and high on this axis. Many of these axes of variations could be named incorrectly or redundant with other axes. I want to cluster these axes so that I can better understand the general patterns seen in these models without having to look through so many axes. Please cluster this large list of axes into a minimal set of parent axes that cover the entire axis list. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Your new set of axes should be distinct so each of the above axes fit under exactly one of your new axes.
                        
    Here are the axes of varaiation (note each axis is formatted {{axis name}}: High: {{high description}} Low: {{low description}}):
    {axes}

    Again I want to cluster these axes into a minimal set of parent axes that cover the entire axis list. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Your new set of axes should be distinct so each of the above axes fit under exactly one of your new axes. Please ensure each axis and parent axis contains an axis name and descriptions of what it means to score high or low on that axis in the same format as the provided axes.  Please ensure the descriptions of what is considered high and low on each axis is clear, concise, under 10 words. Please focus on patterns that are important for understanding the behavior of a language model, as these will later be used to help debug an important system""", 
                                    
    """thanks! Now can you please convert this into a list that I can parse in python? Here are the original axes again for reference:
    {axes}

    Please structure your response as a list which can be parsed with ast.literal_eval() in Python. The format should be as follows:

    ["{{axis name}}:  High: {{new axis high description}} Low: {{new axis low description}}", ...]"""]

    conversion_prompt = """I have a list of axes that I would like to convert into a list that I can parse in python. Here are the original axes again for reference:
    
    {}
    
    My goal is to convert this into a list that I can parse with  with ast.literal_eval() in python. The format should be as follows:"""
    smaller_systems_prompt = "You are a helpful assistant. Your outputs adhere to the format given by the user."
    cluster = set(cluster)
    cluster_batch = random.sample(cluster, min(batch, len(cluster)))
    cluster_batch = sorted(cluster_batch)

    prompt_1 = cluster_axes_descriptions_prompt[0].format(axes="\n".join(cluster_batch))
    cluster_1_reduced_axes = get_llm_output(prompt_1, model=model, system_prompt=smaller_systems_prompt)
    cluster_1_reduced_axes = cluster_1_reduced_axes.replace("*", "")

    history = [{"role": "user", "content": prompt_1}, {"role": "assistant", "content": cluster_1_reduced_axes}]
    prompt_2 = cluster_axes_descriptions_prompt[1].format(axes="\n".join(cluster_batch))
    cluster_1_reduced_axes_categorized = get_llm_output(prompt_2, model=model, system_prompt=smaller_systems_prompt, history=history)
    # cut any thing before the [ and after the ]
    cluster_1_reduced_axes_categorized = cluster_1_reduced_axes_categorized[cluster_1_reduced_axes_categorized.find("["):cluster_1_reduced_axes_categorized.rfind("]") + 1]
    cluster_1_reduced_axes = ast.literal_eval(cluster_1_reduced_axes_categorized)
    # except:
    #     history = [{"role": "user", "content": prompt_1}, {"role": "assistant", "content": cluster_1_reduced_axes}]
    #     prompt_2 = cluster_axes_descriptions_prompt[1].format(axes="\n".join(cluster_batch))
    #     cluster_1_reduced_axes_categorized = get_llm_output(prompt_2, model=model, system_prompt=smaller_systems_prompt, history=history, cache=False)
    #     # cut any thing before the [ and after the ]
    #     cluster_1_reduced_axes_categorized = cluster_1_reduced_axes_categorized[cluster_1_reduced_axes_categorized.find("["):cluster_1_reduced_axes_categorized.rfind("]") + 1]
    #     cluster_1_reduced_axes = ast.literal_eval(cluster_1_reduced_axes_categorized)

    return prompt_1, cluster_1_reduced_axes

def match_axis_to_subaxis(axes, parent_axes, embedding_model):
    if embedding_model == 'all-MiniLM-L6-v2':
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Generate embeddings
        axes_embeddings = model.encode(axes)
        parent_axes_embeddings = model.encode(parent_axes)
    else:
        print(f"Using LLM embeddings ({embedding_model}) for clustering")
        # embeddings = np.stack([get_llm_embedding(d, embedding_model) for d in all_axis_descriptions])
        axes_embeddings = np.stack([get_llm_embedding(d, embedding_model) for d in axes])
        parent_axes_embeddings = np.stack([get_llm_embedding(d, embedding_model) for d in parent_axes])


    # Function to find the closest parent axis for each axis
    def find_closest_parent(axes_embeddings, parent_axes_embeddings):
        similarity_matrix = cosine_similarity(axes_embeddings, parent_axes_embeddings)
        closest_parent_indices = np.argmax(similarity_matrix, axis=1)
        return [parent_axes[index] for index in closest_parent_indices]

    # Categorize each axis
    categorized_axes = find_closest_parent(axes_embeddings, parent_axes_embeddings)
    return categorized_axes

def simplify_axes(parent_axes):
    remove_duplicates = """Below is a list of axes with a description of what makes a piece of text low or high on this axis. Are there any axes that have similar meanings based off their low and high descriptions? Could any of the low and high descriptions be simplified? Please remove any axes with roughly the same meaning and simplify the descriptions of what makes a piece of text low or high on this axis. Please ensure that the descriptions of what makes a piece of text low or high on this axis are distinct and mutually exclusive such that given any pair of text outputs, a human could easily and reliably determine which model is higher or lower on that axis. 

Here is the list of axes:
{axes}

Please return the simplified list of axes with any redundant axes removed and the descriptions of what makes a piece of text low or high on this axis simplified. Please maintain the format of the original axes and return a list like ["{{axis_name}}: High: {{high description}} Low: {{low description}}", ...]. I should be able to parse this output into a string using ast.literal_eval. If the original list does not contain any redundant axes, please return the original list."""
    prompt = remove_duplicates.format(axes="\n".join(parent_axes))
    old_parent_axes = parent_axes
    for i in range(3):
        try:
            response = get_llm_output(prompt, model="gpt-4", system_prompt=smaller_systems_prompt)
            print(f"\n\n{response}\n\n")
            parent_axes = ast.literal_eval(response[response.find("["):response.rfind("]") + 1])
            print(f"\n\nParent axes before: {old_parent_axes}\n\nParent axes after: {parent_axes}\n\n")
        except:
            print(f"Error in iteration {i}")
            continue
    return parent_axes

class AxisReducer(Reducer):

    def __init__(self, args):
        super().__init__(args)
        random.seed(args.seed)

    @staticmethod
    def reduce_texts(texts, 
                     embedding_model="text-embedding-3-small", 
                     summarization_model="gpt-4",
                     n_clusters=5, 
                     cluster_method="dbscan",
                     kwargs={}):
        """
        Given a list of hypotheses, reduce them to a smaller set of hypotheses
        Return a list of reduced hypotheses along with a list of (text, parent_text) pairs
        """
        grouped_axes = get_clusters(texts, embedding_model, n_clusters=n_clusters, cluster_method=cluster_method) 
        all_df_cluster, llm_logs = [], {}
        for cluster, axes in grouped_axes.items():
            prompt_1, parent_axes = get_cluster_axes(sorted(axes), model=summarization_model)
            llm_logs[cluster] = {"prompt_1": prompt_1, "output_1": parent_axes}
            df_cluster = {"axis": [], "cluster": []}
            for axis in parent_axes:
                df_cluster['axis'].append(axis)
                df_cluster['cluster'].append(cluster + 1)
            # all_cluster_axes.append(cluster_axes)
            all_df_cluster.append(pd.DataFrame(df_cluster))
            print(f"Cluster {cluster + 1} (length = {len(axes)}) (df length = {len(df_cluster)}):")
            print("")  # New line for readability between clusters

        llm_outputs = pd.DataFrame(llm_logs).T
        df_cluster = pd.concat(all_df_cluster)
        print(df_cluster)
        parent_axes = df_cluster['axis'].unique()
        parent_axes = simplify_axes(parent_axes)
        print("Parent Axes before simplification:", parent_axes)
        print("Parent axes AFTER simplification:", parent_axes)
        # match each axis to its parent
        categorized_axes = match_axis_to_subaxis(texts, parent_axes, embedding_model)
        df_cluster['parent_axis'] = match_axis_to_subaxis(list(df_cluster['axis']), parent_axes, embedding_model)
        print("Categorized axes:", categorized_axes)
        # sort categories by frequency
        ordered_parent_axes = df_cluster['parent_axis'].value_counts().index.tolist()
        wandb.summary["num_axes"] = len(texts)
        wandb.summary["num_parent_axes"] = len(parent_axes)

        return ordered_parent_axes, categorized_axes, {"df_cluster": df_cluster, "llm_outputs": llm_outputs}

    def reduce(self, hypotheses):
        return self.reduce_texts(hypotheses, n_clusters=self.args.k, cluster_method=self.args.cluster_method, embedding_model=self.args.embedding_model)
    

    