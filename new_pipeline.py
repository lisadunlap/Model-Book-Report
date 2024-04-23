import pandas as pd
import os
from PIL import Image
import json

from serve.utils_llm import get_llm_output, get_llm_embedding
import ast
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb 
import re
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from components.proposer_prompts import *
from components.parsing_utils import *

systems_prompt = "Given a dataset of text outputs from two different large language models (LLMs), your task is to analyze and summarize the data based on specific characteristics. The goal is to identify and cluster similar behaviors or traits within the outputs, summarizing these into a concise list of commonly observed behaviors for each model. This analysis will help in understanding the general behaviors of these models for auditing, error discovery, and comparison purposes. Your outputs adhere to the format given by the user."
smaller_systems_prompt = "You are a helpful assistant. Your outputs adhere to the format given by the user."

def get_cluster_axes(cluster, batch = 50):
    cluster_axes_descriptions_prompt = ["""The following are the axes of variation that you can consider when comparing the two model outputs along with a description of how two models (A and B) vary along that axis. Each axis has a name as well as a description of what it means to be low and high on this axis. Many of these axes of variations could be named incorrectly or redundant with other axes. I want to cluster these axes so that I can better understand the general patterns seen in these models without having to look through so many axes. Please cluster this large list of axes into a minimal set (<=5) of parent axes that cover the entire axis list. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Your new set of axes should be distinct so each of the above axes fit under exactly one of your new axes.
                        
    Here are the axes of varaiation (note each axis is formatted {{axis name}}: High: {{high description}} Low: {{low description}}):
    {axes}

    Again I want to cluster these axes into a minimal set (<=5) of parent axes that cover the entire axis list. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Your new set of axes should be distinct so each of the above axes fit under exactly one of your new axes. Please ensure each axis and parent axis contains an axis name and descriptions of what it means to score high or low on that axis in the same format as the provided axes.  Please ensure the descriptions of what is considered high and low on each axis is clear, concise, under 10 words. Please focus on patterns that are important for understanding the behavior of a language model, as these will later be used to help debug an important system""", 
                                     
    """thanks! Now can you please convert this into a list that I can parse in python? Here are the original axes again for reference:
    {axes}

    Please structure your response as a list which can be parsed with ast.literal_eval() in Python. The format should be as follows:

    ["{{axis name}}:  High: {{new axis high description}} Low: {{new axis low description}}", ...]"""]
    smaller_systems_prompt = "You are a helpful assistant. Your outputs adhere to the format given by the user."
    cluster_batch = random.sample(cluster, min(batch, len(cluster)))
    cluster_batch = sorted(cluster_batch)

    prompt_1 = cluster_axes_descriptions_prompt[0].format(axes="\n".join(cluster_batch))
    cluster_1_reduced_axes = get_llm_output(prompt_1, model="gpt-4", system_prompt=smaller_systems_prompt)
    cluster_1_reduced_axes = cluster_1_reduced_axes.replace("*", "")

    history = [{"role": "user", "content": prompt_1}, {"role": "assistant", "content": cluster_1_reduced_axes}]
    prompt_2 = cluster_axes_descriptions_prompt[1].format(axes="\n".join(cluster_batch))
    cluster_1_reduced_axes_categorized = get_llm_output(prompt_2, model="gpt-4", system_prompt=smaller_systems_prompt, history=history)
    # cut any thing before the [ and after the ]
    cluster_1_reduced_axes_categorized = cluster_1_reduced_axes_categorized[cluster_1_reduced_axes_categorized.find("["):cluster_1_reduced_axes_categorized.rfind("]") + 1]
    cluster_1_reduced_axes = ast.literal_eval(cluster_1_reduced_axes_categorized)

    return prompt_1, cluster_1_reduced_axes
        
remove_duplicates = """Below is a list of axes with a description of what makes a piece of text low or high on this axis. Are there are duplicates in this list? Could any of the low and high descriptions be simplified? Please remove any duplicates and simplify the descriptions of what makes a piece of text low or high on this axis. Please ensure that the descriptions of what makes a piece of text low or high on this axis are distinct and mutually exclusive such that given any pair of text outputs, a human could easily and reliably determine which model is higher or lower on that axis. 

Here is the list of axes:
{axes}

Please return the list of axes with any duplicates removed and the descriptions of what makes a piece of text low or high on this axis simplified. Please maintain the format of the original axes and return a list like ["{{axis_name}}: High: {{high description}} Low: {{low description}}", ...]. I should be able to parse this output into a string using ast.literal_eval."""

def match_axis_to_subaxis(axes, parent_axes, embedding_model):
    # Load a pre-trained model

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

def cluster_hierarchical(embeddings, n_clusters=5):
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings)
    unique_labels = np.unique(clustering.labels_)
    print({i: np.sum(clustering.labels_ == i) for i in unique_labels})
    return clustering.labels_

def cluster_kmeans(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed)
    kmeans.fit(embeddings)
    return kmeans.labels_

def get_score(row, eval_axes, dummy_eval=False):
    if dummy_eval:
        return "Model A Score: high\nModel B Score: low\nReason: Because I said so."
    else:
        scoring = """I am trying to explain differences in the behavior of two LLM's (A and B) by comparing their outputs over a dataset of question answer tuples. I have of found axes of variation with the meanings of what it means to be low and high on this axis.

        For the following question answer tuple, please score the two models on the following axis of variation found in the dataset. The axis of variation is as follows:
        {axes}

        Here is the question answer tuple:
        {question}

        Please score where the two models fall on the above axis. The score for a given model could be ("low", "high").This will help me understand the differences between the two models in a more structured way. Please return the score followed by an explanantion of your thought process in the format:
        Model A Score: {{high/low}}
        Model B Score: {{high/low}}
        Reason: {{reasoning}}

        """
        if row['parent_axis'] not in eval_axes:
            return None
        scoring_prompt = scoring.format(axes=row["parent_axis"], question=row["prompt"])
        scoring_output = get_llm_output(scoring_prompt, model="gpt-3.5-turbo")
        return scoring_output
    
def get_embedding_score(axis_low, axis_high, embeddings_a, embeddings_b):
    # compute similarity between the embeddings of the low and high descriptions of the axis
    low_embedding = np.expand_dims(np.array(get_llm_embedding(axis_low, "text-embedding-3-small")), axis=0)
    high_embedding = np.expand_dims(np.array(get_llm_embedding(axis_high, "text-embedding-3-small")), axis=0)
    embeddings_a = np.expand_dims(np.array(embeddings_a), axis=0)
    embeddings_b = np.expand_dims(np.array(embeddings_b), axis=0)

    #normalize the embeddings
    low_embedding = low_embedding / np.linalg.norm(low_embedding)
    high_embedding = high_embedding / np.linalg.norm(high_embedding)
    embeddings_a = embeddings_a / np.linalg.norm(embeddings_a)
    embeddings_b = embeddings_b / np.linalg.norm(embeddings_b)

    low_similarity_a = cosine_similarity(embeddings_a, low_embedding)
    high_similarity_a = cosine_similarity(embeddings_a, high_embedding)
    low_similarity_b = cosine_similarity(embeddings_b, low_embedding)
    high_similarity_b = cosine_similarity(embeddings_b, high_embedding)
    return {"low_similarity_a": low_similarity_a[0][0], "high_similarity_a": high_similarity_a[0][0], "low_similarity_b": low_similarity_b[0][0], "high_similarity_b": high_similarity_b[0][0]}

import argparse
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--wandb', action='store_true', help='log to wandb')
    parser.add_argument('--num-samples', type=int, help='number of samples to use')
    parser.add_argument('--data-path', type=str, default='data/all.csv', help='path to data')
    parser.add_argument('--model-a-column', type=str, default='human_answers', help='column name for model A')
    parser.add_argument('--model-b-column', type=str, default='chatgpt_answers', help='column name for model B')
    parser.add_argument('--k', type=int, default=3, help='number of clusters')
    parser.add_argument('--batch-size', default=50, type=int, help='batch size for LLM')
    parser.add_argument('--num-eval', default=3, type=int, help='model to use')
    parser.add_argument('--oz', action='store_true', help='use oz prompt')
    parser.add_argument('--dummy-eval', action='store_true', help='use dummy eval prompt')
    parser.add_argument('--embedding-model', type=str, default='all-MiniLM-L6-v2', help='embedding model to use')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--group-column', type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # tirn off wandb logging
    if not args.wandb:
        os.environ["WANDB_MODE"] = "dryrun"
    proj_name = "llm_eval_presentable" if not args.dummy_eval else f"llm_eval_refactor_debug"
    global_df = pd.read_csv(args.data_path)

    if args.group_column:
        groups = global_df[args.group_column].unique()
    else:
        groups = ["all"]
    for group in groups:
        if args.group_column:
            df = global_df[global_df[args.group_column] == group]
        else:
            df = global_df
        model_group = f"{args.model_a_column}_{args.model_b_column}"
        wandb.init(project=proj_name, entity="lisadunlap", config=vars(args), group=model_group, name=group)

        # create str of datapath for savins
        save_str = args.data_path.split("/")[-1].split(".")[0] + f"_{group}"
        tag = f"{args.model_a_column}_{args.model_b_column}_{args.k}" if not args.num_samples else f"{args.model_a_column}_{args.model_b_column}_{args.k}_{args.num_samples}"
        tag = f"{tag}_oz" if args.oz else tag
        tag = f"{tag}_dummy_eval" if args.dummy_eval else tag
        if not os.path.exists(f"pipeline_results/{save_str}"):
            os.makedirs(f"pipeline_results/{save_str}")

        # randomly sample 10 rows, set random seed for reproducibility
        if args.num_samples:
            # df.drop_duplicates(subset=[args.model_a_column, args.model_b_column], inplace=True)
            old_len = df.shape[0]
            # filter out rows where the model outputs are similar
            df['similarity'] = df.apply(lambda x: fuzz.ratio(x[args.model_a_column], x[args.model_b_column]), axis=1)
            df = df[df['similarity'] < 80]
            print(f"Filtered out {old_len - df.shape[0]} rows")
            # remove any entired where the model outputs are the same
            df = df[df[args.model_a_column] != df[args.model_b_column]]
            df = df.sample(args.num_samples, random_state=args.seed)
            # df[f"{args.model_a_column}_embedding"] = df[["question", args.model_a_column]].apply(lambda x: get_llm_embedding(f"User:{x['question']}\Assistant:{x[args.model_a_column]}", args.embedding_model), axis=1)
            # df[f"{args.model_b_column}_embedding"] = df[["question", args.model_b_column]].apply(lambda x: get_llm_embedding(f"User:{x['question']}\Assistant:{x[args.model_b_column]}", args.embedding_model), axis=1)

        model_columns = [args.model_a_column, args.model_b_column]
        oz_axes = ["Tone", "Format", "Level of Detail", "Ability to answer", "Safety", "Approach", "Creativity", "Fluency and grammatical correctness", "Adherence to prompt"]

        ######################################
        #### get per question differences ####
        ######################################
        results = {"question":[], "answer_a":[], "answer_b":[], "prompt": [], "response": [], "axes": [], "axis_response": []}
        
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            texts = f"{row['question']}\nModel A: {row[model_columns[0]]}\nModel B: {row[model_columns[1]]}\n"
            if args.oz:
                prompt = OZ_PROMPT.format(text=texts, axes="\n".join([f"* {axis}" for axis in oz_axes]))
            else:
                prompt = DEFAULT_PROMPT.format(text=texts)
            response = get_llm_output(prompt, model="gpt-3.5-turbo", system_prompt=systems_prompt, trace_name="per question differences").replace("**", "")
            results["prompt"].append(texts)
            results["question"].append(row['question'])
            results["answer_a"].append(row[model_columns[0]].strip('[]'))
            results["answer_b"].append(row[model_columns[1]].strip('[]'))
            results["response"].append(response)
            results["axes"].append(extract_entities(response))
            axis_prompt = AXIS_CONVERSION.format(axes=response)
            axis_response = get_llm_output(axis_prompt, model="gpt-3.5-turbo", system_prompt=smaller_systems_prompt, trace_name="convert per question axes")
            results["axis_response"].append(axis_response)
        # save results
        results = pd.DataFrame(results)
        wandb.log({"per_sample_differences": wandb.Table(dataframe=results)})

        results.to_csv(f"pipeline_results/{save_str}/{tag}-per_question_results.csv", index=False)
        # remove any rows where is_match(results['response'], "No differences found") == True
        results = results[results['response'].apply(lambda x: "No differences found" not in x)]

        # cluster per axis differences
        results['axis_description'] = results['axis_response'].apply(extract_axis_descriptions)
        results = results.explode('axis_description')

        ######################################
        #### cluster per question axes    ####
        ######################################
        all_axis_descriptions = list(set(results['axis_description']))
        all_axis_descriptions = [x.replace("*", "") for x in all_axis_descriptions]
        # all_axis_descriptions = [desc.split(": ", 1)[1] for desc in all_axis_descriptions]
        # Load a pre-trained sentence transformer model
        if args.embedding_model == 'all-MiniLM-L6-v2':
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(all_axis_descriptions)
        else:
            print(f"Using LLM embeddings ({args.embedding_model}) for clustering")
            embeddings = np.stack([get_llm_embedding(d, args.embedding_model) for d in all_axis_descriptions])

        # clusters = cluster_kmeans(embeddings, n_clusters=args.k)
        clusters = cluster_hierarchical(embeddings, n_clusters=args.k)

        # Group axes by cluster
        grouped_axes = {i: [] for i in range(args.k)}
        for axis, cluster in zip(all_axis_descriptions, clusters):
            grouped_axes[cluster].append(axis)


        all_cluster_axes, all_df_cluster, llm_logs = [], [], {}
        for cluster, axes in grouped_axes.items():
            prompt_1, parent_axes = get_cluster_axes(sorted(axes))
            llm_logs[cluster] = {"prompt_1": prompt_1, "output_1": parent_axes}
            df_cluster = {"axis": [], "cluster": []}
            for axis in parent_axes:
                df_cluster['axis'].append(axis)
                df_cluster['cluster'].append(cluster + 1)
            # all_cluster_axes.append(cluster_axes)
            all_df_cluster.append(pd.DataFrame(df_cluster))
            print(f"Cluster {cluster + 1} (length = {len(axes)}) (df length = {len(df_cluster)}):")
            print("")  # New line for readability between clusters

        df_cluster = pd.concat(all_df_cluster)
        parent_axes = df_cluster['axis'].unique()
        parent_axes = list(set(df_cluster['axis']))
        old_parent_axes = parent_axes
        # remove/simplify axes 
        prompt = remove_duplicates.format(axes="\n".join(parent_axes))
        for i in range(3):
            try:
                response = get_llm_output(prompt, model="gpt-4", system_prompt=smaller_systems_prompt)
                print(f"\n\n{response}\n\n")
                parent_axes = ast.literal_eval(response[response.find("["):response.rfind("]") + 1])
                print(f"\n\nParent axes before: {old_parent_axes}\n\nParent axes after: {parent_axes}\n\n")
            except:
                print(f"Error in iteration {i}")
                continue
        results['parent_axis'] = match_axis_to_subaxis(list(results['axis_description']), parent_axes, args.embedding_model)
        df_cluster['parent_axis'] = match_axis_to_subaxis(list(df_cluster['axis']), parent_axes, args.embedding_model)
        wandb.summary["num_axes"] = len(all_axis_descriptions)
        wandb.summary["num_parent_axes"] = len(parent_axes)
        df_cluster.to_csv(f"pipeline_results/{save_str}/{tag}-clustering.csv", index=False) 
        results.to_csv(f"pipeline_results/{save_str}/{tag}-results.csv", index=False)
        llm_outputs = pd.DataFrame(llm_logs).T
        llm_outputs.to_csv(f"pipeline_results/{save_str}llm_outputs.csv", index=False)

        results['parent_axis_deets'] = results['parent_axis'].apply(parse_high_low) # returns {"parent_axis_name": "error", "parent_high": "error", "parent_low": "error"}
        results = pd.concat([results.drop(['parent_axis_deets'], axis=1), results['parent_axis_deets'].apply(pd.Series)], axis=1)


        ######################################
        ############  score axes  ############
        ######################################
        def extract_scores(text):
            text = text.replace("*", "")
            # Create a dictionary to hold the results
            results = {}
            
            # Regex patterns to match the scores and reasoning
            score_pattern = re.compile(r'Model (A|B) Score: (high|low)', re.IGNORECASE)
            reasoning_pattern = re.compile(r'Reason:\s*({{reasoning}})', re.IGNORECASE)

            # Find all matches for model scores
            scores = score_pattern.findall(text)
            for model, score in scores:
                if model.upper() == 'A':
                    results["Model A Score"] = score.lower()
                elif model.upper() == 'B':
                    results["Model B Score"] = score.lower()
            try:
                if 'high' in results["Model A Score"].lower() and 'low' in results["Model B Score"].lower():
                    return 1
                elif 'low' in results["Model A Score"].lower() and 'high' in results["Model B Score"].lower():
                    return -1
                elif "low" in results["Model A Score"].lower() and "low" in results["Model B Score"].lower():
                    return 0
                elif "high" in results["Model A Score"].lower() and "high" in results["Model B Score"].lower():
                    return 0
                else:
                    raise ValueError(f"No score found\n{text}")
            except:
                print(f"No score found\n{text}")        

        def ensemble_scores(scores):
            score_1, score_2 = extract_scores(scores[0]), extract_scores(scores[1])
            # Utility function to ensemble scores
            # If they disagree, return 0
            return (score_1 + -1 * score_2)/2, score_1 == score_2 and score_1 != 0


        def get_score2(row, eval_axes, dummy_eval=False):
            if dummy_eval:
                return ["Model A Score: high\nModel B Score: low\nReason: Because I said so.", "Model A Score: high\nModel B Score: low\nReason: Because I said so."]
            else:
                if row['parent_axis'] not in eval_axes:
                    return None

                # Original prompt
                prompt_a = f"Question:{row['question']}\nModel A: {row['answer_a']}\nModel B: {row['answer_b']}\n"
                # Swapped prompt
                prompt_b = f"Question:{row['question']}\nModel A: {row['answer_b']}\nModel B: {row['answer_a']}\n"

                scoring = """I am trying to explain differences in the behavior of two LLM's (A and B) by comparing their outputs over a dataset of question answer tuples. I have of found axes of variation with the meanings of what it means to be low and high on this axis.

                For the following question answer tuple, please score the two models on the following axis of variation found in the dataset. The axis of variation is as follows:
                {axes}

                Here is the question answer tuple:
                {question}

                Please score where the two models fall on the above axis. The score for a given model could be ("low", "high").This will help me understand the differences between the two models in a more structured way. Please provide your thought process when scoring the models before providing the score. Please respond in the following format:
                Reasoning: {{reasoning}}
                Model A Score: {{high/low}}
                Model B Score: {{high/low}}

                """

                # Generate scoring prompts for both orderings
                scoring_prompt_a = scoring.format(axes=row["parent_axis"], question=prompt_a)
                scoring_prompt_b = scoring.format(axes=row["parent_axis"], question=prompt_b)

                # Get LLM outputs for both prompts
                scoring_output_a = get_llm_output(scoring_prompt_a, model="gpt-3.5-turbo", system_prompt=smaller_systems_prompt)
                scoring_output_b = get_llm_output(scoring_prompt_b, model="gpt-3.5-turbo", system_prompt=smaller_systems_prompt)
                print(scoring_output_a)
                print(scoring_output_b)

                # Ensemble scores and return
                return [scoring_output_a, scoring_output_b]
            
        results["parsed_axis_responses"] = results[['axis_response', 'axis_description']].apply(lambda x: parse_axis_responses(x['axis_response'], x['axis_description']), axis=1)

        #turn the values in the parsed_axis_responses column into separate columns
        results = pd.concat([results.drop(['parsed_axis_responses'], axis=1), results['parsed_axis_responses'].apply(pd.Series)], axis=1)
        parent_axes_logging = results['parent_axis'].value_counts().reset_index()
        eval_axes = results['parent_axis'].value_counts()[:args.num_eval].index.tolist()
        print(f"\n\n{results['parent_axis'].value_counts()}\n{eval_axes}\n\n")

        # get score after parent axis generation
        results['ensamble_scores'] = results.apply(lambda x: get_score2(x, eval_axes=eval_axes, dummy_eval=args.dummy_eval), axis=1)
        results = results.dropna(subset=["ensamble_scores"])
            
        results["one_sided_score_and_reasoning"] = results["ensamble_scores"].apply(lambda x: x[0])
        results["one_sided_score"] = results["one_sided_score_and_reasoning"].apply(extract_scores)
        results["final_score_and_reasoning"] = results["ensamble_scores"]
        results["final_score"] = results["ensamble_scores"].apply(ensemble_scores)
        results["scores_disagree"] = results["final_score"].apply(lambda x: x[1])
        results["final_score"] = results["final_score"].apply(lambda x: x[0])
        wandb.summary["percentage_scores_disagree"] = results["scores_disagree"].sum() / results.shape[0]
        wandb.summary["percentage_neutral"] = results[((results["final_score"] == 0) & (results["scores_disagree"] == False))].shape[0] / results.shape[0]
        results.ensamble_scores.value_counts()
        results.to_csv(f"pipeline_results/{save_str}/{tag}-results.csv", index=False)
        for c in llm_outputs.columns:
            llm_outputs[c] = llm_outputs[c].astype(str)

        summary_results = results.groupby('parent_axis').agg({'final_score': 'mean', 'one_sided_score': 'mean', 'question': 'count'}).reset_index()
        wandb.log({"results": wandb.Table(dataframe=results), "df_cluster": wandb.Table(dataframe=df_cluster), "llm_outputs": wandb.Table(dataframe=llm_outputs), "summary_results": wandb.Table(dataframe=summary_results), "all_parent_axes": wandb.Table(dataframe=parent_axes_logging)})

# make main function
if __name__ == "__main__":
    main()
