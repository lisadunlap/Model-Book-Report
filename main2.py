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

# from components.proposer import LLMPairwiseProposerWithQuestion
# from components.reducer import AxisReducer
import components.ranker as rankers
import components.proposer as proposers
import components.reducer as reducers


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
    parser.add_argument('--embedding-model', type=str, default='text-embedding-3-small', help='embedding model to use')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--group-column', type=str)
    parser.add_argument('--cluster-method', type=str, default='hierarchical', help='clustering method')
    parser.add_argument('--ranker', type=str, default='LLMOnlyRanker', help='ranker to use')
    parser.add_argument('--proposer', type=str, default='LLMPairwiseProposerWithQuestion', help='proposer to use')
    parser.add_argument('--reducer', type=str, default='AxisReducer', help='reducer to use')
    parser.add_argument('--proposer-batch-size', type=str, default=5, help='batch of questions to get differences for')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # tirn off wandb logging
    if not args.wandb:
        os.environ["WANDB_MODE"] = "dryrun"
    proj_name = "llm_eval_presentable" if not args.dummy_eval else f"llm_eval_refactor_debug"
    global_df = pd.read_csv(args.data_path)
    # remove duplicate question-answer
    global_df.drop_duplicates(subset=[args.model_a_column, args.model_b_column], inplace=True)

    if args.group_column:
        groups = global_df[args.group_column].unique()
        print(f"Running VibeCheck on group {args.group_column}({groups})")
        print(f"Group value counts: {global_df[args.group_column].value_counts()}")
    else:
        groups = ["all"]
    for group in groups:
        if args.group_column:
            df = global_df[global_df[args.group_column] == group]
        else:
            df = global_df
        model_group = f"{args.model_a_column}_{args.model_b_column}"
        wandb.init(project=proj_name, entity="lisadunlap", config=vars(args), group=model_group, name=f"{args.group_column}-{group}")

        # create str of datapath for savins
        num_samples = min(args.num_samples, df.shape[0]) if args.num_samples else df.shape[0]
        save_str = args.data_path.split("/")[-1].split(".")[0] + f"_{group}"
        tag = f"{args.model_a_column}_{args.model_b_column}_{args.k}" if not args.num_samples else f"{args.model_a_column}_{args.model_b_column}_{args.k}_{num_samples}"
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
            df = df.sample(num_samples, random_state=args.seed)


        # ######################################
        # #### get per question differences ####
        # ######################################
        # proposer = LLMPairwiseProposerWithQuestion(args)
        proposer = getattr(proposers, args.proposer)(args)
        all_axis_descriptions, llm_logs, pairwise_differences, results = proposer.propose(df)
        wandb.log({"per_sample_differences": wandb.Table(dataframe=results), "pairwise_diff_llm_logs": wandb.Table(dataframe=llm_logs)})


        ######################################
        #### cluster per question axes    ####
        ######################################
        all_axis_descriptions = list(results['axis_description'])
        all_axis_descriptions = [x.replace("*", "") for x in all_axis_descriptions]

        # reducer = AxisReducer(args)
        reducer = getattr(reducers, args.reducer)(args)
        parent_axes, child_parent_map, tables = reducer.reduce(all_axis_descriptions)
        print("Len child parent", len(child_parent_map), len(results))
        results['parent_axis'] = child_parent_map
        wandb.log({k: wandb.Table(dataframe=v) for k, v in tables.items()})
        # results['parent_axis_deets'] = results['parent_axis'].apply(parse_high_low) # returns {"parent_axis_name": "error", "parent_high": "error", "parent_low": "error"}


        ######################################
        ############  score axes  ############
        ######################################
        eval_axes = results['parent_axis'].value_counts()[:args.num_eval].index.tolist()
        print(f"\n\n{results['parent_axis'].value_counts()}\n{eval_axes}\n\n")

        # evaluator = LLMOnlyRanker(args)
        evaluator = getattr(rankers, args.ranker)(args)
        metrics, results, scoring_logs = evaluator.score(eval_axes, results.to_dict("records"))
        if args.ranker == "LLMOnlyRanker":
            summary_results = results.groupby('parent_axis').agg({'final_score': 'mean', 'one_sided_score': 'mean', 'question': 'count'}).reset_index()
        else:
            summary_results = results.groupby('parent_axis').agg({'final_score': 'mean', 'score_a_score': 'mean', 'score_b_score': 'mean', 'question': 'count'}).reset_index()

        results.to_csv(f"pipeline_results/{save_str}/{tag}-results.csv", index=False)
        selected_cols = ['question', 'answer_a', 'answer_b', 'response', 'axis_description', 'parent_axis', 'score_a_score', 'score_b_score', 'final_score', 'score_a_reasoning','score_b_reasoning']
        result_plot_table = wandb.Table(dataframe=results[selected_cols])
        score_distribution_a = [(i, len(results[results['score_a_score'] == i])) for i in range(-2, 3)]
        score_distribution_b = [(i, len(results[results['score_b_score'] == i])) for i in range(-2, 3)]
        score_distribution_a = wandb.Table(data=score_distribution_a, columns=["score", "count"])
        score_distribution_b = wandb.Table(data=score_distribution_b, columns=["score", "count"])
        score_distribution = wandb.Table(data=[(i, len(results[results['final_score'] == i])) for i in range(-4,5)],  columns=["score", "count"])

        
        wandb.log({"summary_results": wandb.Table(dataframe=summary_results), 
                   "results": result_plot_table, 
                   "scoring_logs": wandb.Table(dataframe=scoring_logs),
                   "metrics": metrics,
                   "all_parent_axes": wandb.Table(dataframe=results['parent_axis'].value_counts().reset_index()),
                   "score_a_value_counts: ": wandb.plot.bar(score_distribution_a, "score", "count", title="Score A Value Counts"),
                   "score_b_value_counts: ": wandb.plot.bar(score_distribution_b, "score", "count", title="Score A Value Counts"),
                   "final_score_counts": wandb.plot.bar(score_distribution, "score", "count", title="Final Score Counts"),
                   })
        wandb.finish()

# make main function
if __name__ == "__main__":
    main()
