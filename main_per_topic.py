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
from omegaconf import OmegaConf

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
import components.sampler as samplers

def rank(args, save_str, tag, heldout_df, eval_axes, axis_to_topic):
    # evaluator = LLMOnlyRanker(args)
    evaluator = getattr(rankers, args.ranker)(args)
    metrics, results, scoring_logs = evaluator.score(eval_axes, heldout_df.to_dict("records"), axis_to_topic)
    results.to_json(f"{args.save_dir}/{save_str}/{tag}-eval-results.json", orient='records')
    metrics.to_json(f"{args.save_dir}/{save_str}/{tag}-eval-metrics.json", orient='records')
    scoring_logs.to_json(f"{args.save_dir}/{save_str}/{tag}-scoring-logs.json", orient='records')
    results = results.drop_duplicates(subset=['question', 'axis'])

    wandb.log({
                "scoring_logs": wandb.Table(dataframe=scoring_logs),
                "summary_results": wandb.Table(dataframe=results), 
                "metrics": metrics,
                })

def get_save_str(args,num_samples):
    # get first 3 letters of ech model if length is too long (>50)
    model_group = '-'.join(args.models).replace(' ', '')
    model_group = '-'.join([x[:3] for x in args.models]).replace(' ', '') if len(model_group) > 50 else model_group
    # create str of datapath for savins
    save_str = args.data_path.split("/")[-1].split(".")[0] + "_per_topic"
    save_str = f"{save_str}/{args.output_name}" if args.output_name else save_str
    save_str = f"{save_str}/{args.proposer}-{args.sampler}_{args.num_topic_clusters}-{args.ranker}"
    tag = f"{model_group}_k{args.k}_seed{args.seed}" if not args.num_samples else f"{model_group}_{args.k}_samples{num_samples}_seed{args.seed}"
    tag = f"{tag}_oz" if args.oz else tag
    tag = f"{tag}_dummy_eval" if args.dummy_eval else tag
    tag = f"{tag}_axes_provided" if args.axes else tag
    if not os.path.exists(f"{args.save_dir}/{save_str}"):
        os.makedirs(f"{args.save_dir}/{save_str}", exist_ok=True)
    return save_str, tag

import argparse
def main():
    # add in args to override defaults
    parser = argparse.ArgumentParser(description='CLIP Advice')
    parser.add_argument('--config', default='configs/multi_llm.yaml', help="config file")
    parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                    "(use dots for.nested=overrides)")
    # flags = parser.parse_args()
    flags, unknown = parser.parse_known_args()

    overrides = OmegaConf.from_cli(flags.overrides)
    base_cfg  = OmegaConf.load("configs/base.yaml")
    cfg       = OmegaConf.load(flags.config)
    args      = OmegaConf.merge(base_cfg, cfg, overrides)
    args.yaml = flags.config

    np.random.seed(args.seed)
    random.seed(args.seed)

    # tirn off wandb logging
    if not args.wandb:
        os.environ["WANDB_MODE"] = "dryrun"
    proj_name = args.project if not args.dummy_eval else f"llm_eval_refactor_debug"
    proj_name = f"{proj_name}_test" if args.test else proj_name
    df = pd.read_csv(args.data_path)
    print(f"Models: {args.models}")
    print(f"Eval Axes: {args.axes}")
    # remove duplicate question-answer
    df.drop_duplicates(subset=args.models, inplace=True)

    

    if args.group_column:
        groups = df[args.group_column].unique()
        print(f"Running VibeCheck on group {args.group_column}({groups})")
        print(f"Group value counts: {df[args.group_column].value_counts()}")
    else:
        groups = ["all"]
        
    num_samples = min(args.num_samples, df.shape[0]) if args.num_samples else df.shape[0]
    save_str, tag = get_save_str(args, num_samples)

    # randomly sample 10 rows, set random seed for reproducibility
    if args.num_samples:
        df = df.sample(num_samples, random_state=args.seed)

    old_df = df
    df = df[['question', *args.models]]
    # add in group_column if it exists
    if args.group_column:
        df[args.group_column] = old_df[args.group_column]
    # heldout_len = int(df.shape[0] * args.heldout_percentage)
    # heldout_df = df.sample(heldout_len, random_state=args.seed)
    heldout_df = df
    
    # sample/cluster
    sampler = getattr(samplers, args.sampler)(args)
    topics, centroids = sampler.sample(df)
    np.save(f"{args.save_dir}/{save_str}/{tag}-topic-centroids.np", centroids)
    df["topic"] = topics
    topic_counts = df['topic'].value_counts()

    global_df = df
    time = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    print(f"Topic Counts: {topic_counts}")
    for topic in df.topic.unique():
        # add a dattetime to the tag
        wandb.init(project=proj_name, entity="lisadunlap", config=dict(args), group=model_group+time, name=topic)
        df = global_df[global_df['topic'] == topic]
        wandb.summary[f"Topic count"] = len(df)

        print(f"Running a VibeCheck on {len(df)} samples")
        if args.eval_only:
            print(f"Running eval only on {heldout_df.shape[0]} samples")
        elif args.axes:
            print(f"Running eval on {len(args.axes)} axes : {args.axes}")
        else:
            # ######################################
            # #### get per question differences ####
            # ######################################
            # sample at most 20 samples per topic
            proposal_df = df.sample(min(20, len(df)), random_state=args.seed)
            proposer = getattr(proposers, args.proposer)(args)
            all_axis_descriptions, llm_logs, pairwise_differences, results = proposer.propose(proposal_df)
            wandb.log({"per_sample_differences": wandb.Table(dataframe=results), "pairwise_diff_llm_logs": wandb.Table(dataframe=llm_logs)})

            ######################################
            ####  cluster per question axes   ####
            ######################################
            all_axis_descriptions = list(results['axis_description'])
            all_axis_descriptions = [x.replace("*", "") for x in all_axis_descriptions]

            reducer = getattr(reducers, args.reducer)(args)
            parent_axes, child_parent_map, tables = reducer.reduce(all_axis_descriptions)
            results['axis'] = child_parent_map
            axis_to_topic = results.groupby('axis')['topic'].apply(set).reset_index()
            results.to_csv(f"{args.save_dir}/{save_str}/{tag}-reducer_results.csv", index=False)
            axis_to_topic.to_json(f"{args.save_dir}/{save_str}/{tag}-axes_to_topic.json", orient='records')
            eval_axes = results['axis'].value_counts()[:args.num_eval].index.tolist()
            wandb.log({k: wandb.Table(dataframe=v) for k, v in tables.items()})
            wandb.log({"eval_axes": results['axis'].value_counts().reset_index(), "results": wandb.Table(dataframe=results), "axis_to_topic": wandb.Table(dataframe=axis_to_topic)})

        ######################################
        ############  score axes  ############
        ######################################
        print("STARTING EVAL")
        # load if eval only
        if args.eval_only:
            if not os.path.exists(f"{args.save_dir}/{save_str}/{tag}-results.csv"):
                raise ValueError(f"Results file not found at {args.save_dir}/{save_str}/{tag}-results.csv")
            print(f"Loading {args.save_dir}/{save_str}/{tag}-results.csv...")
            results = pd.read_csv(f"{args.save_dir}/{save_str}/{tag}-results.csv")
            axis_to_topic = pd.read_json(f"{args.save_dir}/{save_str}/{tag}-axes_to_topic.json", orient='records')
            eval_axes = results['axis'].value_counts()[:args.num_eval].index.tolist()
            print(f"\n\n{results['axis'].value_counts()}\n{eval_axes}\n\n")
        elif args.axes:
            eval_axes = args.axes
            axis_to_topic = {x: df['topic'].unique().tolist() for x in eval_axes}
            
        if not args.proposer_only:
            # evaluator = LLMOnlyRanker(args)
            evaluator = getattr(rankers, args.ranker)(args)
            metrics, results, scoring_logs = evaluator.score(eval_axes, heldout_df.to_dict("records"), axis_to_topic)
            results.to_json(f"{args.save_dir}/{save_str}/{tag}-eval-results.json", orient='records')
            metrics.to_json(f"{args.save_dir}/{save_str}/{tag}-eval-metrics.json", orient='records')
            scoring_logs.to_json(f"{args.save_dir}/{save_str}/{tag}-scoring-logs.json", orient='records')
            results = results.drop_duplicates(subset=['question', 'axis'])

            wandb.log({
                        "scoring_logs": wandb.Table(dataframe=scoring_logs),
                        "summary_results": wandb.Table(dataframe=results), 
                        "metrics": metrics,
                        })

# make main function
if __name__ == "__main__":
    main()
