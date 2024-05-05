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


import argparse
def main():
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--wandb', action='store_true', help='log to wandb')
    # parser.add_argument('--project', type=str, default='llm_eval_presentable', help='wandb project name')
    # parser.add_argument('--num-samples', type=int, help='number of samples to use')
    # parser.add_argument('--data-path', type=str, help='path to data')
    # parser.add_argument('--model-a-column', type=str, help='column name for model A')
    # parser.add_argument('--model-b-column', type=str, help='column name for model B')
    # parser.add_argument('--k', type=int, help='number of clusters')
    # parser.add_argument('--batch-size', type=int, help='batch size for LLM')
    # parser.add_argument('--num-eval', default=3, type=int, help='model to use')
    # parser.add_argument('--oz', action='store_true', help='use oz prompt')
    # parser.add_argument('--dummy-eval', action='store_true', help='use dummy eval prompt')
    # parser.add_argument('--embedding-model', type=str, default='text-embedding-3-small', help='embedding model to use')
    # parser.add_argument('--seed', default=42, type=int, help='random seed')
    # parser.add_argument('--group-column', type=str)
    # parser.add_argument('--cluster-method', type=str, default='hierarchical', help='clustering method')
    # parser.add_argument('--ranker', type=str, default='LLMOnlyRanker', help='ranker to use')
    # parser.add_argument('--proposer', type=str, default='LLMBatchProposer', help='proposer to use')
    # parser.add_argument('--reducer', type=str, default='AxisReducer', help='reducer to use')
    # parser.add_argument('--proposer-batch-size', type=int, default=10, help='batch of questions to get differences for')
    # parser.add_argument("--save-dir", type=str, default="pipeline_results", help="directory to save results")
    # parser.add_argument("--eval-only", action="store_true", help="only run evaluation")
    # parser.add_argument("--heldout-percentage", type=float, default=0.5, help="percentage of data to holdout")
    # parser.add_argument("--axes", nargs="+", help="axes to evaluate")
    # parser.add_argument("--test", action="store_true", help="run test")
    # args = parser.parse_args()
    # add in args to override defaults
    parser = argparse.ArgumentParser(description='CLIP Advice')
    parser.add_argument('--config', default='configs/multi_llm.yaml', help="config file")
    parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                    "(use dots for.nested=overrides)")
    # flags = parser.parse_args()
    flags, unknown = parser.parse_known_args()

    overrides = OmegaConf.from_cli(flags.overrides)
    cfg       = OmegaConf.load(flags.config)
    args      = OmegaConf.merge(cfg, overrides)
    args.yaml = flags.config

    np.random.seed(args.seed)
    random.seed(args.seed)

    # tirn off wandb logging
    if not args.wandb:
        os.environ["WANDB_MODE"] = "dryrun"
    proj_name = args.project if not args.dummy_eval else f"llm_eval_refactor_debug"
    proj_name = f"{proj_name}_test" if args.test else proj_name
    global_df = pd.read_csv(args.data_path)
    print(f"Models: {args.models}")
    print(f"Eval Axes: {args.axes}")
    # remove duplicate question-answer
    global_df.drop_duplicates(subset=args.models, inplace=True)

    

    if args.group_column:
        groups = global_df[args.group_column].unique()
        print(f"Running VibeCheck on group {args.group_column}({groups})")
        print(f"Group value counts: {global_df[args.group_column].value_counts()}")
    else:
        groups = ["all"]

    if args.test:
        groups = groups[:3]
    for group in groups:
        if args.group_column:
            df = global_df[global_df[args.group_column] == group]
        else:
            df = global_df
        model_group = '-'.join(args.models).replace(' ', '')[:30]
        wandb.init(project=proj_name, entity="lisadunlap", config=dict(args), group=model_group, name=f"{args.group_column}-{group}")

        # create str of datapath for savins
        num_samples = min(args.num_samples, df.shape[0]) if args.num_samples else df.shape[0]
        save_str = args.data_path.split("/")[-1].split(".")[0] + f"_{group}"
        tag = f"{model_group}_k{args.k}_seed{args.seed}" if not args.num_samples else f"{model_group}_{args.k}_samples{num_samples}_seed{args.seed}"
        tag = f"{tag}_oz" if args.oz else tag
        tag = f"{tag}_dummy_eval" if args.dummy_eval else tag
        if not os.path.exists(f"{args.save_dir}/{save_str}"):
            os.makedirs(f"{args.save_dir}/{save_str}")

        # randomly sample 10 rows, set random seed for reproducibility
        if args.num_samples:
            df = df.sample(num_samples, random_state=args.seed)

        # nonsense_names = [
        # "Blorptex", "Qwinku", "Flumzog", "Veptinar", "Juxlimp",
        # "Sproonch", "Gribwol", "Ploxinor", "Vlurnip", "Snudlewock",
        # "Zemfrip", "Clorxib", "Dwibzut", "Jinktor", "Vemplic",
        # "Glozwurk", "Friptal", "Qwembix", "Hulmop", "Zurkwin"]
        # args.models = args.models[:len(nonsense_names)]
        # for i, model in enumerate(args.models):
        #     wandb.summary[model] = nonsense_names[i]
        df = df[['question', *args.models]]
        heldout_len = int(df.shape[0] * args.heldout_percentage)
        heldout_df = df.sample(heldout_len, random_state=args.seed)
        df = df.drop(heldout_df.index)

        print(f"Running a VibeCheck on {df.shape[0]} samples")
        if args.eval_only:
            print(f"Running eval only on {heldout_df.shape[0]} samples")
        elif args.axes:
            print(f"Running eval on {len(args.axes)} axes : {args.axes}")
        else:
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
            results.to_csv(f"{args.save_dir}/{save_str}/{tag}-results.csv", index=False)
            eval_axes = results['parent_axis'].value_counts()[:args.num_eval].index.tolist()

        ######################################
        ############  score axes  ############
        ######################################
        # load if eval only
        if args.eval_only:
            if not os.path.exists(f"{args.save_dir}/{save_str}/{tag}-results.csv"):
                raise ValueError(f"Results file not found at {args.save_dir}/{save_str}/{tag}-results.csv")
            print(f"Loading {args.save_dir}/{save_str}/{tag}-results.csv...")
            results = pd.read_csv(f"{args.save_dir}/{save_str}/{tag}-results.csv")

            print(results.columns)
            eval_axes = results['parent_axis'].value_counts()[:args.num_eval].index.tolist()
            print(f"\n\n{results['parent_axis'].value_counts()}\n{eval_axes}\n\n")
            results.to_csv(f"{args.save_dir}/{save_str}/{tag}-eval-results.csv", index=False)
            metrics.to_csv(f"{args.save_dir}/{save_str}/{tag}-eval-metrics.csv", index=False)
        elif args.axes:
            eval_axes = args.axes

        # evaluator = LLMOnlyRanker(args)
        evaluator = getattr(rankers, args.ranker)(args)
        metrics, results, scoring_logs = evaluator.score(eval_axes, heldout_df.to_dict("records"))
        results.to_csv(f"{args.save_dir}/{save_str}/{tag}-eval-results.csv", index=False)
        results = results.drop_duplicates(subset=['question', 'axis'])

        wandb.log({"summary_results": wandb.Table(dataframe=results), 
                   "scoring_logs": wandb.Table(dataframe=scoring_logs),
                   "metrics": metrics,
                   })

        ######################################
        ############  test scores  ###########
        ######################################   
        # TODO
        
        wandb.finish()

# make main function
if __name__ == "__main__":
    main()
