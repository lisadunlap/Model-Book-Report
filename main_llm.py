import logging
from typing import Dict, List, Tuple

import click
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
import ast

import wandb
from components.evaluator import GPTEvaluator, NullEvaluator
from components.proposer import (
    LLMProposer,
    LLMPairwiseProposerWithQuestion,
    DualSidedLLMProposer,
    DualSidedLLMProposerNoQuestion
)
from components.ranker import NullRanker, LLMOnlyRanker, DualClusterRanker, DualClusterRanker2


def load_config(config: str) -> Dict:
    base_cfg = OmegaConf.load("configs/base.yaml")
    cfg = OmegaConf.load(config)
    final_cfg = OmegaConf.merge(base_cfg, cfg)
    args = OmegaConf.to_container(final_cfg)
    args["config"] = config
    if args["wandb"]:
        wandb.init(
            project=args["project"],
            entity="clipinvariance",
            name=args["data"]["name"],
            group=f'{args["data"]["group1"]} - {args["data"]["group2"]} ({args["data"]["purity"]})',
            config=args,
        )
    return args


def load_data(args: Dict) -> Tuple[List[Dict], List[Dict], List[str]]:
    data_args = args["data"]

    df = pd.read_csv(f"{data_args['root']}/{data_args['name']}.csv")

    if data_args["subset"]:
        old_len = len(df)
        df = df[df["subset"] == data_args["subset"]]
        print(
            f"Taking {data_args['subset']} subset (dataset size reduced from {old_len} to {len(df)})"
        )

    dataset1 = df[df["group_name"] == data_args["group1"]].to_dict("records")
    dataset2 = df[df["group_name"] == data_args["group2"]].to_dict("records")
    group_names = [data_args["group1"], data_args["group2"]]

    if data_args["purity"] < 1:
        logging.warning(f"Purity is set to {data_args['purity']}. Swapping groups.")
        assert len(dataset1) == len(dataset2), "Groups must be of equal size"
        n_swap = int((1 - data_args["purity"]) * len(dataset1))
        dataset1 = dataset1[n_swap:] + dataset2[:n_swap]
        dataset2 = dataset2[n_swap:] + dataset1[:n_swap]
    return dataset1, dataset2, group_names


def propose(args: Dict, dataset1: List[Dict], dataset2: List[Dict]) -> List[str]:
    proposer_args = args["proposer"]
    proposer_args["seed"] = args["seed"]
    proposer_args["captioner"] = args["captioner"]

    proposer = eval(proposer_args["method"])(proposer_args)
    hypotheses, logs, images = proposer.propose(dataset1, dataset2)
    if args["wandb"]:
        wandb.log({"logs": wandb.Table(dataframe=pd.DataFrame(logs))})
        wandb.log({"llm_outputs": wandb.Table(dataframe=pd.DataFrame(images))})
    all_outputs = pd.DataFrame(images)
    all_outputs['question'] = all_outputs['question'].apply(lambda x: x[0])
    all_outputs['group_1_hypotheses'] = all_outputs['group_1_hypotheses'].apply(lambda x: x[0])
    all_outputs['group_2_hypotheses'] = all_outputs['group_2_hypotheses'].apply(lambda x: x[0])
    all_outputs['group_1_answers'] = all_outputs['group_1_answers'].apply(lambda x: x[0])
    all_outputs['group_2_answers'] = all_outputs['group_2_answers'].apply(lambda x: x[0])
    all_outputs.to_csv('all_outputs.csv', index=False)
    wandb.log({'all_outputs': wandb.Table(dataframe=all_outputs)})
    # all_outputs['group_1_hypotheses'] = all_outputs['group_1_hypotheses'].apply(lambda x: ast.literal_eval(x[0]))
    # all_outputs['group_2_hypotheses'] = all_outputs['group_2_hypotheses'].apply(lambda x: ast.literal_eval(x[0]))
    return hypotheses, all_outputs


def rank(
    args: Dict,
    hypotheses: List[str],
    dataset1: List[Dict],
    dataset2: List[Dict],
    group_names: List[str],
) -> List[str]:
    ranker_args = args["ranker"]
    ranker_args["seed"] = args["seed"]
    ranker_args['group_names'] = group_names

    ranker = eval(ranker_args["method"])(ranker_args)

    scored_hypotheses = ranker.rerank_hypotheses(hypotheses, dataset1, dataset2)
    if args["wandb"]:
        print(scored_hypotheses.keys())
        for key in scored_hypotheses.keys():
            table_hypotheses = wandb.Table(dataframe=pd.DataFrame(scored_hypotheses[key]))
            wandb.log({f"scored {key}": table_hypotheses})
        # make the keys of the dictionary the columns of the dataframe
        dfs = []
        for key in scored_hypotheses.keys():
            df = pd.DataFrame(scored_hypotheses[key])
            df['supercluster'] = key
            dfs.append(df)
        df = pd.concat(dfs)
        df.to_csv('scored_hypotheses.csv', index=False)
        wandb.log({"score_hypotheses": wandb.Table(dataframe=df)})

    return []


@click.command()
@click.option("--config", help="config file")
def main(config):
    logging.info("Loading config...")
    args = load_config(config)
    # print(args)

    logging.info("Loading data...")
    dataset1, dataset2, group_names = load_data(args)
    # print(dataset1, dataset2, group_names)

    logging.info("Proposing hypotheses...")
    hypotheses, logs = propose(args, dataset1, dataset2)
    print(hypotheses)
    print("######################################")
    print("######################################")
    print("######################################")

    logging.info("Ranking hypotheses...")
    ranked_hypotheses = rank(args, hypotheses, dataset1, dataset2, group_names)
    # print(ranked_hypotheses)


if __name__ == "__main__":
    main()
