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
from components.sampler import match_set_to_centroids, classify_centroids

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def get_save_str(args,num_samples, model_group):
    # create str of datapath for savins
    save_str = args.data_path.split("/")[-1].split(".")[0]
    save_str = f"{save_str}/{args.output_name}" if args.output_name else save_str
    save_str = f"{save_str}/{args.proposer}-{args.sampler}_{args.num_topic_clusters}-{args.ranker}"
    tag = f"{model_group}_k{args.k}_seed{args.seed}" if not args.num_samples else f"{model_group}_{args.k}_samples{num_samples}_seed{args.seed}"
    tag = f"{tag}_oz" if args.oz else tag
    tag = f"{tag}_dummy_eval" if args.dummy_eval else tag
    tag = f"{tag}_axes_provided" if args.axes else tag
    if not os.path.exists(f"{args.save_dir}/{save_str}"):
        os.makedirs(f"{args.save_dir}/{save_str}", exist_ok=True)
    return save_str, tag

def load_experiment(results_dir, tag, args):
    results = pd.read_csv(f"{results_dir}/{tag}-reducer_results.csv")
    axis_to_topic = pd.read_json(f"{results_dir}/{tag}-axes_to_topic.json", orient='records')
    eval_axes = results['axis'].value_counts()[:min(args.num_eval, len(results['axis'].unique()))].index.tolist()
    print(f"\n\n{results['axis'].value_counts()}\n{eval_axes}\n\n")
    if os.path.exists(f"{results_dir}/{tag}-topic_to_example.json"):
        topic_to_example = pd.read_json(f"{results_dir}/{tag}-topic_to_example.json", orient='records')
    else:
        topic_to_example = None
    if os.path.exists(f"{results_dir}/{tag}-topic-centroids.np.npy"):
        topic_centroids = np.load(f"{results_dir}/{tag}-topic-centroids.np.npy", allow_pickle=True).item()
    else:
        topic_centroids = None
    if os.path.exists(f"{results_dir}/{tag}-scoring-logs.json"):
        rubric = pd.read_json(f"{results_dir}/{tag}-scoring-logs.json")
        print(f"Rubric: {rubric} \t {len(rubric)}")
        rubric = pd.DataFrame() if len(rubric.columns) == 0 else rubric
    else:
        rubric = pd.DataFrame()
    return eval_axes, axis_to_topic, topic_to_example, topic_centroids, rubric

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
    print(f"df columns: {df.columns}")
    # remove duplicate question-answer
    df.drop_duplicates(subset=args.models, inplace=True)

    if args.group_column:
        groups = df[args.group_column].unique()
        print(f"Running VibeCheck on group {args.group_column}({groups})")
        print(f"Group value counts: {df[args.group_column].value_counts()}")
    else:
        groups = ["all"]

    # # model_group = '-'.join(args.models).replace(' ', '')[:30]
    # # get first 3 letters of ech model if length is too long (>50)
    model_group = '-'.join(args.models).replace(' ', '')
    model_group = '-'.join([x[:3] for x in args.models]).replace(' ', '') if len(model_group) > 50 else model_group
    wandb.init(project=proj_name, entity="lisadunlap", config=dict(args), group=model_group, name=f"{args.group_column}")

    num_samples = min(args.num_samples, df.shape[0]) if args.num_samples else df.shape[0]
    num_samples = 10 if args.test else num_samples
    save_str, tag = get_save_str(args, num_samples, model_group)

    # randomly sample 10 rows, set random seed for reproducibility
    if args.num_samples or args.test:
        if args.new_sample:
            df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)[:num_samples]
        else:
            df = df.sample(num_samples, random_state=args.seed)

    old_df = df
    df = df[['question', *args.models]]
    # add in group_column if it exists
    if args.group_column:
        df[args.group_column] = old_df[args.group_column]
    # heldout_len = int(df.shape[0] * args.heldout_percentage)
    # heldout_df = df.sample(heldout_len, random_state=args.seed)
    heldout_df = df
    
    # sample
    sampler = getattr(samplers, args.sampler)(args)
    df, centroids = sampler.sample(df)
    np.save(f"{args.save_dir}/{save_str}/{tag}-topic-centroids.np", centroids)
    topics = df["topic"]
    rubric = pd.DataFrame()

    print(f"Running a VibeCheck on {len(df)} samples")
    if args.eval_only or args.rubric_path:
        print(f"Running eval only on {heldout_df.shape[0]} samples")
    elif args.axes:
        print(f"Running eval on {len(args.axes)} axes : {args.axes}")
    else:
        # sample at most 20 samples per topic
        proposal_df = df.groupby('topic').sample(min(args.num_proposal_samples, df['topic'].value_counts().min()), random_state=args.seed)
        topic_counts = df['topic'].value_counts()
        for topic, count in topic_counts.items():
            wandb.summary[f"{topic} count"] = count

        topic_to_example = {"topic": [], "example": [], "prompt": []} if args.topic_based_rubric_example else None
        if args.topic_based_rubric_example:
            for topic in df.topic.unique():
                rubric_example = samplers.get_example_prompt(proposal_df[proposal_df["topic"] == topic]["question"].tolist())
                print(f"Rubric Example for topic {topic}\n{rubric_example}\n-------------------")
                topic_to_example["topic"].append(topic)
                topic_to_example["example"].append(rubric_example["response"])
                topic_to_example["prompt"].append(rubric_example["example_generation_prompt"])
            topic_to_example = pd.DataFrame(topic_to_example)
            topic_to_example.to_json(f"{args.save_dir}/{save_str}/{tag}-topic_to_example.json", orient='records')
            wandb.log({"topic_to_example": wandb.Table(dataframe=topic_to_example)})

        # ######################################
        # #### get per question differences ####
        # ######################################
        proposer = getattr(proposers, args.proposer)(args)
        all_axis_descriptions, llm_logs, pairwise_differences, results = proposer.propose(proposal_df)
        wandb.log({"per_sample_differences": wandb.Table(dataframe=results), "pairwise_diff_llm_logs": wandb.Table(dataframe=llm_logs)})

        ######################################
        #### cluster per question axes    ####
        ######################################
        all_axis_descriptions = list(results['axis_description'])
        all_axis_descriptions = [x.replace("*", "") for x in all_axis_descriptions]

        # reducer = AxisReducer(args)
        reducer = getattr(reducers, args.reducer)(args)
        parent_axes, child_parent_map, tables = reducer.reduce(all_axis_descriptions)
        results['axis'] = child_parent_map
        axis_to_topic = results.groupby('axis')['topic'].apply(set).reset_index()
        df.to_json(f"{args.save_dir}/{save_str}/{tag}-reducer-df.json", orient='records')
        results.to_csv(f"{args.save_dir}/{save_str}/{tag}-reducer_results.csv", index=False)
        axis_to_topic.to_json(f"{args.save_dir}/{save_str}/{tag}-axes_to_topic.json", orient='records')
        eval_axes = results['axis'].value_counts()[:min(args.num_eval, len(results['axis'].unique()))].index.tolist()
        wandb.log({k: wandb.Table(dataframe=v) for k, v in tables.items()})
        wandb.log({"eval_axes": results['axis'].value_counts().reset_index(), "results": wandb.Table(dataframe=results), "axis_to_topic": wandb.Table(dataframe=axis_to_topic)})
        if args.proposer_only:
            for topic in df.topic.unique():
                print(f"Topic {topic} axes: {axis_to_topic[axis_to_topic['topic'].apply(lambda x: topic in x)]['axis'].tolist()}")
            wandb.finish()
            exit(0)

    ######################################
    ############  score axes  ############
    ######################################
    print("STARTING EVAL")
    # load if eval only
    if args.eval_only:
        eval_axes, axis_to_topic, topic_to_example, _, rubric = load_experiment(f"{args.save_dir}/{save_str}", tag, args)
    elif args.rubric_path:
        rubric = pd.read_json(args.rubric_path)
        eval_axes = rubric['axis'].unique().tolist()
        axis_to_topic = rubric
        topic_to_example = pd.DataFrame()
        print(axis_to_topic)
    elif args.axes:
        eval_axes = args.axes
        axis_to_topic = {"axis": [], "topic": []}
        for axis in eval_axes:
            axis_to_topic["axis"].append(axis)
            axis_to_topic["topic"].append(df['topic'].unique().tolist())
        axis_to_topic = pd.DataFrame(axis_to_topic)
        topic_to_example = pd.DataFrame()
        print(f"Axis to topic: {axis_to_topic}")
        

    # evaluator = LLMOnlyRanker(args)
    evaluator = getattr(rankers, args.ranker)(args)
    metrics, results, scoring_logs = evaluator.score(eval_axes, heldout_df.to_dict("records"), axis_to_topic, topic_to_example, rubric)
    results.to_json(f"{args.save_dir}/{save_str}/{tag}-eval-results.json", orient='records')
    metrics.to_json(f"{args.save_dir}/{save_str}/{tag}-eval-metrics.json", orient='records')
    scoring_logs.to_json(f"{args.save_dir}/{save_str}/{tag}-scoring-logs.json", orient='records')
    # results = results.drop_duplicates(subset=['question', 'axis'])

    wandb.log({
                "scoring_logs": wandb.Table(dataframe=scoring_logs),
                "summary_results": wandb.Table(dataframe=results), 
                "metrics": metrics,
                })

    ######################################
    ############  test scores  ###########
    ######################################
    if args.test_data_path:
        print("STARTING TEST EVAL")   
        test_df = pd.read_csv(args.test_data_path)
        test_df = test_df[['question', *args.models]]
        print(f"Models: {args.models}")
        print(f"Eval Axes: {args.axes}")
        # remove duplicate question-answer
        test_df.drop_duplicates(subset=args.models, inplace=True)

        # randomly sample 10 rows, set random seed for reproducibility
        if args.num_samples or args.test:
            if args.new_sample:
                test_df = test_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)[:num_samples]
            else:
                test_df = test_df.sample(num_samples, random_state=args.seed)

        test_df["topic"] = match_set_to_centroids(test_df, centroids, np.array(df['topic_label'].tolist()), np.stack(df["embedding"].tolist()))
        # test_df["topic"] = classify_centroids(test_df, centroids['summary'])
        print(f"DF topic breadkdown: {df['topic'].value_counts()}")
        print(f"Test DF topic breakdown: {test_df['topic'].value_counts()}")

        # print out rows which differ between test_df and heldout_df
        print(f"Test df shape: {test_df.shape}")
        print(f"Heldout df shape: {heldout_df.shape}")
        print(f"Test df topics: {test_df['topic'].value_counts()}")
        print(f"Heldout df topics: {heldout_df['topic'].value_counts()}")
        
        # def expand_dataframe_with_axes(df):
        #     unique_axes = df['axis'].unique()
        #     existing_pairs = set(zip(df['question'], df['axis']))
        #     new_rows = []
        #     for question in df['question'].unique():
        #         for axis in unique_axes:
        #             if (question, axis) not in existing_pairs:
        #                 new_row = {
        #                     'question': question,
        #                     'axis': axis,
        #                     'avg_scores': [0, 0],
        #                     'avg_diff_scores': [0, 0]
        #                 }
        #                 new_rows.append(new_row)
        #     new_rows_df = pd.DataFrame(new_rows)
        #     return pd.concat([df, new_rows_df], ignore_index=True)
        
        # def prepare_data_for_decision_tree(df, models):
        #     df = df.copy()
        #     results_short = df[['question', 'topic', 'axis', 'avg_scores', 'avg_diff_scores']]
        #     df = expand_dataframe_with_axes(results_short)

        #     # Create separate rows for each model in the models list
        #     expanded_rows = []
        #     for i, model in enumerate(models):
        #         model_rows = df.copy()
        #         model_rows['label'] = model
        #         model_rows['avg_diff_scores'] = model_rows['avg_diff_scores'].apply(lambda x: x[i])  # Select the element for the current model
        #         expanded_rows.append(model_rows)

        #     # Concatenate all model rows
        #     expanded_df = pd.concat(expanded_rows, ignore_index=True)

        #     # Pivot the data to create feature columns for each axis
        #     pivot_df = expanded_df.pivot_table(index=['question', 'label'], columns='axis', values='avg_diff_scores', fill_value=0).reset_index()

        #     # Prepare features and target
        #     X = pivot_df.drop(columns=['question', 'label'])
        #     y = pivot_df['label']
            
        #     return X, y

        from components.metrics_utils import prepare_data_for_decision_tree, train_decision_tree, expand_dataframe_with_axes, train_individual_feature_impact
        
        print("RUNNING ON HELDOUT SET")
        args.early_stopping = False
        evaluator = getattr(rankers, args.ranker)(args)
        test_metrics, test_results, test_scoring_logs = evaluator.score(eval_axes, test_df.to_dict("records"), axis_to_topic, topic_to_example, rubric)
        test_results.to_json(f"{args.save_dir}/{save_str}/{tag}-test-results.json", orient='records')
        test_metrics.to_json(f"{args.save_dir}/{save_str}/{tag}-test-metrics.json", orient='records')
        test_scoring_logs.to_json(f"{args.save_dir}/{save_str}/{tag}-test-scoring-logs.json", orient='records')

        wandb.log({
                    "test_summary_results": wandb.Table(dataframe=test_results), 
                    "test_metrics": test_metrics,
                    })
            
        # X_train, y_train = prepare_data_for_decision_tree(results, args.models)
        # X_test, y_test = prepare_data_for_decision_tree(test_results, args.models)

        # # if there are any test features which are all 0, remove them from the train and test data
        # zero_features = X_train.columns[X_train.sum() == 0]
        # print(f"Zero features: {zero_features}")
        # X_train = X_train.drop(columns=zero_features)
        # X_test = X_test.drop(columns=zero_features)

        # # Train the Decision Tree Classifier
        # clf = DecisionTreeClassifier(random_state=42)
        # clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # y_train_pred = clf.predict(X_train)
        # # save image of decision tree
        # from sklearn.tree import plot_tree
        # # Create the figure with a specified size
        # fig = plt.figure(figsize=(40,25))


        # # Plot the tree with adjusted parameters for node and text size
        # plot_tree(clf, 
        #         filled=True, 
        #         feature_names=[c.replace("High", "\nHigh").replace("Low", "\nLow") for c in X_train.columns], 
        #         class_names=list(args.models), 
        #         fontsize=12,  # Set the font size
        #         proportion=True,  # Set nodes to be proportional to the number of samples
        #         rounded=True  # Round the nodes
        #         )
        # plt.savefig(f"{args.save_dir}/{save_str}/{tag}-decision-tree.png", bbox_inches='tight')
        # wandb.log({"decision_tree": wandb.Image(f"{args.save_dir}/{save_str}/{tag}-decision-tree.png")})

        # # Print and save classification report
        # print(classification_report(y_test, y_pred))
        # # save train and test classification reports as tables
        # wandb.log({"classification_report": wandb.Table(dataframe=pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))), "classification_report_train": wandb.Table(dataframe=pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True)))})
        # pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).to_json(f"{args.save_dir}/{save_str}/{tag}-classification-report.json")
        # pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True)).to_json(f"{args.save_dir}/{save_str}/{tag}-classification-report-train.json")
        
        # # train logistic regression
        # from sklearn.linear_model import LogisticRegression
        # clf = LogisticRegression(random_state=42)
        # clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # y_train_pred = clf.predict(X_train)
        # # Print and save classification report
        # print("Logistic Regression")
        # print(classification_report(y_test, y_pred))
        # # save train and test classification reports as tables
        # wandb.log({"classification_report_logistic": wandb.Table(dataframe=pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))), "classification_report_logistic_train": wandb.Table(dataframe=pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True)))})
        # pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).to_json(f"{args.save_dir}/{save_str}/{tag}-classification-report-logistic.json")
        # pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True)).to_json(f"{args.save_dir}/{save_str}/{tag}-classification-report-logistic-train.json")
        # # save test predictions and true labels
        # pd.DataFrame({"true": y_test, "pred": y_pred}).to_json(f"{args.save_dir}/{save_str}/{tag}-predictions.json")

    wandb.finish()

# make main function
if __name__ == "__main__":
    main()
