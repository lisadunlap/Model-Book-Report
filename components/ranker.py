import random
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind, ttest_rel, mode
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, trange
import re
import itertools
import plotly.graph_objects as go

import wandb
from serve.utils_clip import get_embeddings
from serve.utils_llm import get_llm_output
from serve.utils_vlm import get_vlm_output

from components.proposer import LLMProposer

smaller_systems_prompt = "You are a helpful assistant. Your outputs adhere to the format given by the user."


class Ranker:
    def __init__(self, args: Dict):
        random.seed(args["seed"])
        self.args = args
        if "group_names" in args:
            self.group_names = args['group_names']
        else:
            self.group_names = ["Group A", "Group B"]

    def score_hypothesis(self, hypothesis: str, dataset: List[dict]) -> List[float]:
        raise NotImplementedError

    def rerank_hypotheses(
        self, hypotheses: List[str], dataset1: List[dict], dataset2: List[dict]
    ) -> List[dict]:
        if len(dataset1) > self.args["max_num_samples"]:
            random.seed(self.args["seed"])
            dataset1 = random.sample(dataset1, self.args["max_num_samples"])
        if len(dataset2) > self.args["max_num_samples"]:
            random.seed(self.args["seed"])
            dataset2 = random.sample(dataset2, self.args["max_num_samples"])

        scored_hypotheses = []
        for hypothesis in tqdm(hypotheses):
            scores1 = self.score_hypothesis(hypothesis, dataset1)
            scores2 = self.score_hypothesis(hypothesis, dataset2)

            metrics = self.compute_metrics(scores1, scores2, hypothesis)
            scored_hypotheses.append(metrics)
        scored_hypotheses = sorted(
            scored_hypotheses, key=lambda x: x["auroc"], reverse=True
        )
        return scored_hypotheses
    
class NullRanker(Ranker):
    def __init__(self, args: Dict):
        super().__init__(args)

    def score_hypothesis(self, hypothesis: str, dataset: List[dict]) -> List[float]:
        return [0.0] * len(dataset)

class RubricRanker(Ranker):
    def __init__(self, args: Dict):
        super().__init__(args)
        random.seed(args.seed)
        self.diff_proposer = LLMProposer(args)

    @staticmethod
    def generate_rubric(axis, running_example=False):

        prompt = """I am performing qualitative analysis on LLM outputs. I have an axis in which I would like to generate a rubric that I could give a person such that they can rate a prompt-output pair on a scale of -2 to 2 on this axis. 

        Here is my axis name along with what it means to be high or low on this axis:
        {axis}

        Please be clear and specific for your definitions of what makes a prompt-output pair a score of -2, -1, 0, etc. To assist understanding, please provide a examples of what a -2, -1, 0, 1, 2 would look like on the same prompt. Please ensure this rubric is easy to understand by people and would result in the same scores across multiple human graders.
        
        Please provide your thought process when creating the rubric before providing the rubric. Please structure your response with "Rubric of {{axis name}}" as the title and the scores in the following format:
        
        Score {{-2, -1, 0, 1, 2}}: {{description}}
        Definition: {{definition}}
        Example: {{example}}"""

        prompt_running_example = """I am performing qualitative analysis on LLM outputs. I have an axis in which I would like to generate a rubric that I could give a person such that they can rate a prompt-output pair on a scale of -2 to 2 on this axis.

        Here is my axis name along with what it means to be high or low on this axis:
        {axis}

        Please be clear and specific for your definitions of what makes a prompt-output pair a score of -2, -1, 0, etc. To assist understanding, please provide a examples of what a -2, -1, 0, 1, 2 would look like on the prompt below. Please ensure this rubric is easy to understand by people and would result in the same scores across multiple human graders, and encourage users to vote a sample as 0 sparingly, as we want to see the differences between the models.

        Here is the prompt:
        {running_example}

        Please provide your thought process when creating the rubric before providing the rubric. Please structure your response with "Rubric of {{axis name}}" as the title and the scores in the following format:

        Score {{-2, -1, 0, 1, 2}}: {{description}}
        Definition: {{definition}}
        Example: {{example}}"""

        prompt = prompt.format(axis=axis) if not running_example else prompt_running_example.format(axis=axis, running_example=running_example)

        rubric_output = get_llm_output(prompt, model="gpt-4-0125-preview")

        convert_prompt = """Below is the output of an LLM asked to generate a rubric. I want to feed this rubric directly into an LLM to score items and remove any beginning or end paragraphs talking to the user about the creation of the rubric. Please extract the rubric from the following text:
        {output}
        
        Please do not make any edits to the rubric itself. Please output only the rubric."""
        converted = get_llm_output(convert_prompt.format(output=rubric_output), model="gpt-4")
        return converted, {"axis": axis, "rubric": rubric_output, "converted_rubric": converted}

    def get_score(self, row, axis, rubric, dummy_eval=False):
        if dummy_eval:
            return ["Reasoning: Because I said so\nScore: 0", "Reasoning: Because I said so\nScore: 0"]
            
        prompt = """I would like to score a given prompt-output pair on the following axis of variation: {axis}. Each prompt-output pair will be scored on a scale of -2 to 2 based on the following rubric:
        {rubric}
        
        Given the above rubric, please objectively score the following prompt and output:
        {prompt}
        
        Please provide your thought process when scoring the prompt and output before providing the score. Please respond in the following format:
        
        Reasoning: {{reasoning}}
        Score: {{-2, -1, 0, 1, 2}}"""

        prompt_a = prompt.format(axis=axis, rubric=rubric, prompt=f"Prompt: {row['question']}\Output: {row['answer_a']}")
        prompt_b = prompt.format(axis=axis, rubric=rubric, prompt=f"Prompt: {row['question']}\Output: {row['answer_b']}")
        output_a = get_llm_output(prompt_a, model="gpt-3.5-turbo", system_prompt="You are a fair and objective judge of model outputs. Your evaluations are clear, concise, and free from exaggerative language. You strictly adhere to the format and guidelines provided by the user, ensuring each decision is well-supported by the evidence within the outputs themselves.")
        output_b = get_llm_output(prompt_b, model="gpt-3.5-turbo", system_prompt="You are a fair and objective judge of model outputs. Your evaluations are clear, concise, and free from exaggerative language. You strictly adhere to the format and guidelines provided by the user, ensuring each decision is well-supported by the evidence within the outputs themselves.")
        return [output_a, output_b]
    
    @staticmethod
    def extract_scores(text):
        def helper(text):
            text = text.replace("*", "").replace("+", "")
            score_pattern = re.compile(r'Score: (-?\d)', re.IGNORECASE)
            score = score_pattern.findall(text)
            if "n/a" in score or "N/A" in score:
                return 0
            elif int(score[0]) not in [-2, -1, 0, 1, 2]:
                print(f"Error extracting scores from text: {text}")
                return 0
            return int(score[0])
        try:
            return helper(text)
        except:
            print(f"Error extracting scores from text: {text}")
            print("fixing....")
            prompt = """I have an LLM output which is the reasoning and score of a piece of text. Can you convert this string into an output that adheres to the following format:
            Reasoning: {{reasoning}}
            Score: {{-2, -1, 0, 1, 2}}

            Here is the string:
            {output}

            If the score is not present, please provide a score of 0. Please only respond with the reasoning and score so I can feed this output directly into my string parser.
            """
            text = get_llm_output(prompt.format(output=text), model="gpt-3.5-turbo")
            print(f"fixed?\n{text}\n")
            try:
                extracted = helper(text)
            except:
                extracted = 0
            return extracted

    def score_hypothesis(self, hypothesis: str, dataset: List[dict]) -> List[float]:
        """
        Generate rubric for each hypothesis
        """
        rubric, logs = self.generate_rubric(hypothesis)
        assert "question" in dataset[0] and "answer_a" in dataset[0] and "answer_b" in dataset[0], "Dataset must contain 'question', 'answer_a', and 'answer_b' keys."
        scores = []
        dataset_scores = []
        for row in dataset:
            score = self.get_score(row, hypothesis, rubric, dummy_eval=self.args.dummy_eval)
            if score is not None:
                row['score_a_reasoning'] = score[0]
                row['score_b_reasoning'] = score[1]
                row["score_a_score"] = self.extract_scores(score[0])
                row["score_b_score"] = self.extract_scores(score[1])
                row["final_score"] = row["score_a_score"] - row["score_b_score"]
                row["axis"] = hypothesis
                scores.append((row["score_a_score"], row["score_b_score"]))
                dataset_scores.append(row)
        return scores, dataset_scores, logs
    
    def score(self, axes: List[str], dataset: List[dict]):
        all_scores, all_dataset_scores, all_logs, axis_metrics = [], [], [], []
        group_based_metrics = []
        for axis in axes:
            scores, dataset_scores, logs = self.score_hypothesis(axis, dataset)
            all_scores.extend(scores)
            all_dataset_scores.extend(dataset_scores)
            all_logs.append(logs)
            metrics = self.compute_metrics(axis, scores)
            axis_metrics.append(metrics)

        return pd.DataFrame(axis_metrics), pd.DataFrame(all_dataset_scores), pd.DataFrame(all_logs)
    
    @staticmethod
    def find_max_mean_diff_subset(scores_a, scores_b, threshold=1):
        diffs = np.array([a - b for a, b in zip(scores_a, scores_b)])
        neutral_indices = np.where(diffs == 0)[0]
        positive_indices = np.where(diffs >= threshold)[0]
        negative_indices = np.where(diffs < -threshold)[0]

        return {"average positive score": np.mean(diffs[positive_indices]) if len(positive_indices) > 0 else 0, 
                "positive score support": len(positive_indices),
                "positive score idxs": positive_indices.tolist(),
                "average negative score": np.mean(diffs[negative_indices]) if len(negative_indices) > 0 else 0, 
                "negative score support": len(negative_indices),
                "negative score idxs": negative_indices.tolist(),
                "average neutral score": np.mean(diffs[neutral_indices]) if len(neutral_indices) > 0 else 0,
                "neutral score support": len(neutral_indices),
                "neutral score idxs": neutral_indices.tolist()
                }
    
    def compute_metrics(self, axis, scores):
        scores_a, scores_b = zip(*scores)
        scores_a, scores_b = np.array(scores_a), np.array(scores_b)

        # Compute the mean difference
        mean_diff = np.mean(scores_a - scores_b)

        # Perform the paired t-test
        t_statistic, p_value = ttest_rel(scores_a, scores_b)

        # find the max mean difference across all subsets of size n
        subset_results = self.find_max_mean_diff_subset(scores_a, scores_b, 10)

        return {
            "axis": axis,
            "mean_diff": mean_diff,
            "t_statistic": t_statistic,
            "p_value": p_value,
            "support": len(scores_a),
            **subset_results
        }

def fleiss_kappa(M):
    """
    Calculate Fleiss' kappa for a matrix of shape (n_items, n_categories).
    M is a matrix where each column represents a category and each row represents a different subject.
    Each cell in the matrix represents the number of raters who assigned the corresponding category to that subject.
    """
    N, k = M.shape  # N is number of items, k is number of categories
    n = np.sum(M[0, :])  # number of ratings per item, assumed to be the same across items

    # The proportion of all assignments which were to category j
    p_j = np.sum(M, axis=0) / (N * n)

    # The extent to which raters agree for the ith item
    P_i = (np.sum(M**2, axis=1) - n) / (n * (n - 1))

    # Mean of P_i over all items
    P_bar = np.mean(P_i)

    # Expected agreement by chance
    P_e = np.sum(p_j**2)

    kappa = (P_bar - P_e) / (1 - P_e)

    return kappa if (1 - P_e) != 0 else 0  # return 0 if division by zero occurs (degenerate case)

def aggregate_scores(scores):
    # given a  num_items x num_judges matrix of scores, aggregate the scores into a single score per item
    mode_score, count = mode(scores, axis=0)
    average_score = np.mean(scores, axis=0)
    # round the average score
    # average_score = np.round(average_score)
    majority_vote = mode_score[0] if mode_score.size > 0 else None  # Handle empty input

    return {
        "Majority Vote": majority_vote,
        "Average Score, Rounded": np.round(average_score),
        "Average Score": average_score
    }

def top_high_variance_indices(scores, top_n=5):
    """
    Identify indices with the highest variance in scores across judges.
    """
    judges = list(scores.keys())
    scores_matrix = np.array([scores[judge] for judge in judges]).T  # transpose to have judges as columns
    variances = np.var(scores_matrix, axis=1)
    top_variance_indices = np.argsort(-variances)[:top_n]  # argsort in descending order by using negative variances

    return np.array(top_variance_indices.tolist())

class RubricRankerJury(RubricRanker):
    def __init__(self, args: Dict):
        super().__init__(args)
        random.seed(args.seed)
        self.diff_proposer = LLMProposer(args)
        self.num_judges = 3

    def get_score(self, row, axis, rubric, dummy_eval=False):
        if dummy_eval:
            return ["Reasoning: Because I said so\nScore: 0", "Reasoning: Because I said so\nScore: 0"]
            
        prompt = """I would like to score a given prompt-output pair on the following axis of variation: {axis}. Each prompt-output pair will be scored on a scale of -2 to 2 based on the following rubric:
        {rubric}
        
        Given the above rubric, please objectively score the following prompt and output:
        {prompt}
        
        Please provide your thought process when scoring the prompt and output before providing the score. Please respond in the following format:
        
        Reasoning: {{reasoning}}
        Score: {{-2, -1, 0, 1, 2}}"""

        prompt_a = prompt.format(axis=axis, rubric=rubric, prompt=f"Prompt: {row['question']}\Output: {row['answer_a']}")
        prompt_b = prompt.format(axis=axis, rubric=rubric, prompt=f"Prompt: {row['question']}\Output: {row['answer_b']}")
        judge_systems_prompt = "You are a fair and objective judge of model outputs. Your evaluations are clear, concise, and free from exaggerative language. You strictly adhere to the format and guidelines provided by the user, ensuring each decision is well-supported by the evidence within the outputs themselves."
        judge_outputs = []
        for judge in ["gpt-3.5-turbo", "claude-3-haiku-20240307"]:
            output_a = get_llm_output(prompt_a, model=judge, system_prompt=judge_systems_prompt)
            output_b = get_llm_output(prompt_b, model=judge, system_prompt=judge_systems_prompt)
            judge_outputs.append([output_a, output_b])
        output_a = get_llm_output(prompt_a, model="gpt-3.5-turbo", system_prompt="You are an expert mechanical turk worker.")
        output_b = get_llm_output(prompt_b, model="gpt-3.5-turbo", system_prompt="You are an expert mechanical turk worker.")
        judge_outputs.append([output_a, output_b])
        return judge_outputs

    def score_hypothesis(self, hypothesis: str, dataset: List[dict]) -> List[float]:
        """
        Generate rubric for each hypothesis
        """
        rubric, logs = self.generate_rubric(hypothesis)
        assert "question" in dataset[0] and "answer_a" in dataset[0] and "answer_b" in dataset[0], "Dataset must contain 'question', 'answer_a', and 'answer_b' keys."
        judge_scores = {f"Judge_{i}_scores": [] for i in range(3)} 
        judge_scores["majority_scores"] = []
        dataset_scores = []
        for row in dataset:
            scores = self.get_score(row, hypothesis, rubric, dummy_eval=self.args.dummy_eval)
            if scores is not None:
                for i, score in enumerate(scores):
                    row[f'Judge_{i}_scores_reasoning'] = score
                    row[f"Judge_{i}_score"] = [self.extract_scores(s) for s in score]
                    row[f"Judge_{i}_final_score"] = [row[f"Judge_{i}_score"][0] - row[f"Judge_{i}_score"][1]]
                    row["axis"] = hypothesis
                    judge_scores[f"Judge_{i}_scores"].append(row[f"Judge_{i}_score"])
                    dataset_scores.append(row)
            row["majority_scores"] = aggregate_scores(np.array([row[f"Judge_{i}_score"] for i in range(self.num_judges)]))
            judge_scores["majority_scores"].append(row["majority_scores"])
        return judge_scores, dataset_scores, logs

    def score(self, axes: List[str], dataset: List[dict]):
        all_dataset_scores, all_logs, axis_metrics = [], [], []
        example_disagreement = []
        for axis in axes:
            scores, dataset_scores, logs = self.score_hypothesis(axis, dataset)
            all_dataset_scores.extend(dataset_scores)
            all_logs.append(logs)
            metrics = self.compute_metrics(axis, scores)
            top_variance_examples = top_high_variance_indices({k: np.average(v, axis=1) for k, v in scores.items()})
            example_disagreement.append({"axis": axis, "top_variance_examples": pd.DataFrame(dataset_scores).iloc[top_variance_examples]})
            wandb.log({f"{axis}_top_variance_examples": wandb.Table(dataframe=pd.DataFrame(dataset_scores).iloc[top_variance_examples])})
            axis_metrics.append(metrics)

        return pd.DataFrame(axis_metrics), pd.DataFrame(all_dataset_scores), pd.DataFrame(all_logs)
    
    def compute_metrics(self, axis, scores):
        metrics = {"axis": axis}
        # Prepare data for Fleiss' Kappa
        category_labels = [-2, -1, 0, 1, 2]
        score_counts = np.zeros((len(scores[next(iter(scores))]), len(category_labels)))  # Rows are items, columns are categories
        for judge in range(3):
            scores_a, scores_b = zip(*scores[f"Judge_{judge}_scores"])
            scores_a, scores_b = np.array(scores_a), np.array(scores_b)

            # Update matrix for Fleiss' Kappa
            for i, (score_a, score_b) in enumerate(zip(scores_a, scores_b)):
                score_counts[i, category_labels.index(score_a)] += 1
                score_counts[i, category_labels.index(score_b)] += 1

            mean_diff = np.mean(scores_a - scores_b)
            t_statistic, p_value = ttest_rel(scores_a, scores_b)
            metrics[f"Judge_{judge}_mean_diff"] = mean_diff
            metrics[f"Judge_{judge}_t_statistic"], metrics[f"Judge_{judge}_p_value"] = t_statistic, p_value
            metrics[f"Judge_{judge}_support"] = len(scores_a)
            self.plot_score_distribution(axis, judge, scores_a, scores_b)

        kappa = fleiss_kappa(score_counts)
        metrics["Inter-annotator Agreement"] = kappa

        # compute stats for majority_score
        scores_a, scores_b = zip(*scores[f"majority_scores"])
        scores_a, scores_b = np.array(scores_a), np.array(scores_b)
        mean_diff = np.mean(scores_a - scores_b)
        t_statistic, p_value = ttest_rel(scores_a, scores_b)
        metrics[f"Judge_majority_mean_diff"] = mean_diff
        metrics[f"Judge_majority_t_statistic"], metrics[f"Judge_majority_p_value"] = t_statistic, p_value
        self.plot_score_distribution(axis, "majority", scores_a, scores_b)

        return metrics
    
    @staticmethod
    def plot_score_distribution(axis, judge, scores_a, scores_b):
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Histogram(x=scores_a - scores_b, nbinsx=5))
        fig_diff.update_layout(title=f"Judge {judge} Difference Scores")
        
        # Plotting individual scores
        fig_scores = go.Figure()
        fig_scores.add_trace(go.Histogram(x=scores_a, nbinsx=5, name='Scores A', marker_color='red', opacity=0.5))
        fig_scores.add_trace(go.Histogram(x=scores_b, nbinsx=5, name='Scores B', marker_color='blue', opacity=0.5))
        fig_scores.update_layout(title=f"Judge {judge} Scores")
        
        # Logging to wandb
        wandb.log({
            f"judge_{judge}_{axis}_scores": wandb.Plotly(fig_scores),
            f"judge_{judge}_{axis}_diff_scores": wandb.Plotly(fig_diff)
        })

class MuliRubricRankerJury(RubricRankerJury):
    def __init__(self, args: Dict):
        super().__init__(args)
        random.seed(args.seed)
        self.diff_proposer = LLMProposer(args)
        self.num_judges = len(self.args.judges)

    def get_score(self, row, axis, rubric, dummy_eval=False):
        if dummy_eval:
            return ["Reasoning: Because I said so\nScore: 0", "Reasoning: Because I said so\nScore: 0"]
            
        prompt = """I would like you to evaluate a language model's response based on a specific criterion: {axis}. Use the following rubric to assign a score from -2 to 2:

        {rubric}

        With the rubric in mind, review the provided response to the prompt below:

        {prompt}

        Based on your analysis, please explain your reasoning before assigning a score. Use the following format for your response:

        Analysis: {{reasoning}}
        Score: {{choose from: -2, -1, 0, 1, 2}}"""

        judge_systems_prompt = "You are a fair and objective judge of model outputs. Your evaluations are clear, concise, and free from exaggerative language. You strictly adhere to the format and guidelines provided by the user, ensuring each decision is well-supported by the evidence within the outputs themselves."
        judge_outputs = []
        for judge in self.args.judges:
            # print(f"Getting judgement for Judge = {judge}")
            model_outputs = []
            for model in self.args.models:
                scoring_prompt = prompt.format(axis=axis, rubric=rubric, prompt=f"Prompt: {row['question']}\Output: {row[model]}")
                output_a = get_llm_output(scoring_prompt, model=judge, system_prompt=judge_systems_prompt)
                model_outputs.append(output_a)
            judge_outputs.append(model_outputs)
        return judge_outputs
    
    def score_hypothesis(self, hypothesis: str, dataset: List[dict], axis_to_topic: dict, topic_example: str = None) -> List[float]:
        """
        Generate rubric for each hypothesis
        """
        print(f"Scoring hypothesis {hypothesis}")
        rubric, logs = self.generate_rubric(hypothesis, topic_example)
        judge_scores = {f"Judge_{i}_scores": [] for i in range(self.num_judges)} 
        judge_scores["avg_scores"] = []
        judge_scores["avg_diff_scores"] = []
        dataset_scores = []
        for row in tqdm(dataset):
        # for row in dataset:
            scores = self.get_score(row, hypothesis, rubric, dummy_eval=self.args.dummy_eval)
            if scores is not None:
                for i, score in enumerate(scores):
                    row[f'Judge_{i}_scores_reasoning'] = score
                    row[f"Judge_{i}_score"] = [self.extract_scores(s) for s in score]
                    row[f"Judge_{i}_diff_score"] = [row[f"Judge_{i}_score"][j] - np.mean(row[f"Judge_{i}_score"]) for j in range(len(row[f"Judge_{i}_score"]))]
                    row["axis"] = hypothesis
                    judge_scores[f"Judge_{i}_scores"].append(row[f"Judge_{i}_score"])
            else:
                print("No scores found")
            row["avg_scores"] = aggregate_scores(np.array([row[f"Judge_{i}_score"] for i in range(self.num_judges)]))["Average Score"]
            row["avg_diff_scores"] = aggregate_scores(np.array([row[f"Judge_{i}_diff_score"] for i in range(self.num_judges)]))["Average Score"]
            judge_scores["avg_scores"].append(row["avg_scores"])
            judge_scores["avg_diff_scores"].append(row["avg_diff_scores"])
            dataset_scores.append(row)
        return judge_scores, dataset_scores, logs
    
    def score(self, axes: List[str], dataset: List[dict], axis_to_topic: dict, topic_to_example: pd.DataFrame = pd.DataFrame()):
        all_dataset_scores, all_logs, axis_metrics = [], [], []
        print(f"axes: {axes}")
        for axis in axes:
            topics = axis_to_topic[axis_to_topic["axis"] == axis].iloc[0]["topic"]
            print(topics)
            for topic in topics:
                print(f"Scoring for axis {axis} for topic {topic}")
                topic_sample = topic_to_example[topic_to_example["topic"] == topic].iloc[0]['example'] if len(topic_to_example) > 0 else None
                axis_dataset = [d for d in dataset if d["topic"] == topic]
                if self.args.early_stopping:
                    # try on 10 rows, if the score differences are < 0.1, continue
                    scores, dataset_scores, logs = self.score_hypothesis(axis, axis_dataset[:10], axis_to_topic, topic_sample)
                    metrics = self.compute_metrics(axis, scores, topic)
                    print("LISA LOOK HERE")
                    print([np.abs(metrics[f"Judge_avg_{model}_mean_diff"]) for model in self.args.models])
                    mean_dff = np.max([np.abs(metrics[f"Judge_avg_{model}_mean_diff"]) for model in self.args.models])
                    print(mean_dff)
                    if mean_dff < self.args.early_stopping_threshold:
                        print(f"Skipping topic {topic} for axis {axis}")
                        continue

                scores, dataset_scores, logs = self.score_hypothesis(axis, axis_dataset, axis_to_topic, topic_sample)
                all_dataset_scores.append(pd.DataFrame(dataset_scores))
                all_logs.append(logs)
                metrics = self.compute_metrics(axis, scores, topic)
                axis_metrics.append(metrics)
        if len(axis_metrics) == 0:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        return pd.DataFrame(axis_metrics), pd.concat(all_dataset_scores), pd.DataFrame(all_logs)
    
    def compute_metrics(self, axis, scores, topic="all"):
        from sklearn.metrics import cohen_kappa_score
        from itertools import combinations
        metrics = {"axis": axis}
        # Prepare data for Fleiss' Kappa
        category_labels = [-2, -1, 0, 1, 2]
        
        for m, model in enumerate(self.args.models):
            score_counts = np.zeros((len(scores[next(iter(scores))]), len(category_labels)))  # Rows are items, columns are categories
            judge_pairs = list(combinations(range(self.num_judges), 2))  # List of all pairs of judges
            for judge_pair in judge_pairs:
                judge_1, judge_2 = judge_pair
                # Get scores for the two judges
                scores_1 = np.array(scores[f"Judge_{judge_1}_scores"])[:, m]
                scores_2 = np.array(scores[f"Judge_{judge_2}_scores"])[:, m]
                # Calculate Cohen's kappa between the two judges
                kappa = cohen_kappa_score(scores_1, scores_2, weights='linear')
                # Add result to metrics with a unique key for this pair
                metrics[f"{model} Cohen's Kappa (Judge {judge_1} vs Judge {judge_2})"] = kappa
            for judge in range(self.num_judges):
                scores_list = np.array(scores[f"Judge_{judge}_scores"])
                # get the m col of scores list
                scores_model = scores_list[:, m]
                # get average across the models
                model_avg_score = np.average(scores_list, axis=1)
                score_diff = scores_model - model_avg_score
                
                metrics[f"Judge_{judge}_{model}_mean_score"] = np.average(scores_model)
                metrics[f"Judge_{judge}_{model}_std_score"] = np.std(scores_model)
                metrics[f"Judge_{judge}_{model}_mean_diff"] = np.mean(score_diff)
            
            # compute stats for majority_score
            scores_list = np.array(scores["avg_scores"])
            scores_model = scores_list[:, m]
            model_avg_score = np.average(scores_list, axis=1)
            score_diff = scores_model - model_avg_score
            metrics[f"Judge_avg_{model}_mean_score"] = np.average(scores_model)
            metrics[f"Judge_avg_{model}_std_score"] = np.std(scores_model)
            metrics[f"Judge_avg_{model}_mean_diff"] = np.mean(score_diff)

        # do a paired t_test for the per-sample scores averaged across judges for each set of models
        model_pairs = list(combinations(self.args.models, 2))
        for model_pair in model_pairs:
            model_1, model_2 = model_pair
            model_idxs = [self.args.models.index(model_1), self.args.models.index(model_2)]
            scores_1 = np.average(np.array([np.array(scores[f"Judge_{judge}_scores"])[:, model_idxs[0]] for judge in range(self.num_judges)]), axis=0)
            scores_2 = np.average(np.array([np.array(scores[f"Judge_{judge}_scores"])[:, model_idxs[1]] for judge in range(self.num_judges)]), axis=0)
            t_statistic, p_value = ttest_rel(scores_1, scores_2)
            metrics[f"t_statistic_{model_1}_{model_2}"] = t_statistic
            metrics[f"p_value_{model_1}_{model_2}"] = p_value
        metrics["support"] = len(scores_list)
        metrics["topic"] = topic

        return metrics
    
# class MuliRubricRankerJuryTopicSpecific(MuliRubricRankerJury):
    
#     def score_hypothesis(self, hypothesis: str, dataset: List[dict], axis_to_topic: dict) -> List[float]:
#         """
#         Generate rubric for each hypothesis
#         """
#         print(f"Scoring hypothesis {hypothesis}")
#         rubric, logs = self.generate_rubric(hypothesis)
#         judge_scores = {f"Judge_{i}_scores": [] for i in range(self.num_judges)} 
#         judge_scores["avg_scores"] = []
#         judge_scores["avg_diff_scores"] = []
#         dataset_scores = []
#         for row in tqdm(dataset):
#         # for row in dataset:
#             scores = self.get_score(row, hypothesis, rubric, dummy_eval=self.args.dummy_eval)
#             if scores is not None:
#                 for i, score in enumerate(scores):
#                     row[f'Judge_{i}_scores_reasoning'] = score
#                     row[f"Judge_{i}_score"] = [self.extract_scores(s) for s in score]
#                     row[f"Judge_{i}_diff_score"] = [row[f"Judge_{i}_score"][j] - np.mean(row[f"Judge_{i}_score"]) for j in range(len(row[f"Judge_{i}_score"]))]
#                     row["axis"] = hypothesis
#                     judge_scores[f"Judge_{i}_scores"].append(row[f"Judge_{i}_score"])
#             else:
#                 print("No scores found")
#             row["avg_scores"] = aggregate_scores(np.array([row[f"Judge_{i}_score"] for i in range(self.num_judges)]))["Average Score"]
#             row["avg_diff_scores"] = aggregate_scores(np.array([row[f"Judge_{i}_diff_score"] for i in range(self.num_judges)]))["Average Score"]
#             judge_scores["avg_scores"].append(row["avg_scores"])
#             judge_scores["avg_diff_scores"].append(row["avg_diff_scores"])
#             dataset_scores.append(row)
#         return judge_scores, dataset_scores, logs
    
class JuryRanker(MuliRubricRankerJury):

    def __init__(self, args: Dict):
        super().__init__(args)
        random.seed(args.seed)
        self.diff_proposer = LLMProposer(args)
        self.num_judges = len(self.args.judges)

    def get_score(self, row, axis, dummy_eval=False):
        if dummy_eval:
            return ["Analysis: Because I said so\nScore: 0"]  * len(self.args.models)
            
        prompt = """I would like you to evaluate a language model's response with responect to the follwing axis: {axis}. 

        Please give the follwing output a score from -2 to 2 based on the descriptions of what it means to be low and high on this axis. Here is the prompt and output:

        {prompt}
        
        Please explain your reasoning before assigning a score. Use the following format for your response:

        Analysis: {{reasoning}}
        Score: {{choose from: -2, -1, 0, 1, 2}}"""

        judge_systems_prompt = "You are a fair and objective judge of model outputs. Your evaluations are clear, concise, and free from exaggerative language. You strictly adhere to the format and guidelines provided by the user, ensuring each decision is well-supported by the evidence within the outputs themselves."
        judge_outputs = []
        for judge in self.args.judges:
            print(f"Getting judgement for Judge = {judge}")
            model_outputs = []
            for model in self.args.models:
                scoring_prompt = prompt.format(axis=axis, prompt=f"Prompt: {row['question']}\Output: {row[model]}")
                output_a = get_llm_output(scoring_prompt, model=judge, system_prompt=judge_systems_prompt)
                model_outputs.append(output_a)
            judge_outputs.append(model_outputs)
        return judge_outputs
    
    def score_hypothesis(self, hypothesis: str, dataset: List[dict], axis_to_topic: dict) -> List[float]:
        """
        Generate rubric for each hypothesis
        """
        print(f"Scoring hypothesis {hypothesis}")
        judge_scores = {f"Judge_{i}_scores": [] for i in range(self.num_judges)} 
        judge_scores["avg_scores"] = []
        judge_scores["avg_diff_scores"] = []
        dataset_scores = []
        for row in tqdm(dataset):
        # for row in dataset:
            scores = self.get_score(row, hypothesis, dummy_eval=self.args.dummy_eval)
            if scores is not None:
                for i, score in enumerate(scores):
                    row[f'Judge_{i}_scores_reasoning'] = score
                    row[f"Judge_{i}_score"] = [self.extract_scores(s) for s in score]
                    row[f"Judge_{i}_diff_score"] = [row[f"Judge_{i}_score"][j] - np.mean(row[f"Judge_{i}_score"]) for j in range(len(row[f"Judge_{i}_score"]))]
                    row["axis"] = hypothesis
                    judge_scores[f"Judge_{i}_scores"].append(row[f"Judge_{i}_score"])
            else:
                print("No scores found")
            row["avg_scores"] = aggregate_scores(np.array([row[f"Judge_{i}_score"] for i in range(self.num_judges)]))["Average Score"]
            row["avg_diff_scores"] = aggregate_scores(np.array([row[f"Judge_{i}_diff_score"] for i in range(self.num_judges)]))["Average Score"]
            judge_scores["avg_scores"].append(row["avg_scores"])
            judge_scores["avg_diff_scores"].append(row["avg_diff_scores"])
            dataset_scores.append(row)
        return judge_scores, dataset_scores, {}
    