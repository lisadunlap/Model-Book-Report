import random
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind, ttest_rel
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, trange
import re

import wandb
from serve.utils_clip import get_embeddings
from serve.utils_llm import get_llm_output
from serve.utils_vlm import get_vlm_output

smaller_systems_prompt = "You are a helpful assistant. Your outputs adhere to the format given by the user."


class Ranker:
    def __init__(self, args: Dict):
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

# LLMDifferenceRanker(Ranker):


class LLMOnlyRanker(Ranker):
    def __init__(self, args: Dict):
        super().__init__(args)
        random.seed(args.seed)

    @staticmethod
    def extract_scores(text):
        def extract_helper(text):
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
                elif "medium" in results["Model A Score"].lower() and "medium" in results["Model B Score"].lower():
                    return 0
                elif "medium" in results["Model A Score"].lower() and "low" in results["Model B Score"].lower():
                    return 1
                elif "medium" in results["Model A Score"].lower() and "high" in results["Model B Score"].lower():
                    return -1
                else:
                    raise ValueError(f"No score found\n{text}")
            except:
                # print(f"No score found\n{text}")     
                raise ValueError(f"No score found\n{text}")
        try:
            return extract_helper(text)
        except:
            print(f"Error extracting scores from text: {text}")
            print("fixing....")
            prompt = """I am trying to parse this string but am getting an error. Here is my expected format:

            Reason: {{reasoning}}
            Model A Score: {{high/medium/low}} # this should only be the word high, medium, or low
            Model B Score: {{high/medium/low}} # this should only be the word high, medium, or low

            And here is my string:
            {text}

            Please reformat the string in the above format for me to parse. Please only respond with the scores and reasoning so I can feed this output directly into my string parser."""
            text = get_llm_output(prompt.format(text=text), model="gpt-3.5-turbo", system_prompt=smaller_systems_prompt)
            print(f"fixed?\n{text}\n")
            extracted = extract_helper(text)
            print(extracted)
            return extracted

    def ensemble_scores(self, scores):
        score_1, score_2 = self.extract_scores(scores[0]), self.extract_scores(scores[1])
        # Utility function to ensemble scores
        # If they disagree, return 0
        return (score_1 + -1 * score_2)/2, score_1 == score_2 and score_1 != 0

    @staticmethod
    def get_score(row, eval_axes, dummy_eval=False):
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

            # Ensemble scores and return
            return [scoring_output_a, scoring_output_b]

    def score_hypothesis(self, hypothesis: str, dataset: List[dict]) -> List[float]:
        """Given an axis and list of question, answer pairs, score the models on the axis."""
        print(dataset[0])
        assert "question" in dataset[0] and "answer_a" in dataset[0] and "answer_b" in dataset[0], "Dataset must contain 'question', 'answer_a', and 'answer_b' keys."
        scores = []
        dataset_scores = []
        for row in dataset:
            score = self.get_score(row, [hypothesis], dummy_eval=self.args.dummy_eval)
            if score is not None:
                row['scores'] = score
                row["one_sided_score_and_reasoning"] = score[0]
                row["one_sided_score"] = self.extract_scores(score[0])
                row["final_score_and_reasoning"] = score
                ensamble_score, disagree = self.ensemble_scores(score)
                row["scores_disagree"] = disagree
                row["final_score"] = ensamble_score
                scores.append(ensamble_score)
                dataset_scores.append(row)
            
        wandb.summary["percentage_scores_disagree"] = sum([r["scores_disagree"] for r in dataset_scores]) / len(dataset_scores)
        wandb.summary["percentage_neutral"] = sum([r["final_score"] == 0 for r in dataset_scores if not r["scores_disagree"]]) / len(dataset_scores)
        return scores, dataset_scores
    
    def score(self, axes: List[str], dataset: List[dict]):
        all_scores, all_dataset_scores = [], []
        for axis in axes:
            scores, dataset_scores = self.score_hypothesis(axis, dataset)
            all_scores.extend(scores)
            all_dataset_scores.extend(dataset_scores)
        return all_scores, all_dataset_scores

class RubricRanker(Ranker):
    def __init__(self, args: Dict):
        super().__init__(args)
        random.seed(args.seed)

    @staticmethod
    def generate_rubric(axis):

        prompt = """I am performing qualitative analysis on LLM outputs. I have an axis in which I would like to generate a rubric that I could give a person such that they can rate a prompt-output pair on a scale of -2 to 2 on this axis. 

        Here is my axis name along with what it means to be high or low on this axis:
        {axis}

        Please be clear and specific for your definitions of what makes a prompt-output pair a score of -2, -1, 0, etc. To assist understanding, please provide a examples of what a -2, -1, 0, 1, 2 would look like on the same prompt. Please ensure this rubric is easy to understand by people and would result in the same scores across multiple human graders.
        
        Please provide your thought process when creating the rubric before providing the rubric. Please structure your response in the following format:
        
        Score {{-2, -1, 0, 1, 2}}: {{description}}
        Definition: {{definition}}
        Example: {{example}}"""

        prompt = prompt.format(axis=axis)

        rubric_output = get_llm_output(prompt, model="gpt-4-0125-preview")
        print(rubric_output)

        convert_prompt = """Below is the output of an LLM asked to generate a rubric. I want to feed this rubric directly into an LLM to score items and remove any beginning or end paragraphs talking to the user about the creation of the rubric. Please extract the rubric from the following text:
        {output}
        
        Please do not make any edits to the rubric itself. Please output only the rubric."""
        converted = get_llm_output(convert_prompt.format(output=rubric_output), model="gpt-4")
        print(f"\n\nconverted\n{converted}")
        return converted, {"axis": axis, "rubric": rubric_output, "converted_rubric": converted}

    @staticmethod
    def get_score(row, axis, rubric, dummy_eval=False):
        if dummy_eval:
            return ["Reasoning: Because I said so\nScore: 0", "Reasoning: Because I said so\nScore: 0"]
        else:
            if row['parent_axis'] != axis:
                return None
            
        prompt = """I would like to score a given prompt-output pair on the following axis of variation: {axis}. Each prompt-output pair will be scored on a scale of -2 to 2 based on the following rubric:
        {rubric}
        
        Here is the prompt-output pair:
        {prompt}
        
        Please provide your thought process when scoring the prompt-output pair before providing the score. Please respond in the following format:
        
        Reasoning: {{reasoning}}
        Score: {{-2, -1, 0, 1, 2}}"""

        prompt_a = prompt.format(axis=axis, rubric=rubric, prompt=f"Prompt: {row['question']}\nResponse: {row['answer_a']}")
        prompt_b = prompt.format(axis=axis, rubric=rubric, prompt=f"Prompt: {row['question']}\nResponse: {row['answer_b']}")
        print(prompt_a)
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
            extracted = helper(text)
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
                scores.append((row["score_a_score"], row["score_b_score"]))
                dataset_scores.append(row)
        return scores, dataset_scores, logs
    
    def score(self, axes: List[str], dataset: List[dict]):
        all_scores, all_dataset_scores, all_logs, axis_metrics = [], [], [], []
        for axis in axes:
            scores, dataset_scores, logs = self.score_hypothesis(axis, dataset)
            all_scores.extend(scores)
            all_dataset_scores.extend(dataset_scores)
            all_logs.append(logs)
            axis_metrics.append(self.compute_metrics(axis, all_scores))

        return pd.DataFrame(axis_metrics), pd.DataFrame(all_dataset_scores), pd.DataFrame(all_logs)
    
    def compute_metrics(self, axis, scores):
        scores_a, scores_b = zip(*scores)
        scores_a, scores_b = np.array(scores_a), np.array(scores_b)

        # Compute the mean difference
        mean_diff = np.mean(scores_a - scores_b)

        # Perform the paired t-test
        t_statistic, p_value = ttest_rel(scores_a, scores_b)

        return {
            "axis": axis,
            "mean_diff": mean_diff,
            "t_statistic": t_statistic,
            "p_value": p_value,
            "support": len(scores_a)
        }