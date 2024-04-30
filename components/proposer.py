import hashlib
import json
import os
import random
import torch
import ast
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image, PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

import components.prompts as prompts
import wandb
from serve.utils_general import save_data_diff_image
from serve.utils_llm import get_llm_output
from serve.utils_vlm import get_embed_caption_blip, get_vlm_output
from tqdm import tqdm
import omegaconf

from components.proposer_prompts import *
from components.parsing_utils import *


class Proposer:
    def __init__(self, args: Dict):
        self.args = args
        # load default config from yaml configs/base.yaml
        default_args = omegaconf.OmegaConf.load("configs/base.yaml")
        self.args = omegaconf.OmegaConf.merge(default_args, self.args)

    def propose(
        self, dataset1: List[Dict], dataset2: List[Dict]
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        Given two datasets, return a list of hypotheses
        """
        all_hypotheses = []
        all_logs = []
        all_images = []
        random.seed(self.args["seed"])
        for i in range(self.args["num_rounds"]):
            sampled_dataset1 = self.sample(dataset1, self.args["num_samples"])
            sampled_dataset2 = self.sample(dataset2, self.args["num_samples"])
            hypotheses, logs = self.get_hypotheses(sampled_dataset1, sampled_dataset2)
            images = self.visualize(sampled_dataset1, sampled_dataset2)
            all_hypotheses += hypotheses
            all_logs.append(logs)
            all_images.append(images)
        return all_hypotheses, all_logs, all_images

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Tuple[List[str], Dict]:
        raise NotImplementedError

    def sample(self, dataset: List[Dict], n: int) -> List[Dict]:
        if self.args['sampling_method'] == 'random':
            return random.sample(dataset, n)
        # elif self.args["proposer"]["sampling_method"] == 'classifier':
        #     model = torch.load("classifier.pt")
        #     return classifier_sampler(dataset, model, n)

class LLMProposer(Proposer):

    question_diff_prompt = """I have a list of user questions group into either A or B and I would like to understand the differences between these groups. Please list any noticeable differences in these groups. lease output a list differences between the two outputs with relation to specific axes of variation. Are there any general patterns, clusters, or variations you notice in the outputs? Try to give patterns that are specific enough that someone could reliably produce new questions that would belong to group A or group B, and that they could understand what it means to be higher or lower on that specific axis. 
    
    Here are the questions:
    {text}
    
    Please output a numbered list of differences between the two groups of questions. If there are no clear differences, please output "No differences found"."""

    conversion = """The following is an LLM output containing the axes of variation that you can consider when comparing two lists of questions (A and B):

        {axes}

        I want to formalize exactly what it means to be high and low on each axis. For each axis, I want you to provide a description of what it means to be high and low on that axis, as well as a score of where the two models fall on that axis. The score for a given model could be ("low", "high").This will help me understand the differences between the two models in a more structured way. Your output should be in this format:

        - {{axis_1}}:
            High: {{description of high}}
            Low: {{description of low}}
            Model A Score: {{score for Model A}}
            Model B Score: {{score for Model B}}

        - {{axis_2}}:
            High: {{description of high}}
            Low: {{description of low}}
            Model A Score: {{score for Model A}}
            Model B Score: {{score for Model B}}

        Please ensure that the description what is high and low on the axis are distinct and mutually exclusive such that given any unseen pair of text outputs, a human could easily and reliably determine which model is higher or lower on that axis. Please the axis name and descriptions of what is high and low are less than 5 words each, and ensure the scores are accurate representations of the outputs of the two models.
    """

    def __init__(self, args: Dict):
        super().__init__(args)
        self.batch_size = self.args.batch_size

    def propose_one_side(self, texts1: List[str], texts2: List[str]) -> Tuple[List[str], Dict]:
        # batch the texts and call llm to get differences
        prompt = self.question_diff_prompt.format(text="Group A:" + "\n".join(texts1) + "\n\nGroup B:" + "\n".join(texts2))
        output = get_llm_output(prompt, 'claude-3-opus-20240229')
        converted = get_llm_output(self.conversion.format(axes=output), 'claude-3-opus-20240229')
        logs = {"prompt": prompt, "output": output, "conversion_prompt": self.conversion.format(axes=output), "converted": converted}
        return output, converted, logs
    
    def propose(self, texts1: List[str], texts2: List[str]) -> Tuple[List[str] | List[Dict]]:
        max_size = 30
        sample_texts_1, sample_texts_2 = random.sample(texts1, min(len(texts1), max_size)), random.sample(texts2, min(len(texts2), max_size))
        left_output, left_converted, left_logs = self.propose_one_side(sample_texts_1, sample_texts_2)
        right_output, right_converted, right_logs = self.propose_one_side(sample_texts_2, sample_texts_1)
        return left_output, left_converted, left_logs, right_output, right_converted, right_logs

# class LLMProposer(Proposer):
#     def __init__(self, args: Dict):
#         super().__init__(args)

#     def get_hypotheses(
#         self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
#     ) -> Tuple[List[str], Dict]:
#         self.captioning(sampled_dataset1)
#         self.captioning(sampled_dataset2)
#         captions1 = [
#             f"Group A: {item['answer']}".replace("\n", " ").strip()
#             for item in sampled_dataset1
#         ]
#         captions2 = [
#             f"Group B: {item['answer']}".replace("\n", " ").strip()
#             for item in sampled_dataset2
#         ]
#         caption_concat = "\n".join(captions1 + captions2)
#         prompt = self.prompt.format(text=caption_concat)
#         output = get_llm_output(prompt, self.args["model"])
#         hypotheses = [line.replace("* ", "") for line in output.splitlines()]
#         logs = {"prompt": prompt, "output": output}
#         return hypotheses, logs
    
class LLMPairwiseProposerWithQuestion(Proposer):
    def __init__(self, args: Dict):
        super().__init__(args)
        self.systems_prompt = "Given a dataset of text outputs from two different large language models (LLMs), your task is to analyze and summarize the data based on specific characteristics. The goal is to identify and cluster similar behaviors or traits within the outputs, summarizing these into a concise list of commonly observed behaviors for each model. This analysis will help in understanding the general behaviors of these models for auditing, error discovery, and comparison purposes. Your outputs adhere to the format given by the user."
        self.smaller_systems_prompt = "You are a helpful assistant. Your outputs adhere to the format given by the user."
        self.model_columns = [args.model_a_column, args.model_b_column]
        self.model_a, self.model_b = args.model_a_column, args.model_a_column

    def propose(
        self, df
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        Given two datasets, return a list of hypotheses
        """
        assert "question" in df.columns, "'question' column not in dataset"
        random.seed(self.args.seed)
        oz_axes = ["Tone", "Format", "Level of Detail", "Ability to answer", "Safety", "Approach", "Creativity", "Fluency and crammatical correctness", "Adherence to prompt"]

        # get per question differences
        results = {"question":[], "answer_a":[], "answer_b":[], "response": [], "axis_response": []}
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            texts = f"{row['question']}\nModel A:\n{row[self.model_columns[0]]}\n\nModel B:\n{row[self.model_columns[1]]}\n"
            if self.args.oz:
                prompt = OZ_PROMPT.format(text=texts, axes="\n".join([f"* {axis}" for axis in oz_axes]))
            else:
                prompt = DEFAULT_PROMPT.format(text=texts)
            response = get_llm_output(prompt, model="gpt-3.5-turbo", system_prompt=self.systems_prompt, trace_name="per question differences").replace("**", "")
            # results["prompt"].append(texts)
            results["question"].append(row['question'])
            results["answer_a"].append(row[self.model_columns[0]].strip('[]'))
            results["answer_b"].append(row[self.model_columns[1]].strip('[]'))
            results["response"].append(response)
            # results["axes"].append(extract_entities(response))
            axis_prompt = AXIS_CONVERSION.format(axes=response)
            axis_response = get_llm_output(axis_prompt, model="gpt-3.5-turbo", system_prompt=self.smaller_systems_prompt, trace_name="convert per question axes")
            results["axis_response"].append(axis_response)
            
        results = pd.DataFrame(results)
        pairwise_differences = results[['question', 'answer_a', 'answer_b', 'response', 'axis_response']]
        llm_logs = results[['response', 'axis_response']]

        results["no_difference_detected"] = results["response"].apply(lambda x: is_match(x, "No differences found"))
        results = results[~results["no_difference_detected"]]

        # cluster per axis differences
        results['axis_description'] = results['axis_response'].apply(extract_axis_descriptions)
        results = results.explode('axis_description')

        all_axis_descriptions = list(set(results['axis_description']))
        return all_axis_descriptions, llm_logs, pairwise_differences, results
    
class LLMBatchProposer(LLMPairwiseProposerWithQuestion):
    def __init__(self, args: Dict):
        super().__init__(args)
        self.systems_prompt = "Given a dataset of text outputs from two different large language models (LLMs), your task is to analyze and summarize the data based on specific characteristics. The goal is to identify and cluster similar behaviors or traits within the outputs, summarizing these into a concise list of commonly observed behaviors for each model. This analysis will help in understanding the general behaviors of these models for auditing, error discovery, and comparison purposes. Your outputs adhere to the format given by the user."
        self.smaller_systems_prompt = "You are a helpful assistant. Your outputs adhere to the format given by the user."
        self.model_columns = [args.model_a_column, args.model_b_column]
        self.model_a, self.model_b = args.model_a_column, args.model_a_column
        self.batch_size = args.proposer_batch_size

    def propose_batch(self, df):
        """
        Get differences over a list of prompts
        """
        assert "question" in df.columns, "'question' column not in dataset"
        random.seed(self.args.seed)
        oz_axes = ["Tone", "Format", "Level of Detail", "Ability to answer", "Safety", "Approach", "Creativity", "Fluency and crammatical correctness", "Adherence to prompt"]

        # get per question differences
        texts = []
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            texts.append(f"{row['question']}\nModel A:\n{row[self.model_columns[0]]}\n\nModel B:\n{row[self.model_columns[1]]}\n")
        texts = "\n".join(texts)
        if self.args.oz:
            prompt = OZ_PROMPT.format(text=texts, axes="\n".join([f"* {axis}" for axis in oz_axes]))
        else:
            prompt = DEFAULT_PROMPT.format(text=texts)
        response = get_llm_output(prompt, model="gpt-3.5-turbo", system_prompt=self.systems_prompt, trace_name="per question differences").replace("**", "")
        axis_prompt = AXIS_CONVERSION.format(axes=response)
        axis_response = get_llm_output(axis_prompt, model="gpt-3.5-turbo", system_prompt=self.smaller_systems_prompt, trace_name="convert per question axes")
        return response, axis_response, {"proposal_prompt": prompt, "response": response, "conversion_prompt": axis_prompt, "axis_response": axis_response}

    def propose(
        self, df
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        Given two datasets, return a list of hypotheses
        """
        assert "question" in df.columns, "'question' column not in dataset"
        random.seed(self.args.seed)
        oz_axes = ["Tone", "Format", "Level of Detail", "Ability to answer", "Safety", "Approach", "Creativity", "Fluency and crammatical correctness", "Adherence to prompt"]

        # get per question differences
        results = {"question":[], "answer_a":[], "answer_b":[], "response": [], "axis_response": []}
        llm_logs = []
        for batch_start in range(0, len(df), self.batch_size):
            batch = df.iloc[batch_start:batch_start + self.batch_size]
            response, axis_response, logs = self.propose_batch(batch)
            results["question"].extend(batch['question'].tolist())
            results["answer_a"].extend(batch[self.model_columns[0]].tolist())
            results["answer_b"].extend(batch[self.model_columns[1]].tolist())
            results["response"].extend([response] * len(batch))
            results["axis_response"].extend([axis_response] * len(batch))
            llm_logs.append(logs)

        # print([(key, len(results[key])) for key in results.keys()])
        results = pd.DataFrame(results)
        pairwise_differences = results[['question', 'answer_a', 'answer_b', 'response', 'axis_response']]
        llm_logs = pd.DataFrame(llm_logs)

        results["no_difference_detected"] = results["response"].apply(lambda x: is_match(x, "No differences found"))
        results = results[~results["no_difference_detected"]]

        # cluster per axis differences
        results['axis_description'] = results['axis_response'].apply(extract_axis_descriptions)
        results = results.explode('axis_description')

        all_axis_descriptions = list(set(results['axis_description']))
        return all_axis_descriptions, llm_logs, pairwise_differences, results
        
def test_proposers():
    dataset = pd.read_csv("data/diffusion_plates.csv")
    dataset = dataset.to_dict("records")
    dataset1 = [item for item in dataset if item["set"] == "a_plate"]
    dataset2 = [item for item in dataset if item["set"] == "a_dinner_plate"]

    args = {
        "num_rounds": 2,
        "num_samples": 10,
        "num_hypotheses": 10,
        "seed": 0,
        "prompt": "CLIP_FRIENDLY",
        "model": "gpt-4",
        "captioner": {
            "prompt": "Describe this image",
            "model": "llava",
        },
    }

    proposer = LLMProposer(args)
    hypotheses, _, _ = proposer.propose(dataset1, dataset2)
    print(hypotheses)

    args = {
        "num_rounds": 2,
        "num_samples": 10,
        "num_hypotheses": 10,
        "seed": 0,
        "prompt": "VLM_PROMPT",
        "model": "llava",
    }


if __name__ == "__main__":
    test_proposers()
