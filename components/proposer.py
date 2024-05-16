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

class LLMProposer(Proposer):

    question_diff_prompt = """I have a list of user questions group into either A or B and I would like to understand the differences between these groups. Please list any noticeable differences in these groups. lease output a list differences between the two outputs with relation to specific axes of variation. Are there any general patterns, clusters, or variations you notice in the outputs? Try to give patterns that are specific enough that someone could reliably produce new questions that would belong to group A or group B, and that they could understand what it means to be higher or lower on that specific axis. 
    
    Here are the questions:
    {text}
    
    Please output a numbered list of differences between the two groups of questions. If there are no clear differences, please output "No differences found"."""

    combine_two_sides = """
    I have two lists of questions, 1 and 2, and I would like to understand the differences between these two groups. To do this I have fed in the questions from both groups into a language model and asked for the differences between the two groups. Here is the output of comparing group 1 and 2 (named A and B):
    
    {left_output}

    To ensure that the differences are not due to the order of the questions, I have also compared group 2 and 1 (group 2 is now A and group 1 is now B). Here is the output of comparing group 2 and 1:

    {right_output}

    Please use this to determine if there are any differences between the two groups of questions that are consistent across both comparisons. For instance, if group 1 was given quality 1 and group 2 quality 2 when comaring groups 1 and 2, this would be correct if when comparing group 2 to group 1 the output gives group 2 quality 1 and group 1 quality 2. If none of the differences are consistent across both comparisons, please output "No consistent differences found".
    """

    def __init__(self, args: Dict):
        super().__init__(args)
        self.batch_size = self.args.batch_size

    def propose_one_side(self, texts1: List[str], texts2: List[str]) -> Tuple[List[str], Dict]:
        # batch the texts and call llm to get differences
        prompt = self.question_diff_prompt.format(text="Group A:" + "\n".join(texts1) + "\n\nGroup B:" + "\n".join(texts2))
        output = get_llm_output(prompt, 'claude-3-opus-20240229')
        # converted = get_llm_output(self.conversion.format(axes=output), 'claude-3-opus-20240229')
        logs = {"prompt": prompt, "output": output, "conversion_prompt": self.conversion.format(axes=output)}
        return output, logs
    
    def propose(self, texts1: List[str], texts2: List[str]):
        max_size = 30
        sample_texts_1, sample_texts_2 = random.sample(texts1, min(len(texts1), max_size)), random.sample(texts2, min(len(texts2), max_size))
        left_output, left_logs = self.propose_one_side(sample_texts_1, sample_texts_2)
        right_output, right_logs = self.propose_one_side(sample_texts_2, sample_texts_1)

        combined = get_llm_output(self.combine_two_sides.format(left_output=left_output, right_output=right_output), 'claude-3-opus-20240229')
        return {
            "left_output": left_output,
            "right_output": right_output,
            "combined": combined,
            "logs": {
                "left": left_logs,
                "right": right_logs,
                "combined": combined
            }
        
        }
    
def extract_questions(text):
    # Remove leading/trailing whitespace and newlines
    text = text.strip()

    # Split the text into lines
    lines = text.split('\n')

    questions = []
    current_question = ''

    for line in lines:
        # Check if the line starts with a number or a bullet point
        if re.match(r'^(\d+|[-*])\.\s', line.strip()):
            if current_question:
                questions.append(current_question.strip())
            current_question = line.strip()
        else:
            current_question += ' ' + line.strip()

    # Append the last question
    if current_question:
        questions.append(current_question.strip())

    return questions
    
class LLMProposerWithQuestion(LLMProposer):

    question_diff_prompt = """I have two lists of questions, A and B, and I would like to understand the differences between these two groups. To do this I have fed in the questions from both groups into a language model and asked for the differences between the two groups and got the following output:

    {output}

    I want validate these differences by coming up with a set of questions that i can ask and llm (e.g. how open ended is this question?) such that i can correctly categorize a question in group a or b given the answers. Please output a bulleted list of questions that i can ask to categorize a question into group A or B. If the questions are nearly identical, please write "No differences found."
    """

    def __init__(self, args: Dict):
        super().__init__(args)
        self.batch_size = self.args.batch_size
    
    def propose(self, texts1: List[str], texts2: List[str]):
        diff_results = super().propose(texts1, texts2)
        output = diff_results["combined"]
        questions = get_llm_output(self.question_diff_prompt.format(output=output), 'claude-3-opus-20240229')
        return extract_questions(questions), questions
    
class LLMPairwiseProposerWithQuestion(Proposer):
    def __init__(self, args: Dict):
        super().__init__(args)
        self.systems_prompt = "Given a dataset of text outputs from two different large language models (LLMs), your task is to analyze and summarize the data based on specific characteristics. The goal is to identify and cluster similar behaviors or traits within the outputs, summarizing these into a concise list of commonly observed behaviors for each model. This analysis will help in understanding the general behaviors of these models for auditing, error discovery, and comparison purposes. Your outputs adhere to the format given by the user."
        self.smaller_systems_prompt = "You are a helpful assistant. Your outputs adhere to the format given by the user."
        # self.model_columns = [args.model_a_column, args.model_b_column]
        # self.model_a, self.model_b = args.model_a_column, args.model_a_column

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
            texts = f"{row['question']}\nModel A:\n{row[self.args.model_a_column]}\n\nModel B:\n{row[self.args.model_b_column]}\n"
            if self.args.oz:
                prompt = OZ_PROMPT.format(text=texts, axes="\n".join([f"* {axis}" for axis in oz_axes]))
            else:
                prompt = DEFAULT_PROMPT.format(text=texts)
            response = get_llm_output(prompt, model="gpt-3.5-turbo", system_prompt=self.systems_prompt).replace("**", "")
            # results["prompt"].append(texts)
            results["question"].append(row['question'])
            results["answer_a"].append(row[self.args.model_a_column].strip('[]'))
            results["answer_b"].append(row[self.args.model_b_column].strip('[]'))
            results["response"].append(response)
            # results["axes"].append(extract_entities(response))
            axis_prompt = AXIS_CONVERSION.format(axes=response)
            axis_response = get_llm_output(axis_prompt, model="gpt-3.5-turbo", system_prompt=self.smaller_systems_prompt)
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
        # self.model_columns = [args.model_a_column, args.model_b_column]
        # self.model_a, self.model_b = args.model_a_column, args.model_a_column
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
            texts.append(f"{row['question']}\nModel A:\n{row[self.args.model_a_column]}\n\nModel B:\n{row[self.args.model_b_column]}\n")
        texts = "\n".join(texts)
        if self.args.oz:
            prompt = OZ_PROMPT.format(text=texts, axes="\n".join([f"* {axis}" for axis in oz_axes]))
        else:
            prompt = DEFAULT_PROMPT.format(text=texts)
        response = get_llm_output(prompt, model="gpt-4-0125-preview", system_prompt=self.systems_prompt).replace("**", "")
        axis_prompt = AXIS_CONVERSION.format(axes=response)
        axis_response = get_llm_output(axis_prompt, model="gpt-4-0125-preview", system_prompt=self.smaller_systems_prompt)
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
            results["answer_a"].extend(batch[self.args.model_a_column].tolist())
            results["answer_b"].extend(batch[self.args.model_b_column].tolist())
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

def parse_bullets(text):
    # Use regex to extract bullet sections, supporting "-", "*", numerical bullets, and others
    bullet_sections = re.split(r"\n\s*-\s*", text.strip())
    print(bullet_sections)
    print("-----------")
    
    result = []
    reslts_str = [] # string comprised of category and details
    
    for section in bullet_sections:
        # Normalize section by removing leading markers and spaces
        section = re.sub(r"^\s*[-*\d.]+", "", section).strip()
        
        # Split each section based on High/Low points using regular expressions
        title, *details = section.splitlines()
        parsed_details = {}
        
        for line in details:
            match = re.match(r"\s*(High|Low):\s*(.+)", line)
            if match:
                key, value = match.groups()
                parsed_details[key] = value
        
        result.append({
            "Category": title.strip(": \n"),
            "Details": parsed_details
        })
        reslts_str.append(title + " " + str(parsed_details))
    
    return [r.replace("{", "").replace("}", "") for r in reslts_str]

class LLMProposerMultiModel(LLMBatchProposer):

    def __init__(self, args: Dict):
        super().__init__(args)
        self.systems_prompt = "Given a dataset of text outputs from two different large language models (LLMs), your task is to analyze and summarize the data based on specific characteristics. The goal is to identify and cluster similar behaviors or traits within the outputs, summarizing these into a concise list of commonly observed behaviors for each model. This analysis will help in understanding the general behaviors of these models for auditing, error discovery, and comparison purposes. Your outputs adhere to the format given by the user."
        self.smaller_systems_prompt = "You are a helpful assistant. Your outputs adhere to the format given by the user."
        self.model_columns = args.models
        self.batch_size = args.proposer_batch_size

    def propose_batch(self, df):
        """
        Get differences over a list of prompts
        """

        oz_proposer_prompt = """
            The following are the result of asking a set language models to generate an answer for the same questions:

            {text}

            I am a machine learning researcher trying to figure out the major differences between these two LLM outputs so I can better compare the behavior of these models. Are there any general patterns, clusters, or variations you notice in the outputs? 

        Please output a list differences between these sets of outputs with relation to specific axes of variation. Try to give axes that a human could easily interpret and they could understand what it means to be higher or lower on that specific axis. Please ensure that the concepts used to explain what is high and low on the axis are distinct and mutually exclusive such that given any tuple of text outputs, a human could easily and reliably determine which model is higher or lower on that axis.
        
        Here are some axes of variation to consider:

        {axes}

        This list is not exhaustive, please add new axes in your response even if it does not fit under one of these categories. If the outputs are roughly the same along one of the provided axes do not include it. 

        The format of response should be a bulleted list of differences, one bullet for each axis. The format should be
        - {{axis_1}}: {{difference}}
        - {{axis_2}}: {{difference}}
            
            Please output differences which have a possibility of showing up in future unseen data and which would be useful for a human to know about when deciding with LLM to use. For each axis, define clearly and succinctly what constitutes a high or low score, ensuring these definitions are mutually exclusive. For each axis, also provide an explanation of what in the text proves this difference exists. Please order your response in terms of the most prominent differences between the two outputs. If the outputs are nearly identical, please write "No differences found."
        """
        proposer_prompt = """
            The following are the result of asking a set language models to generate an answer for the same questions:

            {text}

            I am a machine learning researcher trying to figure out the major differences between these two LLM outputs so I can better compare the behavior of these models. Are there any general patterns, clusters, or variations you notice in the outputs? 

        Please output a list differences between these sets of outputs with relation to specific axes of variation. Try to give axes that a human could easily interpret and they could understand what it means to be higher or lower on that specific axis. Please ensure that the concepts used to explain what is high and low on the axis are distinct and mutually exclusive such that given any tuple of text outputs, a human could easily and reliably determine which model is higher or lower on that axis.
        This list is not exhaustive, please add new axes in your response even if it does not fit under one of these categories. If the outputs are roughly the same along one of the provided axes do not include it. 

        The format should be
        - {{axis_1}}: {{difference}}
        - {{axis_2}}: {{difference}}
            
            Please output differences which have a possibility of showing up in future unseen data and which would be useful for a human to know about when deciding with LLM to use. For each axis, define clearly and succinctly what constitutes a high or low score, ensuring these definitions are mutually exclusive. For each axis, also provide an explanation of what in the text proves this difference exists. Please order your response in terms of the most prominent differences between the two outputs. If the outputs are nearly identical, please write "No differences found."
        """

        axis_convert = """The following are the axes of variation that you can consider when comparing the two outputs along with a description of how language model outputs vary along that axis:

            {axes}

            I want to formalize exactly what it means to be high and low on each axis. For each axis, I want you to provide a description of what it means to be high and low on that axis so that I can place future model outputs along this axis. Your output should be in this format:

            - {{axis_1}}:
                High: {{description of high}}
                Low: {{description of low}}

            - {{axis_2}}:
                High: {{description of high}}
                Low: {{description of low}}

            Please ensure that the description what is high and low on the axis are distinct and mutually exclusive such that given any unseen pair of text outputs, a human could easily and reliably determine which model is higher or lower on that axis. Please keep the axis name and descriptions of what is high and low are less than 5 words each.
        """

        assert "question" in df.columns, "'question' column not in dataset"
        random.seed(self.args.seed)
        oz_axes = ["Tone", "Format", "Level of Detail", "Ability to answer", "Safety", "Approach", "Creativity", "Fluency and crammatical correctness", "Adherence to prompt"]

        # get per question differences
        texts = []
        # shuffle args.models
        shuffled_cols = random.sample(self.model_columns, len(self.model_columns))
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            for j, model in enumerate(shuffled_cols):
                texts.append(f"{row['question']}\nModel {j}:\n{row[model]}\n")
        texts = "\n".join(texts)
        if self.args.oz:
            prompt = oz_proposer_prompt.format(text=texts, axes="\n".join([f"* {axis}" for axis in oz_axes]))
        else:
            prompt = proposer_prompt.format(text=texts)
        response = get_llm_output(prompt, model=self.args.proposer_model, system_prompt=self.systems_prompt).replace("**", "")
        if "LLM Error" in response:
            exit(0)
        axis_prompt = axis_convert.format(axes=response)
        axis_response = get_llm_output(axis_prompt, model="gpt-4", system_prompt=self.smaller_systems_prompt)
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
        results = {"question":[], "response": [], "axis_response": [], "topic": []}
        llm_logs = []
        # partition df by column topic then batch 
        topic_dfs = [df[df['topic'] == topic] for topic in df['topic'].unique()]
        for topic_df in topic_dfs:
            print(f"Proporing for topic {topic_df['topic'].iloc[0]} of length {len(topic_df)}")
            # add tqdm
            for batch_start in range(0, len(topic_df), self.batch_size):
                batch = topic_df.iloc[batch_start:batch_start + self.batch_size]
                assert batch["topic"].nunique() == 1, "Batch should have the same topic"
                response, axis_response, logs = self.propose_batch(batch)
                results["question"].extend(batch['question'].tolist())
                results["response"].extend([response] * len(batch))
                results["axis_response"].extend([axis_response] * len(batch))
                results["topic"].extend(batch['topic'].tolist())
                llm_logs.append(logs)

        results = pd.DataFrame(results)
        pairwise_differences = results[['question', 'response', 'axis_response']]
        llm_logs = pd.DataFrame(llm_logs)

        results["no_difference_detected"] = results["response"].apply(lambda x: is_match(x, "No differences found"))
        results = results[~results["no_difference_detected"]]

        # cluster per axis differences
        results['axis_description'] = results['axis_response'].apply(parse_bullets)
        results = results.explode('axis_description')

        all_axis_descriptions = list(set(results['axis_description']))
        return all_axis_descriptions, llm_logs, pairwise_differences, results