# general imports
import sys
import json
import openai
import time
import pandas as pd
import datetime
from tqdm import tqdm
import argparse
import os
import io
import re, string
import numpy as np
import statistics
import heapq


# to load the api implementation
# Add parent directory to path (works whether running from MIRAI root or agents/ dir)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "APIs"))
sys.path.insert(0, os.path.join(parent_dir, "agent_prompts"))

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn" 

import APIs.api_implementation as api
from APIs.api_implementation import (Date, DateRange, ISOCode, Country, CAMEOCode, Relation, Event, NewsArticle,
                        map_country_name_to_iso, map_iso_to_country_name, map_relation_description_to_cameo,
                        map_cameo_to_relation,
                        get_parent_relation, get_child_relations, get_sibling_relations, count_events, get_events,
                        get_entity_distribution, get_relation_distribution, count_news_articles, get_news_articles,
                        browse_news_article,
                        set_default_end_date, get_default_end_date, use_end_date)
print('loaded api_implementation')

# to load libraries for prompting and planning
import importlib
from typing import List, Dict, Any
import tiktoken
from pandas import DataFrame
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import random

# to load prompt template
from agent_prompts.prompt_extraction import extraction_prompt

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


# get number of gpus
def get_num_gpus():
    global num_gpus
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    return


# color print
def red(msg):
    return "\033[91m" + msg + "\033[0m"


def set_seed(seed_value=42):
    """Sets the seed for reproducibility across different libraries used."""

    # 1. Set `numpy` seed
    np.random.seed(seed_value)

    # 2. Set `random` library seed
    random.seed(seed_value)

    # 3. Set `torch` seed for CPU and CUDA
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True # May reduce performance
        torch.backends.cudnn.benchmark = False # Disable benchmark for determinism

    # 4. Set `PYTHONHASHSEED` environment variable
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # Note: Restart Python interpreter after setting for full effect, especially in Jupyter.

    # 5. Set seed for `transformers` library (if you are using models from it)
    try:
        import transformers
        transformers.set_seed(seed_value) # Sets seed for numpy, torch, and python.random in transformers
    except ImportError:
        print("Transformers library not found, skipping transformers seed setting.")

    # 6. Set seed for `vllm` library (if you are using it for LLM inference) -> in function vllm engine and random sample params

    print(f"Seed set to {seed_value} for NumPy, random, torch, transformers (if available), vLLM (if available), and PYTHONHASHSEED.")
    print("Note: Determinism of OpenAI API responses and complex operations may still vary.")


class DirectAgent:
    def __init__(self,
                 prompt_module,
                 direct_llm_name = 'Llama-3.1-8B-Instruct',
                 temperature: float = 0.4,
                 seed: int = 42,
                 extractor_model_name: str = "gpt-4o-mini-2024-07-18"
                 ) -> None:

        self.answer = ''
        self.scratchpad = ''
        self.prompt = ''
        
        self.finished = False
        self.end_state = ''

        self.step_n = 1

        self.direct_name = direct_llm_name
        self.prompt_module = prompt_module

        self.sys_prompt = prompt_module.sys_relation_prompt
        self.agent_prompt = prompt_module.relation_prompt

        self.json_log = []

        # the temperature of the model generation
        self.temp = temperature


        if 'gpt-3.5' in direct_llm_name:
            self.max_token_length = 15000
            self.llm = ChatOpenAI(temperature=self.temp,
                     max_tokens=2048,
                     model_name=direct_llm_name,
                     openai_api_key=OPENAI_API_KEY)
        elif 'gpt-4o-mini' in direct_llm_name:
            self.max_token_length = 16384
            self.llm = ChatOpenAI(temperature=self.temp,
                        max_tokens=2048,
                        model_name="gpt-4o-mini-2024-07-18",
                        openai_api_key=OPENAI_API_KEY)            
        elif 'gpt-4' in direct_llm_name:
            self.max_token_length = 128000
            self.llm = ChatOpenAI(temperature=self.temp,
                     max_tokens=2048,
                     model_name=direct_llm_name,
                     openai_api_key=OPENAI_API_KEY)
        
        
        else:
            if "deepseek" in direct_llm_name.lower():
                llm_name = "deepseek-ai/" + direct_llm_name
                if "llama" in direct_llm_name.lower():
                    self.max_token_length = 32000
                else:
                    self.max_token_length = 32000
            elif 'llama' in direct_llm_name.lower():
                if "GPTQ" in direct_llm_name:
                    llm_name = "TechxGenus/" +  direct_llm_name
                    self.max_token_length = 128000
                else:
                    llm_name = "meta-llama/" + direct_llm_name
                    self.max_token_length = 32000
            elif 'mistral' in direct_llm_name.lower():
                llm_name = 'mistralai/' + direct_llm_name
                self.max_token_length = 8000

            self.llm = LLM(model=llm_name,
                           tensor_parallel_size=num_gpus, # 2 for 2 GPUs
                           dtype=torch.float16,
                           gpu_memory_utilization=0.9,
                           disable_log_stats=True,
                           max_model_len=8192,  # Explicitly set to avoid rope_scaling issues
                           trust_remote_code=True)
            self.sample_params = SamplingParams(temperature=self.temp,
                                                max_tokens=2048,
                                                seed=seed,
                                                # stop=self.stop_list,
                                                include_stop_str_in_output=False)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
            self.max_prompt_len = 0

        
        # self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # self.answer_extractor = ChatOpenAI(temperature=0.2,
        #              max_tokens=2048,
        #              model_name="gpt-3.5-turbo-0125",
        #              openai_api_key=OPENAI_API_KEY)
            
        
        self.enc = tiktoken.encoding_for_model(extractor_model_name)
        self.answer_extractor = ChatOpenAI(temperature=0.0,
                     max_tokens=2048,
                     model_name=extractor_model_name,
                     openai_api_key=OPENAI_API_KEY)

        self.__reset_agent()

    def run(self, query_info, reset=True):

        if reset:
            self.__reset_agent()

        self.query_info = query_info
        sys_prompt = self._build_sys_prompt()

        prompt, answer = self.prompt_agent()

        self.step_n += 1
        self.scratchpad += f'{answer}'

        if len(prompt) == 0: # openai error
            self.finished = True
            self.end_state = answer
            print(f'\n======\nAnswer with Error: {answer}')
        else:
            self.finished = True
            self.end_state = 'Final Answer'
            print(f'\n======\nFinal Answer: {answer}')

        # use gpt-4o-mini to extract the answer
        ext_prompt, ext_request, self.answer = self.extract_answer(answer)
        return self.end_state, self.step_n-1, self.answer, self.scratchpad, self.json_log,  sys_prompt, prompt, ext_prompt, ext_request

    def extract_answer(self, final_info_str):
        print('\n==\nExtracting final answer...')

        curr_date_str = get_default_end_date()
        curr_date = datetime.datetime.strptime(curr_date_str, '%Y-%m-%d')
        curr_date_nlp = curr_date.strftime('%B %d, %Y')
        ext_prompt = extraction_prompt.format(
            current_date_nlp=curr_date_nlp,
            actor1_name=self.query_info['Actor1CountryName'],
            actor2_name=self.query_info['Actor2CountryName'],
            future_date_nlp=self.query_info['DateNLP'],
            future_date=self.query_info['DateStr'],
            actor1_code=self.query_info['Actor1CountryCode'],
            actor2_code=self.query_info['Actor2CountryCode'],
            info=final_info_str
            )
        ext_request = self.answer_extractor([HumanMessage(content=ext_prompt)]).content
        print('\nExtraction request:\n', ext_request)
        answer = self.extract_and_verify_dictionary(ext_request)
        print('\nFinal answer:\n', answer if len(answer) > 0 else 'No answer extracted.')
        return ext_prompt, ext_request, answer

    def extract_and_verify_dictionary(self, input_string):
        # Remove spaces, newlines, and any other characters that might cause issues
        cleaned_input = re.sub(r'\s+', '', input_string)

        # Regular expression to find content inside <answer> tags
        pattern = r'<answer>(.*?)</answer>'
        # Search for the pattern
        match = re.search(pattern, cleaned_input)

        # Check if a match was found
        if match:
            # Extract the content between the tags
            content = match.group(1)
            content.strip(' \n')
            try:
                # Try to parse the content as JSON
                parsed_dict = json.loads(content)

                # Check if the parsed content is a dictionary
                if isinstance(parsed_dict, dict):
                    return json.dumps(parsed_dict)  # Return the string representation of the dictionary
                else:
                    return ''  # Not a dictionary
            except json.JSONDecodeError:
                return ''  # Content was not valid JSON
        else:
            return ''  # No content found between tags

    def prompt_agent(self):
        trial = 0
        sys_prompt = self._build_sys_prompt()
        prompt = self._build_agent_prompt()
        messages = [SystemMessage(content=sys_prompt),
                    HumanMessage(content=prompt)]
        
        if "gpt" in self.direct_name:
            while trial < 3:
                try:
                    request = self.llm(messages).content
                    # print(request)
                    return prompt, request.strip(' \n')
                except Exception as e:
                    print(f"Error: {e}")
                    print('prompt len:' + str(len(self.enc.encode(sys_prompt + prompt))))
                    time.sleep(5)
                    trial += 1
                    err = str(e)
                    return '', err
        
        elif 'llama' in self.direct_name.lower():
            # format llama3 instruct prompt
            instruct_prompt = self.generate_llama3_instruct_prompt(sys_prompt, prompt, self.scratchpad)
            # compute the token length of the instruct prompt
            prompt_len = len(self.tokenizer.tokenize(instruct_prompt))
            self.max_prompt_len = max(prompt_len, self.max_prompt_len)
            if prompt_len > self.max_token_length:
                print(f"Max prompt length {self.max_prompt_len} exceeds the max token length {self.max_token_length}")

            try:
                response = self.llm.generate(instruct_prompt, self.sample_params)[0].outputs[0].text
                # print("-------instruct_prompt-------")
                # print(instruct_prompt)
                # print("-------instruct_prompt-------")
                return prompt, response.strip(' \n')
            except Exception as e:
                print(red(f"Error: {e}"))
                return '', str(e)
    

        elif 'mistral' in self.direct_name.lower():
            instruct_prompt = self.generate_mistral_instruct_prompt(sys_prompt, prompt, self.scratchpad)
            # compute the token length of the instruct prompt
            prompt_len = len(self.tokenizer.tokenize(instruct_prompt))
            self.max_prompt_len = max(prompt_len, self.max_prompt_len)
            if prompt_len > self.max_token_length:
                print(f"Max prompt length {self.max_prompt_len} exceeds the max token length {self.max_token_length}")

            try:
                response = self.llm.generate(instruct_prompt, self.sample_params)[0].outputs[0].text
                # print("-------instruct_prompt-------")
                # print(instruct_prompt)
                # print("-------instruct_prompt-------")
                return prompt, response.strip(' \n')
            except Exception as e:
                print(red(f"Error: {e}"))
                return '', str(e)


    def generate_llama3_instruct_prompt(self, sys_prompt, user_prompt, assistant_prompt):
        instruct_prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
        instruct_prompt += sys_prompt.strip(' \n') + '<|eot_id|>'
        instruct_prompt += '<|start_header_id|>user<|end_header_id|>\n\n'
        instruct_prompt += user_prompt.strip(' \n') + '<|eot_id|>'
        instruct_prompt += '<|start_header_id|>assistant<|end_header_id|>\n\n'
        instruct_prompt += assistant_prompt.strip(' \n')
        return instruct_prompt        

        
    def generate_mistral_instruct_prompt(self, sys_prompt, user_prompt, assistant_prompt):
        instruct_prompt = '<s>[INST] '
        instruct_prompt += sys_prompt.strip(' \n') + '\n\n' + user_prompt.strip(' \n')
        instruct_prompt += ' </INST> '
        instruct_prompt += assistant_prompt.strip(' \n')
        return instruct_prompt


    def _build_sys_prompt(self) -> str:
        curr_date_str = get_default_end_date()
        curr_date = datetime.datetime.strptime(curr_date_str, '%Y-%m-%d')
        curr_date_nlp = curr_date.strftime('%B %d, %Y')
        return self.sys_prompt.format(current_date_nlp = curr_date_nlp)

    def _build_agent_prompt(self) -> str:
        curr_date_str = get_default_end_date()
        curr_date = datetime.datetime.strptime(curr_date_str, '%Y-%m-%d')
        curr_date_nlp = curr_date.strftime('%B %d, %Y')
        # print(self.query_info)
        return self.agent_prompt.format(
            current_date_nlp = curr_date_nlp,
            actor1_name = self.query_info['Actor1CountryName'],
            actor2_name = self.query_info['Actor2CountryName'],
            future_date_nlp = self.query_info['DateNLP'],
            future_date = self.query_info['DateStr'],
            actor1_code = self.query_info['Actor1CountryCode'],
            actor2_code = self.query_info['Actor2CountryCode'],
            information = self.query_info['Information']
            )
    
    def is_finished(self) -> bool:
        return self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ''
        self.scratchpad = ''
        self.json_log = []

    def extract_content(self, data):
        # Pattern matches optional ``` followed by optional language spec and newline, then captures all content until optional ```
        pattern = r'```(?:\w+\n)?(.*?)```|(.+)'
        match = re.search(pattern, data, re.DOTALL)
        if match:
            # Return the first non-None group
            return match.group(1) if match.group(1) is not None else match.group(2)
        return data  # Return data if no pattern matched



dict_binary2first = {
    "mediation": ['01', '02', '03', '04', '05', '06', '07', '08'],
    "conflict": ['09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
}
dict_first2binary = {item: key for key, value in dict_binary2first.items() for item in value}

dict_quad2first = {
    "verbal cooperation": ['01', '02', '03', '04'],
    "material cooperation": ['05', '06', '07', '08'],
    "verbal conflict": ['09', '10', '11', '12', '13', '14', '15', '16'],
    "material conflict": ['17', '18', '19', '20']
}
dict_first2quad = {item: key for key, value in dict_quad2first.items() for item in value}

dict_code2relation = json.load(open(os.path.join(parent_dir, "data", "info", "dict_code2relation.json"), 'r'))
codes = list(dict_code2relation.keys())
first_level_codes = [code for code in codes if len(code) == 2]
second_level_codes = [code for code in codes if len(code) == 3]

# calculate micro precision, recall, f1
def calculate_metrics(preds, golds):
    tp, fp, fn = 0, 0, 0
    for pred, gold in zip(preds, golds):
        for p in pred:
            if p in gold:
                tp += 1
            else:
                fp += 1
        for g in gold:
            if g not in pred:
                fn += 1
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


def calculate_macro_metrics(preds, golds):
    precisions, recalls, f1s = [], [], []
    for pred, gold in zip(preds, golds):
        tp, fp, fn = 0, 0, 0
        for p in pred:
            if p in gold:
                tp += 1
            else:
                fp += 1
        for g in gold:
            if g not in pred:
                fn += 1
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        precision *= 100
        recall *= 100
        f1 *= 100
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    precision = sum(precisions) / len(precisions) if len(precisions) > 0 else 0
    recall = sum(recalls) / len(recalls) if len(recalls) > 0 else 0
    f1 = sum(f1s) / len(f1s) if len(f1s) > 0 else 0
    # print(precisions)


    return precision, recall, f1

# calculate discrete KL divergence between two list of distributions
def calculate_kl_divergence(preds, golds, dict_all_items):
    kl_divergences = []
    epsilon = 1e-10  # Small constant to prevent log(0)
    items = dict_all_items.keys()  # All possible items

    for pred, gold in zip(preds, golds):
        # Calculate distributions with all items included, adding epsilon to prevent division by zero
        pred_dist = {item: pred.count(item) / len(pred) if len(pred) > 0 else 0 for item in items}
        gold_dist = {item: gold.count(item) / len(gold) if len(gold) > 0 else 0 for item in items}

        # Adding epsilon to all probabilities to handle cases where count might be zero
        pred_dist = {item: pred_dist[item] + epsilon for item in items}
        gold_dist = {item: gold_dist[item] + epsilon for item in items}

        # Calculate KL divergence
        kl_divergence = sum(gold_dist[item] * np.log(gold_dist[item] / pred_dist[item]) for item in items)
        kl_divergences.append(kl_divergence)

    # Calculate average KL divergence
    avg_kl_divergence = sum(kl_divergences) / len(kl_divergences)
    return avg_kl_divergence

# load predictions
def load_predictions(pred_file, task="relation"):
    logs = json.load(open(pred_file))
    answer_str = logs[-1]["answer"]
    try:
        answer = eval(answer_str)
        if task == "relation":
            first_level = list(answer.keys())
            first_level = list(set([item for item in first_level if item in first_level_codes]))

            second_level = []
            for first in first_level:
                value = answer[first]
                for item in value:
                    if item[:2] == first and item in second_level_codes:
                        second_level.append(item)
            second_level = list(set([item for item in second_level]))
            return first_level, second_level
        else:
            return answer
    except:
        if task == "relation":
            return [], []
        else:
            return []

def load_end_state(pred_file):
    logs = json.load(open(pred_file))
    end_state = logs[-1]["end_state"]
    n_steps = logs[-1]["n_steps"]
    return end_state, n_steps

# eval relation predictions
def eval_relation(data_query, setting_output_dir, args):
    # load gold relation predictions
    golds_first_level, golds_second_level = [], []
    gold_binary_level, gold_quad_level = [], []
    gold_binary_level_dedup, gold_quad_level_dedup = [], []
    for rowid, row in tqdm(data_query.iterrows(), total=len(data_query)):
        answer = eval(row["AnswerDict"])
        first_level = list(answer.keys())
        second_level = [item for sublist in answer.values() for item in sublist]
        golds_first_level.append(first_level)
        golds_second_level.append(second_level)

        binary_level = [dict_first2binary[item] for item in first_level]
        quad_level = [dict_first2quad[item] for item in first_level]
        gold_binary_level.append(binary_level)
        gold_quad_level.append(quad_level)

        binary_level_dedup = list(set(binary_level))
        quad_level_dedup = list(set(quad_level))
        gold_binary_level_dedup.append(binary_level_dedup)
        gold_quad_level_dedup.append(quad_level_dedup)

    # load all round predictions and calculate metrics
    round_metrics = {}
    dict_end_states_count = {}
    n_steps_total = 0
    for curr_round in range(args.rounds):
        preds_first_level, preds_second_level = [], []
        preds_binary_level, preds_quad_level = [], []
        preds_binary_level_dedup, preds_quad_level_dedup = [], []

        print(f"Round {curr_round + 1}")
        curr_round_output_dir = os.path.join(setting_output_dir, f"round{curr_round + 1}")

        for rowid, row in tqdm(data_query.iterrows(), total=len(data_query)):
            query_id = row['QueryId']
            pred_file = os.path.join(curr_round_output_dir, f"{query_id}.json")
            first_level, second_level = load_predictions(pred_file, task="relation")
            preds_first_level.append(first_level)
            preds_second_level.append(second_level)

            binary_level = [dict_first2binary[item] for item in first_level]
            quad_level = [dict_first2quad[item] for item in first_level]
            preds_binary_level.append(binary_level)
            preds_quad_level.append(quad_level)

            binary_level_dedup = list(set(binary_level))
            quad_level_dedup = list(set(quad_level))
            preds_binary_level_dedup.append(binary_level_dedup)
            preds_quad_level_dedup.append(quad_level_dedup)

            end_state, n_steps = load_end_state(pred_file)
            n_steps_total += n_steps
            if end_state not in dict_end_states_count:
                dict_end_states_count[end_state] = 0
            dict_end_states_count[end_state] += 1

        # calculate micro metrics
        precision_first_level, recall_first_level, f1_first_level = calculate_macro_metrics(preds_first_level, golds_first_level)
        precision_second_level, recall_second_level, f1_second_level = calculate_macro_metrics(preds_second_level, golds_second_level)
        precision_binary_level, recall_binary_level, f1_binary_level = calculate_macro_metrics(preds_binary_level_dedup, gold_binary_level_dedup)
        precision_quad_level, recall_quad_level, f1_quad_level = calculate_macro_metrics(preds_quad_level_dedup, gold_quad_level_dedup)

        # calculate KL divergence
        kl_binary_level = calculate_kl_divergence(preds_binary_level, gold_binary_level, dict_binary2first)
        kl_quad_level = calculate_kl_divergence(preds_quad_level, gold_quad_level, dict_quad2first)

        round_metrics[curr_round+1] = {
            "binary_level": {
                "precision": precision_binary_level,
                "recall": recall_binary_level,
                "f1": f1_binary_level,
                "kl": kl_binary_level},
            "quad_level": {
                "precision": precision_quad_level,
                "recall": recall_quad_level,
                "f1": f1_quad_level,
                "kl": kl_quad_level},
            "first_level": {
                "precision": precision_first_level,
                "recall": recall_first_level,
                "f1": f1_first_level},
            "second_level": {
                "precision": precision_second_level,
                "recall": recall_second_level,
                "f1": f1_second_level}
        }

    # calculate average metrics
    round_metrics["average"] = {}
    for level in ["binary_level", "quad_level", "first_level", "second_level"]:
        round_metrics["average"][level] = {}
        for metric in round_metrics[1][level]:
            round_metrics["average"][level][metric] = sum([round_metrics[round][level][metric] for round in range(1, args.rounds+1)]) / args.rounds

    # calculate max metrics: for each single query, calculate each metric for all rounds and keep the max
    max_preds_first_level, max_preds_second_level = [], []
    max_preds_binary_level, max_preds_quad_level = [], []
    max_preds_binary_level_dedup, max_preds_quad_level_dedup = [], []
    for rowid, row in tqdm(data_query.iterrows(), total=len(data_query)):
        query_id = row['QueryId']
        answer = eval(row["AnswerDict"])
        gold_second_level = [item for sublist in answer.values() for item in sublist]

        max_second_level_f1 = 0
        max_first_level, max_second_level = None, None

        for curr_round in range(args.rounds):
            curr_round_output_dir = os.path.join(setting_output_dir, f"round{curr_round + 1}")
            pred_file = os.path.join(curr_round_output_dir, f"{query_id}.json")
            first_level, second_level = load_predictions(pred_file, task="relation")
            precision_second_level, recall_second_level, f1_second_level = calculate_macro_metrics([second_level], [gold_second_level])
            if f1_second_level >= max_second_level_f1:
                max_second_level_f1 = f1_second_level
                max_first_level, max_second_level = first_level, second_level
        max_preds_first_level.append(max_first_level)
        max_preds_second_level.append(max_second_level)

        max_preds_binary_level.append([dict_first2binary[item] for item in max_first_level])
        max_preds_quad_level.append([dict_first2quad[item] for item in max_first_level])

        max_preds_binary_level_dedup.append(list(set([dict_first2binary[item] for item in max_first_level])))
        max_preds_quad_level_dedup.append(list(set([dict_first2quad[item] for item in max_first_level])))

    # calculate micro metrics
    precision_binary_level, recall_binary_level, f1_binary_level = calculate_macro_metrics(max_preds_binary_level_dedup, gold_binary_level_dedup)
    precision_quad_level, recall_quad_level, f1_quad_level = calculate_macro_metrics(max_preds_quad_level_dedup, gold_quad_level_dedup)
    precision_first_level, recall_first_level, f1_first_level = calculate_macro_metrics(max_preds_first_level, golds_first_level)
    precision_second_level, recall_second_level, f1_second_level = calculate_macro_metrics(max_preds_second_level, golds_second_level)

    # calculate KL divergence
    kl_binary_level = calculate_kl_divergence(max_preds_binary_level, gold_binary_level, dict_binary2first)
    kl_quad_level = calculate_kl_divergence(max_preds_quad_level, gold_quad_level, dict_quad2first)

    round_metrics["max"] = {
        "binary_level": {
            "precision": precision_binary_level,
            "recall": recall_binary_level,
            "f1": f1_binary_level,
            "kl": kl_binary_level},
        "quad_level": {
            "precision": precision_quad_level,
            "recall": recall_quad_level,
            "f1": f1_quad_level,
            "kl": kl_quad_level},
        "first_level": {
            "precision": precision_first_level,
            "recall": recall_first_level,
            "f1": f1_first_level},
        "second_level": {
            "precision": precision_second_level,
            "recall": recall_second_level,
            "f1": f1_second_level}
    }


    # keeps answer item only if the same item is given in the answer of at least 2 rounds
    if args.rounds >= 2:
        repeated_preds_first_level, repeated_preds_second_level = [], []
        repeated_preds_binary_level, repeated_preds_quad_level = [], []
        repeated_preds_binary_level_dedup, repeated_preds_quad_level_dedup = [], []
        for rowid, row in tqdm(data_query.iterrows(), total=len(data_query)):
            query_id = row['QueryId']
            pred_files = [os.path.join(setting_output_dir, f"round{round + 1}", f"{query_id}.json") for round in range(args.rounds)]
            answers = [load_predictions(pred_file, task="relation") for pred_file in pred_files]
            first_level = [answer[0] for answer in answers]

            first_level = [item for sublist in first_level for item in sublist]
            second_level = [answer[1] for answer in answers]
            second_level = [item for sublist in second_level for item in sublist]
            first_level = [item for item in first_level if first_level.count(item) >= 2]
            first_level = list(set(first_level))
            second_level = [item for item in second_level if second_level.count(item) >= 2]
            second_level = list(set(second_level))
            repeated_preds_first_level.append(first_level)
            repeated_preds_second_level.append(second_level)

            repeated_preds_binary_level.append([dict_first2binary[item] for item in first_level])
            repeated_preds_quad_level.append([dict_first2quad[item] for item in first_level])

            repeated_preds_binary_level_dedup.append(list(set([dict_first2binary[item] for item in first_level])))
            repeated_preds_quad_level_dedup.append(list(set([dict_first2quad[item] for item in first_level])))

        # calculate micro metrics
        precision_binary_level, recall_binary_level, f1_binary_level = calculate_macro_metrics(repeated_preds_binary_level_dedup, gold_binary_level)
        precision_quad_level, recall_quad_level, f1_quad_level = calculate_macro_metrics(repeated_preds_quad_level_dedup, gold_quad_level)
        precision_first_level, recall_first_level, f1_first_level = calculate_macro_metrics(repeated_preds_first_level, golds_first_level)
        precision_second_level, recall_second_level, f1_second_level = calculate_macro_metrics(repeated_preds_second_level, golds_second_level)

        # calculate KL divergence
        kl_binary_level = calculate_kl_divergence(repeated_preds_binary_level, gold_binary_level, dict_binary2first)
        kl_quad_level = calculate_kl_divergence(repeated_preds_quad_level, gold_quad_level, dict_quad2first)

        round_metrics["repeated"] = {
            "binary_level": {
                "precision": precision_binary_level,
                "recall": recall_binary_level,
                "f1": f1_binary_level,
                "kl": kl_binary_level},
            "quad_level": {
                "precision": precision_quad_level,
                "recall": recall_quad_level,
                "f1": f1_quad_level,
                "kl": kl_quad_level},
            "first_level": {
                "precision": precision_first_level,
                "recall": recall_first_level,
                "f1": f1_first_level},
            "second_level": {
                "precision": precision_second_level,
                "recall": recall_second_level,
                "f1": f1_second_level}
        }

    round_metrics["end_states"] = dict_end_states_count
    round_metrics["n_steps_avg"] = n_steps_total / len(data_query) / args.rounds

    # create a dataframe of the max round predictions and repeated round predictions
    data_max_round = []
    data_repeated_round = []
    column_names = []
    for level in ["binary_level", "quad_level", "first_level", "second_level"]:
        for metric in round_metrics["max"][level]:
            data_max_round.append(round_metrics["max"][level][metric])
            column_names.append(f"{level}_{metric}")
            if args.rounds >= 2:
                data_repeated_round.append(round_metrics["repeated"][level][metric])
    df_max_round = pd.DataFrame([data_max_round], columns=column_names)
    if args.rounds >= 2:
        df_repeated_round = pd.DataFrame([data_repeated_round], columns=column_names)

    # concatenate the two dataframes to two rows
    if args.rounds >= 2:
        df_max_round = pd.concat([df_max_round, df_repeated_round], axis=0)

    print(round_metrics['max']['second_level']['f1'])
    if args.rounds >= 2:
        print(round_metrics['repeated']['second_level']['f1'])

    return round_metrics, df_max_round

def load_data(dataset_name, react_result_dir, type, num_examples=None):
    data = []

    if num_examples is None:
        if dataset_name == "test_subset":
            num_examples = 100
        elif dataset_name == "test":
            num_examples = 705
    for i in range(num_examples):
        result_i = json.load(open(react_result_dir + f"{i+1}.json"))[0]
        json_log = result_i["json_log"]
        input_i = []
        for step in json_log:
            if step["include_in_extraction"] == False:
                continue
            if step["state"] == "Detect Final Answer":
                continue
            if type == "thought_observation":
                input_i.append("The expert's thought is:\n" + step['thought'] + "\nThe gathered information is:\n" + step['observation'])
            elif type == "thought_action_observation":
                input_i.append("The expert's thought is:\n" + step['thought'] + "\nThe function executed is:\n" + step['action'] + "\nThe gathered information is:\n" + step['observation'])
            elif type == "observation":
                input_i.append("The function executed is:\n" + step['action'] + "\nThe gathered information is:\n" + step['observation'])
            else:
                raise ValueError(f"Invalid type of information to gather: \n\n {type}")
            
        data.append("\n\n".join(input_i))
    return data

def define_arguments():
    # Define the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="test_subset", choices=["test", "test_subset"])
    parser.add_argument("--timediff", type=int, default=1, help="date difference from the query date to the current date")

    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125",
                        choices=["gpt-3.5-turbo-0125", 
                                "gpt-4-turbo-2024-04-09",
                                "gpt-4-1106-preview",
                                "gpt-4o-2024-05-13",
                                "gpt-4o-mini",
                                "Meta-Llama-3-8B-Instruct",
                                "Llama-3.1-8B-Instruct",
                                "Llama-3.2-3B-Instruct",
                                "Mistral-7B-Instruct-v0.2", # Mistral 7B model (?)
                                "DeepSeek-R1-Distill-Qwen-7B",
                                "DeepSeek-R1-Distill-Llama-8B",
                                "Meta-Llama-3-70B-Instruct-GPTQ"])
    parser.add_argument("--extractor_model", type=str, default="gpt-4o-mini-2024-07-18",
                        help="Model for answer extraction (default: gpt-4o-mini-2024-07-18)")
    parser.add_argument("--temperature", type=float, default=0.4, help="temperature of the model")
    parser.add_argument("--rounds", type=int, default=1, help="number of rounds")

    parser.add_argument("--plan", type=str, default="direct", help="planning strategy (direct, cot or all_once), diversification strategies (3rd_party, temporal).")
    parser.add_argument("--event_k", type=int, default=30, help="number of events to consider")
    parser.add_argument("--news_k", type=int, default=15, help="number of news articles to consider")
    parser.add_argument("--diversity", type=float, default=0.0, help="diversity of the retrieval results")

    # parser.add_argument("--max_steps", type=int, default=0, help="maximum action steps")

    parser.add_argument("--output_dir", type=str, default=os.path.join(parent_dir, "output"))
    parser.add_argument("--data_dir", type=str, default=os.path.join(parent_dir, "data", "MIRAI"))
    parser.add_argument("--api_dir", type=str, default=os.path.join(parent_dir, "APIs", "api_description_full_with_title.py"))

    parser.add_argument("--alias", type=str, default="", help="alias for the output file")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=42, help="control random seed")
    parser.add_argument("--top10_rel", action="store_true", help="use top 10 relations")

    parser.add_argument("--start_idx", type=int, default=0, help="start index for the query")
    parser.add_argument("--end_idx", type=int, default=1000, help="end index for the query")



    # Parse the arguments
    args = parser.parse_args()
    return args


# main

if __name__ == "__main__":

    args = define_arguments()

    set_seed(args.seed)

    # Access the arguments
    print("Dataset:", args.dataset)
    print("Time difference:", args.timediff)
    print("Model name:", args.model_name)
    print("Temperature:", args.temperature)
    print("Rounds:", args.rounds)
    print("Plan:", args.plan)
    print("Event k:", args.event_k)
    # print("News k:", args.news_k)
    print("Diversity:", args.diversity)

    print("Output directory:", args.output_dir)
    print("Data directory:", args.data_dir)
    print("API directory:", args.api_dir)
    print("Alias:", args.alias)


    # Unified extraction prompt
    print("Using unified extraction prompt")

    if args.top10_rel:
        os.environ['TOP_10_REL'] = "True"
    else:
        os.environ['TOP_10_REL'] = "False"

    os.environ["EVENT_DIVERSITY"] = str(args.diversity)
    # ARTICLE_DIVERSITY = 0.0
    os.environ["TOTAL_EVENT_LIMIT"] = str(args.event_k)

    os.environ["TIME_DIFF"] = str(args.timediff)

    if args.diversity > 0:
        os.environ["BOOL_DIVERSITY"] = "True"
    else:
        os.environ["BOOL_DIVERSITY"] = "False"

    # make output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    get_num_gpus()



    setting_output_dir = os.path.join(args.output_dir, args.dataset, args.model_name, "timediff{}-maxsteps{}-{}-react-{}-{}-temp{}".format(args.timediff, 20, args.plan, "func", "full", args.temperature))

    if args.alias != "":
        setting_output_dir = setting_output_dir + '-' + args.alias

    if not os.path.exists(setting_output_dir):
        os.makedirs(setting_output_dir)

    # import prompt module
    prompt_module_name = f'prompts_{args.plan}'
    prompt_module = importlib.import_module(prompt_module_name)

    # load database
    data_kg = pd.read_csv(os.path.join(args.data_dir, "data_kg.csv"), sep='\t', dtype=str)
    data_news = pd.read_csv(os.path.join(args.data_dir, "data_news.csv"), sep='\t', dtype=str)


    # load query dataset
    data_query = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'relation_query.csv'), sep='\t', dtype=str)

    query_ids = [i for i in range(1, len(data_query) + 1)]

    agent = DirectAgent(prompt_module=prompt_module,
                        direct_llm_name=args.model_name, temperature=args.temperature, seed=args.seed,
                        extractor_model_name=args.extractor_model)
    

    with get_openai_callback() as cb:
        for curr_round in range(args.rounds):
            print(f"Round {curr_round + 1}")

            alias_suffix = f"-{args.alias}" if args.alias else ""
            old_setting_output_dir = os.path.join("../output/react", args.dataset, args.model_name, f"timediff{args.timediff}-maxsteps20-react-func-full-temp{args.temperature}{alias_suffix}/round{curr_round+1}/")
            # Determine number of examples to load
            if args.debug:
                num_load = 3
            elif args.start_idx > 0 or args.end_idx < (100 if args.dataset == "test_subset" else 705):
                num_load = args.end_idx - args.start_idx
            else:
                num_load = None  # Load all
            information_data = load_data(args.dataset, old_setting_output_dir, args.plan[10:], num_examples=num_load)
            
            # llama_70_dir = os.path.join("../output/react", args.dataset, "Meta-Llama-3-70B-Instruct-GPTQ", f"timediff{args.timediff}-maxsteps20-react-func-full-temp{args.temperature}-{args.alias}/round{curr_round+1}/")
            # information_data = load_data(args.dataset, llama_70_dir, args.plan[10:])

            
            # make output directory
            curr_round_output_dir = os.path.join(setting_output_dir, f"round{curr_round + 1}")
            if not os.path.exists(curr_round_output_dir):
                os.makedirs(curr_round_output_dir)

            end_id = len(data_query)
            if os.path.exists(os.path.join(curr_round_output_dir, f"{end_id}.json")):
                print(f"The round {curr_round_output_dir} is already completed.")
                continue

            if args.start_idx > 0 and args.end_idx < len(data_query):
                data_query = data_query[args.start_idx:args.end_idx]

            # run the agent
            if args.debug:
                data_query = data_query[:3]

            for rowid, row in tqdm(data_query.iterrows(), total=len(data_query)):
                # rowid starts with 0, query_id starts with 1

                query_id = row['QueryId']
                query_date = row['DateStr']


                curr_date = datetime.datetime.strptime(query_date, '%Y-%m-%d') - datetime.timedelta(days=args.timediff)
                curr_date_str = curr_date.strftime('%Y-%m-%d')

                print(f'Current date: {curr_date_str}')
                set_default_end_date(curr_date_str)
                print(f'End date: {get_default_end_date()}')

                use_end_date()

                curr_date_nlp = curr_date.strftime('%B %d, %Y')
                future_date_nlp = row['DateNLP']

                row["Information"] = information_data[rowid]

                output_file_dir = os.path.join(curr_round_output_dir, query_id + '.json')

                if os.path.exists(output_file_dir):
                    continue

                result = [{}]

                end_state, n_steps, answer, scratchpad, json_log, sys_prompt, query_prompt, ext_prompt, ext_request  = agent.run(row)

                result[-1]['query_id'] = query_id
                result[-1]['n_steps'] = n_steps
                result[-1]['end_state'] = end_state
                result[-1]['answer'] = answer
                result[-1]['gt_answer'] = row['AnswerDict']
                result[-1]['json_log'] = json_log
                result[-1]['sys_prompt'] = sys_prompt
                result[-1]['query_prompt'] = query_prompt
                result[-1]['scratchpad'] = scratchpad
                result[-1]['ext_prompt'] = ext_prompt
                result[-1]['ext_request'] = ext_request

                # write to json file
                with open(output_file_dir, 'w') as f:
                    json.dump(result, f, indent=4)
        
    print(cb)
    if "llama" in args.model_name.lower() or "mistral" in args.model_name.lower():
        print("CHECK_TOKEN_LENGTH: max token length over these test samples is :", agent.max_prompt_len)

    ########### Evaluate the results ###########

    
    # NO NEED TO RELOAD DATA_QUERY

    # load query dataset
    # data_query = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'relation_query.csv'), sep='\t', dtype=str)

    eval_results, eval_df = eval_relation(data_query, setting_output_dir, args)

    # save evaluation results
    eval_dir = os.path.join("../output_eval/simpleReact/", args.dataset)

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # eval_file = "round{}-{}-timediff{}-maxsteps{}-{}-{}-{}-temp{}".format(args.rounds, args.model_name, args.timediff, args.max_steps, args.plan, args.action, args.api, args.temperature)
    # eval_file += f"-diversity{diversity}"
    # eval_file += f'-eventk{eventk}'
    # eval_file += f'-article_diversity{a_diversity}'


    # eval_file = f"round1-Llama-3.1-8B-Instruct-timediff1-maxsteps0-{args.plan}-none-none-temp0.0"

    # eval_file = "round{}-{}-timediff{}-{}-eventK{}-div{}-temp{}".format(args.rounds, args.model_name, args.timediff, args.plan, args.event_k, args.
    # diversity, args.temperature)
    

    eval_file = f"round{args.rounds}-{args.model_name}-timediff{args.timediff}-maxsteps{20}-{args.plan}-react-func-full-temp{args.temperature}"
    if args.alias != "":
        eval_file = eval_file + '-' + args.alias

    json.dump(eval_results, open(os.path.join(eval_dir, eval_file + '.json'), 'w'), indent=4)
    eval_df.to_csv(os.path.join(eval_dir, eval_file + '.csv'), index=False, sep='\t')



