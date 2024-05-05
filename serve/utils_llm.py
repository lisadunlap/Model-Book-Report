import json
import logging
import os
import threading
from typing import List

import lmdb
import openai
from openai import OpenAI
import anthropic
import datetime
from wandb.sdk.data_types.trace_tree import Trace

from serve.global_vars import LLM_CACHE_FILE, VICUNA_URL, LLM_EMBED_CACHE_FILE, OPENAI_API_KEY, ANTHROPIC_API_KEY, LLAMA_URL
from serve.utils_general import get_from_cache, save_to_cache, save_emb_to_cache, get_emb_from_cache

logging.basicConfig(level=logging.ERROR)

if not os.path.exists(LLM_CACHE_FILE):
    os.makedirs(LLM_CACHE_FILE)

llm_cache = lmdb.open(LLM_CACHE_FILE, map_size=int(1e11))
llm_embed_cache = lmdb.open(LLM_EMBED_CACHE_FILE, map_size=int(1e11))
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]
anthropic.api_key = os.environ["ANTHROPIC_API_KEY"]


def get_llm_output(prompt: str, model: str, cache = True, system_prompt = None, history=[], trace_name="root_span") -> str:

    openai.api_base = "https://api.openai.com/v1" if model != "vicuna" else VICUNA_URL
    if 'claude' not in model:
        client = OpenAI()
    else:
        client = anthropic.Anthropic()
        
    systems_prompt = "You are a helpful assistant." if not system_prompt else system_prompt

    if model in ["gpt-3.5-turbo", "gpt-4", "gpt-4-0125-preview"]:
        messages = [{"role": "system", "content": systems_prompt}] + history + [
            {"role": "user", "content": prompt},
        ]
    elif 'claude' in model:
        messages = history + [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = prompt
    key = json.dumps([model, messages])

    cached_value = get_from_cache(key, llm_cache) if cache else None
    if cached_value is not None:
        print("LLM Cache Hit")
        logging.debug(f"LLM Cache Hit")
        # create a span in wandb
        # create_and_log_trace(trace_name, model, system_prompt, prompt, cached_value, cached=True)
        return cached_value
    else:
        print("LLM Cache Miss")
        logging.debug(f"LLM Cache Miss")

    for _ in range(3):
        try:
            if model in ["gpt-3.5-turbo", "gpt-4", "gpt-4-0125-preview"]:
                start_time_ms = datetime.datetime.now().timestamp() * 1000
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                end_time_ms = round(datetime.datetime.now().timestamp() * 1000)  # logged in milliseconds
                response = completion.choices[0].message.content.strip()
            elif 'claude' in model:
                completion = client.messages.create(
                    model="claude-3-opus-20240229",
                    messages=messages,
                    max_tokens=1024,
                    system=systems_prompt,
                )
                response = completion.content[0].text
            elif model == "vicuna":
                completion = client.chat.completions.create(
                    model="lmsys/vicuna-7b-v1.5",
                    prompt=prompt,
                    max_tokens=256,
                    temperature=0.7,  # TODO: greedy may not be optimal
                )
                response = completion.choices[0].message.content.strip()
            save_to_cache(key, response, llm_cache)

            # create a span in wandb
            # create_and_log_trace(trace_name, model, system_prompt, prompt, response, start_time_ms, end_time_ms, cached=False)
            return response

        except Exception as e:
            logging.error(f"LLM Error: {e}")
            continue
    return "LLM Error: Cannot get response."

def create_and_log_trace(trace_name, model, system_prompt, query, response, start_time_ms=None, end_time_ms=None, cached=False):
    if cached:
        # For cached responses, start_time_ms and end_time_ms might not be applicable
        start_time_ms = end_time_ms = datetime.datetime.now().timestamp() * 1000
    trace = Trace(
        name=trace_name,
        kind="llm",
        metadata={"model_name": model, "cached": cached},
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        inputs={"system_prompt": system_prompt, "query": query},
        outputs={"response": response},
    )
    trace.log(name="openai_call")

def get_llm_embedding(prompt: str, model: str) -> str:
    openai.api_base = "https://api.openai.com/v1" if model != "vicuna" else VICUNA_URL
    client = OpenAI()
    key = json.dumps([model, prompt])

    cached_value = get_emb_from_cache(key, llm_embed_cache)

    if cached_value is not None:
        logging.debug(f"LLM Cache Hit")
        return cached_value
    else:
        logging.debug(f"LLM Cache Miss")

    for _ in range(3):
        try:
            text = prompt.replace("\n", " ")
            embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
            save_emb_to_cache(key, embedding, llm_embed_cache)
            return embedding
        except Exception as e:
            logging.error(f"LLM Error: {e}")
            continue

    return "LLM Error: Cannot get response."



def prompt_differences(captions1: List[str], captions2: List[str]) -> str:
    caption1_concat = "\n".join(
        [f"Image {i + 1}: {caption}" for i, caption in enumerate(captions1)]
    )
    caption2_concat = "\n".join(
        [f"Image {i + 1}: {caption}" for i, caption in enumerate(captions2)]
    )
    prompt = f"""Here are two groups of images:

Group 1:
```
{caption1_concat}
```

Group 2:
```
{caption2_concat}
```

What are the differences between the two groups of images?
Think carefully and summarize each difference in JSON format, such as:
```
{{"difference": several words, "rationale": group 1... while group 2...}}
```
Output JSON only. Do not include any other information.
"""
    return prompt


def get_differences(captions1: List[str], captions2: List[str], model: str) -> str:
    prompt = prompt_differences(captions1, captions2)
    differences = get_llm_output(prompt, model)
    try:
        differences = json.loads(differences)
    except Exception as e:
        logging.error(f"Difference Error: {e}")
    return differences


def test_get_llm_output():
    prompt = "hello"
    model = "gpt-4"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "gpt-3.5-turbo"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "vicuna"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")


def test_get_llm_output_parallel():
    threads = []

    for _ in range(3):
        thread = threading.Thread(target=test_get_llm_output)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def test_get_differences():
    captions1 = [
        "A cat is sitting on a table",
        "A dog is sitting on a table",
        "A pig is sitting on a table",
    ]
    captions2 = [
        "A cat is sitting on the floor",
        "A dog is sitting on the floor",
        "A pig is sitting on the floor",
    ]
    differences = get_differences(captions1, captions2, "gpt-4")
    print(f"{differences=}")


if __name__ == "__main__":
    test_get_llm_output()
    test_get_llm_output_parallel()
    test_get_differences()