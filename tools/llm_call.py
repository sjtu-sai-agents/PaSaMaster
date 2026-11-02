# Call the model for answers via openai
from config_loader import get_config_path
import json

import openai
import sys,os
import math
current_path = os.path.dirname(os.path.realpath(__file__))
with open(get_config_path(), 'r') as f:
    model_config = json.load(f)['model_config']
from openai import AsyncOpenAI
import asyncio
from typing import Any, AsyncGenerator, Dict, List, Tuple, Union
from litellm import acompletion
import time

async def llm_call_async(query: str, model_name: str = "glm") -> AsyncGenerator[str, None]:
    """
    Async streaming LLM call:
    - Returns both the model's reasoning process (reasoning_content) and the final answer (content)
    - Since the prompt already stipulates that "all concatenation happens inside <think>",
      here we simply stream reasoning_content as-is without any extra wrapping.
    """
    cfg = model_config[model_name]
    client = AsyncOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
    )

    # If some models require explicitly enabling thinking, configure a "thinking" field in cfg
    # e.g. cfg["thinking"] = {"type": "enabled"}
    extra_params = {}
    thinking_cfg = cfg.get("thinking")
    if thinking_cfg:
        extra_params["thinking"] = thinking_cfg

    stream = await client.chat.completions.create(
        model=model_config[model_name].get("model", model_name),
        messages=[{"role": "user", "content": query}],
        timeout=60,
        stream=True,
        **extra_params,
    )

    async for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta

        # 1. First try to get the reasoning process (reasoning_content)
        reasoning_delta = getattr(delta, "reasoning_content", None)
        # Handle dict structures for compatibility (some third-party SDKs return dict)
        if reasoning_delta is None and isinstance(delta, dict):
            reasoning_delta = delta.get("reasoning_content")

        if reasoning_delta:
            # Emit the model's reasoning process as-is here
            # Control the <think>...</think> wrapping yourself in reasoner_prompt
            yield reasoning_delta

        # 2. Then get the normal answer content
        content_delta = getattr(delta, "content", None)
        if content_delta is None and isinstance(delta, dict):
            content_delta = delta.get("content")

        if content_delta:
            yield content_delta
            
            
async def llm_call_flash(
    messages,
    config,
    model: str = "litellm_proxy/gemini-2.5-flash",
):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    start = time.time()
    thinking = {"type": "disabled", "budget_tokens": 0}

    stream = await acompletion(
        model=config[model].get("model", model),
        messages=messages,
        api_base=config[model]["base_url"],
        api_key=config[model]["api_key"],
        stream=True,
        thinking=thinking,
        stream_options={"include_usage": True},
    )

    first_token = True
    full_text = ""

    async for chunk in stream:
        choice = chunk.choices[0]

        # ✅ Only streaming is considered here: use delta directly
        delta = choice.delta
        text = getattr(delta, "content", None)
        reasoning = getattr(delta, "reasoning_content", None)

        if (text or reasoning) and first_token:
            print(f"⏱ First token latency: {time.time() - start:.3f} s")
            first_token = False

        if text:
            full_text += text

    return full_text


async def llm_call_tool(query: str, model_name: str = "qwen2.5-72b-instruct", max_retries: int = 3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            if 'gemini' in model_name:
                content = await llm_call_flash(query, model_config, model_name)
                return content
            else:
                config = model_config
                client = AsyncOpenAI(
                    api_key=config[model_name]["api_key"],
                    base_url=config[model_name]["base_url"],
                )
                kwargs = {}
                if "qwen3.5" in model_name:
                    kwargs["max_tokens"] = 32768
                    kwargs["temperature"] = 0.7
                    kwargs["top_p"] = 0.8
                    kwargs["presence_penalty"] = 1.5
                    kwargs["extra_body"] = {
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                response = await client.chat.completions.create(
                    model=model_config[model_name].get("model", model_name),
                    messages=[{"role": "user", "content": query}],
                    timeout=60,
                    **kwargs,
                )
                return response.choices[0].message.content
        except Exception as e:
            retry_count += 1
            print(f"Attempt {retry_count} failed: {e}")
            if retry_count == max_retries:
                raise RuntimeError(f"llm_call_async failed after {max_retries} retries.") from e
            await asyncio.sleep(1)
    return None


def llm_call(
    query: str,
    model_name: str = "qwen3-30b",
    return_token_probs: bool = False,
    top_logprobs: int = 1,
    timeout: int = 30,
) -> Union[str, Tuple[str, List[Dict[str, Any]]]]:
    client = openai.OpenAI(api_key=model_config[model_name]["api_key"], base_url=model_config[model_name]["base_url"])
    kwargs: Dict[str, Any] = {}
    if return_token_probs:
        # OpenAI-compatible backends may return:
        # response.choices[0].logprobs.content = [{token, logprob, top_logprobs?}, ...]
        kwargs["logprobs"] = True
        if isinstance(top_logprobs, int) and top_logprobs > 0:
            kwargs["top_logprobs"] = top_logprobs

    response = client.chat.completions.create(
        model=model_config[model_name].get("model", model_name),
        messages=[{"role": "user", "content": query}],
        timeout=timeout,
        **kwargs,
    )

    content = response.choices[0].message.content
    if not return_token_probs:
        return content

    token_probs: List[Dict[str, Any]] = []
    choice0 = response.choices[0]
    logprobs_obj = getattr(choice0, "logprobs", None)
    content_logprobs = None
    if logprobs_obj is not None:
        content_logprobs = getattr(logprobs_obj, "content", None)
        if content_logprobs is None and isinstance(logprobs_obj, dict):
            content_logprobs = logprobs_obj.get("content")

    if content_logprobs:
        for t in content_logprobs:
            token = getattr(t, "token", None) if not isinstance(t, dict) else t.get("token")
            logprob = getattr(t, "logprob", None) if not isinstance(t, dict) else t.get("logprob")
            top_lp = getattr(t, "top_logprobs", None) if not isinstance(t, dict) else t.get("top_logprobs")
            prob = None
            if isinstance(logprob, (int, float)):
                try:
                    prob = math.exp(float(logprob))
                except Exception:
                    prob = None
            token_probs.append(
                {
                    "token": token,
                    "prob": prob,
                    # "top_logprobs": top_lp,
                }
            )

    return content, token_probs

def get_title_from_reference(reference: str):
    query = f"Get the title from the following reference: {reference}, and return the title in a json format:{{'title': 'title'}}.For example, if the reference is '1. [1] Smith, J. (2021). A study on the effects of AI on society. Journal of AI Research, 10(2), 1-10.', you should return {{'title': 'A study on the effects of AI on society'}}."
    # try:
    #     response = llm_call(query, model_name="gpt-4.1-nano-2025-04-14")
    # except:
    #     response = llm_call(query, model_name="qwen-max")
    response = llm_call(query, model_name="qwen-72b")
    json_begin = response.find("{")
    json_end = response.rfind("}") + 1
    json_str = response[json_begin:json_end].replace("'", '"')
    try:
        result = json.loads(json_str)
        return result["title"]
    except:
        return ""

def conclude_abstract(abstract: str,question: str):
    query = f"Please conclude the abstract of the following paper according to the question: {question}, and return the conclusion in a json format:{{'conclusion': 'conclusion'}}. Your answer should be concise and to the point, and should not be too long. You can conclude it in different aspects, such as the problem, the solution, the result, the limitation, etc. Abstract: {abstract}."
    
    try:
        response = llm_call(query, model_name="qwen-72b")
    except:
        response = llm_call(query, model_name="gpt-4.1-nano-2025-04-14")
    json_begin = response.find("{")
    json_end = response.rfind("}") + 1
    json_str = response[json_begin:json_end].replace("'", '"')
    try:
        result = json.loads(json_str)
        return result["conclusion"]
    except:
        return ""

def llm_call_demo(
    query: str,
    model_name: str = "qwen3-30b",
    timeout: int = 30,
) -> Union[str, Tuple[str, List[Dict[str, Any]]]]:
    client = openai.OpenAI(api_key=model_config[model_name]["api_key"], base_url=model_config[model_name]["base_url"])
    response = client.chat.completions.create(
        model=model_config[model_name].get("model", model_name),
        messages=[{"role": "user", "content": query}],
    )

    content = response.choices[0].message.content
    print(content)

async def async_main():
    import time
    query = "hello world"
    begin = time.time()
    num = 1
    while True:
        # Use asyncio.gather to run multiple async calls concurrently
        tasks = [llm_call_tool(query, model_name="qwen-30b-train") for _ in range(num)]
        results = await asyncio.gather(*tasks)
        print(results[0])
        break
    end = time.time()
    print(f"Time taken: {end - begin} seconds")

if __name__ == "__main__":
    asyncio.run(async_main())
    # user_query = "Our team’s code helper looks great when we test it on tiny, self‑contained snippets with simple inputs, but it struggles when we ask it to close real tickets that touch multiple files and need the project to be run to confirm the fix. We want a fair way to judge whether it’s actually improving at these day‑to‑day fixes, not just the toy checks. What kind of test setup should we build so it can try changes, run checks safely, and report results, and how do we compare its progress on these tougher tasks against the easier ones we currently use?"
    # query = f"You are an academic expert, and you need to carefully analyze and understand the user's question. Upon seeing this question, identify the core keywords for academic paper retrieval, providing 3-7 professional terms along with their respective weights. Ensure the keywords are as much as possible **academic, conceptual terms** rather than general words. The more closely the keyword is related to the user’s query, the higher the weight, which should be between 0 and 1. Based on this, generate search queries for literature that will help solve the user's problem, ensuring the queries are concise and focused. **Research Question**: {user_query}\n\nReturn in strict JSON format: \n{{\n  \"keywords\": [\"keyword1\", \"keyword2\", \"keyword3\"],\n  \"weights\": [0.9, 0.7, 0.5],\n  \"queries\": [\"search query 1\", \"search query 2\", \"search query 3\"]\n}}.\nNote that the order of keywords and weights should correspond one-to-one."
    # print(llm_call(query, model_name="glm"))
    # abstract = """ 'The demand for synthetic data in mathematical reasoning has increased due to its potential to enhance the mathematical capabilities of large language models (LLMs). However, ensuring the validity of intermediate reasoning steps remains a significant challenge, affecting data quality. While formal verification via theorem provers effectively validates LLM reasoning, the autoformalisation of mathematical proofs remains error-prone. In response, we introduce iterative autoformalisation, an approach that iteratively refines theorem prover formalisation to mitigate errors, thereby increasing the execution rate on the Lean prover from 60% to 87%. Building upon that, we introduce Theorem Prover as a Judge (TP-as-a-Judge), a method that employs theorem prover formalisation to rigorously assess LLM intermediate reasoning, effectively integrating autoformalisation with synthetic data generation. Finally, we present Reinforcement Learning from Theorem Prover Feedback (RLTPF), a framework that replaces human annotation with theorem prover feedback in Reinforcement Learning from Human Feedback (RLHF). Across multiple LLMs, applying TP-as-a-Judge and RLTPF improves benchmarks with only 3,508 samples, achieving 5.56% accuracy gain on Mistral-7B for MultiArith, 6.00% on Llama-2-7B for SVAMP, and 3.55% on Llama-3.1-8B for AQUA.'"""
    # query = "LLM-as-judge synthetic data scoring"
    # print(conclude_abstract(abstract, query))
    # import time
    # begin = time.time()
    # print(llm_call_demo("Hello, world", model_name="qwen-72b"))
    # end = time.time()
    # print(f"Time taken: {end - begin} seconds")
