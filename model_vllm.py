# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from config_loader import get_config_path
import math
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor
import threading, time

from loguru import logger
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    wait_random,
)
import openai

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

with open(get_config_path(), "r") as f:
    config = json.load(f)

RANKER_USE_STREAM = config["ranker_use_stream"]
USE_STREAM = config["use_stream"]


class EmptyResponseError(Exception):
    """Exception raised when the model returns an empty response."""
    pass


def _extract_true_false_logprob(logprobs_content, final_response_text=None, is_streaming=False):
    """
    Extract the probability of the first True/False token inside the <relevant> tag from logprobs.
    If there is no <relevant> tag, return 0.

    Logic:
    - If there is a </think> tag, skip all tokens before it
    - Find the first <relevant> tag after </think>
    - Find the first True/False token inside the <relevant> tag
    - If it is True, probs = exp(logprob)
    - If it is False, probs = 1 - exp(logprob)

    Args:
        logprobs_content: the logprobs.content list (non-streaming) or the event stream (streaming)
        final_response_text: the actual response text (used for non-streaming mode)
        is_streaming: whether it is streaming mode

    Returns:
        float: the probability of True (0-1); returns 0 if there is no <relevant> tag or nothing is found
    """
    import re
    
    # Check whether there is a <relevant> tag
    if not final_response_text or not re.search(r"<relevant>", final_response_text, re.IGNORECASE):
        return 0.0
    
    # Check whether there is a </think> tag
    has_think_tag = "</think>" in final_response_text.lower()
    
    # Extract the content after <relevant> from the text to determine the target token
    # If there is a </think>, only look for the <relevant> that comes after it
    if has_think_tag:
        match = re.search(r"</think>.*?<relevant>\s*([^<]+?)\s*</relevant>", final_response_text, re.IGNORECASE | re.DOTALL)
    else:
        match = re.search(r"<relevant>\s*([^<]+?)\s*</relevant>", final_response_text, re.IGNORECASE | re.DOTALL)
    
    if not match:
        return 0.0
    
    relevant_content = match.group(1).strip().lower()
    if "true" in relevant_content:
        target_token = "true"
    elif "false" in relevant_content:
        target_token = "false"
    else:
        return 0.0
    
    # Iterate over logprobs to find the target token inside the first <relevant> after </think>
    passed_think_tag = False
    in_relevant_tag = False
    if is_streaming:
        # Streaming mode
        for event in logprobs_content:
            if not event.choices[0].logprobs or len(event.choices[0].logprobs.content) == 0:
                continue
            
            step = event.choices[0].logprobs.content[0]
            token_text = step.token
            token_lower = token_text.lower()
            
            # If there is a </think> tag, first skip all tokens before it
            if has_think_tag and not passed_think_tag:
                if "</think>" in token_lower:
                    passed_think_tag = True
                continue
            
            # Only start looking for <relevant> after passing </think>
            if passed_think_tag or not has_think_tag:
                # Detect the start of the <relevant> tag
                if not in_relevant_tag:
                    if "<relevant" in token_lower or "relevant" in token_lower:
                        in_relevant_tag = True
                        continue
                else:
                    # Already inside the tag
                    if "</relevant>" in token_lower:
                        break
                    
                    token_clean = token_text.strip().lower()
                    if token_clean == target_token:
                        logprob = getattr(step, "logprob", None)
                        if logprob is not None:
                            prob = math.exp(float(logprob))
                            return prob if target_token == "true" else 1 - prob
    else:
        # Non-streaming mode
        if not logprobs_content:
            return 0.0
        
        for idx, t in enumerate(logprobs_content):
            token_text = t.token
            token_lower = token_text.lower()
            
            # If there is a </think> tag, first skip all tokens before it
            if has_think_tag and not passed_think_tag:
                if "</think>" in token_lower:
                    passed_think_tag = True
                continue
            
            # Only start looking for <relevant> after passing </think>
            if passed_think_tag or not has_think_tag:
                # Detect the start of the <relevant> tag
                if not in_relevant_tag:
                    if "<relevant" in token_lower or "relevant" in token_lower:
                        in_relevant_tag = True
                        continue
                else:
                    # Already inside the tag
                    if "</relevant>" in token_lower:
                        break
                    
                    token_clean = token_text.strip().lower()
                    if token_clean == target_token:
                        logprob = getattr(t, "logprob", None)
                        if logprob is not None:
                            prob = math.exp(float(logprob))
                            return prob if target_token == "true" else 1 - prob
    
    return 0.0


class Agent:
    def _call_llm(
        self,
        prompt,
        max_tokens=4096,
        temperature=0.0,
        top_p=1.0,
        logprobs=False,
        **kwargs,
    ):
        """
        Unified low-level model call interface; all reasoning-related methods call the model through this function.
        Args:
            prompt: the input prompt string
            max_tokens: the maximum number of tokens to generate
            temperature: the sampling temperature
            top_p: top_p sampling
            kwargs: other openai parameters
        Returns:
            the response object returned by the openai interface
        """
        model_name_lower = self.model_name.lower()
        supports_logprobs = logprobs and "deepseek" not in model_name_lower and "kimi" not in model_name_lower
        
        # Choose the correct parameter name based on the model type
        if "gpt-5" in model_name_lower:
            # GPT-5 models use max_completion_tokens and do not support temperature=0
            api_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt.strip()}],
                "max_completion_tokens": max_tokens,
                "top_p": top_p,
                **kwargs,
            }
            if supports_logprobs:
                api_params["logprobs"] = logprobs
                api_params["top_logprobs"] = 1
        else:
            # Other models use max_tokens
            if "qwen3.5" in self.model_name:
                kwargs["max_tokens"] = 32768
                kwargs["temperature"] = 0.7
                kwargs["top_p"] = 0.8
                kwargs["presence_penalty"] = 1.5
                kwargs["extra_body"] = {
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            api_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt.strip()}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": supports_logprobs and ((logprobs and RANKER_USE_STREAM) or USE_STREAM),
                **kwargs,
            }
            # Only add logprobs when the model supports it
            if supports_logprobs:
                api_params["logprobs"] = logprobs
        max_retries = kwargs.get("max_retries_on_empty", 3)  # Retry 3 times by default
        chat_id = None

        try:

            @retry(
                retry=retry_if_exception_type(openai.RateLimitError),
                wait=wait_random(0.5, 16),
                stop=stop_after_attempt(20),
                reraise=True,
            )
            def _call_llm_with_retry():
                t0 = time.time()
                response = self.client.chat.completions.create(**api_params)
                logger.info(
                    f"llm time cost: {time.time() - t0:.3f}s, prompt: {prompt[:100]}"
                )

                return response

            @retry(
                retry=retry_if_exception_type(EmptyResponseError),
                wait=wait_random(0.5, 2),
                stop=stop_after_attempt(max_retries),
                reraise=True,
            )
           
            def _get_response_with_retry():
                nonlocal chat_id
                response = _call_llm_with_retry()
                final_response = None
                probs = 0
                passed_relevant_tag = False
                find_target_token = False
                if supports_logprobs and logprobs and RANKER_USE_STREAM:
                    final_response = ""
                    for event in response:
                        # Record the streamed chat_id when possible, to help locate errors
                        if getattr(event, "id", None) is not None:
                            chat_id = event.id

                        # In a vLLM/OpenAI-style stream, not every chunk carries logprobs,
                        # and logprobs.content may be an empty list; thorough null checks are required to avoid IndexError
                        choices = getattr(event, "choices", None)
                        choice0 = choices[0] if choices else None
                        logprobs_obj = getattr(choice0, "logprobs", None) if choice0 else None
                        logprobs_content = getattr(logprobs_obj, "content", None) if logprobs_obj else None

                        if logprobs_content:
                            step = logprobs_content[0]
                            token_text = getattr(step, "token", "") or ""
                            token_lower = token_text.lower()
                            if "relevant" in token_lower:
                                passed_relevant_tag = True
                            else:
                                if (not find_target_token) and passed_relevant_tag:
                                    if "true" in token_lower or "false" in token_lower:
                                        logprob = getattr(step, "logprob", None)
                                        if logprob is not None:
                                            prob = math.exp(float(logprob))
                                            probs = prob if "true" in token_lower else 1 - prob
                                            find_target_token = True
                                        else:
                                            logger.error(f"❌ [EmptyResponse] Model response is missing logprob")
                                            
                        delta = getattr(choice0, "delta", None) if choice0 else None
                        delta_content = getattr(delta, "content", None) if delta else None
                        if delta_content:
                            final_response += delta_content
                else:
                    # First get message.content (the actual response content)
                    final_response = response.choices[0].message.content if hasattr(response.choices[0], 'message') else None
                    
                    # Only try to extract when the model supports logprobs
                    if supports_logprobs and logprobs == True and final_response:
                        # Extract logprob (skip the content inside the <think> tag)
                        probs = _extract_true_false_logprob(
                            response.choices[0].logprobs.content if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs else [],
                            final_response_text=final_response,
                            is_streaming=False
                        )
                    elif supports_logprobs and logprobs == True:
                        probs = 0
                
                # Check whether final_response is None
                if final_response is None:
                    error_msg = (
                        f"❌ [EmptyResponse] Model returned an empty response\n"
                        f"   Model: {self.model_name}\n"
                        f"   URL: {self.url}\n"
                        f"   Stream mode: {logprobs and RANKER_USE_STREAM}\n"
                        f"   Response object: {type(response)}\n"
                    )
                    if hasattr(response, 'choices') and len(response.choices) > 0:
                        error_msg += f"   Choice object: {response.choices[0]}\n"
                        if hasattr(response.choices[0], 'message'):
                            error_msg += f"   Message object: {response.choices[0].message}\n"
                    logger.error(error_msg)
                    raise EmptyResponseError(error_msg)
                
                return response, final_response, probs
            
            response, final_response, probs = _get_response_with_retry()
                    
        except openai.RateLimitError as e:
            logger.exception(f"❌ ----->Model call failed, rate limit error: {e}")
            raise e
        except Exception as e:
            # Special error handling for local models
            if "localhost" in self.url or "127.0.0.1" in self.url:
                logger.info(f"❌ Local model call failed: {e},url:{self.url}")
                logger.info(f"💡 Please check the local model service status:")
                logger.info(f"   1. Confirm the model service is running")
                logger.info(
                    f"   2. Check whether port {self.url.split(':')[-1].split('/')[0]} is in use"
                )
                logger.info(
                    f"   3. Use the command: netstat -tlnp | grep {self.url.split(':')[-1].split('/')[0]}"
                )
                logger.info(f"   4. Check the model service logs")
                raise e
            else:
                # Remote model error handling
                logger.exception(f"❌ Remote model call failed: {e} ")
                # Note: raising a NameError inside exception handling would mask the original error, so this must be absolutely safe
                logger.error(
                    f"llm chat id: {locals().get('chat_id', None)} ; {e}\n prompt: {prompt}"
                )
                raise e
        if logprobs:
            return final_response, probs
        else:
            return response

    def _call_llm_with_cot(
        self,
        prompt,
        max_tokens=4096,
        temperature=0.0,
        top_p=1.0,
        logprobs=False,
        **kwargs,
    ):
        """
        Chain-of-thought model call interface that lets the model think first and then give an answer.
        Output format: <think>reasoning process...</think><answer>final answer...</answer>
        Args:
            prompt: the input prompt string
            max_tokens: the maximum number of tokens to generate
            temperature: the sampling temperature
            top_p: top_p sampling
            kwargs: other openai parameters
        Returns:
            the parsed answer string (returns only the content inside the <answer> tag)
        """
        # Build the system prompt asking the model to output a chain of thought
        system_prompt = """You are an intelligent assistant. Please answer using the following format:

1. First, think step by step inside the <think> tag: analyze the problem and your reasoning process.
2. Then, provide the final answer inside the <answer> tag.

If the user requests a specific format (e.g., JSON), output it strictly inside the <answer> tag in that format.

Example output format:
<think>
Let me analyze this problem...
First, I should consider...
Then...
</think>
<answer>
Here is the final answer, following the user's requested format.
</answer>"""

        # Build the complete message list
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.strip()},
        ]

        # logger.info(f"🤔 Calling the chain-of-thought model...")
        # logger.info(f"📝 User prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        try:
            # Call the model
            # Choose the correct parameter name based on the model type
            if "gpt-5" in self.model_name.lower():
                # GPT-5 models use max_completion_tokens and do not support temperature=0
                api_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                    "top_p": top_p,
                    logprobs: logprobs,
                    **kwargs,
                }
                # Only add the temperature parameter when temperature is not 0
                if temperature != 0:
                    api_params["temperature"] = temperature
            else:
                # Other models use max_tokens
                api_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "logprobs": logprobs,
                    **kwargs,
                }

            @retry(
                retry=retry_if_exception_type(openai.RateLimitError),
                wait=wait_random(0.5, 16),
                stop=stop_after_attempt(20),
                reraise=True,
            )
            def _call_llm_with_retry():
                t0 = time.time()
                response = self.client.chat.completions.create(**api_params)
                logger.info(
                    f"llm time cost: {time.time() - t0:.3f}s, prompt: {prompt[:100]}"
                )
                return response

            response = _call_llm_with_retry()

            # Get the model's raw response
            raw_response = response.choices[0].message.content
            # logger.info(f"🔍 Model raw response:")
            # logger.info(f"   {raw_response}")

            # Parse the content inside the <answer> tag
            answer = self._extract_answer_from_cot(raw_response)

            if answer:
                logger.info(f"✅ Answer parsed successfully")
                # logger.info(f"   {answer}")
                # Reassign the parsed answer to the response object to keep the interface consistent
                response.choices[0].message.content = answer
                return response
            else:
                logger.info(f"⚠️  Unable to parse answer, returning the raw result")
                return response

        except Exception as e:
            logger.exception(f"❌ Chain-of-thought model call failed")
            # On failure, fall back to a regular call
            logger.info(f"🔄 Falling back to a regular model call...")
            try:
                response = self._call_llm(
                    prompt, max_tokens, temperature, top_p, **kwargs
                )
                return response.choices[0].message.content
            except Exception as fallback_e:
                logger.exception(f"❌ Fallback call also failed")
                return ""

    def _extract_answer_from_cot(self, response_text):
        """
        Extract the content inside the <answer> tag from a chain-of-thought response.
        Args:
            response_text: the full text returned by the model
        Returns:
            the extracted answer string, or None if not found
        """
        try:
            # Look for the <answer> tag
            start_tag = "<answer>"
            end_tag = "</answer>"

            start_idx = response_text.find(start_tag)
            if start_idx == -1:
                logger.info(f"⚠️  Start tag <answer> not found")
                return None

            start_idx += len(start_tag)
            end_idx = response_text.find(end_tag, start_idx)

            if end_idx == -1:
                logger.info(f"⚠️  End tag </answer> not found")
                return None

            # Extract the answer content
            answer = response_text[start_idx:end_idx].strip()

            if not answer:
                logger.info(f"⚠️  Content inside the <answer> tag is empty")
                return None

            return answer

        except Exception as e:
            logger.exception(f"❌ Error while parsing the answer")
            return None

    def __init__(
        self, model_name=None, url=None, key="EMPTY", config_path="config/config.json"
    ):
        """
        Agent initialization
        Args:
            model_name: the model name; if provided, the URL and key are read from the config file
            url: the model URL; ignored if model_name is provided
            key: the API key; ignored if model_name is provided
            config_path: the path to the config file
        """
        # If model_name is provided, read the configuration from the config file
        if model_name and not url:
            config = self._load_model_config(config_path)
            model_configs = config.get("model_config", {})
            if model_name not in model_configs:
                available_models = ", ".join(model_configs.keys())
                raise ValueError(
                    f"Model '{model_name}' is not in the config file. Available models: {available_models}"
                )

            model_config = model_configs[model_name]
            self.model_name = model_config.get("model", model_name)
            self.url = model_config.get("base_url")
            self.key = model_config.get("api_key", "EMPTY")
            self.model_config = model_config  # Save the full configuration

            if not self.url:
                raise ValueError(f"The configuration for model '{model_name}' is missing the 'base_url' field")

            logger.info(f"✅ Loaded model from config file: {model_name}")
            logger.info(f"   Base URL: {self.url}")
            logger.info("   API key configured")

            # Test the local model connection
            if "localhost" in self.url or "127.0.0.1" in self.url:
                self._test_local_model_connection()

        # If no model_name is provided, use the legacy approach (backward compatible)
        else:
            self.model_name = model_name if model_name else "unknown"
            self.url = url
            self.key = key
            self.model_config = None

        # Initialize statistics variables
        self.call_count = 0
        self.total_tokens = 0
        self._usage_lock = threading.Lock()

        # Initialize the OpenAI client, reusing the connection
        self.client = openai.OpenAI(base_url=self.url, api_key=self.key)

    def _test_local_model_connection(self):
        """Test whether the local model connection is working."""
        import requests
        import time

        logger.info(f"🔍 Testing local model connection: {self.url}")

        # Build the test request
        test_url = self.url.replace("/v1", "/v1/models")
        test_data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        }

        try:
            # First test the model list interface
            logger.info(f"   📋 Testing the model list interface...")
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                logger.info(f"   ✅ Model list interface is working")
            else:
                logger.info(f"   ⚠️  Model list interface returned status code: {response.status_code}")

            # Test the chat interface
            logger.info(f"   💬 Testing the chat interface...")
            chat_url = self.url.replace("/v1", "/v1/chat/completions")
            response = requests.post(chat_url, json=test_data, timeout=10)

            if response.status_code == 200:
                logger.info(f"   ✅ Local model connection test succeeded!")
                logger.info(f"   🚀 Model {self.model_name} is ready")
            else:
                logger.info(f"   ⚠️  Chat interface returned status code: {response.status_code}")
                logger.info(f"   📝 Response content: {response.text[:200]}")

        except requests.exceptions.ConnectionError:
            logger.exception(f"   ❌ Connection failed: unable to connect to {self.url}")
            logger.info(f"   💡 Please check:")
            logger.info(f"      1. Whether the model service is running")
            logger.info(f"      2. Whether the port is correct")
            logger.info(f"      3. Firewall settings")
            logger.info(
                f"      4. Use the command: netstat -tlnp | grep {self.url.split(':')[-1].split('/')[0]}"
            )
        except requests.exceptions.Timeout:
            logger.exception(f"   ⏰ Connection timed out: the model took too long to respond")
        except Exception as e:
            logger.exception(f"   ❌ Connection test failed: {e}")

        logger.info(f"   🔍 Local model connection test completed")

    def _load_model_config(self, config_path):
        """Load the model config file."""
        # Handle relative paths
        if not os.path.isabs(config_path):
            config_path = os.path.join(current_dir, config_path)

        try:
            with open(get_config_path(config_path), "r", encoding="utf-8") as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Model config file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Model config file has an invalid format: {e}")

    def infer(self, prompt, sample=False, use_cot=False, logprobs=False):
        try:
            caller = self._call_llm_with_cot if use_cot else self._call_llm

            response = caller(
                prompt,
                max_tokens=4096,
                temperature=2.0 if sample else 0.0,
                top_p=0.8 if sample else 1.0,
                logprobs=logprobs,
            )
            return (
                response.choices[0].message.content
                if hasattr(response, "choices")
                else response
            )
        except Exception as e:
            logger.exception(str(e))
            return ""

    def batch_infer(
        self, prompts, batch_size=20, sample=False, use_cot=False, logprobs=False
    ):
        if len(prompts) == 0:
            return []

        def get_response(prompt):
            caller = self._call_llm_with_cot if use_cot else self._call_llm
            response = caller(
                prompt,
                max_tokens=4096,
                temperature=2.0 if sample else 0.0,
                top_p=0.8 if sample else 1.0,
                logprobs=logprobs,
            )
            return (
                response.choices[0].message.content
                if hasattr(response, "choices")
                else response
            )

        try:
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                responses = list(executor.map(get_response, prompts))
            return responses
        except Exception as e:
            logger.exception(str(e))
            return []

    def batch_infer_safe(
        self, prompts, batch_size=20, sample=False, use_cot=False, logprobs=False
    ):
        """
        Safe batch inference method that ensures the order of returned results exactly matches the order of the input prompts.

        Args:
            prompts: the list of input prompts
            batch_size: the batch size
            sample: whether to use sampling mode
            use_cot: whether to enable chain-of-thought calls

        Returns:
            responses: the list of responses returned in order
        """
        if len(prompts) == 0:
            return []

        logger.info(f"🔒 [SafeBatchInfer] Starting safe batch inference for {len(prompts)} prompts")
        logger.info(f"🔒 [SafeBatchInfer] Batch size: {batch_size}")

        # Add an index identifier to each prompt
        indexed_prompts = [(i, prompt) for i, prompt in enumerate(prompts)]

        def get_response_with_index(indexed_item):
            index, prompt = indexed_item
            try:
                caller = self._call_llm_with_cot if use_cot else self._call_llm
                response = caller(
                    prompt,
                    max_tokens=4096,
                    temperature=2.0 if sample else 0.0,
                    top_p=0.8 if sample else 1.0,
                    logprobs=logprobs,
                )
                if logprobs:
                    return index, (response[0], response[1])
                return index, (
                    response.choices[0].message.content
                    if hasattr(response, "choices")
                    else response
                )
            except Exception as e:
                logger.exception(f"❌ [SafeBatchInfer] prompt {index+1} processing failed: {e}")
                if logprobs:
                    return index, ("", 0)
                return index, ""

        try:
            # Use ThreadPoolExecutor for concurrent processing while keeping index information
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                indexed_responses = list(
                    executor.map(get_response_with_index, indexed_prompts)
                )

            # Re-sort by the original index to ensure the correct order
            indexed_responses.sort(key=lambda x: x[0])
            responses = [response for index, response in indexed_responses]

            logger.info(
                f"✅ [SafeBatchInfer] Safe batch inference completed, returning {len(responses)} responses"
            )
            logger.info(f"🔒 [SafeBatchInfer] Verified that the response order matches the input order")

            return responses

        except Exception as e:
            logger.exception(f"❌ [SafeBatchInfer] Batch inference failed: {e}")
            return []

    def get_usage_stats(self):
        avg_token = self.total_tokens / self.call_count if self.call_count > 0 else 0
        return {
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "avg_token": avg_token,
        }

    def close(self):
        """Close the underlying client connection."""
        if hasattr(self, "client"):
            self.client.close()


if __name__ == "__main__":
    print("🧪 Testing the Agent model call functionality")

    # Test creating an Agent using a model name
    try:
        print("\n=== Test 1: Create an Agent using a model name ===")
        agent = Agent(model_name="qwen3.5-35b")
        print(f"✅ Agent created successfully: {agent.model_name}")
        print(f"📊 URL: {agent.url}")
        user_query = "Find research papers about training an LLM agent and involving websailor as a baseline."
        title = "WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization"
        abstract = "The advent of Large Language Model (LLM)-powered agents has revolutionized artificial intelligence by enabling solutions to complex, open-ended tasks through web-based information-seeking (IS) capabilities. The scarcity of high-quality training data has limited the development of IS agents. Existing approaches typically adopt an information-driven paradigm that first collects web data and then generates questions based on the retrieval. However, this may lead to inconsistency between information structure and reasoning structure, question and answer. To mitigate, we propose a formalization-driven IS data synthesis framework WebShaper to construct a dataset. WebShaper systematically formalizes IS tasks through set theory. Central to the formalization is the concept of Knowledge Projections (KP), which enables precise control over reasoning structure by KP operation compositions. During synthesis, we begin by creating seed tasks, then use a multi-step expansion process. At each step, an agentic Expander expands the current formal question more complex with retrieval and validation tools based on our formalization. We train our model on the synthesized dataset. Experiment results demonstrate that WebShaper achieves state-of-the-art performance among open-sourced IS agents on GAIA and WebWalkerQA benchmarks.[score=0.8406596779823303] ### 3.4 Agent Training\n\nTo train our information-seeking agent, similar to WebDancer (Wu et al., 2025a ) , we implement supervised fine-tuning (SFT) followed by reinforcement learning (RL).\n\n---\n\n[score=0.8127291202545166] Thus, constructing training data through a carefully designed agent pipeline becomes the cornerstone of effective IS agent development. !\n\n---\n\n[score=0.8098067045211792] ## 1 Introduction\n\nThe emergence of Large Language Model (LLM)-powered language agents has marked a paradigm-shifting advance in artificial intelligence, enabling transformative solutions to previously intractable challenges across domains (Guo et al., 2024 ; Wang et al., 2024 ; AutoGPT, 2023 ; Wu et al., 2023 ; Ye et al., 2023 ) . Information-seeking (IS) represents a core component of the cognitive autonomy of language agents. This capability not only underpins their adaptability in open-ended tasks but also powers a range of powerful commercial systems such as Deep Research of OpenAI (OpenAI, 2025 ) , Gemini (Gemini, 2025 ) , and Perplexity (Perplexity, 2025 ) . Current agentic systems for unlocking this capability typically follow a well-established pipeline in agent development: (1) First, construct task-specific trajectories of question-answer pairs; (2) Employ supervised fine-tuning (SFT) to acquire foundational skills (Sun et al., 2025 ) . (3) Generalize strategic decision-making through on-policy reinforcement learning (RL) (Jin et al., 2025 ) . The entire development of the IS agent originates from and its ultimate effectiveness depends on high-quality IS task training data. However, due to its complexity, such a high-quality dataset is both sparse and difficult to construct through crowdsourcing. Thus, constructing training data through a carefully designed agent pipeline becomes the cornerstone of effective IS agent development.\n\n====\n\n[score=0.807582676410675] ### 3.4 Agent Training\n\nTo train our information-seeking agent, similar to WebDancer (Wu et al., 2025a ) , we implement supervised fine-tuning (SFT) followed by reinforcement learning (RL). In SFT, given a trajectory in a sequence of tokens ${\\mathcal{T}}=(\\tau_{1},\\alpha_{1},o_{1},...,\\tau_{n},\\alpha_{n},o_{n})$ , we mask out loss from observation leading to loss:\n\n$$\nL=-\\frac{1}{\\sum_{i=1}^{|\\mathcal{{\\mathcal{T}}}|}\\mathbb{I}[x_{i}\\in o]}\\sum_{i=1}^{|{\\mathcal{T}}|}\\mathbb{I}[x_{i}\\in o]\\cdot\\log\\pi_{\\theta}(x_{i}\\mid x_{<i})\n$$ (12)\n\nwhere $\\pi_{\\theta}$ is the model to train. Later in RL, we further optimize $\\pi_{\\theta}$ use the GRPO algorithm (Shao et al., 2024 ) . For a question-answer pair $(q,a)$ , GRPO samples rollouts $\\{y_{i}\\}_{i}^{|G|}$ and updates the policy model by:\n\nwhere $context$ includes all the model completions and tool responses. $\\varepsilon$ is the clipping range of the importance sampling ratio $r_{i,t}(\\theta)$ . $\\hat{A}_{i,t}$ is an estimator of the advantage of the $i$ -th rollout at $t$ -th step. **Table 1: Main results on GAIA and WebWalkerQA benchmarks. We compare WebShaper with several cutting-edge baselines methods. bolded number stands for the best results on the corresponding settings. Blue scores are the highest among all open-sourced methods.\n\n---\n\n[score=0.8068673610687256] • We propose an agentic Expander that iteratively generates and validates questions in alignment with the formalization. • We conduct extensive experiments across multiple benchmarks to evaluate the effectiveness of WebShaper. Empirical results demonstrate that models trained with WebShaper consistently outperform baselines, confirming the value of our formalization and synthesis approach.\n\n---\n\n[score=0.8010835647583008] Further discussions demonstrate the validity of each module of our method. We summarize our contributions as:\n\n• We introduce WebShaper, a formalization-driven data synthesis method for information-seeking agents, grounded in our proposed task formalization. Leveraging this method, we construct the WebShaper dataset, which enables systematic generation of IS instances. • We propose an agentic Expander that iteratively generates and validates questions in alignment with the formalization."
        constraints = [
            "(1) The paper must be about training an LLM agent",
            "(2) The paper must involve WebSailor as a baseline",
        ]
        meta_info = ""
        # Test simple questions
        simple_questions = [
            # f"You are an academic expert and constraint extractor. You need to carefully analyze and understand the user's question.\n\nYour job has two parts:\n\n(1) Search-query decomposition and generation:\n Given a user question, Step 1. Identify literature-oriented concepts From the user query, extract: (A) Core concepts: - Fundamental research topics or entities that MUST appear in the target papers - Prefer terms that would likely appear in paper titles or abstracts (B) Synonyms / aliases: - Alternative technical terms used in the literature for the same concept - Include abbreviations and variant phrasings if commonly used (C) Optional qualifiers: - Methods, settings, time ranges, application domains, or constraints - These should be optional and used only to refine search Step 2. Construct Google Scholar queries - Generate 3–5 concise search queries - Each query should: - Be suitable for Google Scholar(combined by different keywords) - Use quotation marks ONLY for multi-word technical phrases - Contain no natural language sentences - Prefer AND-style keyword combination over long phrases - Target a *class of relevant papers*, not a single specific paper. - NEVER include Exclusion Constraints or negative terms. \n\n(2) From the same question, decompose it into 1–5 'checkpoints' (constraints) that will be used later to judge whether a paper is relevant. Each checkpoint must cover a distinct requirement from the query. Step:1. Keyword extraction 2.Checklist construction\n- Combine concepts when they jointly define a single requirement.\n- Keep task requirements as a separate checkpoint. Note: Each keyword should be highlighted in the checklist. For example: My query is \"I'm looking for papers on using multimodal LLM agents to diagnose medical images, rather than traditional CNN-based medical image classifiers.\n\nKeyword extraction:\n1. multimodal LLM agents\n2. diagnose medical images\n3. CNN-based medical image classifiers\n\nChecklist construction:\n(1) use multimodal LLM agents to perform medical image diagnosis (not CNN-based classifiers)\n(2) the primary task must be diagnosing medical images (not general vision tasks or unrelated clinical tasks)\" \n\nResearch Question: {user_query}\n\nYou must only return the queries and checkpoints, and nothing else (no explanations, no reasoning, no additional text). \n\nReturn in exactly the following format:\n<keywords>keyword1,keyword2...</keywords> <checkpoints> checkpoint 1;checkpoint 2;checkpoint 3 </checkpoints> <query> search query 1;search query 2;search query 3 </query>\n\nAll search queries and checkpoints must be in English. You need to reflect on whether each checkpoint was requested in the original query. Do not add extra assumptions, capabilities, behaviors, or tasks that the query does not mention."
            f"You are a strict relevance evaluator for paper retrieval. Evaluate the relevance between the provided paper and the overall user query. Be a nitpicky critic: If the query has multiple constraints, extract them and test each one. Make sure you fully understand the intended meaning of each evaluation constraint, and carefully check whether the critic paper satisfies each of those meanings. \n\nPaper Title: {title}\nPaper Abstract: {abstract}\n\nUser Query Context: {user_query}\n\nThe metainfo you can refer to is: {meta_info}. Instructions:\n1)You have provided the final conclusion; Did you determine that the literature was relevant or irrelevant and why you think so 2) The constraints is {constraints}. For each constraint Ci, give a relavant score for it. Per-constraint scoring rubric (integer 1-5):\n5 = Fully satisfies + explicit paper evidence.\n4 = core keywords satisfies + explicit evidence for core aspects.\n3 = Partially satisfies + some evidence; OR plausible but weakly supported (must mark INFERENCE).\n2 = Insufficient information / unknown: the paper does not provide evidence that it satisfies the constraint, and there is also no clear evidence that it violates the constraint.\n1 = Clearly violates the constraint: there is explicit evidence that the paper does NOT satisfy the constraint. **Ensure your reasoning, evidence and status are logically consistent.**\n3) List irrelevancies / red flags (divergent focus, wrong task/domain/dataset, missing constraints).\n 4)If the average score across all constraint conditions is ≥ 3, then relevant is True; otherwise it is False.\n\nRequired Output Format (STRICT, output ONLY these tags; no extra text):\n<relevant> True/False </relevant>\n<reason>\n- <bullet 1>\n- <bullet 2>\n</reason>\n<constraint> C1:<constraint content>,Score:<1-5>,Evidence:\"<paper snippet>\" vs \"<constraint content>\",reason; C2:<constraint content>,Score:<1-5>,Evidence:\"<paper snippet>\" vs \"<constraint content>\",reason </constraint>\n<irrelevant>\n- <point 1>\n- <point 2>\n</irrelevant>\nUse ';' between different constraints inside <constraint>. Return ONLY the Output, with no preface or follow-up text. All the output should be limited in 300 words and keep the output complete. Your score should be serious, expecially when you give 5 or 4, you must make sure there have exact evidence in paper"

        ]

        # print("\n=== Test 2: Simple question answering ===")
        for i, question in enumerate(simple_questions, 1):
            print(f"\nQuestion {i}: {question}")
            try:
                answer = agent.batch_infer_safe([question], logprobs=True)
                print(f"Answer: {answer}")
            except Exception as e:
                print(f"❌ Answer failed: {e}")

        print("\n=== Test 3: Batch inference ===")
        # try:
        #     answers = agent.batch_infer(simple_questions * 40, batch_size=40)
        #     # for q, a in zip(simple_questions[:2], answers):
        #     #     print(f"Q: {q}")
        #     #     print(f"A: {a[:80]}..." if len(a) > 80 else f"A: {a}")
        #     #     print()
        # except Exception as e:
        #     print(f"❌ Batch inference failed: {e}")

        # Show usage statistics
        # stats = agent.get_usage_stats()
        # print(
        #     f"\n📊 Call stats: {stats['call_count']} total calls, {stats['total_tokens']} total tokens, avg {stats['avg_token']:.1f} tokens/call"
        # )

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "agent" in locals():
            agent.close()
