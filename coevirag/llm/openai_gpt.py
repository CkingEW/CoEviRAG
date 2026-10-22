import functools
import hashlib
import json
import os
import sqlite3
import threading
from copy import deepcopy
from typing import List, Tuple

import httpx
import openai
from filelock import FileLock
from openai import OpenAI
from openai import AzureOpenAI
from packaging import version
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)

def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # Cache entries must be specific to every generation parameter. In
        # particular, a short QA completion must not reuse an earlier response
        # produced with a different token budget.
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        effective_params = deepcopy(gen_params)
        effective_params.update(kwargs)
        effective_params.pop("messages", None)

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,
            "generation_params": effective_params,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                # return cached result and mark as hit
                return message, metadata, True

        # if cache miss, call the original function to get the result
        result = func(self, *args, **kwargs)
        message, metadata = result

        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper

def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)  
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)
    return wrapper

class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        config_dict = global_config.__dict__
        config_dict['max_retries'] = global_config.max_retry_attempts
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir, global_config, cache_filename: str = None,
                 high_throughput: bool = True,
                 **kwargs) -> None:

        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url
        self.request_timeout = float(getattr(global_config, "llm_request_timeout", 60.0))

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            client = httpx.Client(limits=limits, timeout=httpx.Timeout(self.request_timeout, read=self.request_timeout))
        else:
            client = None

        self.max_retries = int(getattr(self.global_config, "max_retry_attempts", kwargs.get("max_retries", 1)))

        if self.global_config.azure_endpoint is None:
            pooled_keys = [
                key.strip()
                for key in os.environ.get("OPENAI_API_KEYS", "").split(",")
                if key.strip()
            ]
            if not pooled_keys:
                pooled_keys = [os.environ.get("OPENAI_API_KEY")]
            self.openai_clients = [
                OpenAI(
                    api_key=key,
                    base_url=self.llm_base_url,
                    http_client=client,
                    max_retries=self.max_retries,
                )
                for key in pooled_keys
            ]
            self.openai_client = self.openai_clients[0]
            self._client_index = 0
            self._client_lock = threading.Lock()
            logger.info("Initialized LLM client pool with %d API key(s).", len(self.openai_clients))
        else:
            self.openai_client = AzureOpenAI(api_version=self.global_config.azure_endpoint.split('api-version=')[1],
                                             azure_endpoint=self.global_config.azure_endpoint, max_retries=self.max_retries)
            self.openai_clients = [self.openai_client]
            self._client_index = 0
            self._client_lock = threading.Lock()

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__

        config_dict['llm_name'] = self.global_config.llm_name
        config_dict['llm_base_url'] = self.global_config.llm_base_url
        generate_params = {
                "model": self.global_config.llm_name,
                "max_completion_tokens": config_dict.get("max_new_tokens", 400),
                "n": config_dict.get("num_gen_choices", 1),
                "seed": config_dict.get("seed", 0),
                "temperature": config_dict.get("temperature", 0.0),
            }
        reasoning_effort = str(
            getattr(self.global_config, "llm_reasoning_effort", "low")
        ).strip().lower()
        base_url = str(self.llm_base_url).lower()
        model_name = str(self.llm_name).lower()
        is_gpt_oss = "gpt-oss" in model_name
        if "api.cloudflare.com" in base_url:
            # Workers AI rejects seed=0 even though it is accepted by other
            # OpenAI-compatible providers.
            generate_params["seed"] = max(1, int(generate_params["seed"]))
        if is_gpt_oss and reasoning_effort != "none":
            if "openrouter.ai" in base_url:
                generate_params["extra_body"] = {
                    "reasoning": {"effort": reasoning_effort}
                }
            elif "api.nvidia.com" in base_url:
                generate_params["extra_body"] = {
                    "reasoning_effort": reasoning_effort
                }
            elif "api.groq.com" in base_url:
                generate_params["extra_body"] = {
                    "reasoning_effort": reasoning_effort,
                    "reasoning_format": "hidden",
                }
        elif "api.nvidia.com" in base_url:
            if model_name.startswith(("deepseek-ai/", "moonshotai/")):
                if reasoning_effort != "none":
                    generate_params["extra_body"] = {
                        "reasoning_effort": reasoning_effort
                    }
            elif "nemotron-3-" in model_name:
                generate_params["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": reasoning_effort != "none"
                    }
                }
            elif model_name.startswith("google/gemma-4-"):
                generate_params["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": reasoning_effort != "none"
                    }
                }
        config_dict['generate_params'] = generate_params

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"): # if we use vllm to call openai api or if we use openai but the version is too old to use 'max_completion_tokens' argument
            # TODO strange version change in openai protocol, but our current vllm version not changed yet
            params['max_tokens'] = params.pop('max_completion_tokens')

        with self._client_lock:
            client = self.openai_clients[self._client_index % len(self.openai_clients)]
            self._client_index += 1
        response = client.chat.completions.create(**params)

        response_message = response.choices[0].message.content
        assert isinstance(response_message, str), "response_message should be a string"
        
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens, 
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        return response_message, metadata
