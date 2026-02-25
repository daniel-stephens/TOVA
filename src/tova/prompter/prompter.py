import hashlib
import logging
import os
import pathlib

from typing import List, Union

from dotenv import load_dotenv
from joblib import Memory # type: ignore
import requests

from ollama import Client # type: ignore
from openai import OpenAI # type: ignore

from tova.utils.common import (
    load_yaml_config_file,
    init_logger)

memory = Memory(location='cache', verbose=0)

def hash_input(*args):
    return hashlib.md5(str(args).encode()).hexdigest()

class Prompter:
    def __init__(
        self,
        model_type: str,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path("config/config.yaml"),
        temperature: float = None,
        seed: int = None,
        max_tokens: int = None,
        ollama_host: str = None,
        custom_endpoint: str = None,
        custom_api_key: str = None,
        provider: str = None,
    ):
        self._logger = logger if logger else init_logger(config_path, __name__)
        self.config = load_yaml_config_file(config_path, "llm", logger)

        self.GPT_MODELS = self.config.get(
            "gpt", {}).get("available_models", {})
        self.OLLAMA_MODELS = self.config.get(
            "ollama", {}).get("available_models", {})

        self.model_type = model_type
        self.context = None
        self.params = self.config.get("parameters", {})
        self._ollama_host = (ollama_host or "").strip() or None
        self._custom_endpoint = (custom_endpoint or "").strip() or None
        self._custom_api_key = (custom_api_key or "").strip() or None
        self._provider = (provider or "").strip().lower() or None
        
        # We can override the temperature and seed from the config file if given as arguments
        if temperature is not None:
            self.params["temperature"] = temperature
            self._logger.info(f"Setting temperature to: {temperature}")
        if seed is not None:
            self.params["seed"] = seed
            self._logger.info(f"Setting seed to: {seed}")
        if max_tokens is not None:
            # set max_tokens only if provided by the user; otherwise the default values are used
            
            # for gpt models, the parameter is 'max_completion_tokens'
            if model_type in self.GPT_MODELS:
                self.params["max_completion_tokens"] = max_tokens
                self._logger.info(f"Setting max_completion_tokens to: {max_tokens}")
            # for ollama models, the parameter is 'num_predict'
            # https://github.com/ollama/ollama/blob/main/docs/modelfile.md
            elif model_type in self.OLLAMA_MODELS:
                self.params["num_predict"] = max_tokens
                self._logger.info(f"Setting num_predict to: {max_tokens}")
            else:
                self.params["max_tokens"] = max_tokens

        # Explicit provider (e.g. from training UI) takes precedence over inferring from model_type
        if self._provider in ("openai", "gpt"):
            load_dotenv(self.config.get("gpt", {}).get("path_api_key", ".env"))
            self.backend = "openai"
            self._logger.info(f"Using OpenAI API (provider={self._provider}): model={model_type}")
        elif self._provider == "ollama":
            host = self._ollama_host or self.config.get("ollama", {}).get("host", "http://localhost:11434")
            self._ollama_host = host
            os.environ['OLLAMA_HOST'] = host
            self.backend = "ollama"
            Prompter.ollama_client = Client(host=host, headers={'x-some-header': 'some-value'})
            self._logger.info(f"Using OLLAMA API (provider=ollama): host={host}, model={model_type}")
        elif self._provider == "llama_cpp":
            self.llama_cpp_host = self._ollama_host or self.config.get("llama_cpp", {}).get(
                "host", "http://kumo01:11435/v1/chat/completions"
            )
            self.backend = "llama_cpp"
            self._logger.info(f"Using llama_cpp API (provider=llama_cpp): host={self.llama_cpp_host}")
        elif self._provider in ("rchat", "custom") and self._custom_endpoint:
            base_url = self._custom_endpoint.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = base_url + "/v1"
            self.backend = "custom"
            self._custom_base_url = base_url
            self._logger.info(f"Using custom API (provider={self._provider}): {base_url}, model={model_type}")
        # Fall back to host/endpoint and then model_type list membership
        elif self._custom_endpoint:
            base_url = self._custom_endpoint.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = base_url + "/v1"
            self.backend = "custom"
            self._custom_base_url = base_url
            self._logger.info(f"Using custom API (user preference): {base_url}, model={model_type}")
        elif self._ollama_host:
            os.environ['OLLAMA_HOST'] = self._ollama_host
            self.backend = "ollama"
            Prompter.ollama_client = Client(host=self._ollama_host, headers={'x-some-header': 'some-value'})
            self._logger.info(f"Using OLLAMA API (user preference): host={self._ollama_host}")
        elif model_type in self.GPT_MODELS:
            load_dotenv(self.config.get("gpt", {}).get("path_api_key", ".env"))
            self.backend = "openai"
            self._logger.info(f"Using OpenAI API with model: {model_type}")
        elif model_type in self.OLLAMA_MODELS:
            ollama_host = self.config.get("ollama", {}).get(
                "host", "http://kumo01.tsc.uc3m.es:11434"
            )
            os.environ['OLLAMA_HOST'] = ollama_host
            self.backend = "ollama"
            Prompter.ollama_client = Client(
                host=ollama_host,
                headers={'x-some-header': 'some-value'}
            )
            self._logger.info(f"Using OLLAMA API with host: {ollama_host}")
        elif model_type == "llama_cpp":
            self.llama_cpp_host = self.config.get("llama_cpp", {}).get(
                "host", "http://kumo01:11435/v1/chat/completions"
            )
            self.backend = "llama_cpp"
            self._logger.info(
                f"Using llama_cpp API with host: {self.llama_cpp_host}"
            )
        else:
            raise ValueError(
                f"Unsupported model_type: {model_type}. Set provider in config or use custom_endpoint."
            )
        if self.backend == "ollama" and "num_predict" not in self.params and "max_tokens" in self.params:
            self.params["num_predict"] = self.params["max_tokens"]

    @staticmethod
    @memory.cache
    def _cached_prompt_impl(
        template: str,
        question: str,
        model_type: str,
        backend: str,
        params: tuple,
        context=None,
        use_context: bool = False,
        custom_base_url: str = None,
        custom_api_key: str = None,
    ) -> dict:
        """Caching setup."""
        print("Cache miss: computing results...")
        p = dict(params)
        if backend == "openai":
            result, logprobs = Prompter._call_openai_api(
                template=template, question=question, model_type=model_type, params=p
            )
        elif backend == "ollama":
            result, logprobs, context = Prompter._call_ollama_api(
                template=template, question=question, model_type=model_type,
                params=p, context=context,
            )
        elif backend == "llama_cpp":
            result, logprobs = Prompter._call_llama_cpp_api(
                template=template, question=question, params=p,
            )
        elif backend == "custom":
            result, logprobs = Prompter._call_custom_api(
                template=template, question=question, model_type=model_type,
                params=p, base_url=custom_base_url, api_key=custom_api_key,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return {
            "inputs": {
                "template": template,
                "question": question,
                "model_type": model_type,
                "backend": backend,
                "params": p,
                "context": context if use_context else None,
                "use_context": use_context,
            },
            "outputs": {
                "result": result,
                "logprobs": logprobs,
            },
        }

    @staticmethod
    def _call_openai_api(template, question, model_type, params):
        """Handles the OpenAI API call."""

        if template is not None:
            messages = [
                {"role": "system", "content": template},
                {"role": "user", "content": question},
            ]
        else:
            messages = [
                {"role": "user", "content": question},
            ]

        open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = open_ai_client.chat.completions.create(
            model=model_type,
            messages=messages,
            stream=False,
            temperature=params["temperature"],
            max_tokens=params.get("max_tokens", 1000),
            seed=params.get("seed", 1234),
            logprobs=True,
            top_logprobs=10,
        )
        result = response.choices[0].message.content
        logprobs = response.choices[0].logprobs.content
        return result, logprobs

    @staticmethod
    def _call_ollama_api(template, question, model_type, params, context):
        """Handles the OLLAMA API call."""

        if Prompter.ollama_client is None:
            raise ValueError("OLLAMA client is not initialized. Check the model type configuration.")

        if template is not None:
            response = Prompter.ollama_client.generate(
                system=template,
                prompt=question,
                model=model_type,
                stream=False,
                options=params,
                context=context,
            )
        else:
            response = Prompter.ollama_client.generate(
                prompt=question,
                model=model_type,
                stream=False,
                options=params,
                context=context,
            )
        result = response["response"]
        logprobs = None
        context = response.get("context", None)
        return result, logprobs, context

    @staticmethod
    def _call_llama_cpp_api(template, question, params, llama_cpp_host="http://kumo01:11435/v1/chat/completions"):
        """Handles the llama_cpp API call."""
        payload = {
            "messages": [
                {"role": "system", "content": template},
                {"role": "user", "content": question},
            ],
            "temperature": params.get("temperature", 0),
            "max_tokens": params.get("max_tokens", 100),
            "logprobs": 1,
            "n_probs": 1,
        }
        response = requests.post(llama_cpp_host, json=payload)
        response_data = response.json()

        if response.status_code == 200:
            result = response_data["choices"][0]["message"]["content"]
            logprobs = response_data.get("completion_probabilities", [])
        else:
            raise RuntimeError(f"llama_cpp API error: {response_data.get('error', 'Unknown error')}")

        return result, logprobs

    @staticmethod
    def _call_custom_api(template, question, model_type, params, base_url, api_key):
        """OpenAI-compatible API (user preference: custom endpoint + api_key)."""
        if not base_url:
            raise ValueError("Custom backend requires base_url.")
        client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        messages = (
            [{"role": "system", "content": template}, {"role": "user", "content": question}]
            if template else [{"role": "user", "content": question}]
        )
        kwargs = {
            "model": model_type, "messages": messages, "stream": False,
            "temperature": params.get("temperature", 0),
            "max_tokens": params.get("max_tokens", 1024),
        }
        if "seed" in params:
            kwargs["seed"] = params["seed"]
        r = client.chat.completions.create(**kwargs)
        result = r.choices[0].message.content
        logprobs = getattr(r.choices[0].message, "logprobs", None)
        return result, logprobs

    def prompt(
        self,
        system_prompt_template_path: str,
        question: str,
        use_context: bool = False,
        temperature: float = None,
    ) -> Union[str, List[str]]:
        """Public method to execute a prompt given a system prompt template and a question."""
        system_prompt_template = None
        if system_prompt_template_path is not None:
            with open(system_prompt_template_path, "r") as file:
                system_prompt_template = file.read()
        if temperature is not None:
            self.params["temperature"] = temperature
        params_tuple = tuple(sorted(self.params.items()))
        custom_base_url = getattr(self, "_custom_base_url", None)
        custom_api_key = getattr(self, "_custom_api_key", None)
        print("Cache key:", hash_input(system_prompt_template, question, self.model_type, self.backend, params_tuple, self.context, use_context, custom_base_url))
        cached_data = self._cached_prompt_impl(
            template=system_prompt_template,
            question=question,
            model_type=self.model_type,
            backend=self.backend,
            params=params_tuple,
            context=self.context if use_context else None,
            use_context=use_context,
            custom_base_url=custom_base_url,
            custom_api_key=custom_api_key,
        )

        result = cached_data["outputs"]["result"]
        logprobs = cached_data["outputs"]["logprobs"]

        # Update context if necessary
        if use_context:
            self.context = cached_data["inputs"]["context"]

        return result, logprobs