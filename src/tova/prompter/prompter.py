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
        llm_server: str = None,
        llm_provider: str = None,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path("config/config.yaml"),
        temperature: float = None,
        seed: int = None,
        max_tokens: int = None,
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

        # Determine backend based on llm_provider
        if llm_provider is not None:
            self.backend = llm_provider
            self._logger.info(f"Using provider override: {llm_provider}")
        elif model_type in self.GPT_MODELS:
            self.backend = "openai"
        elif model_type in self.OLLAMA_MODELS:
            self.backend = "ollama"
        elif model_type == "llama_cpp":
            self.backend = "llama_cpp"
        else:
            raise ValueError("Unsupported model_type specified.")

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
            if self.backend == "openai":
                self.params["max_completion_tokens"] = max_tokens
                self._logger.info(f"Setting max_completion_tokens to: {max_tokens}")
            # for ollama models, the parameter is 'num_predict'
            # https://github.com/ollama/ollama/blob/main/docs/modelfile.md
            elif self.backend == "ollama":
                self.params["num_predict"] = max_tokens
                self._logger.info(f"Setting num_predict to: {max_tokens}")
            else:
                self.params["max_tokens"] = max_tokens
                self._logger.info(f"Setting max_tokens to: {max_tokens}")

        if self.backend == "openai":
            load_dotenv(self.config.get("gpt", {}).get("path_api_key", ".env"))
            # llm_server can be used as a custom base_url (e.g. for OpenAI-compatible APIs)
            self.openai_base_url = llm_server
            self._logger.info(
                f"Using OpenAI API with model: {model_type}"
                + (f", base_url: {llm_server}" if llm_server else "")
            )
        elif self.backend == "ollama":
            ollama_host = llm_server or self.config.get("ollama", {}).get(
                "host", "http://kumo01.tsc.uc3m.es:11434"
            )
            os.environ['OLLAMA_HOST'] = ollama_host
            # Initialize as class-level variable to be able to use it in the cache function
            Prompter.ollama_client = Client(
                host=ollama_host,
                headers={'x-some-header': 'some-value'}
            )
            available_models = [m.model for m in Prompter.ollama_client.list().models]
            if model_type not in available_models:
                raise ValueError(
                    f"Model '{model_type}' is not available on the Ollama server at {ollama_host}. "
                    f"Available models: {available_models}"
                )
            self._logger.info(f"Using OLLAMA API with host: {ollama_host}")
        elif self.backend == "llama_cpp":
            self.llama_cpp_host = llm_server or self.config.get("llama_cpp", {}).get(
                "host", "http://kumo01:11435/v1/chat/completions"
            )
            self._logger.info(f"Using llama_cpp API with host: {self.llama_cpp_host}")
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

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
        openai_base_url: str = None,
    ) -> dict:
        """Caching setup."""

        print("Cache miss: computing results...")

        if backend == "openai":
            result, logprobs = Prompter._call_openai_api(
                template=template,
                question=question,
                model_type=model_type,
                params=dict(params),
                base_url=openai_base_url,
            )
        elif backend == "ollama":
            result, logprobs, context = Prompter._call_ollama_api(
                template=template,
                question=question,
                model_type=model_type,
                params=dict(params),
                context=context,
            )
        elif backend == "llama_cpp":
            result, logprobs = Prompter._call_llama_cpp_api(
                template=template,
                question=question,
                params=dict(params),
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return {
            "inputs": {
                "template": template,
                "question": question,
                "model_type": model_type,
                "backend": backend,
                "params": dict(params),
                "context": context if use_context else None,
                "use_context": use_context,
            },
            "outputs": {
                "result": result,
                "logprobs": logprobs,
            },
        }

    @staticmethod
    def _call_openai_api(template, question, model_type, params, base_url=None):
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

        client_kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        open_ai_client = OpenAI(**client_kwargs)
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

    def prompt(
        self,
        system_prompt_template_path: str,
        question: str,
        use_context: bool = False,
        temperature: float = None,
    ) -> Union[str, List[str]]:
        """Public method to execute a prompt given a system prompt template and a question."""

        # Load the system prompt template
        system_prompt_template = None
        if system_prompt_template_path is not None:
            with open(system_prompt_template_path, "r") as file:
                system_prompt_template = file.read()

        # Ensure hashable params for caching and get cached data / execute prompt
        if temperature is not None:
            self.params["temperature"] = temperature
        params_tuple = tuple(sorted(self.params.items()))
        
        print("Cache key:", hash_input(system_prompt_template, question, self.model_type, self.backend, params_tuple, self.context, use_context))
        cached_data = self._cached_prompt_impl(
            template=system_prompt_template,
            question=question,
            model_type=self.model_type,
            backend=self.backend,
            params=params_tuple,
            context=self.context if use_context else None,
            use_context=use_context,
            openai_base_url=getattr(self, "openai_base_url", None),
        )

        result = cached_data["outputs"]["result"]
        logprobs = cached_data["outputs"]["logprobs"]

        # Update context if necessary
        if use_context:
            self.context = cached_data["inputs"]["context"]

        return result, logprobs