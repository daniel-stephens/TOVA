def stub_vllm_if_missing():
    """Inject a no-op vllm stub into sys.modules before topicgpt_python imports it. Only injects if vllm is not installed (e.g. ARM/CPU builds); if the real vllm is installed it will be used as-is."""
    import importlib.util
    import sys
    if "vllm" not in sys.modules and importlib.util.find_spec("vllm") is None:
        from types import ModuleType
        stub = ModuleType("vllm")

        class LLM:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "vllm is not available in this build. Use ollama or openai.")

        class SamplingParams:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "vllm is not available in this build. Use ollama or openai.")

        stub.LLM = LLM
        stub.SamplingParams = SamplingParams
        sys.modules["vllm"] = stub


def patch_topicgpt_apiclient():
    import logging
    import os
    import pathlib

    import topicgpt_python.utils as u  # type: ignore

    from tova.utils.common import init_logger, load_yaml_config_file

    Original = u.APIClient

    def patched_init(
        self,
        api,
        model,
        host=None,
        config_path: pathlib.Path = pathlib.Path(
            "./static/config/config.yaml"),
        logger: logging.Logger = None,
    ) -> None:
        self.api = api
        self.model = model
        self.client = None

        self._logger = logger if logger else init_logger(config_path, __name__)
        config = load_yaml_config_file(config_path, "llm", logger)

        if api == "ollama":
            from openai import OpenAI
            base_url = config.get("ollama", {}).get(
                "host", "http://localhost:11434")
            if not base_url.endswith("/v1"):
                base_url = base_url + "/v1"
            self.client = OpenAI(base_url=base_url, api_key="ollama")
            return

        Original.__init__(self, api, model, host=host)

    u.APIClient.__init__ = patched_init


stub_vllm_if_missing()
patch_topicgpt_apiclient()
