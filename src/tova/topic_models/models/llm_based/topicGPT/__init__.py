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


patch_topicgpt_apiclient()