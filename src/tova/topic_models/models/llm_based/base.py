from tova.topic_models.models.base_model import BaseTMModel

class LLMTModel(BaseTMModel):
    def use_llm_embeddings(self, data):
        print("Using LLM-based embedding pipeline")
