from .base import LLMTModel

class GPTopicTMmodel(LLMTModel):
    def save_model(self, path): print(f"Saving GPTopic model to {path}")
    def load_model(self, path): print(f"Loading GPTopic model from {path}")
    def train_model(self, data): print("Training GPTopic model...")
    def infer(self, new_data): print("Inferring with GPTopic model...")
