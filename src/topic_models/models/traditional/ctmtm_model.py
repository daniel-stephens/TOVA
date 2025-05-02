from .base import TradTMmodel

class CTMTMmodel(TradTMmodel):
    def save_model(self, path): print(f"Saving CTM model to {path}")
    def load_model(self, path): print(f"Loading CTM model from {path}")
    def train_model(self, data): print("Training CTM model...")
    def infer(self, new_data): print("Inferring with CTM model...")
