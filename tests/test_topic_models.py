from pyexpat import model
import unittest
import pathlib
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tova.topic_models.models.traditional.base import BaseTMModel

class TestTopicModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load model registry and prepare common resources."""
        with open("./static/config/modelRegistry.json", "r") as f:
            model_classes = json.load(f)
        
        # Debugging: Print the loaded model registry
        print("Loaded model registry:", model_classes)

        # Ensure model_classes contains string paths
        cls.model_registry = {
            key: path for key, path in model_classes.items() if isinstance(path, str)
        }
        cls.sample_file = "data_test/bills_sample_100.csv"
        cls.sample_data = pd.read_csv(cls.sample_file)
        cls.train_data = cls.sample_data.rename(columns={"summary": "raw_text"})
        cls.train_data = cls.train_data[["id", "raw_text"]].to_dict(orient="records")

    @staticmethod
    def load_class_from_path(class_path: str):
        """Dynamically load a class from a given module path."""
        module_path, class_name = class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    # def test_train_and_infer(self):
    #     """Test training and inference for all registered topic models."""
    #     for model_name, model_path in self.model_registry.items():
    #         with self.subTest(model=model_name):
    #             model_cls = self.load_class_from_path(model_path)

    #             model = model_cls(
    #                 model_name=f"test_{model_name}",
    #                 corpus_id="test_corpus",
    #                 id="test_id",
    #                 model_path=pathlib.Path(f"./test_{model_name}"),
    #                 logger=logging.getLogger(f"test_logger_{model_name}"),
    #                 config_path=pathlib.Path("./static/config/config.yaml"),
    #                 load_model=False,
    #                 preprocess_text=True,
    #             )

    #             # Train the model
    #             model.train_model(self.train_data)
                
    #             self.model_path = model_path

    #             # Perform inference
    #             infer_data = self.sample_data.rename(columns={"summary": "raw_text"})
    #             infer_data = infer_data[["id", "raw_text"]].to_dict(orient="records")
    #             thetas, time_taken = model.infer(infer_data)

    #             self.assertIsInstance(thetas, np.ndarray)
    #             print(f"Inference completed in {time_taken} seconds for model {model_name}")

    def test_load_and_infer(self):
        # Load the model
        for model_name, model_path in self.model_registry.items():
            with self.subTest(model=model_name):
                model_cls = self.load_class_from_path(model_path)
                tm_model = model_cls.from_saved_model(f"test_{model_name}")
            
                # Perform inference
                infer_data = self.sample_data.rename(columns={"summary": "raw_text"})
                infer_data = infer_data[["id", "raw_text"]].to_dict(orient="records")
                thetas, duration = tm_model.infer(infer_data)

                # Assertions
                self.assertIsNotNone(thetas, "Thetas should not be None")
                self.assertGreater(len(thetas), 0, "Thetas should contain results")
                self.assertGreater(duration, 0, "Duration should be greater than 0")

if __name__ == "__main__":
    unittest.main()