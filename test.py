import importlib
import json
import pathlib
import logging
import pandas as pd
import numpy as np
import pyLDAvis
import multiprocessing

from tova.topic_models.models.llm_based.topicrag.open_topic_rag_model import OpenTopicRAGModel # A veces es útil importarlo explícitamente

def main():
    
    with open("./static/config/modelRegistry.json", "r") as f:
        model_classes = json.load(f)
    
    def load_class_from_path(class_path: str):
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


    MODEL_REGISTRY = {
        key: load_class_from_path(path) for key, path in model_classes.items()
    }

    print(MODEL_REGISTRY)

    model = "OpenTopicRAGModel"
    model_name = "test_nb"
    id = "xxx"
    data_file = "data_test/bills_sample_100.csv"
    train_data = pd.read_csv(data_file).sample(1000, random_state=42)
    train_data = train_data.rename(columns={"summary": "raw_text"})
    train_data = train_data[["id", "raw_text"]].to_dict(orient="records")

    model_cls = MODEL_REGISTRY.get(model)
    if model_cls is None:
        raise ValueError(f"Unknown model: {model}")

    tr_params = {
        #"num_topics": 50,
        #"preprocess_text": False,
        "run_from_web": False
    }

    tm_model = model_cls(
        model_name=model_name,
        corpus_id="c_4e3634ace8f94d8e899142ef637348c0",
        id=id,
        model_path=pathlib.Path(f"data/tests/test_{model_name}"),
        load_model=False,
        logger=logging.getLogger(f"test_logger_{model_name}"),
        **tr_params
    )
    time = tm_model.train_model(train_data)
    print("Entrenamiento finalizado con éxito.")
    
    
    mo_reload = OpenTopicRAGModel.from_saved_model(
        pathlib.Path(f"data/tests/test_{model_name}"),
       
    )
    mo_reload.print_topics(verbose=True)

if __name__ == '__main__':
    # En macOS es recomendable añadir esta línea también, aunque el if suele bastar
    multiprocessing.freeze_support() 
    
    main()