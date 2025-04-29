from flask import Flask, request, jsonify, render_template, redirect
import os
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
import tempfile
# from .util import *
from .util import *
from .adapters.lda_adapter import LDAAdapter
import dash_table
import plotly.express as px
import pandas as pd
import requests
from collections import Counter
from dash import Dash, html, dcc, dash_table, Input, Output
from .dashboard import init_dash_app
# from .src.commands.train import run



server = Flask(__name__)

# Open and load the file
with open("static/config/modelRegistry.json", "r") as f:
    model_registry = json.load(f)

modelurl = "http://localhost:8000"


client = chromadb.PersistentClient(path="database/myDB")
collection = client.get_or_create_collection(name="documents")
# collect = client.get_or_create_collection(name="doc")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@server.route('/validate', methods=['POST'])
def validate_route():
    file = request.files.get('files')
    text_column = request.form.get('text_column')
    label_column = request.form.get('label_column')

    if not file:
        return jsonify({
            "status": "error",
            "message": "No file uploaded."
        }), 400

    if not file.filename.endswith(('.xls', '.xlsx')):
        return jsonify({
            "status": "error",
            "message": "Only Excel files are supported for validation here."
        }), 400

    try:
        # Call your validation utility and pass column names
        result = excel_confirmation(file, text_column, label_column)
        status_code = 200 if result["status"] == "success" else 400
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Validation failed: {str(e)}"
        }), 500



@server.route('/upload', methods=['POST'])
def upload_route():
    # for eachFile in finalFiles:
    #     print(eachFile)

    return jsonify({
            "status": "Success",
            "message": "This code worked"
        }), 200


@server.route('/loadDB', methods=['POST'])
def loadDB_route():
    files = request.files.getlist('files')
    textColumn = request.form.get('text_column')
    corpusName = request.form.get('corpusName')

    if not files:
        return jsonify({"status": "error", "message": "No files received."}), 400

    inserted = 0
    for file in files:
        print("📥", file.filename)
        ext = os.path.splitext(file.filename)[1].lower().replace('.', '')  # get extension without the dot

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            # ✅ Preprocess & Embed directly
            df = preprocess_and_embed(
            file_path=tmp_path,
            file_type=ext,
            text_columns=[textColumn],
            # label_column=labelColumn
        )

            # 🧹 Delete previous entries for this file (if any)
            # collection.delete(where={"file_name": file.filename})

            documents = df["processed_text"].tolist()

            metadatas = [{
                "original_content": row["calculate_on"],
                "file_name": file.filename,
                "corpus_name": corpusName  # ✅ Key line to enable filtering/grouping by corpus
            } for _, row in df.iterrows()]

            ids = [f"{file.filename}_{i}" for i in range(len(df))]

            collection.add(
                documents=documents,
                metadatas=metadatas,
                # embeddings=df["embedding"].tolist(),  # Uncomment if embeddings are used
                ids=ids
            )

            inserted += len(df)

        except Exception as e:
            print(f"❌ Error processing {file.filename}: {e}")

        finally:
            os.remove(tmp_path)

    return jsonify({
        "status": "success",
        "message": f"{inserted} document(s) processed and added."
    }), 200



@server.route('/preview')
def preview():
    results = collection.get(include=['documents', 'metadatas'])
    data = []

    for id_, doc, meta in zip(results['ids'], results['documents'], results['metadatas']):
        data.append({
            "id": id_,
            "processed_text": doc,
            "content": meta.get("original_content", ""),
        })

    return jsonify(data)


@server.route('/train_model', methods=['POST'])
def run_model():
    try:
        data = request.get_json()

        model_name = data.get("model")
        save_name = data.get("save_name")
        num_topics = data.get("num_topics", 10)  # Default to 10 if missing
        corpus = data.get("corpus")

        # Advanced settings (optional)
        advanced_settings = data.get("advanced_settings", {})

        # Basic validations
        if not model_name:
            return jsonify({"status": "error", "message": "Missing 'model_name' in request."}), 400

        if model_name not in model_registry:
            return jsonify({"status": "error", "message": f"Model '{model_name}' not found in registry."}), 400

        if not save_name:
            return jsonify({"status": "error", "message": "Missing 'save_name' (name to save the model)."}), 400

        # Try to ensure num_topics is a valid number
        try:
            num_topics = int(num_topics)
            if num_topics < 2:
                return jsonify({"status": "error", "message": "'num_topics' must be greater than or equal to 2."}), 400
        except ValueError:
            return jsonify({"status": "error", "message": "'num_topics' must be an integer."}), 400

        # If all validations pass, you can print or continue processing
        # print("Parsed Model Name:", model_name)
        # print("Save Name:", save_name)
        # print("Number of Topics:", num_topics)
        # print("Advanced Settings:", advanced_settings)
        # === 2. Load data from ChromaDB ===
        
        # chroma_data = collection.get(include=["documents", "metadatas"])
        # # print(chroma_data)

        print(corpus)
        
        documents, metadatas, ids = get_corpus_data(corpus, collection)

        print(f"Found {len(documents)} documents in corpus '{corpus}'.")
        # # print(documents)

        # if not documents or len(documents) == 0:
        #     return jsonify({"status": "error", "message": "No documents found in the database."}), 404

        # # === 3. Initialize and fit model ===
        

        payload = {
            "model": model_name,
            "data": "/Users/danielstephens/Desktop/TOVA/data/dat/bills_sample_100.csv",
            "text_col": "tokenized_text",
            "output": "models/tomotopy"
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(modelurl+"/train/", json=payload, headers=headers)

        # Print result
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())

        # model = run(model=model, data=documents, text_col="text_col", output="models/tomotopy")

        # print("==== SAVING THE MODEL ===")

        # model.save(f"model/{save_name}")



        # === 4. Get topic assignments ===
        # doc_topics = model.get_document_topics()

        

        return jsonify({
            "status": "success",
            "message": "The system works is ready to use."
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Model run failed: {str(e)}"
        }), 500
    



@server.route('/')
def home():
    return render_template('index.html')

@server.route('/model')
def loadModel():
    return render_template('loadModel.html')

@server.route('/corpora')
def get_corpus_names():
    print("Looking for corpus names...")
    results = collection.get(include=["metadatas"], limit=10000)
    all_corpora = [
        meta.get("corpus_name", "").strip()
        for meta in results["metadatas"]
        if meta.get("corpus_name") and isinstance(meta.get("corpus_name"), str)
    ]
    unique_corpora = sorted(set(all_corpora))
    
    print("Found:", unique_corpora)
    return jsonify(unique_corpora)

@server.route("/model-config")
def get_model_config():
    with open("static/config/model_config.json") as f:
        
        return jsonify(json.load(f))
    
@server.route("/model-registry")
def get_model_registry():
    with open("static/config/modelRegistry.json") as f:
        return jsonify(json.load(f))



dash_app = init_dash_app(server)
if __name__ == '__main__':
    server.run(debug=True)
