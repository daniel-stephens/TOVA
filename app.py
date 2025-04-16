from flask import Flask, request, jsonify, render_template, redirect
import os
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
import tempfile
# from .util import *
from .adapters.lda_adapter import LDAAdapter
import dash_table
import plotly.express as px
import pandas as pd
import requests
from collections import Counter
from dash import Dash, html, dcc, dash_table, Input, Output
from .dashboard import init_dash_app
from .util import *




server = Flask(__name__)

# Optional model registry
model_registry = {
    "lda": LDAAdapter,
    # "bertopic": BERTopicAdapter,
    # "top2vec": Top2VecAdapter
}

client = chromadb.PersistentClient(path="database/myDB")
collection = client.get_or_create_collection(name="documents")
collect = client.get_or_create_collection(name="doc")

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
    labelColumn = request.form.get('label_column')

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
            label_column=labelColumn
        )

            # 🧹 Delete previous entries for this file (if any)
            collection.delete(where={"file_name": file.filename})

            # 🆕 Insert new entries
            for i, row in df.iterrows():
                collection.add(
                documents=[row["processed_text"]],
                metadatas=[{
                    "original_content": row["calculate_on"],
                    "file_name": file.filename,
                    "label": row.get(labelColumn) or ""  # ✅ Convert None to empty string
                }],
                # embeddings=[row["embedding"]],
                ids=[f"{file.filename}_{i}"]
)

                inserted += 1

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


@server.route('/run_model', methods=['POST'])
def run_model():
    try:
        # === 1. Parse input ===
        data = request.get_json()
        model_name = data.get("model_name")
        num_topics = int(data.get("num_topics", 10))
        save_name = data.get("save_name")

        print(model_name)

        if not model_name or model_name.lower() not in model_registry:
            return jsonify({"status": "error", "message": "Invalid model name"}), 400

        # === 2. Load data from ChromaDB ===
        
        chroma_data = collection.get(include=["documents", "metadatas"])
        # print(chroma_data)
        documents = chroma_data["documents"]
        metadatas = chroma_data["metadatas"]
        ids = chroma_data["ids"]

        # print(documents)

        if not documents or len(documents) == 0:
            return jsonify({"status": "error", "message": "No documents found in the database."}), 404

        # === 3. Initialize and fit model ===
        ModelClass = model_registry[model_name.lower()]
        model = ModelClass(num_topics=num_topics)
        print("fitting the data in the model")
        model.fit(documents)
        print("fit complete")

        print("==== SAVING THE MODEL ===")

        model.save(f"model/{save_name}")



        # === 4. Get topic assignments ===
        # doc_topics = model.get_document_topics()

        

        return jsonify({
            "status": "success",
            "message": f"{model.get_model_name()} is ready to use."
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


dash_app = init_dash_app(server)
if __name__ == '__main__':
    server.run(debug=True)
