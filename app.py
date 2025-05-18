from flask import Flask, request, jsonify, render_template, redirect
import os, tempfile, traceback
import chromadb
from .util import *
import requests
import uuid
from flask import session
 

server = Flask(__name__)

# Open and load the file
with open("static/config/modelRegistry.json", "r") as f:
    model_registry = json.load(f)

modelurl = "http://localhost:8989"


client = chromadb.PersistentClient(path="database/myDB")
collection = client.get_or_create_collection(name="documents")
# collect = client.get_or_create_collection(name="doc")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



####################################################################
# Render Pages

# 1. Home Page
@server.route('/')
def home():
    return render_template('index.html')


# 2. Select Model Page
@server.route('/model')
def loadModel():
    return render_template('loadModel.html')

# 3. Inference Page
@server.route("/infer-page")
def infer_page():
    return render_template("inference.html")



#####################################################################

# This Route Validates the selected files

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



###########################################################################

# This is called when you when the data has been validated

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
            # ✅ Load and prepare text columns
            df = processFile(
                file_path=tmp_path,
                file_type=ext,
                text_column=textColumn
            )

            print("Processed Complete")

            # ✅ Check if 'Context' column exists
            if "Context" not in df.columns:
                raise ValueError(f"'Context' column not found in file. Found columns: {list(df.columns)}")

            documents = df["Context"].astype(str).tolist()  # Ensure strings
            print("Documents Retrieved")

            # ✅ Get safe filename
            file_name = getattr(file, "filename", str(file))

            print("\nBuilding Metadata")

            # ✅ Build metadatas
            metadatas = [{
                "file_name": file_name,
                "corpus_name": corpusName,
                "models_used": ""  # Start with an empty list, update later
            } for _ in documents]  # Not df.iterrows()

            print("\nGetting the IDs")
            # ✅ Build IDs
            ids = [f"{file_name}_{i}" for i in range(len(documents))]

            # ✅ Confirm all lengths match
            assert len(documents) == len(metadatas) == len(ids), "Mismatch in data lengths"

            print("\n\n Inserting into the Database")
            # ✅ Insert into collection
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print("\n\nInsert done")
            inserted += len(df)

        except Exception as e:
            
            print(f"❌ Failed to insert file {file.filename}: {str(e)}")
            traceback.print_exc()


        finally:
            os.remove(tmp_path)

    return jsonify({
        "status": "success",
        "message": f"{inserted} document(s) processed and added."
    }), 200



##############################################################################

# This function

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



########################################################################
# This is called to train the model


@server.route('/train_model', methods=['POST'])
def run_model():
    try:
        data = request.get_json()

        model_name = data.get("model")
        save_name = data.get("save_name")
        corpuses = data.get("corpuses")
        training_params = data.get("training_params")
        num_topics = training_params["num_topics"]


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


        for corpus in corpuses:
            documents, metadatas, ids = get_corpus_data(corpus, collection)
            
            print(f"Found {len(documents)} documents in corpus '{corpus}'.")

        



        # # === 3. Initialize and fit model ===
        

        payload = {
            "model": model_name,
            "output": "data/models/"+ save_name,
            "text_col": "tokenized_text",
            "data": "data/dat/bills_sample_100.csv",
            # "training_params": training_params
        }

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }

        response = requests.post(modelurl+"/train", json=payload, headers=headers)

        # Print result
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())

        return jsonify({
            "status": "success",
            "message": "The system works is ready to use."
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Model run failed: {str(e)}"
        }), 500
    





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

@server.route('/infer', methods=['POST'])  # ✅ POST supports JSON
def infer_topic():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        print(text)
        if not text:
            return jsonify({"error": "Text input is required."}), 400


        return jsonify({
                    "topics": [
                        {"label": "Urban Mobility", "score": 0.42, "keywords": ["transit", "commute", "city"]},
                        {"label": "Policy", "score": 0.25, "keywords": ["regulation", "planning"]},
                        {"label": "Sustainability", "score": 0.18, "keywords": ["climate", "green", "emissions"]},
                    ]
                })


    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



    

if __name__ == '__main__':
    server.run(debug=True)