from flask import Flask, request, jsonify, render_template, redirect
import os, tempfile, traceback
import chromadb
from .util import *
import requests
import uuid
import shutil
from flask import session
from zoneinfo import ZoneInfo
 

server = Flask(__name__)

# Open and load the file
with open("static/config/modelRegistry.json", "r") as f:
    model_registry = json.load(f)

modelurl = "http://localhost:8989/"
inferurl = "http://127.0.0.1:8989/infer/json"
topicinfourl = "http://127.0.0.1:8989//queries/model-info"


client = chromadb.PersistentClient(path="database/myDB")
collection = client.get_or_create_collection(name="documents")
registry = client.get_or_create_collection("corpus_model_registry")

####################################################################
# Render Pages

# 1. Home Page
@server.route('/')
def home():
    return render_template('index.html')

# This function makes you view your data before it is uploaded

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



# This is called when you when the data has been validated to insert data into the vector database

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




# 2. Select Model Page
@server.route('/model')
def loadModel():
    return render_template('loadModel.html')



# This function trains the selected model on the data selected
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


        final_output = []

        for corpus in corpuses:
            documents, metadatas, ids = get_corpus_data(corpus, collection)

            for doc, doc_id in zip(documents, ids):
                final_output.append({
                    "id": doc_id,
                    "raw_text": doc
                })
            
            print(f"Found {len(documents)} documents in corpus '{corpus}'.")

        payload = {
            "config_path": "static/config/config.yaml",
            "model": model_name,
            "output": "models/"+ save_name,
            "text_col": "raw_text",
            "data": final_output,
            "training_params": training_params,
            "do_preprocess": True,
            "id_col": "id",
        }

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }

        response = requests.post(modelurl+"train/json", json=payload, headers=headers)

        # Print result
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())

        trained_on = datetime.utcnow().isoformat()

        
        joined_corpuses = ", ".join(corpuses)  # for display
        joined_id = "_".join(corpuses)         # for ID, safer string

        registry.add(
            documents=[f"Trained {model_name} on {joined_corpuses}"],
            metadatas=[{
                "model_id" : f"{model_name}_{joined_id}_{trained_on}",
                "model_type": model_name,
                "model_name": save_name,
                "corpus_names": joined_corpuses,  # you *can* keep this as a list in metadata
                "trained_on": trained_on
            }],
            ids=[f"{model_name}_{joined_id}_{trained_on}"]
        )

        return jsonify({
            "status": "success",
            "message": "The system works is ready to use."
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Model run failed: {str(e)}"
        }), 500


# This function fetches the corpora to be selected
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
    return jsonify(unique_corpora)


# This function gets the configuration file for the Models
@server.route("/model-config")
def get_model_config():
    with open("static/config/model_config.json") as f:
        return jsonify(json.load(f))
    

# This function gets the model registry
@server.route("/model-registry")
def get_model_registry():
    with open("static/config/modelRegistry.json") as f:
        return jsonify(json.load(f))



@server.route('/delete-corpus/', methods=['POST'])
def delete_corpus():
    try:
        data = request.get_json()
        print("Received request to delete corpus:", data)

        corpus_name = data.get("corpus_name")
        if not corpus_name:
            return jsonify({"status": "error", "message": "No corpus_name provided."}), 400

        collection.delete(where={"corpus_name": corpus_name})

        return jsonify({"status": "success", "message": f"Corpus '{corpus_name}' deleted."})
    except Exception as e:
        print("Error during corpus deletion:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


##################################################################################
# 3. Trained Models
@server.route("/trained-models")
def trained_models():
    results = registry.get()


    # Build list of model entries
    models = [
        {
            "model_id":meta.get("model_id", ""),
            "document": doc,
            "model_type":meta.get("model_type", ""),
            "model_name": meta.get("model_name", ""),
            "corpus_names": meta.get("corpus_names", ""),
            "trained_on": datetime.fromisoformat(meta.get("trained_on", "")).replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M %p %Z")
 
        }
        for id_, doc, meta in zip(results["ids"], results["documents"], results["metadatas"])
    ]

    return render_template("trained_models.html", models=models)


@server.route('/delete-model/', methods=['POST'])
def delete_model():
    data = request.get_json()
    model_id = data.get("model_id")
    model_name = data.get("model_name")
    modelpath = f"./models/{model_name}"

    if os.path.exists(modelpath) and os.path.isdir(modelpath):
        shutil.rmtree(modelpath)
        print(f"✅ Deleted model folder: {modelpath}")
    else:
        print(f"⚠️ Model path does not exist: {modelpath}")

    if not model_id:
        return jsonify({"status": "error", "message": "No model_id provided"}), 400

    try:
        registry.delete(where={"model_id": model_id})
        return jsonify({"status": "success", "message": f"Model '{model_id}' deleted."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


###################################################################################
# 4. Visualization and Inferences
@server.route("/dashboard", methods=["POST", "GET"])
def dashboard():
    return render_template("dashboard.html")

####################################################################################

# 5. Inference Page
@server.route("/infer-page")
def infer_page():
    return render_template("inference.html")

@server.route('/get_model_info', methods=['POST'])  # Change to POST to accept JSON
def get_model_info():
    try:
        data = request.get_json()
        modelName = data.get("model", "").strip()
        if not modelName:
            return jsonify({"error": "Model name is required."}), 400

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }

        topicJson = {
            "config_path": "static/config/config.yaml",
            "model_path": f"data/models/{modelName}"
        }

        # Call your external topic-info URL (replace with your real endpoint)
        response = requests.post(topicinfourl, json=topicJson, headers=headers)

        # print(response)

        if response.status_code != 200:
            return jsonify({"error": "Failed to retrieve model info."}), 500

        topic_info = response.json()
        print(topic_info)
        return jsonify(topic_info)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@server.route('/infer', methods=['POST'])  # ✅ POST supports JSON
def infer_topic():
    
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        modelName = data.get("model").strip()

        payload =  {
            "config_path": "static/config/config.yaml",
            "data": [
                {"id": "myid",
                "raw_text" : text}],
            "id_col": "id",
            "model_path": "models/"+modelName,
            "text_col": "raw_text"
            }
        if not text:
            return jsonify({"error": "Text input is required."}), 400

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        response = requests.post(inferurl, json=payload, headers=headers)

        topicJson = {
            "config_path": "static/config/config.yaml",
            "model_path": "data/models/"+modelName
            }

        topic_info = requests.post(topicinfourl, json=topicJson).json()

        print(topic_info)

        inference_result = response.json()
                # Step 3: Format the top 5 predictions
        theta = inference_result["thetas"][0]["myid"]

        top_topics = sorted(
            [{"id": tid, "score": score} for tid, score in theta.items()],
            key=lambda x: x["score"],
            reverse=True
        )[:7]  # Get top 5

        # Step 4: Merge with metadata
        enriched = []
        for topic in top_topics:
            tid = topic["id"]
            info = topic_info.get(tid, {})
            enriched.append({
                "label": info.get("tpc_labels", tid),
                "score": round(topic["score"], 4),
                "text": text,
            })

        # Final output for frontend
        response = {"topics": enriched}


        return jsonify(response)


    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
        


@server.route('/infer-file', methods=['POST'])
def infer_file():
    try:
        file = request.files.get("file")
        id_col = request.form.get("id_col", "").strip()
        text_col = request.form.get("text_col", "").strip()
        modelName = request.form.get("model", "").strip()

        if not file or not id_col or not text_col or not modelName:
            return jsonify({"error": "Missing file, id_col, text_col, or model name"}), 400

        # Load file
        filename = file.filename.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif filename.endswith(".json"):
            df = pd.read_json(file)
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # Validate columns
        if id_col not in df.columns or text_col not in df.columns:
            return jsonify({"error": f"'{id_col}' or '{text_col}' not found in file"}), 400

        # Create input data for inference
        records = [
            {"id": str(row[id_col]), "raw_text": str(row[text_col])}
            for _, row in df.iterrows()
            if pd.notna(row[id_col]) and pd.notna(row[text_col])
        ]
        id_to_text = {r["id"]: r["raw_text"] for r in records}

        payload = {
            "config_path": "static/config/config.yaml",
            "data": records,
            "id_col": "id",
            "model_path": f"models/{modelName}",
            "text_col": "raw_text"
        }

        headers = {"accept": "application/json", "Content-Type": "application/json"}
        response = requests.post(inferurl, json=payload, headers=headers)
        inference_result = response.json()

        # Topic metadata
        topic_metadata = requests.post(topicinfourl, json={
            "config_path": "static/config/config.yaml",
            "model_path": f"models/{modelName}"
        }, headers=headers).json()

        # Format output
        final_output = []
        for theta_entry in inference_result["thetas"]:
            for doc_id, topic_scores in theta_entry.items():
                top_topics = sorted(
                    [{"id": tid, "score": score} for tid, score in topic_scores.items()],
                    key=lambda x: x["score"],
                    reverse=True
                )[:3]

                enriched = []
                for topic in top_topics:
                    tid = topic["id"]
                    info = topic_metadata.get(tid, {})
                    enriched.append({
                        "label": info.get("tpc_labels", tid),
                        "score": round(topic["score"], 4),
                        "keywords": info.get("tpc_descriptions", "").split(", ")
                    })

                final_output.append({
                    "doc_id": doc_id,
                    "text": id_to_text.get(doc_id, ""),
                    "top_topics": enriched
                })

        return jsonify({"topics": final_output})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@server.route("/inference/<model_id>")
def inference_page(model_id):
    return render_template("inference_page.html", model_id=model_id)

    
######################################################################################




# This function displays the available models
@server.route('/get_models', methods=['GET'])
def list_models():
    model_dir = os.path.join("models")
    
    if not os.path.exists(model_dir):
        return jsonify([])  # Return empty if no directory

    # List all directories/files inside model_dir
    models = [
        name for name in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, name))
    ]
    print(models)

    return jsonify(models)








if __name__ == '__main__':
    server.run(debug=True)