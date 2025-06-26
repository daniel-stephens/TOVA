from flask import Flask, request, jsonify, render_template, abort
import os, tempfile, traceback
import chromadb
from .util import *
import requests
import uuid
import shutil
from flask import session
from zoneinfo import ZoneInfo
import random
from datetime import datetime
import sqlite3


server = Flask(__name__)
server.secret_key = 'your-secret-key'
dashboard_data = load_or_create_dashboard_json()



# Open and load the file
with open("static/config/modelRegistry.json", "r") as f:
    model_registry = json.load(f)

modelurl = "http://localhost:8989/"
inferurl = "http://127.0.0.1:8989/infer/json"
topicinfourl = "http://127.0.0.1:8989//queries/model-info"
textinfourl = "http://localhost:8989/queries/thetas-by-docs-ids"


client = chromadb.PersistentClient(path="database/myDB")
collection = client.get_or_create_collection(name="documents")
registry = client.get_or_create_collection("corpus_model_registry")

####################################################################
# Render Pages

# 1. Home Page
@server.route('/')
def home():
    create_normalized_schema()
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
    text_column = request.form.get('text_column')
    corpus_name = request.form.get('corpusName')
    original_id_column = request.form.get('originalID')  # Optional

    if not files:
        return jsonify({"status": "error", "message": "No files received."}), 400

    inserted = 0

    # Connect to SQLite
    conn = sqlite3.connect("database/mydatabase.db")
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Insert corpus (if it doesn't exist)
    cursor.execute("""
        INSERT OR IGNORE INTO corpus (name) VALUES (?)
    """, (corpus_name,))

    for file in files:
        # print("📥", file.filename)
        ext = os.path.splitext(file.filename)[1].lower().replace('.', '')

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            # Load file into DataFrame
            df = processFile(
                file_path=tmp_path,
                file_type=ext,
                text_column=text_column
            )
            print("Processed Complete")

            if "Context" not in df.columns:
                raise ValueError(f"'Context' column not found in file. Found columns: {list(df.columns)}")

            documents = df["Context"].astype(str).tolist()
            original_ids = (
                df[original_id_column].astype(str).tolist()
                if original_id_column and original_id_column in df.columns
                else [None] * len(documents)
            )

            file_name = getattr(file, "filename", str(file))
            ids = [f"{file_name}_{i}" for i in range(len(documents))]

            assert len(ids) == len(documents) == len(original_ids), "Length mismatch"

            # Prepare and insert data
            insert_query = """
                INSERT OR IGNORE INTO documents (
                    id, original_text_id, document, file_name, corpus_name, models_used
                ) VALUES (?, ?, ?, ?, ?, ?)
            """
            data_to_insert = list(zip(
                ids,
                original_ids,
                documents,
                [file_name] * len(documents),
                [corpus_name] * len(documents),
                [''] * len(documents)  # models_used
            ))
            cursor.executemany(insert_query, data_to_insert)
            conn.commit()
            print("Inserted into documents")
            inserted += len(documents)

        except Exception as e:
            print(f"❌ Failed to insert file {file.filename}: {str(e)}")
            traceback.print_exc()

        finally:
            os.remove(tmp_path)

    conn.close()

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
        num_topics = training_params.get("num_topics")

        if not model_name:
            return jsonify({"status": "error", "message": "Missing 'model_name'."}), 400

        if model_name not in model_registry:
            return jsonify({"status": "error", "message": f"Model '{model_name}' not found in registry."}), 400

        if not save_name:
            return jsonify({"status": "error", "message": "Missing 'save_name'."}), 400

        try:
            num_topics = int(num_topics)
            if num_topics < 2:
                return jsonify({"status": "error", "message": "'num_topics' must be >= 2."}), 400
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "'num_topics' must be an integer."}), 400

        # Connect to SQLite
        conn = sqlite3.connect("database/mydatabase.db")
        cursor = conn.cursor()

        final_output = []

        for corpus in corpuses:
            cursor.execute("SELECT id, document FROM documents WHERE corpus_name = ?", (corpus,))
            rows = cursor.fetchall()

            # print(f"Found {len(rows)} documents in corpus '{corpus}'.")
            for doc_id, doc_text in rows:
                final_output.append({
                    "id": doc_id,
                    "raw_text": doc_text
                })

        if not final_output:
            return jsonify({"status": "error", "message": "No documents found for selected corpora."}), 400

        # Payload for model training API
        payload = {
            "config_path": "static/config/config.yaml",
            "model": model_name,
            "output": f"models/{save_name}",
            "text_col": "raw_text",
            "data": final_output,
            "training_params": training_params,
            "do_preprocess": True,
            "id_col": "id"
        }

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }

        response = requests.post(modelurl + "train/json", json=payload, headers=headers)

        print("Status Code:", response.status_code)
        # print("Response JSON:", response.json())

        trained_on = datetime.utcnow().isoformat()
        joined_corpuses = ", ".join(corpuses)
        joined_id = "_".join(corpuses)
        model_id = f"{model_name}_{joined_id}_{trained_on}"

        # Insert model metadata
        # Insert model metadata
        cursor.execute("""
            INSERT INTO model_registry (
                model_id, model_type, num_topic, model_name, trained_on, description, training_params
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            model_name,
            num_topics,
            save_name,
            trained_on,
            f"Trained {model_name} on {joined_corpuses}",
            json.dumps(training_params)  # 👈 Serialize training params to JSON
        ))


        # Map model to each corpus
        model_corpus_pairs = [(model_id, corpus) for corpus in corpuses]
        cursor.executemany("""
            INSERT INTO model_corpus_map (model_id, corpus_name)
            VALUES (?, ?)
        """, model_corpus_pairs)

        conn.commit()
        conn.close()

        return jsonify({
            "status": "success",
            "message": f"Model '{model_name}' trained and saved as '{save_name}'."
        }), 200

    except Exception as e:
        print("❌ Error in run_model:", str(e))
        return jsonify({
            "status": "error",
            "message": f"Model run failed: {str(e)}"
        }), 500


@server.route('/corpora')
def get_corpus_names():
    print("Looking for corpus names...")

    try:
        conn = sqlite3.connect("database/mydatabase.db")
        cursor = conn.cursor()

        # Get all corpus names from the corpus table
        cursor.execute("SELECT name FROM corpus")
        rows = cursor.fetchall()

        # Extract and sort
        unique_corpora = sorted(row[0] for row in rows)

        return jsonify(unique_corpora)

    except Exception as e:
        print(f"❌ Error fetching corpus names: {e}")
        return jsonify({"status": "error", "message": "Failed to retrieve corpus names"}), 500

    finally:
        conn.close()



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
        print("Received request to delete corpus:")

        corpus_name = data.get("corpus_name")
        if not corpus_name:
            return jsonify({"status": "error", "message": "No corpus_name provided."}), 400

        # Connect to database
        conn = sqlite3.connect("database/mydatabase.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")

        # First delete all documents that belong to the corpus
        cursor.execute("DELETE FROM documents WHERE corpus_name = ?", (corpus_name,))

        # Then delete the corpus itself
        cursor.execute("DELETE FROM corpus WHERE name = ?", (corpus_name,))

        conn.commit()
        conn.close()

        return jsonify({
            "status": "success",
            "message": f"Corpus '{corpus_name}' and its documents deleted."
        })

    except Exception as e:
        print("Error during corpus deletion:", str(e))
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


##################################################################################
# 3. Trained Models
from zoneinfo import ZoneInfo
from datetime import datetime
import sqlite3
from flask import render_template, jsonify

@server.route("/trained-models")
def trained_models():
    try:
        conn = sqlite3.connect("database/mydatabase.db")
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM model_registry")
        rows = cursor.fetchall()

        # Get column names so we can access fields by name (safer if schema changes)
        columns = [desc[0] for desc in cursor.description]

        models = []
        for row in rows:
            row_dict = dict(zip(columns, row))

            # Format timestamp
            try:
                dt = datetime.fromisoformat(row_dict["trained_on"]).replace(tzinfo=ZoneInfo("UTC"))
                local_time = dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M %p %Z")
            except Exception:
                local_time = row_dict["trained_on"]

            models.append({
                "model_id": row_dict["model_id"],
                "document": row_dict["description"],
                "model_type": row_dict["model_type"],
                "model_name": row_dict["model_name"],
                "num_topics": row_dict["num_topic"],
                "training_params": json.loads(row_dict["training_params"]) if row_dict.get("training_params") else {},
                "corpus_names": get_model_corpora(row_dict["model_id"]),
                "trained_on": local_time
            })

        return render_template("trained_models.html", models=models)

    except Exception as e:
        print("❌ Error fetching trained models:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        conn.close()


@server.route('/delete-model/', methods=['POST'])
def delete_model():
    data = request.get_json()
    model_id = data.get("model_id")
    model_name = data.get("model_name")
    modelpath = f"./models/{model_name}"

    # ✅ Delete model folder (if exists)
    if os.path.exists(modelpath) and os.path.isdir(modelpath):
        shutil.rmtree(modelpath)
        print(f"✅ Deleted model folder: {modelpath}")
    else:
        print(f"⚠️ Model path does not exist: {modelpath}")

    if not model_id:
        return jsonify({"status": "error", "message": "No model_id provided"}), 400

    try:
        conn = sqlite3.connect("database/mydatabase.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")

        # ✅ Delete from model_corpus_map first (due to FK constraints)
        cursor.execute("DELETE FROM model_corpus_map WHERE model_id = ?", (model_id,))
        # ✅ Delete from model_registry
        cursor.execute("DELETE FROM model_registry WHERE model_id = ?", (model_id,))
        conn.commit()
        conn.close()

        # ✅ Optionally update dashboard data
        dashboard_data = read_dashboard_json()
        if model_name in dashboard_data:
            del dashboard_data[model_name]
            with open('data.json', 'w') as file:
                json.dump(dashboard_data, file, indent=4)

        return jsonify({
            "status": "success",
            "message": f"Model '{model_id}' deleted."
        })

    except Exception as e:
        print("❌ Error deleting model:", str(e))
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


###################################################################################

# 4. Visualization and Inferences
@server.route("/dashboard/<model_name>", methods=["GET", "POST"])
def dashboard(model_name):

    # print("Model Name (from URL):", model_name)

    themeSummary, themeDetails, modelMetrics = fetch_and_process_model_info(model_path=model_name)
    add_model_to_dashboard(model_name, themeSummary=themeSummary, themeDetails=themeDetails, modelLevelMetrics=modelMetrics)
    # print(model_name)
    return render_template("dashboard.html", model_name=model_name)



@server.route("/api/themes", methods=["POST"])
def get_themes():
    try:
        data = request.get_json(force=True)  # 🔁 force=True to parse even without header
        # print("Raw JSON from request:", data)

        if not data or "model" not in data:
            return jsonify({"error": "Missing 'model' in request body"}), 400

        model_name = data["model"]

        dashboard_data = read_dashboard_json()
        model_data = dashboard_data.get(model_name)

        if not model_data:
            return jsonify({"error": f"No data found for model '{model_name}'"}), 404

        summary = model_data.get("Theme Summary", [])
        return jsonify(summary)

    except Exception as e:
        print("Exception:", str(e))
        return jsonify({"error": "Invalid JSON or server error"}), 400



@server.route("/text-info", methods=["POST"])
def text_info():
    
    data = request.get_json()
    text = data.get("text", "").strip()
    text_id = data.get("id", "").strip()
    model_name = data.get("model", "").strip()

    if not text or not model_name:
        return jsonify({"error": "Both 'text' and 'model' are required."}), 400

    dashboard_data = read_dashboard_json()

    thetas = get_thetas_by_doc_ids(text_id, model_name)

    textData = format_theta_output_dict(thetas, dashboard_data[model_name]["Theme Summary"])

    return jsonify(textData[text_id])


@server.route("/api/theme/<theme_id>", methods=["GET"])
def get_theme_detail(theme_id):
    model_name = request.args.get("model")
    if not model_name:
        return jsonify({"error": "Missing model name"}), 400

    dashboard = read_dashboard_json()
    model_data = dashboard.get(model_name)

    if not model_data:
        return jsonify({"error": f"No model found for '{model_name}'"}), 404

    theme_details = model_data.get("Theme Details", {})
    theme = theme_details.get(theme_id)

    if not theme:
        return jsonify({"error": f"No theme '{theme_id}' found for model '{model_name}'"}), 404

    return jsonify(theme)

@server.route("/infer-text", methods=["POST"])
def infer_text():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        text_id = data.get("id", "").strip()
        model_name = data.get("model", "").strip()

        if not text or not model_name:
            return jsonify({"error": "Both 'text' and 'model' are required."}), 400

        model_path = "models/" + model_name

        # Step 1: Call inference API
        infer_payload = {
            "config_path": "static/config/config.yaml",
            "model_path": model_path,
            "id_col": text_id,
            "text_col": "raw_text",
            "data": [{"id": text_id, "raw_text": text}]
        }

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }

        infer_response = requests.post(inferurl, json=infer_payload, headers=headers)
        infer_response.raise_for_status()
        inference_result = infer_response.json()
        dashboard_data = read_dashboard_json()

        model_data = dashboard_data.get(model_name)
        if not model_data:
            return jsonify({"error": f"No data found for model '{model_name}'"}), 404

        theme_summary = model_data.get("Theme Summary", [])

        # ✅ Build lookup maps
        theme_map = {
            item["id"]: {
                "label": item.get("label", item["id"]),
                "keywords": item.get("keywords", "")
            }
            for item in theme_summary
        }

        # ✅ Process inference results
        theta = inference_result["thetas"][0]["id"]
        all_themes = [
            {
                "theme_id": tid,
                "label": theme_map.get(tid, {}).get("label", tid),
                "score": round(score, 4),
                "keywords": theme_map.get(tid, {}).get("keywords", "")
            }
            for tid, score in theta.items()
        ]

        # Sort and pick top theme
        all_themes.sort(key=lambda x: x["score"], reverse=True)
        top_theme = all_themes[0]["label"]

        return jsonify({
            "theme": top_theme[:5],
            "top_themes": all_themes[:5]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@server.route('/api/documents', methods = ["POST"])
def get_documents():

    data = request.get_json()
    modelName = data.get("model", "").strip()
    data = read_dashboard_json()
    allDocuments = [
    {
        **doc,
        "theme": details.get("label", theme),
        "model": modelName
    }
    for theme, details in data[modelName]["Theme Details"].items()
    for doc in details.get("documents", [])
]
    # random.shuffle(allDocuments)

    return jsonify(allDocuments)





@server.route("/api/model-info", methods=["POST"])
def model_info():
    data = request.get_json()
    model_name = data.get("model_name", "")
    print(f"Requested model_name: {model_name}")

    if not model_name:
        return jsonify({"status": "error", "message": "Missing model_name"}), 400

    try:
        conn = sqlite3.connect("database/mydatabase.db")
        cursor = conn.cursor()

        cursor.execute("""
            SELECT model_id, model_type, num_topic, model_name, trained_on, description, training_params
            FROM model_registry
            WHERE model_name = ?
            LIMIT 1
        """, (model_name,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return jsonify({"status": "error", "message": "Model not found."}), 404

        model_id, model_type, num_topic, model_name, trained_on, description, training_params_json = row

        # Format training date
        try:
            trained_on_dt = datetime.fromisoformat(trained_on).replace(tzinfo=ZoneInfo("UTC"))
            trained_on_local = trained_on_dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M %p %Z")
        except Exception:
            trained_on_local = trained_on

        # Get corpus names
        conn = sqlite3.connect("database/mydatabase.db")
        cursor = conn.cursor()
        cursor.execute("SELECT corpus_name FROM model_corpus_map WHERE model_id = ?", (model_id,))
        corpus_rows = cursor.fetchall()
        conn.close()

        corpus_names = ", ".join(row[0] for row in corpus_rows)

        try:
            training_params = json.loads(training_params_json) if training_params_json else {}
        except Exception:
            training_params = {}

        model = {
            "model_id": model_id,
            "document": description,
            "model_type": model_type,
            "model_name": model_name,
            "num_topics": num_topic,
            "corpus_names": corpus_names,
            "trained_on": trained_on_local,
            "training_params": training_params
        }

        return jsonify(model)

    except Exception as e:
        print("❌ Error in /api/model-info:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500



@server.route('/api/theme-metrics', methods=["POST", "GET"])
def get_theme_metrics():
    data = request.get_json()
    # print(data)
    model_name = data.get("model", "")
    dashboard_data = read_dashboard_json()
    model_metrics = dashboard_data[model_name]["Model-Level Metrics"]
    converted_metrics = {
    "metrics": []
    }

    for key, value in model_metrics.items():
        metric_entry = {
            "label": key,
            "value": value,
            "id": key.lower()
                    .replace("(", "")
                    .replace(")", "")
                    .replace(" ", "_")
                    .replace("-", "_")
        }
        converted_metrics["metrics"].append(metric_entry)
    return jsonify(converted_metrics)





@server.route("/api/diagnostics", methods=["GET"])
def get_diagnostics():
    model_name = request.args.get("model")
    if not model_name:
        return jsonify({"error": "Missing model name"}), 400

    dashboard = read_dashboard_json()
    theme_details = dashboard.get(model_name, {}).get("Theme Details", {})

    metrics = extract_metrics_from_theme_details(theme_details)
    return jsonify(metrics)




@server.route("/api/theme-coordinates", methods=["POST"])
def get_theme_coordinates():
    data = request.get_json()
    model_name = data.get("model")

    if not model_name:
        return abort(400, description="Missing 'model' in request body")

    dashboard_data = read_dashboard_json()
    model_data = dashboard_data.get(model_name)

    if not model_data:
        return abort(404, description=f"No data found for model: {model_name}")

    theme_entries = model_data.get("Theme Details", {})

    theme_coords = []
    for key, entry in theme_entries.items():
        coords = entry.get("Coordinates", [None, None])
        theme_coords.append({
            "id": entry.get("topic_id") or entry.get("id"),
            "label": entry.get("label") or entry.get("theme"),
            "size": float(entry.get("size")[:-1]) or float(entry.get("prevalence")[:-1]),
            "x": coords[0],
            "y": coords[1],
            "keywords":entry.get("keywords")
        })

        # print(theme_coords)

    return jsonify(theme_coords)


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
        # print(topic_info)
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
    # print(models)

    return jsonify(models)







if __name__ == '__main__':
    server.run(debug=True)