from flask import Flask, request, jsonify, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import pandas as pd
import json
import fitz
from .utils import *

from werkzeug.utils import secure_filename
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

# Configure Upload Folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER']
    )

# Ensure preprocess_file function is imported or defined earlier

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file0' not in request.files and not request.is_json:
        return jsonify({"error": "No files uploaded"}), 400

    uploaded_files = request.files.to_dict()
    saved_files = []

    if request.is_json:
        try:
            pdf_data = request.json.get("pdf_data", [])
            if not pdf_data:
                return jsonify({"error": "No extracted PDF text received"}), 400

            pdf_json_filename = os.path.join(app.config['UPLOAD_FOLDER'], "extracted_pdfs.json")

            with open(pdf_json_filename, 'w', encoding='utf-8') as json_file:
                json.dump(pdf_data, json_file, indent=2, ensure_ascii=False)

            # processed_df = preprocess_file(pdf_json_filename, 'json')
            # processed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{filename.split('.')[0]}.csv")
            # print(processed_df)
            # processed_df.to_csv(processed_file_path, index=False)
            # saved_files.append(processed_file_path)

            

            return jsonify({"message": "PDF text successfully saved as JSON.", "file": pdf_json_filename}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to save extracted PDF text: {str(e)}"}), 500

    for key, f in uploaded_files.items():
        if f.filename == '':
            continue

        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)

        try:
            file_ext = filename.split('.')[-1].lower()

            if file_ext in ["csv", "json", "jsonl", "xls", "xlsx"]:
                processed_df = preprocess_file(file_path, file_ext)
                processed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{filename.split('.')[0]}.csv")
                processed_df.to_csv(processed_file_path, index=False)
                saved_files.append(processed_file_path)
                print(f"Processed and saved: {processed_file_path}")

            elif file_ext == "pdf":
                extracted_text = extract_text_from_pdf(file_path)
                pdf_json_filename = file_path.replace(".pdf", ".json")
                pdf_data = [{"Document Number": i+1, "Content": text} for i, text in enumerate(extracted_text)]

                with open(pdf_json_filename, 'w', encoding='utf-8') as json_file:
                    json.dump(pdf_data, json_file, indent=2, ensure_ascii=False)

                processed_df = preprocess_file(pdf_json_filename, 'json')
                processed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{filename.split('.')[0]}.csv")
                print(processed_df)
                processed_df.to_csv(processed_file_path, index=False)
                saved_files.append(processed_file_path)

        except Exception as e:
            return jsonify({"error": f"Failed to process {filename}: {str(e)}"}), 500

    if not saved_files:
        return jsonify({"error": "No valid files uploaded"}), 400

    return jsonify({"message": f"{len(saved_files)} file(s) uploaded and processed successfully!", "files": saved_files}), 200
 

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file0' not in request.files and not request.is_json:
#         return jsonify({"error": "No files uploaded"}), 400

#     uploaded_files = request.files.to_dict()
#     saved_files = []

#     # Handle extracted PDF text JSON separately
#     if request.is_json:
#         try:
#             pdf_data = request.json.get("pdf_data", [])
#             if not pdf_data:
#                 return jsonify({"error": "No extracted PDF text received"}), 400

#             pdf_json_filename = os.path.join(app.config['UPLOAD_FOLDER'], "extracted_pdfs.json")

#             # Save the extracted PDF text JSON
#             with open(pdf_json_filename, 'w', encoding='utf-8') as json_file:
#                 json.dump(pdf_data, json_file, indent=2, ensure_ascii=False)

#             return jsonify({"message": "PDF text successfully saved as JSON.", "file": pdf_json_filename}), 200
#         except Exception as e:
#             return jsonify({"error": f"Failed to save extracted PDF text: {str(e)}"}), 500

    # # Process file uploads
    # for key, f in uploaded_files.items():
    #     if f.filename == '':
    #         continue  # Skip empty file inputs

    #     filename = secure_filename(f.filename)
    #     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #     f.save(file_path)
    #     saved_files.append(file_path)

    #     try:
    #         file_ext = filename.split('.')[-1].lower()

    #         if file_ext == "csv":
    #             df = pd.read_csv(file_path)
    #             print(f"\nUploaded CSV: {filename}")
    #             print(df.head())  # Print first 5 rows

    #         elif file_ext == "json":
    #             with open(file_path, 'r', encoding='utf-8') as json_file:
    #                 data = json.load(json_file)
    #                 print(f"\nUploaded JSON: {filename}")
    #                 print(json.dumps(data[:3], indent=2))  # Print first 3 items

    #         elif file_ext == "jsonl":
    #             with open(file_path, 'r', encoding='utf-8') as jsonl_file:
    #                 data = [json.loads(line) for line in jsonl_file]
    #                 print(f"\nUploaded JSONL: {filename}")
    #                 print(json.dumps(data[:3], indent=2))  # Print first 3 lines

    #         elif file_ext in ["xls", "xlsx"]:
    #             df = pd.read_excel(file_path)
    #             print(f"\nUploaded Excel File: {filename}")
    #             print(df.head())  # Print first 5 rows

    #         elif file_ext == "pdf":
    #             extracted_text = extract_text_from_pdf(file_path)
    #             pdf_json_filename = file_path.replace(".pdf", ".json")

    #             # Save extracted PDF text to JSON
    #             pdf_data = [{"document_number": i+1, "text": text, "category": "PDF"} for i, text in enumerate(extracted_text)]
    #             with open(pdf_json_filename, 'w', encoding='utf-8') as json_file:
    #                 json.dump(pdf_data, json_file, indent=2, ensure_ascii=False)

    #             print(f"\nExtracted text saved to JSON: {pdf_json_filename}")

    #     except Exception as e:
    #         return jsonify({"error": f"Failed to process {filename}: {str(e)}"}), 500

    # if not saved_files:
    #     return jsonify({"error": "No valid files uploaded"}), 400

    # return jsonify({"message": f"{len(saved_files)} file(s) uploaded successfully!", "files": saved_files}), 200

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF (fitz)."""
    text_content = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                text_content.append(f"Page {page_num}: {text}")

        return text_content  # Return list of text per page
    except Exception as e:
        return [f"Error extracting text from PDF: {str(e)}"]
    
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     f = request.files['file']
    
#     if f.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     # Secure the filename
#     data_filename = secure_filename(f.filename)

#     # Save file to uploads folder
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
#     f.save(file_path)

#     # Load the CSV file using pandas
#     try:
#         df = pd.read_csv(file_path)
#         print(f"\nUploaded CSV: {data_filename}")
#         print(df.head())  # Print first 5 rows to terminal
#     except Exception as e:
#         return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 500

#     return jsonify({"message": "File uploaded and loaded successfully!", "filename": data_filename}), 200

 

    # if 'files' not in request.files:
    #     return jsonify({"error": "No file uploaded"}), 400
    
    # files = request.files.getlist('files')  # Get multiple files

    # if not files:
    #     return jsonify({"error": "No file selected"}), 400
    
    # uploaded_files = []
    
    # for file in files:
    #     if file.filename != '':
    #         file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #         file.save(file_path)
    #         uploaded_files.append(file.filename)
            
    #         # Print uploaded file names and content in Flask terminal
    #         print(f"Uploaded: {file.filename}")
    #         with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    #             print(f.read())  # Print file content to terminal

    # return jsonify({"message": "Files uploaded successfully!", "files": "My file"}), 200



@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/load-files')
def load_files():
    return render_template('loadFiles.html')

    

@app.route('/preprocess', methods=['POST'])
def preprocess():
    files = request.files.getlist('files')
    if not files:
        return jsonify({'success': False, 'error': 'No files uploaded'}), 400

    # Simulate file preprocessing
    try:
        for file in files:
            print(f"Processing {file.filename}...")  # Replace with actual preprocessing logic
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return "No file part"
    
#     files = request.files.getlist('file')  # Get multiple files
#     if not files:
#         return "No file selected"

#     for file in files:
#         if file.filename != '':
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    
#     return redirect(url_for('index'))

@app.route('/show_data')
def show_data():
    files = os.listdir(UPLOAD_FOLDER)
    return f"Uploaded Files: {', '.join(files)}"


if __name__ == '__main__':
    app.run(debug=True)





