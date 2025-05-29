# 🧠 TOVA: Topic Visualization & Analysis

**TOVA** is an interactive, end-to-end platform for working with topic models. It provides:
- ✅ A **command-line interface** for training and inferring models
- ✅ A **REST API** for integrating into apps or pipelines
- ✅ A sleek **web dashboard** for visualizing and exploring topics, keywords, documents, and trends

Whether you're a researcher, data scientist, or a curious explorer, TOVA helps you understand the hidden structure of large collections of text.

---

## 🚀 Features

- 🔍 Train traditional or LLM-based topic models
- 🧩 Ingest CSV, Excel, or JSON files with custom columns
- 📊 Visualize themes by prevalence, coherence, and keyword uniqueness
- 📝 Run single or batch document inference
- 📈 Track themes over time with trend graphs
- ⚡ Modular architecture (CLI/API/UI) that shares one backend core

---

## 🛠️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/tova.git
cd tova
```


### Install Python Dependencies in your virtual environment
```bash
    pip install -r requirements.txt
```
### Start the Both Servers - Backend API and Flaskapp

```bash
    python run.py
```
Then visit  http://localhost:5000 to access the Web interface. You can visit http://localhost:8989 for API instructions