# TOVA Version 1

This document describes the **Version 1** scope of TOVA (Topic Visualization & Analysis).

## Version 1 scope

The current release is **Version 1**. It includes:

- **Traditional topic models**: Training, inference, and exploration of classical topic models (e.g. Tomotopy LDA, CTM) via CLI, REST API, and web UI.
- **Web dashboard**: Corpus and dataset management, training workflows, model exploration (topics, documents, coherence, similar topics), inference on new text, and optional LLM-based topic labelling/summarization for traditional models.
- **Deployment**: CLI, Python package, and Docker-based stack (API, Web UI, Solr, Postgres).

## Not included in Version 1

The following are **not** part of Version 1 and are planned for a later release:

1. **Implementation of the LLM-based topic models**  
   Topic models that are fully driven by LLMs (e.g. TopicGPT, OpenTopicRAG) are not yet implemented in the active model registry or training/inference pipelines. Configuration and code structure exist for them; they are not part of the Version 1 feature set.

2. **The chatbot for the topic models**  
   A dedicated chatbot experience for interacting with and querying topic models is not part of Version 1. Chat-related UI/API or database support may exist as groundwork for a future release but are not considered part of the Version 1 deliverables.

---

For a detailed list of what *is* included, see [FEATURES.md](FEATURES.md).
