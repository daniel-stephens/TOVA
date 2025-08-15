import dash
from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import chromadb

import os
from sklearn.decomposition import PCA
import numpy as np

# === Helper to list models and datasets ===
def list_models():
    return [f.replace(".model", "") for f in sorted(os.listdir("model")) if f.endswith(".model")]

def list_datasets(model_name):
    return ["documents"]  # For now, static. You can expand this.


def init_dash_app(server):
    # === ChromaDB connection ===
    client = chromadb.PersistentClient(path="database/myDB")

    # Get defaults
    default_model = list_models()[-1] if list_models() else None
    default_dataset = list_datasets(default_model)[0] if default_model else None

    # === Initialize App === #
    app = Dash(__name__, server=server, url_base_pathname="/dashboard/", external_stylesheets=[dbc.themes.BOOTSTRAP])


    header_style = {"padding": "10px", "textAlign": "center", "marginBottom": "10px"}
    table_style_header_base = {"fontWeight": "bold"}

    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("📊 Document Topic Dashboard", id="header-title", style=header_style), width=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col(dcc.Dropdown(
                        id="model-selector",
                        options=[{"label": name, "value": name} for name in list_models()],
                        placeholder="Select Model",
                        value=default_model
                    ), width=6),
                    dbc.Col(dcc.Dropdown(
                        id="dataset-selector",
                        placeholder="Select Dataset",
                        value=default_dataset
                    ), width=6)
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col(dbc.Input(id="search-input", placeholder="Search document text...", type="text"), width=8),
                    dbc.Col(dbc.Button("Reset Filters", id="reset-button", color="secondary", className="w-100"), width=4)
                ], className="mb-3 g-2"),

                dbc.Row([
                    dbc.Col(dcc.Graph(id="topic-bar-chart", style={"height": "320px"}), width=12)
                ])
            ], width=6),

            dbc.Col([
                dbc.Row([
                    dbc.Col(dcc.Graph(id="pca-plot", style={"height": "100%"}), width=12)
                ])
            ], width=6)
        ]),

        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id="document-table",
                    columns=[
                        {"name": "Topic ID", "id": "topic_id"},
                        {"name": "Preview", "id": "preview"},
                        {"name": "Score", "id": "topic_score"}
                    ],
                    page_size=8,
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "maxWidth": "280px",
                        "whiteSpace": "nowrap",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "cursor": "pointer",
                        "fontSize": "13px"
                    },
                    row_selectable="single",
                    sort_action="native",
                    filter_action="native"
                )
            ], width=6),

            dbc.Col([
                html.H5("📝 Full Document Viewer", className="my-3"),
                html.Div(id="full-text-display", style={
                    "whiteSpace": "pre-wrap",
                    "border": "1px solid #ccc",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "backgroundColor": "#f8f9fa",
                    "height": "80%",
                    "overflowY": "auto",
                    "fontSize": "14px"
                })
            ], width=6)
        ])
    ], fluid=True)

    # === Global Cache for Data ===
    df_cache = pd.DataFrame()
    topic_colors = {}

    # === Callbacks === #
    @app.callback(
        Output("dataset-selector", "options"),
        Input("model-selector", "value")
    )
    def update_datasets(model):
        if model:
            datasets = list_datasets(model)
            return [{"label": ds, "value": ds} for ds in datasets]
        return []

    @app.callback(
        Output("topic-bar-chart", "figure"),
        Output("document-table", "data"),
        Output("search-input", "value"),
        Output("pca-plot", "figure"),
        Output("header-title", "style"),
        Output("document-table", "style_header"),
        Output("document-table", "style_data"),
        Input("search-input", "value"),
        Input("model-selector", "value"),
        Input("dataset-selector", "value"),
        Input("reset-button", "n_clicks"),
        Input("topic-bar-chart", "clickData")
    )
    def load_model_dashboard(search_value, model_name, dataset_name, reset_clicks, bar_click):
        global df_cache, topic_colors
        ctx = dash.callback_context

        if not model_name or not dataset_name:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        lda_model = LDAAdapter.load(f"model/{model_name}")
        collection = client.get_or_create_collection(name=dataset_name)
        results = collection.get(include=["documents", "metadatas"])
        documents = results["documents"]

        tokenized_docs = [doc.split() for doc in documents]
        bows = [lda_model.dictionary.doc2bow(doc) for doc in tokenized_docs]
        lda_model.documents = tokenized_docs
        lda_model.corpus = bows
        doc_topics = lda_model.get_document_topics()

        records = []
        topic_distributions = []
        topic_ids = []

        for doc_id, topic_info in doc_topics.items():
            doc_text = documents[doc_id]
            distribution = topic_info["topic_distribution"]
            topic_distributions.append(list(distribution.values()))
            topic_ids.append(str(topic_info["topic_id"]))
            records.append({
                "doc_id": str(doc_id),
                "topic_id": str(topic_info["topic_id"]),
                "topic_score": round(topic_info["topic_score"], 3),
                "document": doc_text,
                "preview": doc_text[:60] + "..."
            })

        df = pd.DataFrame(records).dropna(subset=["topic_id"])
        df_cache = df.copy()

        triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else None
        active_topic = None

        if triggered == "topic-bar-chart.clickData" and bar_click:
            active_topic = bar_click["points"][0]["x"]
            df = df[df["topic_id"] == active_topic]
        elif triggered == "reset-button.n_clicks":
            search_value = ""

        if search_value:
            df = df[df["document"].str.contains(search_value, case=False, na=False)]

        chart_data = df_cache.groupby("topic_id").size().reset_index(name="document_count")
        chart_data = chart_data.sort_values(by="document_count", ascending=False)

        unique_topics = sorted(chart_data["topic_id"].unique())
        palette = px.colors.qualitative.Plotly
        topic_colors = {tid: palette[i % len(palette)] for i, tid in enumerate(unique_topics)}

        fig_bar = px.bar(
            chart_data,
            x="topic_id",
            y="document_count",
            title="📈 Document Count by Topic ID",
            text_auto=True,
            color="topic_id",
            color_discrete_map=topic_colors
        )
        fig_bar.update_layout(showlegend=False)

        topic_distributions = np.array(topic_distributions)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(topic_distributions)
        pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
        pca_df["topic_id"] = topic_ids
        pca_df["doc_id"] = df_cache["doc_id"]
        pca_df["hover"] = "Doc ID: " + df_cache["doc_id"]

        fig_pca = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="topic_id",
            title="📌 PCA: Document Distribution by Topic",
            hover_name="hover",
            custom_data=[pca_df["doc_id"]],
            color_discrete_map=topic_colors,
            opacity=0.7
        )

        selected_color = topic_colors.get(active_topic, None)
        header_style = {
            "padding": "10px",
            "textAlign": "center",
            "marginBottom": "10px",
            "color": selected_color or "black"
        }

        style_header = table_style_header_base.copy()
        if selected_color:
            style_header["backgroundColor"] = selected_color

        style_data = {}
        if selected_color:
            style_data["backgroundColor"] = selected_color + "20"

        return fig_bar, df.to_dict("records"), search_value or "", fig_pca, header_style, style_header, style_data

    @app.callback(
        Output("full-text-display", "children"),
        Input("document-table", "selected_rows"),
        Input("pca-plot", "clickData"),
        State("document-table", "data")
    )
    
    
    def display_full_text(selected_rows, pca_click, table_data):
        ctx = dash.callback_context

        if not table_data:
            return "Data not loaded."

        if ctx.triggered and ctx.triggered[0]["prop_id"] == "pca-plot.clickData" and pca_click:
            doc_id = str(pca_click["points"][0]["customdata"][0])
            # Safely match the document ID from available table_data
            for row in table_data:
                if str(row.get("doc_id")) == doc_id:
                    return row.get("document", "Document not found.")
            return "Document not found."

        if selected_rows:
            return table_data[selected_rows[0]].get("document", "Document not found.")

        return "Click a row or PCA point to see the full document."

    
    return app


# # === Run the app === #
# if __name__ == '__main__':
#     app.run(debug=True)