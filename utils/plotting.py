import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

def generate_color_map(unique_topics):
    palette = px.colors.qualitative.Plotly
    return {tid: palette[i % len(palette)] for i, tid in enumerate(unique_topics)}

def generate_bar_chart(chart_data, topic_colors):
    fig = px.bar(
        chart_data,
        x="topic_id",
        y="document_count",
        title="📈 Document Count by Topic ID",
        text_auto=True,
        color="topic_id",
        color_discrete_map=topic_colors
    )
    fig.update_layout(showlegend=False)
    return fig

def generate_pca_plot(topic_distributions, topic_ids, doc_ids, topic_colors):
    coords = PCA(n_components=2).fit_transform(topic_distributions)
    df_pca = pd.DataFrame(coords, columns=["PC1", "PC2"])
    df_pca["topic_id"] = topic_ids
    df_pca["doc_id"] = doc_ids
    df_pca["hover"] = "Doc ID: " + df_pca["doc_id"]
    
    fig = px.scatter(
        df_pca,
        x="PC1",
        y="PC2",
        color="topic_id",
        title="📌 PCA: Document Distribution by Topic",
        hover_name="hover",
        custom_data=[df_pca["doc_id"]],
        color_discrete_map=topic_colors,
        opacity=0.7
    )
    return fig
