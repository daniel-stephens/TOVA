import pandas as pd
import plotly.express as px
import chromadb
import json


client = chromadb.PersistentClient(path="database/myDB")
# collection = client.get_or_create_collection(name="documents")
collect = client.get_or_create_collection(name="doc")

# Assuming you've already fetched everything
results = collect.get(include=["documents", "metadatas"])

# Create a DataFrame
df = pd.DataFrame({
    "id": results["ids"],
    "document": results["documents"],
    "topic_id": [meta.get("topic_id", -1) for meta in results["metadatas"]],
    
})

# Group by topic
topic_counts = df["topic_id"].value_counts().reset_index()
topic_counts.columns = ["topic_id", "document_count"]
finaldf = topic_counts.sort_values(by="document_count", ascending=False)

# Convert topic_id to string (optional, for cleaner x-axis labels)
finaldf["topic_id"] = finaldf["topic_id"].astype(str)

# Sort topic_id axis based on document_count
finaldf["topic_id"] = pd.Categorical(
    finaldf["topic_id"],
    categories=finaldf["topic_id"],
    ordered=True
)

# print(topic_counts)
# Plot
fig = px.bar(
    finaldf,
    x="topic_id",
    y="document_count",
    title="Document Count by Topic ID",
)
fig.show()
