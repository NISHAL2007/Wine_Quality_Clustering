import gradio as gr
import joblib
import numpy as np

# Load your saved pipeline
scaler, pca, kmeans = joblib.load("wine_clustering_pipeline.joblib")

# List of features expected in the UI
features = [
    "Alcohol", "Malic_Acid", "Ash_Alcanity", "Magnesium", "Total_Phenols",
    "Flavanoids", "Proanthocyanins", "Color_Intensity", "OD280", "Proline"
]

# Cluster descriptions (replace with your summary!)
cluster_desc = [
    "Cluster 0: Lighter wines, with lower alcohol, magnesium, and proline.",
    "Cluster 1: Wines with higher acid and color, but lowest flavor complexity.",
    "Cluster 2: Rich, robust wines with highest alcohol and flavor compounds.",
]

def predict_cluster(*inputs):
    # Convert user inputs to array
    input_array = np.array(inputs).reshape(1, -1)
    # Transform through pipeline
    scaled = scaler.transform(input_array)
    pca_out = pca.transform(scaled)
    cluster = int(kmeans.predict(pca_out)[0])
    return f"Assigned to Cluster {cluster}.\n{cluster_desc[cluster]}"

# Gradio UI input fields
inputs = [gr.Number(label=feat) for feat in features]

# Gradio Interface
app = gr.Interface(
    fn=predict_cluster,
    inputs=inputs,
    outputs="text",
    title="Wine Clustering App",
    description="Enter wine sample features to assign it to a cluster.\nCluster details describe chemical and flavor profile for each group."
)

if __name__ == "__main__":
    app.launch()
