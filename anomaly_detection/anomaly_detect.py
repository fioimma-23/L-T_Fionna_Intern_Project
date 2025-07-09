import pandas as pd
import numpy as np
from pyod.models.iforest import IForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

file_path = r"D:\intern\l&t\anomaly_detection\log2.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_csv(file_path)
print("Loaded Columns:", df.columns.tolist())
print(df.head())

numeric_df = df.select_dtypes(include=["float64", "int64"]).dropna().copy()
numeric_df["row_index"] = numeric_df.index

scaler = StandardScaler()
X = scaler.fit_transform(numeric_df.drop("row_index", axis=1))

clf = IForest(contamination=0.1, random_state=42)
clf.fit(X)

labels = clf.labels_
scores = clf.decision_scores_

numeric_df["anomaly"] = labels
numeric_df["score"] = scores

for col in numeric_df.columns:
    if col not in ["anomaly", "score", "row_index"]:
        if numeric_df[col].dtype in ["float64", "int64"]:
            mean = numeric_df[col].mean()
            numeric_df[f"{col}_diff"] = abs(numeric_df[col] - mean)

anomalies = numeric_df[numeric_df["anomaly"] == 1].copy()

diff_cols = [col for col in anomalies.columns if col.endswith("_diff")]
diff_data = anomalies[diff_cols]

scaler_cluster = StandardScaler()
X_cluster = scaler_cluster.fit_transform(diff_data)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
anomalies["anomaly_type"] = kmeans.fit_predict(X_cluster)

cluster_labels = {
    0: "High Byte Transfer",
    1: "Port Spike",
    2: "Long Duration"
}

anomalies["anomaly_label"] = anomalies["anomaly_type"].map(cluster_labels)

output_path = "anomalies_detected.csv"
anomalies.to_csv(output_path, index=False)

print(f"\nTotal anomalies detected: {len(anomalies)}")
print(anomalies[["row_index", "score", "anomaly_type", "anomaly_label"] + diff_cols].head())
print(f"Anomalies saved to: {output_path}")

plt.figure(figsize=(10, 6))
anomalous_points = X[numeric_df["anomaly"] == 1]
plt.scatter(anomalous_points[:, 0], anomalous_points[:, 1], c="red", edgecolor="k", label="Anomaly")
plt.legend()
plt.title("All Detected Anomalies")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.tight_layout()
plt.show()
