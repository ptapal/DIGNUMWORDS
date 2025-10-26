import os
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.utils import resample
from scipy.stats import entropy
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
results_dir = "umap_hdbscan_oddeven_parity"   # folder with *_clusters.csv
output_summary = "unsup_parity_cluster_quality_summary_augmented.csv"
max_samples = 5000   # max points for silhouette/CH/DB metrics

# ---------------- HELPER FUNCTIONS ----------------
def compute_intrinsic_metrics(df, max_samples=max_samples):
    """Compute clustering quality metrics with optional subsampling."""
    X = df[['umap_1', 'umap_2']].values
    labels = df['cluster'].values
    mask = labels != -1
    X_valid, labels_valid = X[mask], labels[mask]

    if len(np.unique(labels_valid)) < 2 or len(labels_valid) < 50:
        return np.nan, np.nan, np.nan

    if len(labels_valid) > max_samples:
        X_valid, labels_valid = resample(X_valid, labels_valid, n_samples=max_samples, random_state=42)

    try:
        sil = silhouette_score(X_valid, labels_valid)
    except:
        sil = np.nan
    try:
        ch = calinski_harabasz_score(X_valid, labels_valid)
    except:
        ch = np.nan
    try:
        db = davies_bouldin_score(X_valid, labels_valid)
    except:
        db = np.nan

    return sil, ch, db


def compute_cluster_entropy(df):
    """Compute Shannon entropy of cluster size distribution (excluding noise)."""
    valid_clusters = df.loc[df['cluster'] != -1, 'cluster']
    counts = valid_clusters.value_counts(normalize=True)
    if len(counts) == 0:
        return np.nan
    return entropy(counts, base=2)


def qualitative_verdict(sil, ch, db, entropy_val, frac_noise):
    """Assign qualitative verdicts to clustering."""
    if np.isnan(sil):
        return "invalid / degenerate"
    if frac_noise > 0.5 and entropy_val > 6:
        return "highly fragmented / noisy"
    elif sil > 0.3 or ch > 1000:
        return "structured / emerging separation"
    elif sil > 0.5 and db < 1:
        return "very cohesive / distinct"
    else:
        return "ambiguous structure"

# ---------------- MAIN ----------------
all_stats = []

for fname in os.listdir(results_dir):
    if not fname.endswith("_clusters.csv"):
        continue

    fpath = os.path.join(results_dir, fname)
    # load only needed columns
    try:
        df = pd.read_csv(fpath, usecols=['umap_1','umap_2','cluster'])
    except Exception as e:
        print(f"Skipping {fname} — could not read: {e}")
        continue

    frac_noise = (df['cluster'] == -1).mean()
    n_clusters = len(np.unique(df['cluster'])) - (1 if -1 in df['cluster'].values else 0)
    sil, ch, db = compute_intrinsic_metrics(df)
    entropy_val = compute_cluster_entropy(df)

    parts = fname.replace("_clusters.csv", "").split("_")
    category = parts[0]
    font = parts[1] if len(parts) > 1 else "unknown"
    specific_font = parts[2] if len(parts) > 2 else "None"

    verdict = qualitative_verdict(sil, ch, db, entropy_val, frac_noise)

    all_stats.append({
        "category": category,
        "font": font,
        "specific_font": specific_font,
        "num_clusters": n_clusters,
        "frac_noise": frac_noise,
        "cluster_entropy": entropy_val,
        "silhouette_score": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "verdict": verdict
    })

# ---------------- SAVE SUMMARY ----------------
summary_df = pd.DataFrame(all_stats)

# compute normalized entropy per category
summary_df['entropy_norm'] = summary_df.groupby('category')['cluster_entropy'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

summary_df.to_csv(output_summary, index=False)
print("\n✅ Saved augmented intrinsic cluster quality summary to", output_summary)
print(summary_df)