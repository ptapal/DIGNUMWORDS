import os
import pandas as pd

# -------------------- Config --------------------
results_dir = "umap_hdbscan_figs"  # folder with per-subject CSVs (UMAP+HDBSCAN output)
summary_files = [f for f in os.listdir('.') if f.startswith('umap_hdbscan_results_') and f.endswith('.csv')]

# -------------------- 1. Load summary accuracy CSVs --------------------
all_results = []
for file in summary_files:
    df = pd.read_csv(file)
    df['source_file'] = file
    all_results.append(df)

results_df = pd.concat(all_results, ignore_index=True)
results_df['specific_font'] = results_df['specific_font'].replace({None:'None'})

# -------------------- 2. Compute mean and std accuracy by group --------------------
grouped_acc = results_df.groupby(['category', 'font', 'specific_font'])['parity_vs_control'].agg(['mean', 'std']).reset_index()

print("\nMean and Std Accuracy per category/font/specific_font:")
print(grouped_acc)

# -------------------- 3. Optional: per-subject summary --------------------
subject_acc = results_df.groupby(['subject', 'category', 'font', 'specific_font'])['parity_vs_control'].mean().reset_index()
subject_acc.to_csv("per_subject_accuracy_summary.csv", index=False)
print("\nSaved per-subject accuracy summary to 'per_subject_accuracy_summary.csv'")

# -------------------- 4. Optional: cluster purity from per-subject CSVs --------------------
# Only meaningful if you still have the 'parity' column in the per-subject CSVs
all_meta = []
for file in os.listdir(results_dir):
    if file.endswith(".csv"):
        meta = pd.read_csv(os.path.join(results_dir, file))
        # Extract category/font/subject info from filename
        parts = file.replace('.csv','').split('_')
        if len(parts) >= 5:  # e.g., Dig_subj-S02_font_specific.csv
            meta['category'] = parts[0]
            meta['subject'] = parts[2]
            meta['font'] = parts[3]
            meta['specific_font'] = parts[4]
        all_meta.append(meta)

meta_df = pd.concat(all_meta, ignore_index=True)

# Compute cluster purity (mean parity per cluster)
cluster_purity = meta_df.groupby('cluster')['parity'].mean().reset_index()
cluster_counts = meta_df['cluster'].value_counts().reset_index()
cluster_counts.columns = ['cluster', 'count']

cluster_summary = cluster_purity.merge(cluster_counts, on='cluster')
print("\nCluster Purity and Sizes:")
print(cluster_summary.sort_values('cluster'))

# -------------------- 5. Noise fraction --------------------
noise_fraction = (meta_df['cluster'] == -1).mean()
print(f"\nFraction of points labeled as noise (-1): {noise_fraction:.3f}")
