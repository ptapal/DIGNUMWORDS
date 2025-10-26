import os
import pandas as pd
import numpy as np

results_dir = "umap_hdbscan_global"

all_stats = []

for fname in os.listdir(results_dir):
    if not fname.endswith(".csv"):
        continue
    fpath = os.path.join(results_dir, fname)
    df = pd.read_csv(fpath)

    if 'cluster' not in df.columns or 'condition' not in df.columns:
        print(f"Skipping {fname} — missing columns.")
        continue

    # Define parity (1 for Par, 0 for NonPar)
    df['parity'] = df['condition'].apply(lambda c: 1 if 'Par' in str(c) else 0)

    # Compute LOSO-style cluster-weighted purity accuracy
    acc = 0
    for c in np.unique(df['cluster']):
        if c == -1:  # skip noise
            continue
        mask = df['cluster'] == c
        maj = np.round(df.loc[mask, 'parity'].mean())  # majority label
        acc += (df.loc[mask, 'parity'] == maj).sum()
    acc /= len(df)

    # Fraction of noise points
    frac_noise = (df['cluster'] == -1).mean()

    # Number of valid clusters
    num_clusters = (df['cluster'].nunique() - (1 if -1 in df['cluster'].unique() else 0))

    # Parse metadata from filename
    parts = fname.replace('.csv', '').split('_')
    category = parts[0]
    font = parts[1] if len(parts) > 1 else 'None'
    specific_font = parts[2] if len(parts) > 2 else 'None'

    all_stats.append({
        'category': category,
        'font': font,
        'specific_font': specific_font,
        'mean_cluster_purity': acc,
        'num_clusters': num_clusters,
        'frac_noise': frac_noise
    })

# Save summary
results_df = pd.DataFrame(all_stats)
print("\nGlobal UMAP + HDBSCAN summary (LOSO-style accuracy):")
print(results_df)

results_df.to_csv("global_umap_hdbscan_summary_LOSOstyle.csv", index=False)
print("\nSaved summary to global_umap_hdbscan_summary_LOSOstyle.csv")
