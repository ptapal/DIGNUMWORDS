import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import hdbscan
import gc

# ------------------ 1. Data loader ------------------
def load_subject_windows_unlabeled(h5_file, subject_key, category=None, specific_font=None, sample_per_seq=200):
    """Yield X and meta for one subject at a time (fully unsupervised)."""
    with h5py.File(h5_file, 'r') as h5f:
        subj_group = h5f[subject_key]
        for category_key in subj_group.keys():
            if category and category_key != category: 
                continue
            for font_key in subj_group[category_key].keys():
                for spec_key in subj_group[category_key][font_key].keys():
                    if specific_font and spec_key != specific_font: 
                        continue
                    seq_group = subj_group[category_key][font_key][spec_key]
                    for cond_key, cond in seq_group.items():
                        '''
                        if 'Par' not in cond_key:
                            continue
                        '''
                        for seq_name, ds in cond.items():
                            if isinstance(ds, h5py.Dataset):
                                X = ds[:]
                            else:
                                first = next(iter(ds.values()))
                                X = first[:]
                            if len(X) > sample_per_seq:
                                idx = np.random.choice(len(X), sample_per_seq, replace=False)
                                X = X[idx]
                            meta = pd.DataFrame({
                                'subject':[subject_key]*X.shape[0],
                                'category':[category_key]*X.shape[0],
                                'font':[font_key]*X.shape[0],
                                'specific_font':[spec_key]*X.shape[0],
                                'condition':[cond_key]*X.shape[0],
                                'window_name':[seq_name]*X.shape[0]
                            })
                            yield X, meta

# ------------------ 2. Global UMAP + HDBSCAN ------------------
def global_umap_hdbscan_with_odd_even(h5_file, subjects, category=None, font_name=None, specific_font=None,
                                      output_dir="umap_hdbscan_global", n_neighbors=15, min_dist=0.1,
                                      min_cluster_size=10, sample_per_seq=200):
    os.makedirs(output_dir, exist_ok=True)
    gc.collect()

    # ---- Collect all data ----
    print(f"Collecting data from all subjects for {category}...")
    X_all_list, meta_all_list = [], []
    for subj in subjects:
        for X, meta in load_subject_windows_unlabeled(h5_file, subj, category, specific_font, sample_per_seq):
            X_all_list.append(X)
            meta_all_list.append(meta)
    X_all = np.vstack(X_all_list)
    meta_all = pd.concat(meta_all_list, ignore_index=True)
    del X_all_list, meta_all_list; gc.collect()

    # ---- Scale and embed ----
    print("Scaling and embedding with UMAP...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42, n_components=2, n_jobs=-1)
    X_emb = reducer.fit_transform(X_scaled)
    meta_all['umap_1'] = X_emb[:,0]
    meta_all['umap_2'] = X_emb[:,1]
    del X_all, X_scaled, X_emb; gc.collect()

    # ---- HDBSCAN clustering ----
    print("Running HDBSCAN on embedded data...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(meta_all[['umap_1','umap_2']])
    meta_all['cluster'] = cluster_labels

    # ---- Compute odd/even fractions per cluster ----
    cluster_logs = []
    for c in np.unique(cluster_labels):
        cluster_points = meta_all[meta_all['cluster'] == c]
        size = len(cluster_points)
        frac_noise = int(c == -1)
        cluster_logs.append({
            'cluster': c,
            'size': size,
            'noise': frac_noise,
        })
    cluster_log_df = pd.DataFrame(cluster_logs)

    # ---- Save results ----
    out_csv = f"{category}_{font_name or 'all'}_{specific_font or 'all'}_clusters.csv"
    meta_all.to_csv(os.path.join(output_dir, out_csv), index=False)
    cluster_log_df.to_csv(os.path.join(output_dir, f"{category}_{font_name or 'all'}_{specific_font or 'all'}_cluster_log.csv"), index=False)

    # ---- Plot clusters ----
    plt.figure(figsize=(8,6))
    plt.scatter(meta_all['umap_1'], meta_all['umap_2'], c=cluster_labels, cmap='tab20', alpha=0.7, s=20)
    plt.title(f"{category} - {font_name or 'all'} - {specific_font or 'all'}")
    plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2')
    plt.colorbar(label='Cluster')
    plt.savefig(os.path.join(output_dir, f"{category}_{font_name or 'all'}_{specific_font or 'all'}_clusters.png"),
                dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved clustered CSV and plot to {output_dir}")
    return meta_all, cluster_log_df

# ------------------ 3. Run over categories/fonts ------------------
fonts = ['1F', '20F']
categories = ['NumWo', 'Dig']
fixed_subjects = [
    'S02','S03','S04','S05','S07','S08','S09','S10','S11','S12',
    'S13','S15','S16','S18','S19','S20','S21','S22','S23','S24',
    'S25','S26','S28'
]
h5_file = 'h5_freq/features_windowed_unlabeled_experimental.h5'

for category in categories:
    for font in fonts:
        specific_fonts = [None] if font=='1F' else ['A','S']
        for spec_font in specific_fonts:
            try:
                print(f"Running {category}, font={font}, specific_font={spec_font}")
                meta_all, cluster_log_df = global_umap_hdbscan_with_odd_even(
                    h5_file, fixed_subjects, category=category, font_name=font,
                    specific_font=spec_font, output_dir="umap_hdbscan_oddeven",
                    sample_per_seq=500, n_neighbors=15, min_dist=0.1, min_cluster_size=10
                )
            except Exception as e:
                print(f"Error with {category}, font={font}, specific_font={spec_font}: {e}")
                continue
