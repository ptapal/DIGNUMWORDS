import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import hdbscan
import gc

def load_subject_windows_generator(h5_file, subject_key, category=None, specific_font=None):
    """Yield X, y, meta for one subject at a time."""
    with h5py.File(h5_file, 'r') as h5f:
        subj_group = h5f[subject_key]
        for category_key in subj_group.keys():
            if category and category_key != category:
                continue
            cat_group = subj_group[category_key]
            for font_key in cat_group.keys():
                font_group = cat_group[font_key]
                for specific_font_key in font_group.keys():
                    if specific_font and specific_font_key != specific_font:
                        continue
                    spec_group = font_group[specific_font_key]
                    for cond_key, cond_group in spec_group.items():
                        for seq_name, seq_group in cond_group.items():
                            if isinstance(seq_group, h5py.Dataset):
                                X = seq_group[:]
                            else:
                                first = next(iter(seq_group.values()))
                                X = first[:]
                            y = np.full(X.shape[0], 1 if 'Par' in cond_key else 0)
                            meta = pd.DataFrame({
                                'subject': [subject_key]*X.shape[0],
                                'category': [category_key]*X.shape[0],
                                'font': [font_key]*X.shape[0],
                                'specific_font': [specific_font_key]*X.shape[0],
                                'condition': [cond_key]*X.shape[0],
                                'window_name': [seq_name]*X.shape[0]
                            })
                            yield X, y, meta

def global_umap_hdbscan(h5_file, subjects, category=None, font_name=None, specific_font=None,
                        output_dir="umap_hdbscan_global", n_neighbors=15, min_dist=0.1,
                        min_cluster_size=10, sample_per_seq=2000):
    """Run UMAP + HDBSCAN on all subjects together (no LOSO)."""
    os.makedirs(output_dir, exist_ok=True)
    gc.collect()

    # ------------------- 1. Collect all data -------------------
    print("Collecting data from all subjects...")
    X_all_list, y_all_list, meta_all_list = [], [], []

    for subj in subjects:
        for X, y, meta in load_subject_windows_generator(h5_file, subj, category, specific_font):
            if len(X) > sample_per_seq:
                idx = np.random.choice(len(X), sample_per_seq, replace=False)
                X = X[idx]
                y = y[idx]
                meta = meta.iloc[idx]
            X_all_list.append(X)
            y_all_list.append(y)
            meta_all_list.append(meta)

    X_all = np.vstack(X_all_list)
    y_all = np.concatenate(y_all_list)
    meta_all = pd.concat(meta_all_list, ignore_index=True)
    del X_all_list, y_all_list, meta_all_list; gc.collect()

    # ------------------- 2. Scale and embed -------------------
    print("Scaling and embedding with UMAP...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=42, n_components=2, n_jobs=-1)
    X_emb = reducer.fit_transform(X_scaled)
    meta_all['umap_1'] = X_emb[:,0]
    meta_all['umap_2'] = X_emb[:,1]
    del X_all, X_scaled, X_emb; gc.collect()

    # ------------------- 3. HDBSCAN clustering -------------------
    print("Running HDBSCAN on embedded data...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(meta_all[['umap_1','umap_2']])
    meta_all['cluster'] = cluster_labels

    # ------------------- 4. Save results -------------------
    out_file = f"{category}_{font_name or 'all'}_{specific_font or 'all'}.csv"
    meta_all.to_csv(os.path.join(output_dir, out_file), index=False)

    # ------------------- 5. Plot clusters -------------------
    plt.figure(figsize=(8,6))
    plt.scatter(meta_all['umap_1'], meta_all['umap_2'], c=cluster_labels,
                cmap='tab20', alpha=0.7, s=20)
    plt.title(f"{category} - {font_name or 'all'} - {specific_font or 'all'}")
    plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2')
    plt.colorbar(label='Cluster')
    plt.savefig(os.path.join(output_dir, f"{category}_{font_name or 'all'}_{specific_font or 'all'}.png"),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved clustered CSV and plot to {output_dir}")
    return meta_all

# ----------------- Run over categories/fonts -----------------
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
                global_umap_hdbscan(h5_file, fixed_subjects, category=category,
                                    font_name=font, specific_font=spec_font,
                                    output_dir="umap_hdbscan_global",
                                    sample_per_seq=500, n_neighbors=15,
                                    min_dist=0.1, min_cluster_size=10)
            except Exception as e:
                print(f"Error with {category}, font={font}, specific_font={spec_font}: {e}")
                continue
