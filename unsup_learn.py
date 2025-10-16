import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import hdbscan

# ----------------- Data Loading -----------------
def load_h5_windows(h5_file, subjects=None, categories=None):
    X_list, y_list, meta_list = [], [], []
    with h5py.File(h5_file, 'r') as h5f:
        for subject_key in h5f.keys():
            if subjects and subject_key not in subjects:
                continue
            subj_group = h5f[subject_key]
            for category_key in subj_group.keys():
                if categories and category_key not in categories:
                    continue
                cat_group = subj_group[category_key]
                for font_key in cat_group.keys():
                    font_group = cat_group[font_key]
                    for specific_font_key in font_group.keys():
                        spec_font_group = font_group[specific_font_key]
                        for cond_key in spec_font_group.keys():  # 'Par' or 'C1'
                            cond_group = spec_font_group[cond_key]
                            for seq_name, seq_group in cond_group.items():
                                if isinstance(seq_group, h5py.Dataset):
                                    X_list.append(pd.DataFrame(seq_group[:]))
                                    y_val = 1 if 'Par' in cond_key else 0
                                    y_list.append(np.full(seq_group.shape[0], y_val))
                                    meta_list.append(pd.DataFrame({
                                        'subject': [subject_key]*seq_group.shape[0],
                                        'category': [category_key]*seq_group.shape[0],
                                        'font': [font_key]*seq_group.shape[0],
                                        'specific_font': [specific_font_key]*seq_group.shape[0],
                                        'condition': [cond_key]*seq_group.shape[0],
                                        'window_name': [seq_name]*seq_group.shape[0]
                                    }))
                                elif isinstance(seq_group, h5py.Group):
                                    for window_name, window_ds in seq_group.items():
                                        if isinstance(window_ds, h5py.Dataset):
                                            X_list.append(pd.DataFrame(window_ds[:]))
                                            y_val = 1 if 'Par' in cond_key else 0
                                            y_list.append(np.full(window_ds.shape[0], y_val))
                                            meta_list.append(pd.DataFrame({
                                                'subject': [subject_key]*window_ds.shape[0],
                                                'category': [category_key]*window_ds.shape[0],
                                                'font': [font_key]*window_ds.shape[0],
                                                'specific_font': [specific_font_key]*window_ds.shape[0],
                                                'condition': [cond_key]*window_ds.shape[0],
                                                'window_name': [window_name]*window_ds.shape[0]
                                            }))
    if not X_list:
        return pd.DataFrame(), np.array([]), pd.DataFrame()
    X = pd.concat(X_list, ignore_index=True)
    y = np.concatenate(y_list)
    meta = pd.concat(meta_list, ignore_index=True)
    return X, y, meta


# ----------------- LOSO + UMAP + HDBSCAN -----------------
def loso_umap_hdbscan_fixed_group(
    h5_file,
    subjects,
    category=None,
    font_name=None,
    specific_font=None,
    random_state=42,
    n_neighbors=15,
    min_dist=0.1,
    min_cluster_size=10,
    output_dir="umap_hdbscan_figs",
    run_odd_even=True
):
    os.makedirs(output_dir, exist_ok=True)
    X_all, y_all, meta_all = load_h5_windows(h5_file, subjects=subjects, categories=[category] if category else None)
    results = {}

    for test_subj in subjects:
        train_mask = (meta_all['subject'].isin(subjects)) & (meta_all['subject'] != test_subj)
        test_mask = meta_all['subject'] == test_subj

        X_train, X_test = X_all.loc[train_mask], X_all.loc[test_mask]
        y_test = y_all[test_mask]
        meta_test = meta_all.loc[test_mask].copy()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        X_train_emb = reducer.fit_transform(X_train_scaled)
        X_test_emb = reducer.transform(X_test_scaled)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(X_test_emb)

        meta_test['cluster'] = cluster_labels
        meta_test['umap_1'] = X_test_emb[:, 0]
        meta_test['umap_2'] = X_test_emb[:, 1]
        meta_test['parity'] = y_test

        # ---------------- Parity vs Control ----------------
        accuracy = 0
        for c in np.unique(cluster_labels):
            if c == -1: continue
            mask = cluster_labels == c
            majority_label = np.round(y_test[mask].mean())
            accuracy += (y_test[mask] == majority_label).sum()
        accuracy /= len(y_test)
        results[test_subj] = {'parity_vs_control': accuracy}

        # Plot Parity vs Control
        plt.figure(figsize=(8,6))
        plt.scatter(meta_test['umap_1'], meta_test['umap_2'],
                    c=meta_test['parity'], cmap='coolwarm', alpha=0.7)
        plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2')
        plt.title(f"{category} - {test_subj} (Parity vs Control)")
        plt.colorbar(label='Parity / Control')
        plt.savefig(os.path.join(output_dir,f"{category}_subj-{test_subj}_{font_name or 'all'}_{specific_font or 'all'}_parity.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        if not run_odd_even:
            continue

        # ---------------- Unsupervised clustering within Parity ----------------
        parity_mask = y_test == 1
        if parity_mask.sum() > 0:
            X_parity = X_test_scaled[parity_mask]
            meta_parity = meta_test.loc[parity_mask].copy()

            # Unsupervised HDBSCAN on parity trials
            clusterer_parity = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            cluster_labels_parity = clusterer_parity.fit_predict(X_parity)
            meta_parity['cluster'] = cluster_labels_parity

            # Plot clusters (no labels, purely data-driven)
            plt.figure(figsize=(8, 6))
            plt.scatter(meta_parity['umap_1'], meta_parity['umap_2'],
                        c=meta_parity['cluster'], cmap='tab10', alpha=0.7)
            plt.xlabel('UMAP 1', fontsize=14)
            plt.ylabel('UMAP 2', fontsize=14)
            plt.title(f"{category} - {test_subj} Parity-only clusters", fontsize=16)
            plt.colorbar(label='Cluster ID')
            plt.savefig(os.path.join(output_dir,
                        f"{category}_subj-{test_subj}_{font_name or 'all'}_{specific_font or 'all'}_parity_clusters.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

            # Store cluster stats (optional)
            n_clusters = len(set(cluster_labels_parity) - {-1})
            cluster_sizes = pd.Series(cluster_labels_parity).value_counts().to_dict()
            results[test_subj]['parity_n_clusters'] = n_clusters
            results[test_subj]['parity_cluster_sizes'] = cluster_sizes

        # ---------------- Unsupervised clustering on Parity + Control ----------------
        if len(y_test) > parity_mask.sum():
            X_exp = X_test_scaled
            y_exp = y_test
            meta_exp = meta_test.copy()
            clusterer_exp = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            cluster_labels_exp = clusterer_exp.fit_predict(X_exp)
            meta_exp['cluster'] = cluster_labels_exp

            plt.figure(figsize=(8,6))
            plt.scatter(meta_exp['umap_1'], meta_exp['umap_2'],
                        c=meta_exp['cluster'], cmap='plasma', alpha=0.7)
            plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2')
            plt.title(f"{category} - {test_subj} Parity+Control clusters")
            plt.colorbar(label='Cluster ID')
            plt.savefig(os.path.join(output_dir,f"{category}_subj-{test_subj}_{font_name or 'all'}_{specific_font or 'all'}_parity_control_clusters.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

            n_clusters_exp = len(set(cluster_labels_exp) - {-1})
            cluster_sizes_exp = pd.Series(cluster_labels_exp).value_counts().to_dict()
            results[test_subj]['parity_control_n_clusters'] = n_clusters_exp
            results[test_subj]['parity_control_cluster_sizes'] = cluster_sizes_exp

    return results


# ----------------- Run over categories/fonts -----------------
fonts = ['1F','20F']
categories = ['NumWo','Dig']
h5_file = 'h5_freq/features_windowed_unlabeled_experimental.h5'

# >>>> Define the fixed 5 subjects <<<<
fixed_subjects = ['S03', 'S19']  # modify these names as per your dataset

for category in categories:
    all_results = []

    for font in fonts:
        specific_fonts = [None] if font == '1F' else ['A', 'S']

        for spec_font in specific_fonts:
            try: 
                print(f"Running {category}, font={font}, specific_font={spec_font}, subjects={fixed_subjects}")
                results = loso_umap_hdbscan_fixed_group(
                    h5_file,
                    subjects=fixed_subjects,
                    category=category,
                    font_name=font,
                    specific_font=spec_font,
                    run_odd_even=True,
                    output_dir="umap_hdbscan_figs"
                )

                for test_subj in results.keys():
                    record = {
                        'category': category,
                        'font': font,
                        'specific_font': spec_font or 'None',
                        'subject': test_subj,
                        'parity_vs_control': results[test_subj].get('parity_vs_control'),
                        'parity_n_clusters': results[test_subj].get('parity_n_clusters'),
                        'parity_control_n_clusters': results[test_subj].get('parity_control_n_clusters')
                    }
                    all_results.append(record)
            except Exception as e:
                print(f"Error processing {category}, font={font}, specific_font={spec_font}: {e}")
                continue

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"umap_hdbscan_results_{category}_fixed5.csv", index=False)
    print(f"Saved results to umap_hdbscan_results_{category}_fixed5.csv")