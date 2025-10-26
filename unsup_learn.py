import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import hdbscan
import os, gc
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import hdbscan

def load_subject_windows_generator(h5_file, subject_key, category=None, specific_font=None):
    """Yield X, y, meta for one subject at a time."""
    with h5py.File(h5_file, 'r') as h5f:
        subj_group = h5f[subject_key]
        for category_key in subj_group.keys():
            if category and category_key != category: continue
            cat_group = subj_group[category_key]
            for font_key in cat_group.keys():
                font_group = cat_group[font_key]
                for specific_font_key in font_group.keys():
                    if specific_font and specific_font_key != specific_font: continue
                    spec_group = font_group[specific_font_key]
                    for cond_key, cond_group in spec_group.items():
                        for seq_name, seq_group in cond_group.items():
                            if isinstance(seq_group, h5py.Dataset):
                                X = seq_group[:]
                                y = np.full(X.shape[0], 1 if 'Par' in cond_key else 0)
                                meta = pd.DataFrame({
                                    'subject':[subject_key]*X.shape[0],
                                    'category':[category_key]*X.shape[0],
                                    'font':[font_key]*X.shape[0],
                                    'specific_font':[specific_font_key]*X.shape[0],
                                    'condition':[cond_key]*X.shape[0],
                                    'window_name':[seq_name]*X.shape[0]
                                })
                                yield X, y, meta
                            else:
                                for win_name, win_ds in seq_group.items():
                                    X = win_ds[:]
                                    y = np.full(X.shape[0], 1 if 'Par' in cond_key else 0)
                                    meta = pd.DataFrame({
                                        'subject':[subject_key]*X.shape[0],
                                        'category':[category_key]*X.shape[0],
                                        'font':[font_key]*X.shape[0],
                                        'specific_font':[specific_font_key]*X.shape[0],
                                        'condition':[cond_key]*X.shape[0],
                                        'window_name':[win_name]*X.shape[0]
                                    })
                                    yield X, y, meta

def loso_umap_hdbscan_fastmem(h5_file, subjects, category=None, font_name=None, specific_font=None,
                              output_dir="umap_hdbscan_fastmem", n_neighbors=15, min_dist=0.1,
                              min_cluster_size=10, random_state=42, run_odd_even=True,
                              sample_per_subj=2000):
    """
    Memory- and time-efficient LOSO:
      • builds one shared UMAP on small training samples
      • then reuses it for all subjects
    """
    import gc
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # ------------------- 1. Collect small global sample -------------------
    print("Building global training sample for UMAP...")
    scaler = StandardScaler()
    X_sample_list = []

    with h5py.File(h5_file, 'r') as h5f:
        for subj in subjects:
            if subj not in h5f: continue
            subj_group = h5f[subj]
            for cat_key in subj_group.keys():
                if category and cat_key != category: continue
                for font_key in subj_group[cat_key].keys():
                    for spec_key in subj_group[cat_key][font_key].keys():
                        if specific_font and spec_key != specific_font: continue
                        for cond_key, cond_group in subj_group[cat_key][font_key][spec_key].items():
                            for seq_name, seq_ds in cond_group.items():
                                if isinstance(seq_ds, h5py.Dataset):
                                    X = seq_ds[:]
                                else:
                                    first = next(iter(seq_ds.values()))
                                    X = first[:]
                                if len(X) > sample_per_subj:
                                    idx = np.random.choice(len(X), sample_per_subj, replace=False)
                                    X = X[idx]
                                X_sample_list.append(X)
                                break  # just one sequence per subject/category for speed
                            break
                if len(X_sample_list) >= len(subjects): break

    X_sample = np.vstack(X_sample_list)
    scaler.fit(X_sample)
    X_sample_scaled = scaler.transform(X_sample)

    # ------------------- 2. Fit one global UMAP -------------------
    print("Fitting shared UMAP on sample...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=random_state, n_components=2)
    reducer.fit(X_sample_scaled)
    del X_sample, X_sample_scaled, X_sample_list; gc.collect()

    # ------------------- 3. LOSO loop -------------------
    for test_subj in subjects:
        print(f"\nLOSO: test subject {test_subj}")
        X_test_list, y_test_list, meta_test_list = [], [], []

        for X, y, meta in load_subject_windows_generator(h5_file, test_subj, category, specific_font):
            X_scaled = scaler.transform(X)
            X_emb = reducer.transform(X_scaled)
            X_test_list.append(X_emb)
            y_test_list.append(y)
            meta_test_list.append(meta)
            del X, X_scaled, X_emb; gc.collect()

        if not X_test_list:
            print(f"  No data for {test_subj}, skipping")
            results[test_subj] = {'parity_vs_control': np.nan}
            continue

        X_test_emb = np.vstack(X_test_list)
        y_test = np.concatenate(y_test_list)
        meta_test = pd.concat(meta_test_list, ignore_index=True)
        del X_test_list, y_test_list, meta_test_list; gc.collect()

        # ---------------- HDBSCAN ----------------
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(X_test_emb)

        meta_test['cluster'] = cluster_labels
        meta_test['umap_1'] = X_test_emb[:, 0]
        meta_test['umap_2'] = X_test_emb[:, 1]
        meta_test['parity'] = y_test

        # Accuracy
        acc = 0
        for c in np.unique(cluster_labels):
            if c == -1: continue
            mask = cluster_labels == c
            maj = np.round(y_test[mask].mean())
            acc += (y_test[mask] == maj).sum()
        acc /= len(y_test)
        results[test_subj] = {'parity_vs_control': acc}

        # Plot
        plt.figure(figsize=(8,6))
        plt.scatter(meta_test['umap_1'], meta_test['umap_2'],
                    c=meta_test['parity'], cmap='coolwarm', alpha=0.7)
        plt.title(f"{category} - {test_subj}")
        plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2')
        plt.colorbar(label='Parity')
        plt.savefig(os.path.join(output_dir,
                    f"{category}_subj-{test_subj}_{font_name or 'all'}_{specific_font or 'all'}.png"),
                    dpi=200, bbox_inches='tight')
        plt.close()

        # Save per-subject details
        meta_test.to_csv(os.path.join(output_dir,
                     f"{category}_subj-{test_subj}_{font_name or 'all'}_{specific_font or 'all'}.csv"), index=False)
        del meta_test, X_test_emb, y_test; gc.collect()

    return results





# ----------------- Run over categories/fonts -----------------
fonts = ['1F', '20F']
categories = ['NumWo', 'Dig']
fixed_subjects = [
    'S02', 'S03', 'S04', 'S05', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12',
    'S13', 'S15', 'S16', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24',
    'S25', 'S26', 'S28'
]

h5_file = 'h5_freq/features_windowed_unlabeled_experimental.h5'


for category in categories:
    all_results = []

    for font in fonts:
        specific_fonts = [None] if font == '1F' else ['A', 'S']

        for spec_font in specific_fonts:
            try:
                print(f"\nRunning {category}, font={font}, specific_font={spec_font}")
                results = loso_umap_hdbscan_fastmem(
                    h5_file,
                    subjects=fixed_subjects,
                    category=category,
                    font_name=font,
                    specific_font=spec_font,
                    run_odd_even=True,
                    output_dir="umap_hdbscan_figs"
                )

                for test_subj, res in results.items():
                    all_results.append({
                        'category': category,
                        'font': font,
                        'specific_font': spec_font or 'None',
                        'subject': test_subj,
                        **res
                    })
            except Exception as e:
                print(f"Error processing {category}, font={font}, specific_font={spec_font}: {e}")
                continue

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"umap_hdbscan_results_{category}_fixed5.csv", index=False)
    print(f"Saved results to umap_hdbscan_results_{category}_fixed5.csv")
