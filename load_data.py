import os
import h5py
import pandas as pd
import numpy as np

def load_modality_data(base_dir, subjects, modality, font='1F', condition='Par'):
    X, y, subj_ids = [], [], []
    
    for subject in subjects:
        subject_dir = f"S{subject:02d}"
        for cond, label in [('0', 0), ('1', 1)]:
            h5_path = os.path.join(base_dir, subject_dir, modality, font, f"{condition}_{cond}.h5")
            
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as f:
                    group_name = f"{subject_dir}/{modality}/{font}"
                    if group_name in f:
                        group = f[group_name]
                        dataset_name = f"{condition}_{cond}"
                        if dataset_name in group:
                            data = group[dataset_name][:]
                            columns = [col.decode() for col in group.attrs['columns']]
                            
                            X.append(pd.DataFrame(data, columns=columns))
                            y.extend([label] * len(data))
                            subj_ids.extend([subject] * len(data))
    
    if not X: 
        return pd.DataFrame(), np.array([]), np.array([])
    
    return pd.concat(X), np.array(y), np.array(subj_ids)