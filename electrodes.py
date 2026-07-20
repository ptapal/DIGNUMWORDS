# BioSemi 68-channel EEG system electrode order
biosemi_68_order = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7',
    'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7',
    'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7',
    'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz',
    'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'F2', 'F4', 'F6',
    'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'C2', 'C4', 'C6',
    'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6',
    'P8', 'P10', 'PO8', 'PO4', 'O2', 'FCz', 'Cz', 'E1',
    'E2', 'E3', 'E4', 'E5', 'E6',  'EOG1', 'EOG2'
]

# region mapping for electrodes
region_map = {
    'Frontal': [
        'Fp1', 'Fp2', 'Fpz', 'AF3', 'AF4', 'AF7', 'AF8',
        'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'
    ],
    'Frontocentral': [
        'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz',
        'FT7', 'FT8'
    ],
    'Central': [
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'Cz'
    ],
    'Centroparietal': [
        'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz',
        'TP7', 'TP8'
    ],
    'Parietal': [
        'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'Pz',
        'P9', 'P10'
    ],
    'Parietooccipital': [
        'PO3', 'PO4', 'PO7', 'PO8', 'POz'
    ],
    'Occipital': [
        'O1', 'O2', 'Oz', 'Iz'
    ],
    'EOG': [
        'EOG1', 'EOG2'
    ],
    'Externals': [
        'E1', 'E2', 'E3', 'E4', 'E5', 'E6'
    ]
}

import mne
import numpy as np

def get_biosemi68_mne_montage():
    # Start from MNE's biosemi64
    montage64 = mne.channels.make_standard_montage('biosemi64')
    ch_pos = montage64.get_positions()['ch_pos']

    # add extra channels (PO3, PO4, PO7, PO8)
    extra_channels = {
        'PO3': (-0.028, -0.09, 0.0),
        'PO4': (0.028, -0.09, 0.0),
        'PO7': (-0.045, -0.09, 0.0),
        'PO8': (0.045, -0.09, 0.0),
        'E1': (-0.05, 0.05, 0.0),
        'E2': (0.05, 0.05, 0.0),
        'E3': (-0.05, -0.05, 0.0),
        'E4': (0.05, -0.05, 0.0),
        'E5': (-0.05, 0.0, 0.0),
        'E6': (0.05, 0.0, 0.0),
        'EOG1': (-0.04, 0.06, 0.0),
        'EOG2': (0.04, 0.06, 0.0)
    }

    ch_pos.update(extra_channels)

    montage68 = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    return montage68

# Example usage
if __name__ == "__main__":
    montage = get_biosemi68_mne_montage()
    print(montage.ch_names)  