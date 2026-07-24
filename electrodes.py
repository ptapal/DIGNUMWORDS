biosemi_68_order = [
    'Fp1', 'AF7', 'AF3', 'F1',  'F3',  'F5',  'F7',          # 0-6
    'FT7', 'FC5', 'FC3', 'FC1', 'C1',  'C3',  'C5',  'T7',   # 7-14
    'TP7', 'CP5', 'CP3', 'CP1', 'P1',  'P3',  'P5',  'P7',   # 15-22
    'P9',  'PO7', 'PO3', 'O1',  'Iz',  'Oz',  'POz', 'Pz',   # 23-30
    'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz',          # 31-37
    'F2',  'F4',  'F6',  'F8',  'FT8', 'FC6', 'FC4', 'FC2',  # 38-45
    'FCz', 'Cz',  'C2',  'C4',  'C6',  'T8',  'TP8',         # 46-52
    'CP6', 'CP4', 'CP2', 'P2',  'P4',  'P6',  'P8',  'P10',  # 53-60
    'PO8', 'PO4', 'O2',                                        # 61-63
    'PO9', 'I1',  'I2',  'PO10',                              # 64-67
]

RETTER_ROI     = ['PO7', 'O1', 'PO8', 'O2', 'PO9', 'I1', 'I2', 'PO10']
RETTER_ROI_IDX = [biosemi_68_order.index(ch) for ch in RETTER_ROI]

region_map = {
    'Frontal': [
        'Fp1', 'Fp2', 'Fpz', 'AF3', 'AF4', 'AF7', 'AF8', 'AFz',
        'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Fz',
    ],
    'Frontocentral': [
        'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz',
        'FT7', 'FT8',
    ],
    'Central': [
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'Cz',
    ],
    'Centroparietal': [
        'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz',
        'TP7', 'TP8',
    ],
    'Parietal': [
        'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'Pz',
        'P9', 'P10',
    ],
    'Parietooccipital': [
        'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'PO9', 'PO10',
    ],
    'Occipital': [
        'O1', 'O2', 'Oz', 'Iz', 'I1', 'I2',
    ],
}

import mne
import numpy as np


def get_biosemi68_mne_montage():
    montage64 = mne.channels.make_standard_montage('biosemi64')
    ch_pos = dict(montage64.get_positions()['ch_pos'])

    # PO9/PO10 not in biosemi64 — estimated as weighted midpoint of P9 and PO7/PO8
    ch_pos['PO9']  = 0.55 * ch_pos['P9']  + 0.45 * ch_pos['PO7']
    ch_pos['PO10'] = 0.55 * ch_pos['P10'] + 0.45 * ch_pos['PO8']

    # I1/I2 not in biosemi64 — estimated from Iz and O1/O2
    ch_pos['I1'] = 0.5 * ch_pos['Iz'] + 0.5 * ch_pos['O1'] + np.array([-0.010, 0.000, -0.005])
    ch_pos['I2'] = 0.5 * ch_pos['Iz'] + 0.5 * ch_pos['O2'] + np.array([ 0.010, 0.000, -0.005])

    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')


if __name__ == '__main__':
    montage = get_biosemi68_mne_montage()
    print('Total channels in montage:', len(montage.ch_names))
    for ch in RETTER_ROI:
        print(f'  {ch}: index {biosemi_68_order.index(ch)}')
