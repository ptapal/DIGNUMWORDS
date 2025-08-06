import matplotlib.pyplot as plt
import numpy as np
from electrodes import get_biosemi68_positions, biosemi_68_order
from mne.viz import plot_topomap

def plot_electrode_labels(ch_names, pos_dict, output_file="results/electrode_labels.png"):
    pos = np.array([pos_dict[ch] for ch in ch_names])
    
    plt.figure(figsize=(8, 8), facecolor='white')
    ax = plt.subplot(111, aspect='equal')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    dummy_data = np.zeros(len(ch_names))
    im, _ = plot_topomap(
        dummy_data,
        pos, 
        cmap='binary',
        axes=ax,
        show=False,
        sensors=False,
        res=300,
        outlines='head',
        extrapolate='head',
        border=0,
        names=ch_names,
        vlim=(0, 1)
    )

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')

    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

positions = get_biosemi68_positions() 
plot_electrode_labels(biosemi_68_order, positions)