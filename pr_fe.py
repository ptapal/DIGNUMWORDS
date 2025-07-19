import pandas as pd
import numpy as np
from scipy.signal import spectrogram, butter, filtfilt
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import mne

'''
Implementing Time Frequency Analysis
It avoids the loss of frequency information, the loss of signal waveform transients
It extracts features that cannot be simultaneously expressed in a single domain
'''

class FeatureExtractor:
    # initialising sr for each dataset, lowcut and highcut can be adjusted
    def __init__(self, sampling_rate=512):
        self.sampling_rate = sampling_rate


    # computing spectrogram (short time fourier transform) for a single electrode
    def compute_spectrogram(self, signal):
        f, t, Sxx = spectrogram(signal, fs=self.sampling_rate, nperseg=256, noverlap=128)
        return f, t, Sxx

    # extract gray-level co-occurrence matrix texture features from a spectrogram.
    def compute_glcm_features(self, spectrogram_img):
        spectrogram_img = img_as_ubyte(spectrogram_img / np.max(spectrogram_img))
        bins = np.linspace(0, 255, 17)[:-1]
        quantized_img = np.digitize(spectrogram_img, bins)
        glcm = graycomatrix(quantized_img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=17)
        return {
            'contrast': np.mean(graycoprops(glcm, 'contrast')),
            'dissimilarity': np.mean(graycoprops(glcm, 'dissimilarity')),
            'homogeneity': np.mean(graycoprops(glcm, 'homogeneity')),
            'energy': np.mean(graycoprops(glcm, 'energy')),
            'correlation': np.mean(graycoprops(glcm, 'correlation')),
            'ASM': np.mean(graycoprops(glcm, 'ASM'))
        }
    
    # frequency band analysis
    def compute_frequency_features(self, signal, n_fft=256):
        psds, freqs = mne.time_frequency.psd_array_welch(signal.get_data(), signal.info['sfreq'], fmin= 1, fmax=80, n_fft=n_fft)

        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta':  (13, 30),
            'gamma': (30, 80)
        }

        features = {}
        for band, (low, high) in bands.items():
            band_idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.mean(psds[:, band_idx], axis=1)  
            features[band] = band_power

        frequency_df = pd.DataFrame(features)
        frequency_df['Electrode'] = signal.info['ch_names']

        return frequency_df

    # process eeg data and extract glcm features for each electrode
    def extract_glcm_features(self, df):
        feature_list = []
        electrodes = list(df.columns)

        for i in electrodes:
            if i in df.columns:
                signal = df[i].values
                _, _, power_spectrogram = self.compute_spectrogram(signal)
                glcm_features = self.compute_glcm_features(power_spectrogram)
                glcm_features['Electrode'] = i

                feature_list.append(glcm_features)

        # df conversion
        glcm_df = pd.DataFrame(feature_list)
        return glcm_df
    
    # merging datasets
    def merging_feature_data(self, signal, df):
        return pd.merge(self.compute_frequency_features(signal), self.extract_glcm_features(df), on='Electrode', how='inner')