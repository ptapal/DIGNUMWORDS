import pandas as pd
import numpy as np
from scipy.signal import spectrogram, butter, filtfilt
from scipy.fft import fft, fftfreq
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
        nperseg = max(32, min(256, len(signal)))  # minimum segment length 32
        noverlap = min(128, nperseg // 2)
        f, t, Sxx = spectrogram(signal, fs=self.sampling_rate, nperseg=nperseg, noverlap=noverlap)
        return f, t, Sxx

    # extract gray-level co-occurrence matrix texture features from a spectrogram.
    def compute_glcm_features(self, spectrogram_img):
        # normalize to [0,1] safely
        max_val = np.max(spectrogram_img)
        if max_val == 0:
            spectrogram_img = np.zeros_like(spectrogram_img)
        else:
            spectrogram_img = spectrogram_img / max_val

        spectrogram_img = img_as_ubyte(spectrogram_img)

        # quantize safely
        bins = np.linspace(0, 255, 17)
        quantized_img = np.digitize(spectrogram_img, bins) - 1
        quantized_img[quantized_img >= 16] = 15

        # if the image is constant, return defaults
        if np.min(quantized_img) == np.max(quantized_img):
            return {'contrast':0,'dissimilarity':0,'homogeneity':1,'energy':1,'correlation':0,'ASM':1}

        glcm = graycomatrix(quantized_img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16)

        features = {}
        for prop in ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']:
            val = np.mean(graycoprops(glcm, prop))
            features[prop] = 0 if np.isnan(val) else val

        return features
    
    def compute_ssvep_feature(self, signal, freq_of_interest=3.75):
        ssvep_list = []

        # Case 1: MNE RawArray (multi-channel)
        if hasattr(signal, "get_data"):
            data = signal.get_data()
            ch_names = signal.info['ch_names']
            for i, ch_name in enumerate(ch_names):
                sig = data[i]
                n_samples = len(sig)
                yf = np.abs(fft(sig))
                xf = fftfreq(n_samples, 1/self.sampling_rate)
                idx = np.argmin(np.abs(xf - freq_of_interest))
                ssvep_list.append({'Electrode': ch_name, 'ssvep_amp': yf[idx]})

        return pd.DataFrame(ssvep_list)

    def compute_frequency_features(self, signal, n_fft=None):
        n_times = signal.n_times
        if n_fft is None or n_fft > n_times:
            n_fft = n_times  # avoid error if bin is smaller than default n_fft

        psds, freqs = mne.time_frequency.psd_array_welch(
            signal.get_data(),
            sfreq=signal.info['sfreq'],
            fmin=1,
            fmax=80,
            n_fft=n_fft
        )

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
            band_power = np.mean(psds[:, band_idx], axis=1)  # average across freq bins
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
        freq_df = self.compute_frequency_features(signal)
        glcm_df = self.extract_glcm_features(df)

        # SSVEP feature
        ssvep_list = []
        for i in df.columns:
            amp = self.compute_ssvep_feature(df[i].values)
            ssvep_list.append({'Electrode': i, 'ssvep_amp': amp})
        ssvep_df = pd.DataFrame(ssvep_list)

        # merge all features
        merged_df = freq_df.merge(glcm_df, on='Electrode').merge(ssvep_df, on='Electrode')
        return merged_df