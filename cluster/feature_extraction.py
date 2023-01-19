'''
1. Prosodic features
2. Spectracl features
3. Mel Frequency Cepstral Coefficients (MFCC)
4. Linear prediction cepstral coefficients
5. Gammatone Frequency Cepstral Coefficients
6. Voice Quality Features
7. Teager Energy Operator Based Features
'''
import librosa
import numpy as np


def prosodic_features(wave):
    # Prosodic features
    # 1. Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(wave)[0]

    # 2. Spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(wave)[0]

    # 3. Spectral rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(wave)[0]

    # 4. Spectral bandwidth
    spec_bandwidth = librosa.feature.spectral_bandwidth(wave)[0]

    # 5. Spectral contrast
    spec_contrast = librosa.feature.spectral_contrast(wave)[0]

    # 6. Chroma Frequencies
    chroma_freq = librosa.feature.chroma_stft(wave)[0]

    rhythm_tempogram = librosa.feature.tempogram(wave)[0]

    return {
        'zcr': zcr,
        'spec_centroid': spec_centroid,
        'spec_rolloff': spec_rolloff,
        'spec_bandwidth': spec_bandwidth,
        'spec_contrast': spec_contrast,
        'chroma_freq': chroma_freq,
        'rhythm_tempogram': rhythm_tempogram
    }


def energy(wave):
    x, sr = librosa.load(wave)
    return np.sum(x**2)


def rmse_energy(wave):
    x, sr = librosa.load(wave)
    return np.sqrt(np.mean(x**2))


def zero_crossing_rate(wave):
    x, sr = librosa.load(wave)
    return sum(librosa.zero_crossings(x, pad=False))


def mfcc(wave):
    return librosa.feature.mfcc(wave)
