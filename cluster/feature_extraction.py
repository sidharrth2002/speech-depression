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
import scipy
import torch
import opensmile

# gemaps features from wave
def gemaps(wave):
    # load SMILE extractor
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    # load wave
    x, sr = librosa.load(wave)
    # extract features
    features = smile.process_signal(x, sr)
    # return torch tensor
    return features

if __name__ == '__main__':
    # test
    wave = 'test_files/test1.wav'
    feature_df = gemaps(wave)
    feature_df.to_csv('test_files/test1_features.csv')
    print(feature_df)



def describe_frequency(wave):
    x, sr = librosa.load(wave)
    # fractional fast fourier transform
    freqs = np.fft.fftfreq(x.size)
    mean = np.mean(freqs)
    std = np.std(freqs)
    maxv = np.amax(freqs)
    minv = np.amin(freqs)
    median = np.median(freqs)
    skew = scipy.stats.skew(freqs)
    kurt = scipy.stats.kurtosis(freqs)
    q1 = np.quantile(freqs, 0.25)
    q3 = np.quantile(freqs, 0.75)
    mode = scipy.stats.mode(freqs)
    iqr = scipy.stats.iqr(freqs)

    # return torch tensor
    return torch.tensor([mean, std, maxv, minv, median, skew, kurt, q1, q3, mode, iqr])


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
    x, sr = librosa.load(wave)
    return librosa.feature.mfcc(x)


def spectral_features(wave):
    # Spectral features
    # 1. Spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(wave)[0]

    # 2. Spectral rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(wave)[0]

    # 3. Spectral bandwidth
    spec_bandwidth = librosa.feature.spectral_bandwidth(wave)[0]

    # 4. Spectral contrast
    spec_contrast = librosa.feature.spectral_contrast(wave)[0]

    # 5. Chroma Frequencies
    chroma_freq = librosa.feature.chroma_stft(wave)[0]

    # return as torch tensor
    return torch.tensor([spec_centroid, spec_rolloff, spec_bandwidth, spec_contrast, chroma_freq])


def teager_energy(wave):
    x, sr = librosa.load(wave)
    return librosa.feature.rmse(x)[0]


def polyfeatures(wave):
    x, sr = librosa.load(wave)
    return librosa.feature.poly_features(x)[0]


def tempogram(wave):
    x, sr = librosa.load(wave)
    return librosa.feature.tempogram(x)[0]


def spectral_features(wave):
    x, sr = librosa.load(wave)
    spec_centroid = librosa.feature.spectral_centroid(x)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(x)[0]
    spec_contrast = librosa.feature.spectral_contrast(x)[0]
    spec_flatness = librosa.feature.spectral_flatness(x)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(x)[0]

    return torch.tensor([spec_centroid, spec_bandwidth, spec_contrast, spec_flatness, spec_rolloff])


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
