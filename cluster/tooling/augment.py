import nlpaug.augmenter.audio as ag
import librosa
import logging

logging.basicConfig(level=logging.INFO)


def augment_audio(file, method='vtlp'):
    if method == 'vltp':
        # read audio file
        logging.info(f'Augmenting audio file {file} with method {method}')
        audio, sr = librosa.load(file)
        aug = ag.VtlpAug(sr, factor=(0.9, 1.1))
        # augment audio
        augmented_audio = aug.augment(audio)
        return augmented_audio
    elif method == 'noise_pitch':
        # (1) Adding noise: mix the audio signal with random noise. Each mix z is generated using z = x + α × rand(x) where x is the audio signal and α is the noise factor. In our experiments, α = 0.01, 0.02 and 0.03.
        logging.info(f'Augmenting audio file {file} with method {method}')
        audio, sr = librosa.load(file)
        aug = ag.NoiseAug(zone=(0.01, 0.03))
        # augment audio
        augmented_audio = aug.augment(audio)
        # Pitch Shifting: lower the pitch of the audio sample by 3 values (in semitones): (0.5, 2 and 5).
        aug = ag.PitchAug(sampling_rate=sr, zone=(0.5, 2))
        augmented_audio = aug.augment(augmented_audio)
        return augmented_audio