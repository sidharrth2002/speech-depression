import nlpaug.augmenter.audio as ag
import librosa
import logging

logging.basicConfig(level=logging.INFO)


def augment_audio(file, method='vtlp'):
    # read audio file
    logging.info(f'Augmenting audio file {file} with method {method}')
    audio, sr = librosa.load(file)
    aug = ag.VtlpAug(sr, factor=(0.9, 1.1))
    # augment audio
    augmented_audio = aug.augment(audio)
    return augmented_audio
