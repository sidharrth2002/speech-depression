import random
import nlpaug.augmenter.audio as ag
import librosa
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


def augment_audio(file, method='vtlp'):
    if method == 'vltp':
        # read audio file
        logging.info(f'Augmenting audio file {file} with method {method}')
        audio, sr = librosa.load(file, sr=16000)
        aug = ag.VtlpAug(sr, factor=(0.9, 1.1))
        # augment audio
        augmented_audio = aug.augment(audio)
        return augmented_audio
    elif method == 'noise_pitch':

        # • Noise Injection: adds random values into the data with a noise
        # factor of 0.05.
        # • Pitch Augmenter: randomly changes the pitch of the signal. The
        # pitch factor is 1.5.
        # • Shift Augmenter: randomly shifts the audio signal to left or right.
        # The shift max value used is 0.2 s. If the audio signal is shifted to the
        # left with x seconds, the first x seconds are marked as silence (Fast
        # Forwarding the audio signal). If the audio signal is shifted to the right
        # with x seconds, the last x seconds are marked as silence (Back Forwarding the audio signal).
        # • Speed Augmenter: stretches times series by a fixed rate with a speed
        # factor of 1.5. 

        # choose a type of augmentation
        methods = ['pitch', 'shift', 'speed']
        choice = random.choice(methods)
        logging.info(f'Augmenting audio file {file} with method {choice}')

        audio, sr = librosa.load(file, sr=16000)

        logging.info(f"Shape of audio before augmentation: {np.array(audio).shape}")

        if choice == 'noise':
            noise_factor = 0.05
            white_noise = np.random.randn(len(audio)) * noise_factor
            augmented_audio = audio + white_noise
            logging.info(f"Shape of augmented audio: {np.array(augmented_audio).shape}")
        elif choice == 'pitch':
            pitch_factor = 1.5
            aug = ag.PitchAug(sampling_rate=sr, factor=(1.4, 1.6))
            augmented_audio = aug.augment(audio)
            logging.info(f"Shape of augmented audio: {np.array(augmented_audio).shape}")
        elif choice == 'shift':
            shift_factor = 0.2
            aug = ag.ShiftAug(sampling_rate=sr, duration=shift_factor)
            augmented_audio = aug.augment(audio)
            logging.info(f"Shape of augmented audio: {np.array(augmented_audio).shape}")
        else:
            speed_factor = 1.5
            aug = ag.SpeedAug(factor=(1.4, 1.6))
            augmented_audio = aug.augment(audio)
            logging.info(f"Shape of augmented audio: {np.array(augmented_audio).shape}")


        return augmented_audio

        # # (1) Adding noise: mix the audio signal with random noise. Each mix z is generated using z = x + α × rand(x) where x is the audio signal and α is the noise factor. In our experiments, α = 0.01, 0.02 and 0.03.
        # logging.info(f'Augmenting audio file {file} with method {method}')
        # audio, sr = librosa.load(file)
        # aug = ag.NoiseAug(zone=(0.01, 0.03))
        # # augment audio
        # augmented_audio = aug.augment(audio)
        # # Pitch Shifting: lower the pitch of the audio sample by 3 values (in semitones): (0.5, 2 and 5).
        # aug = ag.PitchAug(sampling_rate=sr, zone=(0.5, 2), factor=
        # augmented_audio = aug.augment(augmented_audio)
        # return augmented_audio