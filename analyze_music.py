import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# returns chromagram and chord annotation results
def analyze_chromagram(y: np.ndarray, sr: int):
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)

    # TODO get chord annotation using libfmp library

    return chromagram

def run_feature_extraction(filename: str):
    # loads an audio file as float ndarray
    y, sr = librosa.load(filename, duration=12)

    # separate harmonics and percussives
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)

    # compute chromagram
    chromagram = analyze_chromagram(y=y_harmonic, sr=sr)

    # plot pitch information 
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    img = librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time', ax=ax[0])
    ax[0].set(title='chroma')
    ax[0].label_outer()

    fig.colorbar(img, ax=ax)

    plt.show()

# testing
run_feature_extraction("./sample/pretend-to-be.mp3")