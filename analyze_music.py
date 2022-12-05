import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import libfmp.c5
import libfmp.b

# returns chromagram and chord annotation results
def analyze_chromagram(y: np.ndarray, sr: int):
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)

    # TODO get chord annotation using libfmp library
    # chord recognition
    wav_filename = "./sample/pretend-to-be.wav"
    X_CQT, Fs_X, x, Fs, x_dur = libfmp.c5.compute_chromagram_from_filename(wav_filename, N=4096, H=2048, version="CQT")
    chord_sim, chord_max = libfmp.c5.chord_recognition_template(X_CQT, norm_sim="max")
    chord_labels = libfmp.c5.get_chord_labels()

    # print(chord_sim, chord_max)
    print(chord_max)
    print(chord_labels)

    #Plot
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.03], 
                                            'height_ratios': [1, 2]}, figsize=(9, 6))

    title = 'Chromagram (N = %d)' % X_CQT.shape[1]
    libfmp.b.plot_chromagram(X_CQT, Fs=1, ax=[ax[0, 0], ax[0, 1]],
                            chroma_yticks = [0, 4, 7, 11], clim=[0, 1], cmap='gray_r',
                            title=title, ylabel='Chroma', colorbar=True)

    title = 'Time–chord representation of chord recognition result (N = %d)' % X_CQT.shape[1]
    libfmp.b.plot_matrix(chord_max, ax=[ax[1, 0], ax[1, 1]], Fs=1, 
                        title=title, ylabel='Chord', xlabel='Time (frames)')
    ax[1, 0].set_yticks(np.arange(len(chord_labels)))
    ax[1, 0].set_yticklabels(chord_labels)
    ax[1, 0].grid()
    plt.tight_layout()

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