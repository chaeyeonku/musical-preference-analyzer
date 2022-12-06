import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import libfmp.c5
import libfmp.b

# returns chromagram and chord recognition results
def analyze_chromagram(X_CQT, Fs_X, x, Fs, x_dur):

    # chord recognition
    chord_sim, chord_max = libfmp.c5.chord_recognition_template(X_CQT, norm_sim="max")
    chord_labels = libfmp.c5.get_chord_labels()

    # figure settings
    fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.03], 
                                            'height_ratios': [1, 1, 2]}, figsize=(9, 6))

    # plot waveform
    libfmp.b.plot_signal(x, Fs, ax=ax[0,0], title='Waveform of audio signal')

    # plot chromagram
    title = 'Chromagram (N = %d)' % X_CQT.shape[1]
    libfmp.b.plot_chromagram(X_CQT, Fs=1, ax=[ax[1, 0], ax[1, 1]],
                            chroma_yticks = [0, 4, 7, 11], clim=[0, 1], cmap='gray_r',
                            title=title, ylabel='Chroma', colorbar=True)

    # plot chord recognition
    title = 'Time–chord representation of chord recognition result (N = %d)' % X_CQT.shape[1]
    libfmp.b.plot_matrix(chord_max, ax=[ax[2, 0], ax[2, 1]], Fs=1, 
                        title=title, ylabel='Chord', xlabel='Time (frames)')
    ax[2, 0].set_yticks(np.arange(len(chord_labels)))
    ax[2, 0].set_yticklabels(chord_labels)
    ax[2, 0].grid()

    plt.tight_layout()
    plt.show()

def analyze_tempo(filename: str):
    y, sr = librosa.load(filename)

    temp, beats = librosa.beat.beat_track(y=y, sr=sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)

    hop_length = 512
    fig, ax = plt.subplots(nrows=2, sharex=True)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
                            y_axis='mel', x_axis='time', hop_length=hop_length,
                            ax=ax[0])
    ax[0].label_outer()
    ax[0].set(title='Mel spectrogram')
    ax[1].plot(times, librosa.util.normalize(onset_env),
            label='Onset strength')
    ax[1].vlines(times[beats], 0, 1, alpha=0.5, color='r',
            linestyle='--', label='Beats')
    ax[1].legend()

    plt.show()

def run_feature_extraction(filename: str):
    # load audio from file
    X_CQT, Fs_X, x, Fs, x_dur = libfmp.c5.compute_chromagram_from_filename(filename, N=4096, H=2048, version="CQT")

    analyze_chromagram(X_CQT, Fs_X, x, Fs, x_dur)

    analyze_tempo(filename)

# testing
run_feature_extraction("./sample/pretend-to-be.wav")