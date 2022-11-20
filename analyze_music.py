import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# filename = librosa.example('nutcracker')
filename = "./sample/Pretend to be.mp3"

y, sr = librosa.load(filename, duration=30)

# Separate harmonics and percussives into two waveforms
y_harmonic, y_percussive = librosa.effects.hpss(y)

tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

# print(beat_frames)

# estimate tempo
print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# print(beat_times)

# compute chromagram
chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

# print(chromagram)

# plot pitch information 
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
img = librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time', ax=ax[0])
ax[0].set(title='chroma')
ax[0].label_outer()

fig.colorbar(img, ax=ax)

plt.show()