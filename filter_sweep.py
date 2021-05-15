import numpy as np
import stream as strm
import matplotlib.pyplot as plt
from filters import HPF1, LPF1
import simpleaudio as sa
import wavio

# stream length, in seconds
length = 30.

# sample frequency, in Hz (usually always 44100)
sampfreq = 44100

# frequency of note to generate in Hz
nfreq = 400

# initialise stream
stream = strm.Stream(length, sampfreq)

# generate & plot simple square wave
stream.values = np.where((stream.samples * 0.5 * nfreq / sampfreq) % 2 > 1, -1., 1.)
plt.plot(stream.values)

# create filter sweep mapping function
sfunc = lambda x: (1. - ((x*60.) % 1))**7

# apply filter sweep
stream.bufferize(0.02)
stream.filt_sweep(LPF1, sfunc)
wavio.write(f"filttest.wav", np.clip(stream.values,-10,10), sampfreq, sampwidth=2) 

# play audio
outwave = stream.values * 32767 / np.max(np.abs(stream.values))
play_obj = sa.play_buffer(outwave.astype(np.int64), 1, 2, 44100)

plt.plot(stream.values)
plt.show()
