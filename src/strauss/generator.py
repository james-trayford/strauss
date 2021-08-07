import numpy as np
import stream
import notes
import glob
import wavio
from scipy.interpolate import interp1d

# TO DO:
# - Ultimately have Synth and Sampler classes that own their own stream (stream.py) object
#   allowing ADSR volume and filter enveloping, LFO implementation etc.
# - Functions here will generally be called from a "Score" class that is provided with the
#   musical choices and uses these to generate sound, but can be interfaced with directly.

def gen_chord(stream, chordname, rootoctv=3):
    """ 
    generate chord over entire stream given chord name and optional
    octave of root note
    """
    frqs = notes.parse_chord(chordname, rootoctv)
    frqsamp = frqs/stream.samprate 
    for f in frqsamp:
        stream.values += detuned_saw(stream.samples, f)
    
def detuned_saw(samples, freqsamp, oscdets=[1,1.005,0.995]):
    """
    Three oscillator sawtooth wave generator with slight detuning for
    texture
    """
    saw = lambda freqsamp, samp: 1-((samples*(freqsamp)/2) % 2)
    signal = np.zeros(samples.size)
    for det in oscdets:
        freq = freqsamp*det
        signal += saw(freq, samples+freq*np.random.random())
    return signal

class Generator:
    pass

class Synthesizer(Generator):
    pass

class Sampler(Generator):
    def __init__(self, sampfiles, params=None):
        if isinstance(sampfiles, dict):
            self.sampdict = sampfiles
        if isinstance(sampfiles, str):
            wavs = glob.glob(sampfiles+"/*")
            self.sampdict = {}
            for w in wavs:
                note = w.split('/')[-1].split('_')[-1].split('.')[0]
                self.sampdict[note] = w
        self.load_samples()
        
    def load_samples(self):
        self.samples = {}
        self.samplens = {}
        for note in self.sampdict.keys():
            wavobj = wavio.read(self.sampdict[note])
            # force to mono
            wavdat = wavobj.data.mean(axis=1)
            wavdat /= abs(wavdat).max()
            samps = range(wavdat.size)
            self.samples[note] = interp1d(samps, wavdat,
                                          bounds_error=False,
                                          fill_value = (0.,0.),
                                          assume_sorted=True)
            self.samplens[note] = wavdat.size
