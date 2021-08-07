from .stream import Stream
from .channels import audio_channels
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Sonification:
    def __init__(self, score, sources, generator, audio_setup='stereo', samprate=44100):

        # sampling rate in Hz
        self.samprate = samprate
        
        # sonification owns an instance of the Score
        self.score = score
        
        # sonification owns an instance of the Sources
        self.sources = sources

        # sonification owns an instance of the Generator
        self.generator = generator
        
        # set up the audio channel routing for the sonification
        self.channels = audio_channels(setup=audio_setup)

        # ...and the corresponding Stream objects 
        self.out_channels = {}
        for c in range(self.channels.Nmics):
            self.out_channels[str(c)] = Stream(self.score.length, self.samprate)

    def render(self):

        cbin = np.digitize(self.sources.mapping['time'], self.score.fracbins, 0)
        cbin = np.clip(cbin-1, 0, self.score.nchords-1)
        pitchfrac = self.sources.mapping['pitch'].argsort()/self.sources.nevents
        Nsamp = self.out_channels['0'].values.size
        Nchan = len(self.out_channels.keys())
        for event in tqdm(range(self.sources.nevents)):
            # identify note
            t = self.sources.mapping['time'][event]
            tsamp = int(Nsamp * t)
            chord = self.score.note_sequence[cbin[event]]
            nints = self.score.nintervals[cbin[event]]
            pitch = pitchfrac[event]
            note = chord[int(pitch * nints)]
            sfunc = self.generator.samples[note]
            vol   = self.sources.mapping['volume'][event]
            for i in range(Nchan):
                samplen = self.generator.samplens[note] 
                phi     = self.sources.mapping['phi'][event] * 2 * np.pi
                theta   = self.sources.mapping['theta'][event] * 2 * np.pi
                panenv = self.channels.mics[i].antenna(phi,theta)
                samps = np.arange(samplen)
                values = sfunc(samps) * vol * panenv
                self.out_channels[str(i)].values[tsamp:min(tsamp+samplen, Nsamp-1)] += values[:min(samplen, Nsamp-tsamp-1)]

            
        

        
