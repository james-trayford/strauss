from .stream import Stream
from .channels import audio_channels
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import ffmpeg as ff
import wavio as wav

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
        lastsamp = Nsamp - 1
        Nchan = len(self.out_channels.keys())
        for event in tqdm(range(self.sources.nevents)):
            t = self.sources.mapping['time'][event]
            tsamp = int(Nsamp * t)
            chord = self.score.note_sequence[cbin[event]]
            nints = self.score.nintervals[cbin[event]]
            pitch = pitchfrac[event]
            note = chord[int(pitch * nints)]
            sfunc = self.generator.samples[note]
            phi     = self.sources.mapping['phi'][event] * 2 * np.pi
            theta   = self.sources.mapping['theta'][event] * 2 * np.pi

            sourcemap = {}
            for k in self.sources.mapping.keys():
                sourcemap[k] = self.sources.mapping[k][event]
            sourcemap['note'] = note

            sstream = self.generator.play(sourcemap)
            playlen = sstream.values.size
            truncdx = min(playlen, lastsamp-tsamp)
            enddx   = truncdx + tsamp
            for i in range(Nchan):
                panenv = self.channels.mics[i].antenna(phi,theta)
                self.out_channels[str(i)].values[tsamp:enddx] += sstream.values[:truncdx]
        
    def save_combined(self, fname, ffmpeg_output=False):
        """ Save rendered sonification as a combined multi-channel audio file """
        # setup list to house wav stream data 
        inputs = [None]*len(self.out_channels)

        # first pass - find max amplitude value to normalise output
        vmax = 0.
        for c in range(len(self.out_channels)):
            vmax = max(
                abs(self.out_channels[str(c)].values.max()),
                abs(self.out_channels[str(c)].values.min()),
                vmax
            )
            
        print("Creating temporary .wav files...")
        
        for c in range(len(self.out_channels)):
            tempfname = f"./.TEMP_{c}.wav"
            wav.write(tempfname, 
                      self.out_channels[str(c)].values,
                      self.samprate, 
                      scale = (-vmax,vmax),
                      sampwidth=3)
            inputs[self.channels.forder[c]] = ff.input(tempfname)
            
        print("Joning temporary .wav files...")
        (
            ff.filter(inputs, 'join', inputs=len(inputs), channel_layout=self.channels.setup)
            .output(fname)
            .overwrite_output()
            .run(quiet=~ffmpeg_output)
        )
        
        print("Cleaning up...")
        for c in range(len(self.out_channels)):
            os.remove(f"./.TEMP_{c}.wav")
            
        print("Saved.")
