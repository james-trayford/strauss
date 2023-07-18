""" :obj:`sonification`: generate sonification, combining submodules.

This Submodule handles the combining of all the constituent
subroutines into  a single :obj:`sonification` object that can then
render and output/save the resultant sonification. This handles
feeding of information between :obj:`strauss` modules, including
taking the :obj:`sources` mapping, applying any musical constraints
from :obj:`score` running the :obj:`generators` to make sound and
combining them into the output channels for the overall spatialised
sonificiation.

Todo:
  * Delegate more musical process to the :obj:`score` module
"""

from .stream import Stream
from .channels import audio_channels
from .utilities import const_or_evo, nested_dict_idx_reassign
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import ffmpeg as ff
import wavio as wav
import IPython.display as ipd
from IPython.core.display import display
from scipy.io import wavfile
import warnings

class Sonification:
    """Representing the overall sonification

    This class combines the data sources, musical score constraints
    and generator together to generate and render the ultimate
    sonification for saving or playing in the :obj:`jupyter-notebook`
    environment 

    Args:
      score (:class:`~strauss.score.Score`): Sonification :obj:`Score`
    	object 
      sources (:class:`~strauss.sources.Source`): Sonification
    	:obj:`Sources` child object (:class:`~strauss.sources.Events`
    	or :class:`~strauss.sources.Objects`)  
      generator (:class:`~strauss.generator.Generator`): Sonification
    	:obj:`Generator` child object
    	(:class:`~strauss.generator.Synthesizer` or
    	:class:`~strauss.generator.Sampler`)
      audio_setup (:obj:`str`) The requested audio setup preset to
    	pass to :class:`~strauss.channels.audio_channels`
      samprate (:obj:`int`) Integer sample rate in samples per second
        (Hz), typically :obj:`44100` or :obj:`48000` for most audio
    	applications. 

    Todo:
      * Support custom audio setups here too.
    """
    def __init__(self, score, sources, generator, audio_setup='stereo', samprate=48000):

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

        # check Generator and Sonification sampling rates match...
        if self.samprate != self.generator.samprate:
            # if not, revert to Generator sampling rate.
            warnings.warn("warning: global and generator sampling rates disagree, " \
            f"reverting to generator value of {self.generator.samprate} Hz")
            self.samprate = self.generator.samprate
        
        # ...and the corresponding Stream objects 
        self.out_channels = {}
        for c in range(self.channels.Nmics):
            self.out_channels[str(c)] = Stream(self.score.length, self.samprate)

    def render(self, downsamp=1):
        """Render the sonification.
        
        Generates the sonification by running the  Synthesizer
        :func:`~strauss.generator.Synthesizer.play` or Sampler
        :func:`~strauss.generator.Sampler.play` functions, and
        combining these into the output channel streams using any
        spatialisation for the specified
        :class:`~strauss.channels.audio_channels`. 

        Args:
          downsamp (optional, :obj:`int`): Optionally downsample
          sources for multi-source sonifications for a quicker test
          render by some integer factor.
        """
        
        # first determine if time is provided, if not assume all start at zero
        # and last the duration of sonification

        if "time" not in self.sources.mapping:
            self.sources.mapping['time'] = [0.] * self.sources.n_sources
            self.sources.mapping['note_length'] = [self.score.length] * self.sources.n_sources
            
        # index each chord
        cbin = np.digitize(self.sources.mapping['time'], self.score.fracbins, 0)
        cbin = np.clip(cbin-1, 0, self.score.nchords-1)

        # pitch rank of each source divided by the number of sources
        pitchfrac = np.empty_like(self.sources.mapping['pitch'])
        pitchfrac[np.argsort(self.sources.mapping['pitch'])] = np.arange(self.sources.n_sources)/self.sources.n_sources

        # get some relevant numbers before iterating through sources
        Nsamp = self.out_channels['0'].values.size
        lastsamp = Nsamp - 1
        Nchan = len(self.out_channels.keys())
        indices = range(0,self.sources.n_sources, downsamp)

        for source in tqdm(indices):

            # index note properties
            t = self.sources.mapping['time'][source]
            tsamp = int(Nsamp * t)
            chord = self.score.note_sequence[cbin[source]]
            nints = self.score.nintervals[cbin[source]]
            pitch = pitchfrac[source]
            note = chord[int(pitch * nints)]
            # make dictionary for feeding to play function with each notes properties
            sourcemap = {}
            # for k in self.sources.mapping.keys():
            #     sourcemap[k] = self.soures.mapping[k][source]
            nested_dict_idx_reassign(self.sources.mapping, sourcemap, source)
            sourcemap['note'] = note

            # run generator to play each note
            sstream = self.generator.play(sourcemap)
            playlen = sstream.values.size
            if 'phi' in sourcemap:
                phi     = const_or_evo(sourcemap['phi'], sstream.sampfracs) * 2 * np.pi
            else:
                phi     = const_or_evo(self.generator.preset['phi'], sstream.sampfracs) * 2 * np.pi
            if 'theta' in sourcemap:
                theta   = const_or_evo(sourcemap['theta'], sstream.sampfracs) * np.pi
            else:
                theta   = const_or_evo(self.generator.preset['theta'], sstream.sampfracs) * np.pi

            # compute sample indices for truncating notes overshooting sonification length
            trunc_note = min(playlen, lastsamp-tsamp)
            trunc_soni   = trunc_note + tsamp

            # spatialise audio by computing relative volume in each speaker
            for i in range(Nchan):
                panenv = self.channels.mics[i].antenna(phi,theta)
                self.out_channels[str(i)].values[tsamp:trunc_soni] += (sstream.values*panenv)[:trunc_note]

    def save_stereo(self, fname, master_volume=1.):
        """ Save stereo or mono sonifications
        
        Can use this function to save :obj:`"stereo"` or :obj:`"mono"`
        sonifications while avoiding ffmpeg processing.

        Args:
          fname (:obj:`str`) Filename or filepath
          master_volume (:obj:`float`) Amplitude of the largest volume
            peak, from 0-1

        Todo:
          * Support :obj:`master_volume` in decibels
        """

        if len(self.out_channels) > 2:
            print("Warning: sonification has > 2 channels, only first 2 will be used. See 'save_combined' method.")
        
        # first pass - find max amplitude value to normalise output
        # and concatenate channels to list
        vmax = 0.
        channels = []
        for c in range(min(len(self.out_channels), 2)):
            vmax = max(
                abs(self.out_channels[str(c)].values.max()),
                abs(self.out_channels[str(c)].values.min()),
                vmax
            ) / master_volume
            channels.append(self.out_channels[str(c)].values)
            
        wav.write(fname, 
                  np.column_stack(channels),
                  self.samprate, 
                  scale = (-vmax,vmax),
                  sampwidth=3)
            
        print("Saved.")

                
    def save_combined(self, fname, ffmpeg_output=False, master_volume=1.):
        """ Save render as a combined multi-channel wav file 
        
        Can use this function to save sonification of any audio_setup,
        using ffmpeg processing, and unscrampling to the correct
        channel order.

        Args:
          fname (:obj:`str`) Filename or filepath
          ffmpeg_output (:obj:`bool`) If True, print :obj:`ffmpeg`
            output to screen 
          master_volume (:obj:`float`) Amplitude of the largest volume
            peak, from 0-1

        Todo:
          * Either find a way to avoid the need to unscramble channle
        	order, or find alternative to save wav files
        """
        # setup list to house wav stream data 
        inputs = [None]*len(self.out_channels)

        # first pass - find max amplitude value to normalise output
        vmax = 0.
        for c in range(len(self.out_channels)):
            vmax = max(
                abs(self.out_channels[str(c)].values.max()),
                abs(self.out_channels[str(c)].values.min()),
                vmax
            ) / master_volume
            
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

    def save(self, fname, master_volume=1.):
        """ Save render as a combined multi-channel wav file 
        
        Can use this function to save sonification of any audio_setup
        to a 32-bit depth WAV using `scipy.io.wavfile`

        Args:
          fname (:obj:`str`) Filename or filepath
          master_volume (:obj:`float`) Amplitude of the largest volume
            peak, from 0-1

        Todo:
          * Raise `scipy` issue if common 24-bit WAV can be supported
        """
        
        # first pass - find max amplitude value to normalise output
        vmax = 0.
        for c in range(len(self.out_channels)):
            vmax = max(
                abs(self.out_channels[str(c)].values.max()),
                abs(self.out_channels[str(c)].values.min()),
                vmax
            )

        # normalisation for conversion to int32 bitdepth wav
        norm = master_volume * (pow(2, 31)-1) / vmax

        # setup array to house wav stream data 
        chans = np.zeros((self.out_channels['0'].values.size,
                          len(self.out_channels)), dtype="int32")
        
        # normalise and collect channels into a list
        for c in range(len(self.out_channels)):
            vals = self.out_channels[str(c)].values
            chans[:,c] = (vals*norm).astype("int32")
            
        # finally combine and write out wav file
        wavfile.write(fname, self.samprate, chans)
        print(f"Saved {fname}")

        
    def notebook_display(self):
        """ plot the waveforms and embed player in the notebook

        Show waveforms and embed an audio player in the python
        notebook for direct playback. the notebook player only
        supports up to stereo, so if more than two channels, only the
        first two are used as left and right.
        """
        time = self.out_channels['0'].samples / self.out_channels['0'].samprate

        vmax = 0.
        for c in range(len(self.out_channels)):
            vmax = max(
                abs(self.out_channels[str(c)].values.max()),
                abs(self.out_channels[str(c)].values.min()),
                vmax
            ) * 1.05
        
        for i in range(len(self.out_channels)):
            plt.plot(time[::20], self.out_channels[str(i)].values[::20]+2*i*vmax, label=self.channels.labels[i])

        plt.xlabel('Time (s)')
        plt.ylabel('Relative Amplitude')
        plt.legend(frameon=False, loc=5)
        plt.xlim(-time[-1]*0.05,time[-1]*1.2)
        for s in plt.gca().spines.values():
            s.set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

        if len(self.channels.labels) == 1:
            # we have used 48000 Hz everywhere above as standard, but to quickly hear the sonification sped up / slowed down,
            # you can modify the 'rate' argument below (e.g. multiply by 0.5 for half speed, by 2 for double speed, etc)
            outfmt = np.column_stack([self.out_channels['0'].values, self.out_channels['0'].values]).T
        else:
            outfmt = np.column_stack([self.out_channels['0'].values, self.out_channels['1'].values]).T
        plt.show()
        display(ipd.Audio(outfmt,rate=self.out_channels['0'].samprate, autoplay=False))
