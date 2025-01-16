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
from .utilities import const_or_evo, nested_dict_idx_reassign, NoSoundDevice
from .tts_caption import render_caption, get_ttsMode, default_tts_voice
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import ffmpeg as ff
import wavio as wav
import IPython.display as ipd
from IPython.core.display import display
from scipy.io import wavfile
import warnings
import tempfile
from pathlib import Path
try:
    import sounddevice as sd
except (OSError, ModuleNotFoundError) as sderr:
    sd = NoSoundDevice(sderr)
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = list

class Sonification:
    """Representing the overall sonification

    This class combines the data sources, musical score constraints
    and generator together to generate and render the ultimate
    sonification for saving or playing in the :obj:`jupyter-notebook`
    environment 


    Todo:
      * Support custom audio setups here too.
    """
    def __init__(self, score, sources, generator, audio_setup='stereo',
                 caption=None, samprate=48000,
                 ttsmodel=default_tts_voice):
        """
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
    	  applications
         ttsmodel (:obj:`str` or :obj:`PosixPath`) file path to the
          text-to-speech model used for captions. 
        """
        
        # sampling rate in Hz
        self.samprate = samprate
        
        # tts model name
        self.ttsmodel = ttsmodel
        
        # caption
        self.caption = caption
        
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
        if self.score.pitch_binning == 'adaptive':
            pitchfrac[np.argsort(self.sources.mapping['pitch'])] = np.arange(self.sources.n_sources)/self.sources.n_sources
        elif self.score.pitch_binning == 'uniform':
            pitchfrac = np.clip(self.sources.mapping['pitch'], 0, 9.999999e-1)
            
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
                azi     = const_or_evo(sourcemap['phi'], sstream.sampfracs) * 2 * np.pi
            elif 'azimuth' in sourcemap:
                azi     = const_or_evo(sourcemap['azimuth'], sstream.sampfracs) * 2 * np.pi
            else:
                azi     = const_or_evo(self.generator.preset['azimuth'], sstream.sampfracs) * 2 * np.pi
            if 'theta' in sourcemap:
                polar   = const_or_evo(sourcemap['theta'], sstream.sampfracs) * np.pi
            elif 'polar' in sourcemap:
                polar   = const_or_evo(sourcemap['polar'], sstream.sampfracs) * np.pi                
            else:
                polar   = const_or_evo(self.generator.preset['polar'], sstream.sampfracs) * np.pi

            # compute sample indices for truncating notes overshooting sonification length
            trunc_note = min(playlen, lastsamp-tsamp)
            trunc_soni   = trunc_note + tsamp

            # spatialise audio by computing relative volume in each speaker
            for i in range(Nchan):
                panenv = self.channels.mics[i].antenna(azi,polar)
                self.out_channels[str(i)].values[tsamp:trunc_soni] += (sstream.values*panenv)[:trunc_note]

        # produce mono audio of caption, if one is provided
        if str(self.caption or '').strip():
            ttsMode = get_ttsMode() # determine if using coqui-ai or pyttsx3

            # use a temporary directory to ensure caption file cleanup
            with tempfile.TemporaryDirectory() as cdir:
                cpath = Path(cdir, 'caption.wav')
                render_caption(self.caption, self.samprate,
                               self.ttsmodel, str(cpath))
                
                rate_in, wavobj = wavfile.read(cpath)
                wavobj = np.array(wavobj)
            # Set up the Stream objects for TTS
            self.caption_channels = {}
            caption_norm = wavobj.max()
            for c in range(Nchan):
                self.caption_channels[str(c)] = Stream(wavobj.shape[0], self.samprate, ltype='samples')
                
                # place caption straight ahead spatially
                panenv = self.channels.mics[c].antenna(0, 0.5*np.pi)
                
                cnorm = abs(self.out_channels[str(c)].values).max()/caption_norm
                self.caption_channels[str(c)].values += (wavobj*cnorm*panenv)
        else:
            self.caption_channels = {}
            for c in range(Nchan):
                self.caption_channels[str(c)] = Stream(0, self.samprate) 
        
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
            
            # combine caption + sonification streams at display time
            channel_values = np.concatenate([self.out_channels[str(c)].values,
                                self.caption_channels[str(c)].values])   
            
            channels.append(channel_values)
           
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

        # combine caption + sonification streams at display time
        for c in range(len(self.out_channels)):
            tempfname = Path('.', f'.TEMP_{c}.wav')
            self.out_channels[str(c)].values += self.caption_channels[str(c)].values
            wav.write(tempfname, 
                      self.out_channels[str(c)].values,
                      self.samprate, 
                      scale = (-vmax,vmax),
                      sampwidth=3)
            inputs[self.channels.forder[c]] = ff.input(tempfname)
            
        print("Joining temporary .wav files...")
        (
            ff.filter(inputs, 'join', inputs=len(inputs), channel_layout=self.channels.setup)
            .output(fname)
            .overwrite_output()
            .run(quiet=~ffmpeg_output)
        )
        
        print("Cleaning up...")
        for c in range(len(self.out_channels)):
            Path('.', f'.TEMP_{c}.wav').unlink()
            
        print("Saved.")

    def save(self, fname, master_volume=1., embed_caption=True):
        """ Save render as a combined multi-channel wav file 
        
        Can use this function to save sonification of any audio_setup
        to a 32-bit depth WAV using `scipy.io.wavfile`

        Args:
          fname (:obj:`str`) Filename or filepath
          master_volume (:obj:`float`) Amplitude of the largest volume
            peak, from 0-1
          embed_caption (:obj:`bool`) Whether or not to embed caption
            at the start of the output audio

        Todo:
          * Raise `scipy` issue if common 24-bit WAV can be supported
        """

        channels = []
        vmax = 0.
        
        # first pass - find max amplitude value to normalise output
        for c in range(len(self.out_channels)):
            
            channel_values = np.concatenate(int(embed_caption)*[self.caption_channels[str(c)].values,]+
                                            [self.out_channels[str(c)].values])   
            channels.append(channel_values)
            vmax = max(
                abs(channels[c].max()),
                abs(channels[c].min()),
                vmax
            ) * 1.05

        # normalisation for conversion to int32 bitdepth wav
        norm = master_volume * (pow(2, 31)-1) / vmax

        # setup array to house wav stream data 
        chans = np.zeros((channels[0].size, len(channels)), dtype="int32")
        
        # normalise and collect channels into a list
        for c in range(len(self.out_channels)):
            vals = channels[c]
            chans[:,c] = (vals*norm).astype("int32")
            
        # finally combine and write out wav file
        wavfile.write(fname, self.samprate, chans)
        print(f"Saved {fname}")

        
    def notebook_display(self, show_waveform=True):
        """ plot the waveforms and embed player in the notebook

        Show waveforms and embed an audio player in the python
        notebook for direct playback. the notebook player only
        supports up to stereo, so if more than two channels, only the
        first two are used as left and right.
        """

        time = self.out_channels['0'].samples / self.out_channels['0'].samprate

        channels = []
        fig = plt.figure(figsize=(18,12))
        vmax = 0.
        
        # combine caption + sonification streams at display time
        for c in range(len(self.out_channels)):
            channel_values = np.concatenate([self.caption_channels[str(c)].values,
                                             self.out_channels[str(c)].values])   
            channels.append(channel_values)
            vmax = max(
                abs(channels[c].max()),
                abs(channels[c].min()),
                vmax
            ) * 1.05
        
        if show_waveform:
            for i in range(len(self.out_channels)):
                plt.plot(time[::20], self.out_channels[str(i)].values[::20]+2*i*vmax, label=self.channels.labels[i])
            plt.xlabel('Time (s)')
            plt.ylabel('Relative Amplitude')
            plt.legend(frameon=False, loc=5)
            plt.xlim(-time[-1]*0.05,time[-1]*1.2)
            for s in plt.gca().spines.values():
                s.set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
            plt.show()
        
        if len(self.channels.labels) == 1:             
            # we have used 48000 Hz everywhere above as standard, but to quickly hear the sonification sped up / slowed down,
            # you can modify the 'rate' argument below (e.g. multiply by 0.5 for half speed, by 2 for double speed, etc)
            outfmt = np.column_stack(channels*2).T / vmax
        else:
            outfmt = np.column_stack(channels[:2]).T / vmax
        if len(self.channels.labels) > 2:
            print("Warning: for more than two channels, only first two channels are mapped to L and R, respectively.")
        display(ipd.Audio(outfmt,rate=self.out_channels['0'].samprate, autoplay=False))
        
    def hear(self):
        """ Play audio directly to the sound device, for command-line playback.

        If available, use the ``sounddevice`` module to stream the sonification to
        the sound device directly (speakers, headphones, etc.) via the underlying
        ``PortAudio`` C-library. if unavaialable, raise error.

        Todo:
          * Add more options to control the streamed audio
        """

        channels = []
        vmax = 0.
        
        # combine caption + sonification streams at display time
        for c in range(len(self.out_channels)):
            channel_values = np.concatenate([self.caption_channels[str(c)].values,
                                             self.out_channels[str(c)].values])   
            channels.append(channel_values)
            vmax = max(
                abs(channels[c].max()),
                abs(channels[c].min()),
                vmax
            ) * 1.05
                
        if len(self.channels.labels) == 1:             
            # we have used 48000 Hz everywhere above as standard, but to quickly hear the sonification sped up / slowed down,
            # you can modify the 'rate' argument below (e.g. multiply by 0.5 for half speed, by 2 for double speed, etc)
            outfmt = np.column_stack(channels*2)/vmax
        else:
            outfmt = np.column_stack(channels[:2])/vmax

        dur = int(np.round(outfmt.shape[0]/self.out_channels['0'].samprate))
        playback_msg = f"Playing Sonification ({dur} s): "
        print(playback_msg)
        try:
            sd.play(outfmt,self.out_channels['0'].samprate,blocking=1)
        except OSError as error: 
            print(error) 
            print("The Sonification.hear() function requires the PortAudio C-library. This may be missing from your system or \n"
                  "unsupported in this context. This should be installed by pip on Windows and OSx automatically with the \n "
                  "sounddevice library, but on Linux you may need to install manually using e.g.:\n"
                  "\t 'sudo apt-get install libportaudio2.'\n")

    def _make_seamless(self, overlap_dur=0.05):
        """ Make a seamlessly looping audio signal.

        Audio signal is made seamless by cross-fading end of signal back into start
        over a duration (in seconds) defined by ``overlap_dur``

        Args:
          overlap_dur (:obj:`float`): cross-fade duration in seconds.        
        """
        self.loop_channels = {}
        buffsize = int(overlap_dur*self.samprate)
        ramp = np.linspace(0,1, buffsize+1)
        for c in range(len(self.out_channels)):
            self.loop_channels[str(c)] = Stream(self.out_channels[str(c)].values.size - buffsize,
                                                self.samprate, ltype='samples')
            self.loop_channels[str(c)].values = self.out_channels[str(c)].values[:-buffsize]
            self.loop_channels[str(c)].values[:buffsize] *= ramp[:-1]
            self.loop_channels[str(c)].values[:buffsize] += ramp[::-1][:-1] * self.out_channels[str(c)].values[-buffsize:]
            
