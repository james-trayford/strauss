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
from .sources import Events
from .utilities import const_or_evo, nested_dict_idx_reassign, apply_fades, rescale_values, NoSoundDevice, is_notebook
from .tts_caption import render_caption, get_ttsMode, default_tts_voice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import subprocess as sp
import wavio as wav
import IPython.display as ipd
from IPython.display import display
from scipy.io import wavfile
import warnings
import tempfile
from pathlib import Path
import ffmpeg
try:
    import sounddevice as sd
except (OSError, ModuleNotFoundError) as sderr:
    sd = NoSoundDevice(sderr)
try:
    if is_notebook:
        from tqdm.notebook import tqdm
    else:
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
                 caption=None, samprate=48000, declick_time=0.03,
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
         declick_time (:obj:`float`) duration of start and end fades applied
          on save and dispolay to remove audible clicks from sample
          discontinuity
         ttsmodel (:obj:`str` or :obj:`PosixPath`) file path to the
          text-to-speech model used for captions. 
        """
        
        # sampling rate in Hz
        self.samprate = samprate
        
        # tts model name
        self.ttsmodel = ttsmodel

        # fade duration to de-click audio
        self.declick_time = declick_time
        
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

    def clear(self):
        """
        Clears the audio buffers in all output channels by setting values to 0.
        This prevents audio from accumulating if render() is called multiple times.
        """
        # Iterate over the dictionary of channels (usually '0', '1', etc.)
        for chan in self.out_channels:
            if hasattr(self.out_channels[chan], 'values'):
                self.out_channels[chan].values[:] = 0.
            # Fallback if the channel is a raw numpy array
            elif isinstance(self.out_channels[chan], np.ndarray):
                self.out_channels[chan][:] = 0.
            
    def _assign_notes(self):
        """Determine the note played by each source, and when.

        Combines the :obj:`Sources` `time` and `pitch` mappings with the
        :obj:`Score` chord sequence to decide which note each source
        sounds, and at what point in the sonification. Used by
        :meth:`render`, and by the table methods so that a table can be
        produced without rendering any audio.

        Note:
          As in :meth:`render`, sources with no `time` mapping are all
          assumed to start at zero and last the full sonification.

        Returns:
          notes (:obj:`list(str)`): note played by each source, in
            scientific pitch notation (e.g. :obj:`'A4'`)
          times (:obj:`ndarray`): start time of each source in seconds
        """
        # determine if time is provided, if not assume all start at zero
        # and last the duration of sonification
        if "time" not in self.sources.mapping:
            self.sources.mapping['time'] = [0.] * self.sources.n_sources
            self.sources.mapping['note_length'] = [self.score.length] * self.sources.n_sources

        # index each chord
        cbin = np.digitize(self.sources.mapping['time'], self.score.fracbins, 0)
        cbin = np.clip(cbin-1, 0, self.score.nchords-1)

        # pitch rank of each source divided by the number of sources
        pitch = np.asarray(self.sources.mapping['pitch'])
        pitchfrac = np.empty_like(pitch)
        if self.score.pitch_binning == 'adaptive' and np.unique(pitch).size > 1:
            idxs = np.argsort(pitch)
            pitchfrac[idxs] = np.arange(self.sources.n_sources)/self.sources.n_sources
        else:
            # a single pitch value has no ranking to adapt to - ranking it
            # would spread sources over the chord in whatever order they
            # arrived in, so bin it as a fixed pitch, as uniform binning does
            pitchfrac = np.clip(pitch, 0, 9.999999e-1)

        notes = []
        for source in range(self.sources.n_sources):
            chord = self.score.note_sequence[cbin[source]]
            nints = self.score.nintervals[cbin[source]]
            notes.append(chord[int(pitchfrac[source] * nints)])

        # mapped time is a fraction of the sonification length
        times = np.array(self.sources.mapping['time']) * self.score.length

        return notes, times

    def render(self, downsamp=1, progress=True):
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

        # first, clear the audio channels
        self.clear()
        
        # determine the note played by each source and when it starts
        # (this also defaults the time mapping, if none was provided)
        notes, _ = self._assign_notes()

        # get some relevant numbers before iterating through sources
        Nsamp = self.out_channels['0'].values.size
        lastsamp = Nsamp - 1
        Nchan = len(self.out_channels.keys())
        indices = range(0,self.sources.n_sources, downsamp)

        if progress:
            print('Processing sonification..')
        for source in tqdm(indices) if progress else indices:

            # index note properties
            t = self.sources.mapping['time'][source]
            tsamp = int((Nsamp-1) * t)
            note = notes[source]

            # make dictionary for feeding to play function with each notes properties
            sourcemap = {}
            # for k in self.sources.mapping.keys():
            #     sourcemap[k] = self.soures.mapping[k][source]
            nested_dict_idx_reassign(self.sources.mapping, sourcemap, source)

            sourcemap['note'] = note

            # run generator to play each note
            sstream = self.generator.play(sourcemap)
            playlen = sstream.values.size

            # place source on listener plane (quarter rotation) by default
            polar = 0.5 * np.pi
            if 'pan' in sourcemap:
                # in pan mode, put everything on the 
                azi     = (const_or_evo(sourcemap['pan'], sstream.sampfracs) + 0.5) * np.pi
            else:
                # TODO: generic handling of alias parameters (beyond 3D angles)
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


    def _check_can_tabulate(self):
        """Check the sources carry the mapped values a table needs."""
        if not getattr(self.sources, 'mapped_samples', {}):
            raise Exception("Sources have no mapped values to tabulate - run "
                            "'apply_mapping_functions' on the sources first.")

    def event_table(self, include_input=False):
        """Tabulate the events of the sonification.

        Produces a table with a row per event, giving the time at which
        it sounds, the note played, and the value of each user-specified
        mapped parameter. Parameters added automatically or held at fixed
        values are excluded, and are instead listed by
        :meth:`fixed_table`.

        Note:
          The `time` and `note` columns replace any mapped `time` and
          `pitch` parameters, which are internal fractions rather than
          what is heard - `time` is the time of the event in the
          sonification in seconds, and `note` the note it ultimately
          sounds.

        Args:
          include_input (`optional`, :obj:`bool`): if True, also give the
            input data value of each parameter, before mapping, in a
            column suffixed `'_input'`.

        Returns:
          table (:obj:`pandas.DataFrame`): one row per event
        """
        self._check_can_tabulate()
        if not isinstance(self.sources, Events):
            raise TypeError(f"'event_table' is for Events sources, but these "
                            f"sources are {type(self.sources).__name__}.")

        notes, times = self._assign_notes()
        table = {'time': times, 'note': notes}

        for key in self.sources.mapped_quantities:
            if self.sources.origin.get(key, 'mapped') != 'mapped':
                continue
            if key not in ('time', 'time_evo', 'pitch'):
                # time and pitch are already given by the time and note
                # of each row, in the terms actually heard
                table[key] = np.asarray(self.sources.mapped_samples[key])
            if include_input:
                table[f'{key}_input'] = np.asarray(self.sources.raw_mapping[key])

        return pd.DataFrame(table)

    def fixed_table(self):
        """Tabulate parameters the user did not map.

        Companion to :meth:`event_table`, listing the parameters the
        user did not map - those held at a fixed value, and those
        STRAUSS assigned itself where no mapping was given (`'fixed'`
        and `'auto'` in the `origin` column, respectively) - alongside
        the value each takes.

        Note:
          Parameters varying from source to source (e.g. the `pitch`
          assigned to each Object of a chord) have no one value to
          report, and are left out.

        Returns:
          table (:obj:`pandas.DataFrame`): one row per unmapped parameter
        """
        self._check_can_tabulate()

        rows = []
        for key in self.sources.mapped_quantities:
            origin = self.sources.origin.get(key, 'mapped')
            if origin == 'mapped':
                continue
            values = np.unique(np.asarray(self.sources.mapped_samples[key]))
            if values.size != 1:
                # not held at one value, so don't claim it is
                continue
            rows.append({'parameter': key,
                         'value': values[0],
                         'origin': origin})

        return pd.DataFrame(rows, columns=['parameter', 'value', 'origin'])

    def add_ticks(self, increment, duration=0.04, tick_vol=0.25):
        # TODO this should probably use a dedicated generator...

        # add tick volume to Sonification object
        self.tick_vol = tick_vol
        
        tick_samples = 2*(np.random.random(self.out_channels['0'].values.shape)-0.5)
        k = 'time'
        if k not in self.sources.lims.keys():
            k = 'time_evo'
            if k not in self.sources.lims.keys():
                raise Exception("""
                Sonification doesn't have a time base! only sonifications with a 'time'
                or 'time_evo' mapping can have time increment ticks...
                """)
        inc = self.score.length*rescale_values(self.sources.lims[k][0]+increment,
                                               self.sources.lims[k],
                                               self.sources.plims[k])
        self.t_per_inc = np.linspace(0, self.score.length/inc, tick_samples.shape[0])
        self.tdur_per_inc = inc/duration
        tickenv = np.clip(1/self.tdur_per_inc - self.t_per_inc%1, 0, np.inf)
        tickenv /= tickenv.max()
        tick_samples = tick_samples*tickenv
        Nchan = len(self.out_channels.keys())
        self.tick_channels = {}
        for i in range(Nchan):
            panenv = self.channels.mics[i].antenna(0, 0.5*np.pi)
            self.tick_channels[str(i)] = Stream(tick_samples.size, self.samprate, ltype='samples')
            self.tick_channels[str(i)].values += tick_samples * panenv

                
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
        to a file. This first creates a 32-bit depth WAV using
        `scipy.io.wavfile`. If fname has a non-WAV extension, it then attempts
        conversion via ffmpeg, provided ffmpeg is available.
        
        formats

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

        has_ticks = hasattr(self, 'tick_channels')

        # first pass - find max amplitude value to normalise output
        for c in range(len(self.out_channels)):
                
            channel_values = np.concatenate(int(embed_caption)*[self.caption_channels[str(c)].values,]+
                                            [apply_fades(self.out_channels[str(c)].values,
                                                         self.out_channels['0'].samprate,
                                                         fdur=self.declick_time)])
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
            signal = channels[c]*norm
            if has_ticks:
                # add the ticks
                signal += self.tick_channels[str(c)].values*norm*self.tick_vol
            chans[:,c] = (signal).astype("int32")
            
        # finally combine and write out file. first check extension
        fsplit = str(fname).split('.')
        if len(fsplit) < 2:
            warnings.warn('No file extension in provided fname. Assuming WAV...')
        ext = fsplit[-1].lower()
        if ext != 'wav':
            # check we can use ffmpeg binary 
            try:
                sp.run(['ffmpeg','-h'],capture_output=1, check=1)
            except FileNotFoundError as e: 
                raise FileNotFoundError(f"""
                'ffmpeg' doesn't appear to be available in the local environment.
                This may need to be installed manually. To install ffmpeg visit
                https://www.ffmpeg.org/download.html.
                {str(e)}
                """)
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                # now first write the wav to a temporary file
                wavfile.write(tmp.name, self.samprate, chans)
                try:
                    # try (naive) convert with ffmpeg
                    sp.run(['ffmpeg', '-i', f'{tmp.name}', f'{fname}'],
                           capture_output=1, check=1)
                except sp.CalledProcessError as e:
                    # if ffmpeg can't do it for whatever reason, raise
                    raise Exception(f"""
                    'ffmpeg' failed to convert '.wav' to '.{ext}' succesfully:
                    {str(e)}
                    {e.stderr}""")
        else:
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

        has_ticks = hasattr(self, 'tick_channels')
        channels = []
        fig = plt.figure(figsize=(18,12))
        vmax = 0.
        
        # combine caption + sonification streams at display time
        for c in range(len(self.out_channels)):
            # apply fades at display time
            channel_values = np.concatenate([self.caption_channels[str(c)].values,
                                             apply_fades(self.out_channels[str(c)].values,
                                                         self.out_channels['0'].samprate,
                                                         fdur=self.declick_time)])   
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
        if has_ticks:
            # add the ticks
            for c in range(outfmt.shape[0]):
                outfmt[c] += self.tick_channels['0'].values*self.tick_vol / vmax
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
            
    def _make_out_array(self, master_volume=1., embed_caption=True):
        channels = []
        vmax = 0.

        has_ticks = hasattr(self, 'tick_channels')

        # first pass - find max amplitude value to normalise output
        for c in range(len(self.out_channels)):
                
            channel_values = np.concatenate(int(embed_caption)*[self.caption_channels[str(c)].values,]+
                                            [apply_fades(self.out_channels[str(c)].values,
                                                         self.out_channels['0'].samprate,
                                                         fdur=self.declick_time)])
            channels.append(channel_values)
            vmax = max(
                abs(channels[c].max()),
                abs(channels[c].min()),
                vmax
            ) * 1.05

        # normalisation for conversion to int32 bitdepth wav
        norm = master_volume * (pow(2, 31)-1) / vmax

        # setup array to house wav stream data 
        chans = np.zeros((channels[0].size, len(channels)))
        
        # normalise and collect channels into a list
        for c in range(len(self.out_channels)):
            signal = channels[c]*norm
            if has_ticks:
                # add the ticks
                signal += self.tick_channels[str(c)].values*norm*self.tick_vol
            chans[:,c] = (signal)
        return chans
