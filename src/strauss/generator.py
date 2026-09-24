""" The :obj:`generator` submodule: creating sounds for the sonification.

This submodule handles the actual generation of sound for the
sonification, after parameterisation by the :obj:`Sources` and musical
choices dictated by the :obj:`Score`.

Todo:
    * Consolidate more common code into the :obj:`Generator` parent
      class.
    * Support more Envelope and LFO types in the :obj:`play` methods
      (want pitch, volume and filter options for each)
    * Check buffer length consistency for spectralizer - do we hit
      grid points?
    * Throw appropriate errors when rendering with unreasonable length
      and freq combinations
"""

from . import stream
from . import notes
from . import presets
from . import utilities as utils
from . import filters
import numpy as np
import scipy
# can we use FFTW backend in scipy?
try:
    import pyfftw
    scipy.fft.set_backend(pyfftw.interfaces.scipy_fft)
except (OSError, ModuleNotFoundError):
    pass
from scipy.fft import fft, rfft, ifft, fftfreq, rfftfreq
import glob
import copy
import scipy
import json
from scipy.io import wavfile
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
import logging
from sf2utils.sf2parse import Sf2File
from pathlib import Path
import os
import re
import hashlib
from .tts_caption import render_caption, get_ttsMode, TTSIsNotSupported

# ignore wavfile read warning that complains due to WAV file metadata
warnings.filterwarnings("ignore", message=r"Chunk \(non-data\) not understood, skipping it\.")

# TO DO:
# - Ultimately have Synth and Sampler classes that own their own stream (stream.py) object
#   allowing ADSR volume and filter enveloping, LFO implementation etc.
# - Functions here will generally be called from a "Score" class that is provided with the
#   musical choices and uses these to generate sound, but can be interfaced with directly.

def forward_loopsamp(s, start, end):
    """Produce array of sample indices for looping a sample forward.

    Sample indices between values `start` and `end` that will loop the sample
    such that it loops "forward", i.e. start, start+1, ..., end-1, end, start,
    ... etc.

    Args:
      s (:obj:`ndarray`): array of input sample indices
      start (:obj:`int`): Index of sample after which looping should commence
      end (:obj:`int`): Index of sample after which audio loops

    Returns:
      out (:obj:`ndarray`): array of output sample indices
    """
    delsamp = end-start
    return np.piecewise(s, [s < start, s >= start],
                        [lambda x: x, lambda x: (x-start)%(delsamp) + start])
def forward_back_loopsamp(s, start, end):
    """Produce array of sample indices for looping a sample forward-back.

    Sample indices between values `start` and `end` that will loop the sample
    such that it loops "forward-back", i.e. `start, start+1, ..., end-1, end,
    end-1, ..., start+1, start, start+1, ...` etc.
    ... etc.

    Args:
      s (:obj:`ndarray`): array of input sample indices
      start (:obj:`int`): Index of sample after which looping should commence
      end (:obj:`int`): Index of sample after which audio loops

    Returns:
      out (:obj:`ndarray`): array of output sample indices
    """
    delsamp = end-start
    return np.piecewise(s, [s < start, s >= start],
                        [lambda x: x, lambda x: end - abs((x-start)%(2*(delsamp)) - (delsamp))])

class Generator:
    """Generic generator Class, defining common code for child classes

    Generators have common initialisation and methods that are
    defined by this parent class.

    Attributes:
      samprate (:obj:`int`): Samples per second of audio stream (Hz)
      audbuff (:obj:`int`): Samples per audio buffer
      preset (:obj:`dict`): Dictionary of parameters defining the
        generator.
      supports_aliases (:obj:`bool`): whether the generator can play
        sounds named by an alias in the :obj:`Score` (see
        :meth:`Sampler.resolve_alias`) rather than a note.
      makes_aliases (:obj:`bool`): whether the generator can make
        a sound it doesn't have, when asked (see
        :meth:`Speech.make_aliases`).
      sounds_have_pitch (:obj:`bool`): whether the generator's sounds
        have a  meaningful single pitch. Otherwise (for spoken
        phrases, say), the note a sound is filed under is an internal
        detail and is left out of the sonification tables.

    """
    # generators whose sounds can be named by alias override this
    supports_aliases = False

    # ...and those that can make a named sound on demand, this
    makes_aliases = False

    # generators whose sounds carry no pitch to report override this
    sounds_have_pitch = True

    def __init__(self, params={}, samprate=48000):
        """
        Args:
    	params (`optional`, :obj:`dict`): any generator parameters
    	  that differ from the generator :obj:`preset`, where keys and
    	  values are parameter names and values respectively. 
    	samprate (`optional`, :obj:`int`): the sample rate of
  	  the generated audio in samples per second (Hz)
        """
        self.samprate = samprate

        # samples per buffer (use 30Hz as minimum)
        self.audbuff = self.samprate / 30.
        
        # modify or load preset if specified
        if params:
            self.preset = self.modify_preset(params)

    def load_preset(self, preset='default'):
        """Load parameters from a preset YAML file.

        Wrapper method for the :obj:`presets.synth.load_preset` or
        :obj:`presets.sampler.load_preset` functions. Always load the
        :obj:`default` preset first to ensure all parameters are
        defined, and then if necessary reload parameters defined by
        :obj:`preset`

        Args:
          preset (:obj:`str`): name of the preset. Built-in presets
            can be named directly and looks to import the preset from
            the :obj:`<MODULE_PATH>/presets/<GENERATOR>/` directory as
            :obj:`<preset>.yml`, where :obj:`<GENERATOR>` is either
            `synth` or `sampler`, :obj:`<MODULE_PATH>` is the path to
            the strauss module (i.e. :obj:`strauss.__file__`). Custom
            presets can also  be loaded from :obj:`<preset>.yml` if
            :obj:`preset` represents a path containing file
            separators.
        """
        if not hasattr(self, preset):
            self.preset = getattr(presets, self.gtype).load_preset('default')
            self.preset['ranges'] = getattr(presets, self.gtype).load_ranges() 
        if preset != 'default':
            preset = getattr(presets, self.gtype).load_preset(preset)
            self.modify_preset(preset)

    def modify_preset(self, parameters, cleargroup=[]):
        """Modify parameters within current preset

        method allows user to tweak generator parameters directly,
        using a dictionary of parameters and their values. subgroups
        within the preset are represented as nested dictionaries. 

        Args:
          parameters (:obj:`dict`): keys and items are the preset
            parameter names and new values. Nested dictionaries are
            used to redefine grouped parameters,
            e.g. :obj:`{'volume_envelope':{'A':0.5}}`
          cleargroup (optional :obj:`list(str)`): if required, list of
            the group names to be completely reset (e.g. if defining
            new :obj:`oscillators` set for synth). 
        """
        utils.nested_dict_reassign(parameters, self.preset)
        for grp in cleargroup:
            if grp in parameters:
                for k in list(self.preset[grp].keys()):
                    if k not in parameters[grp]:
                        del self.preset[grp][k]
            
    def preset_details(self, term="*"):
        """Print the names and descriptions of presets

        Wrapper for preset_details function. lists the name and description
        of built-in presets with names matching the search term.

        Args:
          term (optional, :obj:`str`): name or glob term for built-in
          preset(s). Default '*' prints all.
        """
        getattr(presets, self.gtype).preset_details(name=term)

    def envelope(self, samp, params, etype='volume'):
        """Envelope function for modulating a single note

        The envelope function takes the pre-defined envelope
        parameters for the specified envelope type and returns the
        envelope value at each sample. envelopes are defined by
        attack, decay, sustain and release (:obj:`'A','D','S' & 'R`)
        values, as well as segment curvatures (:obj:`'Ac','Dc', &
        'Rc`) and a normalisation :obj:`'level'`. See `this article <https://learnmusicproduction.in/blogs/music-production-and-audio-engineering/adsr-fundamentals-in-music-everything-you-need-to-know>`_ for a more detailed explanation of ADSR envelopes.

        Args:
          samp (:obj:`array-like`): Audio sample index
          params (:obj:`dict`): Keys and values of generator
            parameters
          etype (optional , :obj:`str`): type of envelope, indicating
            which :obj:`params` group to read (i.e. if
            :obj:`etype='volume'`, read `from :obj:`volume_envelope`)          
        """
        # TO DO: is it worth it to pre-set this in part if parameters don't change?
        nlen=params['note_length']
        edict=params[f'{etype}_envelope']
        
        # read envelope params from dictionary
        a = edict['A']
        d = edict['D']
        s = edict['S']
        r = edict['R']
        a_k = edict['Ac']
        d_k = edict['Dc']
        r_k = edict['Rc']
        lvl = edict['level']
        
        # effective input sample times, clipped to ensure always defined
        sampt = samp/self.samprate
        
        # handy time values
        t1 = a 
        t2 = a+d
        t3 = nlen+r

        # determine segments and envelope value when note turns off
        a_seg = lambda t: 1-self.env_segment_curve(t, a, 1, -a_k)
        d_seg = lambda t: s+self.env_segment_curve(t-t1, d, 1-s, d_k)
        s_seg = lambda t: s
        o_seg = lambda t: 0.

        if nlen < t1:
            env_off = a_seg(nlen)
        elif nlen < t2:
            env_off = d_seg(nlen)
        else:
            env_off = s

        r_seg = lambda t: self.env_segment_curve(t-nlen, r, env_off, r_k)

        # conditionals to determine which segment a sample is in
        a_cond = sampt < t1
        d_cond = np.logical_and(sampt<min(t2,nlen), sampt>=t1)
        s_cond = np.logical_and(sampt<nlen, sampt>=min(t2,nlen))
        r_cond = sampt >= nlen
        o_cond = sampt >= t3

        # compute envelope for each sample 
        env =  np.piecewise(sampt,
                            [a_cond, d_cond, s_cond, r_cond, o_cond],
                            [a_seg, d_seg, s_seg, r_seg, o_seg])
        return lvl*env

    def env_segment_curve(self, t, t1, y0, k):
        """Formula for segments of the envelope function
        
        Function to evaluate the segments of the envelope, allowing
        for curvature, i.e. concave & convex envelope segments.

        Args:
          t (:obj:`float`): time of each sample along segment
          t1 (:obj:`float`): time of segment endpoint
          y0 (:obj:`float`): starting value of segment
          k (:obj:`float`): curvature value of segment, from (-1,1),
            with positive values indicating concave and negative
            convex curvature.
        """
        return y0/(1 + (1-k)*t / ((k+1)*(t1-t)))

    
    # ||||||||||||||||||||||||||||||||||||||||||||||||||
    # OSC types 
    # ||||||||||||||||||||||||||||||||||||||||||||||||||
    def sine(self, s,f,p):
        """Sine-wave oscillator

        Args:
          s (:obj:`array`-like): sample index
          f (:obj:`float`): samples per cycle
          p (:obj:`float` or :obj:`str`): if numerical, phase in units
            of cycles, :obj:`'random'` indicates randomised.
        Returns:
          v (:obj:`array`-like): values for each sample
        """
        return np.sin(2*np.pi*(s*f+p))
    
    def saw(self,s,f,p):
        """Sawtooth-wave oscillator

        Args:
          s (:obj:`array`-like): sample index
          f (:obj:`float`): samples per cycle
          p (:obj:`float` or :obj:`str`): if numerical, phase in units
            of cycles, :obj:`'random'` indicates randomised.
        Returns:
          v (:obj:`array`-like): values for each sample
        """
        return (2*(s*f+p) +1) % 2 - 1
    
    def square(self,s,f,p):
        """Square-wave oscillator

        Args:
          s (:obj:`array`-like): sample index
          f (:obj:`float`): samples per cycle
          p (:obj:`float` or :obj:`str`): if numerical, phase in units
            of cycles, :obj:`'random'` indicates randomised.
        Returns:
          v (:obj:`array`-like): values for each sample
        """
        return np.sign(self.saw(s,f,p))
    
    def tri(self,s,f,p):
        """Triangle-wave oscillator

        Args:
          s (:obj:`array`-like): sample index
          f (:obj:`float`): samples per cycle
          p (:obj:`float` or :obj:`str`): if numerical, phase in units
            of cycles :obj:`'random'` indicates randomised.
        Returns:
          v (:obj:`array`-like): values for each sample
        """
        return 1 - abs((4*(s*f+p) +1) % 4 - 2)
    def noise(self,s,f,p):
        """White noise oscillator

        Note:
          :obj:`f` and :obj:`p` have no efffect for this oscillator,
          generating a random value for each sample.
        Args:
          s (:obj:`array`-like): sample index
          f (:obj:`float`): unused
          p (:obj:`float` or :obj:`str`): unused
        Returns:
          v (:obj:`array`-like): values for each sample
        """
        return np.random.random(np.array(s).size)*2-1

    def lfo(self, samp, sampfrac, params, ltype='pitch'):
        """Low-Frequency oscillator (LFO)

        This function takes the pre-defined LFO parameters (if
        switched on) for the specified LFO type and returns the
        LFO value at each sample. LFOs are defined by the same values
        as :meth:`strauss.generator.envelope`, with an additional
        :obj:`use` switch, a waveform (:obj:`wave`,
        e.g. :obj:`sine`), an amplitude (:obj:`amount`), a frequncy in
        Hz (:obj:`freq`) a frequency shift in octaves
        (:obj:`freq_shift`) and a :obj:`phase`, either numerical in
        cycles or :obj:`'random'` to indicate randomised.

        Note:
          To modulate the frequency of an oscillator, use the
          :obj:`freq_shift` parameter, rather than :obj:`freq`

        Args:
          samp (:obj:`array-like`): Audio sample index
          sampfrac (:obj:`array-like`): Audio sample as fraction of
            total number of samples
          params (:obj:`dict`): Keys and values of generator
            parameters
          ltype (optional , :obj:`str`): type of LFO, indicating which
            :obj:`params` group to read (i.e. if :obj:`ltype='volume',
            read `from :obj:`pitch_envelope`)

        Returns:
          v (:obj:`array-like`): amplitude of LFO at each input sample
        """
        env_dict = {}
        lfo_key = f'{ltype}_lfo'
        lfo_params = params[lfo_key]
        
        env_dict['note_length'] = params['note_length']
        env_dict['lfo_envelope'] = lfo_params

        freq = lfo_params['freq']/self.samprate
        effsamp = samp.astype(float)
        
        if callable(lfo_params['freq_shift']):
            findex = lfo_params['freq_shift'](sampfrac)
            effsamp = np.cumsum(pow(2, findex))
        elif lfo_params['freq_shift'] != 0:
            effsamp *=  pow(2,lfo_params['freq_shift'])
            
        if callable(lfo_params['amount']):
            amnt  = lfo_params['amount'](sampfrac)
        else:
            amnt = lfo_params['amount']

        if lfo_params['phase'] == 'random':
            phase = np.random.random()
        else:
            phase =lfo_params['phase']

        osc = getattr(self,lfo_params['wave'])(effsamp, freq, phase)
        env = self.envelope(samp, env_dict, 'lfo')
        return amnt * env * osc
        
class Synthesizer(Generator):
    """Synthesizer generator class

    This generator class synthesises sound using mathmatically
    generated waveforms or 'oscillators', from a combination of
    oscillator methods defined in the parent class. The relative
    frequency, phase and amplitude of these oscillators are defined in
    the preset, and linearly combined to produce the sound. defines
    attribute :obj:`self.gtype = 'synth'`.

    Attributes:
      gtype (:obj:`str`): Generator type 

    Todo:
    	* Add other synthesiser types, aside from additive (e.g. FM,
    	  vector, wavetable)? 
    """
    def __init__(self, params=None, samprate=48000):
        """
        Args:
    	  params (`optional`, :obj:`dict`): any generator parameters
    	    that differ from the generator :obj:`preset`, where keys
            and values are parameters names and values respectively. 
    	  samprate (`optional`, :obj:`int`): the sample rate of
  	    the generated audio in samples per second (Hz)
        """
        # default synth preset
        self.gtype = 'synth'
        self.preset = getattr(presets, self.gtype).load_preset()
        self.preset['ranges'] = getattr(presets, self.gtype).load_ranges() 
        
        # universal initialisation for generator objects:
        super().__init__(params, samprate)

        # set up the oscillator banks
        self.setup_oscillators()

    def setup_oscillators(self):
        """Setup and consolidate oscs into a two-variable function.

        Reads the parameterisation of each oscillator from the preset,
        specifying their waveform (:obj:`wave`), relative amplitude
        (:obj:`level`), detuning in cents (:obj:`det`) and
        :obj:`phase`, either a number in units of cycles, or a string
        specifying randomisation (:obj:`'random'`). Sets the
        :obj:`self.generate` method, using the
        :obj:`self.combine_oscs`.

        Note:
          This is deprecated and will likely be removed from future
          versions
        """
        # oscdict = self.preset['oscillators']
        # self.osclist = []
        # for osc in oscdict.keys():
        #     lvl = oscdict[osc]['level']
        #     det = oscdict[osc]['detune']
        #     phase = oscdict[osc]['phase']
        #     form = oscdict[osc]['form']
        #     snorm = self.samprate
        #     fnorm = (1 + det/100.)
        #     if phase == 'random':
        #         oscf = lambda samp, f: lvl * getattr(self,form)(samp/snorm, f*fnorm, np.random.random())
        #     else:
        #         oscf = lambda samp, f: lvl * getattr(self,form)(samp/snorm, f*fnorm, phase)
        #     self.osclist.append(oscf)
        #     flg += 1
        self.generate = self.combine_oscs

    def modify_preset(self, parameters, clear_oscs=True):
        """Synthesizer-specific wrapper for the modify_preset method.

        This gives control over whether or not to clear the arbitrary
        number of oscillators for synthesizer.
        
        Args:
          parameters (:obj:`dict`): keys and items are the preset
            parameter names and new values. Nested dictionaries are
            used to redefine grouped parameters,
            e.g. :obj:`{'volume_envelope':{'A':0.5}}`
          clear_oscs (optional, :obj:`bool`): if True, clear all
            oscillators from the existing preset. Turn off if just
            wishing to tweak non-oscillator parameters.
        """
        if clear_oscs:
            super().modify_preset(parameters, ['oscillators'])
        else:
            super().modify_preset(parameters)
        self.setup_oscillators()
            
    def combine_oscs(self, s, f):
        """ Evaluate and linearly combine oscillators.

        Args:
          s (:obj:`array`-like): Sample index
          f (:obj:`float` or :obj:`str`): If numerical, frequency in
            cycles per second, if string, note name in scientific pitch
            notation (e.g. :obj:`'A4'`)
        Returns:
          tot (:obj:`array`-like): values for each sample
        """
        tot = 0.
        oscdict = self.preset['oscillators']
        if isinstance(f, str):
            # we want a numerical frequency to generate tone
            f = notes.parse_note(f)
        for osc in oscdict:
            lvl = oscdict[osc]['level']
            det = oscdict[osc]['detune']
            phase = oscdict[osc]['phase']
            form = oscdict[osc]['form']
            snorm = self.samprate
            fnorm = (1 + det/100.)
            if phase == 'random':
                tot += lvl * getattr(self,form)(s/snorm, f*fnorm, np.random.random())
            else:
                tot += lvl * getattr(self,form)(s/snorm, f*fnorm, phase)
            # self.osclist.append(oscf)
            # flg += 1
        return tot

    def play(self, mapping):
        """ Play the sound for a given source.

        Play a given source and return the sample values for
        combination into the overall sonification.

        Note:
          :obj:`mapping` is a linear dictionary (not nested, as for 
          :meth:`strauss.generator.modify_preset`) where group members
          are indicated using :obj:`'/'` notation
          (e.g. :obj:`{'volume_envelope/A': 0.5, ...`).

        Args:
          mapping (:obj:`dict`): keys and items are generator
            parameter names and their values. This combines all the
            preset mapped parameters, overwritten by any
            :obj:`Source`-mapped parameters (represented as values or
            interpolation functions for static and evolving
            parameters, respectively). This is a linear dictionary
            (not nested, see :meth:`strauss.generator.modify_preset`)
            where group members are indicated using :obj:`'/'`
            notation (e.g. :obj:`{'volume_envelope/A': 0.5, ...`).

        """
        samprate = self.samprate
        audbuff = self.audbuff

        params = copy.deepcopy(self.preset)
        utils.linear_to_nested_dict_reassign(mapping, params)

        nlength = (params['note_length']+params['volume_envelope']['R'])*samprate
        
        # generator stream (attribute of stream?)
        sstream = stream.Stream(nlength/samprate, samprate)
        samples = sstream.samples
        sstream.get_sampfracs()

        pindex  = np.zeros(samples.size)
        if callable(params['pitch_shift']):
            pindex += params['pitch_shift'](sstream.sampfracs)/12.
        elif params['pitch_shift'] != 0:
            pindex += params['pitch_shift']/12.
        if params['pitch_lfo']['use']:
            pindex += self.lfo(samples, sstream.sampfracs, params, 'pitch')/12.
        if np.any(pindex):
            samples = np.cumsum(pow(2., pindex))
            # if callable(params['pitch_shift']):
            #     samples = np.cumsum(pow(2., pindex))
            # else:
            #     samples = samples * pow(2., pindex)
        
        # generate stream values
        values = self.generate(samples, params['note'])

        # get volume envelope
        env = self.envelope(sstream.samples, params)
        if params['volume_lfo']['use']:
            env *= np.clip(1.-self.lfo(sstream.samples, sstream.sampfracs,
                                       params, 'volume')*0.5, 0, 1)
        
        # apply volume normalisation or modulation (TO DO: envelope, pre or post filter?)
        sstream.values = values * utils.const_or_evo(params['volume'], sstream.sampfracs) * env

        # filter stream
        if params['filter'] == "on":
            if hasattr(params['cutoff'], "__iter__"):
                # if static cutoff, use minimum buffer count
                sstream.bufferize(sstream.length/4)
            else:
                # 30 ms buffer (hardcoded for now)
                sstream.bufferize(0.03)
            sstream.filt_sweep(getattr(filters, params['filter_type']),
                               utils.const_or_evo_func(params['cutoff']))
        return sstream    
            
class Sampler(Generator):
    """Sampler generator class

    This generator class generates sound using pre-loaded audio
    samples, representing different notes. Presets define parameters
    controlling these defines
    attribute :obj:`self.gtype = 'sampler'`.

    Attributes:
      gtype (:obj:`str`): Generator type
      aliases (:obj:`dict`): name each loaded sample can be called by in
        a :obj:`Score`, alongside the sample's home note. Defaults to the
        filename stem (e.g. `'snare'` for `snare.wav`), and can be
        changed with :meth:`set_aliases`.

    Todo:
    	* Add zone mapping for samples (e.g. allow a sample to define
          a range of notes played at different speeds).
        * Have sample loading defined via the preset rather than the
          :obj:`sampfiles` variable?
    """
    supports_aliases = True

    def __init__(self, sampfiles, params=None, samprate=48000, sf_preset=None,
                 assign_notes='sequential'):
        """
        Args:
          sampfiles (`required`, :obj:`str`): string pointing to samples
            to load. This can either point to a directory containing
            samples, where `"path/to/samples"` contains files named
            as `samples_A#4.wav` (ie. `<lowest_directory>_<note>.wav`),
            or a *Soundfont* file, with extension `.sf2`.
    	  params (`optional`, :obj:`dict`): any generator parameters
    	    that differ from the generator :obj:`preset`, where keys and
    	    values are parameters names and values respectively. 
    	  samprate (`optional`, :obj:`int`): the sample rate of
  	    the generated audio in samples per second (Hz)
          sf_preset (`optional`, :obj:`int`) if using a *Soundfont*
            (`.sf2`) file, this is the number of the preset to use.
            All `.sf2` files should contain at least one preset. When
            given default `None` value, will print available presets
            and select the first preset. Note presets are 1-indexed.
          assign_notes (`optional`, :obj:`str`): how samples whose
            filenames carry no note tag are given a home note. With
            `'sequential'` (default) they take consecutive semitones from
            `C3` in filename order, skipping any note a tagged sample has.
            With `'detect'` the note is found from the sample's strongest
            frequency, for tuned samples that are simply untagged.
        
        Note:
          Each sample is assigned a home note in order to choose between
          samples based on the ``pitch`` parameter, following a similar
          approach to a [keyboard sampler]
          (https://support.apple.com/en-lk/guide/logicpro/lgcp4eecaaff/mac)
          where each key can triggers a different chosen sample. If
          `drumset_C1.wav` is a kick drum and `drumset_D1.wav` is a snare
          drum for :obj:`Score` with `chord_sequence=[["C1", "D1"]]`, events
          mapped to a higher (lower) `pitch` will sound as snare (kick) drums.

          Samples whose filename carries no note tag (e.g. `kick.wav`) are
          assigned a home note automatically (see `assign_notes`), and
          can be named in the
          :obj:`Score` by their `alias` instead - the filename stem by
          default, so `chord_sequence=[["kick", "snare"]]`. See
          :meth:`info` to list the assignments and :meth:`set_aliases`
          to change them.

        """
        # default sampler preset
        self.gtype = 'sampler'
        self.preset = getattr(presets, self.gtype).load_preset()
        self.preset['ranges'] = getattr(presets, self.gtype).load_ranges() 
        self.sampsource = None
        self.sampfiles = sampfiles
        self.sf_preset = sf_preset
        self.sf_preset_name = None
        self.sf_note_range = []
        if assign_notes not in ('sequential', 'detect'):
            raise ValueError(f"assign_notes '{assign_notes}' not recognised, "
                             "choose from 'sequential' or 'detect'")
        self.assign_notes = assign_notes
        
        # universal initialisation for generator objects:
        super().__init__(params, samprate)

        # non-note names for samples (e.g. non-tonal samples)
        self.aliases = {}    
        
        if isinstance(sampfiles, dict):
            # catch case sample dictionary provided directly
            self.sampdict = sampfiles
        else:
            # re-cast sampfiles as a string
            sampfiles = str(sampfiles)
            if sampfiles[-4:] == '.sf2':
                self.sampsource = 'soundfont'
                # if a soundfont (.sf2) file, use read routines
                with open(sampfiles, 'rb') as sf2_file:
                    self.sf2 = Sf2File(sf2_file)
                    # number of presets (excluding EOS entry)
                    npres = len(self.sf2.raw.pdta['Phdr'][:-1])
                    if npres == 1:
                        # if there's only one preset, choose it
                        sf_preset = 1
                    if not (isinstance(sf_preset, int) and (sf_preset >= 1) and (sf_preset <= npres)):
                        # if there's more than one, and no valid number specified, ask for one.
                        print("valid 'sf_preset' not provided for soundfont file, available presets are:")
                        print('\n'+''.join(['-']*40))
                        choose_name = ''
                        for i in range(npres):
                            hdr = self.sf2.raw.pdta['Phdr'][i]
                            name = json.dumps(hdr.name.decode('utf-8')).replace(r'\u0000', '')
                            print(f"{i+1}. {name}")
                            if not choose_name:
                                choose_name = name
                        print(''.join(['-']*40)+'\n')
                        # TODO: zero index, but would 1 index be more user friendly?
                        print(f"By default choosing preset 1 ({choose_name}).\n\n"
                              "Re-run 'Sampler' with the 'sf_preset' keyword argument to select a specific\n"
                              f"preset, ie. 'Sampler(\"{sampfiles}\",sf_preset=N)',\n"
                              f"where N is an integer from 1-{i+1}.\n")
                        sf_preset = 1
                        
                    # TODO: isolate the warning suppression better?
                    logger = logging.getLogger()

                    pres = self.sf2.build_presets()
                    logger.disabled = True
                    sf_data = self.get_sfpreset_samples(pres[sf_preset-1])
                    self.sampdict = self.reconstruct_samples(sf_data)
                    logger.disabled = False

                    self.sf_preset = sf_preset
                    hdr = self.sf2.raw.pdta['Phdr'][sf_preset-1]
                    self.sf_preset_name = json.dumps(hdr.name.decode('utf-8')).replace(r'\u0000', '')
                self.load_samples()
            else:
                self.sampsource = 'directory'
                wavs = sorted(Path(sampfiles).glob("*.[wW][aA][vV]"))
                self.sampdict = {}
                unassigned_wavs = []
                for w in wavs:
                    filename = Path(w).name
                    note = filename.split('_')[-1].split('.')[0]
                    if notes.valid_note(note):
                        self.sampdict[note] = str(w)
                    else:
                        unassigned_wavs.append(str(w))
                if len(unassigned_wavs) == 0:
                    # All samples assigned to notes
                    self.load_samples()
                elif len(unassigned_wavs) == len(wavs):
                    # All samples unassigned
                    self.load_samples(unassigned_wavs=unassigned_wavs)
                else:
                    # A mix of assigned and unassigned samples
                    warnings.warn("Mix of sample filenames with and without specified notes. \n"
                                  "Assigning unspecified samples to remaining notes.")
                    self.load_samples(unassigned_wavs=unassigned_wavs)


    def get_sfpreset_samples(self, sfpreset):
        """Reading samples from a soundfont file along with metadata.

        Read in the audio samples from a ``.sf2`` file to populate
        available notes, mapping the MIDI key values to musical notes,
        scaling and tuning samples as appropriate.

        Args:
          sf_preset (`optional`, :obj:`int`) The number of the *Soundfont*
            preset to use. All `.sf2` files should contain at least one
            preset. When given default `None` value, will print available
            presets and select the first preset. Note presets are
            1-indexed.

        Returns:
          sfpre_dict (:obj:`dict`): dictionary of data required to load
          soundfont samples in to the `Sampler`, including raw `samples`,
          `sample_rate`, `original_pitch` of the samples, the `min_note`
          and `max_note` in midi values to use the sample, and the
          `sample_map`, assigning each sample to a note.
        """
        minmidi = np.inf
        maxmidi = -np.inf
        stdvel = 100
        sampdat = {}
        sratedat = {}
        krangedat = {}
        mapsamps = {}
        opitchdat = {}

        # iterate through preset 'bags' containing 
        # sample sets associated to each note
        for bag in sfpreset.bags:
            isvelstd = True
            inst = bag.instrument
            vr = bag.velocity_range
            if vr:
                isvelstd = (vr[0] <= stdvel) and (vr[1] >= stdvel)
            # we support a fixed velocity, choose value stdvel
            # as standard, so only want bags of samples 
            # associated with that range
            if not isvelstd:
                continue
            if inst:
                # if bag is not empty, iterate through samples
                for sbag in inst.bags:
                # for i in range(len(inst.samples)):
                    samp = sbag.sample
                    if not samp:
                        # if sbag not associated with a sample, skip
                        continue
                    # don't support stereo samples, due to spatialisation
                    # in strauss. Only read in mono or left channel samples.  
                    if samp.is_left or samp.is_mono:
                        tune = 0
                        ftun = 0
                        if sbag.tuning:
                            tune += sbag.tuning
                        if sbag.fine_tuning:
                            tune += sbag.fine_tuning/100
                        sample = np.frombuffer(samp.raw_sample_data, dtype='int16')
                        note = samp.original_pitch
                        if sbag.base_note:
                            note = sbag.base_note
                        name = samp.name
                        keys = np.array(sbag.key_range)
                        minmidi = min(minmidi, keys[0])
                        maxmidi = max(maxmidi, keys[1])
                        for i in range(keys[0], keys[1]+1):
                            if i not in mapsamps:
                                mapsamps[i] = []
                            mapsamps[i].append(name)
                        opitchdat[name] = note-tune
                        sratedat[name] = samp.sample_rate
                        sampdat[name] = sample
        self.sf_note_range = [notes.mkey_to_note(minmidi), notes.mkey_to_note(maxmidi)]
        return {'samples': sampdat, 'sample_rate': sratedat, 'original_pitch': opitchdat,
               'min_note': minmidi, 'max_note': maxmidi, 'sample_map': mapsamps}

    def reconstruct_samples(self, sfpre_dict):
        """Interpolate, combine and resample soundfont samples for each note,
           and load into the `Sampler`.

           Args:
             sfpre_dict (:obj:`dict`): dictionary of data required to load
               soundfont samples in to the `Sampler`, including raw `samples`,
               `sample_rate`, `original_pitch` of the samples, the `min_note`
               and `max_note` in midi values to use the sample, and the
               `sample_map`, assigning each sample to a note.
        
           Return:
             sampdict (:obj:`dict`): output dictionary of mapped notes, with
             values of arrays of sample values at the samplerate of the
             `Generator`.
        """
        minkey = sfpre_dict['min_note']
        maxkey = sfpre_dict['max_note']
        smap = sfpre_dict['sample_map']
        sampdict = {}
        for i in range(max(minkey,16), min(maxkey, 115)+1):
            wave_stack = []
            maxlen = 0
            for nme in smap[i]:
                # print((i-sfpre_dict['original_pitch'][nme])/12.)
                semi_shift = pow(2, (i-sfpre_dict['original_pitch'][nme])/12.)
                srate = sfpre_dict['sample_rate'][nme]
                samp = sfpre_dict['samples'][nme]
                vals = utils.resample(semi_shift*srate, self.samprate, samp)
                maxlen = max(maxlen, vals.size)
                wave_stack.append(vals)
            compwave = np.zeros(maxlen, dtype='int16')
            nwave = len(wave_stack)
            for wave in wave_stack:
                compwave[:wave.size] += wave//nwave
            nte = notes.mkey_to_note(i)
            sampdict[nte] = compwave # return notes using sharps
            if nte[1] == '#' and (nte[0] not in ['C', 'F']):
                # if a sharp, also assign flat...
                sampdict[nte.replace('#','b')] = compwave
            # outname = f'../../example_wavs/out_{nte}.wav'
            # write(outname, samprate, compwave)
        return sampdict
            
    def load_samples(self, unassigned_wavs=[], fill_notes=True):
        """Load audio samples into the sampler.

        Read audio samples in from a specified directory or via a
        dictionary of filepaths, generate interpolation functions for
        each, and assign them to a named note in scientific pitch notation
        (e.g. :obj:`'A4'`).

        Note:
          Notes are assigned based on a tag in the filename (see :obj:`Sampler`),
          not by analysing the audio itself. If a tuned sample is tagged as the
          wrong note, this will carry over to the sonification. However, this
          allows non-pitched samples to be assigned notes and triggered.
          
        """
        self.samples = {}
        self.samplens = {}
        mkeys = []
        root_notes = []
        for note in self.sampdict.keys():
            mkey = notes.note_to_mkey(note)
            sintp, slen = process_sample(self.sampdict[note], self.samprate)
            self.samples[note] = sintp
            self.samplens[note] = slen
            if isinstance(self.sampdict[note], str):
                self.aliases[note] = Path(self.sampdict[note]).stem
            else:
                # TODO: reassess auto aliasing for soundfonts
                self.aliases[note] = note
            mkeys.append(mkey)
            root_notes.append(note)
        # untagged samples are placed from C1 in the order given, unless
        # asked to find their pitch, in which case they go at the note
        # detected.
        mkey = notes.note_to_mkey('C1')
        for wav in unassigned_wavs:
            if self.assign_notes == 'detect':
                sintp, slen, mkey = process_sample(wav, self.samprate, find_pitch=True)
                mkey = int(np.clip(mkey, 0, 127))
            else:
                sintp, slen = process_sample(wav, self.samprate)
            note = notes.mkey_to_note(mkey)
            while note in self.samples:
                mkey += 1
                note = notes.mkey_to_note(mkey)
            self.samples[note] = sintp
            self.samplens[note] = slen
            self.sampdict[note] = wav
            self.aliases[note] = Path(wav).stem
            mkeys.append(mkey)
            root_notes.append(note)
        self.sampranges = {}
        self.samporder = list(np.array(root_notes)[np.argsort(mkeys)])

        
        for k in self.sampdict.keys():
            self.sampranges[k] = [[k]]
        if fill_notes:
            self.fill_midi()

    def fill_midi(self):
        keys = np.array(list(self.samples.keys()))
        assigned_mkey = np.zeros(len(keys))
        for k in range(keys.size):
            assigned_mkey[k] = notes.note_to_mkey(keys[k])
        keysort = np.argsort(assigned_mkey)
        assigned_mkey = assigned_mkey[keysort]
        keys = keys[keysort]
        idx = 0
        keylim = keys[0]
        keylims = [['C-1']]
        for i in range(128):
            offsets = i - assigned_mkey
            neardx = abs(offsets).argmin()
            off = offsets[neardx]
            sfac = pow(2, off/12.)
            this_note = notes.mkey_to_note(i)
            nearkey = keys[neardx]
            if (keys[neardx] != keylim):
                keylims[idx].append(notes.mkey_to_note(i-1))
                idx += 1
                keylims.append([])
                keylims[idx].append(this_note)
                keylim = keys[neardx]
            if this_note != nearkey:
                self.samples[this_note] = lambda x, s=sfac, k=nearkey: self.samples[k](x*s)  
                self.samplens[this_note] = sfac*self.samplens[nearkey]
            if this_note[1] == '#':
                # if a sharp, also assign flat...
                flat_note = notes.noteflats[i%12]+this_note[2:]
                if flat_note != nearkey:
                    # avoid recursion assigning reference to itself!
                    self.samples[flat_note] = self.samples[this_note]
                    self.samplens[flat_note] = self.samplens[this_note]                
        keylims[idx].append(notes.mkey_to_note(i))
        self.sampranges = dict(zip(keys, keylims))

    def set_aliases(self, aliases):
        """Name loaded samples, to call them by alias in a :obj:`Score`.

        Each sample may have one alias, so exiting aliases are overwritten. 

        Args:
          aliases (:obj:`dict`): keys are the new aliases, values the
            home note of the sample each names (see :meth:`info`),
            e.g. :obj:`{'kick': 'C1', 'snare': 'D1'}`.

        Raises:
          ValueError: if a note has no sample loaded at it, or an alias
            already names a different sample.
        """
        for alias, note in aliases.items():
            alias = str(alias)
            if note not in self.sampdict:
                raise ValueError(f"No sample is loaded at note '{note}' to alias as "
                                 f"'{alias}', choose from: {[str(n) for n in self.samporder]}")
            # a name must pick out one sample. Renaming the same sample is fine
            taken = [n for n, a in self.aliases.items() if a == alias and n != note]
            if taken:
                raise ValueError(f"Alias '{alias}' already names the sample at "
                                 f"note '{taken[0]}'")
            if notes.valid_note(alias):
                raise ValueError(f"Alias '{alias}' is a note name, so would be "
                                 "read as a note in the Score")
            self.aliases[note] = alias

    def resolve_alias(self, alias):
        """Find the root note of the sample an alias names.

        Args:
          alias (:obj:`str`): alias of a loaded sample

        Returns:
          note (:obj:`str`): the sample's root note

        Raises:
          ValueError: if no loaded sample has this alias.
        """
        for note, name in self.aliases.items():
            if name == alias:
                return note
        raise ValueError(f"Sampler has no sample aliased '{alias}', choose "
                         f"from: {sorted(self.aliases.values())}")

    def info(self, pretty=True):
        """
        Print info about the sampler set-up
        """
        # TODO: we could use a keyboard view, built with something like https://gist.github.com/wbolster/65fc4819120857b9fc8bdc0f7a976ed7
        if pretty:
            bfl, bfr = "\033[1m", "\033[0m"
        else:
            bfl, bfr = '',''
        if self.sampsource in ('directory', 'speech'):

            if self.sampsource == 'speech':
                print(f"{bfl}Phrase Assignment:{bfr}")
                print(f"Engine: \t {self.engine}")
                print(f"Voice: \t\t {self.voice if self.voice is not None else 'default'}\n")
                titles = ['Number', 'Home Pitch', 'Note Range', 'Phrase']
            else:
                print(f"{bfl}Sample Assignment:{bfr}\n")
                titles = ['Number', 'Home Pitch', 'File Name', 'Note Range', 'Alias']
            maxchars = []
            for i in range(len(titles)):
                maxchars.append(len(titles[i]))
            lines = [titles]        

            for i in range(len(self.samporder)):
                note = self.samporder[i]
                fname = ('' if self.sampsource == 'speech'
                         else Path(self.sampdict[note]).name)
                line = [f"{i+1}.",f"{note}",f"{fname}",f"{' - '.join(self.sampranges[note])}", f"\"{self.aliases[note]}\""]
                if self.sampsource == 'speech':
                    # the filename is just the cache entry, so name the phrase
                    line.pop(2)
                for j in range(len(line)):
                    maxchars[j] = max(maxchars[j], len(line[j]))
                lines.append(line)
            llen = len(line)
            np.sum(maxchars)
            for l in range(len(lines)):
                for c in range(llen):
                    lines[l][c] = lines[l][c].ljust(maxchars[c])
                lines[l] = '\t'.join(lines[l])
                print(lines[l])
                if l == 0:
                    print(len(lines[l].expandtabs())*'-')
        elif self.sampsource == 'soundfont':
            print(f"{bfl}Samples from SoundFont:{bfr}")
            print(f'SoundFont: \t {self.sampfiles}')
            print(f"Preset: \t {self.sf_preset}. {self.sf_preset_name} ({' - '.join(self.sf_note_range)})")
                    

    def forward_loopsamp(self, s, start, end):
        """Looping samples forward using indexing

        From a list of samples and start and end loop-points, return a
        new list of samples that index the audio file samples to
        create a forward-looping effect.

        Args:
          s (:obj:`array`-like): Sample indexes for the duration of a
            source's note 
          start (:obj:`int`): Sample index at which to start the loop
          end (:obj:`int`): Sample index at which to end the loop

        Returns:
          s_new (:obj:`array`-like): new sample indices to create a
          forward-looping effect
        """
        delsamp = end-start
        return np.piecewise(s, [s < start, s >= start],
                        [lambda x: x, lambda x: (x-start)%(delsamp) + start])
    def forward_back_loopsamp(self, s, start, end):
        """Looping samples forward-backward alternately using indexing

        From a list of samples and start and end loop-points, return a
        new list of samples that index the audio file samples to
        create a back and forth looping effect.

        Args:
          s (:obj:`array`-like): Sample indexes for the duration of a
            source's note 
          start (:obj:`int`): Sample index at which to start the loop
          end (:obj:`int`): Sample index at which to end the loop

        Returns:
          s_new (:obj:`array`-like): new sample indices to create a
          back and forth looping effect
        
        """
        delsamp = end-start
        return np.piecewise(s, [s < start, s >= start],
                            [lambda x: x,
                             lambda x: end - abs((x-start)%(2*(delsamp)) - (delsamp))])

    def play(self, mapping):
        """ Play the sound for a given source.

        Play a given source and return the sample values for
        combination into the overall sonification.

        Note:
          :obj:`mapping` is a linear dictionary (not nested, as for 
          :meth:`strauss.generator.modify_preset`) where group members
          are indicated using :obj:`'/'` notation
          (e.g. :obj:`{'volume_envelope/A': 0.5, ...`).

        Args:
          mapping (:obj:`dict`): keys and items are generator
            parameter names and their values. This combines all the
            preset mapped parameters, overwritten by any
            :obj:`Source`-mapped parameters (represented as values or
            interpolation functions for static and evolving
            parameters, respectively). This is a linear dictionary
            (not nested, see :meth:`strauss.generator.modify_preset`)
            where group members are indicated using :obj:`'/'`
            notation (e.g. :obj:`{'volume_envelope/A': 0.5, ...`).
        """
        # TO DO: Generator should know samplerate and audbuff
        # TO DO: split this into common and generator-specific functions to minimise code duplication
        samprate = self.samprate
        audbuff = self.audbuff

        params = copy.deepcopy(self.preset)
        utils.linear_to_nested_dict_reassign(mapping, params)
        # for p in self.preset.keys():
        #     if p not in mapping:
        #        mapping[p] = self.preset[p]

        # sample to use
        samplefunc = self.samples[params['note']]
        
        # note length
        if params['note_length'] == 'sample':
            nlength = self.samplens[params['note']]
            params['note_length'] = nlength/samprate
        else:
            nlength = (params['note_length']+params['volume_envelope']['R'])*samprate

        # generator stream (TO DO: attribute of stream?)
        sstream = stream.Stream(nlength/samprate, samprate)
        sstream.get_sampfracs()
        samples = sstream.samples.astype(float)

        pindex  = np.zeros(samples.size)
        if callable(params['pitch_shift']):
            pindex += params['pitch_shift'](sstream.sampfracs)/12.
        elif params['pitch_shift'] != 0:
            pindex += params['pitch_shift']/12.
        if params['pitch_lfo']['use']:
            pindex += self.lfo(samples, sstream.sampfracs, params, 'pitch')/12.
        if np.any(pindex):
            samples = np.cumsum(pow(2., pindex))
        
        # if callable(params['pitch_shift']):
        #     pshift = np.cumsum(params['pitch_shift'](sstream.sampfracs))
        #     samples *= pow(2., pshift/12.)
        # else:
        #     samples *= pow(2., params['pitch_shift']/12.)
        
        # sample looping if specified
        if params['looping'] != 'off':
            startsamp = params['loop_start']*samprate
            endsamp = params['loop_end']*samprate

            # find clean loop points within an audible (< 20Hz) cycle
            startsamp += np.argmin(samplefunc(np.arange(audbuff) + startsamp))
            endsamp += np.argmin(samplefunc(np.arange(audbuff) + endsamp))

            if params['looping'] == 'forwardback':
                samples = forward_back_loopsamp(samples,#sstream.samples,
                                                startsamp,
                                                endsamp)
            elif params['looping'] == 'forward':
                samples = forward_loopsamp(samples,#sstream.samples,
                                           startsamp,
                                           endsamp)
        
                
        # generate stream values
        values = samplefunc(samples)

        # get volume envelope
        env = self.envelope(sstream.samples, params)
        if params['volume_lfo']['use']:
            env *= np.clip(1.-self.lfo(sstream.samples, sstream.sampfracs,
                                       params, 'volume')*0.5, 0, 1)
        # apply volume normalisation or modulation (TO DO: envelope, pre or post filter?)
        sstream.values = values * env * utils.const_or_evo(params['volume'], sstream.sampfracs)
        
        # TO DO: filter envelope (specify as a cutoff array function? or filter twice?)

        # filter stream
        if params['filter'] == "on":
            if hasattr(params['cutoff'], "__iter__"):
                # if static cutoff, use minimum buffer count
                sstream.bufferize(sstream.length/4)
            else:
                # 30 ms buffer (hardcoded for now)
                sstream.bufferize(0.03)
            sstream.filt_sweep(getattr(filters, params['filter_type']),
                               utils.const_or_evo_func(params['cutoff']))
        return sstream    

class Speech(Sampler):
    """Speech generator class

    This generator class speaks phrases, rendering each with
    text-to-speech (see :obj:`strauss.tts_caption`) and playing the
    resultant audio via the :obj:`Sampler` machinery it inherits - so
    the same envelope, filter, LFO and spatialisation handling applies.
    Phrases are named in the :obj:`Score` by `alias`, i.e. the phrase
    itself. Has attribute :obj:`self.gtype = 'speech'`.

    Example:
      ::

        gen = Speech(['a red giant', 'a white dwarf'])
        score = Score([['a red giant', 'a white dwarf']], 10)

    Attributes:
      gtype (:obj:`str`): Generator type
      phrases (:obj:`list(str)`): the phrases loaded, in the order given
      voice: the text-to-speech voice used, or :obj:`None` for the
        current engine's default
      engine (:obj:`str`): name of the text-to-speech engine used
      cache_dir (:obj:`Path`): directory rendered phrases are cached in
      aliases (:obj:`dict`): the phrase each loaded sample is called by
        in a :obj:`Score`, alongside its root note

    Todo:
        * Support :obj:`Style` construction, which needs a way to carry
          a phrase list.
    """
    supports_aliases = True
    makes_aliases = True
    sounds_have_pitch = False

    # a spoken phrase is taken to have begun where it first rises above
    # this fraction of its peak, with `_trim_pad` seconds kept either side
    _trim_level = 0.01
    _trim_pad = 0.02

    def __init__(self, phrases=None, params=None, samprate=48000, voice=None,
                 cache_dir=None, trim=True):
        """
        Args:
          phrases (`optional`, :obj:`list(str)`): the phrases to say,
            each of which becomes a sample that can be called by in a
            :obj:`Score` or by a `call` mapping. Where none are given,
            the generator starts empty and generates whatever the
            sonification asks for (see :meth:`make_aliases`).
          params (`optional`, :obj:`dict`): any generator parameters
            that differ from the generator :obj:`preset`, where keys and
            values are parameters names and values respectively.
          samprate (`optional`, :obj:`int`): the sample rate of
            the generated audio in samples per second (Hz)
          voice (`optional`): text-to-speech voice, passed to
            :meth:`strauss.tts_caption.render_caption` as its `model`
            (see :meth:`strauss.tts_caption.getVoices`). :obj:`None`
            (default) uses the current engine's default voice. The
            ``kokoro`` voices are described at
            https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
          cache_dir (`optional`, :obj:`str`): where rendered phrases are
            cached, by default :obj:`"~/.cache/strauss/speech"` (or
            :obj:`$XDG_CACHE_HOME`, where set).
          trim (`optional`, :obj:`bool`): cut silence a text-to-speech
            engine leaves around a phrase, so that a source is *heard* at
            the time it sounds. :obj:`True` by default - `kokoro`, for
            one, pads about a quarter of a second in front of every
            phrase, which would put every event that late. 

        Raises:
          TTSIsNotSupported: if no text-to-speech engine is installed.
          ValueError: if `phrases` contains an empty phrase.

        Note:
          Phrases are placed at consecutive semitones from `C1` in the
          order given, so `pitch` maps low to high onto the phrase list
          - the same order the :obj:`Score` reads aliases in. 

        """
        if get_ttsMode() == 'None':
            raise TTSIsNotSupported(
                "the 'speech' generator needs a text-to-speech engine.\n"
                "Install one with 'pip install strauss[speech]', or see the\n"
                "strauss.tts_caption module for the engines supported.")

        # default speech preset
        self.gtype = 'speech'
        self.preset = getattr(presets, self.gtype).load_preset()
        self.preset['ranges'] = getattr(presets, self.gtype).load_ranges()
        self.sampsource = 'speech'
        self.sampfiles = None
        self.sf_preset = None
        self.sf_preset_name = None
        self.sf_note_range = []
        self.assign_notes = 'sequential'

        self.phrases = []
        self.voice = voice
        self.trim = trim
        self.engine = get_ttsMode()
        if cache_dir is None:
            cache_dir = Path(os.environ.get('XDG_CACHE_HOME',
                                            Path.home() / '.cache'),
                             'strauss', 'speech')
        self.cache_dir = Path(cache_dir)

        # universal initialisation for generator objects:
        super(Sampler, self).__init__(params, samprate)

        # phrases are the aliases a Score or 'call' mapping names samples by
        self.aliases = {}
        self.samples = {}
        self.samplens = {}
        self.sampdict = {}
        self.samporder = []
        self.sampranges = {}

        if phrases is not None:
            self.make_aliases(phrases)

    def make_aliases(self, names):
        """Say any of the phrases the generator does not yet have.

        Phrases are rendered and loaded as samples, so they can be called
        in a :obj:`Score` or by a `call` mapping. Pre-loaded phrases are
        kept, and rendered audio is cached, so calling repeatedly is
        economical.

        Args:
          names (:obj:`list` of :obj:`str`): phrases to say

        Raises:
          ValueError: if `names` is a single string rather than a list
            of them, or contains an empty phrase.
        """
        if isinstance(names, str):
            raise ValueError("phrases should be a list of strings to say, "
                             f"not a single string ('{names}')")
        names = [str(n).strip() for n in names]
        if not all(names):
            raise ValueError("An empty phrase was given, which cannot be spoken")

        # a phrase corresponds to one sample, so repeats are dropped (each
        # is only rendered once), keeping the order first given
        known = {p.lower(): p for p in self.phrases}
        added = []
        for name in names:
            if name.lower() in known:
                continue
            known[name.lower()] = name
            added.append(name)
        if not added:
            return

        self.phrases = self.phrases + added
        self._load_phrases()

    def _load_phrases(self):
        """Render and load every phrase, as a sampler loads its samples.

        Phrases are placed at consecutive semitones from `C1` in the
        order given. The notes are only labels to file the samples
        under - a phrase is never bent to pitch - so they are allowed to
        run past the top of the midi range where there are enough
        phrases to need it.
        """
        mkey0 = notes.note_to_mkey('C1')
        root_notes = [notes.mkey_to_note(mkey0+i)
                      for i in range(len(self.phrases))]

        # render each phrase up front, then load as any other samples
        self.sampdict = {}
        self.phrase_files = {}
        for note, phrase in zip(root_notes, self.phrases):
            path = self._render_phrase(phrase)
            self.phrase_files[note] = path
            self.sampdict[note] = self._phrase_audio(path)
        self.load_samples()

        # ...and name them by phrase rather than the cache filename
        self.aliases = dict(zip(root_notes, self.phrases))

    def _phrase_path(self, phrase):
        """Path a phrase's rendered audio is cached at.

        Args:
          phrase (:obj:`str`): phrase to be spoken

        Returns:
          path (:obj:`Path`): the wav file path, which may not exist yet
        """
        if isinstance(self.voice, dict):
            voicekey = json.dumps(self.voice, sort_keys=True, default=str)
        else:
            voicekey = str(self.voice)
        # the sample rate is part of the key as phrases are cached
        # already resampled to it
        key = '|'.join([self.engine, voicekey, str(self.samprate),
                        phrase.lower()])
        digest = hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]
        slug = re.sub(r'\W+', '-', voicekey).strip('-')[:24] or 'default'
        return Path(self.cache_dir, f"{self.engine}_{slug}_{digest}.wav")

    def _render_phrase(self, phrase):
        """Render a phrase to audio via text-to-speech, if not cached.

        Args:
          phrase (:obj:`str`): phrase to be spoken

        Returns:
          path (:obj:`str`): wav file of the spoken phrase, at the
          generator's sample rate
        """
        path = self._phrase_path(phrase)
        if path.exists():
            return str(path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # atomic copy to avoid partial renders from interrupted render
        tmp = path.with_name(f"{path.stem}-tmp{os.getpid()}.wav")
        try:
            render_caption(phrase, self.samprate, self.voice, str(tmp))
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return str(path)

    def _phrase_audio(self, path):
        """Read a rendered phrase, with its silence cut off.

        Text-to-speech engines pad a phrase with silence at either end.
        Left in, that silence would delay every source by the same
        amount - a quarter of a second, for `kokoro` - so that a star
        called out on the beat is heard late.

        Args:
          path (:obj:`str`): wav file of the spoken phrase

        Returns:
          audio (:obj:`ndarray`): the samples to sound, trimmed to the
          phrase itself unless :obj:`trim` is False
        """
        rate, wav = wavfile.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype='float64')
        if rate != self.samprate:
            # render_caption resamples as it writes, so this is a
            # belt-and-braces check on anything already cached
            wav = utils.resample(rate, self.samprate, wav)
        if not self.trim:
            return wav

        peak = np.abs(wav).max()
        if not peak:
            return wav
        loud = np.flatnonzero(np.abs(wav) > self._trim_level*peak)
        if loud.size == 0:
            return wav
        # keep a little either side, so that a quiet consonant at the
        # start of a phrase is not clipped off with the silence
        pad = int(self._trim_pad*self.samprate)
        return wav[max(0, loud[0]-pad):min(wav.size, loud[-1]+pad+1)]

    def clear_cache(self, all_phrases=False):
        """Remove cached audio for this generator's phrases.

        Args:
          all_phrases (`optional`, :obj:`bool`): when :obj:`True`,
            remove every phrase cached in :obj:`cache_dir`, not just
            those this generator loaded.

        Returns:
          removed (:obj:`int`): number of cached files removed
        """
        if all_phrases:
            paths = sorted(self.cache_dir.glob('*.wav'))
        else:
            paths = [self._phrase_path(p) for p in self.phrases]
        removed = 0
        for path in paths:
            if path.exists():
                path.unlink()
                removed += 1
        return removed

    def fill_midi(self):
        """Assign a phrase to every midi note, without pitch shifting.

        As :meth:`Sampler.fill_midi`, but each note plays its nearest
        phrase as rendered rather than a varispeed copy of it - a
        phrase spoken faster or slower by note would be a different
        phrase. Deliberate shifts are still available via the
        `pitch_shift` parameter.
        """
        keys = np.array(list(self.samples.keys()))
        assigned_mkey = np.array([notes.note_to_mkey(k) for k in keys])
        keysort = np.argsort(assigned_mkey)
        assigned_mkey = assigned_mkey[keysort]
        keys = keys[keysort]
        idx = 0
        keylim = keys[0]
        keylims = [['C-1']]
        # phrases can go above midi range
        topkey = max(127, int(assigned_mkey[-1]))
        for i in range(topkey+1):
            neardx = abs(i - assigned_mkey).argmin()
            this_note = notes.mkey_to_note(i)
            nearkey = keys[neardx]
            if nearkey != keylim:
                keylims[idx].append(notes.mkey_to_note(i-1))
                idx += 1
                keylims.append([])
                keylims[idx].append(this_note)
                keylim = nearkey
            if this_note != nearkey:
                # the same sample, at the same speed
                self.samples[this_note] = self.samples[nearkey]
                self.samplens[this_note] = self.samplens[nearkey]
            if this_note[1] == '#':
                # if a sharp, also assign flat...
                flat_note = notes.noteflats[i%12]+this_note[2:]
                if flat_note != nearkey:
                    # avoid recursion assigning reference to itself!
                    self.samples[flat_note] = self.samples[this_note]
                    self.samplens[flat_note] = self.samplens[this_note]
        keylims[idx].append(notes.mkey_to_note(i))
        self.sampranges = dict(zip(keys, keylims))

    def resolve_alias(self, alias):
        """Find the root note of the sample a phrase names.

        As :meth:`Sampler.resolve_alias`, but phrases are matched
        case-insensitively.

        Args:
          alias (:obj:`str`): phrase naming a loaded sample

        Returns:
          note (:obj:`str`): the sample's root note

        Raises:
          ValueError: if no loaded sample speaks this phrase.
        """
        for note, phrase in self.aliases.items():
            if phrase.lower() == str(alias).strip().lower():
                return note
        raise ValueError(f"Speech generator has no phrase '{alias}', choose "
                         f"from: {sorted(self.aliases.values())}")


class Spectralizer(Generator):
    """Spectralizer generator class

    This generator class synthesises sound from a spectrum input
    using an *inverse Fast Fourier Transform* (iFFT) algorithm.
    Defining a minimum and maximum frequency in Hz, input spectrum
    is interpolated between these points such that the output
    audio signal has the requested length. Phases are randomised
    to avoid phase correlations.

    Attributes:
      gtype (:obj:`str`): Generator type 

    Todo:
    	* Add other synthesiser types, aside from additive (e.g. FM,
    	  vector, wavetable)? 
    """

    def __init__(self, params=None, samprate=48000):
        """
        Args:
    	  params (`optional`, :obj:`dict`): any generator parameters
    	    that differ from the generator :obj:`preset`, where keys
            and values are parameters names and values respectively. 
    	  samprate (`optional`, :obj:`int`): the sample rate of
  	    the generated audio in samples per second (Hz)
        """
        # default synth preset
        self.gtype = 'spec'
        self.preset = getattr(presets, self.gtype).load_preset()
        self.preset['ranges'] = getattr(presets, self.gtype).load_ranges() 

        self.eq = utils.Equaliser()
        self.freqwarn = True

        # universal initialisation for generator objects:
        super().__init__(params, samprate)

    def spectrum_to_signal(self, spectrum, phases, new_nlen, mindx, maxdx, interp_type):
        """ Convert the input spectrum into sound signal
        
        Performs the inverse fast fourier transform to produce spectral
        sonification.
        
        Args:
          spectrum (:obj:`ndarray`): Values of the spectrum, ordered
            from high to low frequency
          phases (:obj:`ndarray`): Array of values of `[0,2*numpy.pi]`
            representing the complex number argument
          new_nlen (:obj:`int`): Number of samples needed to enclose
            the output signal.
          mindx (:obj:`int`): Index in total Fourier transform
            represnting the minimum audio frequency
          maxdx (:obj:`int`): Index in total Fourier transform
            represnting the maximum audio frequency
          interp_type (:obj:`str`): Interpolation approach, either
            `"sample"` interpolating between samples, or
            `"preserve_power"` where cumulative power is interpolated
            and then differentiated to avoid missing power.
        """        
        
        if interp_type == "sample":
            ps = np.interp(np.linspace(0,1,maxdx-mindx), np.linspace(0, 1, spectrum.size), spectrum)
        elif interp_type == "preserve_power":
            # we don't renormalise by len(ps) / spectrum.size,
            # as renormalising to peak later anyway.
            ps = np.diff(np.interp(np.linspace(0, 1, maxdx-mindx+1),
                                   np.linspace(0, 1, spectrum.size),
                                   np.cumsum(spectrum)))
           
        empt = np.zeros(new_nlen)
        empt[mindx:maxdx] = ps
        ps = empt
        PS = ps*np.cos(phases) + 1j*ps*np.sin(phases)
        return np.real(ifft(PS))[:new_nlen]
        
    def play(self, mapping):
        """ Play the sound for a given source.

        Play a given source and return the sample values for
        combination into the overall sonification.

        Note:
          :obj:`mapping` is a linear dictionary (not nested, as for 
          :meth:`strauss.generator.modify_preset`) where group members
          are indicated using :obj:`'/'` notation
          (e.g. :obj:`{'volume_envelope/A': 0.5, ...`).

        Args:
          mapping (:obj:`dict`): keys and items are generator
            parameter names and their values. This combines all the
            preset mapped parameters, overwritten by any
            :obj:`Source`-mapped parameters (represented as values or
            interpolation functions for static and evolving
            parameters, respectively). This is a linear dictionary
            (not nested, see :meth:`strauss.generator.modify_preset`)
            where group members are indicated using :obj:`'/'`
            notation (e.g. :obj:`{'volume_envelope/A': 0.5, ...`).
        """
        samprate = self.samprate
        audbuff = self.audbuff

        params = copy.deepcopy(self.preset)
        utils.linear_to_nested_dict_reassign(mapping, params)

        duration = (params['note_length']+params['volume_envelope']['R'])
        nlength = int(duration*samprate)
        # generator stream (attribute of stream?)
        sstream = stream.Stream(nlength/samprate, samprate)
        samples = sstream.samples
        sstream.get_sampfracs()

        spectrum = params['spectrum']
        interp_type = params['interpolation_type']
        
        if np.array(spectrum).ndim == 1:
            # number of discrete frequencies available in ifft between freq. limits 
            discrete_freqs = duration*(params['max_freq']-params['min_freq'])

            
            # how many spectra points fit into the available intermediate frequencies
            spectra_multiples = (discrete_freqs - 1)/(spectrum.size - 1)
            
            # the minimum factor by which to increase the stream length to accomodate spectra in whole number multiples
            if params['fit_spec_multiples']:
                buffer_factor = np.ceil(spectra_multiples)/spectra_multiples
            else:
                buffer_factor = 1
                
            # number of samples to generate including buffer
            new_nlen = int(buffer_factor * nlength)
        
            # the frequency bound indices which the spectrum will be mapped into
            mindx = int(params['min_freq'] * duration * buffer_factor)
            maxdx = int(params['max_freq'] * duration * buffer_factor)
            
            if params['equal_loudness_normalisation']:
                freqs =  np.linspace(params['min_freq'], params['max_freq'], len(spectrum))
                norm = self.eq.get_relative_loudness_norm(freqs)
                if not self.eq.factor_rms:
                    self.eq.factor_rms = []
                rms1 = np.sqrt(np.mean(spectrum**2))
                spectrum *= norm
                self.eq.factor_rms.append(np.sqrt(np.mean(spectrum**2))/rms1)
                
            # hardcode phase randomisation for now
            phases = 2*np.pi*np.random.random(new_nlen)

            # generate stream values
            sstream.values = self.spectrum_to_signal(spectrum, phases, new_nlen, mindx, maxdx, interp_type)[:nlength]
        else:

            if 'time_evo' in params:
                np.diff(params['time_evo'])

            nspec, nwlen  = np.array(spectrum).shape
                
            # buffer for each spectrum
            buffdur = duration / (nspec-1)
            sstream.bufferize(buffdur)

            # indices of IFFT input spectrum corresponding to the nearest desired frequencies
            # NOTE: we don't force the spectrum to reproduce desired frequencies to the accuracy of
            # a few samples in the evolving spectrum case.
            mindx = np.round(params['min_freq'] * buffdur).astype(int)
            maxdx = np.round(params['max_freq'] * buffdur).astype(int)

            if ((maxdx - mindx)*10 < nwlen) and self.freqwarn and params['interpolation_type'] == 'sample':

                basewarn = ("\n\n Spectrum strongly undersampled (by more than a factor 10) while using 'sample' \n"
                            "interpolation type. This could miss spectral features. You could consider: \n"
                            "\t - Changing interpolation type to 'preserve power' (i.e. generator.modify_preset({'interpolation_type':'preserve_power'})) \n")

                fdiff = params['max_freq']- params['min_freq']
                newdur = (nwlen/fdiff) *(nspec-1)
                if newdur < 300: # i.e. 5 minutes
                    # suggest increasing duration if reasonable 
                    basewarn += f"\t - Increase the duration of the sonification (e.g. to > {newdur:.0f}) \n"
                if ((2e4 - 30) * buffdur > nwlen):
                    newdiff = np.log10(nwlen / buffdur)
                    flo = pow(10,0.5*(np.log10(30) + np.log10(2e4)) - 0.5*newdiff)
                    fhi = pow(10,0.5*(np.log10(30) + np.log10(2e4)) + 0.5*newdiff)
                    basewarn += f"\t - Increase the sound frequency range eg ({np.floor(flo):.0f} to {np.ceil(fhi):.0f} Hz)\n"

                warnings.warn(basewarn
                    + f"\t - Rebin spectra more coarsely \n")
                    
                # only warn once per instance
                self.freqwarn = False
            
            # length of buffer and therefore IFFT in this case
            new_nlen = sstream.buffers._nsamp_buff
            
            # hardcode phase randomisation for now
            phases = 2*np.pi*np.random.random(new_nlen)
            
            # iterate through buffers and spectra
            nolap = nspec-1
            buffsize = sstream.buffers._nsamp_buff

            for i in range(nspec):
                # print(buffsize)
                buffs = self.spectrum_to_signal(spectrum[i], phases, new_nlen,
                                                mindx, maxdx, interp_type)
                # print(sstream.buffers._nsamp_buff, buffs.size)
                sstream.buffers.buffs_tile[i] = buffs[:sstream.buffers._nsamp_buff]
                if i == nolap:
                    continue
                # print(sstream.buffers.buffs_tile[i][0], sstream.buffers.buffs_tile[i][-1])
                sstream.buffers.buffs_olap[i][buffsize//2:] = sstream.buffers.buffs_tile[i][:buffsize//2]
                sstream.buffers.buffs_olap[i][:buffsize//2] = sstream.buffers.buffs_tile[i][buffsize//2:]

                if params['regen_phases']:
                    # regenerate randomised phases if doing so
                    phases = 2*np.pi*np.random.random(new_nlen)
                  
            sstream.consolidate_buffers()
            
        sstream.values /= abs(sstream.values).max()
        
        # get volume envelope
        env = self.envelope(sstream.samples, params)
        if params['volume_lfo']['use']:
            env *= np.clip(1.-self.lfo(sstream.samples, sstream.sampfracs,
                                       params, 'volume')*0.5, 0, 1)

        pindex  = np.zeros(samples.size)
        if callable(params['pitch_shift']):
            pindex += params['pitch_shift'](sstream.sampfracs)/12.
        elif params['pitch_shift'] != 0:
            pindex += params['pitch_shift']/12.
        if params['pitch_lfo']['use']:
            pindex += self.lfo(samples, sstream.sampfracs, params, 'pitch')/12.
        if np.any(pindex):
            sampfunc = interp1d(samples, sstream.values,
                                bounds_error=False,
                                fill_value = (0.,0.),
                                assume_sorted=True)
            newsamp = np.cumsum(pow(2., pindex))
            sstream.values = sampfunc(newsamp)

        # apply volume normalisation or modulation (TO DO: envelope, pre or post filter?)
        sstream.values *= utils.const_or_evo(params['volume'], sstream.sampfracs) * env
        
        # filter stream
        if params['filter'] == "on":
            if hasattr(params['cutoff'], "__iter__"):
                # if static cutoff, use minimum buffer count
                sstream.bufferize(sstream.length/4)
            else:
                # 30 ms buffer (hardcoded for now)
                sstream.bufferize(0.03)
            sstream.filt_sweep(getattr(filters, params['filter_type']),
                               utils.const_or_evo_func(params['cutoff']))
        return sstream    

def process_sample(sample, samprate, find_pitch=False):
    if isinstance(sample, str):
        rate_in, wavobj = wavfile.read(sample)
        # If it doesn't match the required rate, resample and re-write
        if rate_in != samprate:
            wavobj = utils.resample(rate_in, samprate, wavobj)
            # force to mono array, else convert values to float
        if wavobj.ndim > 1:
            wavdat = np.mean(wavobj.data, axis=1)
        else:
            wavdat = np.array(wavobj.data, dtype='float64')
    else:
        wavdat = sample.astype('float64')
    # remove DC term 
    dc = wavdat.mean()
    slen = wavdat.size
    wavdat -= dc
    wavdat /= abs(wavdat).max()
    samps = range(slen)
    intp = interp1d(samps, wavdat,
                    bounds_error=False,
                    fill_value = (0.,0.),
                    assume_sorted=True)
    
    if find_pitch:

        # need fourier analysis to assign notes...
        yf = rfft(wavdat)
        xf = rfftfreq(slen, 1 / samprate)

        # Don't need phase info - get power peak
        magnitudes = np.abs(yf)

        # double check for any DC term
        peak_index = np.argmax(magnitudes[1:]) + 1

        # Frequency, to closest midi key
        pf =  xf[peak_index]
        mkey = np.round((np.log2(pf/notes.tuneC0)+1)*12).astype(int)
        
        return intp, slen, mkey
    else:
        return intp, slen

    
def gen_chord(stream, chordname, rootoctv=3):
    """DEPRECATED CODE:
    generate chord over entire stream given chord name and optional
    octave of root note
    """
    frqs = notes.parse_chord(chordname, rootoctv)
    frqsamp = frqs/stream.samprate 
    for f in frqsamp:
        stream.values += detuned_saw(stream.samples, f)
    
def detuned_saw(samples, freqsamp, oscdets=[1,1.005,0.995]):
    """DEPRECATED CODE: 
    Three oscillator sawtooth wave generator with slight detuning for
    texture
    """
    saw = lambda freqsamp, samp: 1-((samples*(freqsamp)/2) % 2)
    signal = np.zeros(samples.size)
    for det in oscdets:
        freq = freqsamp*det
        signal += saw(freq, samples+freq*np.random.random())
    return signal

def legacy_env(t, dur,a,d,s,r):
    """ DEPRECATED CODE:
    older function for generating envelopes
    """
    att = lambda t: t/a
    dgrad = (1-s)/d
    dec = lambda t: (a-t)*dgrad + 1
    sus = lambda t: s
    funcs = [att, dec, sus]
    conds = [t<a,
             np.logical_and(t>a, t<(d+a)),
             t > (a+d)]
    vol = np.piecewise(np.clip(t, 0, dur), conds, funcs)
    rel = np.clip(np.exp((dur-t)/r),0, 1)
    print(a,d,s,r)
    return vol * rel
    
if __name__ == "__main__":
    # test volume envelope
    t = np.linspace(0.,11,500)
    dur = 9
    a = 1.4
    d = 2
    s = 0.7
    r = 1
    env = legacy_env(t, dur,a,d,s,r)
    plt.plot(t, env)
    plt.show()
