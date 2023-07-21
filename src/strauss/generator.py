""" The :obj:`generator` submodule: creating sounds for the sonification.

This submodule handles the actual generation of sound for the
sonfication, after parametrisation by the :obj:`Sources` and musical
choices dictated by the :obj:`Score`.

Todo:
    * Consolidate more common code into the :obj:`Generator` parent
      class.
    * Support more Envelope and LFO types in the :obj:`play` methods
      (want pitch, volume and filter options for each)
"""

from . import stream
from . import notes
from . import presets
from . import utilities as utils
from . import filters
import numpy as np
import glob
import copy
import scipy
from scipy.io import wavfile
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings

# ignore wavfile read warning that complains due to WAV file metadata
warnings.filterwarnings("ignore", message="Chunk \(non-data\) not understood, skipping it\.")


# TO DO:
# - Ultimately have Synth and Sampler classes that own their own stream (stream.py) object
#   allowing ADSR volume and filter enveloping, LFO implementation etc.
# - Functions here will generally be called from a "Score" class that is provided with the
#   musical choices and uses these to generate sound, but can be interfaced with directly.

def forward_loopsamp(s, start, end):
    delsamp = end-start
    return np.piecewise(s, [s < start, s >= start],
                        [lambda x: x, lambda x: (x-start)%(delsamp) + start])
def forward_back_loopsamp(s, start, end):
    delsamp = end-start
    return np.piecewise(s, [s < start, s >= start],
                        [lambda x: x, lambda x: end - abs((x-start)%(2*(delsamp)) - (delsamp))])

class Generator:
    """Generic generator Class, defining common code for child classes

    Generators have common initialisation and methods that are
    defined by this parent class.
    
    Args:
    	params (`optional`, :obj:`dict`): any generator parameters
    	  that differ from the generator :obj:`preset`, where keys and
    	  values are parameters names and values respectively. 
    	samprate (`optional`, :obj:`int`): the sample rate of
  	  the generated audio in samples per second (Hz)
    """
    def __init__(self, params={}, samprate=48000):
        """universal generator initialisation"""
        self.samprate = samprate

        # samples per buffer (use 30Hz as minimum)
        self.audbuff = self.samprate / 30.
        
        # modify or load preset if specified
        if params:
            self.preset = self.modify_preset(params)

    def load_preset(self, preset='default'):
        """ load parameters from a preset YAML file.

        Wrapper method for the :obj:`presets.synth.load_preset` or
        :obj:`presets.sampler.load_preset` functions. Always load the
        :obj:`default` preset first to ensure all parameters are
        defined, and then if necessary reload parameters defined by
        :obj:`preset`

        Args:
          preset (:obj:`str`): name of the preset. built-in presets
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
        """modify parameters within current preset

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
        """ Print the names and descriptions of presets

        Wrapper for preset_details function. lists the name and description
        of built-in presets with names matching the search term.

        Args:
          term (optional, :obj:`str`): name or glob term for built-in
          preset(s). Default '*' prints all.
        """
        getattr(presets, self.gtype).preset_details(name=term)

    def envelope(self, samp, params, etype='volume'):
        """ Envelope function for modulating a single note

        The envelope function takes the pre-defined envelope
        parameters for the specified envelope type and returns the
        envelope value at each sample. envelopes are defined by
        attack, decay, sustain and release (:obj:`'A','D','S' & 'R`)
        values, as well as segment curvatures (:obj:`'Ac','Dc', &
        'Rc`) and a normalisation :obj:`'level'`.

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
        """formula for segments of the envelope function
        
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
            of cycles, :obj:`'random'` indicates randomised.
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
          To modulate the frequency of an ocillator, use the
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

    Args:
    	params (`optional`, :obj:`dict`): any generator parameters
    	  that differ from the generator :obj:`preset`, where keys and
    	  values are parameters names and values respectively. 
    	samprate (`optional`, :obj:`int`): the sample rate of
  	  the generated audio in samples per second (Hz)

    Todo:
    	* Add other synthesiser types, aside from additive (e.g. FM,
    	  vector, wavetable)? 
    """
    def __init__(self, params=None, samprate=48000):

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

        Reads the parametrisation of each oscillator from the preset,
        specifying their waveform (:obj:`wave`), relative amplitude
        (:obj:`level`), detuning in cents (:obj:`det`) and
        :obj:`phase`, either a number in units of cycles, or a string
        specifying randomisation (:obj:`'random'`). Sets the
        :obj:`self.generate` method, using the
        :obj:`self.combine_oscs`.
        """
        oscdict = self.preset['oscillators']
        self.osclist = []
        for osc in oscdict.keys():
            lvl = oscdict[osc]['level']
            det = oscdict[osc]['detune']
            phase = oscdict[osc]['phase']
            form = oscdict[osc]['form']
            snorm = self.samprate
            fnorm = (1 + det/100.)
            if phase == 'random':
                oscf = lambda samp, f: lvl * getattr(self,form)(samp/snorm, f*fnorm, np.random.random())
            else:
                oscf = lambda samp, f: lvl * getattr(self,form)(samp/snorm, f*fnorm, phase)
            self.osclist.append(oscf)
        self.generate = self.combine_oscs

    def modify_preset(self, parameters, clear_oscs=True):
        """Synthesizer-specific wrapper for the modify_preset method.

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
            cycles per second, if string, note name in scientific
            notation (e.g. :obj:`'A4'`)
        Returns:
          tot (:obj:`array`-like): values for each sample
        """
        tot = 0.
        if isinstance(f, str):
            # we want a numerical frequency to generate tone
            f = notes.parse_note(f)
        for osc in self.osclist:
            tot += osc(s,f)
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
    samples, representing different notes. The relative
    frequency, phase and amplitude of these oscillators are defined in
    the preset, and linearly combined to produce the sound. defines
    attribute :obj:`self.gtype = 'synth'`.

    Args:
        sampfiles ()
    	params (`optional`, :obj:`dict`): any generator parameters
    	  that differ from the generator :obj:`preset`, where keys and
    	  values are parameters names and values respectively. 
    	samprate (`optional`, :obj:`int`): the sample rate of
  	  the generated audio in samples per second (Hz)

    Todo:
    	* Add zone mapping for samples (e.g. allow a sample to define
          a range of notes played at different speeds).
        * Support non-scientifically named notes? (e.g
    	  :obj:`'cymbal'`, :obj:`'snare'`). 
        * Have sample loading defined via the preset rather than the
          :obj:`sampfiles` variable?
    """

    def __init__(self, sampfiles, params=None, samprate=48000):
        # default sampler preset
        self.gtype = 'sampler'
        self.preset = getattr(presets, self.gtype).load_preset()
        self.preset['ranges'] = getattr(presets, self.gtype).load_ranges() 
        
        # universal initialisation for generator objects:
        super().__init__(params, samprate)
        
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
        """Load audio samples into the sampler.

        Read audio samples in from a specified directory or via a
        dictionary of filepaths, generate interpolation functions for
        each, and assign them to a named note in scientific notation
        (e.g. :obj:`'A4'`).
        """
        self.samples = {}
        self.samplens = {}
        for note in self.sampdict.keys():
            rate_in, wavobj = wavfile.read(self.sampdict[note])
            # If it doesn't match the required rate, resample and re-write
            if rate_in != self.samprate:
                wavobj = utils.resample(rate_in, self.samprate, wavobj)
            # force to mono
            wavdat = np.mean(wavobj.data, axis=1)
            # remove DC term 
            dc = wavdat.mean()
            wavdat -= dc
            wavdat /= abs(wavdat).max()
            samps = range(wavdat.size)
            self.samples[note] = interp1d(samps, wavdat,
                                          bounds_error=False,
                                          fill_value = (0.,0.),
                                          assume_sorted=True)
            self.samplens[note] = wavdat.size

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
