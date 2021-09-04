from . import stream
from . import notes
from . import presets
from . import utilities as utils
from . import filters
import numpy as np
import glob
import wavio
from scipy.interpolate import interp1d

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
    def __init__(self, params, samprate):
        """universal generator initialisation"""
        self.samprate = samprate

        # samples per buffer (use 30Hz as minimum)
        self.audbuff = self.samprate / 30.
        
        # modify or load preset if specified
        if params:
            self.preset = self.modify_preset(params)

    def load_preset(self, preset):
        self.preset = presets.sampler.load_preset(preset)
    def modify_preset(self, parameters):
        utils.nested_dict_reassign(parameters, self.preset)
    def envelope(self, samp):
        # TO DO: is it worth it to pre-set this in part if parameters don't change?
        nlen=self.preset['note_length']
        edict=self.preset['volume_envelope']
        
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
        sampt = np.clip(samp/self.samprate, 0, (nlen+r)*0.99999)
        
        # handy time values
        t1 = a 
        t2 = a+d
        t3 = nlen+r

        # determine segments and envelope value when note turns off
        a_seg = lambda t: 1-env_segment_curve(t, a, 1, -a_k)
        d_seg = lambda t: s+env_segment_curve(t-t1, d, 1-s, d_k)
        s_seg = lambda t: s
        o_seg = lambda t: 0.

        if nlen < t1:
            env_off = a_seg(nlen)
        elif nlen < t2:
            env_off = d_seg(nlen)
        else:
            env_off = s

        r_seg = lambda t: env_segment_curve(t-nlen, r, env_off, r_k)

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
        
class Synthesizer(Generator):
    def __init__(self, params=None, samprate=44100):

        # default synth preset 
        self.preset = presets.synth.load_preset()
        
        # universal initialisation for generator objects:
        super().__init__(params, samprate)

        # set up the oscillator banks
        self.setup_oscillators()

    def setup_oscillators(self):
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

    def modify_preset(self, parameters):
        super().modify_preset(parameters)
        self.setup_oscillators()
        
    # ||||||||||||||||||||||||||||||||||||||||||||||||||
    # OSC types 
    # ||||||||||||||||||||||||||||||||||||||||||||||||||
    def sine(self, s,f,p):
        return np.sin(2*np.pi*(s*f+p))
    def saw(self,s,f,p):
        return (2*(s*f+p) +1) % 2 - 1
    def square(self,s,f,p):
        return np.sign(saw(s,f,p))
    def tri(self,s,f,p):
        return 1 - abs((4*(s*f+p) +1) % 4 - 2)
            
    def combine_oscs(self, s, f):
        tot = 0.
        if isinstance(f, str):
            # we want a numerical frequency to generate tone
            f = notes.parse_note(f)
        for osc in self.osclist:
            tot += osc(s,f)
        return tot

    def play(self, mapping):
        # TO DO: Generator should know samplerate and audbuff
        # TO DO: split this into common and generator-specific functions to minimise code duplication
        # integrate the common tasks into generic generator functions  to avoid code duplication
        # once main features 
        samprate = self.samprate
        audbuff = self.audbuff

        utils.nested_dict_fill(self.preset, mapping)

        # for p in self.preset.keys():
        #     if p not in mapping:
        #        mapping[p] = self.preset[p]
               
        nlength = (mapping['note_length']+mapping['volume_envelope']['R'])*samprate

        # generator stream (TO DO: attribute of stream?)
        sstream = stream.Stream(nlength/samprate, samprate)
        samples = sstream.samples
        sstream.get_sampfracs()

        # generate stream values
        values = self.generate(samples, mapping['note'])

        # get volume envelope
        env = self.envelope(samples)
        
        # apply volume normalisation or modulation (TO DO: envelope, pre or post filter?)
        sstream.values = values * utils.const_or_evo(mapping['volume'], sstream.sampfracs) * env

        # filter stream
        if mapping['filter'] == "on":
            if hasattr(mapping['cutoff'], "__iter__"):
                # if static cutoff, use minimum buffer count
                sstream.bufferize(sstream.length/4)
            else:
                # 30 ms buffer (hardcoded for now)
                sstream.bufferize(0.03)
            sstream.filt_sweep(getattr(filters, mapping['filter_type']),
                               utils.const_or_evo_func(mapping['cutoff']))
        return sstream    
            
class Sampler(Generator):
    def __init__(self, sampfiles, params=None, samprate=44100):

        # default sampler preset 
        self.preset = presets.sampler.load_preset()
        
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
        self.samples = {}
        self.samplens = {}
        for note in self.sampdict.keys():
            wavobj = wavio.read(self.sampdict[note])
            # force to mono
            wavdat = wavobj.data.mean(axis=1)
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

    def forward_loopsamp(s, start, end):
        delsamp = end-start
        return np.piecewise(s, [s < start, s >= start],
                        [lambda x: x, lambda x: (x-start)%(delsamp) + start])
    def forward_back_loopsamp(s, start, end):
        delsamp = end-start
        return np.piecewise(s, [s < start, s >= start],
                            [lambda x: x,
                             lambda x: end - abs((x-start)%(2*(delsamp)) - (delsamp))])

    def play(self, mapping):
        # TO DO: Generator should know samplerate and audbuff
        # TO DO: split this into common and generator-specific functions to minimise code duplication
        samprate = self.samprate
        audbuff = self.audbuff

        for p in self.preset.keys():
            if p not in mapping:
               mapping[p] = self.preset[p]

        # sample to use
        samplefunc = self.samples[mapping['note']]
        
        # note length
        if mapping['note_length'] == 'sample':
            nlength = self.samplens[mapping['note']]
        else:
            nlength = (mapping['note_length']+mapping['volume_envelope']['R'])*samprate

        # generator stream (TO DO: attribute of stream?)
        sstream = stream.Stream(nlength/samprate, samprate)
        sstream.get_sampfracs()

        # sample looping if specified
        if mapping['looping'] == 'off':
            samples = sstream.samples
        else:
            startsamp = mapping['loop_start']*samprate
            endsamp = mapping['loop_end']*samprate

            # find clean loop points within an audible (< 20Hz) cycle
            startsamp += np.argmin(samplefunc(np.arange(audbuff) + startsamp))
            endsamp += np.argmin(samplefunc(np.arange(audbuff) + endsamp))

            if mapping['looping'] == 'forwardback':
                samples = forward_back_loopsamp(sstream.samples,
                                                startsamp,
                                                endsamp)
            elif mapping['looping'] == 'forward':
                samples = forward_loopsamp(sstream.samples,
                                           startsamp,
                                           endsamp)
        # generate stream values
        values = samplefunc(samples)

        # get volume envelope
        env = self.envelope(samples)
        
        # apply volume normalisation or modulation (TO DO: envelope, pre or post filter?)
        sstream.values = values * env * utils.const_or_evo(mapping['volume'], sstream.sampfracs)
        
        # TO DO: filter envelope (specify as a cutoff array function? or filter twice?)

        # filter stream
        if mapping['filter'] == "on":
            if hasattr(mapping['cutoff'], "__iter__"):
                # if static cutoff, use minimum buffer count
                sstream.bufferize(sstream.length/4)
            else:
                # 30 ms buffer (hardcoded for now)
                sstream.bufferize(0.03)
            sstream.filt_sweep(getattr(filters, mapping['filter_type']),
                               utils.const_or_evo_func(mapping['cutoff']))
        return sstream    

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

def env_segment_curve(t, t1, y0, k):
    """formula for segments of the generator.envelope function"""
    return y0/(1 + (1-k)*t / ((k+1)*(t1-t)))


def legacy_env(t, dur,a,d,s,r):
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
