from functools import reduce
import operator
import numpy as np
from scipy.interpolate import interp1d
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
from io import StringIO 
import sys
from pathlib import Path

# Some utility classes (these may graduate to somewhere else eventually)

class NoSoundDevice:
    """
    drop-in replacement for sounddevice module if not working,
    so can still use other functionality.
    """
    def __init__(self, err):
        self.err = err
    def play(self, audio, rate, blocking=1):
        raise self.err

class Equaliser:
    def __init__(self):

        self.factor_rms = None
        parpath = Path(f"{Path(__file__).parent}","data","params.csv")
        # Read in parameters for ISO 226:2024 standard.
        pars = np.genfromtxt(parpath, delimiter=',', names=True)
        self.parfuncs = {}
        for c in pars.dtype.names:
            if c == 'freq':
                continue
            self.parfuncs[c] = interp1d(np.log10(pars['freq']),
                                        pars[c],
                                        fill_value='extrapolate')
            
    def get_relative_loudness_norm(self, freq, phon=70.):
        """
        Relative normalisation of sound frequencies
        To compensate for pereceptual loudness, following
        the ISO 226:2024 standard.

        Args:
          freq (:obj:`array-like`) audio frequencies in Hz
          phon (:obj:`float`) listening level for a 1 kHz
            note 

        Returns:
          rnorm (:obj:`array-like`) volume normalisation
            for spectra
        """
        lfreq = np.log10(freq)
        L_U = self.parfuncs['L_U'](lfreq)
        alpha_f = self.parfuncs['alpha_f'](lfreq)
        T_f = self.parfuncs['T_f'](lfreq)

        A = pow(4e-10, 0.3 - alpha_f)
        B = pow(10., (phon)*3e-2) - pow(10, 7.2e-2)
        C = pow(10., alpha_f * 0.1*(T_f + L_U))

        L_f = 10*np.log10(A*B + C)/alpha_f - L_U
        norm = pow(10., (L_f - phon)/20)
        rnorm = norm/norm.max()
        return rnorm

    
# a load of utility functions used by STRAUSS

def nested_dict_reassign(fromdict, todict):
    """recurse through dictionaries and sub-dictionaries"""
    for k, v in fromdict.items():
        if isinstance(v, dict):
            # recurse through nested dictionaries
            nested_dict_reassign(v, todict[k])
        else:
            # reassign todict value
            todict[k] = v

def nested_dict_fill(fromdict, todict):
    """recurse through dictionaries and sub-dictionaries"""
    for k, v in fromdict.items():
        if k not in todict:
            # assign todict value
            todict[k] = v
        elif isinstance(v, dict):
            # recurse through nested dictionaries
            nested_dict_fill(todict[k], v)
            
def nested_dict_idx_reassign(fromdict, todict, idx):
    """recurse through dictionaries and sub-dictionaries"""
    for k, v in fromdict.items():
        if isinstance(v, dict):
            # recurse through nested dictionaries
            nested_dict_idx_reassign(todict[k], v, idx)
        else:
            # reassign todict value
            todict[k] = v[idx]

def reassign_nested_item_from_keypath(dictionary, keypath, value):
    """
    dictionary: dict, dict object to reassign values of
    keypath: str, 'a/b/c' corresponds to dict['a']['b']['c'] 
             or (for Windows systems): str, 'a\b\c' corresponds to dict['a']['b']['c'] 
    value: any, value to reassign dictionary value with
    """
    p = Path(keypath)
    keylist = list(p.parts)
    get_item = lambda d, kl: reduce(operator.getitem, kl, d)
    get_item(dictionary, keylist[:-1])[keylist[-1]] = value
            
def linear_to_nested_dict_reassign(fromdict, todict):
    """iterate through a linear dictionary to reassign nested values
    using keypaths (d1['a/b/c'] -> d2['a']['b']['c'], d1['a']->d2['a'])"""
    for k, v in fromdict.items():
        reassign_nested_item_from_keypath(todict, k, v)
            
def const_or_evo_func(x):
    """if x is callable, return x, else provide a function that returns x"""
    if callable(x):
        return x
    else:
        return lambda y: y*0 + x

def const_or_evo(x,t):
    """if x is callable, return x(t), else return x"""
    if callable(x):
        return x(t)
    else:
        return x

def rescale_values(x, oldlims, newlims):
    """ rescale x values to range limits such that 0-1 is mapped to limits[0]-limits[1] """
    olo, ohi = oldlims
    nlo, nhi = newlims
    descale = np.clip((x - olo) / (ohi-olo), 0 , 1)
    return (nhi-nlo)*descale + nlo
    
def resample(rate_in, samprate, wavobj):
    """ resample audio from original samplerate to required samplerate """
    duration = wavobj.shape[0] / rate_in

    time_old  = np.linspace(0, duration, wavobj.shape[0])
    time_new  = np.linspace(0, duration,
                            int(wavobj.shape[0] * samprate / rate_in))

    interpolator = interp1d(time_old, wavobj.T)
    new_wavobj = np.round(interpolator(time_new).T).astype(wavobj.dtype)
    return(new_wavobj)

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
            
class Capturing(list):
    """ Context manager for handling stdout (see https://stackoverflow.com/a/16571630) """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
