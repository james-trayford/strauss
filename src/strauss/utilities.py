""" The :obj:`utilities` submodule: useful functions for ``strauss``

This submodule is for useful utility functions used by other
``strauss`` modules. Generally these are not intended for direct
use by the user.
"""
from functools import reduce
import operator
import numpy as np
from scipy.interpolate import interp1d
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
from io import StringIO 
import sys
from pathlib import Path

class NoSoundDevice:
    """
    Drop-in replacement for sounddevice module if not working,
    so we can still use other functionality.

    Attributes:
      err (:obj:`Exception`): Error message from trying to import
        sounddevice
    """
    def __init__(self, err):
        self.err = err
    def play(self, *args, **kwargs):
        """Dummy function replacing `sounddevice.play` when unavailable.

        Args:
          *args: arguments (ignored)
          **kwargs: keyword-only arguments (ignored)
        """
        raise self.err
        
# a load of utility functions used by STRAUSS

def nested_dict_reassign(fromdict, todict):
    """
    Recurse through dictionaries and sub-dictionaries in
    `fromdict` and reassign equivalent values in `todict`

    Args
     fromdict (:obj:`dict`): Dictionary containing values
      to assign
     todict (:obj:`dict`): Dictionary containing values
      to be reassigned
    """
    for k, v in fromdict.items():
        if isinstance(v, dict):
            # recurse through nested dictionaries
            nested_dict_reassign(v, todict[k])
        else:
            # reassign todict value
            todict[k] = v

def nested_dict_fill(fromdict, todict):
    """
    Recurse through dictionaries and sub-dictionaries in
    `fromdict` and assign to any entries missing from
    `todict` 
    
    Args:
     fromdict (:obj:`dict`): Dictionary containing values
      to assign
    todict (:obj:`dict`): Dictionary containing values to
      be reassigned
    """
    for k, v in fromdict.items():
        if k not in todict:
            # assign todict value
            todict[k] = v
        elif isinstance(v, dict):
            # recurse through nested dictionaries
            nested_dict_fill(todict[k], v)
            
def nested_dict_idx_reassign(fromdict, todict, idx):
    """
    Recurse through dictionaries and sub-dictionaries of
    iterables in `fromdict` and index value idx to assign
    or replact value in todict
    
    Args:
     fromdict (:obj:`dict`): Dictionary containing values
      to assign
     todict (:obj:`dict`): Dictionary containing values
      to be reassigned
     idx (:obj:`dict`): Index value for retrieving value
      from iterables
    """
    for k, v in fromdict.items():
        if isinstance(v, dict):
            # recurse through nested dictionaries
            nested_dict_idx_reassign(todict[k], v, idx)
        else:
            # reassign todict value
            todict[k] = v[idx]

def reassign_nested_item_from_keypath(dictionary, keypath, value):
    """
    Reassign item in a nested dictionary to value using keypath syntax,
    to traverse multiple dictionaries

    Args:
     dictionary (:obj:`dict`): dict object to reassign values within
     keypath (:obj:`str`): Using filepath syntax on given OS to
       traverse dictionary, i.e 'a/b/c' ('a\\b\\c') corresponds to
       dict['a']['b']['c'] on Unix (Windows).
     value: value to reassign dictionary value with
    """
    p = Path(keypath)
    keylist = list(p.parts)
    get_item = lambda d, kl: reduce(operator.getitem, kl, d)
    get_item(dictionary, keylist[:-1])[keylist[-1]] = value
            
def linear_to_nested_dict_reassign(fromdict, todict):
    """
    Iterate through a linear dictionary to reassign nested values
    using keypaths (d1['a/b/c'] -> d2['a']['b']['c'], d1['a']->d2['a'])

    Args:
     fromdict (:obj:`dict`):
      Dictionary containing values to assign
     todict (:obj:`dict`):
      Dictionary containing values to be reassigned    
    """
    for k, v in fromdict.items():
        reassign_nested_item_from_keypath(todict, k, v)
            
def const_or_evo_func(x):
    """
    If x is callable, return x, else provide a function that just
    returns x

    Args:
      x: input value, either a numerical value or a
      function
    """
    if callable(x):
        return x
    else:
        return lambda y: y*0 + x

def const_or_evo(x,t):
    """
    If x is callable, return x(t), else return x

    Args:
      x: input value, either a numerical value or a
      function
      t (numerical): values to evaluate x function
    """
    if callable(x):
        return x(t)
    else:
        return x

def rescale_values(x, oldlims, newlims):
    """
    Rescale x values defined by limits oldlims to new limits newlims

    Args:
      x (array-like): Array of input values
      oldlims (:obj:`tuple`): tuple representing the original limits
      of `x` (low, high)
      newlims (:obj:`tuple`): tuple representing the new limits

    Returns:
      x_rs (array-like): Rescaled array
    """
    olo, ohi = oldlims
    nlo, nhi = newlims
    descale = np.clip((x - olo) / (ohi-olo), 0 , 1)
    return (nhi-nlo)*descale + nlo
    
def resample(rate_in, samprate, wavobj):
    """
    Resample audio from original samplerate to required samplerate

    Args:
      rate_in (:obj:`int`) sample rate of input wave object
      samprate (:obj:`int`) desired sample rate for output
      wavobj (:obj:`tuple`) sample rate, sample array tuple, output
      by `scipy.io.wavfile` function

    Returns:
      new_wavobj (:obj:`tuple`) as `wavobj`, with new sample rate
      and resampled sample values
    """
    duration = wavobj.shape[0] / rate_in

    time_old  = np.linspace(0, duration, wavobj.shape[0])
    time_new  = np.linspace(0, duration,
                            int(wavobj.shape[0] * samprate / rate_in))

    interpolator = interp1d(time_old, wavobj.T)
    new_wavobj = np.round(interpolator(time_new).T).astype(wavobj.dtype)
    return(new_wavobj)

@contextmanager
def suppress_stdout_stderr():
    """
    A context manager that redirects stdout and stderr to devnull
    """
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
            
class Capturing(list):
    """
    Context manager for handling stdout (see https://stackoverflow.com/a/16571630)
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
