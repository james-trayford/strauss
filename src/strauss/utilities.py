from functools import reduce
import operator
import numpy as np
from scipy.interpolate import interp1d

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
    value: any, value to reassign dictionary value with
    """
    keylist = keypath.split('/')
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

