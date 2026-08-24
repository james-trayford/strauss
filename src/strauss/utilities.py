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
from scipy.stats import binned_statistic as bs1d
from pathlib import Path


# Some utility classes (these may graduate to somewhere else eventually)

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

def decimals_for_range(span, steps=1000):
    """Decimal places that resolve a range into a number of steps.

    A quantity is worth reporting to about a thousandth of the range it
    covers, which is finer than any of it is heard. Spatial angles over
    360 degrees are then given to a tenth of a degree, and the same
    angles in cycles to a thousandth of a cycle.

    Args:
      span (:obj:`float`): the range the quantity covers
      steps (`optional`, :obj:`int`): steps to resolve it into

    Returns:
      decimals (:obj:`int`): decimal places to round to, never negative
    """
    if (not np.isfinite(span)) or (span <= 0):
        return 3

    return int(max(0, -np.floor(np.log10(span/steps))))

def amplitude_to_db(amplitude):
    """Express an amplitude in decibels.

    Amplitudes are fractions of the loudest signal, which is 0 dB, and
    silence is minus infinity decibels. Decibels of amplitude are used
    throughout, so a factor of 20 rather than the 10 used for power.

    Args:
      amplitude (:obj:`array-like` or :obj:`float`): amplitude, as a
        fraction of the loudest signal

    Returns:
      db (:obj:`array-like` or :obj:`float`): the amplitude in decibels
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return 20*np.log10(np.asarray(amplitude, dtype=float))

def db_to_amplitude(db):
    """Express decibels as an amplitude.

    Inverse of :func:`amplitude_to_db`, giving the amplitude as a
    fraction of the loudest signal.

    Args:
      db (:obj:`array-like` or :obj:`float`): a level in decibels

    Returns:
      amplitude (:obj:`array-like` or :obj:`float`): the corresponding
      amplitude
    """
    return pow(10., np.asarray(db, dtype=float)/20.)

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
    if np.size(x) == 1:
        olo, ohi = (0,1)
    if (ohi == olo):
        return np.ones(np.size(x))*nlo
    else:
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

def get_supported_coqui_voices():
    """Show supported Coqui-AI TTS voices
    
    """
    voices = []

    jenny = {"id": Path('tts_models', 'en', 'jenny', 'jenny'),
             "name": 'jenny',
             "languages": ['en_GB'],
             "age": None}
    
    voices.append(jenny)

    vits = {"id": Path('tts_models', '{iso_code}', 'fairseq', 'vits'),
            "name": 'fairseq-{iso_code}',
            "languages": ['many (https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html for iso codes)'],
            "age": None}

    voices.append(vits)

    vits_ita = {"id": Path('tts_models', 'it', 'mai_female', 'glow-tts'),
                "name": 'mai',
                "languages": ['it_IT'],
                "age": None}

    voices.append(vits_ita)

    thorsten = {"id": Path('tts_models', 'de', 'thorsten', 'vits'),
                "name": 'thorsten',
                "languages": ['de_DE'],
                "age": None}

    voices.append(thorsten)

    return voices

def merge_events_original(duration, max_rate, time, arglist):
    nevents = len(time)
    nbin = int(duration*max_rate)+1
    bins = np.linspace(0,1, nbin)
    newargs = []
    for i in range(len(arglist)):
        mean, _, _ = bs1d(time, arglist[i], statistic='mean', bins=bins)
        newargs.append(mean[~np.isnan(mean)])
    return newargs

def merge_events(duration, max_rate, time, arglist, mode='average',
                 time_index=None, cyclic=None, return_index=False):
    """ Thin events down to a maximum rate.

    Events are sorted by time and grouped into clusters no closer
    together than the maximum rate allows, each cluster then yielding a
    single event. Every cluster sounds at its mean time, which keeps
    events at least a gap apart and so holds the rate to `max_rate`.

    How the remaining properties are decided depends on `mode`:
    `'average'` takes the mean of the cluster, so the event represents
    all of its members but corresponds to no one input event, while
    `'central'` takes them from the member sounding closest to the mean
    time, so the event remains a real data point.

    Args:
      duration (:obj:`float`): sonification duration in seconds
      max_rate (:obj:`float`): maximum events per second
      time (:obj:`ndarray`): event times, as a fraction of the
        sonification length
      arglist (:obj:`list(ndarray)`): input data arrays, one value per
        event
      mode (`optional`, :obj:`str`): `'average'` to average each
        cluster, or `'central'` to take its most central event
      time_index (`optional`, :obj:`int`): index in `arglist` of the
        data mapped to time, which is always averaged so that events
        keep to `max_rate`
      cyclic (`optional`, :obj:`dict`): indices in `arglist` of data
        that wraps around, with entries giving the value it wraps at
        (e.g. 360, for an azimuth in degrees). These are averaged as
        directions rather than as numbers, so that a cluster straddling
        the wrap point does not average to the opposite direction.
      return_index (`optional`, :obj:`bool`): if True, also return how
        the input events fall into clusters. Values that cannot be
        averaged (e.g. names) can be carried through the thinning by
        indexing them with this.

    Returns:
      sort_args (:obj:`list(ndarray)`): merged data arrays, in time order
      clusters (:obj:`dict`, `optional`): describing the cluster behind
      each merged event, with `'central'`, `'first'` and `'last'`
      indexing the input events of its most central, earliest and latest
      members, and `'size'` giving the number of events merged
    """
    min_gap = 1./(max_rate*duration)
    sortdx = np.argsort(time)
    sort_time = time[sortdx]

    start_indices = [0]
    anchor = sort_time[0]

    for i in range(1, len(sort_time)):
        if (sort_time[i] - anchor) >= min_gap:
            start_indices.append(i)
            anchor = sort_time[i]

    start_indices = np.array(start_indices)
    end_indices = np.append(start_indices[1:], len(sort_time))
    cluster_sizes = end_indices - start_indices

    # the event of each cluster sounding closest to the cluster's mean
    # time best represents it, whether or not we take its properties
    mean_times = np.add.reduceat(sort_time, start_indices) / cluster_sizes
    repdx = np.empty(start_indices.size, dtype=int)
    for i in range(start_indices.size):
        lo, hi = start_indices[i], end_indices[i]
        repdx[i] = lo + np.argmin(abs(sort_time[lo:hi] - mean_times[i]))

    cyclic = cyclic or {}

    sort_args = []
    for i in range(len(arglist)):
        sortvals = np.array(arglist[i], dtype=float)[sortdx]

        # non-finite values are left out of the averages, so that one event
        # with missing data does not take its whole cluster with it. A
        # cluster averages to NaN only where none of its members has a value
        good = np.isfinite(sortvals)
        ngood = np.add.reduceat(good.astype(float), start_indices)
        with np.errstate(invalid='ignore', divide='ignore'):
            if (mode == 'central') and (i != time_index):
                merged_values = sortvals[repdx]
            elif i in cyclic:
                # average as a direction, so that a cluster straddling the wrap
                # point averages to the direction it lies in, rather than to the
                # opposite one (e.g. 359 and 1 degrees to 0, not to 180)
                period = cyclic[i]
                phase = 2*np.pi*np.where(good, sortvals, 0.)/period
                sines = np.add.reduceat(np.where(good, np.sin(phase), 0.), start_indices)
                cosines = np.add.reduceat(np.where(good, np.cos(phase), 0.), start_indices)
                merged_values = np.arctan2(sines, cosines) % (2*np.pi)
                merged_values *= period/(2*np.pi)
                merged_values = np.where(ngood > 0, merged_values, np.nan)
            else:
                # time is averaged whatever the mode, to hold the event rate
                merged_values = np.add.reduceat(np.where(good, sortvals, 0.),
                                                start_indices) / ngood
        sort_args.append(merged_values)

    if return_index:
        clusters = {'central': sortdx[repdx],
                    'first': sortdx[start_indices],
                    'last': sortdx[end_indices-1],
                    'size': cluster_sizes}
        return sort_args, clusters

    return sort_args
                
        
def nan_mute_envelope(mask, times, ramp):
    """ Build a gain curve muting the intervals where data is missing.

    Returns the knots of a piecewise-linear curve that sits at full
    volume where a source has data and at zero where it has none, with
    a short ramp between the two so that the transition does not click.

    The ramp down begins at the last sample before the gap, so that
    sample still sounds at full volume, and the ramp up finishes at the
    first sample after it, so that one does too. Where a gap is too
    short to fit both ramps, the curve dips to zero at its midpoint
    instead. A gap running to either end of the data has no sample on
    that side to ramp from, and so simply starts or stays muted.

    Args:
      mask (:obj:`ndarray`): boolean, True where a sample is missing
      times (:obj:`ndarray`): position of each sample on the axis the
        knots are returned in, in ascending order
      ramp (:obj:`float`): duration of the ramps, in the units of `times`

    Returns:
      knots (:obj:`tuple(ndarray)`): the `x` and `y` of the curve, or
        :obj:`None` where nothing is missing and so nothing is muted
    """
    mask = np.asarray(mask, dtype=bool)
    times = np.asarray(times, dtype=float)

    if not mask.any():
        return None

    if mask.all():
        return np.array([times[0], times[-1]]), np.zeros(2)

    # bracket each contiguous run of missing samples, as the padding makes
    # a run at either end of the data show up as a change like any other
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(np.diff(padded.astype(int)))
    starts, ends = changes[0::2], changes[1::2] - 1

    xs = [times[0]]
    ys = [0. if mask[0] else 1.]

    for first, last in zip(starts, ends):
        has_pre, has_post = first > 0, last < mask.size - 1
        pre = times[first-1] if has_pre else None
        post = times[last+1] if has_post else None

        # ramp down from the last sample with data, and up to the next one
        down = min(pre + ramp, times[first]) if has_pre else None
        up = max(post - ramp, times[last]) if has_post else None

        # a gap too short for both ramps dips to zero at its midpoint
        if has_pre and has_post and (down > up):
            down = up = 0.5*(pre + post)

        if has_pre:
            xs += [pre, down]
            ys += [1., 0.]
        if has_post:
            xs += [up, post]
            ys += [0., 1.]

    xs.append(times[-1])
    ys.append(0. if mask[-1] else 1.)

    # knots must be strictly increasing to interpolate between. Sorting the
    # quieter of any pair sharing a position to the front keeps a mute from
    # being widened away by a ramp knot landing on top of it
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    order = np.lexsort((ys, xs))
    xs, ys = xs[order], ys[order]
    unique = np.append(True, np.diff(xs) > 0)

    return xs[unique], ys[unique]


def apply_fades(samples, srate, fdur=0.03):
    """ Apply de-click fades to a sample array

    Args:
      samples (:obj:`ndarray`): array of input sample indices
      samprate (:obj:`int`): Samples per second of audio stream (Hz)
      fdur (`optional`, :obj:`float`): duration of fade in seconds

    Returns:
      out (:obj:`ndarray`): array of output samples
    """
    samps = samples.copy()
    nfade = int(srate*fdur)
    if nfade > 0:
        ramp = np.linspace(0,1,nfade)
        samps[:nfade] *= ramp
        samps[-nfade:] *= ramp[::-1]
    else:
        pass
    return samps


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

def is_notebook() -> bool:
    """
    check if we are in a notebook environment or not
    """
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'google.colab._shell':
            return True   # Google Colab
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    except ImportError:
        return False      # IPython not installed
