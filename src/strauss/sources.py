""" The :obj:`sources` submodule: representing data as sound sources.

This submodule deals with the mapping of input datasets to the parameters
controlling sound in the eventual sonification.   

Attributes:
   mappable (:obj:`list(str)`): List of strings indicating possible
	sonification parameters to which data can be mapped.
   evolvable (:obj:`list(str)`): List of strings indicating the subset of
	`mappable` parameters that can be evolved continuosly for an
	individual Source.
   param_limits (:obj:`list(tuple)`): List of tuples indicating the default
	numerical ranges bounding corresponding mappable parameter
	(e.g. 0-1 for volume).
   param_lim_dict (:obj:`dict`): Dictionary combining `mappable` (keys) and
	`param_limits` (items).
   nan_modes (:obj:`tuple(str)`): Supported values of the `handle_nans`
	keyword, controlling what becomes of non-finite input data.

Todo:
    * Store mappable, evolvable and parameter ranges in YAML files (cleaner). 
    * Specialised Event and Object child classes (eg. spectralisation).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy import signal as sig
from .utilities import rescale_values, amplitude_to_db, nan_mute_envelope
from .stream import filter_freq_lims
import warnings
import copy

mappable = ['polar',
            'azimuth',
            'theta',
            'phi',
            'volume',
            'pitch',
            'time',
            'cutoff',
            'time_evo',
            'spectrum',
            'pitch_shift',
            'pan',
            'volume_envelope/A',
            'volume_envelope/D',
            'volume_envelope/S',
            'volume_envelope/R',
            'volume_lfo/freq',
            'volume_lfo/freq_shift',
            'volume_lfo/amount',
            'pitch_lfo/freq',
            'pitch_lfo/freq_shift',
            'pitch_lfo/amount']     

evolvable = ['polar',
             'azimuth',
             'theta',
             'phi',
             'volume',
             'cutoff',
             'time_evo',
             'pitch_shift',
             'pan',
             'volume_lfo/freq_shift',
             'volume_lfo/amount',
             'pitch_lfo/freq_shift',
             'pitch_lfo/amount']

param_limits = [(0,1),#np.pi),
                (0,1),#2*np.pi),
                (0,1),#np.pi),
                (0,1),#2*np.pi),
                (0,1),
                (0,1),
                (0,1),
                (0,1),
                (0,1),
                (0,1),
                (0,24),
                (0,1),
                (1e-2, 10),
                (1e-2, 10),
                (0,1),
                (1e-2, 10),
                (1,12),
                (0,3),
                (0,1),
                (1,12),
                (0,3),
                (0,2)]     

param_lim_dict = dict(zip(mappable, param_limits))


# readable names for the mapped parameters, for tables and other output.
param_names = {'polar': 'Polar Angle',
               'azimuth': 'Azimuthal Angle',
               'theta': 'Polar Angle',
               'phi': 'Azimuthal Angle',
               'volume': 'Volume',
               'pitch': 'Pitch',
               'time': 'Time',
               'cutoff': 'Cutoff Frequency',
               'time_evo': 'Time',
               'spectrum': 'Spectrum',
               'pitch_shift': 'Pitch Shift',
               'pan': 'Stereo Pan',
               'volume_envelope/A': 'Volume Attack',
               'volume_envelope/D': 'Volume Decay',
               'volume_envelope/S': 'Volume Sustain',
               'volume_envelope/R': 'Volume Release',
               'volume_lfo/freq': 'Volume LFO Frequency',
               'volume_lfo/freq_shift': 'Volume LFO Frequency Shift',
               'volume_lfo/amount': 'Volume LFO Amount',
               'pitch_lfo/freq': 'Pitch LFO Frequency',
               'pitch_lfo/freq_shift': 'Pitch LFO Frequency Shift',
               'pitch_lfo/amount': 'Pitch LFO Amount',
               # not mapped parameters, but reported alongside them
               'note': 'Note',
               'source': 'Source',
               'note_length': 'Note Length'}


# conversions from a mapped value to the quantity it is worth reporting,
def _cutoff_to_freq(values):
    """Convert a mapped filter cutoff to the frequency it cuts at.

    The cutoff is mapped logarithmically between the frequency limits
    of the sweep applied in :meth:`strauss.stream.Stream.filt_sweep`.

    Args:
      values (:obj:`array-like` or :obj:`float`): mapped cutoff values

    Returns:
      freqs (:obj:`array-like` or :obj:`float`): cutoff frequencies in Hz
    """
    lolim, hilim = np.log10(filter_freq_lims)

    return pow(10., np.asarray(values, dtype=float)*(hilim-lolim) + lolim)


# below this, a sound is silent rather than quiet - well under the
# dynamic range of 16-bit audio
quiet_db = -100.

def _amplitude_to_db(values):
    """Convert a mapped amplitude to decibels.

    Amplitudes are mapped as a fraction of the loudest the parameter
    goes, which is 0 dB.

    Note:
      Nothing below `quiet_db` is audible, so everything quieter is
      reported as minus infinity decibels rather than as a number that
      suggests a distinction nobody can hear.

    Args:
      values (:obj:`array-like` or :obj:`float`): mapped amplitudes

    Returns:
      db (:obj:`array-like` or :obj:`float`): the amplitudes in decibels
    """
    db = amplitude_to_db(values)

    return np.where(db < quiet_db, -np.inf, db)


param_converters = {
    # panning is the fraction of the amplitude from the right speaker
    'pan': (lambda values: 100*np.asarray(values), '% right'),
    # loudness is heard logarithmically, so is reported that way
    'volume': (_amplitude_to_db, 'dB'),
    # a cutoff is a fraction of a logarithmic sweep, not a frequency
    'cutoff': (_cutoff_to_freq, 'Hz'),
    'time': (lambda values: np.asarray(values, dtype=float), 'seconds'),
    'time_evo': (lambda values: np.asarray(values, dtype=float), 'seconds'),
}


def display_name(key):
    """A readable name for a mapped parameter.

    Args:
      key (:obj:`str`): name of the parameter

    Returns:
      name (:obj:`str`): its readable name, from `param_names`, or the
      parameter name tidied up where it has no entry there
    """
    if key in param_names:
        return param_names[key]

    return key.replace('/', ' ').replace('_', ' ').title()


spatial_angles = ('azimuth', 'polar', 'theta', 'phi')
z_angles = ('polar', 'theta')
azimuthal_angles = ('azimuth', 'phi')
angle_unit_maxs = {'degrees': 360,
                   'radians': 2*np.pi,
                   'cycles': 1}

# parameter pairs that don't work together
invalid_combos = {('azimuth', 'pan') : 'angle_pan', 
                  ('polar', 'pan') :  'angle_pan',
                  ('phi', 'pan'): 'angle_pan',
                  ('theta', 'pan'): 'angle_pan',
                  ('theta', 'azimuth'): 'alias',
                  ('phi', 'polar'): 'alias'}
invalid_explanations = {'angle_pan': "'pan' and spatial angles are both controlling spatialisation",
                        'alias': "these represent different names for the same quantity"}

# how non-finite input data is handled. 'interpolate' fills the gaps and
# sounds the result, 'silent' fills them and then mutes the audio they
# would have made, so a gap in the data is heard as a gap in the sound
nan_modes = ('silent', 'interpolate')

# parameters whose missing values are set to 0 rather than applying silence
# envelope
nan_zero_filled = ('spectrum',)

def _any_nonfinite(values):
    """Test for any non-finite entry, however the values are nested.

    Args:
      values: array, scalar, or (possibly ragged) list of either

    Returns:
      any_bad (:obj:`bool`): True if any entry is non-finite
    """
    # lists are checked entry by entry, as they may be ragged and so
    # cannot be made into an array to test in one go
    if isinstance(values, (list, tuple)):
        return any(_any_nonfinite(v) for v in values)
    return bool((~np.isfinite(np.asarray(values, dtype=float))).any())

def _replace_nonfinite(values, fill):
    """Substitute non-finite values with a specified value.

    Args:
      values: array, scalar, or list of either
      fill (:obj:`float`): value to substitute

    Returns:
      filled: as `values`, with non-finite entries replaced
    """
    if isinstance(values, (list, tuple)):
        return [_replace_nonfinite(v, fill) for v in values]
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return float(values) if np.isfinite(values) else float(fill)
    # arrays keep their shape, so that a spectrum stays a value per frequency
    # per time rather than being taken apart into rows
    filled = values.copy()
    filled[(~np.isfinite(np.asarray(filled, dtype=float)))] = fill
    return filled

def _interp_fill(values, mask, x=None):
    """Fill the masked entries of an array by linear interpolation.

    Interpolates along `x`, or along the entry positions where no axis
    is given, taking the unmasked entries as the points to interpolate
    between. A masked run at either end has nothing to interpolate
    between and clamps to the nearest unmasked value.

    Args:
      values (:obj:`ndarray`): values to fill
      mask (:obj:`ndarray`): boolean, True where a value needs filling
      x (`optional`, :obj:`ndarray`): axis to interpolate along, which
        must itself be finite

    Returns:
      filled (:obj:`ndarray`): copy of `values` with masked entries
        filled, or unchanged where nothing is left to interpolate from
    """
    filled = np.array(values, dtype=float)
    good = ~mask
    if (not mask.any()) or (not good.any()):
        return filled
    if x is None:
        x = np.arange(filled.size, dtype=float)
    x = np.asarray(x, dtype=float)

    # np.interp needs the points it interpolates between in ascending
    # order, which event times in particular are not guaranteed to be
    order = np.argsort(x[good])
    filled[mask] = np.interp(x[mask], x[good][order], filled[good][order])
    return filled

class Source:
    """ Generic source class defining common methods/attributes
    
    `Source` and its child classes represent the input data, and its
    mapping to sonification parameters.

    Note:
	`Source` isn't used directly, instead use child classes
    	`Events` or `Objects`.

    Attributes:
      mapped_quantities (:obj:`list(str)`): The subset of parameters to
        which data will be mapped.
      raw_mapping (:obj:`dict`): Housing the input mapped parameters
        and data, with keys corresponding to :obj:`mapped_quantities`.
      mapping (:obj:`dict`): processed mapping :obj:`dict` rescaled
        to parameter ranges, or interpolation funtions for evolving
        parameters.
      mapped_samples (:obj:`dict`): as `mapping`, but retaining the
        mapped values of evolving parameters rather than replacing
        them with interpolation functions. Set by
        :meth:`apply_mapping_functions`.
      names (:obj:`list(str)`): name of each source, used to look up
        its table. Defaults to `source_0` to `source_N`.
      table_angle_unit (:obj:`str`): units to report spatial angles
        in, rather than as the mapped fractions the sonification works
        in. Defaults to degrees where unset.
      origin (:obj:`dict`): keys are `mapped_quantities`, entries say
        where each mapping came from - `'mapped'` where requested
        directly, or `'auto'` or `'fixed'` where a higher-level
        interface (e.g. `AudioFigure`) added it.
      handle_nans (:obj:`str`): what becomes of non-finite input data,
        either `'silent'` or `'interpolate'`. See :meth:`__init__`.
      nan_mask (:obj:`ndarray` or :obj:`list(ndarray)`): boolean, True
        where any mapped quantity was non-finite and its value has been
        interpolated. Set by :meth:`apply_mapping_functions`.

    Raises:
    	UnrecognisedProperty: if `mapped_quantities` entry not in `mappable`.
    	ValueError: if `handle_nans` is not a recognised mode.
    """
    def __init__(self, mapped_quantities, handle_nans='silent'):
        """
        Args:
    	  mapped_quantities (:obj:`list(str)`): The subset of parameters to
    	    which data will be mapped.
    	  handle_nans (`optional`, :obj:`str`): how non-finite (`NaN` or
    	    `±inf`) input data is handled. Both modes linearly interpolate
    	    the missing values in time, so that they don't propagate
    	    through the mapping; `'interpolate'` then sounds the
    	    interpolated data as it is, while `'silent'` (the default)
    	    additionally mutes the audio the missing data would have made,
    	    so that a gap in the data is heard as a gap in the sound.
        """

        # check these are all mappable parameters

        for q in mapped_quantities:
            if q not in mappable:
                raise UnrecognisedProperty(
                    f"Property \"{q}\" is not recognised")

        if handle_nans not in nan_modes:
            raise ValueError(f"handle_nans mode \"{handle_nans}\" is not "
                             f"recognised, choose from: {list(nan_modes)}")

        # initialise common structures
        self.mapped_quantities = mapped_quantities
        self.raw_mapping = {}
        self.mapping = {}
        self.mapped_samples = {}

        # non-finite value handling, with the mask filled in once the data
        # has been read in and the mapping functions applied
        self.handle_nans = handle_nans
        self.nan_mask = None

        # names are generated from n_sources on demand, until set
        self._names = None

        # units to report spatial angles in, degrees unless asked otherwise
        self.table_angle_unit = None

        # everything asked for here is user-specified by definition, higher
        # level interfaces overwrite this where they add parameters themselves
        self.origin = {q: 'mapped' for q in mapped_quantities}

    @property
    def names(self):
        """:obj:`list(str)`: name of each source.

        Names identify sources when looking up their tables. Where none
        are set, sources are named `source_0` to `source_N` in the order
        they were provided. Setting requires the data to have been read
        in already, as the number of sources follows from it.

        Raises:
          Exception: if set before any data is read in.
          ValueError: if the number of names does not match the number
            of sources, or if any name repeats.
        """
        if self._names is not None:
            return self._names
        return [f'source_{i}' for i in range(getattr(self, 'n_sources', 0))]

    @names.setter
    def names(self, names):
        if not hasattr(self, 'n_sources'):
            raise Exception("Cannot name sources before reading in data - "
                            "use 'fromdict' or 'fromfile' first.")

        names = [str(n) for n in names]

        if len(names) != self.n_sources:
            raise ValueError(f"Got {len(names)} source names for "
                             f"{self.n_sources} sources.")

        if len(set(names)) != len(names):
            repeats = sorted({n for n in names if names.count(n) > 1})
            raise ValueError("Source names must be unique, but multiple"
                             f"insatances of {repeats} found.")

        self._names = names

    def source_index(self, source):
        """Resolve a source to its index.

        Args:
          source (:obj:`str` or :obj:`int`): name of the source, as in
            :attr:`names`, or its index.

        Returns:
          index (:obj:`int`): index of the source

        Raises:
          KeyError: if no source goes by that name.
          IndexError: if the index falls outside the sources.
        """
        if isinstance(source, (int, np.integer)):
            if not -self.n_sources <= source < self.n_sources:
                raise IndexError(f"Source index {source} is outside the "
                                 f"{self.n_sources} sources.")
            return int(source) % self.n_sources

        names = self.names
        if source not in names:
            raise KeyError(f"No source named '{source}'. Choose from: {names}")

        return names.index(source)

    def in_angle_unit(self, key, values):
        """Express mapped values of a spatial angle in reporting units.

        Spatial angles are mapped to a fraction of the angular range
        they span, as this is what the sonification works in. This
        converts such values to `table_angle_unit`, leaving any other
        parameter alone. Where no unit was asked for, angles are
        reported in degrees as the most readable choice, whatever units
        they were given in.

        Note:
          Angles given `map_lims` are mapped linearly rather than
          wrapped, so are not angular quantities to convert, and are
          left alone too.

        Args:
          key (:obj:`str`): name of the mapped parameter
          values (:obj:`array-like` or :obj:`float`): mapped values

        Returns:
          values (:obj:`array-like` or :obj:`float`): the values in
          units of `table_angle_unit`, if `key` is a spatial angle
        """
        if (key not in spatial_angles) or (key in getattr(self, 'map_lims', {})):
            return values

        amax = angle_unit_maxs[getattr(self, 'table_angle_unit', None) or 'degrees']

        if key in z_angles:
            # polar angles are folded onto half a turn
            amax *= 0.5

        return np.asarray(values) * amax

    def validate_mapping(self):
        """ Validate the mapping choices, warn and/or except on issues.

        Looks through provided mapping for invalid combinations of parameters,
        as well as checking angle unit for the special case of 3D angles
        """

        params = self.mapped_quantities
        errs = []
        warn_text = ""
        
        # check invalid combinations of parameters
        bad_combos = []
        explain = []
        for pair in invalid_combos.keys():
            if (pair[0] in params) and (pair[1] in params):
                bad_combos.append(pair)
                explain.append(invalid_explanations[invalid_combos[pair]])
        if bad_combos:
            err_text = "Invald parameter combinations in mapping:\n"
            for i in range(len(bad_combos)):
                err_text += f" - '{bad_combos[i][0]}' and '{bad_combos[i][1]}' "
                err_text += f"are incompatible, as {explain[i]}. \n"
            err_text += f"Please remove any incompatible parameter combinations from the input mapping.\n\n"
            errs.append(err_text)

        # check we know what unit angles are input in
        for ang in spatial_angles:
            if ang in self.param_lims:
                err_text = f"As spatial angle {ang} is cyclic, a limited range cannot be supported in param_lims. "
                err_text += f"Instead, provide the units of input values for {ang} using the angle_unit argument via"
                err_text += f"of apply_mapping_functions, or use the 'pan' parameter to simply map stereo effects.\n\n"
                errs.append(err_text)
            if (ang in params) and not (ang in self.map_lims) and not (self.angle_unit):
                warn_text += f" - no angle unit or map_lims entry for {ang}, assuming values (0,1] for fractions of a circle (cycles)  \n"
            if (ang in self.map_lims) and (self.angle_unit):
                warn_text += f" - map_lims entry for '{ang}' ('{ang}':{self.map_lims[ang]}) provided alongside "
                warn_text += f"angle_unit={self.angle_unit}. Ignoring angle_unit for '{ang}'.\n"
                
        # Finally, warn or except about any issues after full audit of mapping
        if warn_text:
            warnings.warn("\n\nParameter mapping warning:\n"+warn_text)
        if len(errs):
            err_text = ''
            errnum = 1
            for e in errs:
                err_text += f"{errnum}.  "
                err_text += e
                errnum += 1
            raise Exception(f"Found {len(errs)} critical issues with mapping:\n\n", err_text)

    def _keep_sources(self, keep):
        """Thin the sources down to those flagged, names and all.

        Args:
          keep (:obj:`ndarray`): boolean, True for each source to keep
        """
        keep = np.asarray(keep, dtype=bool)

        for key in self.raw_mapping:
            vals = self.raw_mapping[key]
            if isinstance(vals, np.ndarray):
                self.raw_mapping[key] = vals[keep]
            else:
                self.raw_mapping[key] = [v for v, k in zip(vals, keep) if k]

        # names identify the sources, so they thin along with the data they
        # name, as they do for the events thinned by 'max_notes_per_sec'
        if self._names is not None:
            self._names = [n for n, k in zip(self._names, keep) if k]

        self.n_sources = int(keep.sum())

    def _map_values(self, key):
        """Apply the mapping function to a key, and find its input limits.

        Args:
          key (:obj:`str`): the mapped quantity to convert

        Returns:
          mapvals: the converted input data
          vallims (:obj:`tuple`): limits to descale the data by, as
            absolute values or percentile strings
        """
        rawvals = self.raw_mapping[key]

        # apply mapping functions if specified
        if key in self.map_funcs:
            func = self.map_funcs[key]
            func_list = [func] if callable(func) else func
            mapvals = rawvals
            for f in func_list: # Allow multiple functions to be applied in order given
                mapvals = f(mapvals)
        else:
            mapvals = rawvals

        # set mapping limits if specified
        if key in self.map_lims:
            vallims = self.map_lims[key]
        elif key in spatial_angles:
            # special case for spatial angles - use absolute values
            # set domain based on units unit (by default in cycles)
            amax = 1
            if self.angle_unit:
                amax = angle_unit_maxs[self.angle_unit]
            # for angles make sure conforms to units
            if key in z_angles:
                # triangle wave behaviour to map polar angle domain
                mapvals = (sig.sawtooth(2*np.pi*(np.array(mapvals)/amax),0.5) + 1)/2
            else:
                # sawtooth wave behaviour to map azimuthal angle domain
                mapvals = (np.array(mapvals)%amax)/amax
            vallims = (0, 1)
        else:
            vallims = ('0%','100%')

        return mapvals, vallims

    def _drop_nonfinite_times(self):
        """Discard input data with no finite time to place it at.

        A non-finite time says neither when a source sounds nor what to
        interpolate its other values against, so it is dropped in both
        `handle_nans` modes rather than filled.
        """
        if 'time' not in self.raw_mapping:
            return

        bad = ~np.isfinite(np.asarray(self.raw_mapping['time'], dtype=float))
        if not bad.any():
            return

        warnings.warn(f"Dropping {bad.sum()} of {self.n_sources} sources with "
                      "a non-finite 'time', as there is no point in the "
                      "sonification to place them at.", stacklevel=3)
        self._keep_sources(~bad)

    def _init_nan_mask(self):
        """Set up an empty non-finite value mask, one entry per source."""
        self.nan_mask = np.zeros(self.n_sources, dtype=bool)

    def _update_nan_mask(self, mapvals):
        """Fold one mapped quantity into the non-finite value mask.

        The mask is the union over every mapped quantity, so a source
        counts as missing if any one of the parameters it sounds with
        has no value.

        Args:
          mapvals: converted input data for one mapped quantity
        """
        self.nan_mask |= ~np.isfinite(np.asarray(mapvals, dtype=float))

    def _fill_nan(self, key, mapvals):
        """Interpolate over the non-finite values of one mapped quantity.

        Args:
          key (:obj:`str`): the mapped quantity being filled
          mapvals: its converted input data

        Returns:
          filled: `mapvals` with its non-finite entries interpolated
        """
        mask = ~np.isfinite(np.asarray(mapvals, dtype=float))
        if not mask.any():
            return mapvals

        # sources are interpolated against the time they sound at, so that
        # a gap is filled from the values either side of it in the audio
        return _interp_fill(mapvals, mask, self.raw_mapping.get('time', None))

    def _nan_keep_mask(self):
        """Which sources survive `'silent'` mode, or None to keep them all.

        Returns:
          keep (:obj:`ndarray` or :obj:`None`): boolean, True for each
            source to keep
        """
        return None

    def _warn_fully_missing(self):
        """Warn about sources with no data anywhere in their lifetime.

        Events missing anything are either skipped or interpolated from
        their neighbours, both of which are already reported, so there
        is nothing further to say here - see
        :meth:`Objects._warn_fully_missing`.
        """
        return

    def all_missing(self, index):
        """Whether a source has no data to sound at all.

        Args:
          index (:obj:`int`): index of the source

        Returns:
          missing (:obj:`bool`): True if every value is interpolated
        """
        if self.nan_mask is None:
            return False
        return bool(np.all(self.nan_mask[index]))

    def mute_envelope(self, index, ramp):
        """Gain curve muting the intervals where a source has no data.

        Events are discrete in time, with no interval within one to
        mute, so nothing is returned here - see
        :meth:`Objects.mute_envelope`.

        Args:
          index (:obj:`int`): index of the source
          ramp (:obj:`float`): duration of the ramps into and out of a
            mute, as a fraction of the source's lifetime

        Returns:
          knots (:obj:`tuple(ndarray)` or :obj:`None`): the `x` and `y`
            of the curve, or None where there is nothing to mute
        """
        return None

    def apply_mapping_functions(self, map_funcs={}, map_lims={}, param_lims={}, angle_unit=None):
        """ Taking input data and mapping to parameters.

        This function does the bulk of the work for `Source` classes,
        taking each input data variable and applying the mapping
        function (x' = x by default), descaling by the x' upper and
        lower limits and rescaling to the sonification parameter
        limits. These values are stored for non-evolving parameters,
        while for evolving properties they are converted to interpolation
        functions. 

        Args:
    	   map_funcs (:obj:`dict`, optional): dict with keys that must be
        	a subset self.mapped_quantities. Entries are then
        	function-like objects for converting input data
        	(e.g. taking log of a data set). If not provided,
        	each conversion function is assumed to be  f(x) = x.  
           map_lims (:obj:`dict`, optional): dict with keys that must be
        	a subset self.mapped_quantities. Entries are
        	tuples indicating the lower (index 0) and upper (index
        	1) limits on the converted input data
        	values. numerical values indicate absolute limits,
        	while strings are used to indicate percentiles
        	[e.g. ('10%','95%')]. converted data values are clipped
        	to these limits. If not provided, (0,1) is assumed.
           param_lims (:obj:`dict`, optional): dict with keys that
        	must be a subset self.mapped_quantities. Entries are
        	tuples indicating the lower (index 0) and upper (index
        	1) limits of the mapped sonification parameters. The
        	map_lims ranges are resaled to these ranges to give
        	the parameter values. If not provided, the default
        	param_lim_dict values are taken.
           angle_unit (:obj:`str`, optional): string naming a
                supported unit for any spatial angles used in the mapping
          	(e.g. `'azimuth'` or `'polar'`). Supported units are
        	`'degrees'`, `'radians'` or `'cycles'`. If spatial
        	angles are mapped without any unit system, will default
        	to cycles and warn the user.
        	
        
        Note:
           There is special behaviour for the `polar` and `azimuth`
           parameters, to ensure shortest angular distance when
           interpolating across the 0-2pi and 0-pi boundaries.

        Note:
           Non-finite input values are recorded in :attr:`nan_mask` and
           then interpolated over, so that they neither reach the
           generator nor set the limits any other source is descaled
           by. See the `handle_nans` argument of :meth:`__init__` for
           what is then done with the audio they would have made.

        """

        # first store the chosen mapping variables
        self.map_funcs = map_funcs
        self.map_lims = map_lims
        self.param_lims = param_lims
        self.angle_unit = angle_unit
        
        # then validate the mapping combinations
        self.validate_mapping()

        # a non-finite time says neither when a source sounds nor what to
        # interpolate against, so those go before anything reads the time axis
        self._drop_nonfinite_times()

        # set up dictionaries to store the limits
        self.lims = {}
        self.plims = {}

        # convert every parameter and find where the data is missing before
        # rescaling any of them. Doing this up front matters twice over: the
        # mapping functions make non-finite values of their own (the log of a
        # non-positive value, say), and 'silent' sources are dropped below,
        # which changes the data the limits are then taken over
        mapped = {}
        vallims = {}
        self._init_nan_mask()

        for key in self.mapped_quantities:
            mapped[key], vallims[key] = self._map_values(key)
            if key not in nan_zero_filled:
                self._update_nan_mask(mapped[key])

        # in 'silent' mode an event with missing data is not sounded at all,
        # so drop it rather than synthesise
        keep = self._nan_keep_mask()
        if keep is not None:
            for key in mapped:
                mapped[key] = np.asarray(mapped[key])[keep]
            self.nan_mask = self.nan_mask[keep]
            self._keep_sources(keep)

        if not self.n_sources:
            warnings.warn("No sources are left with data to sound - the "
                          "sonification will be silent.", stacklevel=2)
            return

        self._warn_fully_missing()

        for key in self.mapped_quantities:
            mapvals = mapped[key]

            # set parameter limits if specified
            if key in param_lims:
                plims = param_lims[key]
            else:
                plims = param_lim_dict[key]

            # set the limits in input units. Non-finite values are excluded,
            # so that one missing value cannot set the range for every source
            lims = set_limits(vallims[key], mapvals, warn=True)

            # then fill the gaps, so that they propagate no further than the
            # limits they have just been left out of
            if key in nan_zero_filled:
                # a missing spectral bin is an absence of power at that
                # frequency rather than a gap to interpolate across, so it
                # goes to the bottom of the input range - which descales to
                # the bottom of the parameter range, i.e. silence
                mapvals = _replace_nonfinite(mapvals, lims[0])
            else:
                mapvals = self._fill_nan(key, mapvals)

            # a parameter with no finite values anywhere has neither limits to
            # descale by nor anything to interpolate from. Warn and put it in
            # the middle of its range, rather than raising and taking the rest
            # of the sonification down with it
            if not np.all(np.isfinite(lims)):
                lims = (0., 1.)
            if _any_nonfinite(mapvals):
                warnings.warn(f"Mapped quantity '{key}' has no finite values "
                              "to interpolate from, so it is mapped to the "
                              f"middle of its {tuple(plims)} range. Sources "
                              "missing every value are silent in 'silent' "
                              "mode.", stacklevel=2)
                mapvals = _replace_nonfinite(mapvals, 0.5*(lims[0] + lims[1]))

            # lets store the limits from input for later conversion
            self.lims[key] = lims
            self.plims[key] = plims

            # limit mapped values from 0 to 1 NOTE: do we want to mix and match const and evo?
            if hasattr(mapvals[0], "__iter__"):
                self.mapping[key] = []
                for i in range(self.n_sources):
                    scaledvals = rescale_values(mapvals[i], lims, plims)
                    self.mapping[key].append(scaledvals)
            else:
                scaledvals = rescale_values(np.array(mapvals), lims, plims)
                self.mapping[key] =  list(scaledvals)

        # keep the mapped values themselves before evolving parameters are
        # replaced by interpolation functions below. Deep copy as the angle
        # unwrapping further down modifies the mapped arrays in place.
        self.mapped_samples = copy.deepcopy(self.mapping)

        # finally, iterate through sources and interpolate evo functions
        for key in self.mapping:
            if key == "time_evo":
                continue
            if key == "spectrum":
                # if hasattr(self.mapping[key][0][0], "__iter__"):
                # ^ in case we want to catch and pre process multi-spectra
                continue
            elif hasattr(self.mapping[key][0], "__iter__"):
                for i in range(self.n_sources):
                    if key not in evolvable:
                        raise Exception(f"Mapping error: Parameter \"{key}\" cannot be evolved.")
                    x = self.mapping["time_evo"][i]
                    y = self.mapping[key][i]
                    if key == "phi" or key == "azimuth":
                        # special case: shortest angular distance
                        # between phi points is always assumed
                        ydiff = np.diff(y)
                        discont_bdx = abs(ydiff) > 0.5
                        for j in range(discont_bdx.sum()):
                            xpre = x[:-1][discont_bdx][j]
                            ysense = np.sign(ydiff[discont_bdx][j]) 
                            y[x > xpre] -= ysense
                    self.mapping[key][i] = interp1d(x,y, bounds_error=False,
                                                    fill_value=(y[0],y[-1]))
            
class Events(Source):
    """ Represent data as time-discrete events.

    Child class of `Source`, for `Event`-type sources. Each `Event` is
    discrete in `time` with single data values mapped to each
    sonification parameter.
    
    """
    def fromfile(self, datafile, coldict):
        """Take input data from ASCII file

        Args:
          datafile (:obj:`str`): path to input data file
          coldict (:obj:`dict`): keys are self.mapped_values, with
        	entries integer indexes for their corresponding column.
        """
        data = np.genfromtxt(datafile)
        for key in self.mapped_quantities:
            self.raw_mapping[key] = data[:,coldict[key]] 
        self.n_sources = data.shape[0]
        
    def fromdict(self, datadict):
        """Take input data from dictionary
	
        Args:
          datadict (:obj:`dict`): keys are self.mapped_values, with
        	entries corresponding to the input data. Multiple
        	sources are provided as :obj:`lists`, with data for
       		each source corresponding to the values. Single
        	sources can be represented as single values.
        """
        for key in self.mapped_quantities:
            if key in datadict:
                self.raw_mapping[key] = datadict[key]
            else:
                raise KeyError(f"Mapped property {key} not in datadict.")
        self.n_sources = len(self.raw_mapping[self.mapped_quantities[0]])

    def _nan_keep_mask(self):
        """Which events survive `'silent'` mode, or None to keep them all.

        An event is a single point in time, so there is no gap within it
        to mute - it either sounds or it does not. Events missing any
        mapped value are therefore dropped outright, rather than
        synthesised and then silenced.

        Returns:
          keep (:obj:`ndarray` or :obj:`None`): boolean, True for each
            event to keep
        """
        if (self.handle_nans != 'silent') or (not self.nan_mask.any()):
            return None

        warnings.warn(f"Skipping {self.nan_mask.sum()} of {self.n_sources} "
                      "events with non-finite values, as handle_nans is "
                      "'silent'. Use handle_nans='interpolate' to sound them "
                      "at interpolated values instead.", stacklevel=3)
        return ~self.nan_mask

class Objects(Source):
    """ Represent data as time-continuous objects.
    
    Child class of `Source`. In addition to supporting single values
    for each parameter (see `Events` class), objects also support
    time evolution for `evolvable` parameters, given a `time-evo`
    mapping.

    Todo:
    	* implement :obj:`fromfile` method
    """
    def fromdict(self, datadict):
        """ Take input data from dictionary
	
        Args:
          datadict (:obj:`dict`): keys are self.mapped_values, with
        	entries corresponding to the input data. Multiple
        	sources are provided as either :obj:`lists` or 2D
        	:obj:`numpy.array` objects, with each source
        	corresponding to the entries or columns respectively.
        	Single sources can be represented as single values or
        	1D :obj:`numpy.array` (for evolving parameters). 
        """
        for key in self.mapped_quantities:
            if key in datadict:
                d = datadict[key]
                if (type(d) is not list) and (np.array(d).ndim <= 1):
                    self.raw_mapping[key] = [d]
                else:
                    self.raw_mapping[key] = d
            else:
                raise KeyError(f"Mapped property {key} not in datadict.")
        self.n_sources = len(self.raw_mapping[self.mapped_quantities[0]])

    def _sample_counts(self):
        """Number of time samples of each object, 1 where non-evolving."""
        if 'time_evo' in self.raw_mapping:
            return [np.size(t) for t in self.raw_mapping['time_evo']]
        return [1] * self.n_sources

    def _drop_nonfinite_times(self):
        """Discard samples of each object with no finite time to place them.

        A non-finite `time_evo` sample gives the values alongside it no
        position to be interpolated at, so the whole sample is dropped
        from every array of that object. An object whose times are
        entirely non-finite has no time axis at all, and is dropped.
        """
        if 'time_evo' not in self.raw_mapping:
            return
        if not _any_nonfinite(self.raw_mapping['time_evo']):
            return

        # the arrays are sliced per source, so they cannot stay as the rows
        # of a single 2D array once objects lose different samples
        self.raw_mapping = {k: list(v) for k, v in self.raw_mapping.items()}

        keep = np.ones(self.n_sources, dtype=bool)
        ndropped = 0

        for i in range(self.n_sources):
            times = np.atleast_1d(np.asarray(self.raw_mapping['time_evo'][i],
                                             dtype=float))
            bad = ~np.isfinite(np.asarray(times, dtype=float))
            if not bad.any():
                continue
            if bad.all():
                keep[i] = False
                continue

            ndropped += int(bad.sum())
            for key in self.mapped_quantities:
                vals = np.asarray(self.raw_mapping[key][i])
                # constants have no samples of their own to drop. Anything
                # else is indexed by time along its first axis, a spectrum
                # carrying a whole row of frequencies per entry
                if vals.ndim and (vals.shape[0] == times.size):
                    self.raw_mapping[key][i] = vals[~bad]

        if ndropped:
            warnings.warn(f"Dropping {ndropped} time samples with a non-finite "
                          "'time_evo', as there is no point in the "
                          "sonification to place them at.", stacklevel=3)
        if not keep.all():
            warnings.warn(f"Dropping {(~keep).sum()} of {self.n_sources} "
                          "objects whose 'time_evo' is entirely non-finite.",
                          stacklevel=3)
            self._keep_sources(keep)

    def _init_nan_mask(self):
        """Set up an empty mask per object, one entry per time sample."""
        self.nan_mask = [np.zeros(n, dtype=bool) for n in self._sample_counts()]

    def _update_nan_mask(self, mapvals):
        """Fold one mapped quantity into the non-finite value mask.

        Args:
          mapvals: converted input data for one mapped quantity
        """
        for i in range(self.n_sources):
            # a non-finite constant holds for the whole object's lifetime,
            # and broadcasts over every sample of its mask accordingly
            self.nan_mask[i] |= np.atleast_1d(~np.isfinite(np.asarray(mapvals[i], dtype=float)))

    def _fill_nan(self, key, mapvals):
        """Interpolate over the non-finite values of one mapped quantity.

        Evolving values are interpolated in time within their own
        object. A constant has no time axis to interpolate along, so it
        falls back to the median of the finite values the other objects
        give this parameter.

        Args:
          key (:obj:`str`): the mapped quantity being filled
          mapvals: its converted input data

        Returns:
          filled (:obj:`list`): `mapvals` with non-finite entries filled
        """
        if not _any_nonfinite(mapvals):
            return mapvals

        allvals = np.concatenate([np.atleast_1d(np.asarray(v, dtype=float)).ravel()
                                  for v in mapvals])
        finite = allvals[np.isfinite(allvals)]
        fallback = np.median(finite) if finite.size else np.nan

        filled = list(mapvals)
        for i in range(self.n_sources):
            vals = np.atleast_1d(np.asarray(mapvals[i], dtype=float))
            bad = ~np.isfinite(np.asarray(vals, dtype=float))
            if not bad.any():
                continue
            if vals.size == 1:
                filled[i] = fallback
            else:
                times = None
                if 'time_evo' in self.raw_mapping:
                    times = self.raw_mapping['time_evo'][i]
                filled[i] = _interp_fill(vals, bad, times)

        return filled

    def _warn_fully_missing(self):
        """Warn about objects with no data anywhere in their lifetime.

        Such an object sounds nothing at all in `'silent'` mode, and
        sounds throughout at values interpolated from elsewhere in
        `'interpolate'` mode. Either is worth saying out loud, as
        neither represents data the object itself gave.
        """
        missing = [i for i in range(self.n_sources) if np.all(self.nan_mask[i])]
        if not missing:
            return

        named = ', '.join(str(self.names[i]) for i in missing[:5])
        if len(missing) > 5:
            named += f', ... ({len(missing)} in total)'

        if self.handle_nans == 'silent':
            fate = "will be silent throughout"
        else:
            fate = ("will sound throughout at values interpolated from "
                    "elsewhere, rather than any of their own")

        warnings.warn(f"Objects with no finite data at any point in their "
                      f"lifetime ({named}) {fate}.", stacklevel=3)

    def mute_envelope(self, index, ramp):
        """Gain curve muting the intervals where an object has no data.

        Built over the object's mapped `time_evo`, the same axis its
        evolving parameters are interpolated against, so that the curve
        can be read at the sample fractions of the note in the same way
        they are.

        Args:
          index (:obj:`int`): index of the object
          ramp (:obj:`float`): duration of the ramps into and out of a
            mute, as a fraction of the object's lifetime

        Returns:
          knots (:obj:`tuple(ndarray)` or :obj:`None`): the `x` and `y`
            of the curve, or None where there is nothing to mute
        """
        if self.nan_mask is None:
            return None

        mask = np.atleast_1d(self.nan_mask[index])
        if 'time_evo' in self.mapping:
            times = np.atleast_1d(np.asarray(self.mapping['time_evo'][index],
                                             dtype=float))
        else:
            # without a time evolution an object holds one value throughout,
            # so its lifetime is either wholly muted or not muted at all
            times = np.linspace(0., 1., mask.size)

        return nan_mute_envelope(mask, times, ramp)

def set_limits(vallims, mapvals, warn=True):
    # sources need not share a length - and do not, once samples with no
    # finite time have been dropped from some of them - so flatten source by
    # source rather than stacking, which cannot handle a ragged set
    if isinstance(mapvals, (list, tuple)):
        flat = np.concatenate([np.asarray(v, dtype=float).ravel()
                               for v in mapvals])
    else:
        flat = np.asarray(mapvals, dtype=float).ravel()

    # non-finite values have no place in the range of the data. Taken
    # through np.percentile a single one of them would return NaN, setting
    # the limits every source is then descaled by
    flat = np.where(np.isfinite(flat), flat, np.nan)
    nothing_finite = not np.isfinite(flat).any()

    lims = []
    for l in vallims:
        if isinstance(l, str):
            if ('%' not in l) and warn:
                warnings.warn("Specifying percentiles without appending a '%' character "
                              "(e.g. XX%) currently works but is deprecated for more "
                              "explicit syntax.", stacklevel=2)
            else:
                # string values notate percentile limits
                l = l.strip('%')
            pc = float(l)
            buff = 1
            sub = 0
            if pc > 100:
                # if percentile over 100 we add
                buff = pc/100.
                pc = 100
                sub = lims[0]
            if nothing_finite:
                # no data to take a percentile over - the caller warns and
                # falls back to the middle of the parameter range
                lims.append(np.nan)
                continue
            lim = sub + (np.nanpercentile(flat, pc) - sub)*buff
            lims.append(lim)
        else:
            # numerical values notate absolute limits
            lims.append(l)
    return lims
        
class UnrecognisedProperty(Exception):
    "Error raised when trying to map unrecognised parameters"
    pass
