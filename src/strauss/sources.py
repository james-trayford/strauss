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

Todo:
    * Store mappable, evolvable and parameter ranges in YAML files (cleaner). 
    * Specialised Event and Object child classes (eg. spectralisation).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy import signal as sig
from .utilities import rescale_values
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


spatial_angles = ('azimuth', 'polar', 'theta', 'phi')
z_angles = ('polar', 'theta')
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
      origin (:obj:`dict`): keys are `mapped_quantities`, entries say
        where each mapping came from - `'mapped'` where requested
        directly, or `'auto'` or `'fixed'` where a higher-level
        interface (e.g. `AudioFigure`) added it.

    Raises:
    	UnrecognisedProperty: if `mapped_quantities` entry not in `mappable`.
    """
    def __init__(self, mapped_quantities):
        """
        Args:
    	  mapped_quantities (:obj:`list(str)`): The subset of parameters to
    	    which data will be mapped.
        """
        
        # check these are all mappable parameters
        
        for q in mapped_quantities:
            if q not in mappable:
                raise UnrecognisedProperty(
                    f"Property \"{q}\" is not recognised")
            
        # initialise common structures
        self.mapped_quantities = mapped_quantities
        self.raw_mapping = {}
        self.mapping = {}
        self.mapped_samples = {}

        # names are generated from n_sources on demand, until set
        self._names = None

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
        """Express mapped values of a spatial angle in its input unit.

        Spatial angles are mapped to a fraction of the angular range
        they span, as this is what the sonification works in. This
        converts such values back to the unit they were given in (see
        the `angle_unit` argument of :meth:`apply_mapping_functions`),
        leaving any other parameter alone.

        Args:
          key (:obj:`str`): name of the mapped parameter
          values (:obj:`array-like` or :obj:`float`): mapped values

        Returns:
          values (:obj:`array-like` or :obj:`float`): the values in
          units of `angle_unit`, if `key` is a spatial angle
        """
        if (key not in spatial_angles) or (key in getattr(self, 'map_lims', {})):
            return values

        amax = angle_unit_maxs[getattr(self, 'angle_unit', None) or 'cycles']

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
                   
        """

        # first store the chosen mapping variables
        self.map_funcs = map_funcs
        self.map_lims = map_lims
        self.param_lims = param_lims
        self.angle_unit = angle_unit
        
        # then validate the mapping combinations
        self.validate_mapping()
        
        # set up dictionaries to store the limits
        self.lims = {}
        self.plims = {}
        
        for key in self.mapped_quantities:
            rawvals = self.raw_mapping[key]

            # apply mapping functions if specified
            if key in map_funcs:
                func = map_funcs[key]
                func_list = [func] if callable(func) else func
                mapvals = rawvals
                for f in func_list: # Allow multiple functions to be applied in order given
                    mapvals = f(mapvals)
                
            else:
                mapvals = rawvals

            # set mapping limits if specified
            if key in map_lims:
                vallims = map_lims[key]
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
                
            # set parameter limits if specified
            if key in param_lims:
                plims = param_lims[key]
            else:
                plims = param_lim_dict[key]

            # set the limits in input units
            lims = set_limits(vallims, mapvals, warn=True)

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
                Exception(f"Mapped property {key} not in datadict.")
        self.n_sources = np.array(datadict[key]).shape[0]
 
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
                Exception(f"Mapped property {key} not in datadict.")
        self.n_sources = np.array(self.raw_mapping[key]).shape[0]

def set_limits(vallims, mapvals, warn=True):
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
            lim = sub + (np.percentile(np.hstack([mapvals]), pc) - sub)*buff
            lims.append(lim)
        else:
            # numerical values notate absolute limits
            lims.append(l)
    return lims
        
class UnrecognisedProperty(Exception):
    "Error raised when trying to map unrecognised parameters"
    pass
