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
from .utilities import rescale_values 
import warnings

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

        if ('theta' in mapped_quantities) and ('polar' in mapped_quantities):
            raise Exception(
                "\"theta\" and \"polar\" cannot be combined as " \
                "these represent the same quantity: \"theta\" and " \
                "\"phi\" are deprecated and will be replaced with \"polar\"" \
                " and \"azimuth\" in a future version.")

        if ('phi' in mapped_quantities) and ('azimuth' in mapped_quantities):
            raise Exception(
                "\"phi\" and \"azimuth\" cannot be combined as " \
                "these represent the same quantity: \"theta\" and " \
                "\"phi\" are deprecated and will be replaced with \"polar\"" \
                " and \"azimuth\" in a future version.")            
            
        # initialise common structures
        self.mapped_quantities = mapped_quantities
        self.raw_mapping = {}
        self.mapping = {}
        
    def apply_mapping_functions(self, map_funcs={}, map_lims={}, param_lims={}):
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

        Note:
           There is special behaviour for the `polar` and `azimuth`
           parameters, to ensure shortest angular distance when
           interpolating across the 0-2pi and 0-pi boundaries.
        
        """
        for key in self.mapped_quantities:
            rawvals = self.raw_mapping[key]

            # apply mapping functions if specified
            if key in map_funcs:
                mapvals = map_funcs[key](rawvals)
            else:
                mapvals = rawvals

            # set mapping limits if specified
            if key in map_lims:
                vallims = map_lims[key]
            else:
                vallims = (0,1)

            # set parameter limits if specified
            if key in param_lims:
                plims = param_lims[key]
            else:
                plims = param_lim_dict[key]
                
            lims = []
            # scale mapped values within limits if specified
            for l in vallims:
                if isinstance(l, str):
                    if '%' not in l:
                        warnings.warn("Specifying percentiles without appending a '%' character "
                                      "(e.g. XX%) currently works but is deprecated for more "
                                      "explicit syntax.", stacklevel=2)
                    else:
                        l = l.strip('%')
                    # string values notate percentile limits
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
    
            # limit mapped values from 0 to 1 NOTE: do we want to mix and match const and evo?
            if hasattr(mapvals[0], "__iter__"):
                self.mapping[key] = []
                for i in range(self.n_sources):
                    scaledvals = rescale_values(mapvals[i], lims, plims)
                    self.mapping[key].append(scaledvals)
            else:
                scaledvals = rescale_values(np.array(mapvals), lims, plims)
                self.mapping[key] =  list(scaledvals)
            
        # finally, iterate through sources and interpolate evo functions 
        for key in self.mapping:
            if key == "time_evo":
                continue
            if key == "spectrum":
                # if hasattr(self.mapping[key][0][0], "__iter__"):
                # ^ in case we want to catch and pre process multi-spectra
                continue
            elif hasattr(self.mapping[key][0], "__iter__"):
                # print(key, self.mapping[key][0])
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

class UnrecognisedProperty(Exception):
    "Error raised when trying to map unrecognised parameters"
    pass
