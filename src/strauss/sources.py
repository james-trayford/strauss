import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

mappable = ['theta',
            'phi',
            'volume',
            'pitch',
            'time',
            'cutoff',
            'time_evo']

class Source:
    """ Generic source class """
    def __init__(self, mapped_quantities):
        # check these are all mappable parameters
        for q in mapped_quantities:
            if q not in mappable:
                raise UnrecognisedProperty(
                    f"Property \"{q}\" is not recognised")

        # initialise common structures
        self.mapped_quantities = mapped_quantities
        self.raw_mapping = {}
        self.mapping = {}
        self.mapping_evo = {}
        
    def apply_mapping_functions(self, map_funcs={}, map_lims={}):
        for key in self.mapped_quantities:
            rawvals = self.raw_mapping[key]

            # apply mapping functions if specified
            if key in map_funcs:
                mapvals = map_funcs[key](rawvals)
            else:
                mapvals = rawvals

            # set parameter limits if specified
            if key in map_lims:
                vallims = map_lims[key]
            else:
                vallims = (0,1)

            lims = []
            # scale mapped values within limits if specified
            for l in vallims:
                if isinstance(l, str):
                    # string values notate percentile limits
                    pc = float(l)
                    buff = 1
                    if pc > 100:
                        buff = pc/100.
                        pc = 100
                    lim = np.percentile(np.hstack([mapvals]), pc)*buff
                    lims.append(lim)
                else:
                    # numerical values notate absolute limits
                    lims.append(l)
    
            # limit mapped values from 0 to 1 NOTE: do we want to mix and match const and evo?
            if hasattr(mapvals[0], "__iter__"):
                self.mapping[key] = []
                for i in range(self.n_sources):
                    scaledvals = (mapvals[i] - lims[0]) / np.diff(lims)
                    self.mapping[key].append(np.clip(scaledvals,0,1))
            else:
                scaledvals = (mapvals - lims[0]) / np.diff(lims)
                self.mapping[key] =  np.clip(scaledvals, 0, 1)
                
        # finally, iterate through sources and interpolate evo functions 
        for key in self.mapping:
            if key == "time_evo":
                continue
            elif hasattr(self.mapping[key][0], "__iter__"):
                # print(key, self.mapping[key][0])
                for i in range(self.n_sources):
                    x = self.mapping["time_evo"][i]
                    y = self.mapping[key][i]
                    if key == "phi":
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
            
class MultiEvents(Source):
    def fromfile(self, datafile, mapdict):
        data = np.genfromtxt(datafile)
        for key in self.mapped_quantities:
            self.raw_mapping[key] = data[:,mapdict[key]] 
        self.n_sources = data.shape[0]
        
    def fromdict(self, datadict):
        for key in self.mapped_quantities:
            if key in datadict:
                self.raw_mapping[key] = datadict[key]
            else:
                Exception(f"Mapped property {key} not in datadict.")
        self.n_sources = datadict[key].shape[0]
 
class SingleObject(Source):
    def fromdict(self, datadict):
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
    "Error raised when trying to map unrecognised quantities"
    pass
