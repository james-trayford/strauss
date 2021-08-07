import numpy as np
import pandas as pd

mappable = ['theta',
            'phi',
            'volume',
            'pitch',
            'time',
            'filt']

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
        
    def apply_mapping_functions(self, map_funcs, map_lims):
        for key in self.mapped_quantities:
            rawvals = self.raw_mapping[key]

            # apply mapping functions
            mapvals = map_funcs[key](rawvals)

            # scale mapped values within limits
            lims = []
            for l in map_lims[key]:
                if isinstance(l, str):
                    # string values notate percentile limits
                    pc = float(l)
                    buff = 1
                    if pc > 100:
                        buff = pc/100.
                        pc = 100
                    lim = np.percentile(mapvals, pc)*buff
                    lims.append(lim)
                else:
                    # numerical values notate absolute limits
                    lims.append(l)
    
            mapvals = (mapvals - lims[0]) / np.diff(lims)

            # finally, construct mapped values from 0 to 1
            self.mapping[key] = np.clip(mapvals, 0, 1)
    
            
class MultiEvents(Source):
    def fromfile(self, datafile, mapdict):
        data = np.genfromtxt(datafile)
        for key in self.mapped_quantities:
            self.raw_mapping[key] = data[:,mapdict[key]] 
        self.nevents = data.shape[0]
class UnrecognisedProperty(Exception):
    "Error raised when trying to map unrecognised quantities"
    pass
