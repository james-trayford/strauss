"""STRAUSS (Sonification Tools and Resources for Analysis Using Sound Synthesis)

This module provides a toolkit for *sonification*, i.e. the representation of data using sound."""

# relative imports of submodules
from . import channels
from . import filters
from . import generator
from . import notes
from . import score
from . import sonification
from . import sources
from . import stream
from . import presets
from . import styles

__version__ = "1.0.3"

import numpy as np

# module-level variables ---------------
_active_sonification = None
_active_style = None
_interactive_mode = None


class CompositeSonificationClass:
    def __init__(**kwargs):
        pass

## TODO: audify?
    
def sonify(*args, style='theremin', duration=30, **kwargs):
    
    global _active_sonification
    global _active_style

    print(args)

    if args:
        if len(args) == 1:
            y = args[0]
            x = np.linspace(0,1, len(args[0]))
        if len(args) == 2:
            x,y = args

    style_pars = styles.load_style(style)
    _active_style = style_pars

    # set the sources
    for i in range(len(style_pars['parameters'])):
        
        print(style_pars['parameters'][i])
    
    sources = Objects(style_pars[''])
    sources.fromdict(data)
    sources.apply_mapping_functions()

    

    
    print(style_pars)
    pass
