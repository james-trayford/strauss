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
from . import assets

# relative imports of submodules
from . import channels
from . import filters
from . import generator as generator_module # Renamed to avoid conflict with 'generator' variable
from . import notes
from . import score as score_module # Renamed to avoid conflict with 'score' variable
from . import sonification as sonification_module # Renamed to avoid conflict with 'sonification' variable
from . import sources as sources_module # Renamed to avoid conflict with 'sources' variable
from . import stream
from . import presets

# Import core classes for direct use in the sonify function
from .score import Score
from .sources import Events, Objects
from .generator import Synthesizer, Sampler, Spectralizer
from .sonification import Sonification
from .utilities import nested_dict_reassign # For merging styles

import yaml
import numpy as np
import glob
from pathlib import Path


__version__ = "1.0.3"

p = Path(__file__)
thisdir = p.parent


# --- Matplotlib-like interface ---
_current_sonification = None
_style_cycle_index = 0
_style_cycle = {"theremin",
                "windy",
                "clicker"}

_kw_defaults = {
    'channels': 'stereo',
    'duration': 10,
    'is_mapped': ['pitch', 'time_evo'],
    # Style File
    'style' : None,
    'caption': None,
}

def sonify(*args, **kwargs):
    """
    Generate a sonification in a matplotlib-like interface.

    Args:
        data: The input data for sonification.
        style (dict, optional): A dictionary defining the sonification style,
                                including parameters for score, sources, and generator.
        **kwargs: Additional parameters to override or supplement the style.

    Returns:
        The generated Sonification object.
    """
    global _current_sonification
    sonpars, is_default = fill_from_kwargs(kwargs)            

    if not sonpars['style']:
        sonpars['style'] = 'default'
    style = styles.Style(**load_style(sonpars['style']))
    
    if len(args) == 0:
        raise Exception("No data to sonify!")
    elif len(args) == 1:
        args = [np.arange(len(args[0])), args[0]]        
    
    nmap = min(len(args), len(style.map))
    
    to_map = []
    in_lims = {}
    out_lims = {}
    for i in range(nmap):
        to_map.append(style.map[i].output)
        if style.map[i].input_range:
            in_lims[to_map[-1]] = style.map[i].input_range
        if style.map[i].output_range:
            out_lims[to_map[-1]] = style.map[i].output_range
    map_data = dict(zip(to_map, args[:nmap]))

    if 'pitch' not in to_map:
        to_map.append('pitch')
        nnote = len(style.notes)
        for k in map_data.keys():
            map_data[k] = [map_data[k]]*nnote
        map_data['pitch'] = list(range(nnote))

    snotes = style.notes
    if not isinstance(style.notes, str):
        snotes = [snotes]
    _score = Score(snotes, length=sonpars['duration'])
    _sources = getattr(sources, style.sources.capitalize())(to_map)
    _sources.fromdict(map_data)
    _sources.apply_mapping_functions(map_lims=in_lims, param_lims=out_lims)

    gentype = style.generator.type
    
    if gentype == 'sampler':
         asset = assets.get_asset_path(style.generator.sample.lower())
         _generator = getattr(generator, "Sampler")(asset)
    else:
        _generator = getattr(generator, style.generator.type.capitalize())()
    _generator.load_preset(style.generator.preset)
    if style.generator.mods:
        _generator.modify_preset(style.generator.mods)
    _sonification = Sonification(
        score=_score,
        sources=_sources,
        generator=_generator,
        audio_setup=sonpars['channels'],
        caption=sonpars["caption"],
        samprate=_generator.samprate, # Use generator's samprate for consistency
    )
    
    _current_sonification = _sonification
    return _sonification

def display():
    _current_sonification.render()
    return _current_sonification.notebook_display(False)
    

def fill_from_kwargs(input_kwargs):
    """ function to store provided keyword arguments against defaults
    """
    sonpars = input_kwargs.copy()
    is_default = {}
    for k in _kw_defaults.keys():
        if k not in sonpars:
            sonpars[k] = _kw_defaults[k]
            is_default[k] = True
        else:
            is_default[k] = False
    return sonpars, is_default

def load_style(name="default"):
    if Path(name).name == Path(name):
        # if open user directly
        filename = Path(name)
    else:
        # else load built-in preset of that name
        filename = Path(f"{thisdir}", "styles", f"{name}.yml")
    return read_yaml(filename)

def read_yaml(filename):
    with filename.open(mode='r') as fdata:
        try:
            yamldict = yaml.safe_load(fdata)
        except yaml.YAMLError as err:
            print(err)
    return yamldict
