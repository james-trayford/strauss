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
from .audio_figure import AudioFigure

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
from .sources import Events, Objects, set_limits
from .generator import Synthesizer, Sampler, Spectralizer
from .sonification import Sonification
from .utilities import nested_dict_reassign, merge_events, rescale_values

import yaml
import numpy as np
import glob
from pathlib import Path
import hashlib
import json

__version__ = "1.0.3"

p = Path(__file__)
thisdir = p.parent


# --- Matplotlib-like interface ---
_current_figure = None
_figure_hashes = {}
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
    'name': None,
    'level': 1,
}

_exclude_keys = ['name', 'level']

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
    global _current_figure
    sonpars, is_default = fill_from_kwargs(kwargs)            

    # We first analyse if this is a repeat sonification...
    # Initially filter relevent keys
    kwargs_to_hash = {k: v for k, v in sonpars.items() if k not in _exclude_keys}
    args_to_hash = [str(arg) for arg in args]
    
    # Put it in a dictionary, and get a unique hash
    param_dict = {
        'args': args_to_hash, 
        'kwargs': kwargs_to_hash  # Use the filtered dict here
    }
    param_str = json.dumps(param_dict, sort_keys=True)
    current_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()

    # now check against existing hashes and rename sonification
    # or re-set volume level if necessary
    if current_hash in _figure_hashes.keys():
        name = _figure_hashes[current_hash]
        mssg = 'Matching sonification exists, '
        actions = []
        if ('name' in sonpars) and sonpars['name'] and sonpars['name'] != name:
            actions.append('renaming')
            _current_figure.rename(old=name, new=sonpars['name'])
            name = sonpars['name']
        if (sonpars['level'] != _current_figure.levels[name]):
            actions.append(f"re-setting level to {sonpars['level']}")
            _current_figure.set_level(name, sonpars['level'])
        if not actions:
            actions.append('skipping')
        print(mssg + ' and '.join(actions) + '...')
        
        # replace the named hash
        _figure_hashes[current_hash] = name
        return _current_figure.sonifications[name]
            
    if not sonpars['style']:
        sonpars['style'] = 'default'
    style = styles.Style(**load_style(sonpars['style']))
    
    if len(args) == 0:
        raise Exception("No data to sonify!")
    elif len(args) == 1:
        args = [np.arange(len(args[0])), args[0]]
        tlims = (0,args[0][-1])

    if (style.sources.lower() == 'events') and style.max_notes_per_sec:
        if style.map[0].output == 'time':
            tlims = style.map[0].input_range
        tlims = set_limits(tlims, args[0], warn=False)
        time = rescale_values(args[0], tlims, (0,1))
        
        # lets now thin the data according to max events per second if using events
        args = merge_events(sonpars['duration'], style.max_notes_per_sec, time, args)
            
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
        to_map.append('pitch')
    
    # we now iterate through style fixed values
    if len(args) > nmap: 
        for i in range(nmap, len(args)):
            if style.map[i].fixed:
                if style.map[i].input_range:
                    in_lims[to_map[-1]] = style.map[i].input_range
                if style.map[i].output_range:
                    out_lims[to_map[-1]] = style.map[i].output_range
                fix_array = np.array(args[1])*0 + style.map[i].fixed
                to_map.append(style.map[i].output)
                map_data[style.map[i].output] = fix_array

    # and finally overwrite with any kwarg fixed values:
    for k in sonpars.keys():
        ksplit = k.split('fix_')
        if len(ksplit) > 1:
            prop = ksplit[1]
            if prop in to_map:
                print(f'Overwriting {prop} with fixed value...')
            fix_array = np.array(args[1])*0 + sonpars[k]
            map_data[prop] = fix_array
            in_lims[prop] = (0,1)
            to_map.append(prop)
                
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

    if not _current_figure:
        _current_figure = AudioFigure(sonpars['duration'], system=sonpars['channels'])
    name = _current_figure.add(_sonification, name=sonpars["name"], level=sonpars["level"])
    _figure_hashes[current_hash] = name
    return _sonification

def close():
    global _current_figure
    global _figure_hashes
    _current_figure = None
    _figure_hashes = {}
    
def display():
    _current_figure.render()
    return _current_figure.notebook_display(True)
    
def list_sonifications():
    _current_figure.list_sonifications()

def set_level(name, level):
    _current_figure.set_level(name, level)
    
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

def _get_style_path(name="default"):
    path = Path(name)
    if (len(path.parts) > 1) or (path.suffix != ''):
        # if open user directly
        return Path(name)
    else:
        # else load built-in preset of that name
        return Path(f"{thisdir}", "styles", f"{name}.yml")

def load_style(name="default"):
    filename = _get_style_path(name)
    return read_yaml(filename)

def get_style(name, print_style=False):
    filename = _get_style_path(name)
    with filename.open(mode='r') as fdata:
        # strip unnecessary whitespace
        yaml_string = fdata.read().rstrip().lstrip()
    if print_style:
        print(yaml_string)
    return yaml_string
        
def read_yaml(filename):
    with filename.open(mode='r') as fdata:
        try:
            yamldict = yaml.safe_load(fdata)
        except yaml.YAMLError as err:
            print(err)
    return yamldict

def save(fname):
    _current_figure.save(fname)
