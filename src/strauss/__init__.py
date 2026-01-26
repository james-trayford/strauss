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

__version__ = "1.0.3"

# --- Matplotlib-like interface ---
_current_sonification = None
_style_cycle_index = 0
_style_cycle = {"theremin",
                "windy",
                "clicker"}

_kw_defaults = {
    'spatial_setup': 'stereo',
    'duration': 30,
    'is_mapped': ['pitch', 'time_evo'],
    # Style File
    'style' : {'harmony': ['C3', 'E3', 'G3'],
               'source_type': 'Objects',
               }
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


    args = ()

    sonpars, is_default = fill_from_kwargs(kwargs)            
        
    _score = Score('Cmaj7_3', length=sonpars['duration'])
    _sources = getattr(sources, sonpars['style']['source_type'])(is_mapped)
    _sources.fromdict()
    _generator = 1
    _sonification = 0 #Sonification(

    #     score=_score,
    #     sources=_sources,
    #     generator=_generator,
    #     audio_setup=sonpars['duration'],
    #     caption=sonification_params["caption"],
    #     samprate=_generator.samprate, # Use generator's samprate for consistency
    #     declick_time=sonification_params["declick_time"],
    #     ttsmodel=sonification_params["ttsmodel"]
    # )
    
    _current_sonification = _sonification
    return _sonification
    
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

#     if style:
#         nested_dict_reassign(style, current_style)
#     # Apply kwargs to override top-level style parameters or nested ones if keys match pattern (e.g., 'score/length')
#     for k, v in kwargs.items():
#         if '/' in k:
#             keys = k.split('/')
#             temp_dict = current_style
#             for i, key in enumerate(keys):
#                 if i == len(keys) - 1:
#                     temp_dict[key] = v
#                 else:
#                     if key not in temp_dict or not isinstance(temp_dict[key], dict):
#                         temp_dict[key] = {}
#                     temp_dict = temp_dict[key]
#         else:
#             if k in current_style:
#                 if isinstance(current_style[k], dict) and isinstance(v, dict):
#                     nested_dict_reassign(v, current_style[k])
#                 else:
#                     current_style[k] = v
#             else: # If kwargs are for data, then they need to be passed to Sources.fromdict
#                 pass # This needs to be handled when Sources are instantiated

#     # --- 1. Instantiate Score ---
#     score_params = current_style["score"]
#     _score = Score(
#         chord_sequence=score_params["chord_sequence"],
#         length=score_params["length"],
#         pitch_binning=score_params["pitch_binning"]
#     )

#     # --- 2. Instantiate Sources ---
#     sources_params = current_style["sources"]
#     source_type = sources_params["type"]
#     _sources = None
#     if source_type == "Events":
#         _sources = Events(mapped_quantities=sources_params["mapped_quantities"])
#     elif source_type == "Objects":
#         _sources = Objects(mapped_quantities=sources_params["mapped_quantities"])
#     else:
#         raise ValueError(f"Unknown source type: {source_type}. Choose 'Events' or 'Objects'.")

#     # Assuming 'data' is a dictionary for fromdict method
#     _sources.fromdict(data)

#     _sources.apply_mapping_functions(
#         map_funcs=sources_params["map_funcs"],
#         map_lims=sources_params["map_lims"],
#         param_lims=sources_params["param_lims"],
#         angle_unit=sources_params["angle_unit"]
#     )

#     # --- 3. Instantiate Generator ---
#     generator_params = current_style["generator"]
#     generator_type = generator_params["type"]
#     _generator = None
#     if generator_type == "Synthesizer":
#         _generator = Synthesizer(
#             params=generator_params["params"],
#             samprate=generator_params["samprate"]
#         )
#     elif generator_type == "Sampler":
#         # Sampler requires 'sampfiles' - this needs to be specified in style or kwargs
#         sampfiles = generator_params.get("sampfiles")
#         if not sampfiles:
#              # Check kwargs for sampfiles for sampler
#             sampfiles = kwargs.get("sampfiles")
#             if not sampfiles:
#                 raise ValueError("Sampler generator requires 'sampfiles' to be specified in the style or kwargs.")

#         _generator = Sampler(
#             sampfiles=sampfiles,
#             params=generator_params["params"],
#             samprate=generator_params["samprate"],
#             sf_preset=generator_params.get("sf_preset")
#         )
#     elif generator_type == "Spectralizer":
#         _generator = Spectralizer(
#             params=generator_params["params"],
#             samprate=generator_params["samprate"]
#         )
#     else:
#         raise ValueError(f"Unknown generator type: {generator_type}. Choose 'Synthesizer', 'Sampler', or 'Spectralizer'.")

#     # --- 4. Instantiate Sonification ---
#     sonification_params = current_style["sonification"]
#     _sonification = Sonification(
#         score=_score,
#         sources=_sources,
#         generator=_generator,
#         audio_setup=sonification_params["audio_setup"],
#         caption=sonification_params["caption"],
#         samprate=_generator.samprate, # Use generator's samprate for consistency
#         declick_time=sonification_params["declick_time"],
#         ttsmodel=sonification_params["ttsmodel"]
#     )

#     # --- 5. Render Sonification ---
#     render_params = current_style["render"]
#     _sonification.render(downsamp=render_params["downsamp"])

#     _current_sonification = _sonification
#     return _sonification

# def set_style(style_dict):
#     """
#     Set a global sonification style.

#     Args:
#         style_dict (dict): A dictionary defining the sonification style.
#     """
#     global _default_style
#     nested_dict_reassign(style_dict, _default_style)
#     print("Default sonification style updated.")
