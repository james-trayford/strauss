# relative imports of submodules
from . import generator as generator_module # Renamed to avoid conflict with 'generator' variable
from . import score as score_module # Renamed to avoid conflict with 'score' variable
from . import sonification as sonification_module # Renamed to avoid conflict with 'sonification' variable
from . import sources as sources_module # Renamed to avoid conflict with 'sources' variable

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

# Import core classes for direct use in the sonify function
from .score import Score
from .sources import Events, Objects, set_limits
from .generator import Synthesizer, Sampler, Spectralizer
from .sonification import Sonification
from .utilities import (nested_dict_reassign, merge_events, rescale_values,
                        db_to_amplitude)

import numpy as np
from . import channels
from .stream import Stream
from .channels import audio_channels
from .utilities import const_or_evo, nested_dict_idx_reassign, apply_fades, rescale_values, NoSoundDevice, is_notebook
from .tts_caption import render_caption, get_ttsMode, default_tts_voice
from scipy.io import wavfile
import IPython.display as ipd
import subprocess as sp
import tempfile
try:
    if is_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = list
    
import yaml
import glob
import warnings
from pathlib import Path
import hashlib
import json
import pandas as pd

p = Path(__file__)
thisdir = p.parent

INTMAX32 = (pow(2, 31)-1)

_kw_defaults = {
    'duration': 10,
    'is_mapped': ['pitch', 'time_evo'],
    # units assumed for spatial angles (polar, azimuth, theta, phi), where
    # a cycle is a full turn, as the mapped parameters themselves are
    'angle_unit': 'cycles',
    # names to identify each source by, when looking up its table
    'source_names': None,
    # how events merged by max_notes_per_sec take their values, overriding
    # the style's own merge_mode where given
    'merge_mode': None,
    # Treatment of non-finite (NaN or infinite) input data, overriding the
    # style's own handle_nans where given
    'handle_nans': None,
    # Style File
    'style' : None,
    'caption': None,
    'name': None,
    'level': 1,
}

_exclude_keys = ['name', 'level']

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


def arg_hash(arg):
    if isinstance(arg, np.ndarray):
        return hashlib.md5(arg.tobytes()).hexdigest()
    return str(arg)


class AudioFigure:
    """
    A figure-like wrapper to manage, mix, and render multiple STRAUSS sonifications.
    """
    def __init__(self, samprate=48000, system="stereo"):
        self.length = None             # Master duration in seconds
        self.samprate = samprate         # Master sampling rate
        self.system = system             # Master channel format (mono/stereo/etc.)
        
        self.figure_hashes = {}
        self.style_cycle_index = 0
        self.style_cycle = {"theremin",
                            "windy",
                            "clicker"}

        # Store sonifications in a dict for easy access and modification
        self.sonifications = {}
        
        # Store mixing levels
        self.levels = {}

        # Store any prescribed styles
        self.styles = {}
        
        # The final mixed audio array
        self.master_audio = None

        # has this figure been rendered?
        self.is_rendered = False
    
        
    def sonify(self, *args, **kwargs):
        """
        Generate a sonification in a matplotlib-like interface and add it to the AudioFigure.

        Args:
            data: The input data for sonification. This can be multiple arrays, 
                    a path to a CSV file, or a Pandas DataFrame.
            style (dict, optional): A dictionary defining the sonification style,
                                    including parameters for score, sources, and generator.
            **kwargs: Additional parameters to override or supplement the style.

        Returns:
            The generated Sonification object.
        """
        
        sonpars, is_default = fill_from_kwargs(kwargs)
        
        # Load style
        if not sonpars['style']:
            sonpars['style'] = 'default'
        style = styles.Style(**load_style(sonpars['style']))

        # an entry is a data mapping or a fixed value by what it holds, not by
        # where it sits in the map. Data mappings take the input data in the
        # order they are given
        data_maps = [m for m in style.map if m.fixed is None]
        fixed_maps = [m for m in style.map if m.fixed is not None]
        
        # Check if data is in CSV or Pandas DataFrame format
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, (str, Path)) and Path(arg).suffix.lower() == '.csv':
                df = pd.read_csv(arg)
            elif isinstance(arg, pd.DataFrame):
                df = arg.copy()
            else:
                df = None
            
            # If using CSV or DF, check for specified 'input' columns in style and resolve these
            if df is not None:
                resolved = []
                for i, mapping in enumerate(data_maps):
                    col = getattr(mapping, 'input', None)
                    if col is None:
                        # If no input, default to auto-mapping based on position in style file
                        if i < len(df.columns):
                            resolved.append(df.iloc[:, i].to_numpy(dtype=float, na_value=np.nan))
                        else:
                            break
                    elif isinstance(col, int):
                        # If input is int, use that column index
                        if col >= len(df.columns):
                            raise IndexError(
                                f"Column index {col} out of range. "
                                f"DataFrame has {len(df.columns)} columns."
                            )
                        resolved.append(df.iloc[:, col].to_numpy(dtype=float, na_value=np.nan))
                    elif isinstance(col, str):
                        # If string, use that column name
                        if col not in df.columns:
                            raise KeyError(
                                f"Column '{col}' not found in data. "
                                f"Available columns: {list(df.columns)}"
                            )
                        resolved.append(df[col].to_numpy(dtype=float, na_value=np.nan))
                    else:
                        raise TypeError(
                            f"style.map[i].input must be an int (column index) or str "
                            f"(column name), got {type(col).__name__!r}"
                        )
                args = tuple(resolved)            

        # We analyse if this is a repeat sonification...
        # Initially filter relevent keys
        kwargs_to_hash = {k: v for k, v in sonpars.items() if k not in _exclude_keys}
        args_to_hash = [arg_hash(arg) for arg in args]
        
        # Put it in a dictionary, and get a unique hash
        param_dict = {
            'args': args_to_hash, 
            'kwargs': kwargs_to_hash  # Use the filtered dict here
        }
        param_str = json.dumps(param_dict, sort_keys=True)
        current_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()

        # now check against existing hashes and rename sonification
        # or re-set volume level if necessary
        if current_hash in self.figure_hashes.keys():
            name = self.figure_hashes[current_hash]
            mssg = 'Matching sonification exists, '
            actions = []
            if ('name' in sonpars) and sonpars['name'] and sonpars['name'] != name:
                actions.append('renaming')
                self.rename(old=name, new=sonpars['name'])
                name = sonpars['name']
            if (sonpars['level'] != self.levels[name]):
                actions.append(f"re-setting level to {sonpars['level']}")
                self.set_level(name, sonpars['level'])
            if not actions:
                actions.append('skipping')
            print(mssg + ' and '.join(actions) + '...')
            
            # replace the named hash
            self.figure_hashes[current_hash] = name
            return self.sonifications[name]
                
        if len(args) == 0:
            raise Exception("No data to sonify!")
        elif len(args) == 1:
            args = [np.arange(len(args[0])), args[0]]
            tlims = (0,args[0][-1])
                
        if len(data_maps) > len(args):
            # a style may map more than the data given, drop these quietly for now.
            data_maps = data_maps[:len(args)]
        elif len(args) > len(data_maps):
            # data given that no mapping asks for is more likely a mistake
            warnings.warn(f"{len(args)} data arrays were given but the style "
                          f"maps only {len(data_maps)} parameters, so the last "
                          f"{len(args) - len(data_maps)} will not be used.")

        source_names = sonpars['source_names']

        # a merge_mode keyword overrides the style
        merge_mode = sonpars['merge_mode'] or style.merge_mode
        if merge_mode not in ('average', 'central'):
            raise ValueError(f"merge_mode '{merge_mode}' is not recognised, "
                             "choose from 'average' or 'central'.")

        # a handle_nans keyword overrides the style
        handle_nans = sonpars['handle_nans'] or style.handle_nans
        if handle_nans not in sources.nan_modes:
            raise ValueError(f"handle_nans '{handle_nans}' is not recognised, "
                             f"choose from {list(sources.nan_modes)}.")

        if (style.sources.lower() == 'events') and style.max_notes_per_sec:
            # names are given per input event, so must be downsampled too
            if (source_names is not None) and (len(source_names) != len(args[0])):
                raise ValueError(f"Got {len(source_names)} source names for "
                                 f"{len(args[0])} input events.")

            tdx = None
            for i, mapping in enumerate(data_maps):
                if mapping.output == 'time':
                    tdx = i
                    break

            if tdx is not None:
                # an event with no finite time has nowhere in the sonification
                # to be placed, so cannot be sorted into a cluster to merge
                # with. 
                finite = np.isfinite(np.asarray(args[tdx], dtype=float))
                if not finite.all():
                    warnings.warn(f"Dropping {(~finite).sum()} events with a "
                                  "non-finite 'time'")
                    args = [np.asarray(a)[finite] for a in args]
                    if source_names is not None:
                        source_names = [n for n, f in zip(source_names, finite) if f]

                tlims = data_maps[tdx].input_range
                tlims = set_limits(tlims, args[tdx], warn=False)
                time = rescale_values(args[tdx], tlims, (0,1))

            # azimuthal angles wrap around the circle, so must be merged as
            # directions. Any angle given an input_range is mapped linearly
            # rather than wrapped, so is merged that way too
            amax = sources.angle_unit_maxs[sonpars['angle_unit']]
            cyclic = {i: amax for i, m in enumerate(data_maps)
                      if (m.output in sources.azimuthal_angles)
                      and ('input_range' not in m.model_fields_set)}

            # lets now thin the data according to max events per second if using events
            args, clusters = merge_events(sonpars['duration'], style.max_notes_per_sec,
                                          time, args, mode=merge_mode,
                                          time_index=tdx, cyclic=cyclic,
                                          return_index=True)

            if source_names is not None:
                if merge_mode == 'central':
                    # the merged event is the event representing it, so is named
                    # by it
                    source_names = [source_names[i] for i in clusters['central']]
                else:
                    # the merged event is an average of several, so name it for
                    # the span it covers, from the earliest event to the latest
                    source_names = [source_names[a] if a == b else
                                    f'{source_names[a]}->{source_names[b]}'
                                    for a, b in zip(clusters['first'],
                                                    clusters['last'])]
        
        to_map = []
        in_lims = {}
        out_lims = {}
        mapping_functions = {}

        # track where each mapping came from, so that tables and summaries
        # can tell user-specified mappings from those we add here
        origin = {}
        
        # Iterate through the mappings and add lims and funcs to their own dicts
        for mapping in data_maps:

            if mapping.output == 'time' and style.sources == 'objects':
                # Automatically swap time for time_evo if using Objects
                mapping.output = 'time_evo'
                
            if mapping.output == "cutoff":
                # Automatically turn on cutoff filter
                style.generator.mods = style.generator.mods or {}
                style.generator.mods["filter"] = "on" 
                
            to_map.append(mapping.output)
            origin[to_map[-1]] = 'mapped'

            # only limits explicitly  asked for given angle_unit handling
            if 'input_range' in mapping.model_fields_set:
                in_lims[to_map[-1]] = mapping.input_range
            if mapping.output_range:
                out_lims[to_map[-1]] = mapping.output_range
            if mapping.function:
                funcs = [mapping.function] if isinstance(mapping.function, str) else mapping.function
                mapping_functions[to_map[-1]] = [styles.MAPPING_FUNCTIONS[f] for f in funcs]
        map_data = dict(zip(to_map, args[:len(data_maps)]))
        
        if 'pitch' not in to_map:
            to_map.append('pitch')
            origin['pitch'] = 'auto'

            if style.sources == 'objects':
                nnote = len(style.notes)
                for k in map_data.keys():
                    # Use a list of arrays for Objects
                    map_data[k] = [map_data[k]]*nnote
                map_data['pitch'] = list(range(nnote))
            
            else:  # Events
                map_data["pitch"] = np.zeros(len(map_data[to_map[0]]))
        
        # we now iterate through style fixed values
        for mapping in fixed_maps:
            if mapping.output in to_map:
                warnings.warn(f"'{mapping.output}' is mapped by the style, but "
                              f"is also being fixed at {mapping.fixed}.")
            else:
                to_map.append(mapping.output)
            origin[mapping.output] = 'fixed'

            if mapping.output not in sources.spatial_angles:
                lims = sources.param_lim_dict[mapping.output]
                in_lims[mapping.output] = lims
                out_lims[mapping.output] = lims
            # spatial angles are left to apply_mapping_functions, which
            # scales them by angle_unit and wraps them around the circle

            map_data[mapping.output] = len(map_data[to_map[0]])*[mapping.fixed]

        # and finally overwrite with any kwarg fixed values:
        for k in sonpars.keys():
            ksplit = k.split('fix_')
            if len(ksplit) > 1:
                prop = ksplit[1]
                if prop in to_map:
                    # already mapped, so overwrite it in place rather than
                    # mapping the same parameter twice
                    warnings.warn(f"'{prop}' is mapped by the style, but is being "
                                  f"overwritten with the fixed value {sonpars[k]}.")
                else:
                    to_map.append(prop)
                map_data[prop] = [sonpars[k]]*len(map_data[to_map[0]])
                # as above: angles are scaled and wrapped later
                
                if prop not in sources.spatial_angles:
                    in_lims[prop] = sources.param_lim_dict[prop]
                    out_lims[prop] = sources.param_lim_dict[prop]
                origin[prop] = 'fixed'

       
        _sources = getattr(sources, style.sources.capitalize())(to_map, handle_nans=handle_nans)
        _sources.origin = origin

        # report angles in whatever units they were given in, or in degrees
        # where the user expressed no preference
        if not is_default['angle_unit']:
            _sources.table_angle_unit = sonpars['angle_unit']
        _sources.fromdict(map_data)

        # name the sources, now that we know how many there are
        if source_names is not None:
            _sources.names = source_names
        _sources.apply_mapping_functions(map_funcs=mapping_functions, map_lims=in_lims,
                                         param_lims=out_lims, angle_unit=sonpars['angle_unit'])

        # Set up Generator
        gentype = style.generator.type
        
        if gentype == 'sampler':
            s = style.generator.sample
            if '/' in s or '\\' in s or '.' in s:
                # this is probably intended to be a file path
                asset = s
            else:
                # if not, assume intention is in-built name
                asset = assets.get_asset_path(s.lower())
            _generator = getattr(generator, "Sampler")(asset, sf_preset=style.generator.sf_preset)
        else:
            _generator = getattr(generator, gentype.capitalize())()
        _generator.load_preset(style.generator.preset)
        if style.generator.mods:
            _generator.modify_preset(style.generator.mods)
            
        # Set up Score
        snotes = style.notes
    
        if not isinstance(style.notes, str):    
            snotes = [snotes]
            
        _score = Score(snotes, length=sonpars['duration'], pitch_binning=style.pitch_binning)
        
        # Combine into Sonification object
        _sonification = Sonification(
            score=_score,
            sources=_sources,
            generator=_generator,
            audio_setup=self.system,
            caption=sonpars["caption"],
            samprate=_generator.samprate, # Use generator's samprate for consistency
            handle_nans=handle_nans,
        )
            
        name = self.add(_sonification, name=sonpars["name"], level=sonpars["level"])
        self.figure_hashes[current_hash] = name
        
        return _sonification
    
    
    def list_sonifications(self):
        for i, k in enumerate(self.sonifications.keys()):
            print(f"\t{i+1}.\t {k}")

    def _get_sonification(self, name=None):
        """Look up one of the figure's sonifications by name.

        Args:
          name (`optional`, :obj:`str`): name of the sonification. Can
            be omitted where the figure holds only one.

        Returns:
          soni (:class:`~strauss.sonification.Sonification`): the named
          sonification

        Raises:
          KeyError: if no such sonification exists, or if no name is
            given for a figure holding more than one.
        """
        if not self.sonifications:
            raise KeyError("AudioFigure has no sonifications yet.")

        if name is None:
            if len(self.sonifications) > 1:
                raise KeyError("AudioFigure has more than one sonification, so "
                               "a 'name' is needed. Choose from: "
                               f"{list(self.sonifications)}")
            return next(iter(self.sonifications.values()))

        if name not in self.sonifications:
            raise KeyError(f"No sonification named '{name}' in this AudioFigure. "
                           f"Choose from: {list(self.sonifications)}")

        return self.sonifications[name]

    def get_event_table(self, name=None, include_input=False):
        """Get the table of events for a sonification.

        Wrapper for
        :meth:`~strauss.sonification.Sonification.event_table`, giving a
        row per event with the time it sounds, the note played and the
        value of each user-specified mapped parameter.

        Args:
          name (`optional`, :obj:`str`): name of the sonification. Can
            be omitted where the figure holds only one.
          include_input (`optional`, :obj:`bool`): if True, also give
            the input data value of each parameter, before mapping.

        Returns:
          table (:obj:`pandas.DataFrame`): one row per event
        """
        return self._get_sonification(name).event_table(include_input=include_input)

    def get_object_table(self, name=None, source=None, include_input=False):
        """Get the table of one object's evolution in a sonification.

        Wrapper for
        :meth:`~strauss.sonification.Sonification.object_table`, giving
        a row per point in the object's evolution.

        Args:
          name (`optional`, :obj:`str`): name of the sonification. Can
            be omitted where the figure holds only one.
          source (`optional`, :obj:`str` or :obj:`int`): name or index
            of the source. Can be omitted where the sonification has
            only one.
          include_input (`optional`, :obj:`bool`): if True, also give
            the input data value of each parameter, before mapping.

        Returns:
          table (:obj:`pandas.DataFrame`): one row per point in the
          object's evolution
        """
        soni = self._get_sonification(name)
        return soni.object_table(source, include_input=include_input)

    def get_fixed_table(self, name=None, source=None):
        """Get the table of unmapped parameters for a sonification.

        Wrapper for
        :meth:`~strauss.sonification.Sonification.fixed_table`, giving a
        row per parameter that the user did not map, and the value it
        takes.

        Args:
          name (`optional`, :obj:`str`): name of the sonification. Can
            be omitted where the figure holds only one.
          source (`optional`, :obj:`str` or :obj:`int`): name or index
            of a source to report values for. If omitted, values are
            reported for the sonification as a whole.

        Returns:
          table (:obj:`pandas.DataFrame`): one row per unmapped parameter
        """
        return self._get_sonification(name).fixed_table(source)

    def get_table(self, name=None, source=None, include_input=False):
        """Get the timing table for a sonification.

        Returns the table appropriate to the sonification's source type
        - :meth:`get_event_table` for `Events`, where the sonification
        has a single table of events, and :meth:`get_object_table` for
        `Objects`, where each source has a table of its own evolution.

        Args:
          name (`optional`, :obj:`str`): name of the sonification. Can
            be omitted where the figure holds only one.
          source (`optional`, :obj:`str` or :obj:`int`): name or index
            of the source, for `Objects` sonifications. Can be omitted
            where the sonification has only one.
          include_input (`optional`, :obj:`bool`): if True, also give
            the input data value of each parameter, before mapping.

        Returns:
          table (:obj:`pandas.DataFrame`): the sonification's table of
          events, or the named source's table of evolution

        Raises:
          TypeError: if a `source` is given for an `Events`
            sonification, where events are not divided by source.
        """
        soni = self._get_sonification(name)

        if isinstance(soni.sources, sources.Events):
            if source is not None:
                raise TypeError("Events sonifications have a single table of "
                                f"all events, so cannot take source '{source}'.")
            return soni.event_table(include_input=include_input)

        return soni.object_table(source, include_input=include_input)

    def list_tables(self):
        """Print the tables available from this figure.

        Lists each sonification with the tables it offers - a single
        table of events for `Events` sonifications, or one per named
        source, with the note it plays, for `Objects`.
        """
        for i, (name, soni) in enumerate(self.sonifications.items()):
            stype = type(soni.sources).__name__
            print(f"\t{i+1}.\t {name} ({stype})")

            if isinstance(soni.sources, sources.Events):
                print(f"\t\t - {soni.sources.n_sources} events")
                continue

            notes, _ = soni._assign_notes()
            for sname, note in zip(soni.sources.names, notes):
                print(f"\t\t - {sname} ({note})")

    def rename(self, old, new):
        dicts = [self.sonifications, self.levels, self.styles, self.figure_hashes]
        for d in dicts:
            d[new] = d[old]
            del d[old]
            
    def add(self, soni, name=None, level='0dB', style=None):
        """
        Add a sonification to the figure.
        Enforces the common timebase so it can be mixed seamlessly.
        
        Parameters:
            soni: The sonification object.
            name (str): Optional name for the track.
            level (str or float): Mixing level (e.g., '0dB', '-6dB', 0.5).
        """
        # Enforce global timebase on the child
        soni.samprate = self.samprate
        soni.system = self.system
        
        # Set up channels and channel_arrays on first sonification
        if self.length is None:
            self.length = soni.score.length

            self.channels = channels.audio_channels(setup=self.system)

            self.channel_arrays = {}

            for c in range(self.channels.Nmics):
                self.channel_arrays[str(c)] = Stream(
                    self.length,
                    self.samprate
                )

        if self.sonifications:
            # preliminary check if sonification is consistent with existing timebase
            # TODO: build this out to a better overall system
            pre_soni = list(self.sonifications.values())[0]
            mssg = ''
            if pre_soni.channels.setup != soni.channels.setup:
                mssg += (f"Clashing sonification channel setup: added setup '{soni.channels.setup}' does not "
                         f"match existing '{pre_soni.channels.setup}.'")

            new_length = np.round(soni.score.length, decimals=1)
            old_length = np.round(pre_soni.score.length, decimals=1)
            if old_length != new_length:
                mssg += (f"Clashing sonification length: added duration {new_length} s does not "
                         f"match existing '{old_length} s.'")
            if mssg:
                raise ValueError(mssg)

            
        # Determine key name
        if name is None:
            name = f"sonification_{len(self.sonifications) + 1}"
            
        # will need to re-render now there's a new sonification
        self.is_rendered = False

        # reset flag for reprocessing in style
        # style.reset_processing_flag()
        
        self.sonifications[name] = soni
        self.levels[name] = level
        self.styles[name] = style
        
        return name
        
    def remove(self, name):
        """Remove a sonification from the figure."""
        if name in self.sonifications:
            del self.sonifications[name]
        if name in self.levels:
            del self.levels[name]

        # change in state, need to re-render
        self.is_rendered = False
            
    def set_level(self, name, level):
        """
        Update the mixing level for a specific sonification.
        
        Parameters:
            name (str): The name of the sonification.
            level (str or float): The new level (e.g., '-3dB', 0.8).
        """
        if name in self.sonifications:
            self.levels[name] = level
            
            # change in state, need to re-render
            self.is_rendered = False

        else:
            raise KeyError(f"Sonification '{name}' not found in AudioFigure.")

    def _parse_level(self, level):
        """ 
        Parses a mixing level to a linear amplitude fraction.
        
        Supports:
        - Strings ending in 'dB' (e.g., '-6 dB', '-inf dB')
        - Floats/linear fractions (0.0 to 1.0)
        """
        if isinstance(level, str):
            level_clean = level.strip().lower()
            if level_clean == '-inf db':
                return 0.0
            elif level_clean.endswith('db'):
                try:
                    db_val = float(level_clean.replace('db', '').strip())
                    return float(db_to_amplitude(db_val))
                except ValueError:
                    raise ValueError(f"Invalid dB format: {level}")
            else:
                # specific case for just a number in string format
                try:
                    return float(level)
                except ValueError:
                    raise ValueError(f"Unknown level format: {level}")
        elif isinstance(level, (int, float)):
            return float(level)
        else:
            raise TypeError(f"Level must be str or float, got {type(level)}")

    def render(self, progress=True, normalize='peak'):
        """
        Regenerate all attached sonifications and mix them additively.
        
        Parameters:
        normalize (str): 'peak', 'soft', or None to handle clipping.
        """
        
        # Initialize master audio buffer
        # Shape: (N_channels, N_samples) or (N_samples, N_channels) depending on STRAUSS convention.
        # Assuming (N_channels, N_samples) based on _align_audio and standard audio libs
        n_samples = int(self.length * self.samprate)
        self.master_audio = np.zeros((self.channel_arrays['0'].values.size, len(self.channel_arrays)))

        if progress:
            print('Processing audio figure...')
        self.list_sonifications()
        for name, soni in tqdm(self.sonifications.items()) if progress else self.sonifications.items():
            # 1. Generate the individual track
            soni.render(progress=False)
            
            # 2. Extract the rendered numpy array 
            track_audio = soni._make_out_array(embed_caption=False)

            # 3. Apply Mixing Level
            if name in self.levels:
                amp = self._parse_level(self.levels[name])
                track_audio = track_audio * amp
            # 4. Align timebases (safety catch in case of rounding errors in generation)
            # track_audio = self._align_audio(track_audio, self.channels.Nmics, n_samples)
            
            # 5. Additive mixing
            # Ensure shapes match before adding; if strict alignment isn't enabled, 
            # we assume strauss generation is accurate to the sample.
            if track_audio.shape == self.master_audio.shape:
                self.master_audio += track_audio
            else:
                # Basic safety add if dimensions differ slightly (e.g. 1 sample off)
                aligned_audio = self._align_audio(track_audio, self.channels.Nmics, n_samples)
                self.master_audio += aligned_audio

        # 6. Apply clipping protection
        # self._apply_normalization(normalize)
        self.vmax = abs(self.master_audio).max()
        self.master_audio *= (pow(2, 31)-1)/self.vmax
        self.master_audio = self.master_audio.astype('int32')
        self.is_rendered = True
        
    def notebook_display(self, show_waveform=True):
        if not self.is_rendered:
            self.render()

        has_ticks = hasattr(self, 'tick_channels')
        if len(self.channels.labels) == 1:             
            outfmt = np.column_stack([self.master_audio]*2)
        else:
            outfmt = self.master_audio[:,:2]
        if len(self.channels.labels) > 2:
            print("Warning: for more than two channels, only first two channels are mapped to L and R, respectively.")
        if has_ticks:
            # add the ticks
            for c in range(outfmt.shape[0]):
                outfmt[c] += self.tick_channels['0'].values*self.tick_vol / self.vmax
        ipd.display(ipd.Audio(outfmt.T,rate=self.samprate, autoplay=False))

    def save(self, fname):
        # combine and write out file. first check extension
        if not self.is_rendered:
            self.render()

        fsplit = str(fname).split('.')
        if len(fsplit) < 2:
            warnings.warn('No file extension in provided fname. Assuming WAV...')
        ext = fsplit[-1].lower()
        if ext != 'wav':
            # check we can use ffmpeg binary 
            try:
                sp.run(['ffmpeg','-h'],capture_output=1, check=1)
            except FileNotFoundError as e: 
                raise FileNotFoundError(f"""
                'ffmpeg' doesn't appear to be available in the local environment.
                This may need to be installed manually. To install ffmpeg visit
                https://www.ffmpeg.org/download.html.
                {str(e)}
                """)
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                # now first write the wav to a temporary file
                wavfile.write(tmp.name, self.samprate, self.master_audio)
                try:
                    # try (naive) convert with ffmpeg
                    sp.run(['ffmpeg', '-i', f'{tmp.name}', f'{fname}'],
                           capture_output=1, check=1)
                except sp.CalledProcessError as e:
                    # if ffmpeg can't do it for whatever reason, raise
                    raise Exception(f"""
                    'ffmpeg' failed to convert '.wav' to '.{ext}' succesfully:
                    {str(e)}
                    {e.stderr}""")
        else:
            wavfile.write(fname, self.samprate, self.master_audio)
        
        print(f"Saved {fname}")

        
    def _align_audio(self, audio, expected_channels, expected_samples):
        """
        Helper method to guarantee the audio array perfectly matches the master track shape.
        Pads with zeros or truncates the end if there is a mismatch.
        """
        # Check channel shape
        if audio.shape[0] != expected_channels:
            # If mono source in stereo system, duplication might be needed, 
            # but strictly raising error for now as per original code.
            self.remove()
            raise ValueError(f"Channel mismatch: Source has {audio.shape[0]}, Master has {expected_channels}")
            
        current_samples = audio.shape[1]
        
        if current_samples < expected_samples:
            # Pad with silence at the end
            padding = np.zeros((expected_channels, expected_samples - current_samples))
            return np.concatenate((audio, padding), axis=1)
        elif current_samples > expected_samples:
            # Truncate
            return audio[:, :expected_samples]
        
        return audio

    def _apply_normalization(self, method):
        """Prevents the additive mix from causing digital clipping (>1.0 or <-1.0)."""
        if method == 'peak':
            # Scales everything down proportionally if it breaches 1.0
            peak = np.max(np.abs(self.master_audio))
            if peak > 1.0:
                self.master_audio /= peak
        elif method == 'soft':
            # Applies a soft-clipping limiter (tanh) for a warmer, compressed mix
            self.master_audio = np.tanh(self.master_audio)
