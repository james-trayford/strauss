from pydantic import BaseModel, ConfigDict, PrivateAttr, Field, field_validator, model_validator
from typing import Optional, Literal, Dict, List, Union, Tuple
from ..sources import param_lim_dict as valid_params
from pathlib import Path
import random
import numpy as np

# Helper to ensure the right mapping function is used whether input is np array or list of np arrays
def get_func(func):
    def wrapper(x):
        if isinstance(x, list):
            return [func(v) for v in x]
        return func(x)
    return wrapper

# The list of available mapping functions
MAPPING_FUNCTIONS = {
    "log2": get_func(np.log2), # TODO - need to deal with non-positive inputs for log functions
    "log10": get_func(np.log10),
    "invert": get_func(np.negative),
    "abs": get_func(np.abs),
    "reciprocal": get_func(np.reciprocal)
}

AVAILABLE_FUNCTIONS = ", ".join(MAPPING_FUNCTIONS)

def get_presets(generator_type):

    # Get the path to src/strauss
    BASE_DIR = Path(__file__).resolve().parent.parent

    preset_names = []

    folder = BASE_DIR / 'presets' / generator_type

    if folder.is_dir():
        preset_names.extend(p.stem for p in folder.rglob('*.yml'))

    return preset_names


class MonitoredBaseModel(BaseModel):
    
    # Private attributes are excluded from .model_dump() and schema generation
    _is_touched: bool = PrivateAttr(default=True)
    
    def __setattr___(self, name, value):
        self._is_touched = True
        super().__setattr__(name, value)

    @property
    def needs_reprocessing(self) -> bool:
        return self._is_touched
        
    def reset_processing_flag(self):
        self._is_touched = False
    

class Mapping(MonitoredBaseModel):
    # Input can be a string (column header), an int (column index) or None (to auto-map to available columns in dataset)
    input: Union[str, int, None] = Field(
        default=None,
        title='Input Parameter',
        description=(
            'Source data variable to map from. '
            'Can be a column name (str), column index (int), '
            'or null to auto-map to available columns in the data.'
        ),
        examples=['magnitude', 2, None]
    )

    # 'map_lims' in STRAUSS v1. Must be a tuple/list of two elements which can either be strings (for percentiles) or floats (ints are converted to floats)
    input_range: Tuple[Union[str, float], Union[str, float]] = Field(
        default=('0%','100%'),
        title='Input Range',
        description=(
            'Range of input values to map from. '
            'Known as "map_lims" in STRAUSS v1. '
            'Can either be string percentiles, or absolute values in the data.'
        ),
        examples=[('5%', '95%'), (24, 68.3)]
    )

    # Output is required (...) and must be a string
    output: str = Field(
        ...,
        title='Output Parameter',
        description='The sound parameter to map the input to.',
        examples=['azimuth', 'pitch', 'volume']
    )


    # 'param_lims' in v1. Must be a tuple/list of two numbers. Defaults to None to allow STRAUSS to use the full range for that output
    output_range: Optional[Tuple[float, float]] = Field(
        default=None,
        title='Output Range',
        description=(
            'Range of the output sound parameter to use. '
            'Known as param_lims in STRAUSS v1.'
            ),
        examples=[(0.1, 0.8), (0, 24)]
    )

    # The mapping function can either be a string e.g. 'log(x)' or a list of strings e.g. ['log(x)', '-x']. Defaults to None (x=x).
    function: Union[str, List[str], None] = Field(
        default=None,
        title='Mapping Function',
        description=(
            'The mathematical function to apply when mapping this parameter data. '
            'This is useful for scaling the data (e.g logarithmically) or reversing the polarity (-x) '
            'so that the biggest values become the smallest values. '
            f'The string must correspond to one of the pre-defined functions: {AVAILABLE_FUNCTIONS}'
            'Multiple functions can be applied at once by providing a list of strings in the order you want the functions applied.'
        ),
        examples=['log2', ['invert', 'log10']]
    )

    fixed: Union[int, float, None] = Field(
        default=None,
        title='Fixed Value',
        description=(
            'The value to hold this parameter at for the duration of the sonification.'
            'Some parameters are "evolvable" but we might want to fix them at certain values. '
            'For example, we might want to hard pan a source to the left for the duration of the sonification. '
            'We can achieve this by specifying "output": "pan" and "fixed": 0. '
            'If using a fixed value, any input specified will be ignored, as the fixed value takes precedence.'
            'The fixed value specified needs to be within the limits of the output parameter specified.'
        ),
        examples=[0, 1, 0.5, 0.3, 0.9]
    )

    @field_validator('input')
    @classmethod
    def validate_input(cls, value: Union[str, int, None]):

        if value is None:
            return value
        
        if isinstance(value, int) and value < 0:
            raise ValueError('Column index must be a positive number.')
        
        return value



    @field_validator('input_range')
    @classmethod
    def validate_input_range(cls, value: Tuple[Union[str, float], Union[str, float]]):
        
        for v in value:
           
            # Check string is correct format
            if isinstance(v, str):
                if not v.endswith('%'):
                    raise ValueError('Input range must either be percentile strings e.g. "50%" or ints/floats.')
                if v.startswith('-'):
                    raise ValueError('Input range percentile cannot be negative.')
    
                try:
                    # Try to parse as a number 
                    num_v = float(v.replace('%',''))
                except ValueError:
                    raise ValueError('Percentile must be a number.')

        # Check lims are in correct order e.g. [lower_lim, upper_lim]
        # upper_lim - lower_lim must not = 0 as this will cause a divide by zero error        
        if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(v, (int, float)) for v in value)
        and value[0] >= value[1]):
            
            raise ValueError("Invalid range: lower bound must be < upper bound")
                
        return value
            

    @field_validator('output')
    @classmethod
    def validate_output(cls, value: str):

        # Check if output is a valid mappable parameter in STRAUSS
        if value not in valid_params.keys():
            
            # Provide suggestions
            suggestions = random.sample(list(valid_params.keys()), 2)
            raise ValueError(f'Output "{value}" is not a valid parameter. \nTry "{suggestions[0]}" or "{suggestions[1]}".')
        
        return value
    
    @field_validator('function')
    @classmethod
    def validate_function(cls, value: Union[str, List[str], None]):

        if value is None:
            return value
        
        funcs = [value] if isinstance(value, str) else value
        
        valid_functions = ", ".join(sorted(MAPPING_FUNCTIONS))

        for func in funcs:
            if func not in MAPPING_FUNCTIONS:
                raise ValueError(
                    f"'{func}' is not a valid mapping function. "
                    f"Available functions: {valid_functions}."
                )
            
        return value
    
    @model_validator(mode='after')
    def validate_output_range(self):

        if self.output_range is None:
            return self

        valid_min, valid_max = valid_params[self.output]

        for val in self.output_range:

            # Check range is within valid range
            if val < valid_min or val > valid_max:
                if self.output == 'pitch':
                    continue
                raise ValueError(f'Parameter limits for "{self.output}" must be between those in param_lim_dict.')
            
        return self
    
    @model_validator(mode='after')
    def validate_fixed(self):
        
        if self.fixed is None:
            return self
        
        valid_min, valid_max = valid_params[self.output]
        
        if not valid_min <= self.fixed <= valid_max:
            raise ValueError('Fixed value must be between the limits of the output parameter.')
        
        return self
    

class GeneratorStyle(MonitoredBaseModel):

    # Generator type - defaults to Synthesizer 
    type: Literal['sampler', 'synthesizer', 'synth', 'spectralizer'] = Field(
        default='synthesizer',
        title='Generator Type',
        description='The Generator type to use. This field accepts any capitalisation of the string. Also accepts "synth" as a shortcut for Synthesizer.',
        examples=['sampler', 'Synthesizer', 'SPECTRALIZER', 'synth']
    )

    # Generator preset. Can be either a path to a user preset or the name of a Generator preset. Defaults to 'default' because every Generator has a default.yml
    preset: Union[str, Path] = Field(
        default='default',
        title='Generator Preset',
        description=(
            'The name of the Generator preset to use (must exist in presets/{generator type}/), or a file path to a user-made preset. '
            'Defaults to "default" because every Generator has a default.yml'
        ),
        examples=['staccato', 'windy', 'path/to/my/preset.yml']
    )

    # Name or path to samples if using Sampler type
    sample: Union[Path, str, None] = Field(
        default=None,
        title='Sample',
        description=(
            'If using Sampler, this is the either the name of the sample/instrument, '
            'a file path to a directory of audio files, or a web URL to an audio file. '
            "If it's a name, STRAUSS will first check if the sample(s) are already downloaded, "
            "and if not, it will fetch them from the online library."
        ),
        examples=['Glockenspiel', 'path/to/my/samples', 'http://www.samples.com/mallets.sf2']
    )

    # Soundfont preset number, if using a .sf2 file
    sf_preset: Optional[int] = Field(
        default=None,
        gt=0,
        title='Soundfont Preset',
        description='If using a soundfont (.sf2) file, this is the preset number to use.',
        examples=[1,2,16]
    )

    # Any modifications to the Generator preset. These will be applied with the generator.modify_preset() function
    mods: Optional[dict] = Field(
        default=None,
        title='Generator Modifications',
        description=(
            'Any changes to apply to the chosen Generator preset. '
            'These will be applied with the generator.modify_preset() function'
        ),
        examples=[{'looping': 'forwardback', 'filter': 'on'}, {'loop_start': 0.2}]
        )

    @field_validator('type', mode='before')
    @classmethod
    def valiadate_type(cls, value: str):
        
        # Convert string to lowercase so that 'Sampler', 'sampler', and 'SAMPLER' are all valid.
        value = value.lower()
        value = 'synthesizer' if value == 'synth' else value # allow synth shortcut
       
        return value
    
    @field_validator('sample')
    @classmethod
    def validate_path(cls, value: Union[Path, str, None]):

        if value is None:
            return value
        
        if isinstance(value, str):
            if any(c in value for c in ('/', '\\', '.')):
                # It's a filepath
                return value
            else:
                # It's the name of a built-in sample
                return value.lower()
        
        return value
    
    @model_validator(mode="after")
    def validate_preset(self):

        # Accept explicit paths
        if isinstance(self.preset, (str, Path)):
            preset_path = Path(self.preset)
            if preset_path.exists():
                return self

        # Otherwise treat it as a preset name
        dirs = {
            "synthesizer": "synth",
            "sampler": "sampler",
            "spectralizer": "spec",
        }
        valid_presets = get_presets(generator_type=dirs[self.type])

        if self.preset not in valid_presets:
            raise ValueError(
                f"{self.preset} is not a valid preset for {self.type} "
                "Generator type. Please choose from the preset names or "
                "provide a valid path."
            )

        return self

    
    @model_validator(mode='after')
    def validate_sf_preset(self):

        if self.sf_preset is not None and self.type != 'sampler':
            raise ValueError('sf_preset can only be specified for Sampler Generator type.')
        
        return self
    
    
class Style(MonitoredBaseModel):

    # Style name is required (no default)
    name: str = Field(
        ...,
        title='Style Name',
        description='The name to give the style.',
        examples=['Sci-Fi','Windy', 'My Cool Style']
        )

    # Description is optional
    description: Optional[str] = Field(
        default=None,
        title='Style Description',
        description='An optional description to give the style.',
        examples=['Sawtooth synth paired with filter cutoff.','This style works well for univariate data.', 'This style makes my ears happy :)']
        )

    # All of the generator settings. Defaults to default STRAUSS synth.
    generator: GeneratorStyle = Field(
        default=GeneratorStyle(),
        title='Generator Style',
        description=(
            'All of the settings to set up the Generator. '
            'If omitted entirely, defaults to an empty GeneratorStyle instance, which in turn '
            'sets up the default synthesizer.'
        ),
        examples=[
            # 1. Minimal example (defaults)
            {},

            # 2. Synthesizer with preset
            {
                "type": "synthesizer",
                "preset": "windy"
            },

            # 3. Sampler with named instrument
            {
                "type": "sampler",
                "sample": "Glockenspiel"
            },

            # 4. Sampler with soundfont and preset number
            {
                "type": "sampler",
                "sample": "path/to/piano.sf2",
                "sf_preset": 2
            },

            # 5. User preset path with modifications
            {
                "type": "synthesizer",
                "preset": "path/to/my/preset.yml",
                "mods": {
                    "filter": "on",
                    "cutoff": 0.8,
                    "volume_envolope": {'A': 0.3}
                }
            }
        ]
    )

    # Source type must be Events or Objects
    sources: Literal['objects', 'events'] = Field(
        ...,
        title='Source Type',
        description='Whether to use Objects or Events type (case insensitive).',
        examples=['Objects', 'events']
    )
    
    max_notes_per_sec: Optional[int] = Field(
        default=None,
        title='Maximum Notes per Second',
        description='This optional field defines the maximum number of notes played per second '
                    'when sonifying Events. This prevents high-frequency artefacts appearing '
                    'in the audio by downsampling the input data. If using, must be between 1 and 20.',
        examples=[15, 10, 7]
        )

    # The map is the list of Mapping objects which set up the sonification parameters and limits.
    map: List[Mapping] = Field(
        ...,
        title="Parameter Map",
        description="The Map is the list of Mapping objects which set up the sonification parameters and limits.",
        examples=[
            # 1. Minimal mapping (auto input, default ranges, linear mapping)
            [
                {
                    "output": "pitch"
                }
            ],

            # 2. Simple column-name mapping
            [
                {
                    "input": "magnitude",
                    "output": "volume"
                }
            ],

            # 3. Column index with percentile range
            [
                {
                    "input": 2,
                    "input_range": ["5%", "95%"],
                    "output": "azimuth"
                }
            ],

            # 4. Absolute input range with constrained output range
            [
                {
                    "input": "temperature",
                    "input_range": [250, 350],
                    "output": "cutoff",
                    "output_range": [0.2, 0.75]
                }
            ],

            # 5. Logarithmic scaling
            [
                {
                    "input": "flux",
                    "input_range": ["1%", "99%"],
                    "output": "pitch",
                    "function": "log(x)"
                }
            ],

            # 6. Multiple functions (invert + log)
            [
                {
                    "input": "distance",
                    "output": "volume",
                    "function": ["-x", "log(x)"]
                }
            ],

            # 7. Typical multi-parameter sonification map
            [
                {
                    "input": "magnitude",
                    "input_range": ["5%", "95%"],
                    "output": "pitch"
                    
                },
                {
                    "input": "magnitude",
                    "output": "volume",
                    "output_range": [0.2, 1],
                    "function": "-x"
                },
                {
                    "input": "ra",
                    "output": "azimuth"
                }
            ]
        ]
    )

    # Notes can either be a list of notes, a list of a list of notes (for a chord progression), or the name of a chord/scale e.g. 'Cmaj7', 'D Hirajoshi'
    notes: Union[List[str], List[List[str]], str, None] = Field(
        default=None,
        title='Musical Notes',
        description=('Notes can either be a list of notes, '
        'a list of a list of notes (for a chord progression), '
        'or the name of a chord/scale.'
        ),
        examples=[['G2', 'D3', 'G3', 'B4', 'F#4'], 'Esus4', 'D Mixolydian']
    )
    
    pitch_binning: Literal['adaptive', 'uniform'] = Field(
        default='adaptive',
        title='Pitch Binning Mode',
        description='The method used to determine how the data is binned into discrete pitches (if using pitch as a parameter).'
                    'choose from "adaptive", where sources are binned by the pitch mapping such that each '
                    'interval is represented the same fraction of the time, and "uniform" where the pitch binning '
                    'is based on uniform size bins in the mapped pitch parameter.',
        examples=['adaptive', 'uniform']
    )


    @field_validator('sources', mode='before')
    @classmethod
    def lowercase_type(cls, value: str):
        # Convert string to lowercase so that 'Objects', 'objects', and 'OBJECTS' are all valid.
        return value.lower()
    
    @field_validator('pitch_binning', mode='before')
    @classmethod
    def lowercase_pitch_binning(cls, value: str):
        # Convert string to lowercase so that 'Adaptive', 'adaptive', and 'ADAPTIVE' are all valid.
        return value.lower()
    
    
    @field_validator('max_notes_per_sec')
    @classmethod
    def validate_max_notes_per_sec(cls, value: int):
        
        if not 0 < value <= 20:
            raise ValueError("max_notes_per_sec must be between 1 and 20")
        
        return value
        
    
    @field_validator('notes')
    @classmethod
    def validate_notes(cls, value: Union[List[str], List[List[str]], str, None]):

        if value is None:
            return value 
        
        # Could validate the chords/scales here? E.g. try to parse the chord name with pychord

        return value


