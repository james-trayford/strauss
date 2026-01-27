from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal, Dict, List, Union, Tuple
from sources import param_lim_dict as valid_params
from pathlib import Path
import random

metadata = {
    # Field titles, descriptions, and examples will go here
}

FUNCTION_WHITELIST = [
    # Our 'whitelisted' functions can go here
]

def get_presets():

    # Get the path to the directory this file is in
    BASE_DIR = Path(__file__).resolve().parent

    preset_names = []

    # Iterate through preset folders and add names to list
    for generator_type in ['sampler', 'spec', 'synth']:
        
        folder = BASE_DIR / 'presets' / generator_type

        if folder.is_dir():
            preset_names.extend(p.stem for p in folder.rglob('*.yml'))

    return preset_names


class Mapping(BaseModel):
    # Input can be a string (column header), an int (column index) or None (to auto-map to available columns in dataset)
    input: Union[str, int, None] = Field(default=None, ge=0)

    # 'map_lims' in STRAUSS v1. Must be a tuple/list of two elements which can either be strings (for percentiles) or floats (ints are converted to floats)
    input_range: Tuple[Union[str, float], Union[str, float]] = Field(default=('0%','100%'))

    # Output is required (...) and must be a string
    output: str = Field(...)

    # 'param_lims' in v1. Must be a tuple/list of two numbers. Defaults to None to allow STRAUSS to use the full range for that output
    output_range: Optional[Tuple[float, float]] = Field(default=None)

    # The mapping function can either be a string e.g. 'log(x)' or a list of strings e.g. ['log(x)', '-x']. Defaults to None (x=x).
    function: Union[str, List[str], None] = Field(default=None)


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
        if isinstance(value, Tuple[float,float]) and value[0] >= value[1]:
            raise ValueError('Lower limit must be less than upper limit.')
                
        return value
            

    @field_validator('output')
    @classmethod
    def validate_output(cls, value: str):

        # Check if output is a valid mappable parameter in STRAUSS
        if value not in valid_params.keys():
            
            # Provide suggestions
            suggestions = random.sample(valid_params.keys(), 2)
            raise ValueError(f'Output "{value}" is not a valid parameter. \nTry "{suggestions[0]}" or "{suggestions[1]}".')
        
        return value
    
    @field_validator('function')
    @classmethod
    def validate_function(cls, value: Union[str, List[str], None]):

        if value is None:
            return value
        
        funcs = [value] if isinstance(value, str) else value

        for func in funcs:
            if func not in FUNCTION_WHITELIST:
                raise ValueError(f'{func} is not a valid mapping function.')
            
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
    

class GeneratorStyle(BaseModel):

    # Generator type - defaults to Synthesizer 
    type: Literal['sampler', 'synthesizer', 'spectralizer'] = Field(default='synthesizer')

    # Generator preset (can be written with or without '.yml' suffix). Defaults to 'default' because every Generator has a default.yml
    preset: str = Field(default='default.yml')

    # Path to samples if using Sampler type
    path: Optional[Path] = Field(default=None)

    # Soundfont preset number, if using a .sf2 file
    sf_preset: Optional[int] = Field(default=None, gt=0)

    # Any modifications to the Generator preset. These will be applied with the generator.modify_preset() function
    mods: Optional[Dict] = Field(default=None)

    @field_validator('type', mode='before')
    @classmethod
    def lowercase_type(cls, value: str):
        # Convert string to lowercase so that 'Sampler', 'sampler', and 'SAMPLER' are all valid.
        return value.lower()


    @field_validator('preset')
    @classmethod
    def validate_preset(cls, value: str):

        # Check that the preset exists
        preset_name = value.removesuffix('.yml')
        valid_presets = get_presets()

        if preset_name not in valid_presets:
            raise ValueError(f'{value} is not a valid Generator preset. Please choose from the yaml file names in the /presets/ directory.')
        
        return value
    
    @field_validator('path')
    @classmethod
    def validate_path(cls, value: Optional[Path]):

        if value is None:
            return value

        if str(value).startswith('http'):
            # Do something here to validate URLs?
            return value

        if not value.exists():
            raise ValueError('The Generator path provided does not exist.')
        
        return value
    
    @model_validator(mode='after')
    def validate_sf_preset(self):

        if self.sf_preset and self.type != 'sampler':
            raise ValueError('sf_preset can only be specified for Sampler Generator type.')
        
        return self
    
    
class Style(BaseModel):

    # Style name is required (no default)
    name: str = Field(...)

    # Description is optional
    description: Optional[str] = Field(None)

    # All of the generator settings. Defaults to default STRAUSS synth.
    generator: GeneratorStyle = Field(default=GeneratorStyle())

    # Source type must be Events or Objects
    sources: Literal['objects', 'events'] = Field(...)

    # The map is the list of Mapping objects which set up the sonification parameters and limits.
    map: List[Mapping] = Field(...)

    # Notes can either be a list of notes, a list of a list of notes (for a chord progression), or the name of a chord/scale e.g. 'Cmaj7', 'D Hirajoshi'
    notes: Union[List[str], List[List[str]], str, None] = Field(default=['C3'])

    # Chord mode can either be True/False (also accepts 'on'/'off', 1/0, 'yes'/'no' etc.)
    chord_mode: bool = Field(default=True)


    @field_validator('sources', mode='before')
    @classmethod
    def lowercase_type(cls, value: str):
        # Convert string to lowercase so that 'Objects', 'objects', and 'OBJECTS' are all valid.
        return value.lower()
    
    @field_validator('notes')
    @classmethod
    def validate_notes(cls, value: Union[List[str], List[List[str]], str, None]):

        if value is None:
            return value 
        
        # Could validate the chords/scales here? E.g. try to parse the chord name with pychord

        return value
    

    

   

