from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, List, Union, Tuple
from sources import param_lim_dict as valid_params
from pychord import Chord
import random

metadata = {
    # Field titles, descriptions, and examples will go here
}

class Mapping(BaseModel):
    # Input can be a string (column header), an int (column index) or None (to auto-map to available columns in dataset)
    input: Optional[Union[str, int]] = Field(default=None, ge=0)

    # 'map_lims' in STRAUSS v1. Must be a tuple/list of two elements which can either be strings (for percentiles) or floats (ints are converted to floats)
    input_range: Optional[Tuple[Union[str, float], Union[str, float]]] = Field(default=('0%','100%'))

    # Output is required (...) and must be a string
    output: str = Field(...)

    # 'param_lims' in v1. Must be a tuple/list of two numbers. Default needs to be set once the output parameter is known.
    output_range: Optional[Tuple[float, float]] = Field(default=None)


    @field_validator('input_range')
    @classmethod
    def validate_input_range(cls, value: Optional[Tuple[Union[str, float], Union[str, float]]]):
        
        if value is None:
            return value
        
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
        if value[0] >= value[1]:
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
    
    @model_validator(mode="after")
    def validate_output_range(self):

        if self.output_range is None:
            return self

        valid_min, valid_max = valid_params[self.output]

        for val in self.output_range:

            if val < valid_min or val > valid_max:
                if self.output == 'pitch':
                    continue
                raise ValueError(f'Parameter limits for "{self.output}" must be between those in param_lim_dict.')
            
        return self

