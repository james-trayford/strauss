""" The :obj: `caption` submodule: adding audio captioning

This submodule uses TTS (text to speech) to convert text to a wav file.

Attributes:
   
To do:
   Add choice of voices.
   
"""

from TTS.api import TTS

class Caption:
    """ Class defining audio captioning for the sonification.
    
    The Caption class generates audio captioning in the form of a wav file. It 
    captures text from user input and converts it with text-to-speech software 
    from Coqui-AI - https://pypi.org/project/TTS/
    
    Args:
        
    """
    
    fin = "draft.txt"
    fout = "caption.wav"
    model_name = "tts_models/en/jenny/jenny"

    tts = TTS(model_name, progress_bar = False, gpu = False)
    file_in = open(fin, "r").read().replace("\n", " ")

    tts = TTS(model_name, progress_bar=False, gpu=False)
    tts.tts_to_file(text = file_in, file_path = file_out)


