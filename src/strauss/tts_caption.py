from scipy.io import wavfile
from scipy.interpolate import interp1d
import numpy as np
import strauss.utilities as utils
import re
try:
    from TTS.api import TTS
except (OSError, ModuleNotFoundError) as sderr:
    def TTS(*args, **kwargs):
        raise TTSIsNotSupported("strauss has not been installed with text-to-speech support. \n"
              "This is not installed by default, due to some specific module requirements of the TTS module."
              "Reinstalling strauss with 'pip install strauss[TTS]' will give you access to this function")

class TTSIsNotSupported(Exception):
    pass

def render_caption(caption, samprate, model, caption_path):
    '''The render_caption function generates an audio caption from text input
    and writes it as a wav file. If the sample rate of the model is not equal 
    to that passed from sonification.py, it resamples to the correct rate and
    re-writes the file. Text from user input is converted with text-to-speech
    software from Coqui-AI - https://pypi.org/project/TTS/ . You can view 
    publicly available voice models with 'TTS.list_models()'

    Args:
      caption (:obj:`str`): script to be spoken by the TTS voice
      samprate (:obj:`int`): samples per second
      model (:obj:`str`): valid name of TTS voice from the underying TTS
        module
      model (:obj:`str`): valid name of TTS voice from the underying TTS
        module
      caption_path (:obj:`str`): filepath for spoken caption output
    '''

    # TODO: do this better with logging. We can filter TTS function output, e.g. alert to downloading models...
    print('Rendering caption (this can take a while if the caption is long, or if the TTS model needs downloading)...')
    
    # capture stdout from the talkative TTS module
    with utils.Capturing() as output:
        # Load in the tts model
        tts = TTS(model, progress_bar=False, gpu=False)

        # render to speech, and write as a wav file (allow )
        tts.tts_to_file(text=caption, file_path=caption_path)

        
    # Read the file back in to check the sample rate
    rate_in, wavobj = wavfile.read(caption_path)
    
    #If it doesn't match the required rate, resample and re-write
    if rate_in != samprate:
        new_wavobj = utils.resample(rate_in, samprate, wavobj)
        wavfile.write(caption_path, samprate, new_wavobj)


