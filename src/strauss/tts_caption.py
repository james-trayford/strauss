from TTS.api import TTS
from scipy.io import wavfile
from scipy.interpolate import interp1d
import numpy as np
import utilities as utils
       
def render_caption(caption, samprate, model, caption_path):
    ''' The render_caption function generates an audio caption from text input
    and writes it as a wav file. If the sample rate of the model is not equal 
    to that passed from sonification.py, it resamples to the correct rate and
    re-writes the file. Text from user input is converted with text-to-speech
    software from Coqui-AI - https://pypi.org/project/TTS/ . Yohello_u can view 
    publicly available voice models with 'TTS.list_models()'
    '''
    
    # Load in the tts model, render to speech, and write as a wav file
    tts = TTS(model, progress_bar=False, gpu=False)
    tts.tts_to_file(text=caption, file_path=caption_path)
    
    # Read the file back in to check the sample rate
    rate_in, wavobj = wavfile.read(caption_path)
    
    #If it doesn't match the required rate, resample and re-write
    if rate_in != samprate:
        new_wavobj = utils.resample(rate_in, samprate, wavobj)
        wavfile.write(caption_path, samprate, new_wavobj)

