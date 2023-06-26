from TTS.api import TTS
from scipy.io import wavfile
from scipy.interpolate import interp1d
import numpy as np
       
def render_caption(caption, samplerate, model, caption_path):
    ''' The render_caption function generates an audio caption from text input
    and writes it as a wav file. If the sample rate of the model is not equal 
    to that passed from sonification.py, it resamples to the correct rate and
    re-writes the file. Text from user input is converted with text-to-speech
    software from Coqui-AI - https://pypi.org/project/TTS/ . You can view 
    publicly available voice models with 'TTS.list_models()'
    '''
    
    # Load in the tts model, render to speech, and write as a wav file
    tts = TTS(model, progress_bar=False, gpu=False)
    tts.tts_to_file(text=caption, file_path=caption_path)
    
    # Read the file back in to check the sample rate
    old_samplerate, old_audio = wavfile.read(caption_path)
    
    #If it doesn't match the required rate, resample and re-write
    if old_samplerate != samplerate:
        duration = old_audio.shape[0] / old_samplerate

        time_old  = np.linspace(0, duration, old_audio.shape[0])
        time_new  = np.linspace(0, duration, int(old_audio.shape[0] * samplerate / old_samplerate))

        interpolator = interp1d(time_old, old_audio.T)
        new_audio = interpolator(time_new).T

        wavfile.write(caption_path, samplerate, np.round(new_audio).astype(old_audio.dtype))
        
    pass
    

