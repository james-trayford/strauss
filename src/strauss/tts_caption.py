"""The :obj:`tts_caption` submodule: tool for generating spoken captions

This uses text-to-speech via the the ``TTS`` module to allow captions
represented as strings to be converted to spoken audio to precede the
sonification.
"""

from scipy.io import wavfile
import numpy as np
import strauss.utilities as utils
import ffmpeg as ff
import re
import os
import warnings
from pathlib import Path

# Ordered by preference: Kokoro > Coqui > Pyttsx3
ENGINE_PREFERENCE = ['kokoro', 'coqui-tts', 'pyttsx3']
current_tts_mode = 'None'
pipeline = None
default_tts_voice = None

def set_engine(engine_name):
    """
    Manually set the TTS engine and update the default voice.
    
    Args:
        engine_name (str): One of 'kokoro', 'coqui-tts', or 'pyttsx3'.
    """
    global current_tts_mode, pipeline, default_tts_voice
    
    try:
        if engine_name == 'kokoro':
            from kokoro import KPipeline
            pipeline = KPipeline(lang_code='b')
            current_tts_mode = 'kokoro'
            default_tts_voice = 'bf_emma'
            
        elif engine_name == 'coqui-tts':
            from TTS.api import TTS
            current_tts_mode = 'coqui-tts'
            default_tts_voice = Path('tts_models','en','jenny', 'jenny')
            
        elif engine_name == 'pyttsx3':
            import pyttsx3
            current_tts_mode = 'pyttsx3'
            default_tts_voice = {} 
            
        print(f"TTS engine successfully set to: {current_tts_mode}")
        return True
    except (OSError, ModuleNotFoundError, Exception) as e:
        print(f"Failed to load {engine_name}: {e}")
        return False

def initialize_tts():
    """Initializes the best available engine based on preference."""
    for engine in ENGINE_PREFERENCE:
        if set_engine(engine):
            return
    print("No supported text-to-speech packages found.")

# Run initial detection
initialize_tts()

def get_ttsMode():
    return current_tts_mode

def getVoices(info=False):
    """Get available voices for the currently active TTS engine."""
    voices = []
    getter = dict

    if current_tts_mode == 'pyttsx3':
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        getter = vars
    elif current_tts_mode == 'coqui-tts':
        try:
            voices = utils.get_supported_coqui_voices()
        except:
            voices = []
    elif current_tts_mode == 'kokoro':
        # Hardcoded list from your specification
        voices = [{'name': n} for n in [
            'af_heart', 'af_bella', 'af_nicole', 'af_sarah', 'af_sky', 
            'am_adam', 'am_michael', 'bf_emma', 'bf_isabella', 'bf_lily'
        ]]
    
    if info:
        print(f'\n--- {current_tts_mode.upper()} Voice Options ---')
        for i, v in enumerate(voices):
            props = getter(v)
            print(f"Index {i}: {props.get('name', 'Unknown')}")
    return voices

def render_caption(caption, samprate, model, caption_path):
    """The render_caption function generates an audio caption from text input
    and writes it as a wav file. If the sample rate of the model is not equal 
    to that passed from sonification.py, it resamples to the correct rate and
    re-writes the file. 
    
    If Kokoro is selcted, text from user input is converted with text-to-
    speech software from Kokoro - https://pypi.org/project/kokoro-tts/ .

    If Coqui-AI is selected, text from user input is converted with text-to-
    speech software from Coqui-AI - https://pypi.org/project/TTS/ . 
    You can view publicly available voice models with 'TTS.list_models()'

    If neither Kokoro nor Coqui-AI are installed but pyttsx3 (https://pypi.org/project/pyttsx3/)
    is installed, text from user input is converted offline using pyttsx3.

    Note:
    STRAUSS checks if Coqui-AI is available. If it is, ``ttsMode`` is set to
    ``coqui-ai``. If it is unavailable, STRAUSS checks whether pyttsx3 is 
    available. If it is, ``ttsMode`` is set to ``pyttsx3``.

    Args:
      caption (:obj:`str`): script to be spoken by the TTS voice
      samprate (:obj:`int`): samples per second
      model (:obj:`str` for Coqui-AI; :obj:`dict` for pyttsx3): for Coqui-AI: 
        valid name of TTS voice from the underlying TTS module; for pyttsx3:
        dictionary with keys of 'rate' (percent of speed, signed int16),
        'volume' (float from 0 to 1), and/or 'voice' (the voice 'id' that can
        be chosen from the list given by the TTS.list_models() function).
      caption_path (:obj:`str`): filepath for spoken caption output
    """
    
    if current_tts_mode == 'None':
        raise Exception("TTS not supported. Install with 'pip install strauss[AI-TTS]'")

    # Generate Audio Output
    if current_tts_mode == 'kokoro':
        # Process Kokoro generator into a single audio array
        kokoro_gen = pipeline(caption, voice=model, speed=1, split_pattern=r'\n+')
        all_audio = [audio for _, _, audio in kokoro_gen]
        final_audio = np.concatenate(all_audio)
        # Clip values to avoid distortion during conversion
        final_audio = np.clip(final_audio, -1.0, 1.0)
        # Convert to 16-bit PCM (standard WAV format)
        final_audio = (final_audio * 32767).astype(np.int16)
        wavfile.write(caption_path, 24000, final_audio)


    elif current_tts_mode == 'coqui-tts':
        from TTS.api import TTS
        with utils.Capturing():
            tts = TTS(str(model), progress_bar=False, gpu=False)
            tts.tts_to_file(text=caption, file_path=caption_path)

        # TODO: do this better with logging. We can filter TTS function output, e.g. alert to downloading models...
        print('Rendering caption (this can take a while if the caption is long, or if the TTS model needs downloading)...')

        # strip leading or trailing punctuation
        caption = caption.strip('.!?¿¡') 
      
        # capture stdout from the talkative TTS module
        with utils.Capturing() as output:
            # Load in the tts model
            tts = TTS(str(model), progress_bar=False, gpu=False)

    elif current_tts_mode == 'pyttsx3':
        import pyttsx3
        engine = pyttsx3.init()
        if isinstance(model, dict):
            for key in ['rate', 'volume', 'voice']:
                if key in model:
                    engine.setProperty(key, model[key])
        engine.save_to_file(caption, caption_path)
        engine.runAndWait()

    else:
       # initialise dummy TTS class to raise error.
       TTS()
          
    # Read the file back in to check the sample rate
    try:
        # Try to read in directly...
        rate_in, wavobj = wavfile.read(caption_path)
    except:
        # ...but pttsx3 TTS can produce audio files incompatable
        # with scipy - convert to standard WAV using ffmpeg
        cpre = caption_path.split('.')[0] + '_pre.wav'
        os.rename(caption_path, cpre)
        ff.input(cpre).output(caption_path).run(quiet=1)
        rate_in, wavobj = wavfile.read(caption_path)
        
    # If it doesn't match the required rate, resample and re-write
    if rate_in != samprate:
        new_wavobj = utils.resample(rate_in, samprate, wavobj)
        wavfile.write(caption_path, samprate, new_wavobj)    # Resample to target rate if necessary

