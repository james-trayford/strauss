"""The :obj:`tts_caption` submodule: tool for generating spoken captions

This uses text-to-speech via the the ``TTS`` module to allow captions
represented as strings to be converted to spoken audio to precede the
sonification.
"""

from scipy.io import wavfile
from scipy.interpolate import interp1d
import numpy as np
import strauss.utilities as utils
import re
import ffmpeg as ff
import os

try:
    from TTS.api import TTS
    ttsMode = 'coqui-TTS'
except (OSError, ModuleNotFoundError) as sderr:
    # print('Coqui TTS not found. Trying to import pyttsx3...')
    try:
      import pyttsx3
      ttsMode = 'pyttsx3'
      class TTS:
          def __init__(*args, **kwargs):
              pass
          def list_models(self):
              getVoices(True)
      # print('pyttsx3 has been successfully imported.')
    except (OSError, ModuleNotFoundError) as sderr:
      ttsMode = 'None'
      # print('No supported text-to-speech packages have been found.')
      def TTS(*args, **kwargs):
          raise TTSIsNotSupported("strauss has not been installed with text-to-speech support. \n"
                "This is not installed by default, due to some specific module requirements of the TTS module.\n"
                "Reinstalling strauss with 'pip install strauss[TTS]' will give you access to this function\n"
                "If you run into issues with the TTS package, you can also install pyttsx3 with the command\n" 
                "'pip install pyttsx3'.")
    
class TTSIsNotSupported(Exception):
    pass

def get_ttsMode():
   return ttsMode

def getVoices(info=False):
  '''Get available voices for text-to-speech.

  When info=True, this prints out information
  for each voice option.

    Args:
      info (:obj:`bool`): Print out voice information when True, 
      by default False
      voices (:obj:`list`): List of ``pyttsx3.voice.Voice`` objects
  '''
  if ttsMode == 'pyttsx3':
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    if info==True:
        print('Text-to-speech voice options')
        for ind in range(len(voices)):
            voiceProps = vars(voices[ind])
            print('\nVoice index:', ind)
            for key in voiceProps.keys():
                print('{}: {}'.format(key, voiceProps[key]))
    else:
        pass
    return voices

def render_caption(caption, samprate, model, caption_path):
    '''The render_caption function generates an audio caption from text input
    and writes it as a wav file. If the sample rate of the model is not equal 
    to that passed from sonification.py, it resamples to the correct rate and
    re-writes the file. 
    
    If Coqui-AI is installed, text from user input is converted with text-to-
    speech software from Coqui-AI - https://pypi.org/project/TTS/ . 
    You can view publicly available voice models with 'TTS.list_models()'

    If Coqui-AI is not installed but pyttsx3 (https://pypi.org/project/pyttsx3/)
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
        dictionary with keys of 'rate' (percent of speed), 'volume' (float from 0 to 1), 
        and/or 'voices' (string identifying the chosen voice)
      caption_path (:obj:`str`): filepath for spoken caption output
    '''

    if ttsMode == 'coqui-TTS':
      # TODO: do this better with logging. We can filter TTS function output, e.g. alert to downloading models...
      print('Rendering caption (this can take a while if the caption is long, or if the TTS model needs downloading)...')
      
      # capture stdout from the talkative TTS module
      with utils.Capturing() as output:
          # Load in the tts model
          tts = TTS(model, progress_bar=False, gpu=False)

          # render to speech, and write as a wav file (allow )
          tts.tts_to_file(text=caption, file_path=caption_path)
    
    elif ttsMode == 'pyttsx3':

      # Setup voice model for pyttsx3
      engine = pyttsx3.init() # initialize object

      # check what model info was set; if none were
      # specified, use defaults
      for key in ['rate','volume','voices']:
          if key in model.keys():
              engine.setProperty(key, model[key])
          else:
              pass

      engine.save_to_file(caption, caption_path)

      try:
          # TODO: explore why NSS triggers error hear without catching
          engine.runAndWait()
      except Exception as e:
          print(e)
          if engine._inLoop:
              engine.endLoop()
          pass

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
        wavfile.write(caption_path, samprate, new_wavobj)


