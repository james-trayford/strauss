"""The :obj:`tts_caption` submodule: tool for generating spoken captions

This uses text-to-speech (TTS) to allow captions represented as strings
to be converted to spoken audio to precede the sonification.

Two TTS engines are supported, and the best available is picked at
import time, in order of preference:

1. ``kokoro`` - neural TTS, good quality, runs offline once the model is
   downloaded (``pip install strauss[speech]``)
2. ``pyttsx3`` - system TTS, no model download but platform dependent

The engine can be chosen explicitly with :func:`set_engine`, and the
voices it offers listed with :func:`getVoices`.

Note:
  The ``kokoro`` voices are named by language and gender, e.g.
  ``'af_heart'`` (American female) or ``'bm_george'`` (British male).
  They are described, with sample audio and quality grades, at
  https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md -
  :data:`_kokoro_voices` lists the English ones strauss offers.
"""

from scipy.io import wavfile
import numpy as np
import strauss.utilities as utils
import importlib.util
import os
import warnings

# ordered by preference
ENGINE_PREFERENCE = ('kokoro', 'pyttsx3')

# module name that must be importable for each engine
_engine_modules = {'kokoro': 'kokoro',
                   'pyttsx3': 'pyttsx3'}

# default voice for each engine, given to ``Sonification(ttsmodel=...)``
_engine_default_voices = {'kokoro': 'bf_emma',
                          'pyttsx3': {}} # i.e. system default TTS (if exists)

# Kokoro ships voices on the huggingface hub rather than exposing a list, so
# name the bundled English ones here. The first letter gives the language
# ('a'merican or 'b'ritish English), the second the sex.
# the English kokoro voices, named <language><gender>_<name>. The full
# list, with sample audio and quality grades, is at
# https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
_kokoro_voices = [
    {'name': 'af_heart', 'languages': ['en_US']},
    {'name': 'af_bella', 'languages': ['en_US']},
    {'name': 'af_nicole', 'languages': ['en_US']},
    {'name': 'af_sarah', 'languages': ['en_US']},
    {'name': 'af_sky', 'languages': ['en_US']},
    {'name': 'am_adam', 'languages': ['en_US']},
    {'name': 'am_michael', 'languages': ['en_US']},
    {'name': 'bf_emma', 'languages': ['en_GB']},
    {'name': 'bf_isabella', 'languages': ['en_GB']},
    {'name': 'bf_lily', 'languages': ['en_GB']},
    {'name': 'bm_george', 'languages': ['en_GB']},
    {'name': 'bm_lewis', 'languages': ['en_GB']},
]

ttsMode = 'None'
default_tts_voice = None

# kokoro pipelines are expensive to build, so keep one per language
_kokoro_pipelines = {}

class TTSIsNotSupported(Exception):
    pass

_no_tts_message = (
    "strauss has not been installed with text-to-speech support. \n"
    "This is not installed by default, due to some specific module requirements of the TTS modules.\n"
    "Reinstalling strauss with 'pip install strauss[speech]' will give you access to this function\n"
    "(installing the 'kokoro' engine). You can also install\n"
    "pyttsx3 to use your system's text-to-speech voices instead. Currently the most compatible\n"
    "version is not published on PyPI, but you can install from the test repo with \n"
    "'pip install --no-cache-dir --extra-index-url https://test.pypi.org/simple/ pyttsx3==2.99'")

def _available(engine):
    """Whether an engine's module is installed (without importing it)."""
    return importlib.util.find_spec(_engine_modules[engine]) is not None

def set_engine(engine):
    """Choose the text-to-speech engine, and update the default voice.

    Engines are only checked for here, and loaded when a caption is
    first rendered, so switching is cheap.

    Args:
      engine (:obj:`str`): one of ``'kokoro'`` or ``'pyttsx3'``

    Raises:
      TTSIsNotSupported: if the engine's module isn't installed.
    """
    global ttsMode, default_tts_voice
    if engine not in _engine_modules:
        raise ValueError(f"Unknown TTS engine '{engine}', choose from {list(_engine_modules)}")
    if not _available(engine):
        raise TTSIsNotSupported(f"TTS engine '{engine}' requested but its module "
                                f"'{_engine_modules[engine]}' is not installed.\n" + _no_tts_message)
    ttsMode = engine
    default_tts_voice = _engine_default_voices[engine]

def _init_engine():
    """Pick the most preferred engine that is installed."""
    for engine in ENGINE_PREFERENCE:
        if _available(engine):
            set_engine(engine)
            if engine == 'pyttsx3':
                warnings.warn("Neural TTS module (kokoro) not found, using pyttsx3 instead. Note this is platform \n"
                              "dependent and can be problematic for linux-based systems (using the espeak engine)")
            return

_init_engine()

def get_ttsMode():
   return ttsMode

def get_default_voice():
    """The default voice for the current engine, as passed to ``Sonification(ttsmodel=...)``."""
    return default_tts_voice

def getVoices(info=False):
  '''Get available voices for the current text-to-speech engine.

  When info=True, this prints out information
  for each voice option.

    Args:
      info (:obj:`bool`): Print out voice information when True,
      by default False
      voices (:obj:`list`): List of ``pyttsx3.voice.Voice`` objects
      or ``dict`` objects.

  Note:
    For ``kokoro``, these are the English voices, named
    ``<language><gender>_<name>`` - ``'af_heart'`` is an American
    female voice, ``'bm_george'`` a British male one. They are
    described, with sample audio and quality grades, at
    https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

  '''
  if ttsMode == 'pyttsx3':
      import pyttsx3
      engine = pyttsx3.init()
      voices = engine.getProperty('voices')
      getter = vars
  elif ttsMode == 'kokoro':
      voices = _kokoro_voices
      getter = dict
  else:
      getter = dict
      voices = [{"voices": "None"}]
  if info==True:
      print(f'Text-to-speech voice options ({ttsMode})')
      for ind in range(len(voices)):
          voiceProps = getter(voices[ind])
          print('\nVoice index:', ind)
          for key in voiceProps.keys():
              print('{}: {}'.format(key, voiceProps[key]))
  return voices

def _kokoro_pipeline(voice):
    """Get the kokoro pipeline for a voice's language, building it on first use."""
    from kokoro import KPipeline
    # the voice name is prefixed by its language code
    lang_code = str(voice)[0]
    if lang_code not in _kokoro_pipelines:
        with utils.Capturing():
            _kokoro_pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id='hexgrad/Kokoro-82M')
    return _kokoro_pipelines[lang_code]

def render_caption(caption, samprate, model, caption_path):
    '''The render_caption function generates an audio caption from text input
    and writes it as a wav file. If the sample rate of the model is not equal
    to that passed from sonification.py, it resamples to the correct rate and
    re-writes the file.

    If Kokoro is selected, text from user input is converted with text-to-
    speech software from Kokoro - https://pypi.org/project/kokoro/ .

    If pyttsx3 (https://pypi.org/project/pyttsx3/) is selected, text from
    user input is converted offline using the system voices.

    Note:
    STRAUSS checks which engines are available at import, setting ``ttsMode``
    to the first found of ``kokoro`` or ``pyttsx3``. Choose
    explicitly with ``set_engine``.

    Args:
      caption (:obj:`str`): script to be spoken by the TTS voice
      samprate (:obj:`int`): samples per second
      model (:obj:`str` for Kokoro; :obj:`dict` for pyttsx3):
        for Kokoro: a voice name (see ``getVoices``); for pyttsx3:
        dictionary with keys of 'rate' (percent of speed, signed int16),
        'volume' (float from 0 to 1), and/or 'voice' (the voice 'id' that can
        be chosen from the list given by the ``getVoices`` function).
        ``None`` uses the current engine's default.
      caption_path (:obj:`str`): filepath for spoken caption output
    '''

    if model is None:
        model = default_tts_voice

    # TODO: allow uniform indexing and/or language querying approaches for more consistency between tts modes...
    if ttsMode == 'kokoro':

      print('Rendering caption (this can take a while if the caption is long, or if the TTS model needs downloading)...')
      pipeline = _kokoro_pipeline(model)
      # the pipeline yields one chunk of 24 kHz audio per line of text
      chunks = [audio for _, _, audio in pipeline(caption, voice=str(model), split_pattern=r'\n+')]
      audio = np.clip(np.concatenate(chunks), -1., 1.)
      wavfile.write(caption_path, 24000, (audio * 32767).astype(np.int16))

    elif ttsMode == 'pyttsx3':
      import pyttsx3

      # Setup voice model for pyttsx3
      engine = pyttsx3.init() # initialize object

      # check what model info was set; if none were
      # specified, use defaults
      for key in ['rate','volume','voice']:
          if key in model.keys():
              engine.setProperty(key, model[key])
          else:
              pass

      engine.save_to_file(caption, caption_path, name='caption')
      # note the current PyPI release ()
      engine.runAndWait()

    else:
       raise TTSIsNotSupported(_no_tts_message)

    # Read the file back in to check the sample rate
    try:
        # Try to read in directly...
        rate_in, wavobj = wavfile.read(caption_path)
    except:
        # ...but pttsx3 TTS can produce audio files incompatable
        # with scipy - convert to standard WAV using ffmpeg
        import ffmpeg as ff
        cpre = caption_path.split('.')[0] + '_pre.wav'
        os.rename(caption_path, cpre)
        ff.input(cpre).output(caption_path).run(quiet=1)
        rate_in, wavobj = wavfile.read(caption_path)

    # If it doesn't match the required rate, resample and re-write
    if rate_in != samprate:
        new_wavobj = utils.resample(rate_in, samprate, wavobj)
        wavfile.write(caption_path, samprate, new_wavobj)
