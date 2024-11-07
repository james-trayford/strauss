#!/usr/bin/env python
# coding: utf-8

# ### <u> Generate a sonification with an audio caption in `strauss` </u>
# Import the relevant modules:
# 
# ***Note***: you will need to have some form of python text-to-speech installed (`TTS` or `pyttsx3`) for these examples to work. See the error raised when trying to run the examples below for more info:


from strauss.sonification import Sonification
from strauss.sources import Events
from strauss import channels
from strauss.score import Score
from strauss.tts_caption import render_caption
import numpy as np
from strauss.generator import Sampler
import os
from pathlib import Path
import strauss

mode = strauss.tts_caption.ttsMode


# What text to speech do we have?
print(f"Available text-to-speech (TTS) is: {mode}")


# Generate a placeholder sonification (a short sequence of glockenspiel notes) that we may want to add a caption to:


# platform agnostic absolute path for samples...
strauss_dir = Path(strauss.__file__).parents[2]
sample_path = Path(strauss_dir, 'data','samples','glockenspiels')

# setup used in stars appearing example
chords = [['Db3','Gb3', 'Ab3', 'Eb4','F4']]
length = 6
system = 'stereo'
score =  Score(chords, length)

maplims =  {'time': ('0', '150'),
            'pitch' : ('0', '100'),
           'phi':('0','100'),
            'theta':('0','100')}

events = Events(maplims.keys())

data = {'pitch':np.arange(5),
        'time':np.arange(5),
       'phi': np.arange(5),
       'theta': np.arange(5)}


generator = Sampler(Path(sample_path))

events.fromdict(data)
events.apply_mapping_functions(map_lims=maplims)


# Now, lets look at the avaialble voices for our TTS engine:
from strauss.tts_caption import TTS
voices = TTS().list_models()


# Generate text-to-speech (TTS) for the caption, using the default choice of voice (`"Jenny"` for the `coqui-tts` module, OS default for `pyttsx3`)
caption_en = 'In the following audio, a glockenspiel is used to represent stars of varying colour.'

soni = Sonification(score, events, generator, system,
                    caption=caption_en)
soni.render()
soni.hear()

caption_en = 'In the following audio, a glockenspiel is used to represent stars of varying colour.'

if mode == 'coqui-tts':
    soni = Sonification(score, events, generator, system,
                        caption=caption_en,
                       ttsmodel=str(Path('tts_models', 'en', 'ljspeech', 'tacotron2-DDC')))
elif mode == 'pyttsx3':
    for v in voices[::-1]:
        if v.languages[0][:2] == 'en':
            break
    print(f"Selected voice: {v.name}")
    soni = Sonification(score, events, generator, system,
                        caption=caption_en,
                       ttsmodel={'voice':v.id,
                                 # we can also set a rate for pyttsx3 (int16)...
                                'rate': 217})

soni.render()
soni.hear()


# Other TTS models are available in several languages. We can demonstrate a German voice, for example

caption_de = "In der folgenden Tonspur wird ein Glockenspiel verwendet um Sterne mit unterschiedlichen Farben zu repräsentieren."

if mode == 'coqui-tts':
    soni = Sonification(score, events, generator, system,
                        caption=caption_de, 
                        ttsmodel=str(Path('tts_models', 'de', 'thorsten', 'vits')))
elif mode == 'pyttsx3':
    # find a German-language voice...
    for v in voices:
        if v.languages[0][:2] == 'de':
            break
    soni = Sonification(score, events, generator, system,
                        caption=caption_de,
                        ttsmodel={'voice':v.id})

soni.render()
soni.hear()


# **Note**: the AI-based `TTS` can behave unpredictably when using unrecognised characters or terms. Sometimes these will be mispronounced by the TTS, other times they could be skipped entirely. This can be circumvented by writing out the how symbols should be pronounced, or spelling phonetically to improve pronunciation:

symbol_examples_en = 'The Lyman-α resonance is 1216 Å. The Lyman alpha resonance is twelve hundred and sixteen angstroms. '

for v in voices[::-1]:
        if v.languages[0][:2] == 'en':
            break
                       
soni = Sonification(score, events, generator, system,
                    caption=symbol_examples_en, ttsmodel={'voice':v.id, 'rate': 217})

soni.render()
soni.hear()

