#!/usr/bin/env python
# coding: utf-8

# ### <u> Generate a sonification with an audio caption in `strauss` </u>
# Import the relevant modules:

from strauss.sonification import Sonification
from strauss.sources import Events
from strauss import channels
from strauss.score import Score
from strauss.tts_caption import render_caption
import numpy as np
from strauss.generator import Sampler
import os
import pprint

# Generate a placeholder sonification (a short sequence of glockenspiel notes) that we may want to add a caption to:

# platform agnostic absolute path for samples...
import strauss
sample_path = '/'.join(strauss.__file__.split('/')[:-3] +['data','samples','glockenspiels'])

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


generator = Sampler(sample_path)

events.fromdict(data)
events.apply_mapping_functions(map_lims=maplims)


# Generate text-to-speech (TTS) for the caption, using the default choice of voice (`"Jenny"` from the `TTS` module)

caption_en = 'In the following audio, a glockenspiel is used to represent stars of varying colour.'

print("Example of a caption using the default voice...")

# render at default 48 kHz rate
soni = Sonification(score, events, generator, system,
                    caption=caption_en)
soni.render()
soni.hear()


caption_en = 'In the following audio, a glockenspiel is used to represent stars of varying colour.'

print("Example of a caption using an alternative voice...")

soni = Sonification(score, events, generator, system,
                    caption=caption_en,
                   ttsmodel='tts_models/en/ljspeech/tacotron2-DDC')
soni.render()
soni.hear()


# Other TTS models are available in several languages. We can demonstrate a German voice, for example

caption_de = "In der folgenden Tonspur wird ein Glockenspiel verwendet um Sterne mit unterschiedlichen Farben zu repräsentieren."

print("Example of a caption in a different language (German), selecting a voice supportingh that language ('Thorsten')...")

soni = Sonification(score, events, generator, system,
                    caption=caption_de, 
                    ttsmodel="tts_models/de/thorsten/vits")
soni.render()
soni.hear()


# **Note**: the AI-based `TTS` can behave strangely when using unrecognised characters or terms. Sometimes these will be mispronounced by the TTS, other times they could be skipped entirely. This can be circumvented by writing out the how symbols should be pronounced, or spelling phonetically to improve pronunciation:

symbol_examples_en = 'The Lyman-α resonance is 1216 Å. The Lyman alpha resonance is twelve hundred and sixteen angstroms. '

print("Example of mispronunciation of terms or symbols...")

soni = Sonification(score, events, generator, system,
                    caption=symbol_examples_en+caption_en)
soni.render()
soni.hear()


# Captions can be used to provide context to sonifications, explaining what to listen for.
# 
# We can list available models for the TTS module (including `Jenny` the default `strauss` voice):

print("Print available voice models...")
from strauss.tts_caption import TTS
pprint.pprint(TTS().list_models().list_tts_models())
