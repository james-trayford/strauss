"""Sonify events with spoken phrases, using the `Speech` generator.

Each source is spoken as one of a list of phrases, chosen by the
`pitch` mapping. The phrases are rendered with text-to-speech up front
(and cached), then played by the sampler machinery, so the usual
generator parameters apply.
"""

from strauss.sonification import Sonification
from strauss.sources import Events
from strauss.score import Score
from strauss.generator import Speech
from strauss.tts_caption import get_ttsMode
import numpy as np
import sys

if get_ttsMode() == 'None':
    print("No text-to-speech engine found; install one with\n"
          "'pip install strauss[AI-TTS]' to run this example.")
    sys.exit()

# the phrases to speak, low to high in the pitch mapping
phrases = ['a red dwarf', 'a yellow dwarf', 'a red giant']

# ...naming each of them in the score, so sources are binned onto them
score = Score([phrases], '0m 10s')

# a star type and a time for each of six stars
stype = np.array([0., 2., 1., 0., 2., 1.])
time = np.linspace(0, 0.85, stype.size)

data = {'pitch': stype,
        'time': time}

sources = Events(data.keys())
sources.fromdict(data)
sources.apply_mapping_functions()

generator = Speech(phrases)
generator.info()

soni = Sonification(score, sources, generator, 'mono')
soni.render()

print('\nEvents:')
print(soni.event_table().to_string())
print(f"\nRendered {soni.out_channels['0'].values.size/soni.samprate:.1f}s of audio")

# to listen, run soni.hear() or write the audio out, e.g:
# soni.save('./Speech.wav')
