#!/usr/bin/env python
# coding: utf-8

# ### <u> Generate the "_Stars Appearing_" sonification used in the "_Audible Universe_" planetarium show </u>

# First, we import relevant modules:

import matplotlib.pyplot as plt
import ffmpeg as ff
import wavio as wav
from strauss.sonification import Sonification
from strauss.sources import Events
from strauss import channels
from strauss.score import Score
import numpy as np
from strauss.generator import Sampler
import IPython.display as ipd
import os
from pathlib import Path


# <u> __The Score:__ </u>
# 
# We set up the ***Score***; this is analagous to a musical score and controls what notes can be played over the course of the sonification. We can specify a chord sequence as a single string (`str`) where chord names are separated by a `|` character. The root octave of the chord may also be specified by adding `_X` where `X` is the octave number. <span style="color:gray">_(Note: for now, each chord occupies an equal lenth in the sonification, in the future chord change times can be directly specified and optionally related to events in the data)_</span>
# 
# We can directly specify the ___chord voicing___ as 
# a list of lists containing the notes from low to high in each chord.
# 
# Here, we  are directly specifying a single __`Db6/9`__ chord voicing. These notes will later be played by stars of different colours!


chords = [['Db3','Gb3', 'Ab3', 'Eb4','F4']]
length = "1m 30s"
score =  Score(chords, length)


# <u> __The Sources:__ </u>
# 
# Next, we import the data that will represent ___sources___ of sound. The data is the sky positions, brightness and colour of stars from the _Paranal_ observatory site in _Chile_. This data is contained in an `ascii` file (specified by the `datafile` variable) and organised where each __row__ is a star and each __column__ is a property of that star.
# 
# The idea here is that as night draws in we see the brightet stars first. As it gets darker, and our eyes adjust, we see more dim stars. We "***sonify***" this by having a note play when each star appears. The "***panning***" (a.k.a stereo imaging) of the note is controlled by the ***altitude*** and ***azimuth*** of the stars, as if we were facing south. The ***colour*** of the star contols the note within the chord we've chosen, where notes low to high (short to long wavelength) represent fixed-number bins in colour from blue to red (again, short to long wavelength). Finally, the volume of the note is also related to the brightness of the star (dimmer stars are quieter). This is chosen to give a relatively even volume throught the sonification, as dim stars are much more numerous than bright ones <span style="color:gray">_(Note: in the future we will have the option to scale volumes in this way automatically)_</span>
# 
# We speciify the sound preperty to star property mappimg as as three `dict` objects with keys representing each sound property we dapat in the sonification:
# - **`mapcols`**: entries are the data file columns used to map each property
# - **`mapvals`**: entries are function objects that manipulate each columns values to yield the linear mapping
# - **`maplims`**: entries are `tuple`s representing the (`low`,`high`) limits of each mapping.numerical values represent absolute limits (used here for the angles in degrees to correctly limit the `azimuth` and `polar` mappings to 360° (2π) and 180° (π) respectively. `str` values are taken to be percentiles from 0% to 100%. string values > 100% can also be used, where e.g. 104 is 4% larger than the 100th percentile value. This is used for the time here, so that the last sample doesnt trigger at exactly the end of the sonification, giving time for the sound to die away slowly.


datafile = Path("..", "data", "datasets", "stars_paranal.txt")
mapcols =  {'azimuth':1, 'polar':0, 'volume':2, 'time':2, 'pitch':3}

mapvals =  {'azimuth': lambda x : x,
            'polar': lambda x : 90.-x,
            'time': lambda x : x,
            'pitch' : lambda x: -x,
            'volume' : lambda x : (1+np.argsort(x).astype(float))**-0.2}

maplims =  {'azimuth': (0, 360),
            'polar': (0, 180), 
            'time': ('0%', '104%'),
            'pitch' : ('0%', '100%'),
            'volume' : ('0%', '100%')}

events = Events(mapcols.keys())
events.fromfile(datafile, mapcols)
events.apply_mapping_functions(mapvals, maplims)


# <u> __The Generator:__ </u>
# 
# The final element we need is a ***Generator*** that actually generates the audio given the ***Score*** and ***Sources***. Here, we use a ***Sampler***-type generator that plays an audio sample for each note. The samples and other parameters (not specified here) control the sound for each note. These can be specified in `dict` format note-by-note (keys are note name strings, entries are strings pointing to the `WAV` format audio sample to load) or just using a string that points to a sample directory (each sample filename in that directory ends with `_XX.wav` where `XX` is the note name) we use the example sample back in `./data/samples/glockenspiels` <span style="color:gray">_(Note: rendering can take a while with the long audio samples we use here, shorter samples can be used to render faster, such as those in `./data/samples/mallets`. This is also useful if you want to try different notes or chords, as only the 5 notes specified above are provided in the glockenspiel sample folder.)_</span>

sampler = Sampler(Path("..", "data", "samples", "glockenspiels"))
sampler.preset_details("default")


# <u> __The Sonification:__ </u>
# 
# We consolidate the three elements above in to a sonification object to generate the sound, specifying the audio setup (here `'stereo'` as opposed to `'mono'`, `'5.1'`, etc). <span style="color:gray">_(Note: you can generate the audio in any specified audio setup, but following cells assume stereo and only mono and stereo formats are supported by the jupyter audio player in the final cell)_</span>
# 
# We then `render` the sonification to generate the audio track (may take some time with the glockenspiel samples).

system = "stereo"

print("Generating 'Stars Appearing' sonification...")
soni = Sonification(score, events, sampler, system)
soni.render()


# Finally, let's visualise the waveform, and preview the audio in-notebook*!
# 
# if using a surround sound format (i.e > 2 channels) the preview is stereo, with the first two channels mapped left and right, due to the limitations of the notebook audio player

soni.hear()

# Run `soni.save_combined('<directory/to/filename.wav>')` if you want to save the sonification to a file.
# soni.save_combined(Path('..', '..', 'rendered_stars_stereo.wav'),True)

