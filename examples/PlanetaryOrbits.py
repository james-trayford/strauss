#!/usr/bin/env python
# coding: utf-8

# ### <u> Generate the Planetary Orbit sonifications used in the "_Audible Universe_" planetarium show </u>

# **First, import relevant modules:**

import matplotlib.pyplot as plt
import ffmpeg as ff
import wavio as wav
from strauss.sonification import Sonification
from strauss.sources import Objects
from strauss import channels
from strauss.score import Score
import numpy as np
from strauss.generator import Sampler
import IPython.display as ipd
import os
from pathlib import Path


# **Collate the notes we are using to represent each planet and their orbital periods, as well as the length of each sonification**
# 
# Then, combine these into dictionaries so they can be easily indexed

print("\n Sonifying the stereo planet orbits example...")

# planet names
planets = ['Mercury',
          'Venus',
          'Earth',
          'Mars',
          'Jupiter',
          'Saturn',
          'Uranus',
          'Neptune']

# notes representing each planet
notes = [[['F6']],
         [['Bb5']],
         [['Gb5']],
         [['Db6']],
         [['Gb2']],
         [['Bb2']],
         [['Bb3']],
         [['F3']]]

# orbital period of each planet in days
periods = np.array([88,
                     224.7,
                     365.2,
                     687,
                     4331,
                     10747,
                     30589,
                     59800])

# sonification lengths for each planet used in planetarium show
# lengths = [126]*4 + [84]*4

# for the purposes of this, 30 second sonifications
lengths = [30]*8


# put these into dictionaries
chorddict = dict(zip(planets,notes))
lendict = dict(zip(planets, lengths))
perioddict = dict(zip(planets,periods*(lendict['Neptune']/(0.75 * periods[-1]))))

# Choose your planet...
print(f"\n Avaialable planets:")
for i in range(len(planets)):
    print(f"\t {i+1}. {planets[i]}")
planet_idx = input("Choose a planet's number (5): ")
if not planet_idx:
    planet_idx = 5-1
else:
    planet_idx = int(planet_idx)-1
print(f"\n Generating sonification for {planets[planet_idx]}...")

# **Specify the audio system to use** _(use `'stereo'` by default but for the planetarium `'5.1'` is used)_


# specify audio system (e.g. mono, stereo, 5.1, ...)
system = "stereo"


# **Now, set-up the sampler and the mapping functions and limits of mapped quantities**


# set up sampler
sampler = Sampler(Path("..", "data", "samples", "solar_system"))

# we want to loop the orchestral samples
sampler.modify_preset({'looping':'forward', # looping style
                       'loop_start': 7.0, # start of loop in seconds
                       'loop_end': 9.4}) # end of loop in seconds

# mapping functions and their limits
mapvals =  {'azimuth': lambda x : x,
            'polar': lambda x : x,
            'pitch' : lambda x: x,
            'volume' : lambda x : x,
           'time_evo' : lambda x : x}

maplims =  {'azimuth': (0, 360),
            'polar': (0, 180),
            'pitch' : (0, 1),
            'volume' : (0, 1),
           'time_evo' : (0,1)}


# **Specify which planet you want to sonify:**


# modify these for each planet
planet = planets[planet_idx]
panphase = 0


# **Render sonification for specified planet...**


# volume swell is directly ahead
volphase = panphase + 90

# setup score
score =  Score(chorddict[planet], lendict[planet])

# data dict
n = 10000
orbits_per_sonification = lendict[planet]/perioddict[planet]
orbital_azimuth = (np.linspace(0,orbits_per_sonification,n)%1)*360
data = {'azimuth':orbital_azimuth + panphase,
        'polar':np.ones(n)*90., # constant polar of 90 deg
        'pitch':1,     # constant pitch
        'volume':np.sin((orbital_azimuth + volphase) * np.pi/180.)*0.4+0.6,
        'time_evo': np.linspace(0,1,n)
        }

# set up source
sources = Objects(mapvals.keys())
sources.fromdict(data)
sources.apply_mapping_functions(mapvals, maplims)

soni = Sonification(score, sources, sampler, system)
soni.render()


# **Listen to and plot the waveforms from the sonification:**

soni.hear()

# **Combine and save sonification to a multi-channel wav** 
# 
# NOTE: Change `"../../FILENAME.wav"` to your filepath of choice. By default, the sound file is normalised to that of the highest amplitude sample, but can be set to a lower normalisation by setting the `master_volume` parameter to a value between `0.` and `1.`.

# soni.save(Path("..", "..", "FILENAME.wav"), master_volume=1.0)





