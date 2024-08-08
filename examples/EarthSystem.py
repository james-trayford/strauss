#!/usr/bin/env python
# coding: utf-8

# ### <u> Generate the Earth rotation sound for the Planetarium Show</u>
# **First, import relevant modules:**

import matplotlib.pyplot as plt
import ffmpeg as ff
import wavio as wav
from strauss.sonification import Sonification
from strauss.sources import Objects
from strauss import channels
from strauss.score import Score
import numpy as np
from strauss.generator import Synthesizer
import IPython.display as ipd
import os
from scipy.interpolate import interp1d
from pathlib import Path


# **Then, import the land fraction data**
# 
# The land fraction as a function of longitude is converted to a water fraction (i.e. $1-f_{\rm water}$), and mapped of three rotation cycles to control the LP filter cutoff. This is normalised to a range within the [0,1] range, chosen to sound good.

print("\n Sonifying Earth's rotation, using the land covering fraction with longitude...")

datafile = Path("..", "data", "datasets", "landfrac.txt")
data = np.genfromtxt(datafile)

longitude = data[:,0]
waterfrac = 1-data[:,1]

startlong = 180-(96 + 15./60 + 2.2/3600)
# we travel backwards in longitude per the earth's rotation
longgrid = (np.linspace(startlong,720+startlong,2599)%360 - 180.)[::-1] 
wfrac = interp1d(longitude, waterfrac)

wfracgrid = wfrac(longgrid)*0.75 + 0.15
timegrid = np.linspace(0,1,wfracgrid.size)

# uncomment to show plot...

# plt.plot(timegrid, wfracgrid)
# plt.ylabel("Normalised Water Fraction")
# plt.xlabel(r"${\rm Rotation}\; [6\pi]$")
# plt.show()

# and set up the synthesiser

# chord representing the earth (a Gbsus7 chord)
notes = [['Gb3', 'Db4', 'E4', 'B4']]

# specify audio system (e.g. mono, stereo, 5.1, ...)
system = "stereo"

length = 60.

# set up synth and turn on LP filter
generator = Synthesizer()
generator.modify_preset({'filter':'on'}) 


# Map the data and render sonification for the Earth's rotation...

score =  Score(notes, length)

# volume swell is directly ahead
data = {'cutoff':[wfracgrid]*4,
        'time_evo':[timegrid]*4,
        'pitch':list(range(4))}

# set up source
sources = Objects(data.keys())
sources.fromdict(data)
sources.apply_mapping_functions()

soni = Sonification(score, sources, generator, system)
soni.render()

# **Listen to and plot the waveforms from the sonification:**
soni.hear()

# **Combine and save sonification to a multi-channel wav** 
# NOTE: Change `"../../FILENAME.wav"` to your filepath of choice

# soni.save_combined(Path("..", "..", "earth.wav"), True)

