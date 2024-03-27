#!/usr/bin/env python
# coding: utf-8

# Demonstrate some generic techniques for sonifying 1D data
# First, import relevant modules:

import matplotlib.pyplot as plt
from strauss.sonification import Sonification
from strauss.sources import Objects
from strauss import channels
from strauss.score import Score
from strauss.generator import Synthesizer
import IPython.display as ipd
import os
from scipy.interpolate import interp1d
import numpy as np



# Now, we construct some mock data!
# We use seeded random numbers to generate a mock 1D data set with features and noise:

# seed the randoms...
np.random.seed(0)

# construct arrays of size N for x and y...
N = 300
x = np.linspace(0,1,N)
y = np.zeros(N)

# define a Gaussian function...
gauss = lambda x, m, s: np.exp(-(x-m)**2/s) 

# place some randomised gaussians...
for i in range(10):
    a,b,c = np.random.random(3)
    y += gauss(x, b, 1e-3*c) * a ** 3

# now add some noise and normalise
y += np.random.random(N) * y.mean()
y /= y.max()*1.2
y += 0.15


# uncomment block to display mock data...

# plt.plot(x,y)
# plt.ylabel('Some dependent Variable')
# plt.xlabel('Some independent Variable')
# plt.show()

# Set up some universal sonification parameters and classes for the examples below
# For all examples we use the `Synthesizer` generator to create a 30 second, mono sonification.

# specify audio system (e.g. mono, stereo, 5.1, ...)
system = "stereo"

# length of the sonification in s
length = 15.



# set up synth and turn on LP filter
generator = Synthesizer()
generator.load_preset('pitch_mapper')


# Example: Pitch Mappin
print("\nExample 1: Pitch Mapping...")

# uncomment to see preset details...
# generator.preset_details('pitch_mapper')

notes = [["A2"]]
score =  Score(notes, length)

data = {'pitch':1.,
        'time_evo':x,
        'azimuth':(x*0.5+0.25) % 1,
        'polar':0.5,
        'pitch_shift':y**0.7}

# set up source
sources = Objects(data.keys())
sources.fromdict(data)
sources.apply_mapping_functions()

soni = Sonification(score, sources, generator, system)
soni.render()

soni.hear()


# Example 2: Volume Mapping
print("Example 2: Volume Mapping...")

notes = [["A2"]]
score =  Score(notes, length)

data = {'pitch':1.,
        'time_evo':x,
        'azimuth':(x*0.5+0.25) % 1,
        'polar':0.5,
        'volume':y**0.7}

# set up source
sources = Objects(data.keys())
sources.fromdict(data)
sources.apply_mapping_functions()

soni = Sonification(score, sources, generator, system)
soni.render()

soni.hear()


# Example 3a: Filter Cutoff Mapping - Tonal
print("Example 3a: Filter Cutoff Mapping - Tonal...")

generator = Synthesizer()
generator.modify_preset({'filter':'on'})

notes = [["C2","G2","C3","G3"]]
score =  Score(notes, length)

data = {'pitch':[0,1,2,3],
        'time_evo':[x]*4,
        'azimuth':[(x*0.5+0.25) % 1]*4,
        'polar':[0.5]*4,
        'cutoff':[y**0.8]*4}

# set up source
sources = Objects(data.keys())
sources.fromdict(data)
sources.apply_mapping_functions()

soni = Sonification(score, sources, generator, system)
soni.render()
soni.hear()

# Example 3b: Filter Cutoff Mapping - Textural

print("Example 3b: Filter Cutoff Mapping - Textural...")
generator = Synthesizer()

generator.load_preset('windy')

# uncomment to see preset details...
# generator.preset_details('windy')

data = {'pitch':[0,1,2,3],
        'time_evo':[x],
        'azimuth':[(x*0.5+0.25) % 1],
        'polar':[0.5],
        'cutoff':[y**0.8]}
sources = Objects(data.keys())
sources.fromdict(data)
sources.apply_mapping_functions()

soni = Sonification(score, sources, generator, system)
soni.render()
soni.hear()

