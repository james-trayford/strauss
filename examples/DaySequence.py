#!/usr/bin/env python
# coding: utf-8

# ### <u> Generate the sunrise to sunset sonification used in the "_Audible Universe_" planetarium show </u>

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
import glob
import os
import copy
from pathlib import Path


print("\nSonifying the Sun's motion across the sky...")

# First we download the samples to the local data directory, if they haven't been already:
outdir = Path("..", "data", "samples", "day_sequence")

if list(Path(f"{outdir}").glob("*.wav")):
    print(f"Directory {outdir} already exists.")
else:
    print("Downloading files...")
    import urllib.request
    
    Path('..', 'data', 'samples', 'day_sequence').mkdir(parents=True, exist_ok=True)
    
    files = ("sun_A4.wav", "scatter_B4.wav")
    urls = ("https://drive.google.com/uc?export=download&id=15D7xHEKtKppTvzzwECIq_0UGhifdhrEy",
            "https://drive.google.com/uc?export=download&id=1bnhZ_kagtWMUkj1VtEE6vzQGfnYexQfL")
    for f, u in zip(files, urls):
        with urllib.request.urlopen(u) as response, Path(f"{outdir}", f"{f}").open(mode='wb') as out_file:
            print(f"\t getting {f}")
            data = response.read() # a `bytes` object
            out_file.write(data)
    print("Done.")
    

# **Specify the audio system to use** _(use `'stereo'` by default but for the planetarium `'5.1'` is used)_
# specify audio system (e.g. mono, stereo, 5.1, ...)
system = "stereo"

# **Now, set-up the sampler:**
# set up sampler
sampler = Sampler(str(outdir))
sampler.modify_preset({'filter':'on'}) # want filtering on for sun altitude effect

# **Set mapping limits of mapped quantities** (truncated relative to planetarium show example)
maplims =  {'azimuth': (0, 360),
            'polar': (0, 180),
            'pitch' : (0, 1),
            'cutoff' : (0, 1),
            'volume' : (0,1),
            'time_evo' : (0,75)}

# **Initialise the score:**
# setup score
score =  Score([['A4','B4']], 75)

# **Render sonification for specified planet...**
data = {'azimuth': np.array([90,90, 0, 330, 240,240]),
        'polar': np.array([45,45,0, 40, 0, 0]), # constant polar of 90 deg
        'pitch': 1,     # constant pitch
        'volume': np.ones(6),
        'cutoff': np.array([0.5, 0.5, 1, 0.444, 0.1, 0]),
        'time_evo': np.array([0, 33.5,45, 57.5, 72.5, 147])}

# set up source
events = Objects(maplims.keys())
events.fromdict(data)
events.apply_mapping_functions(map_lims=maplims)

print("Generating sonification of Sun alone...")
soni = Sonification(score, events, sampler, system)
soni.render()

# listen...
soni.hear()

# **Listen to and plot the waveforms from the sonification:**
print("Generating sonification with scattered light sound...")
data2 = {'azimuth': np.ones(8)*0,
        'polar': np.zeros(8), # constant polar of 90 deg
        'pitch': 1,     # constant pitch
        'volume': np.array([0.2,0.2,0.4,0.2,0.1,0.03, 0.01, 0.]),
        'cutoff': np.ones(8),
        'time_evo': np.array([0, 33.5,45, 57.5, 72.5, 90, 100, 147])}

# set up source
events2 = Objects(maplims.keys())
events2.fromdict(data2)
events2.apply_mapping_functions(map_lims=maplims)

sampler2 = copy.deepcopy(sampler)
sampler2.samples['A4'] = sampler2.samples['B4']

soni2 = Sonification(score, events2, sampler2, system)
soni2.out_channels = soni.out_channels
soni2.render()

# listen...
soni2.hear()

# **Combine and save sonification to a multi-channel wav** 
# 
# NOTE: Change `"../../FILENAME.wav"` to your filepath of choice. By default, the sound file is normalised to that of the highest amplitude sample, but can be set to a lower normalisation by setting the `master_volume` parameter to a value between `0.` and `1.`.

# soni2.save_combined(Path("..", "..", "day_sequence.wav"), True, master_volume=1.0)
