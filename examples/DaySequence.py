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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--animate", action="store_true",
                    help="create an animation of plot")
args = parser.parse_args()

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

def animate(raw_mapping, soni):
    '''Make frames for animating a plot showing changes in volume with time.
       Create a sequence to animate this with the sound overlaid, using a temporary directory for intermediate files.'''

    print("\n Creating animation frames. This may take a few minutes. The output, 'final.mp4' is saved to /figure_animations/DaySequence/")
    soni.score.length
    import warnings
    from pathlib import Path
    from strauss.animation import Animate
    import shutil
    import tempfile

    here = Path.cwd()

    # Define the final target directory
    target_dir_name = Path("figure_animations") / "DaySequence"
    
    # Use a temporary directory for all intermediate files
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        print(f"Using temporary directory: {temp_dir}")

        pipe = Animate(temp_dir)
        pipe.register('volume', sonification=soni, pre_caption=f'This video shows how volume is mapped to daylight.', post_caption='Thank you for listening!', stype='animation')
        xp = events2.raw_mapping['time_evo'][0]
        yp = events2.raw_mapping['volume'][0]
        nframe = int(soni.score.length*int(pipe.pars['fps']))
        xf = np.linspace(xp[0], xp[-1], nframe)
        yf = np.interp(xf, xp, yp)
        xp, yp = xf, yf
        for i in range(xp.size)[::1]:
            fig, ax1 = plt.subplots()
            plt.title("Day Sequence")
            ax1.set_xlabel('Time [s]') 
    	    ax2 = ax1.twinx() 
    	    ax1.plot(xp, yp)
    	    ax1.set_ylabel('Data')
    	    ax1.tick_params(axis ='y')
    	    ax1.axvline(xp[i], ls ='--', c='C0',lw=1.5, alpha=0.55)
    	    ax1.axhline(yp[i], ls ='--', c='C0',lw=1.5, alpha=0.55)
    	    ax2.set_ylabel('Volume')
    	    ax2.tick_params(axis ='y')
    	    plt.savefig(pipe.frames["volume"].parent / f'frame_{i:05d}.png', dpi=120)
    	    plt.close()
        print(f"Volume frames created in temporary directory!")
        pipe.render()
        temp_final_mp4 = temp_dir / "final.mp4" 

        target_dir_name.mkdir(parents=True, exist_ok=True)
        final_target_path = target_dir_name / temp_final_mp4.name
        
        if temp_final_mp4.exists():
            shutil.copy(temp_final_mp4, final_target_path)
            print(f"\nFinal animation copied to: {final_target_path}")
        else:
            warnings.warn(f"Could not find {temp_final_mp4} after rendering.")

if args.animate:
    animate(events2.raw_mapping, soni)
