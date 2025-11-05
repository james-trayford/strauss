#!/usr/bin/env python
# coding: utf-8

# ### <u> Generate the Earth rotation sound for the Planetarium Show</u>
# **First, import relevant modules:**

import matplotlib.pyplot as plt
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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--animate", action="store_true",
                    help="create an animation of plot")
args = parser.parse_args()


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

#soni.save_combined(Path("..", "..", "earth.wav"), True)

def animate(timegrid, wfracgrid, soni):
    '''Make frames for animating a plot showing changes in water fraction.
       Create a sequence to animate this with the sound overlaid.'''

    print("\n Creating animation frames. This may take a few minutes.")
    import warnings
    from pathlib import Path
    import shutil
    import tempfile

    from strauss.animation import Animate
    here = Path.cwd()

    # Define the final target directory
    target_dir_name = Path("figure_animations") / "EarthSystem"
    
    # Use a temporary directory for all intermediate files
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        print(f"Using temporary directory: {temp_dir}")

        pipe = Animate(temp_dir)
        pipe.register('cutoff', sonification=soni, pre_caption=f'This video shows how timbre changes with water fraction, during two Earth rotations.', post_caption='Thank you for listening!', stype='animation')
        xp = timegrid
        yp = wfracgrid
        nframe = int(soni.score.length*int(pipe.pars['fps']))
        xf = np.linspace(xp[0], xp[-1], nframe)
        yf = np.interp(xf, xp, yp)
        xp, yp = xf, yf
        for i in range(xp.size)[::1]:
            fig, ax1 = plt.subplots()
            plt.title("Variation of Water Fraction of the Earth by Longitude")
            ax1.set_xlabel(r"${\rm Rotation}\; [6\pi]$") 
            ax2 = ax1.twinx() 
            ax1.plot(xp, yp)
            ax1.set_ylabel("Normalised Water Fraction")
            ax1.tick_params(axis ='y')
            ax1.axvline(xp[i], ls ='--', c='C0',lw=1.5, alpha=0.55)
            ax1.axhline(yp[i], ls ='--', c='C0',lw=1.5, alpha=0.55)
            ax2.set_ylim(min(wfracgrid)*4, max(wfracgrid)*4)
            ax2.set_ylabel('Cutoff')
            ax2.tick_params(axis ='y')
            plt.savefig(pipe.frames["cutoff"].parent / f"frame_{i:05d}.png", dpi=120)
            plt.close()
        print("Cutoff frames created!")
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
    animate(timegrid, wfracgrid, soni)
 
