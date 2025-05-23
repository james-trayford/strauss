#!/usr/bin/env python
# coding: utf-8

# ## <u> Demonstrate `Spectraliser` Generator type to represent Data:</u>

# **First, import relevant modules:**

# In[16]:

import matplotlib.pyplot as plt
from strauss.sonification import Sonification
from strauss.sources import Events, Objects
from strauss import channels
from strauss.score import Score
from strauss.generator import Spectralizer
import IPython.display as ipd
import os
from scipy.interpolate import interp1d
import numpy as np
from pathlib import Path


# In other examples we use a 'parameter mapping' approach for one-dimensional data series, where we map _y_ as a function of _x_ using the change in some expressive property of sound (e.g. `pitch_shift`) as a function of time.
# 
# We consider a direct spectralisation approach where the sopund is generated by treating th 1D data as a sound spectrum! This uses a direct inverse Fourier transform.This seems relatively intuitive for spectral data, particular those with spectral features similar to what we can identify in sound.
# 
# We will use Planetary Nebulae (PNe) data to demonstrate this, objects that are dominated by strong emission lines...

# **First, let's grab some data...**

# In[74]:


spectral_data1 = np.genfromtxt(Path('..', 'data', 'datasets', 'NGC1535.csv'), delimiter=',')
wlen1 = spectral_data1[:,0]
fluxdens1 = spectral_data1[:,1]

spectral_data2 = np.genfromtxt(Path('..', 'data', 'datasets', 'NGC6302.csv'), delimiter=',')
wlen2 = spectral_data2[:,0]
fluxdens2 = spectral_data2[:,1]

# spectrum needs to be provided to the Spectraliser in frequency order (i.e. low to high), 
# so we ensure it is sorted that way... 
spec1 = fluxdens1[np.argsort(1/wlen1)]
spec2 = fluxdens2[np.argsort(1/wlen2)]
wlen1 = wlen1[np.argsort(1/wlen1)]
wlen2 = wlen2[np.argsort(1/wlen2)]

# plot the spectra vs wavlength
plt.plot(wlen1, spec1/spec1.max(), label= 'NGC1535')
plt.plot(wlen2, spec2/spec2.max(), alpha=0.6, label= 'NGC6302')
plt.xlabel('Wavelength [Angstrom]')
plt.ylabel('Flux')
plt.legend(frameon=False)
# plt.show()
plt.close()

# plot the spectra vs frequency
plt.plot(1e-12*3e8/(wlen1*1e-10), spec1/spec1.max(),label= 'NGC1535')
plt.plot(1e-12*3e8/(wlen2*1e-10), spec2/spec2.max(), alpha=0.6,label= 'NGC1535')
plt.xlabel('Frequency [THz]')
plt.ylabel('Flux')
plt.legend(frameon=False)
#plt.show()
plt.close()

# **Set up some universal sonification parameters and classes for the examples below**
# 
# For all examples we use the `Synthesizer` generator to create a 30 second, mono sonification.

# In[75]:


# specify audio system (e.g. mono, stereo, 5.1, ...)
system = "stereo"

# length of the sonification in s
length = 10.


# ### <u>Example 1</u> &nbsp; **Comparing the NGC 1535 & NGC 6302 Spectra**

# Lets compare the _Spectraliser_ representations of these two spectra

# In[77]:


notes = [["A2"]]

score =  Score(notes, length)

spectra = [spec1, spec2]
wlens = [wlen1, wlen2]
names = ['NGC 1535', 'NGC 6302']

for i in range(2):

    #set up spectralizer generator
    generator = Spectralizer()

    # Lets pick the mapping frequency range for the spectrum...
    generator.modify_preset({'min_freq':100, 'max_freq':1000})

    s = np.zeros(spec1.size)
    s[-1] = 1
    # set up spectrum and choose some envelope parameters for fade-in and fade-out
    data = {'spectrum':[spectra[i]], 'pitch':[1],
            'volume_envelope/D':[0.9], 
            'volume_envelope/S':[0.], 
            'volume_envelope/A':[0.05]}
    
    # again, use maximal range for the mapped parameters
    lims = {'spectrum': ('0%','100%')}
    
    # set up source
    sources = Events(data.keys())
    sources.fromdict(data)
    sources.apply_mapping_functions(map_lims=lims)
    
    # render and play sonification!
    soni = Sonification(score, sources, generator, system)
    soni.render()
    print(f"Spectralising {names[i]}...")
    plt.plot(1e-12*3e8/(wlens[i]*1e-10), spectra[i]/spectra[i].max(), alpha=0.7,label=names[i])
    soni.hear()
plt.xlabel('Frequency [THz]')
plt.ylabel('Flux')
plt.legend(frameon=False)
#plt.show()
plt.close()

# What differences do you notice about the sounds? Can you here the presence/absence of spectral lines, and their relative pitches?

# ### <u>Example 2</u> &nbsp; **Evolving Spectra and Image Sonification**

# We could also perform a `Object` type sonification with an evolving Spectrum. 
# 
# An evolving spectrum can be represented as a 2D array, similar to a regular image. Using this similarity, the `Spectraliser` provides a neat way to sonify images!
# 
# Here we sonify the `strauss` logo, lets grab it...

# In[101]:


image = plt.imread(Path('..', 'misc', 'strauss_logo.png'))
image = image[:,:,:-1].sum(axis=-1)
image_inv = 1-image
plt.imshow(image_inv, cmap='gray_r')
plt.axis('off')
#plt.show()
plt.close()

# in `strauss` each row represents a spectrum, ordered from first to last.
# 
# Convention to represent the image is to evolve from left to right, with higher features in the _y_-axis sounding higher pitch. Due to image formatting conventions being different, we need transpse and flip the image array to get the right format for `strauss`

# In[129]:


spec_stack = image_inv[::-1].T
plt.imshow(spec_stack,  cmap='gray_r')
plt.axis('off')
#plt.show()
plt.close()

# Now let's _Spectralise_!  

# In[130]:


#show the image again...
plt.imshow(image_inv, cmap='gray_r')
plt.axis('off')
#plt.show()
plt.close()

score =  Score(notes, 15)

#set up spectralizer generator
generator = Spectralizer()

# Lets pick the mapping frequency range for the spectrum...
generator.modify_preset({'min_freq':20, 'max_freq':10000})

# set up spectrum
data = {'spectrum':[spec_stack], 'pitch':[1]}

# again, use maximal range for the mapped parameters
lims = {'spectrum': ('0%','100%')}

# set up source
sources = Events(data.keys())
sources.fromdict(data)
sources.apply_mapping_functions(map_lims=lims)

# render and play sonification!
soni = Sonification(score, sources, generator, system)
soni.render()
print(f"Spectralising Image...")
soni.hear()

