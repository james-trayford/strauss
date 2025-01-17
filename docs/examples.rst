
.. _examples:

Examples
^^^^^^^^

Here, we explain some of the example sonfications included in the :code:`examples/` directory of the Strauss repo. These are all in both *Python Notebook* (:code:`.ipynb`) and *Python script* (:code:`.py`) format so you can choose which format you prefer to use.

Audio Caption (:code:`AudioCaption.ipynb`)
******************************************
The *Audio Caption* example demonstrates how to add audio captions to a sonification, using a text-to-speech (TTS) module. The TTS module is not included in the standard Strauss installation, but it can be installed by using pip install strauss[TTS]. This example uses the Strauss :code:`Sampler` to play a short sequence of glockenspiel notes, then generates an audio caption using a standard TTS voice. The notebook allows the user to try different voices and languages from TTS.

There are examples of audio captioning with different voices in the following spectrogram videos

#. `English default voice <https://www.youtube.com/watch?v=jcdRNKnbzPs>`_
#. `English alternate voice <https://www.youtube.com/watch?v=fHrbVeTaNbM>`_
#. `Non-English (German) voice <https://www.youtube.com/watch?v=2qE5kk-iCYk>`_.
#. `Special characters, and writing phonetically to improve pronunciation <https://www.youtube.com/watch?v=36J2jYy33DI>`_.

The code is available in a `Google Colab notebook <https://githubtocolab.com/james-trayford/strauss/blob/main/examples/colab/AudioCaption.ipynb>`_.

Day Sequence (:code:`DaySequence.ipynb`)
****************************************
The *Day Sequence* sonification generates the sunrise to sunset sonification used in the `"Audio Universe: Tour of the Solar System" <https://www.audiouniverse.org/education/shows/tour-of-the-solar-system>`_, an immersive planetarium show designed with sonifications so it can be enjoyed and understood irrespective of level of vision. Samples are downloaded from a Google drive to a local directory, and are played using the Strauss :code:`Sampler`. This spatialisation can be mapped to any audio setup, and a :code:`5.1` system was used for the planetarium, but for the example we use :code:`stereo`.

Below is an example of this sonification

#. `Dawn to dusk, sun sound isolated <https://www.youtube.com/watch?v=x2LBs10H5Mc>`_.

The code is available in a `Google Colab notebook <https://githubtocolab.com/james-trayford/strauss/blob/main/examples/colab/DaySequence.ipynb>`_.


Earth System (:code:`EarthSystem.ipynb`)
****************************************
The *Earth System* sonification represents the ratio of ocean to land along lines of longitude as the Earth spins through three rotation cycles. We use the Strauss :code:`Synthesiser` to generate chords. A low pass filter is employed to generate a brighter sound to represent a high water fraction and a duller sound for high land fraction. 

A video of this sequence is available starting at 2:17 in `this video <https://www.youtube.com/watch?v=h1muFAEMmOs&t=137s>`_.

The code is available in a `Google Colab notebook <https://githubtocolab.com/james-trayford/strauss/blob/main/examples/colab/EarthSystem.ipynb>`_.


Light Curve Soundfonts (:code:`LightCurveSoundfonts.ipynb`)
************************************************************
The *Light Curve Soundfonts* example demonstrates how to use imported soundfont files to use virtual musical instruments in a sonification. Soundfont files are widely available online. We download flute and guitar sounds from `Soundfonts 4 U <https://sites.google.com/site/soundfonts4u/>`_. We load the sounds into the Strauss :code:`Sampler`, and select a preset. The soundfonts are used to sonify a light curve for the variable star 55 Cancri, creating the audio equivalent of a scatter plot. The note length and volume envelope can be adjusted to improve the articulation of individual data points. To demonstrate an alternative approach, we use an :code: `Object` source type, where we evolve a sound over time to represent the data. This is analogous to a line graph, representing a continuous data series. We use a held chord, changing the cutoff frequency of the low-pass filter to create a "brighter" timbre when the star is brighter and a "duller" sound when the star is darker.

Two examples using different approaches demonstrated in the notebook can be heard in the videos below

#. `Stellar light curve mapped to pitch on a melodic scale using a flute for an Event-type Source <https://www.youtube.com/watch?v=myYYbFT2JD0>`_
#. `Stellar light curve to cutoff frequency for a held  Electric guitar chord using an Object-type Source <https://www.youtube.com/watch?v=5tDeCN-xCgs>`_.

The code is available in a `Google Colab notebook <https://githubtocolab.com/james-trayford/strauss/blob/main/examples/colab/LightCurveSoundfonts.ipynb>`_.

Planetary Orbits (:code:`PlanetaryOrbits.ipynb`)
************************************************
The *Planetary Orbits* example generates sonifications used in the "Audio Universe: Tour of the Solar System" planetarium show. Each planet in the Solar System is assigned a unique note, with the smallest planets having the highest notes and the largest planets having the lowest notes. The sonification length for each is set according to the planet's orbital period, and the volume is varied by orbital azimuth. The audio system is set as 'stereo' by default but for the planetarium '5.1' is used. This creates the effect of the planets moving in orbits at relative speeds to represent the real relative motions.

Two video/audio examples are available below

#. `The eight planets orbiting the Sun (with graphics) <https://www.youtube.com/watch?v=WI-WPvXeAgk>`_.
#. `Jupiter's orbit <https://www.youtube.com/watch?v=sWnH56i8mJEk>`_.

The code is available in a `Google Colab notebook <https://githubtocolab.com/james-trayford/strauss/blob/main/examples/colab/PlanetaryOrbits.ipynb>`_.

Sonifying Data 1D (:code:`SonifyingData1D.ipynb`)
*************************************************
The *Sonifying Data 1D* example demonstrates some generic techniques for sonifying one-dimensional data series. We construct some mock data with features and noise. For all examples we use the Strauss :code:`Synthesiser` to create a 30 second, mono sonification. We demonstrate a variety of ways to map y as a function of x, using the change in some expressive property of sound (e.g. pitch_shift, volume and filter-cutoff) as a function of time.

The following spectrogram videos illustrate examples of 1D data series sonifications, mapping the data to different sound parameters.

#. `Mapping Pitch <https://www.youtube.com/watch?v=DQUbSYP-Fhw>`_
#. `Mapping Volume <https://www.youtube.com/watch?v=EgfA6M6MoEo>`_
#. `Mapping Filter Cutoff (tonal) <https://www.youtube.com/watch?v=W_tn3kvgcQA>`_
#. `Mapping Filter Cutoff (textural) <https://www.youtube.com/watch?v=t0wGb_IrAQU>`_

The code is available in a `Google Colab notebook <https://githubtocolab.com/james-trayford/strauss/blob/main/examples/colab/SonifyingData1D.ipynb>`_.

Spectral Data (:code:`SpectralData.ipynb`)
******************************************
The *Spectral Data* sonification demonstrates use of the Strauss :code:`Spectraliser` to represent data. We use a direct spectralisation approach where the sound is generated by treating the 1D data as a sound spectrum. This uses a direct inverse Fourier transform, which is relatively intuitive for spectral data, especially where the spectral features are similar to those that can be identified in sound. We use Planetary Nebulae data, objects dominated by strong emission lines, to demonstrate this. We plot the spectra vs wavelength and spectra vs frequency, and use the Strauss :code:`Synthesiser` to create a 30 second, mono sonification. We set the ranges for the mapped parameters and render the sonification. A second example uses an "Object" type sonification with an evolving Spectrum to sonify an image. We represent the image by evolving from left to right, with higher features in the y-axis having a higher pitch.

Three examples using different approaches demonstrated in the notebook can be heard in the videos below

#. `Spectraliser representation of NGC 1535 <https://www.youtube.com/watch?v=ZONZXZGCAEA>`_.
#. `Spectraliser representation of NGC 6302 <https://www.youtube.com/watch?v=xLTns7JmvDA>`_.
#. `Evolving spectraliser representation of STRAUSS logo <https://www.youtube.com/watch?v=MRUO2BWg2Vw>`_.

The code is available in a `Google Colab notebook <https://githubtocolab.com/james-trayford/strauss/blob/main/examples/colab/SpectralData.ipynb>`_.

Stars Appearing (:code:`StarsAppearing.ipynb`)
**********************************************
The *Stars Appearing* sonification demonstrates the generation of a sonification that was used directly in the "Audio Universe: Tour of the Solar System" planetarium show. This is intended to represent the appearance of stars in the night sky to an observer. Over time, the sky darkens and our eyes adjust, allowing us to see more and more stars. To represent this, the brightest stars appear first, with dimmer stars appearing later. Data on the colours of the stars is used to set the note used for each stars sound, with bluer stars having higher notes and redder stars having lower notes. We use the Strauss :code:`Sampler` to play a glockenspiel sound for each star as it appears. The actual positions of stars in the sky is used to spatialise the audio, with westerly stars positioned in the right speaker and easterly stars in the left. This spatialisation can be mapped to any audio setup, and a :code:`5.1` system was used for the planetarium, but for the example we use :code:`stereo`.

A video of this sequence with the audio is available `here <https://www.youtube.com/watch?v=5HS3tRl2Ens>`_.

The code is available in a `Google Colab notebook <https://githubtocolab.com/james-trayford/strauss/blob/main/examples/colab/StarsAppearing.ipynb>`_.
