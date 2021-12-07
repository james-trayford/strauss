
.. _examples:

Examples
^^^^^^^^

Here, we explain some of the example sonfications included in the :code:`examples/` directory of the Strauss repo. These are all in *Python Notebook* (:code:`.ipynb`) format.

Stars Appearing (:code:`StarsAppearing.ipynb`)
**********************************************

The *Stars Appearing* sonification demonstrates the generation of a sonification that was used directly in `a planetarium show for visually impaired children <https://www.audiouniverse.org/tour-of-the-solar-system>`_. This is intended to represent the appearance of stars in the night sky to an observer. Over time, the sky darkens and our eyes adjust, allowing us to see more and more stars. To represent this, the brightest stars appear first, with dimmer stars appearing later. Data on the colours of the stars is used to set the note used for each stars sound, with bluer stars higher notes and redder stars lower notes. We use the Strauss :code:`Sampler` to play a glockenspiel sound for each star as it appears. The actual positions of stars in the sky is used to spatialise the audio, with westerly stars positioned in the right speaker and easterly stars in the left. This spatialisation can be mapped to any audio setup, and a :code:`5.1` system was used fo the planetarium, but for the example we use :code:`stereo`

A video of this sequence with the audio is available `here <https://www.youtube.com/watch?v=5HS3tRl2Ens>`_.
