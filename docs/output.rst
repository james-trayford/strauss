.. _formatting:

Formatting Audio
################

This page covers ways in which you can format the audio output by Strauss. This includes the audio system that governs spatialisation effects (such as stereo or surround-sound formats). It also covers figure and sonification auditory display features, such as spoken captions, spoken annotations and tick-marks, which are added to a figure to aid communication of the data.

Audio systems
*************

.. versionchanged:: 1.5
   ``'ambiX<N>'`` channels with ``m=0`` are now normalised to SN3D. Earlier versions made them 0.84 dB too loud.

A figure's audio system needs to be consistent for all sonifications, and is set when the figure is created (see :doc:`audio_figure`):

.. code-block:: python

   import strauss

   fig = strauss.AudioFigure(system='5.1')

   # or, for the current figure
   fig.close()
   strauss.sonify(x, y, channels='5.1')

The output channel format is chosen by the ``system`` keyword, and can be one of the following:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - System
     - Channels
   * - ``'mono'``
     - A single channel (no spatial effects).
   * - ``'stereo'``
     - Left and right.
   * - ``'5.1'``
     - Front left and right, centre, low-frequency (LFE), and left and right surround. The centre and LFE channels are silent, as sources are panned between the other four.
   * - ``'7.1'``
     - As ``'5.1'``, with two more channels directly left and right.
   * - ``'ambiX<N>'``
     - Ambisonics of order ``N`` (e.g. ``'ambiX1'``, ``'ambiX3'``), with (N+1)² channels in ACN order and SN3D normalisation. Ambisonic files describe the whole sound field, and need decoding for a particular set of speakers or for headphones.
   * - ``'custom'``
     - Your own set of channels, given by ``custom_setup``.

A custom setup lists the azimuthal position (in radians, counter-clockwise from the front), type and label of each channel, in order. This one would be the same as ``'stereo'``:

.. code-block:: python

   import numpy as np

   fig = strauss.AudioFigure(system='custom',
                             custom_setup={'azimuths': [0.5*np.pi, 1.5*np.pi],
                                           'types': ['directional', 'directional'],
                                           'labels': ['L', 'R']})

Note: for custom setups strauss will not check that the channels are valid, or that they are in a sensible order. It is up to you to ensure that your custom setup is correct, and you have a system capable of reproducing them! It may also be the case that additional metadata is required for your custom setup to be correctly interpreted by other software, such as a DAW. Again this can be handled with external tools.

Placing sources in space
************************

.. versionchanged:: 1.5
   ``'5.1'`` and ``'7.1'`` now pan pairwise (``'vbap'``) by default. Use ``panning='cardioid'`` for the earlier behaviour.

Sources are placed spatially for spatialisation by the audio system using the simple, left-right ``pan`` parameter, or fully specified using spatial angles (e.g. setting``azimuth`` and ``polar`` parameters). See :doc:`params` for more on parameter units.

This spatial information determines how a source's sound gets distributed between channels. For surround-sound, ``<N>.1`` systems, the ``panning`` option also controls the algorithm used to localise sources, choosing from one of the following:

* ``'vbap'``: pairwise panning, so each source's azimuthal component is interpolated between one or two channels, with remainder split between all spatial channels. This is the default for ``'5.1'`` and ``'7.1'``.
* ``'cardioid'``: each channel picks up sound over a broad, 'cardioid' antenna pattern, so a source's sound is reproduced in any channel that isn't at 180° to it's angular position. This is the default for ``'stereo'``, and was the universal behaviour of strauss channels prior to v1.5.

.. code-block:: python

   fig = strauss.AudioFigure(system='5.1', panning='vbap')

Saving
******

.. versionchanged:: 1.5
   One ``save`` for every audio system, recording surround channel layouts. The loudest peak now sits exactly at ``master_volume``, where earlier headroom grew with the number of channels.

``save`` writes the figure's audio to a file. The type of file is specified by its extension:

.. code-block:: python

   fig.save('mix.wav')
   fig.save('mix.mp3')

* ``.wav`` files are written directly by Strauss, as 32-bit audio for mono and stereo.
* Other formats attempt conversion with ``ffmpeg``, which needs to be installed locally if not there already (see `ffmpeg.org <https://ffmpeg.org/download.html>`_). Strauss should raise an error if ``ffmpeg`` is required but not found, with further instructions.
* Surround format outputs are also passed through ``ffmpeg`` where it is available, applying extra metadata and using a 24-bit rate to maximise compatibility .

The master audio is scaled so that its loudest peak is at the value specified ``master_volume``, with maximum amplitude (``1``) by default. Give a lower amplitude, or a level in decibels, to leave some headroom:

.. code-block:: python

   fig.save('mix.wav', master_volume='-3 dB')

Captions
********

.. versionchanged:: 1.5
   Captions can use the ``kokoro`` text-to-speech engines.

A *caption* is a phrase spoken using text-to-speech at the start 
(or end) of the audio, and used to provide helpful context for understanding the sonification. Provide one with the ``caption`` keyword when sonifying:

.. code-block:: python

   strauss.sonify(x, y, caption='The brightness of a pulsating star over ten days.')

Captions are spoken in the order their sonifications were added, before the mix plays. The text-to-speech engines used for this are described in :doc:`aliases`, with captioning needing one to be installed.

Ticks (provisional)
*******************

.. versionadded:: 1.5

"Ticks" are short transient sounds for marking regular steps of the input data for the time axis, like the tick marks on a plot's axis.

The ``add_ticks`` method expects a time interval in the sonification's time units, and a ``name`` referring to which sonification the tickds relate to:

.. code-block:: python

   fig.add_ticks(10, name='tone')

Here, a tick sounds to makr multiples of 10 in the units of the ``'tone'`` sonification's input data for the time parameter. ``name`` can be left out when the figure uses a single sonification.

Optional keyword arguments are also available: ``duration`` sets the length of each tick in seconds, and ``level`` sets their volume.
