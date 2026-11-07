.. _audio_figure:

Audio Figures
#############

An *audio figure* allows multiple sonifications to be combined for richer sonic representations of data. The audio figure has (in-development) audio display features, such as axis ticks, spoken captions or an audio legend. Each sonification is added to an *audio figure* as a named component with its own level or "track", which is mixed together as a single piece of audio.

A useful analogy could be a plotting library like `matplotlib <https://matplotlib.org/>`_, where a figure can contain multiple plots, each with its own data and plotting styles.

Ways to work with figures
*************************

The top-level functions used in :doc:`quickstart` (``strauss.sonify``, ``strauss.display``, ``strauss.save``, etc) use audio figures implicitly. If no audio figure currently exists, calling  ``strauss.sonify`` creates one. Further calls of the top-level functions then apply to the last active figure. The active figure  persists until  ``strauss.close()``:

.. code-block:: python

   import numpy as np
   import strauss

   x = np.linspace(0, 90, 400)
   y = np.sin(x/2) * np.exp(-x/60)

   strauss.sonify(x, y, name='tone', level='-10 dB')
   strauss.sonify(x, -y, style='windy', name='wind')
   strauss.save('mix.wav')
   strauss.close()

You can also create audio figures explicitly, allowing multiple figures to be maintained at once, or passed around as a variable. This uses the same methods as the top-level functions:

.. code-block:: python

   fig = strauss.AudioFigure()
   fig.sonify(x, y, name='tone', level='-10 dB')
   fig.sonify(x, -y, style='windy', name='wind')
   fig.save('mix.wav')

Both produce the same output. Below we'll use ``AudioFigure`` methods explicitly. Most of these have a top-level equivalent (i.e. of the form ``strauss.<method>``) for acting on the current active figure.

Channels and duration
*********************

A figure's audio system, e.g. ``'mono'``, ``'stereo'`` or ``'5.1'``, is set on creation, as ``AudioFigure(system=...)``, or with ``channels=`` when calling the ``strauss.sonify`` function that creates a new figure implicitly.

All sonifications in a figure will use the same audio system and ``duration``. Adding a sonification asking for a different duration raises an error.

Sonify keywords
***************

Along with the data, ``sonify`` takes these optional keywords:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Keyword
     - Default
     - Description
   * - ``style``
     - ``'default'``
     - A built-in style name, or the path to a style file. See :doc:`styles`.
   * - ``name``
     - ``'sonification_<N>'``
     - Name to give the sonification, used to reference it later.
   * - ``level``
     - ``1``
     - Relative volume level, either as a linear amplitude (e.g. ``0.5``) or a string in decibels (e.g. ``'-6 dB'``).
   * - ``duration``
     - ``10``
     - Length of any sonifications in seconds.
   * - ``notes``
     - from the style
     - Notes to play overriding any specified in the style, using e.g. a chord name or a list of notes.
   * - ``fix_<parameter>``
     - Depends on the chosen parameter, style, presets, etc.
     - Fix a sound parameter at a value for the sonification, e.g. ``fix_pan=0.25``, in the set units (see the units note in :doc:`params`).
   * - ``handle_nans``, ``merge_mode``, ``angle_unit``
     - from the style
     - Override style behaviour options (described in :doc:`styles`).
   * - ``source_names``
     - ``'source_0'``, ...
     - Set of names referencing each source, used in tables and plots. See :doc:`inspecting`.

Unrecognised keywords are ignored with a warning. A ``fix_`` keyword for a parameter mapped by the style overrides the mapping, providing a warning.

Calling ``sonify`` again with identical data and options avoids making a copy. If only the ``level`` differs, the existing sonification's volume is updated instead.

Managing the audio mix
**********************

You can list the sonifications in a figure, change their levels, or drop one altogether:

.. code-block:: python

   fig.list_sonifications()
   fig.set_level('wind', '-3 dB')
   fig.remove('tone')

``add`` can be used to add a new ``Sonification`` that was built with the lower-level interface (see :doc:`elements`) to a figure, along with a ``name`` and ``level``:

.. code-block:: python

   fig.add(my_sonification, name='custom', level=0.5)

Rendering, playing and saving audio
***********************************

A figure renders sonifications and produces the overall audio mix when it is displayed or saved, or re-renders it if a sonification or its volume level has changed relative to a previous render. The volume mix is scaled by default with the loudest amplitude peak at 100% volume, to maximise volume contrast. When saving, ``master_volume`` can be set to lower this - either as an amplitude from 0 to 1, or a string in decibels:

.. code-block:: python

   fig.save('mix.wav', master_volume='-3 dB')
   fig.notebook_display()

``fig.render()`` can render the mix without saving or displaying it. Formats other than ``.wav`` attempt conversion based on the file extension via ``ffmpeg``, as described in :doc:`quickstart`.

Figures can't yet be played outside a notebook, but should also get a ``hear`` function.

Data input
**********

``sonify`` takes data in the same forms as in :doc:`quickstart`: numpy arrays, pandas Series, a DataFrame or a CSV file.
