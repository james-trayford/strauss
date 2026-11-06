.. _styles:

Styles
######

A style specifies how data gets represented as sound for a given sonification. This includes: the sound to use (i.e. the "generator"), the representation of input data (either as transient events or continuous objects), which expressive properties of sound the data is mapped to, and any set of musical notes to stick to. 

Styles are completely specified by their YAML file, and so this is all you need to share, write or modify your own styles. 

Built-in styles
***************

.. include:: builtin_styles.md
   :parser: myst_parser.sphinx_

Styles using samples or "soundfont" files download any assets you need the first time you use them, and keep them in your local cache.

Using a style
*************

Provide a built-in style's name, or the path to your own custom style file (e.g. ``'/your/custom/style.yml'``) to the ``style`` argument of ``strauss.sonify()``. 

The style's mappings will then be applied to your data, and the sound generated accordingly:

.. code-block:: python

   import numpy as np
   import strauss

   x = np.linspace(0, 90, 400)
   y = np.sin(x/2) * np.exp(-x/60)

   strauss.sonify(x, y, style='windy')
   strauss.close()

To see how a built-in style is written, print it with ``get_style``. It returns the text as a string, to use as a starting point for your own:

.. code-block:: python

   text = strauss.get_style('flute_section', print_style=True)

Writing a style
***************

This style sustains a chord and expresses the data by changing its 'timbre': the 'brightness' evolves with time as the ``'y'`` variable varies with ``'x'``, panned leftwards:

.. code-block:: yaml

   name: 'My Style'

   description: 'A sustained chord whose brightness follows the data.'

   sources: 'objects'

   generator:
     type: 'synthesizer'
     mods:
       volume_envelope:
         A: 0.5

   map:
     - input: 'x'
       output: 'time_evo'

     - input: 'y'
       input_range: ['5%', '95%']
       output: 'cutoff'
       output_range: [0.4, 0.95]

     - output: 'pan'
       fixed: 0.25

   notes: ['G2', 'D3', 'Bb3', 'F4', 'Bb4']

.. code-block:: python

   import pandas as pd

   strauss.sonify(pd.DataFrame({'x': x, 'y': y}), style='my_style.yml')
   strauss.close()

Each entry in ``map`` either maps data to a sound parameter, listed in the :ref:`Mappable Parameters table <mappable-parameters>`, or holds one at a ``fixed`` value:

* **Order:** data mappings are mapped to arrays in the same order they're input to the ``sonify`` function. ``fixed`` entries set values for the whole sonification, without any input data.
* ``input``: the name of the column to pick from a DataFrame or CSV file. It has no effect on arrays.
* ``input_range``: the data values to map from, as absolute values or percentiles of the input data.
* ``output_range``: the range of a parameter's value used for the mapping, in the parameter's units.
* ``function``: a set of useful transformation functions to apply to the data first, including ``log10`` or ``invert``.
* ``fixed``: the value to hold parameters at if they're not intended to be controlled by the data. See the units note in :doc:`params`.

``generator`` selects how the sound is generated (using one of Strauss's set of generators, see :doc:`elements`). ``mods`` changes individual settings of its preset, along with any preset and parameter overrides for each generator, from those available (listed in :doc:`params`).

Behaviour options
*****************

These optional fields control how a style represents the data. Those marked in the last column can also be overridden per sonification, as keywords to ``sonify``:

.. include:: behaviour_options.md
   :parser: myst_parser.sphinx_

``notes``, and fixing a parameter with ``fix_<parameter>``, can also be set by keyword:

.. code-block:: python

   strauss.sonify(x, y, style='windy', fix_pan=0.25, handle_nans='interpolate')
   strauss.close()

Current limitations (TODO for v2.0)
***********************************

* The ``Speech`` generator and ``call`` parameter can't be used through styles yet.
* The ``spectrum`` parameter is provisional: how it works inside the style paradigm still needs to be settled.
* Spatial angles and their units need to be clarified, and the ``angle_unit`` option is provisional. See :doc:`params` for current behaviour.

Style reference
***************

Every other field of a style, generated from the code, with the behaviour options covered above. ``mods`` takes the parameters listed for each generator in :doc:`params`, and can be used to override those in a generator's "presets".

.. include:: style_tables.md
   :parser: myst_parser.sphinx_
