.. _quickstart:

Quickstart
##########

Here, we demonstrate how you can make your first sonifications in just a few lines using the high-level ``strauss.sonify`` interface! First, you'll need ``strauss`` installed. See :doc:`start` for instructions.

Your first sonification
***********************

Let's first generate some data. We'll try a simple oscillating function that gradually decays:

.. code-block:: python

   import numpy as np
   import strauss

   x = np.linspace(0, 90, 400)
   y = np.sin(x/2) * np.exp(-x/60)

   strauss.sonify(x, y)

Without any extra arguments, strauss uses the ``default`` style for ``strauss.sonify``.

This uses a continuous, synthesised tone where pitch varies with time, just as the ``y`` array varies with the  ``x`` array, over a duration of 10 seconds.

Playing and saving it
*********************

When run in a notebook, ``strauss.display()`` gives you an audio player to hear the sonification:

.. code-block:: python

   strauss.display()

In the command line, or elsewhere, play it directly via the default sound output on your system (provided one is available) with ``hear``:

.. code-block:: python

   soni = strauss.sonify(x, y)
   soni.render()
   soni.hear()

.. note::

   ``hear`` currently plays one sonification. To hear the mix outside a notebook, save it to a file.

To save audio, use ``strauss.save``. Just type in the file path where you want your sonification file to go. 

Formats other than ``.wav``, such as ``.mp3``, will attempt to use ``ffmpeg`` for conversion, so require this to be installed. See guidance for ffmpeg at `ffmpeg.org <https://ffmpeg.org/download.html>`_.

.. code-block:: python

   strauss.save('my_sonification.wav')


In ``strauss``, a *sonification* is a particular mapping of data variables to properties of sound. Multiple sonifications can be combined together in an *audio figure*, allowing them to be played at once, as we'll demonstrate below. 

By default, new sonifications are added to the current audio figure by ``strauss.sonify`` until it gets closed. Use ``strauss.close`` to start a new audio figure. 


.. code-block:: python

   strauss.close()

Changing the style
******************

Styles control how our data is represented as sound in ``strauss``. The ``windy`` style, for example, maps higher ``y`` values to a more intense wind-like sound (NB. technically, this is mapping ``y`` to the logarithmic cutoff frequency of a low-pass Butterworth filter, applied to a sustained white noise carrier sound!). 

You can also change the duration for your sonification, in seconds:

.. code-block:: python

   strauss.sonify(x, y, style='windy', duration=20)
   strauss.display()
   strauss.close()

For styles with a pitched sound, ``notes`` changes what musical notes are used. For the ``flute_section`` style these notes represent a sustained chord, with higher ``y`` resulting in a *"brighter"* sound (using the same approach as for ``windy`` above). 

Let's change it to a different chord. You can use a common musical chord name and octave as below. The flute audio that the style uses is downloaded when we first use it, so can take a bit longer:

.. code-block:: python

   strauss.sonify(x, y, style='flute_section', notes="D7_3")
   strauss.display()
   strauss.close()

Or even specify the notes in the chord directly (using scientific notation for the pitch, with the octave number after each note):

.. code-block:: python

   strauss.sonify(x, y, style='flute_section', notes=['G2', 'D3', 'B3', 'F#4', 'B4'])
   strauss.display()
   strauss.close()


See :doc:`styles` for all the built-in styles and how to write your own!

Combining sonifications in an Audio Figure
******************************************

Calling ``sonify`` multiple times in succession adds to the current audio figure, with the sonifications played together. Giving them names lets you identify them and do things like adjust their relative volumes afterwards. Here, the ``default`` and ``windy`` styles follow our wave, with the ``y`` data flipped relative to each other:

.. code-block:: python

   strauss.sonify(x, y, name='tone', level='-10 dB')
   strauss.sonify(x, -y, style='windy', name='wind')
   strauss.set_level('wind', '-6 dB')
   strauss.display()
   strauss.close()

Using your own data
*******************

A style can map the data input to sonify directly in the order it's provided. For example, the ``default`` style maps time and then pitch, respectively, with the table in :doc:`styles` listing the order for each provided style. 

Currently, data can be provided to the sonification function in the following forms:

**multiple numpy Arrays or pandas Series**, following the order they are passed in. Note: If more arguments are provided than are needed by a given style, extra arrays or series are ignored with a warning. If you provide fewer arrays or series than a style can map, the later mappings are dropped.

.. code-block:: python

   strauss.sonify(x, y)
   strauss.close()

**A single array**, which is mapped to the style's second mapping, assuming its points are evenly spaced in time:

.. code-block:: python

   strauss.sonify(y)
   strauss.close()

**A pandas DataFrame or CSV file**, where columns are taken in order, and any extra columns are ignored. A style can also pick columns by name or index, using ``input`` (see :doc:`styles` for more details).

.. code-block:: python

   import pandas as pd

   df = pd.DataFrame({'x': x, 'y': y})
   strauss.sonify(df)
   strauss.close()

   df.to_csv('my_data.csv', index=False)
   strauss.sonify('my_data.csv')
   strauss.close()

Visualising what was sonified
*****************************

To follow how the data is being represented as sound, ``strauss.get_table()`` provides the mapped values as a table in the natural units used for each parameter. 

``strauss.plot_mapping()`` visualises this: pairs of axes show how the input values correspond to time and the expressive properties of sound, along the top and right-hand axes. 

.. code-block:: python

   strauss.sonify(x, y)
   table = strauss.get_table()
   strauss.plot_mapping()
   strauss.close()

Next steps
**********

* :doc:`styles`: read about built-in styles, or write your own.
* :doc:`elements`: read about the lower-level interface that styles are built on for more advanced users.
* :doc:`params`: reference for every sound parameter in Strauss that can be set or mapped.
* :doc:`examples`: try out notebooks applying Strauss to diverse data in different ways!
