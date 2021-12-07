
.. _elements:

Elements of a Strauss Sonification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Strauss makes use of a number of internal Classes to produce the output audio. These handle broadly different aspects of the sonification process. For more in-depth documentation see the :code:`Detailed Documentation`.

Briefly, a :code:`Source` represents the data and the mapping of its variables onto aspects of sound. This is taken by the :code:`Score` which defines any constraints on what notes or sounds can be generated at a given time. A :code:`Generator` object then generates these notes and sounds as an audio signal, including any mapping of :code:`Source` properties to expressive properties of sound. These audio signals are then consolidated by the output :code:`Channels`, object according to any spatialisation (e.g. source position) and choice of audio system (mono, stereo, 5.1, etc).

This processing is all handled inside the :code:`Sonification` class, which can render and output the audio. This scheme is illustrated by the figure below.

.. figure:: figures/strauss_flow.png
   :scale: 8 %
   :alt: flowchart representing the Strauss code

   **Figure:** Basic flow of the Strauss code, from input data to output audio.

.. _sources:

Source Class
************

The :code:`Source` classes in Strauss are so named because they act as `sources of sound` in the sonification. Sources are used to represent the input data, by mapping this data to properties of sound (volume, position, frequency, etc). The choice of how to set up the sources depends on the data being sonified, and what the user wants to convey. For this, Strauss defines two generic classes that inherit the parent :code:`Source` class; :code:`Events` and :code:`Objects`, described below.

**Full documentation of the mappable parameters are coming soon!** See :obj:`./examples` for example mappings in jupyter notebooks.

`Events`
''''''''
The :code:`Events` source class is suited to represent data that can be characterised by an **occurence time**.

In this case, sounds are triggered over the duration of the sonification, with properties of each discrete event (e.g. brightness, size,  etc.) mapped to properties of sound (e.g. volume, pitch, etc). :code:`Events` will therefore typically be defined by single values, including a :code:`time` variable, else all events will occur at the start of the sonification.

.. note::

   Some examples of :code:`Events` in scientific data could be; `stars forming, supernovae explosions, particle detections, lightning strikes, website interactions, etc...`
   
`Objects`
'''''''''
The :code:`Objects` source class is suited to represent data that **evolves continuosly over time**.

In this case, sound is produced continuously by the source, with the evolving properties of the object (e.g. position, brightness, mass, etc) mapped to properties of the sound (e.g. panning, volume, pitch). :code:`Objects` will therefore typically each be defined by arrays of values, and a corresponding :code:`time_evo` array, indicating when the measurements are taken.

.. note::

   Some examples of :code:`Objects` in scientific data could be; `A galaxy evolving, planets orbiting, a plant growing, a glacier flowing, a climate changing etc...`   

.. _score:
