
.. _elements:

Elements of a Strauss Sonification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Strauss makes use of a number of internal Classes to produce the output audio. These handle broadly different aspects of the sonification process, and are detailed in the following subsections.

Briefly, a :code:`Source` represent some aspect of the data and how variables are mapped which is taken by the :code:`Score` which defines any constraints on which notes or sounds can be generated at a given time. A :code:`Generator` object then generates these notes and sounds as an audio signal, including any mapping of :code:`Source` properties to expressive properties of sound. These audio signals are then consolidated by the output :code:`Channels` object according to any spatialisation and choice of audio system (mono, stereo, 5.1, etc). This processing is all handled inside the :code:`Sonification` class, which can render and output the audio. This scheme is illustrated by the figure below.

.. figure:: figures/strauss_flow.png
   :scale: 8 %
   :alt: flowchart representing the Strauss code

   **Figure:** Basic flow of the Strauss code, from input data to output audio.

.. _sources:

Source Class
************

The data being sonified is represented by :code:`Source` classes in Strauss.  

.. _score:

Score Class
***********

*Text goes here*

.. _generator:

Generator Class
***************

*Text goes here*

.. _channels: 

Output Channels Class
*********************

*Text goes here*

.. _sonification: 

Sonification Class
******************

*Text goes here*
