.. _aliases:

Calling Samples and Spoken Phrases
##################################

.. versionadded:: 1.5
   Sample aliases, the ``call`` mapping and the ``Speech`` generator.

The ``Sampler`` generator plays recorded audio clips or "samples", and the ``Speech`` generator speaks phrases using text-to-speech (TTS). Choosing which sample or phrase each source plays, by *alias* or by a ``call`` mapping, is new in v1.5.

.. note::

   Aliases, ``call`` and ``Speech`` can't yet be used through styles or ``strauss.sonify``, so the examples here use the lower-level interface described in :doc:`elements`.

Loading samples
***************

A ``Sampler`` loads either a folder of audio files or a "soundfont" (``.sf2``) file:

.. code-block:: python

   from strauss.generator import Sampler

   generator = Sampler('path/to/drums')
   generator = Sampler('path/to/instrument.sf2', sf_preset=1)

Samples in a folder are given a *"home note"*, at which they play unshifted. Naming files in the format ``<folder>_<note>.wav`` (e.g. ``chime_C1.wav``) sets these notes. Where files have no notes specified in their name, Strauss takes consecutive semitones in filename order, skipping any notes that are explicitly assigned by named files. Alternatively, setting keyword argument ``assign_notes='detect'``, instead chooses the note closest to the peak frequency of untagged samples, by analysing their frequency spectrum.

A soundfont holds one or more presets (typically different instruments). These can be selected via the ``sf_preset`` argument, from preset ``1`` to ``N`` (where ``N`` is the total number of presets). If no preset is specified, the available presets are printed and the first is used.

``info`` provides information about the loaded samples, including each sample's home note, the range of notes it covers and any aliases:

.. code-block:: python

   generator.info()

Built-in samples
****************

The built-in styles use hosted samples that are downloaded (via an internet connection) the first time they're needed, and kept in your local cache. ``assets.get_asset_path`` can be used to fetch one by name, for use in your own sampler:

.. code-block:: python

   from strauss import assets

   generator = Sampler(assets.get_asset_path('mallet'))

As of writing, the names available are ``'glockenspiel'``, ``'mallet'``, ``'flute'`` (a soundfont), ``'space strings'`` and ``'bright mallets'``.

Aliases
*******

Samples can be invoked using an *alias* instead of a note: by default, the filename stem, e.g. ``'snare'`` for ``snare.wav``. Internally, Strauss's ``Score`` can use aliases in place of notes, such that a ``pitch`` mapping chooses between the samples named *{Note: provisional as commandeering the pitch mapping for aliases is potentially confusing. This is assuming aliases are sorted low to high with some ordinal meaning. can consider how this works for v1.5+}*:

.. code-block:: python

   from strauss.score import Score

   score = Score([['kick', 'snare', 'hat']], 10)

``set_aliases`` can be used to add or change aliases for samples, by referring to their home note:

.. code-block:: python

   generator.set_aliases({'kick': 'C1', 'snare': 'D1'})

Each sample can have one alias, and an alias isn't allowed to be formatted as valid note name (e.g. ``'A4'``), as this would get interpreted as a note.

Calling samples directly
************************

While the ``pitch`` mapping currently picks a sample assuming the aliases to be sorted ordinally, the ``call`` mapping instead can be used to reference the sample directly per source. When mapping to ``call``, a list of alias strings refers to the sample representing each source:

.. code-block:: python

   import numpy as np
   from strauss.sources import Events

   hits = ['kick', 'hat', 'snare', 'hat', 'kick', 'kick', 'snare', 'hat']
   data = {'call': hits,
           'time': np.linspace(0, 0.9, len(hits))}

   sources = Events(data.keys())
   sources.fromdict(data)
   sources.apply_mapping_functions()

The musical notes used by Strauss' score are bypassed when ``call`` is used to refer to samples, though the ``note_length`` parameter still sets the duration *{should be looked into, probably want length to be that of the sample by default}*. For the reference tables produced by Strauss (see :doc:`inspecting`), a ``Sound`` column lists the alias used for each event.

Speech
******

The ``Speech`` generator generates spoken audio for a set of phrases, provided as a list of strings, storing each as a sample. With the speech audio produced, the ``Speech`` generator works identically to the ``Sampler``, applying any envelopes, filters, pitch manipulation, spatialisation, etc. The input phrases function as aliases, that can be referenced by a ``pitch`` or a ``call`` mapping:

.. code-block:: python

   from strauss.generator import Speech
   from strauss.sonification import Sonification

   cols = ['red', 'blue', 'green', 'blue']
   data = {'call': cols,
           'time': np.linspace(0, 0.85, len(cols))}

   sources = Events(data.keys())
   sources.fromdict(data)
   sources.apply_mapping_functions()

   soni = Sonification(Score([['C3']], 10), sources, Speech(), 'stereo')
   soni.render()

When the ``Speech`` generator is provided with no initial list of phrases, phrases are generated for whatever the data calls. Given a list, e.g. ``Speech(['low', 'medium', 'high'])``, the phrases are assigned to consecutive notes in order, with a ``pitch`` mapping choosing between them from low to high.

``Speech`` requires a text-to-speech engine, installed as an optional extra with ``pip install strauss[speech]``. In preference order, Strauss will try to use:

1. ``kokoro``: good quality, running offline once models are downloaded.
2. ``kokoro-onnx``: the same model and voices, where ``kokoro`` can't be installed.
3. ``pyttsx3``: makes use of your system's own text-to-speech - platform dependent.

You can choose one with ``strauss.tts_caption.set_engine``, and select a voice via the ``voice`` argument (see ``strauss.tts_caption.getVoices``). Spoken phrases are stored in cache, by default (at ``~/.cache/strauss/speech``), but can be removed with ``clear_cache``. 

These TTS engines are also used to render captions (see :doc:`output`).
