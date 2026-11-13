Getting Started
^^^^^^^^^^^^^^^

This walkthrough will take you through a clean install of the code, including optional dependencies and trying your first sonification. There are also example notebooks available on `Google Colab <https://colab.research.google.com/github/james-trayford/strauss/blob/colab_examples/>`_ which you can run without installing Strauss on your local system.

Installation
************

Strauss can be installed in different ways, depending on whether you want to develop the code or simply use it as it is. It can be installed using pip install from PyPI directly, or you can install it from a local copy of the GitHub repository, with or without the option for development.

.. note::
   To use text-to-speech functionality see the further steps in the next section 

For basic functionality, you can install directly from PyPI with pip:

.. code-block:: bash

   pip install "strauss"

or, more explicitly, with the default dependencies:

.. code-block:: bash

   pip install "strauss[default]"

Alternatively, the Strauss code can be downloaded from **GitHub** at `the repository url <https://github.com/james-trayford/strauss.git>`_

Using :code:`git` make a copy of the STRAUSS repository via SSH,

.. code-block:: bash
  
  git clone git@github.com:james-trayford/strauss.git strauss

or HTTPS if you don't have SSH keys set up,

.. code-block:: bash

  git clone https://github.com/james-trayford/strauss.git strauss

if you just want to use the code, STRAUSS may then be installed using pip, as

.. code-block:: bash
		
	cd strauss
	pip install ".[default]"

If you want to develop the code, you can instead use

.. code-block:: bash
  
	pip install -e ".[default]"

throughout the documentation, I will refer to this as the **strauss repo** or **code directory**.

To build these docs locally, you can install :code:`docs` as an extra, and run :code:`Sphinx` from the :code:`./docs` directory:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html

The built pages are written to :code:`docs/_build/html`.

.. _ffmpeg-install:

ffmpeg
******

Some features use `ffmpeg <https://ffmpeg.org>`_ for media encoding, a separate program rather than a Python package. On certain platforms, this isn't pre-installed and :code:`pip` cannot install it directly. A local :code:`ffmpeg` install is used to:

* Save audio in formats other than :code:`.wav`, such as :code:`.mp3` (see :doc:`output`)
* Correctly encode the channel layout for surround-sound files
* Write videos via the :code:`animate_mapping` method (see :doc:`inspecting`)

Other functionality, including producing :code:`.wav` files end-to-end, works without :code:`ffmpeg`. If it isn't already available, you can install it with a package manager, for example:

.. code-block:: bash

   brew install ffmpeg          # macOS, with Homebrew
   sudo apt install ffmpeg      # Debian and Ubuntu
   conda install -c conda-forge ffmpeg

or see `ffmpeg.org/download <https://ffmpeg.org/download.html>`_ for more options.

.. _tts-install:

Text-to-speech
**************

For text-to-speech (TTS) functionality, there are a couple of options: *system* and *AI* text to speech.

If you would like to use system TTS, you will need to install the (as of writing) cutting-edge `pyttsx3` dependency, with 

.. code-block:: bash

   pip install --no-cache-dir --extra-index-url https://test.pypi.org/simple/ pyttsx3==2.99

If you would like to use AI text-to-speech instead, you can instead install strauss requesting the optional :code:`speech` dependency (also available as :code:`TTS` or :code:`AI-TTS`), which installs the :code:`kokoro` engine (or :code:`kokoro-onnx` on Python 3.13). 

.. code-block:: bash

   pip install "strauss[speech]"

or, for an install from a local repository copy:

.. code-block:: bash

   pip install -e ".[speech]"

.. note::
   This is an optional extra rather than a default as :code:`kokoro` pulls in :code:`torch`, which is
   a few hundred MB; voice models download on first use.

.. note::
   :code:`kokoro` only supports Python<=3.12, so installs :code:`kokoro-onnx` (different pronunciation engine and  lacking Japanese and Mandarin voices). To use it on earlier versions, install
   :code:`strauss[kokoro-onnx]`.

   :code:`kokoro-onnx` doesn't yet allow Python 3.14 due to version limits, in this case you can try to force install with: 

   .. code-block:: bash

      pip install --ignore-requires-python kokoro-onnx==0.6.1

Strauss uses our favoured engine, prioritising :code:`kokoro`, (or
:code:`kokoro-onnx`) with :code:`pyttsx3` as a fall back option. To choose one explicitly, or list voices:

.. code-block:: python

   from strauss import tts_caption
   tts_caption.set_engine('kokoro')
   tts_caption.getVoices(info=True)

A :code:`kokoro` voice is named :code:`<language><gender>_<name>`, so :code:`'af_heart'` is an
American female voice and :code:`'bf_emma'` a British female one. See the voice list at:
`hexgrad/Kokoro-82M VOICES.md <https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md>`_.
A voice is chosen per caption with :code:`Sonification(ttsmodel='af_heart')`, and for spoken
sources with :code:`Speech(voice='af_heart')`.

Your first sonification
***********************

With Strauss installed, try your first sonification in just a few lines following the :doc:`quickstart` guide. In a single block:

.. code-block:: python

   import numpy as np
   import strauss

   # some made-up data to sonify
   x = np.linspace(0, 90, 400)       
   y = np.sin(x/2) * np.exp(-x/60)

   # let's make a sonification
   strauss.sonify(x, y)

   # then save it to a file...
   strauss.save('my_sonification.wav')

   # ...or just listen to it directly
   strauss.hear()

Example jupyter notebooks/scripts
*********************************

There are a number of example applications of Strauss in the :code:`example` subdirectory of the :code:`strauss` repo. These are in Python Notebook (:code:`.ipynb`) format for an interactive, step-by-step experience. They are also provided in Python script format (.py) in the :code:`examples` directory. The Python scripts can be run from the command line.

In order to run the notebook examples, first ensure that :code:`jupyter` is installed on your system. These were developed in :code:`jupyter-lab`, which can also be installed using pip, as:

.. code-block:: bash

   pip install jupyterlab

Then, running :code:`jupyter-lab` in the :code:`strauss` should initiate the :code:`jupyter-lab` server and open a browser window. Navigate to the :code:`examples` directory within the :code:`jupyter-lab` navigation plane, from which a number of examples can be opened and run interactively.

Running some examples
*********************

From the :code:`jupyter-lab` interface, the :code:`StyleSonification.ipynb` Notebook is the quickest start, sonifying data in one line with the built-in styles (see :doc:`styles`). To see more detail on how a sonification is built from scratch using a single data series, try the :code:`SonifyingData1D.ipynb` Notebook. This demonstrates various methods of representing a single 1D dataset sonically, using a single :code:`Object`-type source representation. The code and instruction cells provide a step-by-step guide to setting up, rendering and saving a sonification with Strauss.

For a multivariate :code:`Event`-type sonification, the :code:`StarsAppearing.ipynb` notebook provides a step-by-step example, and demonstrates realistic stereo imaging for panoramic data. The output from this example was used in the `"Audio Universe: Tour of the Solar System" 2021 planetarium show <https://www.audiouniverse.org/education/shows/tour-of-the-solar-system>`_.

For a multivariate, multi-source example using an :code:`Object`-type source representation, see the :code:`PlanetaryOrbits.ipynb` Notebook, the output of which was also used in the "Audio Universe: Tour of the Solar System" planetarium show. An example of a bivariate data series sonification, described in `this paper <https://arxiv.org/abs/2311.16847>`_, can be found `here 
<https://data.ncl.ac.uk/articles/media/Trayford_2023_STRAUSS_ICAD_examples/22241182?file=39529129>`_.

In addition to the above-mentioned examples, there are a number of other Notebooks, each representing the diverse applications and uses of the Strauss code to sonify data in different ways. A more detailed overview of the example Notebooks and scripts can be found in  :ref:`examples`.
