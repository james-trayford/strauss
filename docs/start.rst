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

.. _tts-install:

Text-to-speech
**************

Fot text-to-speech (TTS) functionality, there are a couple of options: _system_ and _AI_ text to speech.

If you would like to use system AI, you will need to install the (as of writing) cutting-edge `pyttsx3` dependency, with 

.. code-block:: bash

   pip install --no-cache-dir --extra-index-url https://test.pypi.org/simple/ pyttsx3==2.99

If you would like to use AI text-to-speech instead, you can instead install strauss requesting the optional :code:`AI-TTS` dependency:

.. code-block:: bash

   pip install "strauss[AI-TTS]"

or, for an install from a local repository copy:

.. code-block:: bash

   pip install -e ".[AI-TTS]"

.. note::
   The AI TTS is currently supported for python version :code:`<= 3.12`.
   
Example jupyter notebooks/scripts
*********************************

There are a number of example applications of Strauss in the :code:`example` subdirectory of the :code:`strauss` repo. These are in Python Notebook (:code:`.ipynb`) format for an interactive, step-by-step experience. They are also provided in Python script format (.py) in the :code:`examples` directory. The Python scripts can be run from the command line.

In order to run the notebook examples, first ensure that :code:`jupyter` is installed on your system. These were developed in :code:`jupyter-lab`, which can also be installed using pip, as:

.. code-block:: bash

   pip install jupyterlab

Then, running :code:`jupyter-lab` in the :code:`strauss` should initiate the :code:`jupyter-lab` server and open a browser window. Navigate to the :code:`examples` directory within the :code:`jupyter-lab` navigation plane, from which a number of examples can be opened and run interactively.

Running some examples
*********************

From the :code:`jupyter-lab` interface, a good starting point is the :code:`SonifyingData1D.ipynb` Notebook. This demonstrates various methods of representing a single 1D dataset sonically, using a single :code:`Object`-type source representation. The code and instruction cells provide a step-by-step gude to setting up, rendering and saving a sonification with Strauss.

For a multivariate :code:`Event`-type sonification, the :code:`StarsAppearing.ipynb` notebook provides a step-by-step example, and demonstrates realistic stereo imaging for panoramic data. The output from this example was used in the `"Audio Universe: Tour of the Solar System" 2021 planetarium show <https://www.audiouniverse.org/education/shows/tour-of-the-solar-system>`_.

For a multivariate, multi-source example using an :code:`Object`-type source representation, see the :code:`PlanetaryOrbits.ipynb` Notebook, the output of which was also used in the "Audio Universe: Tour of the Solar System" planetarium show. An example of a bivariate data series sonification, described in `this paper <https://arxiv.org/abs/2311.16847>`_, can be found `here 
<https://data.ncl.ac.uk/articles/media/Trayford_2023_STRAUSS_ICAD_examples/22241182?file=39529129>`_.

In addition to the above-mentioned examples, there are a number of other Notebooks, each representing the diverse applications and uses of the Strauss code to sonify data in different ways. A more detailed overview of the example Notebooks and scripts can be found in  :ref:`examples`.
