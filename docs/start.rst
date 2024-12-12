Getting Started
^^^^^^^^^^^^^^^

This walkthrough will take you through a clean install of the code, including optional dependencies and trying your first sonification. There are also example notebooks available on `Google Colab <https://colab.research.google.com/github/james-trayford/strauss/blob/colab_examples/>`_ which you can run without installing Strauss on your local system.

Installation
************

Strauss can be installed in three different ways, depending on whether you want to develop the code or simply use it as it is. It can be installed using pip install, with or without the option for development, or you can clone it from the GitHub repository.

if you just want to use the code, STRAUSS may then be installed using pip, as

.. code-block:: bash
		
	cd strauss
	pip install .

If you want to develop the code, you can instead use

.. code-block:: bash
  
	pip install -e .

where the :code:`-e` option allows a local install, such that you can modify and run the source code on the fly without needing to reinstall each time.

Alternatively, the Strauss code can be downloaded from **GitHub** at `the repository url <https://github.com/james-trayford/strauss.git>`_

Using :code:`git` make a copy of the STRAUSS repository via SSH,

.. code-block:: bash
  
  git clone git@github.com:james-trayford/strauss.git strauss

or HTTPS if you don't have SSH keys set up,

.. code-block:: bash

  git clone https://github.com/james-trayford/strauss.git strauss

throughout the documentation, I will refer to this as the **strauss repo** or **code directory**.

Example jupyter notebooks/scripts
*********************************

There are a number of example applications of Strauss in the :code:`example` subdirectory of the :code:`strauss` repo. These are in Python Notebook (:code:`.ipynb`) format for an interactive, step-by-step experience. They are also provided in Python script format (.py) in the :code:`examples` directory. The Python scripts can be run from the command line.

In order to run the notebook examples, first ensure that :code:`jupyter` is installed on your system. These were developed in :code:`jupyter-lab`, which can also be installed using pip, as.

.. code-block:: bash
  
	pip install jupyterlab

Then, running :code:`jupyter-lab` in the :code:`strauss` should initiate the :code:`jupyter-lab` server and open a browser window. Navigate to the :code:`examples` directory within the :code:`jupyter-lab` navigation plane, from which a number of examples can be opened and run interactively.

Running some examples
*********************

From the :code:`jupyter-lab` interface, a good starting point is the :code:`SonifyingData1D.ipynb` notebook. demonstrating various method of representing a single 1D dataset sonically, using both a single :code:`Object`-type source representation and an evolving :code:`Event`-type. You can read more about these sonification types in :ref:`elements`. The code and instruction cells provide a step-by-step gude to setting up, rendering and saving a sonification with Strauss.

For a multivariate :code:`Event`-type sonification, the :code:`StarsAppearing.ipynb` notebook provides a step-by-step example, and demonstrates realistic stereo imaging for panoramic data. The output from this example was used in the `*Audible Universe* 2021 planetarium show <https://www.audiouniverse.org/education/shows/tour-of-the-solar-system>`_.

For a multivariate, multi-source example using an :code:`Object`-type source representation, see ...

In addition to the above-mentioned examples, there are a number of other notebooks, each representing the diverse applications and uses of the Strauss code to sonify data in different ways. A more detailed overview of the example notebooks can be found in  :ref:`examples`.
