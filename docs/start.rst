Getting Started
^^^^^^^^^^^^^^^

This walkthrough will take you through a clean install of the code, including optional dependencies and trying your first sonification

Installation
************

the Strauss code can be downloaded from **GitHub** at `the repository url <https://github.com/james-trayford/strauss.git>`_

Using :code:`git` make a copy of the STRAUSS repository via SSH,

.. code-block:: bash
  
  git clone git@github.com:james-trayford/strauss.git strauss

or HTTPS if you don't have SSH keys set up,

.. code-block:: bash

  git clone https://github.com/james-trayford/strauss.git strauss

throughout the documentation, I will refer to this as the **strauss repo** or **code directory**.

if you just want to use the code, STRAUSS may then be installed using pip, as

.. code-block:: bash
		
	cd strauss
	pip install .

If you want to develop the code, you can instead use

.. code-block:: bash
  
	pip install -e .

where the :code:`-e` option allows a local install, such that you can modify and run the source code on the fly without needing to reinstall each time.

Example jupyter notebooks
*************************

There are a number of example applications of Strauss in the :code:`example` subdirectory of the :code:`strauss` repo. These are in Python Notebook (:code:`.ipynb`) format for an interactive, step-by-step .

In order to run the exampes, first ensure that :code:`jupyter` is installed on your system. These were developed in :code:`jupyter-lab`, which can also be installed using pip, as.

.. code-block:: bash
  
	pip install jupyterlab

Then, running :code:`jupyter-lab` in the :code:`strauss` should initiate the :code:`jupyter-lab` server and open a browser window. Navigate to the :code:`examples` directory within the :code:`jupyter-lab` navigation plane, from which a number of examples can be opened and run interactively.

Running some examples
*********************

From the :code:`jupyter-lab` interface, a good starting point is the :code:`SonifyingData1D.ipynb` notebook. demonstrating various method of representing a single 1D dataset sonically, using a single :code:`Object`-type source representation. The code and instruction cells provide a step-by-step gude to setting up, rendering and saving a sonification with Strauss.

For a multivariate :code:`Event`-type sonification, the :code:`StarsAppearing.ipynb` notebook provides a step-by-step example, and demonstrates realistic stereo imaging for panoramic data. The output from this example was used in the `*Audible Universe* 2021 planetarium show <www.audibletour.net>`_.

For a multivariate, multi-source example using an :code:`Object`-type source representation, see ...

In addition to the above-mentioned examples, there are a number of other notebooks, each representing the diverse applications and uses of the Strauss code to sonify data in different ways. A more detailed overview of the example notebooks can be found in  :ref:`examples`.
