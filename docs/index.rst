.. strauss
   documentation master file, created by
   sphinx-quickstart on Tue Oct 26 14:56:02 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the STRAUSS documentation!
=====================================

Strauss is a python toolkit for data *"sonification"* - the representation of data using sound - with both scientific and outreach applications.

The code aims to make rich and evocative sonification straightforward, with a number of presets and examples enabling a quick start. At the same time, it is intended to be flexible enough to allow high level of control over the sonification and various expressive elements of sound and harmony if required. The project is described in more detail in the paper `Introducing STRAUSS: a flexible sonification Python package <https://arxiv.org/abs/2311.16847/>`_, presented at th-e 28th Proceedings of the `International Community of Auditory Displays (2023) <https://repository.gatech.edu/entities/publication/6f73e8a9-ae9a-447f-a174-64ca28a2f3bb/full/>`_. You can read about the associated `Audio Universe project here <https://www.audiouniverse.org/>`_ for examples of using Strauss for a variety of applications.

Strauss is packaged with a variety of :doc:`example scripts <./examples>`. These mostly follow a `parameter mapping <https://sonification.de/handbook/chapters/chapter15/>`_ approach, but also other approaches, for example a `spectral audification <https://academic.oup.com/rasti/article/2/1/387/7209921>`_ approach in the `Spectral Data <./examples.html#sonifying-data-1d-sonifyingdata1d-ipynb>`_ example.

.. note::
   Strauss and its documentation is in continuous development, with more features in the pipeline. Follow the `GitHub repository <https://github.com/james-trayford/strauss>`_ for the cutting-edge release.

.. toctree::
   :maxdepth: 2

   motivation
   start
   elements
   params
   detailed
   examples
   todo
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
