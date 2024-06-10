# STRAUSS
***S**onification **T**ools and **R**esources for **A**stronomers **U**sing **S**ound **S**ynthesis*

![Sonification Tools & Resources for Astronomers Using Sound Synthesis](/misc/strauss_logo.png "STRAUSS logo")

## Sonification and STRAUSS

*"Sonification"* is the process of conveying data via the medium of sound. Sonification can be used to make scientific data more accessible to those with visual impairments, enhance visualisations and movies, and even convey information more efficiently than by visual means. The *STRAUSS* python package is intended to make sonification simple for both scientific and outreach applications.

## Getting Started

Access the [full documentation here](https://strauss.readthedocs.io/) *(under construction!)* and read more about the associated [Audio Universe project here](https://www.audiouniverse.org/).

Make a copy of the *STRAUSS* repository via SSH,

`git clone git@github.com:james-trayford/strauss.git strauss`

or HTTPS if you don't have [SSH keys set up](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh),

`git clone https://github.com/james-trayford/strauss.git strauss`

*STRAUSS* may then be installed using pip

`cd strauss`

`pip install .`

For development purposes, you can instead use:

`pip install -e .`

where the `-e` option allows a local install, such that you can modify and run the source code on the fly without needing to reinstall each time. 

We recommend using a conda environment to avoid package conflicts. Type

`conda env create -f environment.yml`

before `pip install -e .`

and activate the environment with

`conda activate strauss`

## Acknowledgments
The *STRAUSS* code has benefited from funding via an [Royal Astronomical Society Eduaction & Outreach grant award](https://ras.ac.uk/awards-and-grants/outreach/education-outreach-small-grants-scheme), providing hardware and software for sound development and spatialisation testing.

