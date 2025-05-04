---
title: 'STRAUSS: Sonification Tools \& Resources for Analysis Using Sound Synthesis'
tags:
  - Python
  - sonification
  - data inspection
  - astronomy
authors:
  - name: James W. Trayford
    orcid: 0000-0003-1530-1634
    corresponding: true # (This is how to denote the corresponding author)
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Samantha Youles
    orcid: 0000-0002-7520-5911
    affiliation: 1
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
  - name: Chris Harrison
    orcid: 0000-0001-8618-4223
    affiliation: 2
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
  - name: Rose Shepherd
    orcid: 0009-0002-0369-5146
    affiliation: 2
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
  - name: Nicolas Bonne
    orcid: 0000-0001-9569-8808
    affiliation: 1
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)

affiliations:
 - name: Institute of Cosmology and Gravitation, University of Portsmouth, Dennis Sciama Building, Burnaby Road, Portsmouth PO1 3FX, UK
   index: 1
 - name: School of Mathematics, Statistics and Physics, Newcastle University, NE1 7RU, UK
   index: 2
date: 15 January 2024
bibliography: paper.bib
---

# Summary

_Sonification_, or conveying data using non-verbal audio, is a relatively niche
but growing approach for presenting data across multiple specialist
domains including  astronomy, climate science, and beyond [@Lenzi21;
@Zanella22; @Lindborg23]. The
`strauss` Python package aims to provide such a tool that builds upon previous
approaches to provide a powerful means to explore different ways of expressing
data, with fine control over the output audio and its format.
`strauss` is a free, open source (FOSS) Python package, designed to allow flexible and
effective sonification to be straightforwardly integrated into data workflows,
in analogy to widely used visualisation packages.

The remit of `strauss` is broad; it is intended to be able to bridge between
_ad-hoc_ solutions for sonifying very particular datasets, and highly technical compositional
and sound-design tools that are not optimised for sonification, or may have a steep
learning curve. The code offers a range of approaches to sonification for a variety of
contexts (e.g. science education, science communication, technical data analysis, etc.).
To this end, `strauss` is packaged with a number of examples of different sonification
approaches, and preset configurations to support a _low-barrier, high-ceiling_ approach.
`strauss` has been used to produce both educational resources [@Harrison22], and
analysis tools [@Trayford23b].

# Statement of need

Sonification has great potential as a fundamental approach for interfacing with data.
This provides new perspectives on data that complement visual approaches, as
well as an accessible channel to data for those who cannot access visual
presentation, e.g. those who are blind or visually impaired
(BVI). As with the dominant approach of data
_visualisation_, what can constitute sonification is very broad, with different considerations for aspects
such as the audience, information content and aesthetics of the sonification. In
order for sonification to become more established and realise its potential as
a way of interfacing with data, accessible and flexible tools are needed to make
sonification intuitive and accessible to those who routinely work with data.

Unlike data _visualisation_, however, sonification of data is a far less developed
methodology, with a lack of widely adopted, cross-domain tools and interfaces for those dealing with data
to use. E.g., in the Python programming language, a number of packages
exist for visualising data, such as Matplotlib [@Hunter07], yt [@Turk11],
seaborn [@Waskom21], or Plotly [@plotly]. The lack of dedicated and accessible
tools for sonifying data is a barrier to exploring the approach. Most solutions
are piecemeal, using _ad-hoc_ tools to parse data and map it to properties of sound as
well as generating and outputting sound in different formats. What's more, many
of the tools being repurposed to sonify data are proprietary and platform-dependent, requiring
paid-for licenses.

A number of effective python packages for data sonification have emerged, e.g. `astronify`
[@Brasseur23] and `SonoUno` [@Casado24], but these are typically feature-limited
in how sonification is produced, namely continuous pitch-mapped sonification.

Realising the potential of sonification may require a "crowdsourced" approach; via
broad adoption of the technique and innovation in sonification approaches driven by
the scientific community. In this context, we identify a need for Python package that:

- Provides a full pipeline from data to sonified audio that can be integrated into the
workflows of scientists and data analysts.

- Is modular and enables complex, multi-variate sonification to be produced and fine tuned,
while being simple enough to produce simple sonifications, for instance for novice users or
in educational contexts.

- Is fully open-source and platform-independent, such that sonification is able to be
integrated more broadly into scientific practice.

`strauss` (**S**onification **T**ools and **R**esources for **A**nalysis **U**sing
**S**ound **S**ynthesis) is a Python package for data sonification. The code uses an
object-oriented structure to provide a clear conceptual framework for the different
components of a sonification [@Trayford23]. `strauss` is designed to be used by
analysts to fully sonify their data in a way analogous to a plotting pipeline, to be
analysed independently or in conjunction with visuals.

By choosing the mapping between the variables being communicated and the expressive properties
of sound, `strauss` is designed to be a _low-barrier, high-ceiling_ tool; providing
the means to make diverse and highly customised sonifications with a high degree of
low-level control as the user becomes more experienced, while providing a relatively
simple path to produce sonifications quickly for a novice. This is intended to
provide the bridge between typical analysts who know their data and the facets of
it that they want to communicate, and the sound experts or musicians that understand
how to express data with sound.

The `strauss` code minimises dependencies where possible, implementing built-in signal
generation and audio parsing and encoding based on low-level python libraries like
NumPy [@numpy] and SciPy [@scipy], as well as being tested on multiple platforms (MacOS, Linux,
Windows). This also provides means to interface with commonly used audio formats,
such as _"Soundfont"_ files (`.sf2`) to open a range of possibilities for representing
data using different instruments. This helps to ensure that the free and open source
status of `strauss` is maintained and does not require difficult to install or
proprietary software, that may themselves have steep learning curves.

Towards our _low-barrier_ aspirations for `strauss`, a tutorial-driven development (TDD)
approach is used where each major feature should have an associated tutorial example,
which are packaged with the code, in both Python Notebook (`examples/.ipynb`) and Python script
(`examples/*.py`) formats. `strauss` provides tools for inline playback of sound in both formats.
In addition, a further collection of modified examples is provided specifically for the
_Google Colab_ platform (`examples/colab/*.ipynb`), allowing examples to be run fully
in-browser without requiring a local install of the code.

Offering both formats for examples in `strauss` allows us to exploit the interactivity
and low technical threshold needed to use notebooks, while also having the BVI accessibility
of the raw-text scripts, given the difficulties of using notebooks with screen readers
[@Trayford24]. `strauss` is provided with extensive documentation, maintained and hosted via
[_Read The Docs_](https://strauss.readthedocs.io/en/latest/).

# Acknowledgements

JWT acknowledges support via the STFC Early Stage Research
& Development Grant, reference ST/X004651/1, CMH acknowledge funding from an United Kingdom
Research and Innovation grant (code: MR/V022830/1). RS is supported by a studentship from an
STFC Centre of Doctoral Training in Data Intensive Science (code: ST/W006790/1).

# References
