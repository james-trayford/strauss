.. _params:

Parameter Reference
###################

Here, we provide details of parameters that can be set for each of the generators via the preset ``.yml`` files or using the |mod-preset|_.

A subset of these parameters are also ``'mappable'``, i.e. can be controlled by input data via the ``Sources``:

.. |mod-preset| replace:: ``modify_preset()`` *Generator* method
.. _mod-preset: ./detailed.html#strauss.generator.Generator.modify_preset

.. exec_code::

   # --- hide: start ---
   from strauss.sources import mappable

   for m in mappable:
     # filter deprecated variable names
     if m is not ('phi' or 'theta'):
       if 'time' not in m:
         print(m)
   # --- hide: stop ---

While a further subset of these are also ``'evolvable'``, i.e. can be evolved over time using an ``Object``-type source:

.. exec_code::
   
   # --- hide: start ---
   from strauss.sources import evolvable
   
   for e in evolvable:
     # filter deprecated variable names
     if e is not ('phi' or 'theta'):
       if 'time' not in e:
         print(e)
   # --- hide: stop ---
     
When mapping data input into ``Sources`` to expressive sound parameters, two numerical ranges are relevant - the mapping limits of the input parameters, and the range of values the mapped parameter can assume - the ``map_lims`` and ``param_lims`` arguments of the Sources |mapping-func|_, respectively. 

The mapping limits, ``map_lims``, are the minimum and maximum allowed values of the input data. Data values above or below the range are clipped to the corresponding limits. These can either be absolute values, specified numerically (e.g. ``1.0``)  or percentiles, specified as a string (e.g. ``'100'``). The ``param_lims`` then define the minimum and maximum values of the chosen sound parameters that the data are rescaled between. For example if ``map_lims={'pitch_shift', [0,3]}`` and ``param_lims={'pitch_shift', [0,24]}``, data values of ``1.25``, ``2`` and ``3`` mapped to ``pitch_shift`` will result in pitch shifts of ``10``, ``16`` and ``24`` semitones, respectively.

.. |mapping-func| replace:: ``modify_preset()`` *Sources* method
.. _mapping-func: ./detailed.html#strauss.sources.Source.apply_mapping_functions

.. exec_code::
   :filename: yamls_to_tables.py

.. include:: tables.md
   :parser: myst_parser.sphinx_
