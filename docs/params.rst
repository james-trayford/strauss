.. _params:

Parameter Reference
###################

Here, we provide details of parameters that can be set for each of the generators via the preset ``.yml`` files or using the |mod-preset|_.

A subset of these parameters are also ``'mappable'``, i.e. can be controlled by input data via the ``Sources``, with a smaller subset of these also being ``'evolvable'``, i.e. can be evolved continuosly over time using an ``Object``-type source. Both are listed in the :ref:`Mappable Parameters table <mappable-parameters>` below. 

Note: the legacy ``phi`` and ``theta`` are omitted as aliases of ``azimuth`` and ``polar``.

.. |mod-preset| replace:: ``modify_preset()`` *Generator* method
.. _mod-preset: ./detailed.html#strauss.generator.Generator.modify_preset

When mapping data input into ``Sources`` to expressive sound parameters, two numerical ranges are relevant - the mapping limits of the input parameters, and the range of values the mapped parameter can assume - the ``map_lims`` and ``param_lims`` arguments of the Sources |mapping-func|_, respectively. 

The mapping limits, ``map_lims``, are the minimum and maximum allowed values of the input data. Data values above or below the range are clipped to the corresponding limits. These can either be absolute values, specified numerically (e.g. ``1.0``)  or percentiles, specified as a string (e.g. ``'100%'``). The ``param_lims`` then define the minimum and maximum values of the chosen sound parameters that the data are rescaled between. For example if ``map_lims={'pitch_shift': [0,3]}`` and ``param_lims={'pitch_shift': [0,24]}``, data values of ``1.25``, ``2`` and ``3`` mapped to ``pitch_shift`` will result in pitch shifts of ``10``, ``16`` and ``24`` semitones, respectively.

.. |mapping-func| replace:: ``apply_mapping_functions()`` *Sources* method
.. _mapping-func: ./detailed.html#strauss.sources.Source.apply_mapping_functions

.. warning::

   Units differ depending on how a parameters are set, these conventions should be made clear by version 2.0, but currently arexs summarised here:

   * **Spatial angles:** input values for ``azimuth`` and ``polar`` are in ``angle_unit`` (cycles), whatever the tables below list. ``polar`` folds over a full turn, so ``0``, ``0.25`` and ``0.5`` give the zenith, horizon and nadir, with ``0.75`` and ``1`` folding back. The ``half-cycles`` unit and ``0 to 1`` limits listed for ``polar`` describe the value after folding. Giving ``map_lims`` for an angle skips the folding, and scales the data linearly between zenith and nadir instead. Angles cannot take ``param_lims``.
   * **Fixed values:** ``fix_*`` keywords to ``AudioFigure.sonify`` are fractions of the parameter's range, not its listed units, and are clipped to that range. For example, ``fix_pitch_lfo/freq`` of ``2`` gives 12 Hz, not 2 Hz. Spatial angles are the exception, and use ``angle_unit`` as above.
   * **Mapped values:** ``param_lims``, and ``output_range`` in a style, use the parameter's own units.

.. _mappable-parameters:

.. include:: tables.md
   :parser: myst_parser.sphinx_
