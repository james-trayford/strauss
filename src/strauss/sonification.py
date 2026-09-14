""" :obj:`sonification`: generate sonification, combining submodules.

This Submodule handles the combining of all the constituent
subroutines into  a single :obj:`sonification` object that can then
render and output/save the resultant sonification. This handles
feeding of information between :obj:`strauss` modules, including
taking the :obj:`sources` mapping, applying any musical constraints
from :obj:`score` running the :obj:`generators` to make sound and
combining them into the output channels for the overall spatialised
sonificiation.

Todo:
  * Delegate more musical process to the :obj:`score` module
"""

from .stream import Stream
from .channels import audio_channels
from .sources import (Events, Objects, spatial_angles, display_name,
                      param_converters, param_lim_dict, quiet_db)
from .utilities import decimals_for_range
from .utilities import const_or_evo, nested_dict_idx_reassign, apply_fades, rescale_values, NoSoundDevice, is_notebook
from .utilities import write_audio, ffmpeg_layout, parse_level, MinPixelLocator
from .tts_caption import render_caption, get_ttsMode, default_tts_voice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
import os
import subprocess as sp
import wavio as wav
import IPython.display as ipd
from IPython.display import display
from scipy.io import wavfile
from scipy.interpolate import interp1d
import warnings
import tempfile
from pathlib import Path
import ffmpeg
try:
    import sounddevice as sd
except (OSError, ModuleNotFoundError) as sderr:
    sd = NoSoundDevice(sderr)
try:
    if is_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = list

class Sonification:
    """Representing the overall sonification

    This class combines the data sources, musical score constraints
    and generator together to generate and render the ultimate
    sonification for saving or playing in the :obj:`jupyter-notebook`
    environment 


    Todo:
      * Support custom audio setups here too.
    """
    def __init__(self, score, sources, generator, audio_setup='stereo',
                 caption=None, samprate=48000, declick_time=0.03,
                 ttsmodel=default_tts_voice, handle_nans=None):
        """
        Args:
         score (:class:`~strauss.score.Score`): Sonification :obj:`Score`
    	  object 
         sources (:class:`~strauss.sources.Source`): Sonification
    	  :obj:`Sources` child object (:class:`~strauss.sources.Events`
    	  or :class:`~strauss.sources.Objects`)  
         generator (:class:`~strauss.generator.Generator`): Sonification
    	  :obj:`Generator` child object
    	  (:class:`~strauss.generator.Synthesizer` or
    	  :class:`~strauss.generator.Sampler`)
         audio_setup (:obj:`str`) The requested audio setup preset to
    	  pass to :class:`~strauss.channels.audio_channels`
         samprate (:obj:`int`) Integer sample rate in samples per second
          (Hz), typically :obj:`44100` or :obj:`48000` for most audio
    	  applications
         declick_time (:obj:`float`) duration of start and end fades applied
          on save and dispolay to remove audible clicks from sample
          discontinuity
         ttsmodel (:obj:`str` or :obj:`PosixPath`) file path to the
          text-to-speech model used for captions.
         handle_nans (:obj:`str`) how to treat the audio any source's
          non-finite data corresponds to, either :obj:`'silent'` or
          :obj:`'interpolate'`. Taken from the :obj:`Sources` where not
          given, which is where the data itself is handled - see the
          `handle_nans` argument of
          :meth:`~strauss.sources.Source.__init__`.
        """

        # sampling rate in Hz
        self.samprate = samprate
        
        # tts model name
        self.ttsmodel = ttsmodel

        # fade duration to de-click audio
        self.declick_time = declick_time
        
        # caption
        self.caption = caption
        
        # sonification owns an instance of the Score
        self.score = score
        
        # sonification owns an instance of the Sources
        self.sources = sources

        # sonification owns an instance of the Generator
        self.generator = generator

        # the Sources handle the data, and so decide what is
        # interpolated and which events sound - from here these
        # instructions are taken to part-mute or drop sources.
        self.handle_nans = handle_nans or self.sources.handle_nans
        if handle_nans and (handle_nans != self.sources.handle_nans):
            warnings.warn(f"handle_nans='{handle_nans}' disagrees with the "
                          f"Sources value of '{self.sources.handle_nans}', "
                          "which has already been applied to the data. Pass " 
                          "to the Sources too for correct behaviour.")

        # set up the audio channel routing for the sonification
        self.channels = audio_channels(setup=audio_setup)

        # check Generator and Sonification sampling rates match...
        if self.samprate != self.generator.samprate:
            # if not, revert to Generator sampling rate.
            warnings.warn("warning: global and generator sampling rates disagree, " \
            f"reverting to generator value of {self.generator.samprate} Hz")
            self.samprate = self.generator.samprate
        
        # ...and the corresponding Stream objects 
        self.out_channels = {}
        for c in range(self.channels.Nmics):
            self.out_channels[str(c)] = Stream(self.score.length, self.samprate)

    def clear(self):
        """
        Clears the audio buffers in all output channels by setting values to 0.
        This prevents audio from accumulating if render() is called multiple times.
        """
        # Iterate over the dictionary of channels (usually '0', '1', etc.)
        for chan in self.out_channels:
            if hasattr(self.out_channels[chan], 'values'):
                self.out_channels[chan].values[:] = 0.
            # Fallback if the channel is a raw numpy array
            elif isinstance(self.out_channels[chan], np.ndarray):
                self.out_channels[chan][:] = 0.
            
    def _assign_notes(self):
        """Determine the note played by each source, and when.

        Combines the :obj:`Sources` `time` and `pitch` mappings with the
        :obj:`Score` chord sequence to decide which note each source
        sounds, and at what point in the sonification. Used by
        :meth:`render`, and by the table methods so that a table can be
        produced without rendering any audio.

        Note:
          As in :meth:`render`, sources with no `time` mapping are all
          assumed to start at zero and last the full sonification.

        Returns:
          notes (:obj:`list(str)`): note played by each source, in
            scientific pitch notation (e.g. :obj:`'A4'`)
          times (:obj:`ndarray`): start time of each source in seconds
        """
        # determine if time is provided, if not assume all start at zero
        # and last the duration of sonification
        if "time" not in self.sources.mapping:
            self.sources.mapping['time'] = [0.] * self.sources.n_sources
            self.sources.mapping['note_length'] = [self.score.length] * self.sources.n_sources

        # index each chord
        cbin = np.digitize(self.sources.mapping['time'], self.score.fracbins, 0)
        cbin = np.clip(cbin-1, 0, self.score.nchords-1)

        # pitch rank of each source divided by the number of sources
        pitch = np.asarray(self.sources.mapping['pitch'])
        pitchfrac = np.empty_like(pitch)
        if self.score.pitch_binning == 'adaptive' and np.unique(pitch).size > 1:
            idxs = np.argsort(pitch)
            pitchfrac[idxs] = np.arange(self.sources.n_sources)/self.sources.n_sources
        else:
            # a single pitch value has no ranking to adapt to - ranking it
            # would spread sources over the chord in whatever order they
            # arrived in, so bin it as a fixed pitch, as uniform binning does
            pitchfrac = np.clip(pitch, 0, 9.999999e-1)

        notes = []
        for source in range(self.sources.n_sources):
            chord = self.score.note_sequence[cbin[source]]
            nints = self.score.nintervals[cbin[source]]
            notes.append(chord[int(pitchfrac[source] * nints)])

        # mapped time is a fraction of the sonification length
        times = np.array(self.sources.mapping['time']) * self.score.length

        return notes, times

    def render(self, downsamp=1, progress=True):
        """Render the sonification.
        
        Generates the sonification by running the  Synthesizer
        :func:`~strauss.generator.Synthesizer.play` or Sampler
        :func:`~strauss.generator.Sampler.play` functions, and
        combining these into the output channel streams using any
        spatialisation for the specified
        :class:`~strauss.channels.audio_channels`. 

        Args:
          downsamp (optional, :obj:`int`): Optionally downsample
           sources for multi-source sonifications for a quicker test
           render by some integer factor.
        """

        # first, clear the audio channels
        self.clear()
        
        # determine the note played by each source and when it starts
        # (this also defaults the time mapping, if none was provided)
        notes, _ = self._assign_notes()

        # get some relevant numbers before iterating through sources
        Nsamp = self.out_channels['0'].values.size
        lastsamp = Nsamp - 1
        Nchan = len(self.out_channels.keys())
        indices = range(0,self.sources.n_sources, downsamp)

        if progress:
            print('Processing sonification..')
        mute_nans = (self.handle_nans == 'silent')

        for source in tqdm(indices) if progress else indices:

            # a source with no data at all has nothing to sound, so skip
            if mute_nans and self.sources.all_missing(source):
                continue

            # index note properties
            t = self.sources.mapping['time'][source]
            tsamp = int((Nsamp-1) * t)
            note = notes[source]

            # make dictionary for feeding to play function with each notes properties
            sourcemap = {}
            # for k in self.sources.mapping.keys():
            #     sourcemap[k] = self.soures.mapping[k][source]
            nested_dict_idx_reassign(self.sources.mapping, sourcemap, source)

            sourcemap['note'] = note

            # run generator to play each note
            sstream = self.generator.play(sourcemap)
            playlen = sstream.values.size

            # silence the stretches the source has no data for, ramping in
            # and out so the gaps don't click. Applied before spatialisation
            # so that it costs one multiply rather than one per channel
            if mute_nans:
                nlength = sourcemap.get('note_length',
                                        self.generator.preset.get('note_length'))
                if not isinstance(nlength, (int, float, np.number)) or (nlength <= 0):
                    # the sampler's 'sample' length runs as long as the sample
                    nlength = playlen/self.samprate
                knots = self.sources.mute_envelope(source,
                                                   self.declick_time/nlength)
                if knots is not None:
                    x, y = knots
                    # as with the evolving parameters, the curve is read at
                    # the sample fractions of the note, holding its end value
                    # through the release tail
                    mutenv = interp1d(x, y, bounds_error=False,
                                      fill_value=(y[0], y[-1]))
                    sstream.values = sstream.values * mutenv(sstream.sampfracs)

            # place source on listener plane (quarter rotation) by default
            polar = 0.5 * np.pi
            if 'pan' in sourcemap:
                # in pan mode, put everything on the 
                azi     = (const_or_evo(sourcemap['pan'], sstream.sampfracs) + 0.5) * np.pi
            else:
                # TODO: generic handling of alias parameters (beyond 3D angles)
                if 'phi' in sourcemap:
                    azi     = const_or_evo(sourcemap['phi'], sstream.sampfracs) * 2 * np.pi
                elif 'azimuth' in sourcemap:
                    azi     = const_or_evo(sourcemap['azimuth'], sstream.sampfracs) * 2 * np.pi
                else:
                    azi     = const_or_evo(self.generator.preset['azimuth'], sstream.sampfracs) * 2 * np.pi
                if 'theta' in sourcemap:
                    polar   = const_or_evo(sourcemap['theta'], sstream.sampfracs) * np.pi
                elif 'polar' in sourcemap:
                    polar   = const_or_evo(sourcemap['polar'], sstream.sampfracs) * np.pi                

            # compute sample indices for truncating notes overshooting sonification length
            trunc_note = min(playlen, lastsamp-tsamp)
            trunc_soni   = trunc_note + tsamp

            # spatialise audio by computing relative volume in each speaker
            for i in range(Nchan):
                panenv = self.channels.mics[i].antenna(azi,polar)
                self.out_channels[str(i)].values[tsamp:trunc_soni] += (sstream.values*panenv)[:trunc_note]

        # produce mono audio of caption, if one is provided
        if str(self.caption or '').strip():
            ttsMode = get_ttsMode() # determine if using coqui-ai or pyttsx3

            # use a temporary directory to ensure caption file cleanup
            with tempfile.TemporaryDirectory() as cdir:
                cpath = Path(cdir, 'caption.wav')
                render_caption(self.caption, self.samprate,
                               self.ttsmodel, str(cpath))
                rate_in, wavobj = wavfile.read(cpath)
                wavobj = np.array(wavobj)
            # Set up the Stream objects for TTS
            self.caption_channels = {}
            caption_norm = wavobj.max()
            for c in range(Nchan):
                self.caption_channels[str(c)] = Stream(wavobj.shape[0], self.samprate, ltype='samples')
                
                # place caption straight ahead spatially
                panenv = self.channels.mics[c].antenna(0, 0.5*np.pi)
                
                cnorm = abs(self.out_channels[str(c)].values).max()/caption_norm
                self.caption_channels[str(c)].values += (wavobj*cnorm*panenv)
        else:
            self.caption_channels = {}
            for c in range(Nchan):
                self.caption_channels[str(c)] = Stream(0, self.samprate) 


    def _check_can_tabulate(self):
        """Check the sources carry the mapped values a table needs."""
        if not getattr(self.sources, 'mapped_samples', {}):
            raise Exception("Sources have no mapped values to tabulate - run "
                            "'apply_mapping_functions' on the sources first.")

    def _param_unit(self, key):
        """The unit a mapped parameter is reported in.

        Units come from three places, in order: the table columns that
        replace a mapped parameter with what is heard (`time` in
        seconds, `note` in place of `pitch`), the units spatial angles
        are reported in, and otherwise the `<parameter>_unit` entries of
        the :obj:`Generator` ranges. Anything else, and anything the
        ranges call `unitless`, has no unit to report.

        Args:
          key (:obj:`str`): name of the mapped parameter, or of a table
            column

        Returns:
          unit (:obj:`str`): the unit, or an empty string where the
          quantity has none
        """
        if key in param_converters:
            # the value is converted for reporting, so takes the unit of
            # whatever it is converted into
            return param_converters[key][1]

        if key in spatial_angles:
            return self.sources.table_angle_unit or 'degrees'

        # otherwise ask the generator what it calls this parameter's units
        node = self.generator.preset.get('ranges', {})
        parts = key.split('/')
        for part in parts[:-1]:
            node = node.get(part, {})
            if not isinstance(node, dict):
                return ''
        unit = node.get(f'{parts[-1]}_unit', '')

        return '' if unit == 'unitless' else unit

    def _display_values(self, key, values):
        """Put mapped values into the terms they are reported in.

        A mapped value is not always the quantity worth reporting - a
        spatial angle is a fraction of a turn rather than an angle, and
        `pan` a fraction rather than a share of the output. Parameters
        with an entry in `param_converters` are converted by it, and
        spatial angles by the units angles are reported in. Anything
        else is already in the terms it is reported in.

        Args:
          key (:obj:`str`): name of the mapped parameter
          values (:obj:`array-like` or :obj:`float`): mapped values

        Returns:
          values (:obj:`array-like` or :obj:`float`): the values as they
          are reported
        """
        if key in param_converters:
            convert, _ = param_converters[key]
            return convert(values)

        return self.sources.in_angle_unit(key, values)

    def _display_range(self, key):
        """The range a parameter covers, as it is reported.

        Args:
          key (:obj:`str`): name of the mapped parameter

        Returns:
          span (:obj:`float`): the range it covers, or `None` where the
          parameter has no known limits
        """
        if key in ('time', 'time_evo'):
            return float(self.score.length)

        lims = self.sources.plims.get(key, param_lim_dict.get(key))
        if lims is None:
            return None

        lims = np.asarray(self._display_values(key, np.asarray(lims)),
                          dtype=float)

        return float(abs(np.diff(lims)[0]))

    def _round_numbers(self, table):
        """Round a table's numbers to what is worth reading.

        Each column is rounded to the decimal places that resolve its
        range into a thousand steps, so that a column of angles in
        degrees is given to a tenth of a degree and the same angles in
        cycles to a thousandth. Columns whose range is unknown, input
        data among them, are resolved by the values they hold. Times are
        never coarser than 10 ms, however long the sonification.

        Args:
          table (:obj:`pandas.DataFrame`): table to round

        Returns:
          table (:obj:`pandas.DataFrame`): the same table, rounded
        """
        for name in table.columns:
            # Avoid rounding booleans to e.g. 1.0
            if pd.api.types.is_bool_dtype(table[name]):
                continue
            if not pd.api.types.is_numeric_dtype(table[name]):
                continue

            values = table[name].to_numpy(dtype=float)

            span = self._display_range(name)
            if span is None:
                # no declared range, as for input data in the user's own
                # units, so resolve what the column holds
                finite = values[np.isfinite(values)]
                span = float(finite.max() - finite.min()) if finite.size else 0.

            decimals = decimals_for_range(span)
            if name in ('time', 'time_evo'):
                # times keep to the nearest 10 ms however long the
                # sonification, rather than coarsening with its length
                decimals = max(decimals, 2)

            table[name] = np.round(values, decimals)

        return table

    def _with_units(self, table):
        """Label a table's columns with the units of their contents.

        Columns become a two-level :obj:`pandas.MultiIndex` of name and
        unit, so that units travel with the table rather than living in
        its formatting - they survive `to_csv`, and are read back with
        `header=[0,1]`.

        Note:
          Columns holding no numerical quantity (e.g. `source`, `note`)
          and input data columns, whose units are the user's own, are
          labelled with an empty unit.

        Args:
          table (:obj:`pandas.DataFrame`): table with plain columns

        Returns:
          table (:obj:`pandas.DataFrame`): the same table, with name and
          unit columns
        """
        names, units = [], []
        for column in table.columns:
            if column.endswith('_input'):
                # input values are the data as given, in the user's own units
                key = column[:-len('_input')]
                names.append(f'{display_name(key)} (input)')
                units.append('')
            else:
                names.append(display_name(column))
                unit = self._param_unit(column)
                # bracket it, to read as a unit rather than as a second name
                units.append(f'[{unit}]' if unit else '')

        table.columns = pd.MultiIndex.from_arrays([names, units])

        return table

    def event_table(self, include_input=False):
        """Tabulate the events of the sonification.

        Produces a table with a row per event, giving the name of the
        source it represents, the time at which it sounds, the note
        played, and the value of each user-specified mapped parameter.
        Parameters added automatically or held at fixed values are
        excluded, and are instead listed by :meth:`fixed_table`.

        Note:
          Rows are ordered in time, whatever order the data was given in.
          The `time` and `note` columns replace any mapped `time` and
          `pitch` parameters, which are internal fractions rather than
          what is heard - `time` is the time of the event in the
          sonification in seconds, and `note` the note it ultimately
          sounds. Spatial angles are given in degrees, or in the
          sonification's `angle_unit` where one was asked for, rather
          than as mapped fractions.

        Args:
          include_input (`optional`, :obj:`bool`): if True, also give the
            input data value of each parameter, before mapping, in a
            column suffixed `'_input'`.

        Returns:
          table (:obj:`pandas.DataFrame`): one row per event
        """
        self._check_can_tabulate()
        if not isinstance(self.sources, Events):
            raise TypeError(f"'event_table' is for Events sources, but these "
                            f"sources are {type(self.sources).__name__}.")

        notes, times = self._assign_notes()

        # each event is a source, so is named by it
        table = {'source': self.sources.names,
                 'time': self._display_values('time', times),
                 'note': notes}

        # flag the events sounding using interpolated values
        if (self.sources.nan_mask is not None) and np.any(self.sources.nan_mask):
            table['missing'] = np.asarray(self.sources.nan_mask, dtype=bool)

        for key in self.sources.mapped_quantities:
            if self.sources.origin.get(key, 'mapped') != 'mapped':
                continue
            if key not in ('time', 'time_evo', 'pitch'):
                # time and pitch are already given by the time and note
                # of each row, in the terms actually heard
                table[key] = self._display_values(
                    key, np.asarray(self.sources.mapped_samples[key]))
            if include_input:
                table[f'{key}_input'] = np.asarray(self.sources.raw_mapping[key])


        table = pd.DataFrame(table).sort_values('time', kind='stable',
                                                ignore_index=True)

        return self._with_units(self._round_numbers(table))

    def _resolve_source(self, source=None):
        """Resolve a source name or index, defaulting to a lone source.

        Args:
          source (`optional`, :obj:`str` or :obj:`int`): name or index of
            the source. Can be omitted where there is only one.

        Returns:
          index (:obj:`int`): index of the source

        Raises:
          KeyError: if omitted where there is more than one source.
        """
        if source is None:
            if self.sources.n_sources > 1:
                raise KeyError("Sonification has more than one source, so a "
                               "'source' is needed. Choose from: "
                               f"{self.sources.names}")
            return 0

        return self.sources.source_index(source)

    def object_table(self, source=None, include_input=False):
        """Tabulate the evolution of one object of the sonification.

        Produces a table for a single source, with a row per point in
        its continuous evolution, giving the time and the value of each
        user-specified mapped parameter at that point. Parameters that
        do not evolve hold the same value down their column. Those the
        user did not map are excluded, and are instead listed by
        :meth:`fixed_table`.

        Note:
          Rows are ordered in time, whatever order the data was given in.
          As for :meth:`event_table`, `time` is given in seconds, and
          replaces the mapped `time_evo` parameter, and spatial angles
          are given in degrees, or in the sonification's `angle_unit`
          where one was asked for. The note the
          object plays, and its name, are given by the `note` and
          `source` entries of the table's `attrs`.

        Args:
          source (`optional`, :obj:`str` or :obj:`int`): name or index
            of the source, as in :attr:`Source.names`. Can be omitted
            where the sonification has only one source.
          include_input (`optional`, :obj:`bool`): if True, also give the
            input data value of each parameter, before mapping, in a
            column suffixed `'_input'`.

        Returns:
          table (:obj:`pandas.DataFrame`): one row per point in the
          object's evolution
        """
        self._check_can_tabulate()
        if not isinstance(self.sources, Objects):
            raise TypeError(f"'object_table' is for Objects sources, but these "
                            f"sources are {type(self.sources).__name__}.")

        index = self._resolve_source(source)

        # objects with nothing evolving have no time base, and so are a
        # single unchanging state
        if 'time_evo' in self.sources.mapped_samples:
            times = np.asarray(self.sources.mapped_samples['time_evo'][index])
            times = self._display_values('time', times * self.score.length)
        else:
            times = np.zeros(1)

        def _down_column(values):
            """Broadcast an unevolving value down the time column."""
            values = np.asarray(values)
            if values.ndim == 0:
                return np.broadcast_to(values, times.shape)
            return values

        table = {'time': times}

        # flag the points sounding at interpolated values
        if (self.sources.nan_mask is not None) and np.any(self.sources.nan_mask[index]):
            table['missing'] = _down_column(
                np.asarray(self.sources.nan_mask[index], dtype=bool))

        for key in self.sources.mapped_quantities:
            if self.sources.origin.get(key, 'mapped') != 'mapped':
                continue
            if key not in ('time', 'time_evo', 'pitch'):
                # time and pitch are already given by the time column and
                # the note of the object, in the terms actually heard
                table[key] = self._display_values(
                    key, _down_column(self.sources.mapped_samples[key][index]))
            if include_input:
                table[f'{key}_input'] = _down_column(self.sources.raw_mapping[key][index])


        table = pd.DataFrame(table).sort_values('time', kind='stable',
                                                ignore_index=True)
        table = self._with_units(self._round_numbers(table))

        notes, _ = self._assign_notes()
        table.attrs['source'] = self.sources.names[index]
        table.attrs['note'] = notes[index]

        return table

    def fixed_table(self, source=None):
        """Tabulate parameters the user did not map.

        Companion to :meth:`event_table`, listing the parameters the
        user did not map - those held at a fixed value, and those
        STRAUSS assigned itself where no mapping was given (`'fixed'`
        and `'auto'` in the `origin` column, respectively) - alongside
        the value each takes.

        Note:
          Across the whole sonification, parameters varying from source
          to source (e.g. the `pitch` assigned to each Object of a
          chord) have no one value to report, and are left out. Given a
          `source`, they take their value for that source, so are
          listed. Spatial angles are given in degrees, or in the
          sonification's `angle_unit` where one was asked for. A `pitch`
          is reported as the `note` it resolves to, being what is
          ultimately heard.

        Args:
          source (`optional`, :obj:`str` or :obj:`int`): name or index
            of a source, as in :attr:`Source.names`, to report values
            for. If omitted, values are reported for the sonification as
            a whole.

        Returns:
          table (:obj:`pandas.DataFrame`): one row per unmapped parameter
        """
        self._check_can_tabulate()

        index = None if source is None else self._resolve_source(source)

        rows = []
        for key in self.sources.mapped_quantities:
            origin = self.sources.origin.get(key, 'mapped')
            if origin == 'mapped':
                continue

            if key == 'pitch':
                # a mapped pitch is an internal fraction, of no use to the
                # reader - what is heard is the note it resolves to
                notes, _ = self._assign_notes()
                values = np.unique(notes if index is None else [notes[index]])
                values = values.astype(str).tolist()
                name = 'note'
            else:
                values = self.sources.mapped_samples[key]
                values = np.unique(np.asarray(values if index is None
                                              else values[index]))
                name = key

            if len(values) != 1:
                # not held at one value, so don't claim it is
                continue

            value = self._display_values(key, values[0])

            rows.append({'parameter': display_name(name),
                         'value': value if isinstance(value, str) else
                                  round(float(value),
                                        decimals_for_range(
                                            self._display_range(key) or 0.)),
                         # a row per parameter, so the unit is a column of its
                         # own rather than a second level of the column names
                         'unit': self._param_unit(name),
                         'origin': origin})

        return pd.DataFrame(rows, columns=['parameter', 'value', 'unit',
                                           'origin'])

    def _mapping_forward(self, key):
        """The function taking input data for `key` to what it sounds as.

        Composes the mapping the sources applied with the conversion
        the tables report in - seconds for time, degrees for angles,
        decibels for volume and so on - so the two describe the same
        thing. `pitch` is the exception, as what it sounds as depends
        on the score: it is taken to the fraction of the chord it
        selects, by rank for `'adaptive'` pitch binning.

        Args:
          key (:obj:`str`): the mapped parameter

        Returns:
          forward (:obj:`callable`): takes input values to reported ones
          unit (:obj:`str`): the unit they are reported in
        """
        to_param = self.sources.input_to_param(key)

        if key in ('time', 'time_evo'):
            length = float(self.score.length)
            return (lambda values: to_param(values) * length,
                    self._param_unit(key))

        if key == 'pitch' and self.score.pitch_binning == 'adaptive':
            # the chord fraction is the rank of the pitch among the
            # sources, so the same for any function preserving their order
            ranked = np.sort(np.concatenate([np.ravel(v) for v in
                                             self.sources.mapped_samples[key]]))
            fracs = np.arange(ranked.size) / ranked.size
            return (lambda values: np.interp(to_param(values), ranked, fracs),
                    '')

        if key == 'pitch':
            return to_param, ''

        def forward(values):
            shown = np.asarray(self._display_values(key, to_param(values)),
                               dtype=float)
            # nothing quieter than `quiet_db` is heard, and an axis can't
            # reach minus infinity to say so
            return np.where(np.isneginf(shown), quiet_db, shown)

        return forward, self._param_unit(key)

    @staticmethod
    def _monotone_functions(forward, lo, hi, samples=2001):
        """Sample a function over a range, and invert it if it is monotone.

        The pair returned describe the function as sampled: a mapping
        may be undefined past the data (the log of a negative value,
        say), so both hold their end values outside the range rather
        than returning nothing an axis can be drawn to.

        Args:
          forward (:obj:`callable`): function to sample and invert
          lo, hi (:obj:`float`): range of input to do so over
          samples (:obj:`int`): points to sample the function at

        Returns:
          sampled (:obj:`callable`): the function as sampled
          inverse (:obj:`callable`): takes its values back to the input,
          or `None` where the function turns back on itself over the
          range so there is no single input to return
        """
        if not np.isfinite([lo, hi]).all():
            return None
        if hi == lo:
            lo, hi = lo - 0.5, hi + 0.5

        grid = np.linspace(lo, hi, samples)
        with np.errstate(all='ignore'):
            values = np.asarray(forward(grid), dtype=float)
        # invert only where the function is defined
        finite = np.isfinite(values)
        if finite.sum() < 2:
            return None
        grid, values = grid[finite], values[finite]

        steps = np.diff(values)
        if np.all(steps >= 0):
            sign = 1.
        elif np.all(steps <= 0):
            sign = -1.
        else:
            return None

        if values[-1] == values[0]:
            # flat everywhere - nothing to invert
            return None

        sampled = lambda inputs: np.interp(np.asarray(inputs, dtype=float),
                                           grid, values)
        # `np.interp` wants a strictly increasing grid, so take the first
        # input reaching each value and drop the flat stretches
        signed, first = np.unique(sign*values, return_index=True)
        inverse = lambda shown: np.interp(sign*np.asarray(shown, dtype=float),
                                          signed, grid[first])

        return sampled, inverse

    def _secondary_axis(self, ax, key, which, data, min_pixels):
        """Add an axis showing what input data along one side sounds as.

        Args:
          ax (:obj:`matplotlib.axes.Axes`): axes to add to
          key (:obj:`str`): the mapped parameter shown on that side
          which (:obj:`str`): `'x'` or `'y'`, the side
          data (:obj:`ndarray`): input values plotted on that side
          min_pixels (:obj:`float`): closest ticks may sit on the page

        Returns:
          secondary (:obj:`matplotlib.axes.Axes`): the added axis, or
          `None` where the mapping is not monotone over the data so
          cannot be shown as a rescaling of the input axis
        """
        forward, unit = self._mapping_forward(key)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return None
        # invert over the data alone - past it an angle may wrap or a log
        # run out, and there is nothing there to label anyway
        functions = self._monotone_functions(forward, finite.min(), finite.max())
        if functions is None:
            return None
        forward, inverse = functions

        label = display_name(key) + (f' [{unit}]' if unit else '')
        if which == 'x':
            secondary = ax.secondary_xaxis('top', functions=functions)
            secondary.set_xlabel(label)
            axis = secondary.xaxis
            to_pixels = lambda ticks: ax.transData.transform(
                np.column_stack([inverse(ticks), np.zeros(np.size(ticks))]))[:, 0]
        else:
            secondary = ax.secondary_yaxis('right', functions=functions)
            secondary.set_ylabel(label)
            axis = secondary.yaxis
            to_pixels = lambda ticks: ax.transData.transform(
                np.column_stack([np.zeros(np.size(ticks)), inverse(ticks)]))[:, 1]

        if key == 'pitch':
            # a pitch selects a note from the chord, so mark the chord's
            # notes at the fraction each is chosen for - by name where
            # every chord is the same, else by where they fall
            chords = self.score.note_sequence
            nints = len(chords[0])
            if nints < 2:
                secondary.set_visible(False)
                return secondary
            centres = (np.arange(nints) + 0.5) / nints
            if all(list(c) == list(chords[0]) for c in chords):
                labels = [str(n) for n in chords[0]]
            else:
                labels = ['low'] + ['']*(nints-2) + ['high']
            axis.set_ticks(centres, labels)
            return secondary

        shown = np.asarray(forward(finite), dtype=float)
        axis.set_major_locator(MinPixelLocator(axis.get_major_locator(),
                                               to_pixels, min_pixels,
                                               shown.min(), shown.max()))
        return secondary

    def plot_mapping(self, show=True, panel_size=(8., 2.5), min_tick_pixels=12):
        """Plot each mapped parameter against the input data it came from.

        One panel per parameter the user mapped, showing the input data
        for it against the input data for time - or against source
        number where nothing is mapped to time. The far side of each
        axis then shows what the same values sound as: the time in the
        sonification in seconds along the top, and the parameter along
        the right in the terms the tables report it in - degrees for
        angles, decibels for volume, the notes of the chord for pitch.
        Where the mapping folds back on itself over the data (a polar
        angle over more than half a turn, say) there is no one input
        for each value it sounds as, so the parameter is instead drawn
        as a dashed line against its own axis.

        Note:
          Mapping functions and limits given to the sources are
          honoured, as are the notes of the score, so the plot shows
          the mapping as it sounds - it is the picture the tables
          describe.

        Args:
          show (`optional`, :obj:`bool`): display the figure once made
          panel_size (`optional`, :obj:`tuple`): width and height of each
            panel in inches
          min_tick_pixels (`optional`, :obj:`float`): closest two ticks
            of a converted axis may sit, on the page

        Returns:
          fig (:obj:`matplotlib.figure.Figure`): the figure
        """
        self._check_can_tabulate()
        sources = self.sources
        is_events = isinstance(sources, Events)
        origin = getattr(sources, 'origin', {})

        xkey = 'time' if is_events else 'time_evo'
        if xkey not in sources.mapped_quantities:
            xkey = None

        keys = [k for k in sources.mapped_quantities
                if k != xkey and origin.get(k, 'mapped') == 'mapped']
        if 'spectrum' in keys:
            # a spectrum is data across frequency rather than across time
            keys.remove('spectrum')
        if not keys:
            raise Exception("No mapped parameters to plot against the input.")

        width, height = panel_size
        fig, axes = plt.subplots(len(keys), 1, squeeze=False,
                                 figsize=(width, height*len(keys)))
        axes = axes[:, 0]

        # what the input for each source is plotted against
        if xkey is not None:
            xdata = [np.ravel(np.asarray(x, dtype=float))
                     for x in sources.raw_mapping[xkey]]
        else:
            xdata = [np.array([i], dtype=float) for i in range(sources.n_sources)]

        labelled_time = False
        for ax, key in zip(axes, keys):
            ydata = [np.ravel(np.asarray(y, dtype=float))
                     for y in sources.raw_mapping[key]]
            evolving = any(y.size > 1 for y in ydata)

            by_source = not is_events and not (evolving and xkey is not None)
            if is_events:
                # each event is a point, but still one input value per source
                ax.scatter(np.concatenate(xdata), np.concatenate(ydata), s=6)
                xshown = np.concatenate(xdata)
            elif not by_source:
                # objects evolve over time, so each is a line
                for x, y in zip(xdata, ydata):
                    ax.plot(x, y if y.size > 1 else np.full(x.size, y[0]))
                xshown = np.concatenate(xdata)
            else:
                # objects holding one value each have nothing to draw against
                # time, so go by source instead
                xshown = np.arange(sources.n_sources, dtype=float)
                ax.scatter(xshown, [y[0] for y in ydata], s=12)
                ax.set_xlabel('Source')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            yshown = np.concatenate(ydata)
            ax.set_ylabel(f'{display_name(key)} input')
            ax.tick_params(axis='both', direction='in')
            ax.ticklabel_format(useOffset=False, style='plain')

            secy = self._secondary_axis(ax, key, 'y', yshown, min_tick_pixels)
            if secy is None:
                # not a rescaling of the input, so draw the mapped values as
                # a line of their own
                forward, unit = self._mapping_forward(key)
                twin = ax.twinx()
                if is_events or by_source:
                    twin.scatter(xshown, forward([y[0] for y in ydata] if by_source
                                                 else yshown), marker='x', c='C3', s=12)
                else:
                    for x, y in zip(xdata, ydata):
                        twin.plot(x, forward(y if y.size > 1 else np.full(x.size, y[0])),
                                  ls='--', c='C3', lw=1)
                twin.set_ylabel(display_name(key) + (f' [{unit}]' if unit else ''),
                                color='C3')
                twin.tick_params(axis='y', colors='C3', direction='in')
            else:
                secy.tick_params(axis='y', direction='in')

            if xkey is not None and not by_source:
                secx = self._secondary_axis(ax, xkey, 'x', xshown, min_tick_pixels)
                if secx is not None:
                    secx.tick_params(axis='x', direction='in')
                    # the time in the sonification reads the same on every
                    # panel, so label it once, at the top
                    if labelled_time:
                        secx.set_xlabel('')
                        secx.set_xticklabels([])
                    labelled_time = True

        if xkey is not None:
            axes[-1].set_xlabel(f'{display_name(xkey)} input')

        if xkey is None:
            axes[-1].set_xlabel('Source')

        fig.tight_layout()
        self.mapping_figure = fig
        if show:
            if is_notebook():
                display(fig)
                plt.close(fig)
            else:
                plt.show()

        return fig

    def add_ticks(self, increment, duration=0.04, tick_vol=0.25):
        # TODO this should probably use a dedicated generator...

        # add tick volume to Sonification object
        self.tick_vol = tick_vol
        
        tick_samples = 2*(np.random.random(self.out_channels['0'].values.shape)-0.5)
        k = 'time'
        if k not in self.sources.lims.keys():
            k = 'time_evo'
            if k not in self.sources.lims.keys():
                raise Exception("""
                Sonification doesn't have a time base! only sonifications with a 'time'
                or 'time_evo' mapping can have time increment ticks...
                """)
        inc = self.score.length*rescale_values(self.sources.lims[k][0]+increment,
                                               self.sources.lims[k],
                                               self.sources.plims[k])
        self.t_per_inc = np.linspace(0, self.score.length/inc, tick_samples.shape[0])
        self.tdur_per_inc = inc/duration
        tickenv = np.clip(1/self.tdur_per_inc - self.t_per_inc%1, 0, np.inf)
        tickenv /= tickenv.max()
        tick_samples = tick_samples*tickenv
        Nchan = len(self.out_channels.keys())
        self.tick_channels = {}
        for i in range(Nchan):
            panenv = self.channels.mics[i].antenna(0, 0.5*np.pi)
            self.tick_channels[str(i)] = Stream(tick_samples.size, self.samprate, ltype='samples')
            self.tick_channels[str(i)].values += tick_samples * panenv

                
    def save_stereo(self, fname, master_volume=1.):
        """ Save stereo or mono sonifications
        
        Can use this function to save :obj:`"stereo"` or :obj:`"mono"`
        sonifications while avoiding ffmpeg processing.

        Args:
          fname (:obj:`str`) Filename or filepath
          master_volume (:obj:`str` or :obj:`float`) Amplitude of the
            largest volume peak, from 0-1, or a level in decibels as a
            string, e.g. :obj:`'-6 dB'`
        """
        master_volume = parse_level(master_volume)

        if len(self.out_channels) > 2:
            print("Warning: sonification has > 2 channels, only first 2 will be used. See 'save_combined' method.")

            
        # first pass - find max amplitude value to normalise output
        # and concatenate channels to list
        vmax = 0.
        channels = []
        for c in range(min(len(self.out_channels), 2)):
            vmax = max(
                abs(self.out_channels[str(c)].values.max()),
                abs(self.out_channels[str(c)].values.min()),
                vmax
            ) / master_volume
            
            # combine caption + sonification streams at display time
            channel_values = np.concatenate([self.out_channels[str(c)].values,
                                self.caption_channels[str(c)].values])   
            
            channels.append(channel_values)

        wav.write(fname,
                  np.column_stack(channels),
                  self.samprate, 
                  scale = (-vmax,vmax),
                  sampwidth=3)

        print("Saved.")


    def save_combined(self, fname, ffmpeg_output=False, master_volume=1.):
        """ Save render as a combined multi-channel wav file 
        
        Can use this function to save sonification of any audio_setup,
        using ffmpeg processing, and unscrampling to the correct
        channel order.

        Args:
          fname (:obj:`str`) Filename or filepath
          ffmpeg_output (:obj:`bool`) If True, print :obj:`ffmpeg`
            output to screen 
          master_volume (:obj:`str` or :obj:`float`) Amplitude of the
            largest volume peak, from 0-1, or a level in decibels as a
            string, e.g. :obj:`'-6 dB'`
        """
        master_volume = parse_level(master_volume)

        # setup list to house wav stream data 
        inputs = [None]*len(self.out_channels)

        # first pass - find max amplitude value to normalise output
        vmax = 0.
        for c in range(len(self.out_channels)):
            vmax = max(
                abs(self.out_channels[str(c)].values.max()),
                abs(self.out_channels[str(c)].values.min()),
                vmax
            ) / master_volume
            
        print("Creating temporary .wav files...")

        # combine caption + sonification streams at display time
        for c in range(len(self.out_channels)):
            tempfname = Path('.', f'.TEMP_{c}.wav')
            self.out_channels[str(c)].values += self.caption_channels[str(c)].values
            wav.write(tempfname, 
                      self.out_channels[str(c)].values,
                      self.samprate, 
                      scale = (-vmax,vmax),
                      sampwidth=3)
            inputs[self.channels.forder[c]] = ff.input(tempfname)
            
        print("Joining temporary .wav files...")
        (
            ff.filter(inputs, 'join', inputs=len(inputs), channel_layout=self.channels.setup)
            .output(fname)
            .overwrite_output()
            .run(quiet=~ffmpeg_output)
        )
        
        print("Cleaning up...")
        for c in range(len(self.out_channels)):
            Path('.', f'.TEMP_{c}.wav').unlink()
            
        print("Saved.")

    def save(self, fname, master_volume=1., embed_caption=True):
        """ Save render as a combined multi-channel wav file 
        
        Can use this function to save sonification of any audio_setup
        to a file. This first creates a 32-bit depth WAV using
        `scipy.io.wavfile`. If fname has a non-WAV extension, it then attempts
        conversion via ffmpeg, provided ffmpeg is available. Surround setups
        are additionally passed through ffmpeg where it is available, to
        record their channel layout in the output - which `scipy` cannot -
        and are 24-bit as a result.
        
        formats

        Args:
          fname (:obj:`str`) Filename or filepath
          master_volume (:obj:`str` or :obj:`float`) Amplitude of the
            largest volume peak, from 0-1, or a level in decibels below
            full scale as a string, e.g. :obj:`'-6 dB'`
          embed_caption (:obj:`bool`) Whether or not to embed caption
            at the start of the output audio

        Todo:
          * Raise `scipy` issue if common 24-bit WAV can be supported
        """

        channels = []
        vmax = 0.

        has_ticks = hasattr(self, 'tick_channels')

        # first pass - find max amplitude value to normalise output
        for c in range(len(self.out_channels)):
                
            channel_values = np.concatenate(int(embed_caption)*[self.caption_channels[str(c)].values,]+
                                            [apply_fades(self.out_channels[str(c)].values,
                                                         self.out_channels['0'].samprate,
                                                         fdur=self.declick_time)])
            channels.append(channel_values)
            vmax = max(
                abs(channels[c].max()),
                abs(channels[c].min()),
                vmax
            ) * 1.05

        # normalisation for conversion to int32 bitdepth wav, the
        # master_volume is applied on writing
        norm = (pow(2, 31)-1) / vmax

        # setup array to house wav stream data 
        chans = np.zeros((channels[0].size, len(channels)), dtype="int32")
        
        # normalise and collect channels into a list
        for c in range(len(self.out_channels)):
            signal = channels[c]*norm
            if has_ticks:
                # add the ticks
                signal += self.tick_channels[str(c)].values*norm*self.tick_vol
            chans[:,c] = (signal).astype("int32")
            
        # finally combine and write out file
        write_audio(fname, self.samprate, chans,
                    layout=ffmpeg_layout(self.channels.setup),
                    master_volume=master_volume)

        
    def notebook_display(self, show_waveform=True):
        """ plot the waveforms and embed player in the notebook

        Show waveforms and embed an audio player in the python
        notebook for direct playback. the notebook player only
        supports up to stereo, so if more than two channels, only the
        first two are used as left and right.
        """

        time = self.out_channels['0'].samples / self.out_channels['0'].samprate

        has_ticks = hasattr(self, 'tick_channels')
        channels = []
        fig = plt.figure(figsize=(18,12))
        vmax = 0.
        
        # combine caption + sonification streams at display time
        for c in range(len(self.out_channels)):
            # apply fades at display time
            channel_values = np.concatenate([self.caption_channels[str(c)].values,
                                             apply_fades(self.out_channels[str(c)].values,
                                                         self.out_channels['0'].samprate,
                                                         fdur=self.declick_time)])   
            channels.append(channel_values)
            vmax = max(
                abs(channels[c].max()),
                abs(channels[c].min()),
                vmax
            ) * 1.05
        
        if show_waveform:
            for i in range(len(self.out_channels)):
                plt.plot(time[::20], self.out_channels[str(i)].values[::20]+2*i*vmax, label=self.channels.labels[i])
            plt.xlabel('Time (s)')
            plt.ylabel('Relative Amplitude')
            plt.legend(frameon=False, loc=5)
            plt.xlim(-time[-1]*0.05,time[-1]*1.2)
            for s in plt.gca().spines.values():
                s.set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
            plt.show()
        
        if len(self.channels.labels) == 1:             
            # we have used 48000 Hz everywhere above as standard, but to quickly hear the sonification sped up / slowed down,
            # you can modify the 'rate' argument below (e.g. multiply by 0.5 for half speed, by 2 for double speed, etc)
            outfmt = np.column_stack(channels*2).T / vmax
        else:
            outfmt = np.column_stack(channels[:2]).T / vmax
        if len(self.channels.labels) > 2:
            print("Warning: for more than two channels, only first two channels are mapped to L and R, respectively.")
        if has_ticks:
            # add the ticks
            for c in range(outfmt.shape[0]):
                outfmt[c] += self.tick_channels['0'].values*self.tick_vol / vmax
        display(ipd.Audio(outfmt,rate=self.out_channels['0'].samprate, autoplay=False))
        
    def hear(self):
        """ Play audio directly to the sound device, for command-line playback.

        If available, use the ``sounddevice`` module to stream the sonification to
        the sound device directly (speakers, headphones, etc.) via the underlying
        ``PortAudio`` C-library. if unavaialable, raise error.

        Todo:
          * Add more options to control the streamed audio
        """

        channels = []
        vmax = 0.
        
        # combine caption + sonification streams at display time
        for c in range(len(self.out_channels)):
            channel_values = np.concatenate([self.caption_channels[str(c)].values,
                                             self.out_channels[str(c)].values])   
            channels.append(channel_values)
            vmax = max(
                abs(channels[c].max()),
                abs(channels[c].min()),
                vmax
            ) * 1.05
                
        if len(self.channels.labels) == 1:             
            # we have used 48000 Hz everywhere above as standard, but to quickly hear the sonification sped up / slowed down,
            # you can modify the 'rate' argument below (e.g. multiply by 0.5 for half speed, by 2 for double speed, etc)
            outfmt = np.column_stack(channels*2)/vmax
        else:
            outfmt = np.column_stack(channels[:2])/vmax

        dur = int(np.round(outfmt.shape[0]/self.out_channels['0'].samprate))
        playback_msg = f"Playing Sonification ({dur} s): "
        print(playback_msg)
        try:
            sd.play(outfmt,self.out_channels['0'].samprate,blocking=1)
        except OSError as error: 
            print(error) 
            print("The Sonification.hear() function requires the PortAudio C-library. This may be missing from your system or \n"
                  "unsupported in this context. This should be installed by pip on Windows and OSx automatically with the \n "
                  "sounddevice library, but on Linux you may need to install manually using e.g.:\n"
                  "\t 'sudo apt-get install libportaudio2.'\n")

    def _make_seamless(self, overlap_dur=0.05):
        """ Make a seamlessly looping audio signal.

        Audio signal is made seamless by cross-fading end of signal back into start
        over a duration (in seconds) defined by ``overlap_dur``

        Args:
          overlap_dur (:obj:`float`): cross-fade duration in seconds.        
        """
        self.loop_channels = {}
        buffsize = int(overlap_dur*self.samprate)
        ramp = np.linspace(0,1, buffsize+1)
        for c in range(len(self.out_channels)):
            self.loop_channels[str(c)] = Stream(self.out_channels[str(c)].values.size - buffsize,
                                                self.samprate, ltype='samples')
            self.loop_channels[str(c)].values = self.out_channels[str(c)].values[:-buffsize]
            self.loop_channels[str(c)].values[:buffsize] *= ramp[:-1]
            self.loop_channels[str(c)].values[:buffsize] += ramp[::-1][:-1] * self.out_channels[str(c)].values[-buffsize:]
            
    def _make_out_array(self, master_volume=1., embed_caption=True):
        channels = []
        vmax = 0.

        has_ticks = hasattr(self, 'tick_channels')

        # first pass - find max amplitude value to normalise output
        for c in range(len(self.out_channels)):
                
            channel_values = np.concatenate(int(embed_caption)*[self.caption_channels[str(c)].values,]+
                                            [apply_fades(self.out_channels[str(c)].values,
                                                         self.out_channels['0'].samprate,
                                                         fdur=self.declick_time)])
            channels.append(channel_values)
            vmax = max(
                abs(channels[c].max()),
                abs(channels[c].min()),
                vmax
            ) * 1.05

        # normalisation for conversion to int32 bitdepth wav
        norm = master_volume * (pow(2, 31)-1) / vmax

        # setup array to house wav stream data 
        chans = np.zeros((channels[0].size, len(channels)))
        
        # normalise and collect channels into a list
        for c in range(len(self.out_channels)):
            signal = channels[c]*norm
            if has_ticks:
                # add the ticks
                signal += self.tick_channels[str(c)].values*norm*self.tick_vol
            chans[:,c] = (signal)
        return chans
