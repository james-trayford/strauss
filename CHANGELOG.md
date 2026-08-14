# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v1.5

### Added

- Timing tables, describing what sounded and when. `Events` sonifications have a single
  table of events, `Objects` sonifications one per source, via `Sonification.event_table()`
  and `.object_table(source)`, or `AudioFigure.get_table()` which picks the right one.
  `.fixed_table()` lists the parameters the user didn't map and the value each took.
  `AudioFigure.list_tables()` shows what's tabulated. Tables are built from the mapping,
  prior to render, and report what's heard: notes rather than pitch fractions, seconds
  rather than fractions of the duration and degrees rather than cycles.
- Sources can be named, via `sonify(source_names=[...])` or `Sources.names`, and their
  tables looked up by name. For `Events` give one name per input event, and they are
  thinned along with the data.
- `merge_mode`, a `Style` field also settable per `sonify` call, choosing how events
  merged by `max_notes_per_sec` take their values (see **Changed**).
- Table columns now have a nicely formatted name and can also carry a (square-bracketed)
  unit, which uses pandas 'multi-index' functionality to display nicely - e.g.
  `Cutoff Frequency`, `[Hz]`. These survive export to CSV, read back with `header=[0,1]`.
- Fixed table adds a `unit` column.
- Units are drawn from the generator `ranges` files if not specified elsewhere, then
  nothing for `pitch` (becomes `Note`). Haven't figured out spectraliser `spectrum` yet
  (unitless for now).
- Converters give the most intuitive representation of each parameter - `pan` as % right,
  `volume` as dB with `-inf` below -100 dB, `cutoff` in Hz, `time` in seconds. The cutoff
  conversion shares its frequency limits with `Stream.filt_sweep`, so the two can't drift.
- Custom rounding per parameter for table display, resolving each column's range into
  about a thousand steps - a tenth of a degree for angles, a thousandth of a cycle for the
  same angles in cycles. Times are never coarser than 10 ms, for syncing to video.

### Changed

- Events merged by `max_notes_per_sec` now take the values of the event closest to the
  middle of those merged, rather than the mean of them, so each remains a real data point.
  Merged events still sound at the mean time, so the maximum rate still holds. Use
  `merge_mode: 'average'` for the previous behaviour.
- Azimuthal angles are merged as directions rather than numbers, so that a cluster of
  events straddling the wrap point no longer merges to the opposite direction.
- `angle_unit` defaults to `'cycles'` rather than `'degrees'`, matching the mapped
  parameters themselves. Spatial angles are still reported in degrees in tables unless
  an `angle_unit` is asked for.
- Adaptive pitch binning no longer spreads a constant pitch across the chord in input
  order, instead binning it as `'uniform'` does. Affects `Events` sonifications with no
  `pitch` mapping.
- Generator `ranges` file fixes - `pan` had no entry at all and `volume_lfo/amount` no
  unit, the envelope segments declared limits of 20 s where mapping uses 10, `1e-2`
  parsed as a string rather than a float (YAML needs `1.0e-2`), and the `theta`/`phi`
  units were the wrong way round.
- dB conversion now lives in `utilities`, as `amplitude_to_db` and `db_to_amplitude`, and
  is shared with `AudioFigure._parse_level` - so mixing levels and reported volumes use
  one definition of the convention.

## v1.3

### Added

- Added `CHANGELOG.md`!
- `apply_fades` function in utilities to apply arbitrary fade in and fade out length to "de-click"
  (see **Changed**) a sample array, using a simple linear ramp in time.
- Added `.add_ticks(increment, duration=0.04, tick_vol=0.5)` method to the sonification. 
  This can be run after `.render()` and generates regular ticks in `'time'` or `'time_evo'`. 
  `increment` is specified in the input units, while duration is in seconds of the sonification
  (usually 0.01-0.1s), and `tick_vol` is a linear fraction of the peak sonification amplitude (remember,
  our hearing is generally logarithmic).

### Changed

- Fade in and out sonifications at display- or save-time to avoid clicks owing to sample discontinuity.
  Assumes a default 30ms, set via the new optional arg to `Sonification`, the `declick_time=0.03`.
- `Sonification.save` can now accept non-WAV file extensions, simply handing off the conversion to 
  `ffmpeg`. Requires `ffmpeg` installed - could add more specific directives for certain file types
  **Warning**: format conversions may not preserve channels! Surround formats may require more work.
- Sampler Improvements:
  - Some groundwork for sample `aliases` which will be used to call samples without an assigned pitch
  - `Sampler.info()` method giving a run down of how the `Sampler` is loaded with samples and aliases
  - a new `Sampler.fill_midi()` which is on by default, and fills all note keys in the midi-key pitch range
	(`C-1 - G9`)  with pitch-shifted equivalents of nearest samples
  - Allow loading of samples without needing specifically formatted note name in title - instead analyse pitch
	of sample and assign to nearest available note (TBC)

## To do (this version)
