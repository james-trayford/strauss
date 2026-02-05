# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
