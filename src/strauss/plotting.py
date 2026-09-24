"""Plots and animations of how a sonification maps its data.

The mapping plot of :meth:`strauss.sonification.Sonification.plot_mapping`
and its animation, and the same across the columns of an
:class:`strauss.audio_figure.AudioFigure`. Each takes the sonification
(or the figure's sonifications) rather than living on it, so the
figure and the sonification share the one drawing.
"""
from .sources import Events, display_name, quiet_db, categorical
from .utilities import is_notebook, MinPixelLocator, EdgePrunedLocator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, FuncFormatter
import subprocess as sp
import IPython.display as ipd
from IPython.display import display
import warnings
import tempfile
from pathlib import Path


def _is_dark(ax):
    """Whether axes are drawn on a dark background."""
    from matplotlib.colors import to_rgb
    return sum(to_rgb(ax.get_facecolor())) < 1.5


def preview_animation(fig, draw, duration, preview_fps):
    """A `FuncAnimation` of `draw(t)` over `duration` seconds at
    `preview_fps`, shown in a notebook."""
    from matplotlib.animation import FuncAnimation
    frames = np.arange(0., duration, 1./preview_fps)
    anim = FuncAnimation(fig, draw, frames=frames, blit=False,
                         interval=1000./preview_fps)
    if is_notebook():
        display(ipd.HTML(anim.to_jshtml()))
        plt.close(fig)
    return anim


def stream_video(fig, draw, duration, fps, dpi, write_audio, fname,
                  ffmpeg_output=False):
    """Write a video of `draw(t)` over `duration` seconds, with the audio
    `write_audio(path)` saves.

    Frames go straight down a pipe to ffmpeg as raw pixels, with the
    audio muxed in the same pass, so nothing is written in between.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    fig.set_dpi(dpi)
    # the frame size as drawn - a figure of fractional pixel size is
    # rounded by the canvas, and every frame must match to the pixel or
    # each is read from where the last left off
    canvas.draw()
    height, width = np.asarray(canvas.buffer_rgba()).shape[:2]
    n_frames = int(round(duration*fps))
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp, 'audio.wav')
        write_audio(str(audio))
        command = ['ffmpeg', '-y', '-hide_banner',
                   '-loglevel', 'info' if ffmpeg_output else 'error',
                   '-f', 'rawvideo', '-pixel_format', 'rgba',
                   '-video_size', f'{width}x{height}', '-framerate', str(fps),
                   '-i', 'pipe:0', '-i', str(audio),
                   # the encoder needs even dimensions
                   '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                   '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                   '-c:a', 'aac', '-b:a', '320k', '-shortest', str(fname)]
        proc = sp.Popen(command, stdin=sp.PIPE)
        try:
            for i in range(n_frames):
                draw(i/fps)
                canvas.draw()
                proc.stdin.write(np.asarray(canvas.buffer_rgba()).tobytes())
        finally:
            proc.stdin.close()
            code = proc.wait()
    plt.close(fig)
    if code != 0:
        raise RuntimeError(f"ffmpeg failed while writing {fname}")


# the Okabe-Ito colours, less black, ordered so that neighbours in the
# cycle differ in lightness as well as hue. As they are for the dark
# background the plots default to, the blue is lightened to read on
# it and white takes the place of black, for eight in all
# clearance in inches between a figure's edges and anything drawn in it
MARGIN = 0.1

# height of the waveform panel, as a fraction of a mapping panel's
WAVEFORM_HEIGHT = 0.25

MAPPING_COLOURS = ['#56B4E9', '#E69F00', '#009E73', '#F0E442',
                   '#CC79A7', '#D55E00', '#4A9BE0']


def mapping_style(dark):
    """The style the mapping plot is drawn in.

    Args:
      dark (:obj:`bool`): whether on a dark background

    Returns:
      style (:obj:`list`): style sheets and settings for
      :func:`matplotlib.style.context`
    """
    from cycler import cycler
    settings = {'axes.prop_cycle': cycler(color=MAPPING_COLOURS),
                'font.size': 12}
    return (['dark_background', settings] if dark else [settings])


def _mapping_forward(soni, key):
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
    if key in categorical:
        raise NotImplementedError(
            f"'{key}' values are labels naming a sound, so there is no "
            "mapping from input values to plot.")

    to_param = soni.sources.input_to_param(key)

    if key in ('time', 'time_evo'):
        length = float(soni.score.length)
        return (lambda values: to_param(values) * length,
                soni._param_unit(key))

    if key == 'pitch' and soni.score.pitch_binning == 'adaptive':
        # the chord fraction is the rank of the pitch among the
        # sources, so the same for any function preserving their order
        ranked = np.sort(np.concatenate([np.ravel(v) for v in
                                         soni.sources.mapped_samples[key]]))
        fracs = np.arange(ranked.size) / ranked.size
        return (lambda values: np.interp(to_param(values), ranked, fracs),
                '')

    if key == 'pitch':
        return to_param, ''

    def forward(values):
        shown = np.asarray(soni._display_values(key, to_param(values)),
                           dtype=float)
        # nothing quieter than `quiet_db` is heard, and an axis can't
        # reach minus infinity to say so
        return np.where(np.isneginf(shown), quiet_db, shown)

    return forward, soni._param_unit(key)


def _extrapolating(xp, fp):
    """An interpolant over the points, continuing at the end slopes.

    A mapping clamps outside its limits, but an axis drawn from it
    must run on smoothly past the data to keep in step with the
    panel's own axis, which has margins beyond the data.

    Args:
      xp (:obj:`ndarray`): strictly increasing sample positions
      fp (:obj:`ndarray`): values at each

    Returns:
      function (:obj:`callable`): linear interpolation within the
      samples, and linear extrapolation beyond them
    """
    xp, fp = np.asarray(xp, dtype=float), np.asarray(fp, dtype=float)
    def function(x):
        x = np.asarray(x, dtype=float)
        y = np.interp(x, xp, fp)
        # slopes over a stretch at each end, past any samples the
        # mapping clamps there, so that it continues at its pace
        rising = np.flatnonzero(fp != fp[0])
        falling = np.flatnonzero(fp != fp[-1])
        if rising.size and falling.size:
            reach = 0.05*(xp[-1] - xp[0])
            i = max(rising[0], np.searchsorted(xp, xp[0] + reach))
            j = min(falling[-1], np.searchsorted(xp, xp[-1] - reach) - 1)
            i, j = min(i, xp.size - 1), max(j, 0)
            lo = (fp[i] - fp[0]) / (xp[i] - xp[0])
            hi = (fp[-1] - fp[j]) / (xp[-1] - xp[j])
            y = np.where(x < xp[0], fp[0] + lo*(x - xp[0]), y)
            y = np.where(x > xp[-1], fp[-1] + hi*(x - xp[-1]), y)
        return y
    return function


def _monotone_functions(forward, lo, hi, samples=2001):
    """Sample a function over a range, and invert it if it is monotone.

    The pair returned describe the function as sampled: a mapping
    may be undefined past the data (the log of a negative value,
    say), so both carry on at their end slopes outside the range
    rather than returning nothing an axis can be drawn to.

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

    # a mapping clamped at an end of the range is flat there, which
    # an axis can't follow - so the flat runs go, and the functions
    # continue at their pace instead
    rising = np.flatnonzero(values != values[0])[0]
    falling = np.flatnonzero(values != values[-1])[-1]
    keep = slice(rising, falling + 1)
    grid, values = grid[keep], values[keep]

    sampled = _extrapolating(grid, values)
    # the inverse wants a strictly increasing grid, so take the first
    # input reaching each value and drop the flat stretches
    signed, first = np.unique(sign*values, return_index=True)
    back = _extrapolating(signed, grid[first])
    inverse = lambda shown: back(sign*np.asarray(shown, dtype=float))

    return sampled, inverse


def _axis_functions(soni, key, data, shown=None):
    """The functions taking an input axis to what its values sound as.

    Args:
      key (:obj:`str`): the mapped parameter
      data (:obj:`ndarray`): input values plotted along the axis
      shown (`optional`, :obj:`ndarray`): what the tables say each
        of `data` sounds as, to check the functions against

    Returns:
      functions (:obj:`tuple`): the forward and inverse functions,
      or `None` where the mapping is not monotone over the data, or
      not a function of each value alone, so the axis cannot be
      shown as a rescaling of the input axis
    """
    forward, _ = _mapping_forward(soni, key)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    # invert over the data alone - past it an angle may wrap or a log
    # run out, and there is nothing there to label anyway
    functions = _monotone_functions(forward, finite.min(), finite.max())
    if functions is None:
        return None
    if shown is not None:
        # a mapping function may not act on each value alone - one
        # ranking the data, say - in which case the axis, made by
        # sampling it, would not show what the tables say sounded, so
        # is instead made from what they say
        expected = functions[0](data)
        shown = np.asarray(shown, dtype=float)
        ok = np.isfinite(shown) & np.isfinite(expected)
        span = np.ptp(shown[ok]) if ok.any() else 0.
        if ok.any() and not np.allclose(expected[ok], shown[ok],
                                        atol=1e-2*span + 1e-6, rtol=0):
            return _paired_functions(data[ok], shown[ok])

    return functions


def _paired_functions(data, shown):
    """Functions between an input axis and what sounded, from pairs.

    Interpolates between the values the tables give, for a mapping
    with no function of each value alone to sample.

    Args:
      data (:obj:`ndarray`): input values
      shown (:obj:`ndarray`): what each sounded as

    Returns:
      functions (:obj:`tuple`): forward and inverse functions, or
      `None` where the pairs don't run one way, so there is no
      single value for the axis to show
    """
    # one value per input, in input order
    inputs, first = np.unique(data, return_index=True)
    values = shown[first]
    if inputs.size < 2 or values[-1] == values[0]:
        return None

    steps = np.diff(values)
    if np.all(steps >= 0):
        sign = 1.
    elif np.all(steps <= 0):
        sign = -1.
    else:
        return None

    forward = _extrapolating(inputs, values)
    signed, first = np.unique(sign*values, return_index=True)
    back = _extrapolating(signed, inputs[first])
    inverse = lambda v: back(sign*np.asarray(v, dtype=float))

    return forward, inverse


def _secondary_axis(soni, ax, key, which, functions, data, min_pixels,
                    prune_top=0.):
    """Add an axis showing what input data along one side sounds as.

    Args:
      ax (:obj:`matplotlib.axes.Axes`): axes to add to
      key (:obj:`str`): the mapped parameter shown on that side
      which (:obj:`str`): `'x'` or `'y'`, the side
      functions (:obj:`tuple`): forward and inverse functions between
        the input axis and the parameter, from :func:`_axis_functions`
      data (:obj:`ndarray`): input values plotted on that side
      min_pixels (:obj:`float`): closest ticks may sit on the page,
        in points
      prune_top (:obj:`float`): fraction of a y axis from its top to
        leave without ticks, where a panel above would collide

    Returns:
      secondary (:obj:`matplotlib.axes.Axes`): the added axis, or
      `None` for a pitch where the chord changes over the
      sonification, so no note names a value along the axis
    """
    forward, inverse = functions
    _, unit = _mapping_forward(soni, key)
    label = display_name(key) + (f' [{unit}]' if unit else '')
    # positions in points rather than pixels, so the ticks chosen don't
    # change with the resolution the figure is drawn or saved at
    def to_points(pixels):
        return pixels*72./ax.figure.dpi

    if which == 'x':
        secondary = ax.secondary_xaxis('top', functions=functions)
        secondary.set_xlabel(label)
        axis = secondary.xaxis
        to_pixels = lambda ticks: to_points(ax.transData.transform(
            np.column_stack([inverse(ticks), np.zeros(np.size(ticks))]))[:, 0])
    else:
        secondary = ax.secondary_yaxis('right', functions=functions)
        secondary.set_ylabel(label)
        axis = secondary.yaxis
        to_pixels = lambda ticks: to_points(ax.transData.transform(
            np.column_stack([np.zeros(np.size(ticks)), inverse(ticks)]))[:, 1])
    secondary.tick_params(axis=which, direction='in')

    if key == 'pitch':
        # a pitch selects a note from the chord, so mark the chord's
        # notes at the fraction each is chosen for. With chords
        # changing over time the note depends on when a source sounds
        # too, so only an unchanging chord can be named along an axis
        chords = soni.score.note_sequence
        nints = len(chords[0])
        if nints < 2 or not all(list(c) == list(chords[0]) for c in chords):
            secondary.remove()
            return None
        centres = (np.arange(nints) + 0.5) / nints
        names = dict(zip(np.round(centres, 6), [str(n) for n in chords[0]]))
        # where the notes crowd the axis it can do without some names,
        # keeping those a line of text apart
        axis.set_major_locator(MinPixelLocator(FixedLocator(centres),
                                               to_pixels, 12))
        axis.set_major_formatter(FuncFormatter(
            lambda value, pos: names.get(round(value, 6), '')))
        return secondary

    finite = data[np.isfinite(data)]
    # ticks only where the data reaches - given a little tolerance,
    # so a round number at an end (a shift of 0, say) still shows
    # where the tables' rounding of the input leaves it a hair short
    shown = np.asarray(forward([finite.min(), finite.max()]), dtype=float)
    lo, hi = sorted(shown)
    tolerance = 5e-3*(hi - lo)
    lo, hi = lo - tolerance, hi + tolerance
    if which == 'y' and prune_top:
        # nothing within the margin of the top of the panel
        ylo, yhi = ax.get_ylim()
        edge = float(np.ravel(forward(yhi - prune_top*(yhi - ylo)))[0])
        if float(np.ravel(forward(yhi))[0]) >= float(np.ravel(forward(ylo))[0]):
            hi = min(hi, edge)
        else:
            lo = max(lo, edge)
    # labels along an x axis are wider than they are tall
    if which == 'x':
        min_pixels *= 2.5
    axis.set_major_locator(MinPixelLocator(None, to_pixels, min_pixels, lo, hi,
                                           log=_reads_log(forward, finite)))
    return secondary


def _reads_log(forward, data, samples=200):
    """Whether a mapping is better ticked by decade than evenly.

    A frequency mapped exponentially from the input, say, spreads
    its round values unevenly along the axis, where those of each
    decade sit evenly - so is ticked by decade where its logarithm
    follows the input more closely than the value itself does.

    Args:
      forward (:obj:`callable`): the mapping from the input
      data (:obj:`ndarray`): finite input values along the axis
      samples (:obj:`int`): points to sample the mapping at

    Returns:
      log (:obj:`bool`): True where decades read better
    """
    lo, hi = data.min(), data.max()
    if hi <= lo:
        return False
    grid = np.linspace(lo, hi, samples)
    with np.errstate(all='ignore'):
        values = np.asarray(forward(grid), dtype=float)
    ok = np.isfinite(values) & (values > 0)
    if ok.sum() < 3 or values[ok].max() / values[ok].min() < 10:
        return False
    grid, values = grid[ok], values[ok]
    if np.ptp(values) == 0:
        return False
    linear = abs(np.corrcoef(grid, values)[0, 1])
    log = abs(np.corrcoef(grid, np.log(values))[0, 1])
    return log > linear


def _pitch_limits(soni, functions, pad=0.25):
    """The input range worth showing for a pitch mapping.

    A pitch chooses among the notes of the chord, so input values
    far beyond those choosing the highest and lowest notes - the
    outliers of adaptively binned data, say - say nothing more
    about what sounds.

    Args:
      functions (:obj:`tuple`): forward and inverse functions between
        the input and the fraction of the chord it selects
      pad (:obj:`float`): fraction of the range between the lowest
        and highest notes to show beyond them

    Returns:
      limits (:obj:`tuple`): input values to limit the axis to
    """
    nints = len(soni.score.note_sequence[0])
    centres = np.array([0.5, nints - 0.5]) / nints
    lo, hi = functions[1](centres)
    margin = pad*(hi - lo)
    if lo > hi:
        lo, hi = hi, lo
        margin = -margin
    return lo - margin, hi + margin


def _shade_pitch_bins(soni, ax, functions):
    """Shade alternate pitch bins behind the data, to show them.

    Args:
      ax (:obj:`matplotlib.axes.Axes`): the pitch panel
      functions (:obj:`tuple`): forward and inverse functions between
        the input and the fraction of the chord it selects
    """
    nints = len(soni.score.note_sequence[0])
    edges = np.asarray(functions[1](np.arange(nints + 1) / nints), dtype=float)
    # beyond the outermost notes the mapping clamps, so those sound
    # right out to the panel's edges
    ylo, yhi = sorted(ax.get_ylim())
    if edges[0] < edges[-1]:
        edges[0], edges[-1] = ylo, yhi
    else:
        edges[0], edges[-1] = yhi, ylo
    for i in range(0, nints, 2):
        # a touch off the background, whichever it is
        shade = '0.15' if _is_dark(ax) else '0.95'
        ax.axhspan(*sorted(edges[i:i+2]), color=shade, lw=0, zorder=0)


def _outlined_line(ax, x, y, colour, label, ls='-', lw=1.):
    """Draw a thin line edged in the background colour.

    Lines laid over one another then stay distinct where they
    cross, as each is cut out from those below it.

    Args:
      ax (:obj:`matplotlib.axes.Axes`): axes to draw on
      x, y (:obj:`ndarray`): the line
      colour: its colour
      label (:obj:`str`): its legend label
      ls (:obj:`str`): its line style
      lw (:obj:`float`): its width in points, edged by as much again
        each side
    """
    from matplotlib import patheffects
    edge = patheffects.withStroke(linewidth=3*lw, foreground=ax.get_facecolor())
    return ax.plot(x, y, c=colour, ls=ls, lw=lw, label=label, path_effects=[edge])


def _clipped(values, limits, inset=0.03):
    """Clip values to a panel's limits, a little inside its edges.

    Args:
      values (:obj:`ndarray`): values along the axis
      limits (:obj:`tuple`): the axis limits, or `None` to leave
        the values as they are
      inset (:obj:`float`): fraction of the range inside the edges
        to clip to, so a marker there is drawn whole

    Returns:
      clipped (:obj:`ndarray`): the values clipped
      over, under (:obj:`ndarray`): which lay beyond the upper and
      lower limits
    """
    values = np.asarray(values, dtype=float)
    if limits is None:
        no = np.zeros(values.shape, dtype=bool)
        return values, no, no
    lo, hi = sorted(limits)
    margin = inset*(hi - lo)
    with np.errstate(invalid='ignore'):
        over, under = values > hi, values < lo
    clipped = np.where(over, hi - margin, np.where(under, lo + margin, values))
    return clipped, over, under


def _clipped_line(soni, ax, x, y, colour, limits, gap=0.02):
    """Clip a line to a panel's limits, and mark where it was.

    The stretches clipped are run along the panel's edge, with an
    arrow at intervals pointing the way they went.

    Args:
      ax (:obj:`matplotlib.axes.Axes`): the panel
      x, y (:obj:`ndarray`): the line
      colour: its colour
      limits (:obj:`tuple`): the y axis limits, or `None`
      gap (:obj:`float`): least distance between arrows, as a
        fraction of the x axis

    Returns:
      y (:obj:`ndarray`): the line's values clipped
    """
    y, over, under = _clipped(y, limits)
    step = gap*np.ptp(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.
    for oob, marker in ((over, '^'), (under, 'v')):
        if not oob.any():
            continue
        # an arrow every `step` along a clipped stretch, and one at
        # the start of each
        keep, last = [], -np.inf
        for i in np.flatnonzero(oob):
            if x[i] - last >= step or (i > 0 and not oob[i-1]):
                keep.append(i)
                last = x[i]
        ax.scatter(x[keep], y[keep], s=16, marker=marker, c=colour, zorder=3)
    return y


def _column(table, name):
    """A table column by name, whatever unit it carries.

    Args:
      table (:obj:`pandas.DataFrame`): a table with (name, unit)
        columns, as the table methods return
      name (:obj:`str`): the column's name

    Returns:
      values (:obj:`ndarray`): the column, or `None` if there is none
    """
    if name not in table.columns.get_level_values(0):
        return None
    return table[name].to_numpy().ravel()


def _is_constant(soni, key, tables, is_events):
    """Whether a parameter is unchanging over the sonification.

    Args:
      key (:obj:`str`): the mapped parameter
      tables (:obj:`list` of :obj:`pandas.DataFrame`): the tables of
        :func:`_mapping_tables`
      is_events (:obj:`bool`): whether the sources are `Events`

    Returns:
      constant (:obj:`bool`): True where every event shares one value,
      or no object's value evolves
    """
    name = display_name(key)
    columns = [_column(t, f'{name} (input)') for t in tables]
    columns = [c[np.isfinite(c)] for c in columns if c is not None]
    if is_events:
        columns = [np.concatenate(columns)] if columns else []
    return all(np.ptp(c) == 0 for c in columns if c.size)


def _mapping_tables(soni):
    """The tables the mapping plot draws from, one per line.

    Returns:
      tables (:obj:`list` of :obj:`pandas.DataFrame`): the event
      table alone for `Events`, or a table per source for `Objects`,
      each with its input columns
    """
    if isinstance(soni.sources, Events):
        return [soni.event_table(include_input=True)]
    return [soni.object_table(i, include_input=True)
            for i in range(soni.sources.n_sources)]


def plot_mapping(soni, show=True, panel_size=(4.5, 2.5), min_tick_points=20,
                 title=None, per_source=False, group_gap=0.35, dark=True,
                 waveform=False):
    """Draw :meth:`strauss.sonification.Sonification.plot_mapping`
    for the sonification `soni`, which documents the arguments."""
    spec = mapping_spec(soni, per_source)
    # a stack of panels per parameter, end to end as they share their
    # time axis, with a gap between one parameter's stack and the next
    with plt.style.context(mapping_style(dark)):
        fig = plt.figure(figsize=(panel_size[0],
                                  mapping_height(spec, panel_size, waveform)),
                         layout='constrained')
        info = draw_mapping(soni, fig, spec, panel_size, group_gap,
                            min_tick_points, title, waveform=waveform)
        fig.canvas.draw()
        fig.set_layout_engine('none')
        finish_mapping(info)

    fig.mapping = info
    soni.mapping_figure = fig
    if show:
        if is_notebook():
            display(fig)
            plt.close(fig)
        else:
            plt.show()

    return fig


def mapping_spec(soni, per_source=False):
    """Work out what :func:`plot_mapping` draws: the tables, the
    parameters with a panel, what's plotted against, and how each
    event or object is coloured. Returned as a dict for
    :func:`draw_mapping`."""
    soni._check_can_tabulate()
    sources = soni.sources
    is_events = isinstance(sources, Events)
    origin = getattr(sources, 'origin', {})
    tables = _mapping_tables(soni)
    names = list(sources.names)

    xkey = 'time' if is_events else 'time_evo'
    if xkey not in sources.mapped_quantities:
        xkey = None

    keys = [k for k in sources.mapped_quantities
            if k != xkey and origin.get(k, 'mapped') == 'mapped']
    if 'spectrum' in keys:
        # a spectrum is data across frequency rather than across time
        keys.remove('spectrum')
    # TODO: a categorical mapping wants a panel of its own, with the
    # labels down a discrete axis. The related job is to plot scored
    # aliases as pitch, ticking the axis with the chord's entries
    # ('kick', 'snare', ...) - which _secondary_axis already does, and
    # which needs a pitchless ordering of the aliases.
    categorical_keys = [k for k in keys if k in categorical]
    for k in categorical_keys:
        keys.remove(k)

    # a parameter that doesn't change over the sonification - the same
    # for every event, or not evolving for any object - has nothing to
    # draw against time, so is left to be reported instead
    constant = [k for k in keys if _is_constant(soni, k, tables, is_events)]
    keys = [k for k in keys if k not in constant]
    if constant:
        # TODO: report these in an info panel of the figure
        print("Constant over the sonification, so not plotted: "
              + ', '.join(display_name(k) for k in constant))
    if not keys:
        extra = ""
        if categorical_keys:
            extra = (" Mappings requesting sounds by name ("
                     + ', '.join(f"'{k}'" for k in categorical_keys)
                     + ") not supported in plots yet.")
        raise Exception("No mapped parameters that vary over the "
                        "sonification to plot against the input." + extra)

    # objects are lines over time - a parameter that doesn't evolve
    # is broadcast down their tables, so draws as a flat one - and
    # only go by source where there is no time to draw them against
    by_source = not is_events and xkey is None
    per_source = per_source and not is_events and not by_source
    n_panels = sources.n_sources if per_source else 1

    cycle = MAPPING_COLOURS
    def colour(i):
        # events share a colour, while each object's line takes the
        # next of the default cycle, so that the objects can be told apart.
        # Named outright, as 'C0' and so on would be read off the
        # default cycle once the figure is drawn outside the style
        return cycle[0] if is_events else cycle[i % len(cycle)]
    def style(i):
        # past the end of the colours, lines go dashed, then dotted
        return ['-', ':'][(i // len(cycle)) % 2]

    # a legend naming the objects by colour (events share a colour, so
    # need none)
    if not is_events and not per_source and sources.n_sources > 1:
        handles = [plt.Line2D([], [], color=colour(i), ls=style(i), label=names[i])
                   for i in range(sources.n_sources)]
    else:
        handles = None
    return {'keys': keys, 'tables': tables, 'xkey': xkey, 'names': names,
            'colour': colour, 'style': style,
            'is_events': is_events, 'by_source': by_source,
            'per_source': per_source, 'n_panels': n_panels, 'handles': handles}


def legend_shape(spec, width):
    """Columns of a spec's legend, as fit across a panel `width` inches
    wide, and the height in inches of the strip it takes across the
    top - of its own, so that the panels all keep the full width."""
    handles = spec['handles']
    if not handles:
        return 0, 0.
    cols = max(1, min(len(handles), int(width // 1.3)))
    rows = -(-len(handles) // cols)
    return cols, 0.22*rows + 0.15


def mapping_height(spec, panel_size, waveform=False):
    """Height in inches of the panels :func:`draw_mapping` draws for
    a spec, with their legend and any waveform panel."""
    return (panel_size[1]*len(spec['keys'])*spec['n_panels']
            + legend_shape(spec, panel_size[0])[1]
            + (WAVEFORM_HEIGHT*panel_size[1] if waveform else 0.))


def fit_panels(spec, max_panels, plot, name=None):
    """Keep the panels of a spec to `max_panels`, as animate readably.

    A panel per source asking for more falls back to one per
    parameter, drawn again by `plot(per_source=False)`, with a
    warning naming the sonification's `name`; more still raises.
    """
    n = len(spec['keys'])*spec['n_panels']
    if n > max_panels and spec['per_source']:
        # a tall stack of panels has its text shrunk past reading in
        # a video frame, so fall back to a panel per parameter
        spec = plot(per_source=False)
        what = f"'{name}'" if name else 'the sonification'
        warnings.warn(f"A panel per source would make {n} panels for {what}, "
                      f"more than the {max_panels} readable in a video, so "
                      f"animating the {len(spec['keys'])} panels of one per "
                      "parameter instead.", stacklevel=3)
        n = len(spec['keys'])
    if n > max_panels:
        what = f" for '{name}'" if name else ''
        raise Exception(f"{n} panels{what} are more than the {max_panels} "
                        "readable in a video: animate fewer parameters, "
                        "or raise max_panels.")
    return spec


def animate_mapping(soni, fname=None, fps=30, dpi=100, highlight=0.5,
                    preview_fps=10, max_panels=6, ffmpeg_output=False,
                    waveform=True, **kwargs):
    """Draw :meth:`strauss.sonification.Sonification.animate_mapping`
    for the sonification `soni`, which documents the arguments."""
    if fname is not None and not hasattr(soni, 'out_channels'):
        raise Exception("Render the sonification first, so the video "
                        "has its audio.")

    # settle the panels before drawing, so the figure is drawn once
    spec = fit_panels(
        mapping_spec(soni, kwargs.get('per_source', False)),
        max_panels,
        lambda per_source: mapping_spec(soni, per_source))
    kwargs['per_source'] = spec['per_source']
    fig = plot_mapping(soni, show=False, waveform=waveform, **kwargs)
    draw = mapping_animation(fig.mapping, highlight,
                             1./(preview_fps if fname is None else fps))

    if fname is None:
        return preview_animation(fig, draw, soni.score.length, preview_fps)
    stream_video(fig, draw, soni.score.length, fps, dpi, soni.save_stereo,
                  fname, ffmpeg_output)


def mapping_animation(info, highlight, window=1./30):
    """Add the moving parts of :func:`animate_mapping` to the panels
    of a drawn mapping `info`, and return the `draw(t)` that moves
    them to a time `t` of the sonification."""
    # the waveform panel shows the audio playing over each frame
    wave = info['waveform']
    if wave is not None:
        line, audio, samprate = wave
        n = max(int(round(window*samprate)), 2)
        line.set_data(np.linspace(0., 1., n), np.zeros(n))

    if not info['animate']:
        raise Exception("Nothing to animate: the sources have no time "
                        "to run the plot along.")
    # the playhead's place on the input axis at a time: the time
    # mapping run backwards, or failing that read off the data
    if info['to_input'] is not None:
        to_input = lambda t: float(np.ravel(info['to_input'](t))[0])
    else:
        order = np.argsort(info['seconds'])
        seconds, inputs = info['seconds'][order], info['inputs'][order]
        finite = np.isfinite(seconds) & np.isfinite(inputs)
        seconds, inputs = seconds[finite], inputs[finite]
        to_input = lambda t: float(np.interp(t, seconds, inputs))
    lo = to_input(0.)

    dark = _is_dark(info['axes'][0])
    head_colour = 'w' if dark else 'k'
    heads, points, markers = [], [], []
    for ax in info['axes']:
        heads.append(ax.axvline(lo, c=head_colour, lw=1.5, alpha=0.8, zorder=50))
        for artist, x, y, when in info['drawn'][ax]:
            if info['is_events']:
                points.append((artist, artist.get_sizes().copy(),
                               artist.get_facecolors().copy(), when))
            else:
                # a marker at the value sounding, and a line across at
                # its level to read it off the axis
                marker, = ax.plot([], [], marker='o', ms=6, ls='',
                                  c=artist.get_color(), mec=ax.get_facecolor(),
                                  mew=1., zorder=51)
                level = ax.axhline(np.nan, c=artist.get_color(), lw=1.2,
                                   ls='--', alpha=0.9, zorder=49)
                markers.append((marker, level, x, y, when))

    def draw(t):
        at = to_input(t)
        for head in heads:
            head.set_xdata([at, at])
        for artist, sizes, colours, when in points:
            # a swell at the note, fading over the highlight time
            dt = t - when
            weight = np.where(dt >= 0, np.exp(-dt/highlight),
                              np.exp(dt/(0.1*highlight)))
            weight = np.nan_to_num(weight)
            artist.set_sizes(sizes + 60.*weight)
            colours[:, 3] = 0.35 + 0.65*weight
            artist.set_facecolors(colours)
        for marker, level, x, y, when in markers:
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 1 and x[ok].min() <= at <= x[ok].max():
                # where the playhead crosses the line
                value = np.interp(at, x[ok], y[ok])
                marker.set_data([at], [value])
                level.set_ydata([value, value])
            else:
                marker.set_data([], [])
                level.set_ydata([np.nan, np.nan])
        if wave is not None:
            start = int(round(t*samprate))
            chunk = audio[start:start + n]
            line.set_ydata(np.pad(chunk, (0, n - chunk.size)))
        return heads

    return draw


def draw_mapping(soni, fig, spec, panel_size, group_gap, min_tick_points,
                 title, spare=0., waveform=False):
    """Draw the mapping panels of a `spec` into `fig`, a figure or a
    subfigure of one, as :func:`plot_mapping` lays them out.

    The figure's layout is left to run afterwards - draw the root
    figure, set its layout engine to `'none'`, then call
    :func:`finish_mapping` on the returned info. The panels stretch
    over `spare` inches.
    """
    sources = soni.sources
    keys, tables, xkey, names = (spec[k] for k in ('keys', 'tables', 'xkey', 'names'))
    colour, style = spec['colour'], spec['style']
    is_events, by_source, per_source, n_panels = (
        spec[k] for k in ('is_events', 'by_source', 'per_source', 'n_panels'))
    handles = spec['handles']

    width, height = panel_size
    legend_cols, legend_height = legend_shape(spec, width)
    root = fig.get_figure(root=True)
    # the gridspecs set the spacing, so the layout adds none of its own,
    # bar a margin clear of the figure's edges
    width_in, height_in = root.get_size_inches()
    root.get_layout_engine().set(
        h_pad=0, rect=(MARGIN/width_in, MARGIN/height_in,
                       1 - 2*MARGIN/width_in, 1 - 2*MARGIN/height_in))
    # the panels stretch remaining height
    panels_height = height*n_panels*len(keys) + spare
    rows, ratios = [], []
    strip = None
    if handles:
        rows.append('legend'); ratios.append(legend_height)
    rows.append('panels'); ratios.append(panels_height)
    if waveform:
        rows.append('waveform'); ratios.append(WAVEFORM_HEIGHT*height)
    if len(rows) > 1:
        # the strip sits close above the panels, whatever their spacing
        top = fig.add_gridspec(len(rows), 1, hspace=0.02, height_ratios=ratios)
        if handles:
            strip = fig.add_subplot(top[0])
            strip.axis('off')
            strip.legend(handles=handles, loc='center', ncol=legend_cols,
                         fontsize='small', frameon=False)
        outer = top[rows.index('panels')].subgridspec(len(keys), 1, hspace=group_gap)
        wave = _waveform_panel(soni, fig.add_subplot(top[rows.index('waveform')])) \
            if waveform else None
    else:
        outer = fig.add_gridspec(len(keys), 1, hspace=group_gap)
        wave = None
    groups = []
    for i in range(len(keys)):
        inner = outer[i].subgridspec(n_panels, 1, hspace=0)
        groups.append([fig.add_subplot(inner[j]) for j in range(n_panels)])
    axes = [ax for group in groups for ax in group]

    # what the input for each line is plotted against
    if xkey is not None:
        xdata = [_column(t, f'{display_name(xkey)} (input)') for t in tables]
    else:
        xdata = [np.arange(len(t), dtype=float) for t in tables]

    # what each panel drew, and when in the sonification each point
    # sounds, kept for the animation to pick up
    seconds = [_column(t, 'Time') for t in tables]
    drawn = {ax: [] for ax in axes}

    # the time axis runs over the span of the sonification alone,
    # where the mapping can say - the input beyond it is shown, but
    # sounds at an end, so is never reached by the animation
    xfunctions = xspan = None
    if xkey is not None and not by_source:
        xall = np.concatenate(xdata)
        xfunctions = _axis_functions(soni, xkey, xall)
        if xfunctions is not None:
            xspan = tuple(sorted(np.ravel(xfunctions[1]([0., soni.score.length]))))
            # the mapping clamps past the span, which an axis can't
            # follow - so the axis runs on at its pace instead, taken
            # from the span alone
            xfunctions = _axis_functions(soni, xkey, np.clip(xall, *xspan))

    labelled_time = False
    stacks = []
    for group, key in zip(groups, keys):
        name = display_name(key)
        unit = soni._param_unit(key)
        ydata = [_column(t, f'{name} (input)') for t in tables]
        shown = [_column(t, name) for t in tables]

        if by_source:
            # objects holding one value each have nothing to draw against
            # time, so go by source instead
            xs = [np.arange(sources.n_sources, dtype=float)]
            ys = [np.array([y[0] for y in ydata])]
            heard = [np.array([v[0] for v in shown])] if key != 'pitch' else None
        else:
            xs, ys = xdata, ydata
            heard = shown if key != 'pitch' else None
        xall, yall = np.concatenate(xs), np.concatenate(ys)

        # the parameter is shown as a rescaling of the input axis
        # where it can be, and in place of the input where it can't
        functions = _axis_functions(soni, 
            key, yall, None if heard is None else np.concatenate(heard))
        if functions is None and key != 'pitch':
            ys = heard
            yall = np.concatenate(ys)
            ylabel = name + (f' [{unit}]' if unit else '')
        else:
            ylabel = f'{name} input'

        # a pitch panel is limited to the input choosing the chord's
        # notes, so what lies beyond is drawn clipped to its edges
        limits = _pitch_limits(soni, functions) if (
            functions is not None and key == 'pitch') else None

        for j, ax in enumerate(group):
            if is_events:
                x = xs[0]
                colours = [colour(i) for i in range(len(x))]
                y, over, under = _clipped(ys[0], limits)
                points = ax.scatter(x, y, s=4, c=colours)
                drawn[ax].append((points, x, y, seconds[0]))
                for oob, marker in ((over, '^'), (under, 'v')):
                    if oob.any():
                        arrows = ax.scatter(x[oob], y[oob], s=16, marker=marker,
                                            c=[c for c, o in zip(colours, oob) if o])
                        drawn[ax].append((arrows, x[oob], y[oob], seconds[0][oob]))
            elif by_source:
                ax.scatter(xs[0], ys[0], s=10, c=[colour(i) for i in range(len(xs[0]))])
                ax.set_xlabel('Source')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            elif per_source:
                y = _clipped_line(soni, ax, xs[j], ys[j], colour(j), limits)
                line, = _outlined_line(ax, xs[j], y, colour(j), names[j], style(j))
                drawn[ax].append((line, xs[j], y, seconds[j]))
                ax.text(0.01, 0.95, names[j], transform=ax.transAxes,
                        ha='left', va='top', fontsize='small', color=colour(j),
                        bbox=dict(fc=ax.get_facecolor(), ec='none',
                                  alpha=0.8, pad=1.5))
            else:
                # objects evolve over time, so each is a line
                for i, (x, y) in enumerate(zip(xs, ys)):
                    y = _clipped_line(soni, ax, x, y, colour(i), limits)
                    line, = _outlined_line(ax, x, y, colour(i), names[i], style(i))
                    drawn[ax].append((line, x, y, seconds[i]))

            ax.tick_params(axis='both', direction='in')
            ax.ticklabel_format(useOffset=False, style='plain')
            if j < len(group) - 1:
                # panels of a stack share their time axis, so only the
                # last is labelled along the bottom
                ax.tick_params(axis='x', labelbottom=False)
            if j > 0:
                # panels butt together, so the top tick of each below
                # the first would collide with the one above's bottom
                ax.yaxis.set_major_locator(EdgePrunedLocator(
                    MaxNLocator(nbins='auto'), upper=0.1))

            if limits is not None:
                ax.set_ylim(*limits)
                _shade_pitch_bins(soni, ax, functions)

            if xspan is not None:
                secx = _secondary_axis(soni, ax, xkey, 'x', xfunctions, np.asarray(xspan),
                                            min_tick_points)
                # the time in the sonification reads the same on every
                # panel, so label it once, at the top
                if labelled_time:
                    secx.set_xlabel('')
                    secx.set_xticklabels([])
                labelled_time = True

        # one label centred on the stack, on its middle panel
        group[len(group)//2].set_ylabel(ylabel)
        stacks.append((group, key, functions, yall))

    # every panel spans the same time, and those of a stack the same
    # range of the parameter - limits shared, rather than the axes, so
    # each keeps its own ticks
    xlim = (min(ax.get_xlim()[0] for ax in axes), max(ax.get_xlim()[1] for ax in axes))
    for group in groups:
        ylim = (min(ax.get_ylim()[0] for ax in group),
                max(ax.get_ylim()[1] for ax in group))
        for ax in group:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

    # the converted axes go on once the limits are settled, so their
    # ticks can keep clear of the panel above
    for group, key, functions, yall in stacks:
        if functions is None:
            continue
        for j, ax in enumerate(group):
            secy = _secondary_axis(soni, ax, key, 'y', functions, yall,
                                        min_tick_points,
                                        prune_top=0.1 if j else 0.)
            # as with the input, the stack is labelled once, centrally
            if secy is not None and j != len(group)//2:
                secy.set_ylabel('')

    axes[-1].set_xlabel(f'{display_name(xkey)} input' if xkey is not None
                        else 'Source')

    if title:
        fig.suptitle(title, fontweight='bold')

    # what the animation needs: the panels, what each drew and when
    # each point of it sounds, and where the time runs on the input
    # axis - and what finishing the layout needs
    return {'figure': fig, 'axes': axes, 'groups': groups, 'drawn': drawn,
            'strip': strip, 'title': bool(title), 'waveform': wave,
            'is_events': is_events,
            'animate': xkey is not None and not by_source,
            'to_input': xfunctions[1] if xfunctions else None,
            'seconds': np.concatenate(seconds),
            'inputs': np.concatenate(xdata)}


def _waveform_panel(soni, ax):
    """Draw the sonification's waveform - its channels summed, scaled
    to peak at the panel's edges - in a bare panel `ax`.

    Returns:
      wave (:obj:`tuple`): the line, the scaled audio and its sample
      rate, for the animation to window frame by frame
    """
    audio = sum(np.asarray(c.values, dtype=float) for c in soni.out_channels.values())
    samprate = next(iter(soni.out_channels.values())).samprate
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    ax.axis('off')
    ax.set_xlim(0., 1.)
    ax.set_ylim(-1.05, 1.05)
    # the whole sonification, thinned to what the panel can show
    step = max(audio.size // 4000, 1)
    line, = ax.plot(np.linspace(0., 1., audio[::step].size), audio[::step],
                    lw=0.8, c=ax.yaxis.label.get_color(), alpha=0.9)
    return line, audio, samprate


def finish_mapping(info, title_gap=0.12):
    """Fix the positions of drawn mapping panels once the figure's
    layout has run and been switched off.

    The layout leaves a sliver between panels for their decorations,
    so fix the panels of each stack to butt together, and keep their
    side labels within the (sub)figure they're drawn in - and, under a
    title, give it `title_gap` inches of clearance.
    """
    fig, axes, groups = info['figure'], info['axes'], info['groups']
    if info['title']:
        # the layout sets the title close on the top panel, so make
        # room by drawing everything a little shorter from the bottom
        shrink = 1. - title_gap*fig.dpi/fig.bbox.height
        wave = info['waveform'][0].axes if info['waveform'] else None
        for ax in axes + [info['strip'], wave]:
            if ax is not None:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0*shrink, box.width, box.height*shrink])
    # panels share a time axis, so line up left to right whatever
    # their side decorations
    x0 = max(ax.get_position().x0 for ax in axes)
    x1 = min(ax.get_position().x1 for ax in axes)
    # and keep their side labels within the figure, where the widest
    # tick labels pushed them past it
    renderer = fig.canvas.get_renderer()
    pad = MARGIN*fig.dpi
    for ax in axes:
        ax.set_position([x0, ax.get_position().y0, x1 - x0, ax.get_position().height])
    fig.canvas.draw()
    # the converted axes are children, and their labels are what reach
    # furthest, so look at them all directly
    parts = [a for ax in axes for a in [ax, *ax.child_axes]]
    extents = [a.get_tightbbox(renderer) for a in parts]
    extents += [a.yaxis.label.get_window_extent(renderer) for a in parts
                if a.yaxis.label.get_text()]
    over = max(e.x1 for e in extents) + pad - fig.bbox.x1
    under = min(e.x0 for e in extents) - pad - fig.bbox.x0
    x1 -= max(over, 0)/fig.bbox.width
    x0 -= min(under, 0)/fig.bbox.width
    wave = info['waveform'][0].axes if info['waveform'] else None
    for ax in axes + ([wave] if wave else []):
        box = ax.get_position()
        ax.set_position([x0, box.y0, x1 - x0, box.height])
    for group in groups:
        if len(group) < 2:
            continue
        top = group[0].get_position().y1
        bottom = group[-1].get_position().y0
        height = (top - bottom) / len(group)
        for j, ax in enumerate(group):
            box = ax.get_position()
            ax.set_position([box.x0, top - (j + 1)*height, box.width, height])
        # the stack's label was placed clear of its own panel's tick
        # labels - move it out where another panel's are wider
        def in_view(ax, label):
            lo, hi = sorted(ax.get_ylim())
            return label.get_text() and lo <= label.get_position()[1] <= hi
        left = min(label.get_window_extent(renderer).x0
                   for ax in group for label in ax.get_yticklabels()
                   if in_view(ax, label))
        mid = group[len(group)//2]
        label = mid.yaxis.label
        extent = label.get_window_extent(renderer)
        if extent.x1 > left:
            # the label's right edge sits at its x coordinate
            box = mid.get_window_extent(renderer)
            pad = mid.yaxis.labelpad*fig.dpi/72
            mid.yaxis.set_label_coords((left - pad - box.x0)/box.width, 0.5)


def plot_columns(columns, show=True, panel_size=(4.5, 2.5), title=None,
                 dark=True, group_gap=0.35, per_source=False,
                 min_tick_points=20, waveform=False):
    """Draw the mapping plots of several sonifications side by side.

    Each of `columns`, a dict of sonifications by name, is drawn as
    :func:`plot_mapping` draws it into a subfigure of its own, titled
    by its name, with the panels the same size in every column - a
    shorter column leaves the rest blank below its panels. The figure
    gets `title` across the top. The other arguments are as for
    :meth:`strauss.sonification.Sonification.plot_mapping`.

    Returns:
      fig (:obj:`matplotlib.figure.Figure`): the figure, with the
      drawn info of each column in `fig.mapping`
    """
    specs = {n: mapping_spec(soni, per_source)
             for n, soni in columns.items()}
    fig = _draw_columns(columns, specs, panel_size, title, dark, group_gap,
                        min_tick_points, waveform)
    if show:
        if is_notebook():
            display(fig)
            plt.close(fig)
        else:
            plt.show()
    return fig


def _draw_columns(columns, specs, panel_size, title, dark, group_gap,
                  min_tick_points, waveform=False):
    """Draw the mapping panels of each sonification's spec in a
    subfigure of its own, side by side."""
    heights = {n: mapping_height(specs[n], panel_size, waveform) for n in columns}
    tallest = max(heights.values())
    with plt.style.context(mapping_style(dark)):
        fig = plt.figure(figsize=(panel_size[0]*len(columns), tallest),
                         layout='constrained')
        subfigs = np.atleast_1d(fig.subfigures(1, len(columns)))
        infos = []
        for subfig, (n, soni) in zip(subfigs, columns.items()):
            # a shorter column keeps its panels the size of the
            # others', leaving the rest blank below them
            infos.append(draw_mapping(soni, subfig, specs[n], panel_size,
                                      group_gap, min_tick_points, n,
                                      spare=tallest - heights[n],
                                      waveform=waveform))
        if title:
            fig.suptitle(title, fontweight='bold')
        fig.canvas.draw()
        fig.set_layout_engine('none')
        for info in infos:
            finish_mapping(info)
    fig.mapping = infos
    return fig


def animate_columns(columns, duration, write_audio, fname=None, fps=30,
                    dpi=100, highlight=0.5, preview_fps=10, max_panels=6,
                    ffmpeg_output=False, waveform=True, **kwargs):
    """Animate the mapping plots of several sonifications side by side.

    The figure of :func:`plot_columns`, with a playhead run across
    every column over `duration` seconds, as :func:`animate_mapping`
    runs one. `write_audio(path)` saves the audio for the video, and
    `max_panels` applies to each column. The other arguments are as
    for :meth:`strauss.sonification.Sonification.animate_mapping`,
    with `kwargs` passed to :func:`plot_columns`.
    """
    specs = {}
    for n, soni in columns.items():
        spec = mapping_spec(soni, kwargs.get('per_source', False))
        specs[n] = fit_panels(
            spec, max_panels,
            lambda per_source, soni=soni: mapping_spec(soni, per_source),
            name=n)
    fig = _draw_columns(columns, specs, kwargs.get('panel_size', (4.5, 2.5)),
                        kwargs.get('title'), kwargs.get('dark', True),
                        kwargs.get('group_gap', 0.35),
                        kwargs.get('min_tick_points', 20), waveform)
    window = 1./(preview_fps if fname is None else fps)
    draws = [mapping_animation(info, highlight, window) for info in fig.mapping]
    def draw(t):
        return [head for column in draws for head in column(t)]

    if fname is None:
        return preview_animation(fig, draw, duration, preview_fps)
    stream_video(fig, draw, duration, fps, dpi, write_audio, fname, ffmpeg_output)
