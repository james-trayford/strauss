"""STRAUSS (Sonification Tools and Resources for Analysis Using Sound Synthesis)

This module provides a toolkit for *sonification*, i.e. the representation of data using sound."""

from .audio_figure import AudioFigure, get_style
from contextvars import ContextVar

__version__ = "1.0.3"

_current_figure = ContextVar('current_figure', default=None)

def _get_current_figure():
    """
    Get the current AudioFigure object, or create one if there isn't one.

    Returns:
        AudioFigure: The AudioFigure object.
    """

    fig = _current_figure.get()

    if fig is None:
        fig = AudioFigure()
        _current_figure.set(fig)

    return fig

def sonify(*args, **kwargs):
    fig = _get_current_figure()
    return fig.sonify(*args, **kwargs)

def close():
    _current_figure.set(None)
    
def display():
    fig = _get_current_figure()
    fig.render()
    return fig.notebook_display(True)
    
def list_sonifications():
    fig = _get_current_figure()
    fig.list_sonifications()

def set_level(name, level):
    fig = _get_current_figure()
    fig.set_level(name, level)

def save(fname):
    fig = _get_current_figure()
    fig.save(fname)
