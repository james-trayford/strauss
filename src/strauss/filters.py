"""The :obj:`filters` submodule: containing audio filter functions

These are audio filters that can be applied to the audio signal in
frequency space to attenuate (filter out) frequencies. These can be
applied to individual ``Buffers`` as an evolvable parameter.

Todo:
  * Support More Filter Types
  * Implement resonance or `'Q'` variation
"""
import numpy as np
import scipy.signal as sig

def LPF1(data, cutoff, q, order=5):
    """
    Low-pass filter data array given cutoff, q and LPF order

    Args:
      data (array-like): Array containing signal for filtering
      cutoff (:obj:`float`): Cutoff frequency
      q (:obj:`float`): Filter quality-factor or 'Q' value
      order (:obj:`int`): polynomial order of filter function

    Return
      y (array-like): Filtered array for output
    """
    b, a = sig.butter(order, cutoff, btype='low', analog=False)
    y = sig.lfilter(b, a, data)
    return y

def HPF1(data, cutoff, q, order=5):
    """
    High-pass filter data array given cutoff, q and HPF order

    Args:
      data (array-like): Array containing signal for filtering
      cutoff (:obj:`float`): Cutoff frequency
      q (:obj:`float`): Filter quality-factor or 'Q' value
      order (:obj:`int`): polynomial order of filter function

    Return
      y (array-like): Filtered array for output
    """
    b, a = sig.butter(order, cutoff, btype='high', analog=False)
    y = sig.lfilter(b, a, data)
    return y

