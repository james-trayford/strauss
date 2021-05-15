import numpy as np
import scipy.signal as sig

def LPF1(data, cutoff, q, order=5):
    b, a = sig.butter(order, cutoff, btype='low', analog=False)
    y = sig.lfilter(b, a, data)
    return y

def HPF1(data, cutoff, q, order=5):
    b, a = sig.butter(order, cutoff, btype='high', analog=False)
    y = sig.lfilter(b, a, data)
    return y

