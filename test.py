import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from strauss.sonification import Sonification
from strauss import sources
from strauss.sources import Objects, Events

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 13}
plt.rc('font', **font)
plt.rc('figure', **{'figsize':[16, 9]})

def linearFunc(x,m,c):
    y = m * x + c
    return y

def gaussian(x, mu, sig):
    return (
        np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )

here = os.path.abspath("")

dat1 = pd.read_csv(f"{here}/data_schruba2011.txt", skiprows=16, header=None, delim_whitespace=True)
sfr, hi, h2 = np.array(dat1[3]), np.array(dat1[5]), np.array(dat1[7])
sig_sfr, sig_hi, sig_h2 = np.array(dat1[4]), np.array(dat1[6]), np.array(dat1[8])

bdx = (sfr > 0) & (h2 > 0) & (sig_sfr > 0)

a_fit,cov=curve_fit(linearFunc, np.log10(h2[bdx]), np.log10(sfr[bdx]), sigma=sig_sfr[bdx])

xfine = np.linspace(-0.5, 3, 90)
yfine = xfine*a_fit[0] + a_fit[1]

data = {'pitch':1.,
        'time_evo':xfine,
        'phi':(xfine*0.5+0.25) % 1,
        'theta':0.,
        'pitch_shift':yfine}

sources = Objects(data.keys())
sources.fromdict(data)
sources.apply_mapping_functions(map_lims={'pitch_shift':(yfine[0], yfine[-1])})


