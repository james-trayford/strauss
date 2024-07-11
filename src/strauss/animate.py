import numpy as np
from matplotlib import pyplot as plt

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 13}
plt.rc('font', **font)
plt.rc('figure', **{'figsize':[16, 9]})

#class Animate:
#    """ """
    
#    def __init__(self):
    

def plot_mapping(x, y, data_param, sonif_param, lims, plims):
    """Create a plot with raw and mapped values"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    plt.title('Parameter Mapping',
	      fontweight="bold")
    ax.legend(loc=0)
    ax.set_xlabel(data_param)
    ax.set_ylabel(sonif_param)
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(plims[0], plims[1])
    plt.show()
    return

def plot_frames(x, y, xp, yp, key, lims, plims):
    """Create frames for animation"""
    
    plt.plot(x,y, c='blue')
    plt.scatter(xp, yp, marker = 'x', s=30,alpha=1, c='0.6')
    
    ax = plt.gca()
    ax.axvline(x[i], ls ='--', c='C0',lw=1.5, alpha=0.55)
    ax.axhline(y[i], ls ='--', c='C0',lw=1.5, alpha=0.55)
    plt.legend(frameon=1, loc=2)

    ax_twin = ax.twinx()
    ax.legend(loc=0)
    ax_twin.legend(loc=0)
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel(raw)
    ax_twin.set_ylabel(key)
    ax_twin.set_ylim(lims[0], lims[1])
    ax.set_ylim(plims[0], plims[1])
    plt.show()
    return

for key in map_lims:
    plot_mapping(x,y, data, key, maplims[key], param_lims[key])
