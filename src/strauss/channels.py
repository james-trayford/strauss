""" :obj:`channels` submodule representing the output audio channels.

This submodule defines objects relevant to the output audio channels,
including the :obj:`audio_channels` which defines arrays of microphone
objects that are channeled to different speakers in the sonification
output.

"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
from scipy.special import lpmv, factorial
class mic:
    """Microphone / sound detector object

    This class represents a microphone, or sound-detector,
    corresponding to a particular output channel for audio
    spatialisation in the sonicfication.

    Args: 
      azimuth (:obj:`float`): Angular position of the microphone on the
    	horizontal plane, from 0 to 2pi with 0.5 pi being to the left
    	and 1.5 pi being to the right. In the special case of ambisonic,
    	this is instead an index corresponding to the *Ambisonic Channel
	Number* (ACN)
      mic_type (:obj:`str`): Type of microphone, choose from
    	:obj:`"directional"` (collects using a cardioid antenna pattern),
    	:obj:`"omni"` (collects sound from all directions equally) and
    	:obj:`"mute"` (collects no sound, useful for e.g. muting
    	auxillary channels)
      label (:obj:`str`): A label for the mic
      channel (:obj:`int`) The index of the channel, corresponding to
    	channel ordering in the output (starting from 0 e.g. stereo
    	L=0, R=1) 

    Raises:
      Exception: if mic_type not in allowed options
    """
    def __init__(self, azimuth, mic_type="directional", label="C", channel=1,  order=None, degree=None):
        self.azimuth = azimuth
        self.mic_type = mic_type
        self.label = label
        self.channel = channel
        
        if mic_type == "directional":
            self.antenna = lambda a, b=0.5*np.pi: 0.5*(1+np.cos(a-azimuth)*np.sin(b))
        elif mic_type == "omni":
            self.antenna = lambda a, b=0.5*np.pi: a**0.
        elif mic_type == "mute":
            self.antenna = lambda a, b=0.5*np.pi: a*0.
        elif mic_type == "ambisonic":
            self.antenna = self._ambisonic_antenna(acn=azimuth)
        else:
            raise Exception(f"Mic type \"{mic_type}\" unknown.")
        
    def _ambisonic_antenna(self, acn):        
        # get order and degree of spherical harmonic from ACN using ambiX standard
        # (see iem.kug.ac.at/fileadmin/media/iem/projects/2011/
        # ambisonics11_nachbar_zotter_sontacchi_deleflie.pdf, footnote 5)
        # with modified l expression protecting against zero division in acn == 0 case.
        l = int(acn**0.5)
        m = acn - l*(l+1)
        mabs = abs(m)

        # normalise using SN3D standard (see iem.kug.ac.at/fileadmin/media/
        # iem/projects/2011/ambisonics11_nachbar_zotter_sontacchi_deleflie.pdf,
        # equation 3)
        fctrl = factorial
        normSN3D = np.sqrt((2-(0**mabs)/4*np.pi) * fctrl(l-mabs)/fctrl(l+mabs))

        # trig function to use, dependent on sense of m (see ref eq 2)
        if m < 0:
            tfunc = np.sin
        if m >= 0:
            tfunc = np.cos
        
        # return lambda function, indexing correct spherical harmonic and
        # normalising to  SN3D normalisation (ref equation 2).
        # Note: additonal (-1)^mabs term needed to match ambiX given differing
        # assoc. Legendre polynom. definitions between ambiX and numpy. 
        return lambda a, b=0.5*np.pi: normSN3D * pow(-1,mabs) * lpmv(mabs, l, np.cos(b)) * tfunc(mabs*a)
        
    
class audio_channels:
    """Representing output audio channels.

    Data object representing the output channels of the sonification
    for preset common audio setups, or a custom setup. Stores an array
    of :obj:`mic` objects for each output channel.
    
    Args:
      setup (:obj:`str`): Type of audio setup. Supported options are
        :obj:`"mono"`, :obj:`"stereo"`, :obj:`"5.1"` and
        :obj:`"7.1"`, or :obj:`"custom"`. 
      custom_setup (:obj:`dict`): Dictionary defining a customised
    	audio setup, containing keys for :obj:`"azimuths"`, :obj:`"types"`
    	and :obj:`"labels"`, containing lists parameterising the first
    	three arguments of the :class:`mic` object, respectively
    	in the order of their channel index. Also optionally an forder
    	list to unscramble any channel order scrambling done by ffmpeg
    	processing in the sonification sae routines (this may need to
    	be found empirically, awaiting better multichannel save
    	routine).

    Raises:
	Exception: If custom requested but no parameters provided, or
    	  custom parameters provided without requesting a custom setup.

    """

    def __init__(self, setup="stereo", custom_setup={}):

        ##############################################
        # Channel properties for preset audio setups
        ##############################################
        
        # mic angles in radians
        mono_azimuths = [0.]
        stereo_azimuths = [0.5*np.pi, 1.5*np.pi]
        fivepoint_azimuths = [1./3*np.pi, 5./3*np.pi,
                            0., 0.,
                            2./3*np.pi, 4./3*np.pi]
        sevenpoint_azimuths = fivepoint_azimuths + stereo_azimuths

        # mic type, either omni(-directional) directional,
        # or muted (mute channels not used for spatialisation)

        mono_types = ["omni"]
        stereo_types = ["directional"] * 2
        fivepoint_types = ["directional"] * 2  + \
                          ["mute"] * 2 + \
                          ["directional"] * 2
        fivepoint_types = ["directional"] * 2  + \
                          ["mute"] * 2 + \
                          ["directional"] * 2
        sevenpoint_types = ["directional"] * 2  + \
                           ["mute"] * 2 + \
                           ["directional"] * 4
        
        # mic labels corresponding to each speaker
        mono_labels = ['C']
        stereo_labels = ['L', 'R']
        fivepoint_labels = ['FL', 'FR', 
                            'FC', 'LF',
                            'SL', 'SR']
        sevenpoint_labels = ['FL', 'FR', 
                             'FC', 'LF',
                             'SL', 'SR',
                             'AL', 'AR']

        #ffmpeg channel orders to correctly save combined files
        mono_forder = [0]
        stereo_forder = [0,1]
        fivepoint_forder = [1,2,0,3,4,5]
        sevenpoint_forder = [1,2,0,3,4,5,6,7]
        
        self.setup = setup

        if custom_setup and (setup != "custom"):
            warnings.warn("custom_setup variable non-empty, but not using custom setup. " \
                          "Did you mean to set setup=\"custom\"?")
        if  (not bool(custom_setup)) and (setup == "custom"):
            raise Exception("Custom setup requested but custom_setup parameters empty. " \
                            "Please provide setup dictionary to custom_setup")

            
        if setup == "mono":
            self.setup_channels(mono_azimuths, mono_types, mono_labels)
            self.forder = mono_forder
        elif setup == "stereo":
            self.setup_channels(stereo_azimuths, stereo_types, stereo_labels)
            self.forder = stereo_forder
        elif setup == "5.1":
            self.setup_channels(fivepoint_azimuths, fivepoint_types, fivepoint_labels)
            self.forder = fivepoint_forder
        elif setup == "7.1":
            self.setup_channels(sevenpoint_azimuths, sevenpoint_types, sevenpoint_labels)
            self.forder = sevenpoint_forder
        elif setup[:-1] == 'ambiX':
            # i.e. ambiX3 => 3rd order ambisonics.
            nchan = np.sum(2*np.arange(int(setup[-1])+1).astype(int) + 1)
            labfunc = lambda a, b: str(a)+str(b)
            self.setup_channels(np.arange(nchan, dtype='int'),
                                ['ambisonic']*nchan,
                                list(map(labfunc, ['C']*nchan, range(nchan))))
        elif setup == "custom":
            self.setup_channels(custom_setup['azimuths'],
                                custom_setup['types'],
                                custom_setup['labels'])
            if 'forder' in custom_setup:
                self.forder = self.custom_setup['forder']
            else:
                self.forder = 'unknown'
        else:
            raise Exception(f"setup \"{setup}\" not understood")
            

    def setup_channels(self, azimuths, types, labels, orders=None, degrees=None):
        """initialise audio channel setup for lists of properties

        Subroutine for setting up the audio_channels as arrays of
        :obj:`mic` objects, setting the :obj:`self.mics` list
        attribute to the :obj:`audio_channels`

        Args:
          azimuths (:obj:`list(float)`): list of :obj:`azimuth` values for
        	:obj:`mic` object
          types (:obj:`list(float)`): list of :obj:`mic_types` values
          	for :obj:`mic` object.
          labels (:obj:`list(float)`): list of :obj:`label` values for
          	:obj:`mic` object

        """
        self.azimuths = azimuths
        self.types = types
        self.labels = labels
        self.Nmics = len(azimuths)

        # Note: channel ordering important, sets output channel number
        self.channels = range(1, self.Nmics+1)
        
        self.mics = []
        
        for i in range(self.Nmics):
            microphone = mic(azimuths[i], types[i], labels[i], self.channels[i], orders, degrees)
            self.mics.append(microphone)

    def plot_antenna(self):
        """Plot antennae patterns for chosen audio setup

        Make a :obj:`matplotlib` figure object, representing a radial
        plot demonstrating the antennae patterns of each channel

        Returns:
          fig (:obj:`matplotlib.pyplot.figure`): figure object that
          can be shown or saved using the standard :obj:`matplotlib`
          routines. 
        """

        plot_azimuths = np.linspace(0, 2*np.pi, 100)
        normalised_volume = 0.*plot_azimuths

        # setup axes
        fig = plt.figure(figsize=(8,8))
        ax = fig.subplots(subplot_kw={'projection': 'polar'})
        shift = 0.

        for i in range(self.Nmics):
            labelpos = 1.2
            microphone = self.mics[i]
            antenna = microphone.antenna(plot_azimuths)
            normalised_volume += antenna
            p = ax.plot(plot_azimuths, np.clip(antenna,0, np.inf))
            p2 = ax.plot(plot_azimuths, np.clip(-antenna,0, np.inf),
                         c=p[0].get_color(), ls =':')
            if np.all(microphone.antenna(plot_azimuths) == 0.):
                labelpos += shift
                shift -= 0.15
            ax.scatter(microphone.azimuth, labelpos)
            ax.annotate(microphone.label+f" (ch. {microphone.channel})",
                         (microphone.azimuth, labelpos),
                         (0, 10), textcoords='offset points', ha='center')
        normalised_volume /= normalised_volume.max()
        plt.plot(plot_azimuths, normalised_volume, c='k', ls ='--', label='net amplitude')
        
        # configure axes
        ax = plt.gca()
        ax.set_theta_zero_location("N")
        ax.set_yticklabels([])
        ax.set_xlabel('azimuth')
        ax.set_ylim(0,1.4)
        ax.legend(frameon=0)
        ax.grid(True)
        ax.set_title(f"Audio setup: {self.setup}")

        # return figure object to manipulate, show or save
        return fig
            

        
if __name__ == "__main__":
    # can use setup values, mono, stereo, 5.1, 7.1 or custom
    setup = "5.1"
    if len(sys.argv) > 1:
        # read setup type from command line argument
        setup = sys.argv[1]
    ac = audio_channels(setup=setup)
    fig = ac.plot_antenna()
    plt.show()
