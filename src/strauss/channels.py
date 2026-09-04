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
    	:obj:`"vbap"` (collects only from the arc spanned by its two
    	neighbours on the speaker ring, using pairwise Vector Base
    	Amplitude Panning), :obj:`"omni"` (collects sound from all
    	directions equally) and :obj:`"mute"` (collects no sound, useful
    	for e.g. muting auxillary channels)
      label (:obj:`str`): A label for the mic
      channel (:obj:`int`) The index of the channel, corresponding to
    	channel ordering in the output (starting from 0 e.g. stereo
    	L=0, R=1)
      neighbours (:obj:`tuple(float)`): for :obj:`"vbap"` mics only, the
    	azimuths of the mics either side of this one on the speaker ring,
    	as computed by :meth:`audio_channels._vbap_ring`. :obj:`None` if
    	the mic has no neighbours to pan against.
      nring (:obj:`int`): for :obj:`"vbap"` mics only, the number of mics
    	on the speaker ring, used to spread sources evenly as they
    	approach the poles.

    Raises:
      Exception: if mic_type not in allowed options
    """
    def __init__(self, azimuth, mic_type="directional", label="C", channel=1,  order=None, degree=None,
                 neighbours=None, nring=1):
        self.azimuth = azimuth
        self.mic_type = mic_type
        self.label = label
        self.channel = channel

        if mic_type == "directional":
            self.antenna = lambda a, b=0.5*np.pi: 0.5*(1+np.cos(a-azimuth)*np.sin(b))
        elif mic_type == "vbap":
            self.antenna = self._vbap_antenna(azimuth, neighbours, nring)
        elif mic_type == "omni":
            self.antenna = lambda a, b=0.5*np.pi: a**0.
        elif mic_type == "mute":
            self.antenna = lambda a, b=0.5*np.pi: a*0.
        elif mic_type == "ambisonic":
            self.antenna = self._ambisonic_antenna(acn=azimuth)
        else:
            raise Exception(f"Mic type \"{mic_type}\" unknown.")

    def _vbap_antenna(self, azimuth, neighbours, nring):
        """Pairwise Vector Base Amplitude Panning antenna pattern.

        Unlike the cardioid used by the :obj:`"directional"` mics, which
        picks up sound from all but a single azimuth, this pattern is
        non-zero only over the arc between the mics either side of this
        one on the speaker ring. A source is therefore reproduced by just
        the two speakers that bracket it, giving a much sharper image.
        Gains are normalised so that the total power over the ring is
        unity at any azimuth.

        Args:
          azimuth (:obj:`float`): azimuth of this mic
          neighbours (:obj:`tuple(float)`): azimuths of the mics either
        	side of this one, or :obj:`None` if it has none
          nring (:obj:`int`): number of mics on the speaker ring

        Returns:
          antenna (:obj:`function`): lambda function of azimuth and polar
          angle, as for the other mic types
        """
        if neighbours is None:
            # nothing to pan against, so this mic carries the full signal
            return lambda a, b=0.5*np.pi: np.ones_like(np.asarray(a, dtype=float))

        # widths of the two sectors this mic contributes to, wrapped onto
        # the ring so that the mic azimuths need not be pre-normalised
        prev_azi, next_azi = neighbours
        wprev = (azimuth - prev_azi) % (2*np.pi)
        wnext = (next_azi - azimuth) % (2*np.pi)

        def pair_gain(w):
            """gain of this mic a distance d into a sector of width w"""
            if w >= np.pi:
                # the tangent law below is ill-conditioned for sectors of
                # half the circle or more, so fall back to interpolating
                # at constant power across the sector instead
                warnings.warn(f"vbap channel arc of {np.degrees(w):.0f} degrees is too "
                              "wide to pan across, falling back to a constant-power "
                              "crossfade. Consider adding channels or using "
                              "\"directional\" mics.")
                return lambda d: np.cos(0.5*np.pi*d/w)
            # 2D VBAP gains are sin(w-d) and sin(d), up to a common factor
            # of 1/sin(w) that cancels in the constant-power normalisation
            return lambda d: np.sin(w-d)/np.sqrt(np.sin(w-d)**2 + np.sin(d)**2)

        gprev = pair_gain(wprev)
        gnext = pair_gain(wnext)

        def antenna(a, b=0.5*np.pi):
            # angle to the source, wrapped onto [-pi, pi) about this mic
            d = (np.asarray(a, dtype=float) - azimuth + np.pi) % (2*np.pi) - np.pi
            # clipping to the sector zeroes sources outside it, as the
            # gain vanishes at the neighbouring mic's azimuth
            g = np.where(d < 0,
                         gprev(np.clip(-d, 0., wprev)),
                         gnext(np.clip(d, 0., wnext)))
            # the ring is horizontal, so it can only place the component of
            # the source lying in its plane, which carries a sin(b) share of
            # the amplitude. the out-of-plane remainder has no speakers to
            # place it, so it is spread evenly around the ring instead,
            # holding the total power constant as we go
            inplane = np.sin(b)**2
            return np.sqrt(inplane*g**2 + (1-inplane)/nring)

        return antenna

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
      panning (:obj:`str`): Optionally override the panning law used by
    	the directional channels of the setup, choosing from
    	:obj:`"vbap"` (pairwise panning, the default for :obj:`"5.1"` and
    	:obj:`"7.1"`) or :obj:`"cardioid"` (the broad antenna pattern
    	that is the default for :obj:`"stereo"`, and reproduces the
    	surround spatialisation of STRAUSS versions before 1.5).

    Raises:
	Exception: If custom requested but no parameters provided, or
    	  custom parameters provided without requesting a custom setup,
    	  or panning not in allowed options.

    """

    def __init__(self, setup="stereo", custom_setup={}, panning=None):

        ##############################################
        # Channel properties for preset audio setups
        ##############################################

        # store the setup
        self.setup = setup
        
        # mic angles in radians
        mono_azimuths = [0.]
        stereo_azimuths = [0.5*np.pi, 1.5*np.pi]
        fivepoint_azimuths = [1./3*np.pi, 5./3*np.pi,
                            0., 0.,
                            2./3*np.pi, 4./3*np.pi]
        sevenpoint_azimuths = fivepoint_azimuths + stereo_azimuths

        # mic type, either omni(-directional), directional, vbap
        # or muted (mute channels not used for spatialisation).
        # surround setups pan pairwise (vbap) rather than with the
        # broad cardioid of the directional mics, which would otherwise
        # spread every source over all of the speakers at once
        mono_types = ["omni"]
        stereo_types = ["directional"] * 2
        fivepoint_types = ["vbap"] * 2  + \
                          ["mute"] * 2 + \
                          ["vbap"] * 2
        sevenpoint_types = ["vbap"] * 2  + \
                           ["mute"] * 2 + \
                           ["vbap"] * 4

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
        self.panning = panning

        if custom_setup and (setup != "custom"):
            warnings.warn("custom_setup variable non-empty, but not using custom setup. " \
                          "Did you mean to set setup=\"custom\"?")
        if  (not bool(custom_setup)) and (setup == "custom"):
            raise Exception("Custom setup requested but custom_setup parameters empty. " \
                            "Please provide setup dictionary to custom_setup")

        # swap in the requested panning law for the directional channels
        # of the setup, leaving muted and omni channels as they are
        if panning is None:
            ptypes = lambda types: types
        elif panning in ("vbap", "cardioid"):
            newtype = "vbap" if panning == "vbap" else "directional"
            ptypes = lambda types: [newtype if t in ("vbap", "directional") else t
                                    for t in types]
        else:
            raise Exception(f"panning \"{panning}\" not understood, choose from " \
                            "\"vbap\" or \"cardioid\"")

        if setup == "mono":
            self.setup_channels(mono_azimuths, ptypes(mono_types), mono_labels)
            self.forder = mono_forder
        elif setup == "stereo":
            self.setup_channels(stereo_azimuths, ptypes(stereo_types), stereo_labels)
            self.forder = stereo_forder
        elif setup == "5.1":
            self.setup_channels(fivepoint_azimuths, ptypes(fivepoint_types), fivepoint_labels)
            self.forder = fivepoint_forder
        elif setup == "7.1":
            self.setup_channels(sevenpoint_azimuths, ptypes(sevenpoint_types), sevenpoint_labels)
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
                                ptypes(custom_setup['types']),
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

        # pairwise panning needs to know which mics bracket each mic on
        # the speaker ring, so this is resolved before they are made
        ring = self._vbap_ring(azimuths, types)

        self.mics = []

        for i in range(self.Nmics):
            microphone = mic(azimuths[i], types[i], labels[i], self.channels[i], orders, degrees,
                             neighbours=ring.get(i), nring=len(ring))
            self.mics.append(microphone)

    def _vbap_ring(self, azimuths, types):
        """Find the neighbours bracketing each vbap mic on the azimuth ring

        Subroutine for :meth:`setup_channels`, ordering the mics that pan
        pairwise by azimuth so that each one knows the azimuths of the mics
        either side of it, which is what defines the arc it collects over.

        Args:
          azimuths (:obj:`list(float)`): list of :obj:`azimuth` values for
        	each :obj:`mic` object
          types (:obj:`list(float)`): list of :obj:`mic_types` values for
        	each :obj:`mic` object

        Returns:
          ring (:obj:`dict`): the :obj:`(previous, next)` azimuth pair for
          each index in :obj:`azimuths` panning pairwise, unwrapped so that
          the mic's own azimuth falls between them. :obj:`None` for a mic
          with no neighbours to pan against.

        Raises:
          Exception: if two mics that pan pairwise share an azimuth, which
            leaves the pair to use ambiguous.
        """
        index = [i for i in range(len(types)) if types[i] == "vbap"]

        if len(index) < 2:
            # a lone speaker has no pair to pan across
            return {i: None for i in index}

        angles = {i: azimuths[i] % (2*np.pi) for i in index}
        index.sort(key=lambda i: angles[i])

        if len(set(np.round(list(angles.values()), 9))) < len(index):
            raise Exception("Two \"vbap\" channels share an azimuth, so the speaker " \
                            "pair to pan a source across is ambiguous. Please give " \
                            "each channel a distinct azimuth.")

        ring = {}
        for k, i in enumerate(index):
            # wrapping the ends of the sorted ring back around to each other
            prev_azi = angles[index[k-1]]
            next_azi = angles[index[(k+1) % len(index)]]
            # unwrap so that prev_azi < angles[i] < next_azi
            prev_azi -= 2*np.pi * (prev_azi >= angles[i])
            next_azi += 2*np.pi * (next_azi <= angles[i])
            ring[i] = (prev_azi, next_azi)

        return ring

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
        net_power = 0.*plot_azimuths

        # setup axes
        fig = plt.figure(figsize=(8,8))
        ax = fig.subplots(subplot_kw={'projection': 'polar'})
        shift = 0.

        for i in range(self.Nmics):
            labelpos = 1.2
            microphone = self.mics[i]
            antenna = microphone.antenna(plot_azimuths)
            normalised_volume += antenna
            net_power += antenna**2
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
        # unlike the net amplitude, this is not rescaled, so a setup that
        # places sources at a consistent loudness sits flat at unity
        plt.plot(plot_azimuths, np.sqrt(net_power), c='k', ls ='-.', label='net power')
        
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
