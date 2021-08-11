import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys

class mic:
    """microphone object"""
    def __init__(self, phi, mic_type="directional", label="C", channel=1):
        self.phi = phi
        self.mic_type = mic_type
        self.label = label
        self.channel = channel
        
        if mic_type == "directional":
            self.antenna = lambda a, b=0.5*np.pi: 0.5*(1+np.cos(a-phi)*np.sin(b))
        elif mic_type == "omni":
            self.antenna = lambda a, b=0.5*np.pi: a**0.
        elif mic_type == "mute":
            self.antenna = lambda a, b=0.5*np.pi: a*0.
        else:
            raise Exception(f"Mic type \"{mic_type}\" unknown.")        
        
class audio_channels:
    """audio channels object"""

    def __init__(self, setup="stereo", custom_setup={}):
        """initialise mic_array object"""

        ##############################################
        # Channel properties for preset audio setups
        ##############################################
        
        # mic angles in radians
        mono_phis = [0.]
        stereo_phis = [0.5*np.pi, 1.5*np.pi]
        fivepoint_phis = [1./3*np.pi, 5./3*np.pi,
                            0., 0.,
                            2./3*np.pi, 4./3*np.pi]
        sevenpoint_phis = fivepoint_phis + stereo_phis

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
            self.setup_channels(mono_phis, mono_types, mono_labels)
            self.forder = mono_forder
        elif setup == "stereo":
            self.setup_channels(stereo_phis, stereo_types, stereo_labels)
            self.forder = stereo_forder
        elif setup == "5.1":
            self.setup_channels(fivepoint_phis, fivepoint_types, fivepoint_labels)
            self.forder = fivepoint_forder
        elif setup == "7.1":
            self.setup_channels(sevenpoint_phis, sevenpoint_types, sevenpoint_labels)
            self.forder = sevenpoint_forder
        elif setup == "custom":
            self.setup_channels(custom_setup['phis'],
                                custom_setup['types'],
                                custom_setup['labels'])
            if 'forder' in custom_setup:
                self.forder = self.custom_setup['forder']
            else:
                self.forder = 'unknown'
        else:
            raise Exception(f"setup \"{setup}\" not understood")
            
    def setup_channels(self, phis, types, labels):
        """initialise audio channel setup for lists of properties"""
        self.phis = phis
        self.types = types
        self.labels = labels
        self.Nmics = len(phis)

        # Note: channel ordering important, sets output channel number
        self.channels = range(1, self.Nmics+1)
        
        self.mics = []
        
        for i in range(self.Nmics):
            microphone = mic(phis[i], types[i], labels[i], self.channels[i])
            self.mics.append(microphone)

    def plot_antenna(self):
        """plot antennae patterns for chosen audio setup"""

        plot_phis = np.linspace(0, 2*np.pi, 100)
        normalised_volume = 0.*plot_phis

        # setup axes
        fig = plt.figure(figsize=(8,8))
        ax = fig.subplots(subplot_kw={'projection': 'polar'})
        shift = 0.

        for i in range(self.Nmics):
            labelpos = 1.2
            microphone = self.mics[i]
            normalised_volume += microphone.antenna(plot_phis)
            ax.plot(plot_phis, microphone.antenna(plot_phis))
            if np.all(microphone.antenna(plot_phis) == 0.):
                labelpos += shift
                shift -= 0.15
            ax.scatter(microphone.phi, labelpos)
            ax.annotate(microphone.label+f" (ch. {microphone.channel})",
                         (microphone.phi, labelpos),
                         (0, 10), textcoords='offset points', ha='center')
        normalised_volume /= normalised_volume.max()
        plt.plot(plot_phis, normalised_volume, c='k', ls ='--', label='net amplitude')
        
        # configure axes
        ax = plt.gca()
        ax.set_theta_zero_location("N")
        ax.set_yticklabels([])
        ax.set_xlabel('phi')
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
