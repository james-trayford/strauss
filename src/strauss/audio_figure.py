import numpy as np
from . import channels
from .stream import Stream
from .channels import audio_channels
from .utilities import const_or_evo, nested_dict_idx_reassign, apply_fades, rescale_values, NoSoundDevice
from .tts_caption import render_caption, get_ttsMode, default_tts_voice

class AudioFigure:
    """
    A figure-like wrapper to manage, mix, and render multiple STRAUSS sonifications.
    """
    def __init__(self, length, samprate=48000, system="stereo"):
        self.length = length             # Master duration in seconds
        self.samprate = samprate         # Master sampling rate
        self.system = system             # Master channel format (mono/stereo/etc.)

        # We set up the master channels for the audio figure
        self.channels = channels.audio_channels(setup=system)

        # with the individual channel arrays
        self.channel_arrays = {}
        for c in range(self.channels.Nmics):
            self.channel_arrays[str(c)] = Stream(self.length, self.samprate)

        # Store sonifications in a dict for easy access and modification
        self.sonifications = {}          
        
        # The final mixed audio array
        self.master_audio = None         

    def add(self, soni, name=None, level='0dB'):
        """
        Add a sonification to the figure.
        Enforces the common timebase so it can be mixed seamlessly.
        """
        # Enforce global timebase on the child
        soni.samprate = self.samprate
        soni.system = self.system
        # (Assuming the length is set on the generator, score, or sonification itself in your architecture)
        # soni.generator.length = self.length 
        if name: 
            self.sonifications[name] = soni
        else:
            self.sonifications[f"sonification_{len(self.sonifications.keys())+1}"] = soni

    def remove(self, name):
        """Remove a sonification from the figure."""
        if name in self.sonifications:
            del self.sonifications[name]

    def present(self, normalize='peak'):
        """
        Regenerate all attached sonifications and mix them additively.
        
        Parameters:
        normalize (str): 'peak', 'soft', or None to handle clipping.
        """
        
        for name, soni in self.sonifications.items():
            # 1. Generate the individual track
            soni.render()
            
            # 2. Extract the rendered numpy array 
            # (Change `soni.audio_array` to whatever attribute STRAUSS currently uses)
            track_audio = soni._make_out_array(embed_caption=False)

            # # 3. Align timebases (safety catch in case of rounding errors in generation)
            # track_audio = self._align_audio(track_audio, channels, total_samples)
            
            # 4. Additive mixing
            self.master_audio += track_audio

        # 5. Apply clipping protection
        self._apply_normalization(normalize)

    def _align_audio(self, audio, expected_channels, expected_samples):
        """
        Helper method to guarantee the audio array perfectly matches the master track shape.
        Pads with zeros or truncates the end if there is a mismatch.
        """
        # Check channel shape (adapt to your exact numpy dimensional structure)
        if audio.shape[0] != expected_channels:
            raise ValueError("Channel mismatch detected during mixing.")
            
        current_samples = audio.shape[1]
        
        if current_samples < expected_samples:
            # Pad with silence at the end
            padding = np.zeros((expected_channels, expected_samples - current_samples))
            return np.concatenate((audio, padding), axis=1)
        elif current_samples > expected_samples:
            # Truncate
            return audio[:, :expected_samples]
        
        return audio

    def _apply_normalization(self, method):
        """Prevents the additive mix from causing digital clipping (>1.0 or <-1.0)."""
        if method == 'peak':
            # Scales everything down proportionally if it breaches 1.0
            peak = np.max(np.abs(self.master_audio))
            if peak > 1.0:
                self.master_audio /= peak
        elif method == 'soft':
            # Applies a soft-clipping limiter (tanh) for a warmer, compressed mix
            self.master_audio = np.tanh(self.master_audio)

    def notebook_display(self):
        """Display the mixed master track in a Jupyter environment."""
        from IPython.display import Audio
        # Assumes a 1D or 2D master_audio layout
        return Audio(self.master_audio, rate=self.samprate)
