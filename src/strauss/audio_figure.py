import numpy as np
from . import channels
from .stream import Stream
from .channels import audio_channels
from .utilities import const_or_evo, nested_dict_idx_reassign, apply_fades, rescale_values, NoSoundDevice
from .tts_caption import render_caption, get_ttsMode, default_tts_voice
import IPython.display as ipd

INTMAX32 = (pow(2, 31)-1)

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
        
        # Store mixing levels
        self.levels = {}

        # The final mixed audio array
        self.master_audio = None         

    def add(self, soni, name=None, level='0dB'):
        """
        Add a sonification to the figure.
        Enforces the common timebase so it can be mixed seamlessly.
        
        Parameters:
            soni: The sonification object.
            name (str): Optional name for the track.
            level (str or float): Mixing level (e.g., '0dB', '-6dB', 0.5).
        """
        # Enforce global timebase on the child
        soni.samprate = self.samprate
        soni.system = self.system
        # (Assuming the length is set on the generator, score, or sonification itself in your architecture)
        # soni.generator.length = self.length 
        
        # Determine key name
        if name is None:
            name = f"sonification_{len(self.sonifications) + 1}"
            
        self.sonifications[name] = soni
        self.levels[name] = level

    def remove(self, name):
        """Remove a sonification from the figure."""
        if name in self.sonifications:
            del self.sonifications[name]
        if name in self.levels:
            del self.levels[name]

    def set_level(self, name, level):
        """
        Update the mixing level for a specific sonification.
        
        Parameters:
            name (str): The name of the sonification.
            level (str or float): The new level (e.g., '-3dB', 0.8).
        """
        if name in self.sonifications:
            self.levels[name] = level
        else:
            raise KeyError(f"Sonification '{name}' not found in AudioFigure.")

    def _parse_level(self, level):
        """
        Parses a mixing level to a linear amplitude fraction.
        
        Supports:
        - Strings ending in 'dB' (e.g., '-6 dB', '-inf dB')
        - Floats/linear fractions (0.0 to 1.0)
        """
        if isinstance(level, str):
            level_clean = level.strip().lower()
            if level_clean == '-inf db':
                return 0.0
            elif level_clean.endswith('db'):
                try:
                    db_val = float(level_clean.replace('db', '').strip())
                    # Convert dB to amplitude: 10^(dB/20)
                    return 10 ** (db_val / 20.0)
                except ValueError:
                    raise ValueError(f"Invalid dB format: {level}")
            else:
                # specific case for just a number in string format
                try:
                    return float(level)
                except ValueError:
                    raise ValueError(f"Unknown level format: {level}")
        elif isinstance(level, (int, float)):
            return float(level)
        else:
            raise TypeError(f"Level must be str or float, got {type(level)}")

    def render(self, normalize='peak'):
        """
        Regenerate all attached sonifications and mix them additively.
        
        Parameters:
        normalize (str): 'peak', 'soft', or None to handle clipping.
        """
        
        # Initialize master audio buffer
        # Shape: (N_channels, N_samples) or (N_samples, N_channels) depending on STRAUSS convention.
        # Assuming (N_channels, N_samples) based on _align_audio and standard audio libs
        n_samples = int(self.length * self.samprate)
        self.master_audio = np.zeros((self.channel_arrays['0'].values.size, len(self.channel_arrays)))

        for name, soni in self.sonifications.items():
            # 1. Generate the individual track
            soni.render(progress=False)
            
            # 2. Extract the rendered numpy array 
            track_audio = soni._make_out_array(embed_caption=False)

            # 3. Apply Mixing Level
            if name in self.levels:
                amp = self._parse_level(self.levels[name])
                track_audio = track_audio * amp
            # 4. Align timebases (safety catch in case of rounding errors in generation)
            # track_audio = self._align_audio(track_audio, self.channels.Nmics, n_samples)
            
            # 5. Additive mixing
            # Ensure shapes match before adding; if strict alignment isn't enabled, 
            # we assume strauss generation is accurate to the sample.
            if track_audio.shape == self.master_audio.shape:
                self.master_audio += track_audio
            else:
                # Basic safety add if dimensions differ slightly (e.g. 1 sample off)
                # This uses the helper you had commented out or a similar logic
                aligned_audio = self._align_audio(track_audio, self.channels.Nmics, n_samples)
                self.master_audio += aligned_audio

        # 6. Apply clipping protection
        # self._apply_normalization(normalize)
        vmax = abs(self.master_audio).max()
        self.master_audio *= (pow(2, 31)-1)/vmax
        self.master_audio = self.master_audio.astype('int32')

    def notebook_display(self, show_waveform=True):
        has_ticks = hasattr(self, 'tick_channels')
        if len(self.channels.labels) == 1:             
            outfmt = np.column_stack([self.master_audio]*2)
        else:
            outfmt = self.master_audio[:,:2]
        if len(self.channels.labels) > 2:
            print("Warning: for more than two channels, only first two channels are mapped to L and R, respectively.")
        if has_ticks:
            # add the ticks
            for c in range(outfmt.shape[0]):
                outfmt[c] += self.tick_channels['0'].values*self.tick_vol / vmax
        print(outfmt.max())
        display(ipd.Audio(outfmt.T,rate=self.samprate, autoplay=False))
        
    def _align_audio(self, audio, expected_channels, expected_samples):
        """
        Helper method to guarantee the audio array perfectly matches the master track shape.
        Pads with zeros or truncates the end if there is a mismatch.
        """
        # Check channel shape
        if audio.shape[0] != expected_channels:
            # If mono source in stereo system, duplication might be needed, 
            # but strictly raising error for now as per original code.
            raise ValueError(f"Channel mismatch: Source has {audio.shape[0]}, Master has {expected_channels}")
            
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
