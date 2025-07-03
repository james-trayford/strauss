import unittest
import numpy as np
from strauss.generator import Generator, Synthesizer, Sampler, Spectralizer
from unittest.mock import patch, MagicMock
from pathlib import Path # Import Path
from scipy.interpolate import interp1d # Import interp1d

# We might need to mock some dependencies like sf2utils, pyfftw, or sounddevice

class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.samprate = 48000
        self.gen = Generator(samprate=self.samprate)
        # A minimal default preset structure for Generator related tests
        self.gen.preset = {
            'volume_envelope': {'A': 0.1, 'D': 0.2, 'S': 0.7, 'R': 0.5, 'Ac': 0, 'Dc': 0, 'Rc': 0, 'level': 1.0},
            'pitch_lfo': {'use': False, 'wave': 'sine', 'freq': 5, 'amount': 1, 'phase': 0, 'A':0, 'D':0, 'S':1, 'R':0, 'Ac':0, 'Dc':0, 'Rc':0, 'level':1, 'freq_shift': 0},
            'volume_lfo': {'use': False, 'wave': 'sine', 'freq': 5, 'amount': 0.2, 'phase': 0, 'A':0, 'D':0, 'S':1, 'R':0, 'Ac':0, 'Dc':0, 'Rc':0, 'level':1, 'freq_shift': 0},
            'note_length': 1.0 # Default for envelope testing
        }
        self.gen.gtype = 'synth' # Needed for load_preset to pick a path

    def test_generator_initialization(self):
        self.assertEqual(self.gen.samprate, self.samprate)
        self.assertGreater(self.gen.audbuff, 0) # samples per buffer (samprate / 30)

    @patch('strauss.generator.presets')
    def test_load_preset(self, mock_presets):
        mock_synth_module = MagicMock()
        mock_presets.synth = mock_synth_module

        # Mock load_preset to return a specific dict
        test_preset_data = {'param1': 100, 'volume_envelope': {'A': 0.05}}
        mock_synth_module.load_preset.return_value = test_preset_data

        # Mock load_ranges to return an empty dict or some mock
        mock_synth_module.load_ranges.return_value = {}

        gen = Generator(samprate=self.samprate)
        gen.gtype = 'synth' # Set gtype before loading
        gen.load_preset(preset='test_preset')

        mock_synth_module.load_preset.assert_any_call('default') # Default is always loaded first
        mock_synth_module.load_preset.assert_any_call('test_preset')
        self.assertEqual(gen.preset['param1'], 100)
        self.assertEqual(gen.preset['volume_envelope']['A'], 0.05)
        self.assertIn('ranges', gen.preset)


    def test_modify_preset(self):
        self.gen.modify_preset({'volume_envelope': {'A': 0.01, 'S': 0.6}})
        self.assertEqual(self.gen.preset['volume_envelope']['A'], 0.01)
        self.assertEqual(self.gen.preset['volume_envelope']['S'], 0.6)
        self.assertEqual(self.gen.preset['volume_envelope']['D'], 0.2) # Unchanged

    @patch('strauss.generator.presets')
    def test_preset_details(self, mock_presets):
        mock_synth_module = MagicMock()
        mock_presets.synth = mock_synth_module

        self.gen.preset_details(term="test")
        mock_synth_module.preset_details.assert_called_once_with(name="test")

    def test_envelope(self):
        params = self.gen.preset
        params['note_length'] = 1.0 # seconds

        # Test Attack phase
        # A=0.1, D=0.2, S=0.7, R=0.5
        # At t=0.05 (mid-attack), value should be approx 0.5 if linear (Ac=0)
        samples_attack = np.array([0.05 * self.samprate])
        env_attack = self.gen.envelope(samples_attack, params, etype='volume')
        self.assertGreater(env_attack[0], 0.4)
        self.assertLess(env_attack[0], 0.6)

        # Test Decay phase
        # At t=0.1 (end attack), value = 1.0
        # At t=0.2 (mid-decay, t-t1 = 0.1), S=0.7. (1-0.7)/(0.2) * 0.1 = 0.15. 1 - 0.15 = 0.85
        samples_decay = np.array([0.2 * self.samprate])
        env_decay = self.gen.envelope(samples_decay, params, etype='volume')
        self.assertGreater(env_decay[0], 0.8) # Approx 0.85 for linear
        self.assertLess(env_decay[0], 0.9)

        # Test Sustain phase
        # At t=0.5 (A+D = 0.1+0.2=0.3. Note_length=1.0. So 0.3 < t < 1.0 is sustain)
        samples_sustain = np.array([0.5 * self.samprate])
        env_sustain = self.gen.envelope(samples_sustain, params, etype='volume')
        self.assertAlmostEqual(env_sustain[0], params['volume_envelope']['S'], places=5)

        # Test Release phase
        # Note ends at t=1.0, env_off = S = 0.7. Release time R=0.5
        # At t=1.25 (mid-release, t-nlen = 0.25), val should be S/2 = 0.35 if linear
        samples_release = np.array([1.25 * self.samprate])
        env_release = self.gen.envelope(samples_release, params, etype='volume')
        self.assertGreater(env_release[0], 0.3) # Approx 0.35
        self.assertLess(env_release[0], 0.4)

        # Test after release
        samples_after = np.array([(params['note_length'] + params['volume_envelope']['R'] + 0.1) * self.samprate])
        env_after = self.gen.envelope(samples_after, params, etype='volume')
        self.assertAlmostEqual(env_after[0], 0.0, places=5)

    def test_lfo_pitch(self):
        self.gen.preset['pitch_lfo']['use'] = True
        self.gen.preset['pitch_lfo']['amount'] = 2 # semitones
        self.gen.preset['pitch_lfo']['freq'] = 1 # 1 Hz
        self.gen.preset['note_length'] = 2.0 # 2s note for 1Hz LFO to cycle

        num_samples = int(2.0 * self.samprate)
        samples = np.arange(num_samples)
        sampfracs = samples / float(num_samples -1) if num_samples > 1 else np.array([0.0])

        lfo_values = self.gen.lfo(samples, sampfracs, self.gen.preset, ltype='pitch')

        # Should oscillate between -2 and 2 (amount=2)
        self.assertLessEqual(np.max(lfo_values), 2.0)
        self.assertGreaterEqual(np.min(lfo_values), -2.0)
        # With 1Hz freq over 2s, should complete two full sine cycles (phase=0)
        # Max at 0.25s, min at 0.75s, max at 1.25s, min at 1.75s
        self.assertAlmostEqual(lfo_values[int(0.25*self.samprate)], 2.0, places=1) # Max amount
        self.assertAlmostEqual(lfo_values[int(0.75*self.samprate)], -2.0, places=1) # Min amount

    def test_oscillators(self):
        s = np.array([0, 0.25, 0.5, 0.75, 1.0]) # Time in cycles if f=1, p=0
        f_norm = 1.0 / self.samprate # Frequency for 1 cycle per samprate seconds (dummy)

        # Sine: sin(2*pi*s*f+p)
        sine_wave = self.gen.sine(s * self.samprate, f_norm, 0) # s is sample index
        expected_sine = np.sin(2 * np.pi * s)
        np.testing.assert_array_almost_equal(sine_wave, expected_sine, decimal=5)

        # Saw: (2*(s*f+p) +1) % 2 - 1
        saw_wave = self.gen.saw(s * self.samprate, f_norm, 0)
        expected_saw = (2*s + 1) % 2 - 1
        np.testing.assert_array_almost_equal(saw_wave, expected_saw, decimal=5)

        # Square: sign(saw)
        square_wave = self.gen.square(s * self.samprate, f_norm, 0)
        expected_square = np.sign(expected_saw) # np.sign(0) is 0. This matches the implementation.
        # The implementation is np.sign(self.saw(...)). If self.saw(...) is 0, output is 0.
        np.testing.assert_array_almost_equal(square_wave, expected_square, decimal=5)


        # Triangle: 1 - abs((4*(s*f+p) +1) % 4 - 2)
        tri_wave = self.gen.tri(s * self.samprate, f_norm, 0)
        expected_tri = 1 - np.abs((4*s + 1) % 4 - 2)
        np.testing.assert_array_almost_equal(tri_wave, expected_tri, decimal=5)

        # Noise: Should be random
        noise_wave1 = self.gen.noise(s * self.samprate, f_norm, 0)
        noise_wave2 = self.gen.noise(s * self.samprate, f_norm, 0)
        self.assertEqual(len(noise_wave1), len(s))
        self.assertFalse(np.array_equal(noise_wave1, noise_wave2)) # Highly unlikely to be equal
        self.assertTrue(np.all(noise_wave1 >= -1) and np.all(noise_wave1 <= 1))


class TestSynthesizer(unittest.TestCase):
    def setUp(self):
        self.samprate = 48000
        # Basic preset for synthesizer
        self.default_preset_data = {
            'oscillators': {
                'osc1': {'form': 'sine', 'level': 0.8, 'detune': 0, 'phase': 0},
                'osc2': {'form': 'saw', 'level': 0.2, 'detune': 5, 'phase': 'random'}
            },
            'volume_envelope': {'A': 0.01, 'D': 0.1, 'S': 0.8, 'R': 0.2, 'Ac':0,'Dc':0,'Rc':0, 'level':1.0},
            'pitch_lfo': {'use': False, 'wave': 'sine', 'freq': 5, 'amount': 1, 'phase': 0, 'A':0,'D':0,'S':1,'R':0,'Ac':0,'Dc':0,'Rc':0, 'level':1, 'freq_shift':0},
            'volume_lfo': {'use': False, 'wave': 'sine', 'freq': 5, 'amount': 0.1, 'phase': 0, 'A':0,'D':0,'S':1,'R':0,'Ac':0,'Dc':0,'Rc':0, 'level':1, 'freq_shift':0},
            'filter': 'off', 'filter_type': 'LPF1', 'cutoff': 0.5,
            'pitch_shift': 0, 'volume': 1.0, 'note': 'A4', 'note_length': 1.0,
            'azimuth': 0.0, 'polar': 0.5 * np.pi
        }

    @patch('strauss.generator.presets.synth.load_preset')
    @patch('strauss.generator.presets.synth.load_ranges')
    def test_synthesizer_initialization(self, mock_load_ranges, mock_load_preset):
        mock_load_preset.return_value = self.default_preset_data.copy()
        mock_load_ranges.return_value = {} # Default ranges

        synth = Synthesizer(samprate=self.samprate)
        self.assertEqual(synth.gtype, 'synth')
        self.assertTrue(hasattr(synth, 'preset'))
        self.assertIn('oscillators', synth.preset)
        self.assertTrue(callable(synth.generate)) # Should be combine_oscs

    @patch('strauss.generator.presets.synth.load_preset')
    @patch('strauss.generator.presets.synth.load_ranges')
    def test_synthesizer_modify_preset_clear_oscs(self, mock_load_ranges, mock_load_preset):
        mock_load_preset.return_value = self.default_preset_data.copy()
        mock_load_ranges.return_value = {}
        synth = Synthesizer(samprate=self.samprate)

        new_oscs = {'new_osc': {'form': 'square', 'level': 1.0, 'detune': 0, 'phase': 0}}
        synth.modify_preset({'oscillators': new_oscs, 'volume': 0.7}, clear_oscs=True)

        self.assertEqual(len(synth.preset['oscillators']), 1)
        self.assertIn('new_osc', synth.preset['oscillators'])
        self.assertEqual(synth.preset['oscillators']['new_osc']['form'], 'square')
        self.assertEqual(synth.preset['volume'], 0.7) # Check other params also modified

    @patch('strauss.generator.presets.synth.load_preset')
    @patch('strauss.generator.presets.synth.load_ranges')
    def test_synthesizer_modify_preset_no_clear_oscs(self, mock_load_ranges, mock_load_preset):
        original_preset_copy = self.default_preset_data.copy()
        # Deep copy oscillators to avoid modifying the class default during test
        original_preset_copy['oscillators'] = {k: v.copy() for k,v in self.default_preset_data['oscillators'].items()}

        mock_load_preset.return_value = original_preset_copy
        mock_load_ranges.return_value = {}
        synth = Synthesizer(samprate=self.samprate)

        # Modify an existing oscillator and add a new one
        modifications = {
            'oscillators': {
                'osc1': {'level': 0.9}, # Modify existing
                'osc3': {'form': 'tri', 'level': 0.5, 'detune': -5, 'phase': 0.5} # Add new
            },
            'pitch_shift': 2
        }
        synth.modify_preset(modifications, clear_oscs=False)

        self.assertEqual(len(synth.preset['oscillators']), 3) # osc1, osc2, osc3
        self.assertEqual(synth.preset['oscillators']['osc1']['level'], 0.9) # Modified
        self.assertEqual(synth.preset['oscillators']['osc1']['form'], 'sine') # Original form unchanged
        self.assertIn('osc2', synth.preset['oscillators']) # Original osc2 still there
        self.assertIn('osc3', synth.preset['oscillators']) # New osc3 added
        self.assertEqual(synth.preset['oscillators']['osc3']['form'], 'tri')
        self.assertEqual(synth.preset['pitch_shift'], 2)


    def test_combine_oscs(self):
        # Simpler synth for this specific test
        synth = Synthesizer(samprate=self.samprate)
        synth.preset['oscillators'] = {
            'osc1': {'form': 'sine', 'level': 1.0, 'detune': 0, 'phase': 0}
        }
        synth.setup_oscillators() # Re-run if preset changed manually after init

        test_freq = 440.0 # A4
        samples_indices = np.arange(0, int(0.01 * self.samprate)) # 0.01 seconds of samples

        # Expected output for a single sine wave at 440Hz
        time_array = samples_indices / float(self.samprate)
        expected_wave = 1.0 * np.sin(2 * np.pi * test_freq * time_array + 0)

        combined_wave = synth.combine_oscs(samples_indices, test_freq)
        np.testing.assert_array_almost_equal(combined_wave, expected_wave, decimal=5)

        # Test with two oscillators
        synth.preset['oscillators'] = {
            'osc1': {'form': 'sine', 'level': 0.6, 'detune': 0, 'phase': 0},
            'osc2': {'form': 'sine', 'level': 0.4, 'detune': 0, 'phase': np.pi/2} # 90 deg phase shift (0.25 cycles)
        }
        synth.setup_oscillators()
        expected_wave_osc1 = 0.6 * np.sin(2 * np.pi * test_freq * time_array)
        expected_wave_osc2 = 0.4 * np.sin(2 * np.pi * test_freq * time_array + np.pi/2) # phase is in cycles for preset, so 0.25
                                                                                      # but sine func takes phase in radians for calculation
                                                                                      # The preset phase is float or 'random'
                                                                                      # The sine func takes phase in units of cycles
                                                                                      # So phase = 0.25 for 90 deg
        # Re-check preset. Phase is in cycles for oscillators
        synth.preset['oscillators']['osc2']['phase'] = 0.25 # 90 deg
        synth.setup_oscillators()
        expected_wave_osc2_corrected = 0.4 * np.sin(2 * np.pi * (test_freq * time_array + 0.25))


        combined_wave_two_oscs = synth.combine_oscs(samples_indices, test_freq)
        np.testing.assert_array_almost_equal(combined_wave_two_oscs, expected_wave_osc1 + expected_wave_osc2_corrected, decimal=5)


    @patch('strauss.generator.stream.Stream')
    @patch('strauss.generator.filters')
    def test_synthesizer_play_basic(self, mock_filters_module, mock_stream_class):
        # Setup mock Stream instance and its attributes
        mock_stream_instance = MagicMock()
        mock_stream_instance.samples = np.arange(int(self.samprate * 1.0)) # 1s default note_length
        mock_stream_instance.sampfracs = mock_stream_instance.samples / float(len(mock_stream_instance.samples)-1)
        mock_stream_instance.values = np.zeros_like(mock_stream_instance.samples, dtype=float)
        mock_stream_instance.samprate = self.samprate
        mock_stream_instance.length = 1.0 # from note_length + R
        mock_stream_class.return_value = mock_stream_instance

        synth = Synthesizer(samprate=self.samprate)
        # Use a simple preset for play
        synth.preset = self.default_preset_data.copy()
        synth.preset['oscillators'] = {'osc1': {'form': 'sine', 'level': 1.0, 'detune': 0, 'phase': 0}}
        synth.preset['volume_envelope']['R'] = 0.0 # simplify length calculation
        synth.setup_oscillators()

        mapping = {
            'note': 'A4', # 440 Hz
            'note_length': 1.0,
            'volume': 1.0,
            # other params will use preset defaults
        }
        # Linearize mapping for play method if nested
        linear_mapping = {}
        for k,v in mapping.items(): linear_mapping[k] = v
        # Add envelope params from preset to mapping, as play expects them flattened if modified
        for k,v in synth.preset['volume_envelope'].items(): linear_mapping[f'volume_envelope/{k}'] = v
        # ... and other LFO/filter params if they were active


        returned_stream = synth.play(linear_mapping)
        self.assertIs(returned_stream, mock_stream_instance)

        # Check that stream values were modified (i.e., sound was generated)
        # For a 1s sine wave at full volume, not all values will be zero.
        self.assertFalse(np.all(mock_stream_instance.values == 0))

        # Check if envelope was applied: values at the end should be close to zero due to release (if R > 0)
        # With R=0, and S=0.8, it should end at S level if note_length is hit.
        # Our envelope A=0.01, D=0.1, S=0.8, R=0 (modified for test). Note length = 1.0
        # End of note_length (1.0s) should be S level = 0.8
        # The stream values are multiplied by envelope.
        # Max value of sine is 1. Max value of envelope is 1. So max stream value could be 1.
        # Value at sample corresponding to 0.5s (in sustain phase)
        # Expected value: 1.0 (osc level) * 0.8 (sustain level) * 1.0 (mapping volume)
        # This requires knowing the exact generated wave.
        # A simpler check: stream.values should not be all zeros.
        # And filter should not have been called if filter='off'
        mock_filters_module.LPF1.assert_not_called()
        mock_filters_module.HPF1.assert_not_called()

class TestSampler(unittest.TestCase):
    def setUp(self):
        self.samprate = 48000
        self.base_preset = { # Sampler preset fields needed for play method
            'volume_envelope': {'A': 0.01, 'D': 0.1, 'S': 0.8, 'R': 0.2, 'Ac':0,'Dc':0,'Rc':0, 'level':1.0},
            'pitch_lfo': {'use': False, 'wave': 'sine', 'freq': 5, 'amount': 1, 'phase': 0, 'A':0,'D':0,'S':1,'R':0,'Ac':0,'Dc':0,'Rc':0, 'level':1, 'freq_shift':0},
            'volume_lfo': {'use': False, 'wave': 'sine', 'freq': 5, 'amount': 0.1, 'phase': 0, 'A':0,'D':0,'S':1,'R':0,'Ac':0,'Dc':0,'Rc':0, 'level':1, 'freq_shift':0},
            'filter': 'off', 'filter_type': 'LPF1', 'cutoff': 0.5,
            'pitch_shift': 0, 'volume': 1.0, 'note': 'A4', 'note_length': 1.0,
            'looping': 'off', 'loop_start': 0.1, 'loop_end': 0.5, # Sampler specific
            'azimuth': 0.0, 'polar': 0.5 * np.pi
        }

    @patch('strauss.generator.wavfile.read')
    @patch('strauss.generator.Path.glob')
    @patch('strauss.generator.presets.sampler.load_preset')
    @patch('strauss.generator.presets.sampler.load_ranges')
    def test_sampler_init_wav_files(self, mock_load_ranges, mock_load_preset, mock_glob, mock_wav_read):
        mock_load_preset.return_value = self.base_preset.copy()
        mock_load_ranges.return_value = {}

        # Simulate Path("path/to/samples").glob("*.[wW][aA][vV]")
        mock_sample_file_path = MagicMock(spec=Path)
        mock_sample_file_path.name = "samples_A4.wav"
        mock_glob.return_value = [mock_sample_file_path]

        # Simulate wavfile.read returning (rate, data)
        dummy_wav_data = np.array([0, 1, 2, 1, 0], dtype=np.int16)
        mock_wav_read.return_value = (self.samprate, dummy_wav_data)

        sampler = Sampler(sampfiles="path/to/samples", samprate=self.samprate)

        mock_glob.assert_called_once_with("*.[wW][aA][vV]")
        mock_wav_read.assert_called_once_with(str(mock_sample_file_path))
        self.assertIn('A4', sampler.samples)
        self.assertTrue(callable(sampler.samples['A4']))
        self.assertEqual(sampler.samplens['A4'], len(dummy_wav_data))
        # Check if data (after normalization) is roughly what was input
        # Accessing interpolated values:
        interpolated_func = sampler.samples['A4']
        # Normalized data: (dummy_wav_data - mean) / max_abs_val
        norm_data = (dummy_wav_data - np.mean(dummy_wav_data))
        norm_data = norm_data / np.max(np.abs(norm_data)) if np.max(np.abs(norm_data)) > 0 else norm_data

        # Check a few points from the interpolator
        # Need to be careful with floating point comparisons for interpolated values
        self.assertAlmostEqual(interpolated_func(0), norm_data[0], places=5)
        self.assertAlmostEqual(interpolated_func(len(dummy_wav_data)-1), norm_data[-1], places=5)


    @patch('strauss.generator.Sf2File')
    @patch('strauss.generator.presets.sampler.load_preset')
    @patch('strauss.generator.presets.sampler.load_ranges')
    @patch('builtins.open', new_callable=unittest.mock.mock_open) # Mock open for sf2 file
    def test_sampler_init_sf2_file(self, mock_open, mock_load_ranges, mock_load_preset, mock_sf2file_class):
        mock_load_preset.return_value = self.base_preset.copy()
        mock_load_ranges.return_value = {}

        # Mock Sf2File structure and its methods/attributes
        mock_sf2_instance = MagicMock()
        # Simulate sf2_instance.raw.pdta['Phdr']
        mock_phdr_entry = MagicMock()
        mock_phdr_entry.name = b'TestPreset\x00\x00\x00'
        mock_sf2_instance.raw.pdta = {'Phdr': [mock_phdr_entry, MagicMock()]} # EOS entry

        # Mock build_presets()
        mock_preset_obj = MagicMock()
        mock_preset_obj.bags = [] # Simplified, actual structure is more complex
        mock_sf2_instance.build_presets.return_value = [mock_preset_obj]

        mock_sf2file_class.return_value = mock_sf2_instance

        # Mock get_sfpreset_samples to return some dummy data structure
        # This method is complex, so we simplify its output for this test
        dummy_sf_data = {
            'samples': {'A4_sample': np.array([0,1000,0,-1000,0], dtype=np.int16)},
            'sample_rate': {'A4_sample': self.samprate},
            'original_pitch': {'A4_sample': 69}, # A4
            'min_note': 69, 'max_note': 69,
            'sample_map': {69: ['A4_sample']}
        }
        # Patch the Sampler's own methods
        with patch.object(Sampler, 'get_sfpreset_samples', return_value=dummy_sf_data) as mock_get_sf_samples, \
             patch.object(Sampler, 'reconstruct_samples', return_value={'A4': np.array([0.0,0.5,0.0,-0.5,0.0])}) as mock_reconstruct:

            sampler = Sampler(sampfiles="dummy.sf2", samprate=self.samprate, sf_preset=1)

            mock_open.assert_called_once_with("dummy.sf2", 'rb')
            mock_sf2file_class.assert_called_once()
            mock_get_sf_samples.assert_called_once()
            mock_reconstruct.assert_called_once_with(dummy_sf_data)
            self.assertIn('A4', sampler.samples) # From reconstruct_samples mock
            self.assertTrue(callable(sampler.samples['A4']))

    @patch('strauss.generator.stream.Stream')
    @patch('strauss.generator.filters')
    @patch('strauss.generator.wavfile.read') # Mock read for base Sampler if it tries to load samples
    @patch('strauss.generator.Path.glob')
    def test_sampler_play_basic(self, mock_glob, mock_wav_read, mock_filters_module, mock_stream_class):
        # Setup Sampler with a single mocked sample
        mock_stream_instance = MagicMock()
        mock_stream_instance.samples = np.arange(int(self.samprate * 1.0))
        mock_stream_instance.sampfracs = mock_stream_instance.samples / float(len(mock_stream_instance.samples)-1)
        mock_stream_instance.values = np.zeros_like(mock_stream_instance.samples, dtype=float)
        mock_stream_instance.samprate = self.samprate
        mock_stream_instance.length = 1.2 # note_length 1.0 + R 0.2
        mock_stream_class.return_value = mock_stream_instance

        # Mock sample loading part of Sampler for this play test
        sampler = Sampler.__new__(Sampler) # Create instance without calling __init__
        sampler.samprate = self.samprate
        sampler.audbuff = self.samprate / 30.0
        sampler.preset = self.base_preset.copy()
        sampler.preset['volume_envelope']['R'] = 0.2 # ensure length calc is simple
        sampler.gtype = 'sampler'

        # Mock one sample
        dummy_sample_data = np.sin(np.linspace(0, 2*np.pi*440*(1.0), int(1.0*self.samprate))) # 1s A4 sine
        sampler.samples = {'A4': MagicMock(spec=interp1d)}
        sampler.samples['A4'].return_value = dummy_sample_data # Mocked interpolator returns full array
        sampler.samplens = {'A4': len(dummy_sample_data)}


        mapping = {'note': 'A4', 'note_length': 1.0, 'volume': 0.8}
        linear_mapping = {}
        for k,v in mapping.items(): linear_mapping[k] = v
        for k,v in sampler.preset['volume_envelope'].items(): linear_mapping[f'volume_envelope/{k}'] = v

        returned_stream = sampler.play(linear_mapping)
        self.assertIs(returned_stream, mock_stream_instance)
        self.assertFalse(np.all(mock_stream_instance.values == 0)) # Sound was generated

        # Check if the mocked sample function was called
        sampler.samples['A4'].assert_called()
        mock_filters_module.LPF1.assert_not_called()


class TestSpectralizer(unittest.TestCase):
    def setUp(self):
        self.samprate = 48000
        self.base_preset = { # Spectralizer preset fields
            'volume_envelope': {'A': 0.01, 'D': 0.1, 'S': 0.8, 'R': 0.2, 'Ac':0,'Dc':0,'Rc':0, 'level':1.0},
            'pitch_lfo': {'use': False, 'wave': 'sine', 'freq': 5, 'amount': 1, 'phase': 0, 'A':0,'D':0,'S':1,'R':0,'Ac':0,'Dc':0,'Rc':0, 'level':1, 'freq_shift':0},
            'volume_lfo': {'use': False, 'wave': 'sine', 'freq': 5, 'amount': 0.1, 'phase': 0, 'A':0,'D':0,'S':1,'R':0,'Ac':0,'Dc':0,'Rc':0, 'level':1, 'freq_shift':0},
            'filter': 'off', 'filter_type': 'LPF1', 'cutoff': 0.5, # Usually not used by spectralizer
            'pitch_shift': 0, 'volume': 1.0, 'note_length': 1.0,
            'min_freq': 100, 'max_freq': 1000, # Spectralizer specific
            'spectrum': np.array([1.0, 0.5, 0.1]), # Dummy spectrum
            'interpolation_type': 'sample',
            'equal_loudness_normalisation': False,
            'fit_spec_multiples': False,
            'regen_phases': False,
            'azimuth': 0.0, 'polar': 0.5 * np.pi
        }

    @patch('strauss.generator.presets.spec.load_preset')
    @patch('strauss.generator.presets.spec.load_ranges')
    @patch('strauss.generator.utils.Equaliser')
    def test_spectralizer_initialization(self, MockEqualiser, mock_load_ranges, mock_load_preset):
        mock_load_preset.return_value = self.base_preset.copy()
        mock_load_ranges.return_value = {}

        spec = Spectralizer(samprate=self.samprate)

        self.assertEqual(spec.gtype, 'spec')
        self.assertTrue(hasattr(spec, 'preset'))
        MockEqualiser.assert_called_once()
        self.assertIsInstance(spec.eq, MagicMock) # Check it's the mocked Equaliser

    @patch('strauss.generator.ifft')
    def test_spectrum_to_signal(self, mock_ifft):
        spec = Spectralizer(samprate=self.samprate)
        new_nlen = 100 # Target signal length for this test consistent with phases
        mock_ifft.return_value = np.random.rand(new_nlen) # Dummy ifft output of correct length

        spectrum_data = np.array([1.0, 0.8, 0.6])
        phases = np.random.rand(new_nlen) * 2 * np.pi # Dummy phases matching new_nlen
        new_nlen = 100 # Target signal length
        mindx, maxdx = 10, 40 # Dummy freq indices

        # Test "sample" interpolation
        signal_sample = spec.spectrum_to_signal(spectrum_data, phases, new_nlen, mindx, maxdx, "sample")
        self.assertEqual(len(signal_sample), new_nlen)
        mock_ifft.assert_called() # Check ifft was called

        # Test "preserve_power" interpolation
        # We need to ensure the input to ifft has correct structure based on this mode
        # For this unit test, primarily ensure it runs and calls ifft.
        mock_ifft.reset_mock()
        signal_power = spec.spectrum_to_signal(spectrum_data, phases, new_nlen, mindx, maxdx, "preserve_power")
        self.assertEqual(len(signal_power), new_nlen)
        mock_ifft.assert_called()


    @patch('strauss.generator.stream.Stream')
    @patch('strauss.generator.ifft') # Mock ifft for the play method's internal calls
    @patch('strauss.generator.utils.Equaliser')
    @patch('strauss.generator.presets.spec.load_preset') # Mock preset loading for constructor
    @patch('strauss.generator.presets.spec.load_ranges')
    def test_spectralizer_play_1D_spectrum(self, mock_load_ranges, mock_load_preset, MockEqualiser, mock_ifft, mock_stream_class):
        mock_load_preset.return_value = self.base_preset.copy()
        mock_load_ranges.return_value = {}
        mock_eq_instance = MockEqualiser.return_value
        mock_eq_instance.get_relative_loudness_norm.return_value = np.array([1.0, 1.0, 1.0]) # No change

        # Setup mock Stream
        mock_stream_instance = MagicMock()
        duration = self.base_preset['note_length'] + self.base_preset['volume_envelope']['R']
        num_samples = int(duration * self.samprate)
        mock_stream_instance.samples = np.arange(num_samples)
        mock_stream_instance.sampfracs = mock_stream_instance.samples / float(num_samples -1 if num_samples > 1 else 1)
        mock_stream_instance.values = np.zeros(num_samples, dtype=float)
        mock_stream_instance.samprate = self.samprate
        mock_stream_instance.length = duration
        mock_stream_class.return_value = mock_stream_instance

        mock_ifft.return_value = np.random.rand(num_samples) # ifft returns an array of signal length

        spec = Spectralizer(samprate=self.samprate)
        spec.preset = self.base_preset.copy() # Ensure preset is clean for test
        spec.preset['spectrum'] = np.array([1.0, 0.5, 0.2])
        spec.preset['equal_loudness_normalisation'] = False # Simplify by disabling

        mapping = {'note_length': 1.0, 'spectrum': spec.preset['spectrum']} # Pass spectrum in mapping
        linear_mapping = {}
        for k,v in mapping.items(): linear_mapping[k] = v
        for k_env,v_env in spec.preset['volume_envelope'].items(): linear_mapping[f'volume_envelope/{k_env}'] = v_env

        returned_stream = spec.play(linear_mapping)

        self.assertIs(returned_stream, mock_stream_instance)
        mock_ifft.assert_called() # spectrum_to_signal -> ifft
        self.assertFalse(np.all(mock_stream_instance.values == 0)) # Sound generated

    # TODO: Add test for evolving spectrum (2D) if time permits, involves buffer logic
    # TODO: Add test for equal_loudness_normalisation path

if __name__ == '__main__':
    unittest.main()
