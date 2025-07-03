import unittest
from unittest.mock import patch, MagicMock
# Import necessary classes from strauss
from strauss.sonification import Sonification
import numpy as np # Import numpy
from strauss.score import Score
from strauss.sources import Events # Or Objects, depending on what's easier to mock/test
from strauss.generator import Synthesizer # Or Sampler/Spectralizer
from strauss.channels import audio_channels
from strauss.stream import Stream # Import Stream directly

class TestSonification(unittest.TestCase):
    def setUp(self):
        self.samprate = 48000

        # Mock Score
        self.mock_score = MagicMock(spec=Score)
        self.mock_score.length = 10.0 # seconds
        self.mock_score.note_sequence = [["A4", "C#5", "E5"]] # Single chord for simplicity
        self.mock_score.nchords = 1
        self.mock_score.nintervals = [3]
        self.mock_score.fracbins = np.array([0., 1.])
        self.mock_score.timebins = np.array([0., 10.])
        self.mock_score.pitch_binning = 'adaptive'

        # Mock Sources
        self.mock_sources = MagicMock(spec=Events) # Using Events for simplicity
        self.mock_sources.n_sources = 2
        self.mock_sources.mapping = {
            'time': [0.0, 0.5], # Second source starts at half-way point
            'pitch': [0.2, 0.8], # Values for pitch binning
            'volume': [0.8, 0.7],
            'note_length': [1.0, 1.0], # Mapped note_length
            # 'phi' or 'azimuth' might be needed if generator preset doesn't have default
        }

        # Mock Generator (e.g., Synthesizer)
        self.mock_generator = MagicMock(spec=Synthesizer)
        self.mock_generator.samprate = self.samprate
        # Mock the preset that generator.play will use
        self.mock_generator.preset = {
            'volume_envelope': {'A':0.01,'D':0.1,'S':0.8,'R':0.1, 'Ac':0,'Dc':0,'Rc':0, 'level':1.0},
            'azimuth': 0.0, # Default spatialization if not in source mapping
            'polar': 0.5 * np.pi
        }
        # Mock the stream returned by generator.play()
        # It should have length corresponding to note_length + R from generator's preset
        # Default note_length from mapping is 1.0s. Default R from generator preset is 0.1s.
        # So, actual_play_duration = 1.1s
        actual_play_duration = 1.0 + self.mock_generator.preset['volume_envelope']['R']
        self.mock_played_stream_length = int(actual_play_duration * self.samprate)

        self.mock_played_stream = MagicMock(spec=Stream) # Use imported Stream for spec
        self.mock_played_stream.values = np.random.rand(self.mock_played_stream_length)
        self.mock_played_stream.sampfracs = np.linspace(0,1,len(self.mock_played_stream.values), endpoint=False) # endpoint false typical for linspace over samples
        self.mock_generator.play.return_value = self.mock_played_stream

        # Mock audio_channels (can be simple if not testing detailed spatialization)
        self.mock_audio_setup_obj = MagicMock(spec=audio_channels)
        self.mock_audio_setup_obj.Nmics = 2 # Stereo
        self.mock_audio_setup_obj.mics = [MagicMock(), MagicMock()]
        self.mock_audio_setup_obj.mics[0].antenna.return_value = 1.0 # Mic 0 (Left) gets full signal
        self.mock_audio_setup_obj.mics[1].antenna.return_value = 0.5 # Mic 1 (Right) gets half signal
        self.mock_audio_setup_obj.labels = ['L', 'R']
        self.mock_audio_setup_obj.forder = [0,1]
        self.mock_audio_setup_obj.setup = "stereo"


    @patch('strauss.sonification.audio_channels')
    def test_sonification_initialization(self, mock_audio_channels_class):
        mock_audio_channels_class.return_value = self.mock_audio_setup_obj

        soni = Sonification(self.mock_score, self.mock_sources, self.mock_generator, audio_setup='stereo', samprate=self.samprate)

        self.assertIs(soni.score, self.mock_score)
        self.assertIs(soni.sources, self.mock_sources)
        self.assertIs(soni.generator, self.mock_generator)
        self.assertEqual(soni.samprate, self.samprate)
        mock_audio_channels_class.assert_called_once_with(setup='stereo')
        self.assertEqual(len(soni.out_channels), self.mock_audio_setup_obj.Nmics)
        self.assertIn('0', soni.out_channels)
        self.assertIsInstance(soni.out_channels['0'], Stream) # Use imported Stream

    @patch('strauss.sonification.audio_channels')
    @patch('strauss.sonification.render_caption')
    @patch('strauss.sonification.get_ttsMode')
    def test_sonification_render_no_caption(self, mock_get_tts_mode, mock_render_caption, mock_audio_channels_class):
        mock_audio_channels_class.return_value = self.mock_audio_setup_obj
        mock_get_tts_mode.return_value = 'None' # No TTS for this test

        soni = Sonification(self.mock_score, self.mock_sources, self.mock_generator, audio_setup='stereo', samprate=self.samprate, caption=None)
        soni.render()

        self.assertEqual(self.mock_generator.play.call_count, self.mock_sources.n_sources)
        # Check a call to play for the first source
        first_call_args = self.mock_generator.play.call_args_list[0][0][0]
        self.assertEqual(first_call_args['note'], "A4") # First source, pitch 0.2 of 3 notes = index 0
        self.assertEqual(first_call_args['volume'], 0.8)

        # Check that output channels were modified (not all zeros)
        self.assertFalse(np.all(soni.out_channels['0'].values == 0))
        mock_render_caption.assert_not_called()


    @patch('strauss.sonification.audio_channels')
    @patch('strauss.sonification.render_caption')
    @patch('strauss.sonification.get_ttsMode')
    @patch('strauss.sonification.wavfile.read') # For when caption is rendered and read
    def test_sonification_render_with_caption(self, mock_wav_read, mock_get_tts_mode, mock_render_caption, mock_audio_channels_class):
        mock_audio_channels_class.return_value = self.mock_audio_setup_obj
        mock_get_tts_mode.return_value = 'coqui-tts' # Simulate TTS available
        # Simulate wavfile.read for the rendered caption
        dummy_caption_audio = np.array([1,2,3,2,1], dtype=np.int16)
        mock_wav_read.return_value = (self.samprate, dummy_caption_audio)

        caption_text = "Test caption"
        soni = Sonification(self.mock_score, self.mock_sources, self.mock_generator, audio_setup='stereo', samprate=self.samprate, caption=caption_text)
        soni.render()

        mock_render_caption.assert_called_once()
        # args[0] is caption text, args[1] is samprate, args[2] is model, args[3] is path
        self.assertEqual(mock_render_caption.call_args[0][0], caption_text)
        self.assertTrue(hasattr(soni, 'caption_channels'))
        self.assertFalse(np.all(soni.caption_channels['0'].values == 0))


    @patch('strauss.sonification.wavfile.write')
    @patch('strauss.sonification.audio_channels')
    def test_sonification_save(self, mock_audio_channels_class, mock_scipy_wav_write):
        mock_audio_channels_class.return_value = self.mock_audio_setup_obj
        soni = Sonification(self.mock_score, self.mock_sources, self.mock_generator, caption=None)
        # Simulate some data in out_channels
        soni.out_channels['0'].values = np.array([0.1, 0.2])
        soni.out_channels['1'].values = np.array([-0.1, -0.2])
        soni.caption_channels = {'0': MagicMock(spec=Stream), '1': MagicMock(spec=Stream)} # Use imported Stream
        soni.caption_channels['0'].values = np.array([]) # Empty caption for this test
        soni.caption_channels['1'].values = np.array([])


        soni.save("test_save.wav", master_volume=0.9, embed_caption=False)
        mock_scipy_wav_write.assert_called_once()
        args, _ = mock_scipy_wav_write.call_args
        self.assertEqual(args[0], "test_save.wav") # filename
        self.assertEqual(args[1], soni.samprate)   # rate
        # args[2] is the data. Check shape and type.
        self.assertEqual(args[2].shape, (2, 2)) # (nsamples, nchannels)
        self.assertEqual(args[2].dtype, np.int32)
        # Check normalization based on master_volume and data vmax
        # vmax = max(0.2, 0.2) * 1.05 = 0.21
        # norm = 0.9 * (2**31-1) / 0.21
        # Expected data[0,0] = 0.1 * norm
        # For simplicity, just check it was called. Detailed value check is complex.

    # More tests needed for save_stereo, save_combined, notebook_display, hear
    # These will involve mocking wavio, ffmpeg, IPython.display, plt, sounddevice respectively.

if __name__ == '__main__':
    unittest.main()
