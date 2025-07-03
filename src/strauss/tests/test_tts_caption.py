import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from strauss.tts_caption import render_caption, getVoices, TTSIsNotSupported, default_tts_voice
# We'll need to extensively mock TTS and pyttsx3
from pathlib import Path
import os
import pyttsx3 # Import pyttsx3 to ensure it's findable by patcher

# Mock the TTS and pyttsx3 modules at the module level before they are imported by tts_caption
# This is a common strategy for complex external dependencies.

# Create a fake TTS class that mimics Coqui TTS
class MockCoquiTTS:
    def __init__(self, model_name=None, progress_bar=False, gpu=False):
        self.model_name = model_name
        self.last_tts_text = None
        self.last_tts_file_path = None

    def tts_to_file(self, text, file_path):
        self.last_tts_text = text
        self.last_tts_file_path = file_path
        # Simulate creating a dummy wav file
        # Actual wav content doesn't matter much for this test, just that a file is made.
        # For more realism, we could write a minimal valid WAV header.
        with open(file_path, 'wb') as f:
            # Minimal WAV header for an empty PCM audio file (mono, 16-bit, 22050 Hz, 0 data bytes)
            # This helps avoid issues with wavfile.read if it expects a valid header.
            header = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xac\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
            # header for 1 sample at 48000 Hz, mono, 16-bit
            # RIFF size = 36 + 2 = 38 (0x26000000)
            # Subchunk1Size = 16 (0x10000000)
            # NumChannels = 1 (0x0100)
            # SampleRate = 48000 (0x80bb0000)
            # ByteRate = 48000 * 1 * 2 = 96000 (0x00017700)
            # BlockAlign = 2 (0x0200)
            # BitsPerSample = 16 (0x1000)
            # Subchunk2Size = 2 (0x02000000) - 1 sample * 2 bytes
            # Data = 0x0000 (single zero sample)

            # Let's use a known sample rate for the dummy file, e.g., 24000 Hz
            # Minimal WAV: RIFF (size) WAVE fmt  (fmt_size) (format) (channels) (sample_rate) (byte_rate) (block_align) (bits_sample) data (data_size) (data)
            # For 24000 Hz, mono, 16-bit, 0 data bytes:
            # NumChannels = 1, SampleRate = 24000, BitsPerSample = 16
            # ByteRate = SampleRate * NumChannels * BitsPerSample/8 = 24000 * 1 * 2 = 48000
            # BlockAlign = NumChannels * BitsPerSample/8 = 1 * 2 = 2
            # Subchunk1Size = 16
            # Subchunk2Size = 0 (no data)
            # ChunkSize = 4 + (8 + Subchunk1Size) + (8 + Subchunk2Size) = 4 + 24 + 8 = 36
            header = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00' \
                     b'\x80\xbb\x00\x00\x00\xee\x02\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
            # The above header has sample rate 24000 (0x5dc0) and byterate 48000 (0xbb80)
            # Let's make it 48000 Hz: 0xbb80, byte rate 96000 (0x017700)
            # SR 48000 = 0xBB80. BR 96000 = 0x017700
            header = (b'RIFF' + (36).to_bytes(4, 'little') + b'WAVEfmt ' +
                      (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little') + (1).to_bytes(2, 'little') +
                      (24000).to_bytes(4, 'little') + (48000).to_bytes(4, 'little') +
                      (2).to_bytes(2, 'little') + (16).to_bytes(2, 'little') +
                      b'data' + (0).to_bytes(4, 'little'))
            f.write(header)

    @staticmethod
    def list_models():
        return [{"id": "tts_models/en/jenny/jenny", "name": "jenny", "languages": ["en_GB"]}]

# Create a fake pyttsx3 engine
class MockPyttsx3Engine:
    def __init__(self):
        self.last_saved_text = None
        self.last_saved_filename = None
        self._properties = {'rate': 200, 'volume': 1.0, 'voice': 'default_voice_id'}

    def getProperty(self, name):
        if name == 'voices':
            # Directly return the list for debugging, bypassing the @property temporarily
            class MockVoice:
                def __init__(self, id, name): self.id = id; self.name = name
            return [MockVoice("id1", "Voice1"), MockVoice("id2", "Voice2")]
        return self._properties.get(name)

    def save_to_file(self, text, filename, name=None): # name arg added to match signature
        self.last_saved_text = text
        self.last_saved_filename = filename
        # Simulate creating a dummy wav file, similar to MockCoquiTTS
        with open(filename, 'wb') as f:
            header = (b'RIFF' + (36).to_bytes(4, 'little') + b'WAVEfmt ' +
                      (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little') + (1).to_bytes(2, 'little') +
                      (22050).to_bytes(4, 'little') + (44100).to_bytes(4, 'little') + # pyttsx3 often defaults to 22050
                      (2).to_bytes(2, 'little') + (16).to_bytes(2, 'little') +
                      b'data' + (0).to_bytes(4, 'little'))
            f.write(header)


    def runAndWait(self):
        pass

    # Removed the duplicate, simpler getProperty that was here.
    # The correct one that handles 'voices' is already defined above.

    def setProperty(self, name, value):
        self._properties[name] = value

    @property
    def voices(self): # Make voices a property
        class MockVoice:
            def __init__(self, id, name):
                self.id = id
                self.name = name
        return [MockVoice("id1", "Voice1"), MockVoice("id2", "Voice2")]


class MockPyttsx3:
    def __init__(self):
        self.engine = MockPyttsx3Engine()

    def init(self):
        return self.engine

    def getVoices(self, info=False): # Mimic the getVoices in tts_caption
        return self.engine.voices


# Patching at module import time is tricky. A common way is to use unittest.mock.patch.dict for sys.modules.
# However, since tts_caption.py tries to import TTS and pyttsx3 at its top level,
# we need to ensure these mocks are in place *before* tts_caption is first imported by the test runner.
# A simpler way for testing is to control the `ttsMode` variable within tts_caption.py via patching.

@patch('strauss.tts_caption.utils') # For Capturing
@patch('strauss.tts_caption.wavfile') # To control wavfile.read and write
@patch('strauss.tts_caption.ff') # To control ffmpeg if it's called
class TestTTSCaption(unittest.TestCase):

    def tearDown(self):
        # Clean up any dummy files created
        if Path("dummy_caption.wav").exists():
            os.remove("dummy_caption.wav")
        if Path("dummy_caption_pre.wav").exists():
            os.remove("dummy_caption_pre.wav")

    @patch('strauss.tts_caption.ttsMode', 'coqui-tts')
    @patch('strauss.tts_caption.TTS', MockCoquiTTS) # Patch the TTS class used in the module
    def test_render_caption_coqui_tts_mode(self, mock_ffmpeg, mock_wavfile, mock_utils):
        caption_text = "Hello world"
        target_samprate = 48000
        # MockCoquiTTS creates a file with 24000 Hz. We want to test resampling.
        mock_wavfile.read.return_value = (24000, np.array([0,0], dtype=np.int16)) # rate_in, wavobj

        render_caption(caption_text, target_samprate, "tts_models/en/jenny/jenny", "dummy_caption.wav")

        self.assertTrue(Path("dummy_caption.wav").exists())
        # Check that wavfile.read was called (by render_caption to check rate)
        mock_wavfile.read.assert_called_with("dummy_caption.wav") # Changed Path object to string
        # Check that wavfile.write was called (by render_caption to resample and save)
        mock_wavfile.write.assert_called()
        args, _ = mock_wavfile.write.call_args
        self.assertEqual(args[0], "dummy_caption.wav") # path (string comparison)
        self.assertEqual(args[1], target_samprate) # samprate
        # args[2] is the resampled data, harder to check precisely without actual resampling.
        # Check that strauss.utilities.resample was called (implicitly via mock_wavfile.write args)
        # This is hard because resample is called internally if rates don't match.
        # The key is that write is called with target_samprate.

    @patch('strauss.tts_caption.ttsMode', 'pyttsx3')
    @patch('pyttsx3.init') # Patch pyttsx3.init directly
    def test_render_caption_pyttsx3_mode(self, mock_pyttsx3_init_global, mock_ffmpeg, mock_wavfile, mock_utils):
        # This test now needs pyttsx3 to be importable, or sys.modules['pyttsx3'] to be pre-mocked.
        # For the purpose of this test, we assume that if ttsMode is 'pyttsx3', then 'import pyttsx3' succeeded.
        mock_engine_instance = MockPyttsx3Engine()
        mock_pyttsx3_init_global.return_value = mock_engine_instance

        caption_text = "Hello from pyttsx3"
        target_samprate = 44100
        # MockPyttsx3Engine creates a file with 22050 Hz
        mock_wavfile.read.return_value = (22050, np.array([0,0], dtype=np.int16))

        render_caption(caption_text, target_samprate, {'voice':'id1'}, "dummy_caption.wav")

        self.assertTrue(Path("dummy_caption.wav").exists())
        self.assertEqual(mock_engine_instance.last_saved_text, caption_text)
        self.assertEqual(mock_engine_instance.last_saved_filename, "dummy_caption.wav") # Changed Path to string

        mock_wavfile.read.assert_called_with("dummy_caption.wav") # Changed Path object to string
        mock_wavfile.write.assert_called()
        args, _ = mock_wavfile.write.call_args
        self.assertEqual(args[0], "dummy_caption.wav") # Changed Path object to string
        self.assertEqual(args[1], target_samprate)

    @patch('strauss.tts_caption.ttsMode', 'None')
    def test_render_caption_no_tts_mode(self, mock_ffmpeg, mock_wavfile, mock_utils):
        with self.assertRaises(TTSIsNotSupported):
            render_caption("No TTS", 48000, "any_model", "dummy_caption.wav")

    @patch('strauss.tts_caption.ttsMode', 'coqui-tts')
    # getVoices in coqui mode does not use TTS class, ff, wavfile, or utils.
    # So, only the ttsMode patch is strictly needed by the method's logic.
    # The class patches will still be passed if not overridden.
    # To avoid confusion, we ensure the signature accepts what class patches provide.
    # Based on error "missing 2 required positional arguments: 'mock_wavfile_patch' and 'mock_utils_patch'"
    # when signature was (self, mock_TTS_class_arg, mock_tts_mode_arg, mock_ff_patch, mock_wavfile_patch, mock_utils_patch)
    # This implies it received self + 3 mocks.
    # Method decorators: TTS (1st passed), ttsMode (2nd passed). Innermost class: ff (3rd passed)
    def test_get_voices_coqui(self, mock_TTS_class_arg, mock_tts_mode_arg, mock_ff_patch):
        from strauss import utilities as strauss_utils
        # Patch utils specifically for this call context if not reliably passed
        with patch('strauss.tts_caption.utils') as mock_utils_for_this_call:
            expected_voices_list = strauss_utils.get_supported_coqui_voices()
            mock_utils_for_this_call.get_supported_coqui_voices.return_value = expected_voices_list

            voices = getVoices(info=False)
            self.assertEqual(voices, expected_voices_list)
            mock_utils_for_this_call.get_supported_coqui_voices.assert_called_once()


    @patch('strauss.tts_caption.ttsMode', 'pyttsx3')
    @patch('pyttsx3.init')
    # Method args: pyttsx3_init, ttsMode. Class args: ff, wavfile
    def test_get_voices_pyttsx3(self, mock_pyttsx3_init_arg, mock_tts_mode_arg, mock_ff_patch, mock_wavfile_patch):
        mock_engine_instance = MockPyttsx3Engine()
        mock_pyttsx3_init_arg.return_value = mock_engine_instance

        voices = getVoices(info=False)
        self.assertEqual(len(voices), 2)
        self.assertEqual(voices[0].name, "Voice1")

    @patch('strauss.tts_caption.ttsMode', 'None')
    # Method arg: ttsMode. Class args: ff, wavfile
    def test_get_voices_no_tts(self, mock_tts_mode_arg, mock_ff_patch, mock_wavfile_patch):
        voices = getVoices(info=False)
        self.assertEqual(len(voices), 1)
        self.assertEqual(voices[0]['voices'], "None") # As per current getVoices logic

    # Test for ffmpeg fallback in render_caption if wavfile.read fails initially
    @patch('strauss.tts_caption.ttsMode', 'pyttsx3')
    @patch('pyttsx3.init') # Patch pyttsx3.init directly
    def test_render_caption_pyttsx3_ffmpeg_fallback(self, mock_pyttsx3_init_global, mock_ffmpeg_module, mock_wavfile_module, mock_utils_module):
        mock_engine_instance = MockPyttsx3Engine()
        mock_pyttsx3_init_global.return_value = mock_engine_instance

        # Simulate initial wavfile.read failure, then success after ffmpeg conversion
        mock_wavfile_module.read.side_effect = [
            ValueError("Simulated initial read error"), # First call fails
            (22050, np.array([0,0], dtype=np.int16))    # Second call (after ffmpeg) succeeds
        ]
        # Mock ffmpeg calls
        mock_ffmpeg_input = mock_ffmpeg_module.input.return_value
        mock_ffmpeg_output = mock_ffmpeg_input.output.return_value

        caption_text = "Testing ffmpeg fallback"
        target_samprate = 44100
        render_caption(caption_text, target_samprate, {}, "dummy_caption.wav")

        # The file "dummy_caption.wav" might not exist if wavfile.write is fully mocked.
        # The crucial part is that the ffmpeg process was invoked and then wavfile.write was called for the final output.
        mock_ffmpeg_module.input.assert_called_once_with('dummy_caption_pre.wav')
        mock_ffmpeg_output.run.assert_called_once_with(quiet=1)
        self.assertEqual(mock_wavfile_module.read.call_count, 2) # Initial read (fails), read after ffmpeg (succeeds)
        mock_wavfile_module.write.assert_called_once() # This is the important check for final output
        # Ensure the final write is to "dummy_caption.wav" with the target sample rate
        final_write_args, _ = mock_wavfile_module.write.call_args
        self.assertEqual(final_write_args[0], "dummy_caption.wav")
        self.assertEqual(final_write_args[1], target_samprate)

if __name__ == '__main__':
    unittest.main()
