import unittest
import numpy as np
from unittest.mock import patch # Import patch
from strauss.stream import Stream, Buffers

class TestStream(unittest.TestCase):
    def test_stream_creation_seconds(self):
        stream = Stream(length=1.0, samprate=44100)
        self.assertEqual(stream.length, 1.0)
        self.assertEqual(stream.samprate, 44100)
        self.assertEqual(stream._nsamp_stream, 44100)
        self.assertEqual(len(stream.values), 44100)
        self.assertEqual(len(stream.samples), 44100)
        self.assertEqual(len(stream.samptime), 44100)
        np.testing.assert_array_equal(stream.values, np.zeros(44100))
        self.assertAlmostEqual(stream.samptime[-1], 1.0 - 1/44100, places=5)

    def test_stream_creation_samples(self):
        stream = Stream(length=1000, samprate=1000, ltype='samples')
        self.assertEqual(stream.length, 1.0) # 1000 samples / 1000 Hz = 1.0s
        self.assertEqual(stream.samprate, 1000)
        self.assertEqual(stream._nsamp_stream, 1000)
        self.assertEqual(len(stream.values), 1000)

    def test_get_sampfracs(self):
        stream = Stream(length=10, samprate=100, ltype='samples') # 0.1s
        stream.get_sampfracs()
        self.assertTrue(hasattr(stream, 'sampfracs'))
        self.assertEqual(len(stream.sampfracs), 10)
        np.testing.assert_array_almost_equal(stream.sampfracs, np.linspace(0, 1, 10))

    def test_reset(self):
        stream = Stream(length=0.1, samprate=1000)
        stream.values = np.random.rand(100)
        stream.bufferize(bufflength=0.05) # 50 samples per buffer
        # Fill buffers with some data to check they are reset
        if hasattr(stream, 'buffers'):
             stream.buffers.buffs_tile = np.random.rand(*stream.buffers.buffs_tile.shape)
             stream.buffers.buffs_olap = np.random.rand(*stream.buffers.buffs_olap.shape)

        stream.reset()
        np.testing.assert_array_equal(stream.values, np.zeros(100))
        if hasattr(stream, 'buffers'):
            np.testing.assert_array_equal(stream.buffers.buffs_tile, np.zeros(stream.buffers.buffs_tile.shape))
            np.testing.assert_array_equal(stream.buffers.buffs_olap, np.zeros(stream.buffers.buffs_olap.shape))

    @patch('strauss.stream.wavio.write')
    def test_save_wav(self, mock_wavio_write):
        stream = Stream(length=0.01, samprate=1000)
        stream.values = np.arange(10) / 10.0
        filename = "test_output.wav"
        stream.save_wav(filename)
        mock_wavio_write.assert_called_once()
        args, kwargs = mock_wavio_write.call_args
        self.assertEqual(args[0], filename)
        np.testing.assert_array_equal(args[1], stream.values)
        self.assertEqual(args[2], stream.samprate)
        self.assertEqual(kwargs['sampwidth'], 3)

    # test_filt_sweep requires a filter function and is more complex,
    # will add if time permits or if specifically requested.
    # It also requires buffers to be initialized.

class TestBuffers(unittest.TestCase):
    def setUp(self):
        self.samprate = 1000
        self.stream_length_seconds = 0.1 # 100 samples
        self.stream = Stream(self.stream_length_seconds, self.samprate)
        self.stream.values = np.arange(self.stream._nsamp_stream) # 0 to 99

    def test_buffers_creation_even_split(self):
        # Buffer length 0.02s = 20 samples. Stream 100 samples.
        # _nsamp_halfbuff = 10, _nsamp_buff = 20
        # _nbuffs = 1 + (100 // 20) = 1 + 5 = 6
        # nsamp_padstream = 6 * 20 = 120
        # nsamp_pad = 120 - 100 = 20
        buff = Buffers(self.stream, bufflength=0.02)
        self.assertEqual(buff._nsamp_buff, 20)
        self.assertEqual(buff._nsamp_halfbuff, 10)
        self.assertEqual(buff._nbuffs, 6) # (100 samples / 20 samples_per_buffer) + 1 = 6
        self.assertEqual(buff.nsamp_padstream, 120)
        self.assertEqual(buff.nsamp_pad, 20)

        self.assertEqual(buff.buffs_tile.shape, (6, 20))
        self.assertEqual(buff.buffs_olap.shape, (5, 20)) # _nbuffs - 1

        # Check content of first tile buffer
        expected_first_tile = np.concatenate([np.arange(20), np.zeros(0)]).astype(float) # Explicitly float
        np.testing.assert_array_almost_equal(buff.buffs_tile[0], expected_first_tile)

        # Check content of first overlap buffer (starts at halfbuff of stream values)
        # stream.values[10:10+20] = stream.values[10:30]
        expected_first_olap = np.concatenate([np.arange(10, 30), np.zeros(0)]).astype(float)
        np.testing.assert_array_almost_equal(buff.buffs_olap[0], expected_first_olap)

    def test_buffers_creation_uneven_split(self):
        # Buffer length 0.03s = 30 samples. Stream 100 samples.
        # _nsamp_halfbuff = 15, _nsamp_buff = 30
        # _nbuffs = 1 + (100 // 30) = 1 + 3 = 4
        # nsamp_padstream = 4 * 30 = 120
        # nsamp_pad = 120 - 100 = 20
        buff = Buffers(self.stream, bufflength=0.03)
        self.assertEqual(buff._nsamp_buff, 30)
        self.assertEqual(buff._nsamp_halfbuff, 15)
        self.assertEqual(buff._nbuffs, 4)
        self.assertEqual(buff.nsamp_padstream, 120)
        self.assertEqual(buff.nsamp_pad, 20)

        self.assertEqual(buff.buffs_tile.shape, (4, 30))
        self.assertEqual(buff.buffs_olap.shape, (3, 30))

    def test_to_stream_reconstruction(self):
        # Use a simple signal and short buffers to trace
        self.stream.values = np.array([1.0] * 10 + [2.0] * 10 + [3.0] * 10) # 30 samples
        self.stream._nsamp_stream = 30
        self.stream.samptime = np.arange(30) / self.samprate

        # bufflength 0.02s => 20 samples per buffer (>= 20 sample minimum)
        # For stream of 30 samples:
        # _nsamp_halfbuff = 10, _nsamp_buff = 20
        # _nbuffs = 1 + (30 // 20) = 1 + 1 = 2
        # nsamp_padstream = 2 * 20 = 40. nsamp_pad = 10.
        buff = Buffers(self.stream, bufflength=0.02) # Changed from 0.01 to 0.02

        # Make copies for manipulation, as to_stream modifies them in place if not careful
        original_tiles = buff.buffs_tile.copy()
        original_olaps = buff.buffs_olap.copy()

        # Apply some modification to check reconstruction (e.g., scale them)
        buff.buffs_tile *= 0.5
        buff.buffs_olap *= 0.5

        reconstructed_stream_vals = buff.to_stream()
        self.assertEqual(len(reconstructed_stream_vals), self.stream._nsamp_stream)

        # Manual reconstruction for comparison (simplified, assuming hann window effect)
        # This is hard to perfectly replicate without running the exact fade logic.
        # The test here is more about whether it runs and returns the correct shape.
        # A more robust test would be to use a known signal, filter it manually (conceptually),
        # and then compare the to_stream output.
        # For now, let's check if the energy changes somewhat predictably if we scale buffers

        original_padded_stream = np.pad(self.stream.values, (0, buff.nsamp_pad))
        expected_reconstructed_no_fade_modification = np.zeros_like(original_padded_stream)

        # Simulate the sum with scaling, without exact windowing for simplicity of this test part
        flat_tiles_scaled = (original_tiles * 0.5).flatten()
        flat_olaps_scaled = (original_olaps * 0.5).flatten()

        expected_reconstructed_no_fade_modification[:len(flat_tiles_scaled)] += flat_tiles_scaled

        # Overlap part needs careful indexing
        start_olap_idx = buff._nsamp_halfbuff
        end_olap_idx = start_olap_idx + len(flat_olaps_scaled)
        expected_reconstructed_no_fade_modification[start_olap_idx:end_olap_idx] += flat_olaps_scaled

        # The actual to_stream uses windowing (fade), so values will differ.
        # We are mostly checking that the process runs and output has correct length.
        # A simple check: if input is all zeros, output should be all zeros.
        zero_stream = Stream(self.stream_length_seconds, self.samprate) # 100 samples
        # Use bufflength that results in >= 20 samples (e.g. 0.02s for 1000Hz = 20 samples)
        zero_buff = Buffers(zero_stream, bufflength=0.02)
        reconstructed_zeros = zero_buff.to_stream()
        np.testing.assert_array_almost_equal(reconstructed_zeros, np.zeros_like(reconstructed_zeros))

    def test_buffer_exception_small_length(self):
        # bufflength results in < 20 samples per buffer
        # samprate = 1000. bufflength = 0.01s => 10 samples. This should raise Exception.
        # The Buffers class has: if nbuff < 20: Exception(...)
        # Here nbuff is stream.samprate*bufflength, which is samples per buffer.
        with self.assertRaisesRegex(Exception, "Error: buffer length 10.0 samples below lower limit of 20"):
            Buffers(self.stream, bufflength=0.01) # 10 samples/buffer at 1000Hz

if __name__ == '__main__':
    unittest.main()
