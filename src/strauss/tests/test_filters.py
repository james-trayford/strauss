import unittest
import numpy as np
from strauss.filters import LPF1, HPF1

class TestFilters(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 44100
        self.duration = 1.0
        self.num_samples = int(self.sample_rate * self.duration)
        self.time = np.linspace(0, self.duration, self.num_samples, endpoint=False)
        # Create a test signal: sum of two sines (e.g., 100 Hz and 1000 Hz)
        self.signal_low_freq = np.sin(2 * np.pi * 100 * self.time)
        self.signal_high_freq = np.sin(2 * np.pi * 1000 * self.time)
        self.test_signal = self.signal_low_freq + self.signal_high_freq
        self.test_signal_zeros = np.zeros_like(self.test_signal)

    def get_power_at_freq(self, signal, freq, sample_rate):
        """Helper to estimate power at a specific frequency."""
        fft_signal = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal), 1/sample_rate)
        # Find the closest frequency bin
        idx = np.argmin(np.abs(fft_freq - freq))
        return np.abs(fft_signal[idx])**2

    def test_LPF1_passes_low_attenuates_high(self):
        cutoff_freq = 500  # Hz
        # Cutoff for digital filter is normalized to Nyquist frequency (sample_rate/2)
        # However, the filter functions in strauss.filters seem to take it directly in Hz
        # and then scipy.signal.butter handles the normalization if analog=False.
        # The provided LPF1 and HPF1 use analog=False, and the cutoff is a fraction of Nyquist if < 1.
        # The documentation for scipy.signal.butter states for digital filters:
        # "Wnndarray or float. Critical frequency or frequencies. For digital filters, Wn is normalized from 0 to 1,
        # where 1 is the Nyquist frequency, pi radians/sample. (Wn is thus in half-cycles / sample.)"
        # The strauss code passes cutoff directly. Let's assume it means normalized cutoff frequency.
        # If cutoff is 500Hz and sample_rate is 44100Hz, Nyquist is 22050Hz.
        # Normalized cutoff = 500 / 22050 = 0.02267

        # The filters.py LPF1 takes cutoff, q, order.
        # `sig.butter(order, cutoff, ...)` -> cutoff here is the normalized frequency.
        # It seems the `cutoff` parameter in LPF1/HPF1 is intended to be the normalized frequency (0 to 1).
        # Let's test this assumption.

        normalized_cutoff = cutoff_freq / (self.sample_rate / 2.0)
        q_factor = 0.707 # Typical for Butterworth
        order = 5

        filtered_signal = LPF1(self.test_signal, normalized_cutoff, q_factor, order)

        power_low_before = self.get_power_at_freq(self.test_signal, 100, self.sample_rate)
        power_high_before = self.get_power_at_freq(self.test_signal, 1000, self.sample_rate)

        power_low_after = self.get_power_at_freq(filtered_signal, 100, self.sample_rate)
        power_high_after = self.get_power_at_freq(filtered_signal, 1000, self.sample_rate)

        # Low frequency (100 Hz) should pass with minimal attenuation
        self.assertGreater(power_low_after, 0.1 * power_low_before, "Low frequency component significantly attenuated by LPF")
        # High frequency (1000 Hz) should be significantly attenuated
        self.assertLess(power_high_after, 0.1 * power_high_before, "High frequency component not significantly attenuated by LPF")

    def test_HPF1_passes_high_attenuates_low(self):
        cutoff_freq = 500  # Hz
        normalized_cutoff = cutoff_freq / (self.sample_rate / 2.0)
        q_factor = 0.707
        order = 5

        filtered_signal = HPF1(self.test_signal, normalized_cutoff, q_factor, order)

        power_low_before = self.get_power_at_freq(self.test_signal, 100, self.sample_rate)
        power_high_before = self.get_power_at_freq(self.test_signal, 1000, self.sample_rate)

        power_low_after = self.get_power_at_freq(filtered_signal, 100, self.sample_rate)
        power_high_after = self.get_power_at_freq(filtered_signal, 1000, self.sample_rate)

        # Low frequency (100 Hz) should be significantly attenuated
        self.assertLess(power_low_after, 0.1 * power_low_before, "Low frequency component not significantly attenuated by HPF")
        # High frequency (1000 Hz) should pass with minimal attenuation
        self.assertGreater(power_high_after, 0.1 * power_high_before, "High frequency component significantly attenuated by HPF")

    def test_LPF1_with_zero_signal(self):
        normalized_cutoff = 0.1
        q_factor = 0.707
        order = 5
        filtered_signal = LPF1(self.test_signal_zeros, normalized_cutoff, q_factor, order)
        np.testing.assert_array_almost_equal(filtered_signal, self.test_signal_zeros)

    def test_HPF1_with_zero_signal(self):
        normalized_cutoff = 0.1
        q_factor = 0.707
        order = 5
        filtered_signal = HPF1(self.test_signal_zeros, normalized_cutoff, q_factor, order)
        np.testing.assert_array_almost_equal(filtered_signal, self.test_signal_zeros)

    def test_LPF1_cutoff_at_nyquist(self): # Should pass everything (or be close to original)
        normalized_cutoff = 1.0
        q_factor = 0.707
        order = 5
        # A cutoff of 1.0 for a digital butterworth filter can be problematic in some scipy versions or lead to an identity filter.
        # Let's use a value very close to 1.0 but not exactly 1.0
        almost_nyquist_cutoff = 0.99 # Adjusted from 0.9999
        filtered_signal = LPF1(self.test_signal, almost_nyquist_cutoff, q_factor, order)
        # Check if the signal energy is largely preserved
        original_energy = np.sum(self.test_signal**2)
        filtered_energy = np.sum(filtered_signal**2)
        self.assertAlmostEqual(filtered_energy, original_energy, delta=original_energy*0.1, msg="LPF at Nyquist significantly altered signal energy")


    def test_HPF1_cutoff_at_zero(self): # Should pass everything
        # Set cutoff well below the lowest frequency component (100 Hz)
        # e.g., 10 Hz. Normalized: 10 / (44100/2) = 10 / 22050 = 0.0004535
        normalized_cutoff = 10.0 / (self.sample_rate / 2.0)
        q_factor = 0.707
        order = 5
        filtered_signal = HPF1(self.test_signal, normalized_cutoff, q_factor, order)
        original_energy = np.sum(self.test_signal**2)
        filtered_energy = np.sum(filtered_signal**2)
        self.assertAlmostEqual(filtered_energy, original_energy, delta=original_energy*0.1, msg="HPF at zero significantly altered signal energy")

    def test_LPF1_invalid_cutoff(self):
        # Cutoff should be between 0 and 1 for scipy.signal.butter Wn
        with self.assertRaises(ValueError): # Scipy butter should raise ValueError for invalid Wn
             LPF1(self.test_signal, 1.5, 0.707, 5) # Cutoff > 1
        with self.assertRaises(ValueError):
             LPF1(self.test_signal, -0.5, 0.707, 5) # Cutoff < 0

    def test_HPF1_invalid_cutoff(self):
        with self.assertRaises(ValueError):
             HPF1(self.test_signal, 1.5, 0.707, 5)
        with self.assertRaises(ValueError):
             HPF1(self.test_signal, -0.5, 0.707, 5)

if __name__ == '__main__':
    unittest.main()
