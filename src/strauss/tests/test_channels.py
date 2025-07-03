import unittest
import numpy as np
from unittest.mock import patch
from strauss.channels import mic, audio_channels

class TestMic(unittest.TestCase):
    def test_mic_creation_default(self):
        m = mic(azimuth=0.0)
        self.assertEqual(m.azimuth, 0.0)
        self.assertEqual(m.mic_type, "directional")
        self.assertEqual(m.label, "C")
        self.assertEqual(m.channel, 1)
        self.assertTrue(callable(m.antenna))

    def test_mic_creation_custom(self):
        m = mic(azimuth=1.5, mic_type="omni", label="L", channel=0)
        self.assertEqual(m.azimuth, 1.5)
        self.assertEqual(m.mic_type, "omni")
        self.assertEqual(m.label, "L")
        self.assertEqual(m.channel, 0)
        self.assertTrue(callable(m.antenna))

    def test_mic_types_antenna(self):
        # Directional
        m_dir = mic(azimuth=0.0, mic_type="directional")
        # Antenna should be max at its azimuth, and half at +/- 90 deg
        self.assertAlmostEqual(m_dir.antenna(0.0, 0.5*np.pi), 1.0) # cos(0)*sin(pi/2) = 1 -> 0.5*(1+1)=1
        self.assertAlmostEqual(m_dir.antenna(0.5*np.pi, 0.5*np.pi), 0.5) # cos(pi/2)*sin(pi/2)=0 -> 0.5*(1+0)=0.5
        self.assertAlmostEqual(m_dir.antenna(np.pi, 0.5*np.pi), 0.0) # cos(pi)*sin(pi/2)=-1 -> 0.5*(1-1)=0

        # Omni
        m_omni = mic(azimuth=0.0, mic_type="omni")
        self.assertEqual(m_omni.antenna(0.0), 1.0)
        self.assertEqual(m_omni.antenna(np.pi), 1.0)

        # Mute
        m_mute = mic(azimuth=0.0, mic_type="mute")
        self.assertEqual(m_mute.antenna(0.0), 0.0)
        self.assertEqual(m_mute.antenna(np.pi), 0.0)

        # Ambisonic (basic check, detailed math is complex)
        # ACN 0 (W channel - omni)
        m_ambi_0 = mic(azimuth=0, mic_type="ambisonic") # azimuth is ACN
        self.assertTrue(callable(m_ambi_0.antenna))
        # Test a known value for Y00 (l=0, m=0) up to normalization
        # lpmv(0,0,cos(b)) = P0(cos(b)) = 1. tfunc(0*a) = cos(0) = 1. normSN3D for l=0,m=0 should be 1/sqrt(4pi) * sqrt(4pi) = 1
        # The code has normSN3D = np.sqrt((2-(0**mabs)/4*np.pi) * fctrl(l-mabs)/fctrl(l+mabs))
        # For l=0, m=0, mabs=0: normSN3D = sqrt((2-1)/4pi * 1/1) = sqrt(1/4pi) -- this is standard N3D
        # The code has: normSN3D = np.sqrt((2-(0**mabs)/4*np.pi) * fctrl(l-mabs)/fctrl(l+mabs))
        # For l=0, m=0 (ACN=0): sqrt((2-1)/(4*pi) * 1/1) = 1/sqrt(4pi) - this is N3D norm for Y00
        # The actual formula used is: normSN3D * pow(-1,mabs) * lpmv(mabs, l, np.cos(b)) * tfunc(mabs*a)
        # For ACN=0 (l=0, m=0): (1/sqrt(4pi)) * 1 * 1 * 1. This is Y00 in N3D.
        # The code uses ambiX standard: l = int(acn**0.5), m = acn - l*(l+1)
        # For acn=0, l=0, m=0.
        # normSN3D = np.sqrt((2-(0**0)/(4*np.pi)) * factorial(0)/factorial(0)) = np.sqrt((2-1)/(4*np.pi)) = 1/np.sqrt(4*np.pi)
        # antenna_val = (1/np.sqrt(4*np.pi)) * (1)**0 * lpmv(0,0,np.cos(0.5*np.pi)) * np.cos(0)
        # lpmv(0,0,0) = 1. So antenna_val = 1/np.sqrt(4*np.pi)
        # This seems like the unnormalized spherical harmonic value * N3D constant.
        # For W channel (omni), it should be constant.
        # Let's check for a few angles, should be same (up to normalization factor)
        # self.assertAlmostEqual(m_ambi_0.antenna(0.0, 0.5*np.pi), m_ambi_0.antenna(np.pi, 0.5*np.pi))
        # The test for ambisonic is a bit complex due to normalization constants and spherical harmonics.
        # A simple check: ACN 0 should be omnidirectional.
        val1 = m_ambi_0.antenna(np.random.rand()*2*np.pi, np.random.rand()*np.pi)
        val2 = m_ambi_0.antenna(np.random.rand()*2*np.pi, np.random.rand()*np.pi)
        # For Y00, it is constant.
        # lpmv(0,0,cos(b)) is P_0(cos(b)) = 1. tfunc(0*a)=cos(0)=1. So it should be constant.
        self.assertAlmostEqual(m_ambi_0.antenna(0.1, 0.2), m_ambi_0.antenna(1.0, 1.0))


        # ACN 1 (Y channel - front-back dipole)
        m_ambi_1 = mic(azimuth=1, mic_type="ambisonic") # ACN 1 -> l=1, m=-1 (sin)
        # Y_1^-1 = sqrt(3/8pi) sin(theta) sin(phi)
        # Here, azimuth is phi, 0.5*pi is theta usually.
        # antenna(a, b=0.5*np.pi) where a is azimuth (phi), b is polar angle (theta)
        # l=1, m=-1. mabs=1. tfunc=sin.
        # norm = sqrt((2-0)/(4pi) * fac(0)/fac(2)) = sqrt(1/2pi * 1/2) = sqrt(1/4pi) -- this is wrong.
        # AmbiX standard for SN3D: sqrt((2-delta_m0)/(4pi) * (l-abs(m))! / (l+abs(m))!)
        # For l=1, m=-1: sqrt((2-0)/(4pi) * (1-1)!/(1+1)!) = sqrt(1/(2pi) * 1/2) = 1/sqrt(4pi)
        # lpmv(1,1,cos(b)). (-1)^1 * lpmv(1,1,cos(b)) * sin(1*a)
        # P_1^1(x) = -sqrt(1-x^2). Here lpmv is SciPy's which is (-1)^m P_l^m.
        # So scipy.lpmv(1,1,cos(b)) = P_1^1(cos(b)).
        # It becomes: (1/sqrt(4pi)) * (-1)^1 * P_1^1(cos(b)) * sin(a)
        # For b = pi/2 (horizontal plane), cos(b)=0. P_1^1(0) = -sqrt(1-0^2) = -1.
        # So, (1/sqrt(4pi)) * (-1) * (-1) * sin(a) = (1/sqrt(4pi)) * sin(a)
        # Max at a=pi/2 (left), min at a=3pi/2 (right)
        self.assertGreater(m_ambi_1.antenna(0.5*np.pi), 0) # Positive for sin(pi/2)
        self.assertLess(m_ambi_1.antenna(1.5*np.pi), 0)   # Negative for sin(3pi/2)
        self.assertAlmostEqual(m_ambi_1.antenna(0), 0)    # Zero for sin(0)
        self.assertAlmostEqual(m_ambi_1.antenna(np.pi), 0) # Zero for sin(pi)

    def test_mic_invalid_type(self):
        with self.assertRaisesRegex(Exception, 'Mic type "invalid" unknown.'):
            mic(azimuth=0.0, mic_type="invalid")

class TestAudioChannels(unittest.TestCase):
    def test_audio_channels_presets(self):
        setups = ["mono", "stereo", "5.1", "7.1", "ambiX1"] # ambiX0, ambiX2, etc.
        expected_n_mics = {
            "mono": 1, "stereo": 2, "5.1": 6, "7.1": 8,
            "ambiX0": 1, # l=0 (1 channel)
            "ambiX1": 4, # l=0 (1) + l=1 (3 channels) = 4
            "ambiX2": 9  # l=0 (1) + l=1 (3) + l=2 (5) = 9
        }
        for setup_name in setups:
            if setup_name.startswith("ambiX"):
                order = int(setup_name[-1])
                expected_mics = sum(2 * l + 1 for l in range(order + 1))
            else:
                expected_mics = expected_n_mics[setup_name]

            ac = audio_channels(setup=setup_name)
            self.assertEqual(ac.setup, setup_name)
            self.assertEqual(len(ac.mics), expected_mics)
            self.assertEqual(ac.Nmics, expected_mics)
            for i, m in enumerate(ac.mics):
                self.assertIsInstance(m, mic)
                if not setup_name.startswith("ambiX"):
                     # Ambisonic azimuth is ACN, not angle
                    self.assertIn(m.mic_type, ["omni", "directional", "mute"])
                else:
                    self.assertEqual(m.mic_type, "ambisonic")
                self.assertEqual(m.channel, i + 1) # Channels are 1-indexed in mic obj

    def test_audio_channels_custom(self):
        custom_setup_dict = {
            "azimuths": [0.0, np.pi],
            "types": ["omni", "directional"],
            "labels": ["front", "back"],
            "forder": [0,1]
        }
        ac = audio_channels(setup="custom", custom_setup=custom_setup_dict)
        self.assertEqual(ac.setup, "custom")
        self.assertEqual(ac.Nmics, 2)
        self.assertEqual(ac.mics[0].azimuth, 0.0)
        self.assertEqual(ac.mics[0].mic_type, "omni")
        self.assertEqual(ac.mics[0].label, "front")
        self.assertEqual(ac.mics[1].azimuth, np.pi)
        self.assertEqual(ac.mics[1].mic_type, "directional")
        self.assertEqual(ac.mics[1].label, "back")
        self.assertEqual(ac.forder, [0,1])

    def test_audio_channels_custom_missing_params(self):
        with self.assertRaisesRegex(Exception, "Custom setup requested but custom_setup parameters empty."):
            audio_channels(setup="custom")

    def test_audio_channels_custom_params_not_custom_setup(self):
        custom_setup_dict = {"azimuths": [0.0], "types": ["omni"], "labels": ["C"]}
        with self.assertWarnsRegex(UserWarning, "custom_setup variable non-empty, but not using custom setup."):
            audio_channels(setup="mono", custom_setup=custom_setup_dict)

    def test_audio_channels_invalid_setup(self):
        with self.assertRaisesRegex(Exception, 'setup "invalid_setup" not understood'):
            audio_channels(setup="invalid_setup")

    @patch('strauss.channels.plt') # Patching plt as used in the channels module
    def test_plot_antenna(self, mock_plt):
        ac = audio_channels(setup="stereo")

        # Make sure the mock figure and axes are returned correctly
        mock_figure_instance = mock_plt.figure.return_value
        mock_axes_instance = mock_figure_instance.subplots.return_value
        mock_plt.gca.return_value = mock_axes_instance # if plt.gca() is used

        fig_obj = ac.plot_antenna()

        mock_plt.figure.assert_called_once_with(figsize=(8, 8))
        mock_figure_instance.subplots.assert_called_once_with(subplot_kw={'projection': 'polar'})

        # Check that plotting happened on an axes object (ax.plot)
        # and directly via plt.plot for the summary line
        self.assertTrue(mock_axes_instance.plot.called)
        self.assertTrue(mock_plt.plot.called)

        self.assertIs(fig_obj, mock_figure_instance) # Ensure the created figure is returned

if __name__ == '__main__':
    unittest.main()
