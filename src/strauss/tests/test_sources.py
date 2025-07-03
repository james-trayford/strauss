import unittest
import numpy as np
from unittest.mock import patch # Import patch
from strauss.sources import Source, Events, Objects, mappable, evolvable, param_lim_dict

class TestSource(unittest.TestCase):
    def test_source_creation_valid_quantities(self):
        s = Source(mapped_quantities=['volume', 'pitch'])
        self.assertEqual(s.mapped_quantities, ['volume', 'pitch'])

    def test_source_creation_invalid_quantity(self):
        with self.assertRaisesRegex(Exception, 'Property "invalid_prop" is not recognised'):
            Source(mapped_quantities=['volume', 'invalid_prop'])

    def test_source_creation_theta_phi_polar_azimuth_conflict(self):
        with self.assertRaisesRegex(Exception, '"theta" and "polar" cannot be combined'):
            Source(mapped_quantities=['theta', 'polar'])
        with self.assertRaisesRegex(Exception, '"phi" and "azimuth" cannot be combined'):
            Source(mapped_quantities=['phi', 'azimuth'])

    def test_apply_mapping_functions_basic(self):
        s = Source(mapped_quantities=['volume', 'pitch_shift'])
        s.raw_mapping['volume'] = np.array([0, 0.5, 1])
        s.raw_mapping['pitch_shift'] = np.array([0, 12, 24])
        s.n_sources = 3 # Needs to be set for non-evolving parameters

        s.apply_mapping_functions()

        # Default map_lims (0,1), default param_lims from param_lim_dict
        # volume: (0,1) -> (0,1)
        np.testing.assert_array_almost_equal(s.mapping['volume'], [0, 0.5, 1])
        # pitch_shift: (0,1) input default -> (0,24) output default
        # raw_mapping values are [0, 12, 24].
        # Default map_lims for pitch_shift is (0,1). So these will be clipped to 1.
        # [0,1,1] -> then scaled to (0,24) -> [0, 24, 24]
        # This implies that if map_lims is not provided, the raw data is assumed to be already in [0,1] range for descaling.
        # This needs clarification or careful testing of rescale_values behavior with default lims.
        # rescale_values(x, oldlims=(0,1), newlims=param_lim_dict[key])
        # So, [0,12,24] with oldlims=(0,1) becomes np.clip(([0,12,24]-0)/(1-0),0,1) = [0,1,1]
        # Then [0,1,1] * (24-0) + 0 = [0,24,24]
        np.testing.assert_array_almost_equal(s.mapping['pitch_shift'], [0, 24, 24])

    def test_apply_mapping_functions_custom_map_funcs_and_lims(self):
        s = Source(mapped_quantities=['volume'])
        s.raw_mapping['volume'] = np.array([1, 10, 100]) # e.g. some physical values
        s.n_sources = 3

        map_funcs = {'volume': np.log10}
        # After log10: [0, 1, 2]
        map_lims = {'volume': (0, 2)} # Min/max of log10 values
        param_lims = {'volume': (0, 0.5)} # Target volume 0 to 0.5

        s.apply_mapping_functions(map_funcs=map_funcs, map_lims=map_lims, param_lims=param_lims)
        # Scaled: [0,1,2] from (0,2) to (0,0.5) => [0, 0.25, 0.5]
        np.testing.assert_array_almost_equal(s.mapping['volume'], [0, 0.25, 0.5])

    def test_apply_mapping_percentile_lims(self):
        s = Source(mapped_quantities=['cutoff'])
        s.raw_mapping['cutoff'] = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100]) # 11 points, 100 is outlier
        s.n_sources = 11

        # Use 0% and 90% percentiles for map_lims
        # sorted data: 0,1,2,3,4,5,6,7,8,9,100
        # 0th percentile is 0.
        # 90th percentile is 9 (the 10th element out of 11, index 9).
        map_lims = {'cutoff': ('0%', '90%')}
        # param_lims for cutoff is (0,1) by default
        s.apply_mapping_functions(map_lims=map_lims)

        # Expected processing:
        # map_lims will be (0, 9)
        # raw_data = [0,1,2,3,4,5,6,7,8,9,100]
        # descaled (clipped to 0-1 based on data within (0,9)):
        # [0,1/9,2/9,3/9,4/9,5/9,6/9,7/9,8/9,1,1] (100 gets clipped to 9, then normalized to 1)
        # param_lims for 'cutoff' is (0,1) by default. So mapping is same as descaled.
        expected = np.array([0,1/9,2/9,3/9,4/9,5/9,6/9,7/9,8/9,1,1])
        np.testing.assert_array_almost_equal(s.mapping['cutoff'], expected)

    def test_apply_mapping_evolving_param(self):
        s = Source(mapped_quantities=['volume', 'time_evo']) # 'volume' is evolvable
        s.raw_mapping['time_evo'] = [np.array([0, 1, 2])]
        s.raw_mapping['volume'] = [np.array([0.1, 0.5, 0.9])] # Already in 0-1 for simplicity
        s.n_sources = 1

        s.apply_mapping_functions(map_lims={'volume':(0,1)}, param_lims={'volume':(0,1)})
        self.assertTrue(callable(s.mapping['volume'][0]))
        # Test interpolated values
        interpolated_func = s.mapping['volume'][0]
        self.assertAlmostEqual(interpolated_func(0.5), 0.3) # (0.5-0.1)/(1-0) * (0.5-0.1) + 0.1 is not 0.3.
                                                            # (0.5-0)/(1-0) * (0.5-0.1) + 0.1 = 0.5 * 0.4 + 0.1 = 0.3
                                                            # interp1d: x=[0,1,2], y=[0.1,0.5,0.9]
                                                            # at 0.5 (between x=0, y=0.1 and x=1, y=0.5):
                                                            # slope = (0.5-0.1)/(1-0) = 0.4. y = 0.1 + 0.4 * 0.5 = 0.1 + 0.2 = 0.3
        self.assertAlmostEqual(interpolated_func(1.5), 0.7) # slope = (0.9-0.5)/(2-1) = 0.4. y = 0.5 + 0.4 * 0.5 = 0.5 + 0.2 = 0.7
        self.assertAlmostEqual(interpolated_func(-1), 0.1) # Bounds error false, fill with first value
        self.assertAlmostEqual(interpolated_func(3), 0.9)  # Bounds error false, fill with last value

    def test_apply_mapping_phi_azimuth_wrapping(self):
        s = Source(mapped_quantities=['phi', 'time_evo'])
        # Test wrapping for phi (azimuth)
        # Going from 0.9 (e.g. 324 deg) to 0.1 (e.g. 36 deg) should go via 0/1 (e.g. +72 deg), not -288 deg
        # phi is normalized 0 to 1 (representing 0 to 2pi)
        s.raw_mapping['time_evo'] = [np.array([0, 1])]
        s.raw_mapping['phi'] = [np.array([0.9, 0.1])] # Raw data, assume map_lims (0,1) param_lims (0,1)
        s.n_sources = 1
        s.apply_mapping_functions(map_lims={'phi':(0,1)}, param_lims={'phi':(0,1)})

        interpolated_phi = s.mapping['phi'][0]
        # y = [0.9, 0.1]. ydiff = -0.8. abs(ydiff) > 0.5 is true.
        # ysense = np.sign(-0.8) = -1.
        # y[x > xpre] -= ysense  => y[1] (which is 0.1) -= (-1) => y[1] becomes 1.1
        # So the interpolation is between (0, 0.9) and (1, 1.1)
        # At 0.5: 0.9 + (1.1-0.9)*0.5 = 0.9 + 0.2*0.5 = 0.9 + 0.1 = 1.0
        self.assertAlmostEqual(interpolated_phi(0.5), 1.0)
        # Check value at 0.1: 0.9 + (1.1-0.9)*0.1 = 0.9 + 0.02 = 0.92
        self.assertAlmostEqual(interpolated_phi(0.1), 0.92)


class TestEvents(unittest.TestCase):
    def test_events_from_dict(self):
        data = {'volume': [0.1, 0.2], 'pitch': [0.5, 0.6], 'time': [0, 1]}
        events = Events(mapped_quantities=['volume', 'pitch', 'time'])
        events.fromdict(data)
        self.assertEqual(events.n_sources, 2)
        np.testing.assert_array_equal(events.raw_mapping['volume'], [0.1, 0.2])
        np.testing.assert_array_equal(events.raw_mapping['pitch'], [0.5, 0.6])
        np.testing.assert_array_equal(events.raw_mapping['time'], [0,1])

    @patch('numpy.genfromtxt')
    def test_events_from_file(self, mock_genfromtxt):
        mock_data = np.array([[0.1, 0.5, 0], [0.2, 0.6, 1]])
        mock_genfromtxt.return_value = mock_data

        coldict = {'volume': 0, 'pitch': 1, 'time': 2}
        events = Events(mapped_quantities=['volume', 'pitch', 'time'])
        events.fromfile("dummy.txt", coldict)

        mock_genfromtxt.assert_called_once_with("dummy.txt")
        self.assertEqual(events.n_sources, 2)
        np.testing.assert_array_equal(events.raw_mapping['volume'], mock_data[:,0])
        np.testing.assert_array_equal(events.raw_mapping['pitch'], mock_data[:,1])
        np.testing.assert_array_equal(events.raw_mapping['time'], mock_data[:,2])


class TestObjects(unittest.TestCase):
    def test_objects_from_dict_single_static_values(self):
        # Single source, static values
        data = {'volume': 0.5, 'pitch_shift': 12}
        obj = Objects(mapped_quantities=['volume', 'pitch_shift'])
        obj.fromdict(data)
        self.assertEqual(obj.n_sources, 1)
        self.assertEqual(obj.raw_mapping['volume'], [0.5]) # Should be wrapped in a list
        self.assertEqual(obj.raw_mapping['pitch_shift'], [12])

    def test_objects_from_dict_single_evolving_values(self):
        # Single source, evolving values
        time_evo_data = np.array([0,1,2])
        volume_data = np.array([0.1,0.5,0.9])
        data = {'time_evo': time_evo_data, 'volume': volume_data}
        obj = Objects(mapped_quantities=['time_evo', 'volume'])
        obj.fromdict(data)
        self.assertEqual(obj.n_sources, 1)
        np.testing.assert_array_equal(obj.raw_mapping['time_evo'][0], time_evo_data)
        np.testing.assert_array_equal(obj.raw_mapping['volume'][0], volume_data)

    def test_objects_from_dict_multiple_sources_list_of_lists(self):
        # Multiple sources, list of lists for evolving params
        time_data = [np.array([0,1]), np.array([0,0.5])]
        volume_data = [np.array([0.1,0.2]), np.array([0.8,0.7])]
        data = {'time_evo': time_data, 'volume': volume_data, 'pitch_shift': [5,10]} # Mix of evolving and static

        obj = Objects(mapped_quantities=['time_evo', 'volume', 'pitch_shift'])
        obj.fromdict(data)
        self.assertEqual(obj.n_sources, 2)
        np.testing.assert_array_equal(obj.raw_mapping['time_evo'][0], time_data[0])
        np.testing.assert_array_equal(obj.raw_mapping['time_evo'][1], time_data[1])
        np.testing.assert_array_equal(obj.raw_mapping['volume'][0], volume_data[0])
        np.testing.assert_array_equal(obj.raw_mapping['volume'][1], volume_data[1])
        self.assertEqual(obj.raw_mapping['pitch_shift'], [5,10])

    def test_objects_from_dict_key_missing(self):
        # Test if a mapped quantity is missing from the dict
        data = {'volume': 0.5}
        obj = Objects(mapped_quantities=['volume', 'pitch']) # 'pitch' is missing
        # The code currently raises KeyError when trying to access datadict[key]
        # if the key is missing and it's not the last key in self.mapped_quantities.
        # If 'pitch' was the last key, it would determine n_sources based on the previous key.
        # The line `self.n_sources = np.array(self.raw_mapping[key]).shape[0]` runs for the *last* key.
        # The access `self.raw_mapping[key] = datadict[key]` happens for *all* keys.
        # So, if 'pitch' is not in datadict, it will raise KeyError there.
        with self.assertRaises(KeyError): # Changed from generic Exception to KeyError
            obj.fromdict(data)

    # fromfile for Objects is not implemented in the source code, so no test for it.

if __name__ == '__main__':
    unittest.main()
