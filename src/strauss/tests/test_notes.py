import unittest
import numpy as np # Import numpy
from strauss.notes import parse_note, parse_chord, chord_notes, mkey_to_note

class TestNotes(unittest.TestCase):
    def test_parse_note(self):
        # A4 = 440 Hz
        self.assertAlmostEqual(parse_note("A4"), 440.0)
        # C4 (middle C) - A4 is 9 semitones above C4. C4 = 440 * 2^(-9/12)
        # C0 = 440 * 2^(-9/12-4) = 16.351597...
        # C4 = C0 * 2^4
        expected_c4 = 16.351597831287415 * (2**4) # Approx 261.6255
        self.assertAlmostEqual(parse_note("C4"), expected_c4, places=3)
        self.assertAlmostEqual(parse_note("c4"), expected_c4, places=3) # Test lowercase

        # Test sharps and flats
        # C#4 should be one semitone above C4
        self.assertAlmostEqual(parse_note("C#4"), expected_c4 * (2**(1/12)), places=3)
        # Db4 should be the same as C#4
        self.assertAlmostEqual(parse_note("Db4"), expected_c4 * (2**(1/12)), places=3)
        # B3 should be one semitone below C4
        self.assertAlmostEqual(parse_note("B3"), expected_c4 * (2**(-1/12)), places=3)

        # Test different octaves
        self.assertAlmostEqual(parse_note("A5"), 880.0, places=3)
        self.assertAlmostEqual(parse_note("A3"), 220.0, places=3)

        # Test invalid note format
        with self.assertRaises(Exception): # Should be more specific if possible, but current code might raise generic Exception or KeyError
            parse_note("X4")
        with self.assertRaises(Exception):
            parse_note("A") # Missing octave
        with self.assertRaises(Exception):
            parse_note("A4#") # Invalid format

    def test_parse_chord(self):
        # C major: C, E, G
        # C4, E4, G4
        c4_freq = parse_note("C4")
        e4_freq = parse_note("E4") # C4 + 4 semitones
        g4_freq = parse_note("G4") # C4 + 7 semitones

        expected_c_major_freqs = sorted([c4_freq, e4_freq, g4_freq])

        # pychord.Chord("C").components_with_pitch(4) gives ['C4', 'E4', 'G4']
        parsed_c_major_freqs = sorted(parse_chord("C", 4))
        np.testing.assert_array_almost_equal(parsed_c_major_freqs, expected_c_major_freqs, decimal=3)

        # Am7 chord with root octave 3: A3, C4, E4, G4
        a3_freq = parse_note("A3")
        # c4_freq already defined
        # e4_freq already defined
        # g4_freq already defined
        expected_am7_freqs = sorted([a3_freq, c4_freq, e4_freq, g4_freq])
        parsed_am7_freqs = sorted(parse_chord("Am7", 3))
        np.testing.assert_array_almost_equal(parsed_am7_freqs, expected_am7_freqs, decimal=3)

        # Test with string octave
        #parsed_c_major_str_oct = sorted(parse_chord("C", "4"))
        #np.testing.assert_array_almost_equal(parsed_c_major_str_oct, expected_c_major_freqs, decimal=3)

        # Test invalid chord name (pychord should handle this)
        with self.assertRaises(Exception): # pychord might raise its own error
            parse_chord("InvalidChord", 4)

    def test_chord_notes(self):
        # C major in octave 4
        notes_c_major = chord_notes("C", 4)
        self.assertEqual(sorted(notes_c_major), sorted(['C4', 'E4', 'G4']))

        # Am7 chord with root octave 3
        notes_am7 = chord_notes("Am7", 3)
        self.assertEqual(sorted(notes_am7), sorted(['A3', 'C4', 'E4', 'G4']))

        # Test with string octave
        notes_c_major_str_oct = chord_notes("C", "4")
        self.assertEqual(sorted(notes_c_major_str_oct), sorted(['C4', 'E4', 'G4']))

        # Test invalid chord name (pychord should handle this)
        with self.assertRaises(Exception):
            chord_notes("InvalidChord", 4)

    def test_mkey_to_note(self):
        # MIDI key 60 is C4
        self.assertEqual(mkey_to_note(60), "C4")
        # MIDI key 69 is A4
        self.assertEqual(mkey_to_note(69), "A4")
        # MIDI key 21 is A0
        self.assertEqual(mkey_to_note(21), "A0")
        # MIDI key 108 is C8
        self.assertEqual(mkey_to_note(108), "C8")
        # Test sharps
        self.assertEqual(mkey_to_note(61), "C#4")
        self.assertEqual(mkey_to_note(70), "A#4")

        # Test boundaries if relevant (MIDI 0-127)
        self.assertEqual(mkey_to_note(0), "C-1") # As per formula: 0 // 12 - 1 = -1
        self.assertEqual(mkey_to_note(127), "G9")# 127 // 12 - 1 = 10 - 1 = 9. 127 % 12 = 7 (G)

if __name__ == '__main__':
    unittest.main()
