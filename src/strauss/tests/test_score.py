import unittest
import numpy as np # Import numpy
from strauss.score import Score, parse_chord_sequence

class TestScore(unittest.TestCase):
    def test_score_creation_time_float(self):
        score = Score(chord_sequence=[["C4"]], length=60.0)
        self.assertEqual(score.length, 60.0)
        self.assertEqual(score.note_sequence, [["C4"]])
        self.assertEqual(score.nchords, 1)
        self.assertEqual(score.nintervals, [1])
        np.testing.assert_array_almost_equal(score.fracbins, np.array([0., 1.]))
        np.testing.assert_array_almost_equal(score.timebins, np.array([0., 60.]))
        self.assertEqual(score.pitch_binning, 'adaptive')

    def test_score_creation_time_str(self):
        score = Score(chord_sequence=[["A4", "E5"]], length="1m 30.5s")
        self.assertEqual(score.length, 90.5)
        self.assertEqual(score.note_sequence, [["A4", "E5"]])
        self.assertEqual(score.nchords, 1)
        self.assertEqual(score.nintervals, [2]) # Number of notes in the chord
        np.testing.assert_array_almost_equal(score.fracbins, np.array([0., 1.]))
        np.testing.assert_array_almost_equal(score.timebins, np.array([0., 90.5]))

    def test_score_creation_chord_str(self):
        # This will call parse_chord_sequence internally, which calls notes.chord_notes
        score = Score(chord_sequence="C_4 | G_3", length=120.0)
        # Expected notes from "C_4": ['C4', 'E4', 'G4']
        # Expected notes from "G_3": ['G3', 'B3', 'D4']
        expected_note_seq = [
            sorted(['C4', 'E4', 'G4']),
            sorted(['G3', 'B3', 'D4'])
        ]
        # Sort internal lists for comparison as pychord might not preserve order
        self.assertEqual([sorted(chord) for chord in score.note_sequence], expected_note_seq)
        self.assertEqual(score.nchords, 2)
        self.assertEqual(score.nintervals, [3, 3]) # 3 notes in Cmaj, 3 notes in Gmaj
        np.testing.assert_array_almost_equal(score.fracbins, np.array([0., 0.5, 1.]))
        np.testing.assert_array_almost_equal(score.timebins, np.array([0., 60., 120.]))

    def test_score_creation_multiple_chords_list(self):
        chord_seq_list = [["C4", "E4", "G4"], ["G3", "B3", "D4"]]
        score = Score(chord_sequence=chord_seq_list, length=10.0)
        self.assertEqual(score.note_sequence, chord_seq_list)
        self.assertEqual(score.nchords, 2)
        self.assertEqual(score.nintervals, [3,3])
        np.testing.assert_array_almost_equal(score.fracbins, np.array([0., 0.5, 1.]))
        np.testing.assert_array_almost_equal(score.timebins, np.array([0., 5., 10.]))

    def test_score_pitch_binning_uniform(self):
        score = Score(chord_sequence=[["C4"]], length=60.0, pitch_binning='uniform')
        self.assertEqual(score.pitch_binning, 'uniform')

    def test_score_invalid_pitch_binning(self):
        with self.assertRaisesRegex(Exception, '"invalid_binning" is not a valid pitch_binning mode'):
            Score(chord_sequence=[["C4"]], length=60.0, pitch_binning='invalid_binning')

    def test_score_length_zero(self):
        score = Score(chord_sequence=[["C4"]], length=0.0)
        self.assertEqual(score.length, 0.0)
        np.testing.assert_array_almost_equal(score.timebins, np.array([0.,0.]))


class TestParseChordSequence(unittest.TestCase):
    def test_parse_single_chord(self):
        seq_str = "Am7_3"
        # Am7 at octave 3: A3, C4, E4, G4
        expected = [sorted(["A3", "C4", "E4", "G4"])]
        result = parse_chord_sequence(seq_str)
        self.assertEqual([sorted(r) for r in result], expected)

    def test_parse_multiple_chords(self):
        seq_str = "Cmaj7_4 | F_3 | Gsus4_3"
        # Cmaj7_4: C4 E4 G4 B4
        # F_3: F3 A3 C4
        # Gsus4_3: G3 C4 D4
        expected = [
            sorted(["C4", "E4", "G4", "B4"]),
            sorted(["F3", "A3", "C4"]),
            sorted(["G3", "C4", "D4"])
        ]
        result = parse_chord_sequence(seq_str)
        self.assertEqual([sorted(r) for r in result], expected)

    def test_parse_with_extra_spaces(self):
        seq_str = " C_2   |   Dm7_3  "
        expected = [
            sorted(["C2", "E2", "G2"]),
            sorted(["D3", "F3", "A3", "C4"])
        ]
        result = parse_chord_sequence(seq_str)
        self.assertEqual([sorted(r) for r in result], expected)

    def test_parse_empty_string(self):
        # This depends on how notes.chord_notes handles empty strings or pychord handles it.
        # current chord_list[i].split('_') on "" will be ['']
        # notes.chord_notes('', '') will likely fail.
        with self.assertRaises(Exception):
            parse_chord_sequence("")

    def test_parse_invalid_chord_name_in_sequence(self):
        seq_str = "C_4 | Invalid_3 | G_3"
        # pychord will raise an error when notes.chord_notes tries to parse "Invalid_3"
        with self.assertRaises(Exception):
            parse_chord_sequence(seq_str)

    def test_parse_missing_octave(self):
        seq_str = "C_4 | Am | G_3" # "Am" is missing octave
        # notes.chord_notes("Am", "") will likely fail due to int conversion of octave or pychord error
        with self.assertRaises(Exception):
            parse_chord_sequence(seq_str)

if __name__ == '__main__':
    unittest.main()
