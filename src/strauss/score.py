""" The :obj:`score` submodule: musical constraints on what :obj:`Sources` can play

This submodule is intended to control any musical aspects of the
sonification, by providing constraints on harmonic and rhythmic
aspects. 

Todo:
    * Incorporate more rhythmic options, such as quantisation
    * Have the :obj:`Score` class handle pitch and rhythm assignment
      from :obj:`Sources` directly, instead of the :obj:`Sonification`
      class. 
    * Support chord change intervals as a mapped parameter
    * Provide template chord charts for users 
"""

from . import stream
from . import notes
import numpy as np
import pychord as chrd
import numbers
import re

class Score:
    """ Class defining the musical score for the sonification
        
    The Score class controls the musical aspects of the
    sonification. Currently supports a `chordal` score, defining a set
    of notes that can be played for each section of the
    simulation. The score also controls the length of the
    sonification. 

    Note:
    	Currently, no rhythmic constraints are incorporated in the
    	score. Chords are divided evenly over the length of the
    	sonification. For example for a one minute sonification
    	(:obj:`length = '1m 0s'`) the :obj:`chord_sequence = "Am7_3 |
    	D9_3 | Gmaj7_2"` plays each chord for 20s each. Chaining the 
    	same chord can be used to change the length of these intervals,
    	(e.g. :obj:`chord_sequence = "F_3 | F_3 | C_4"` plays F for
    	40s and C for 20s, `[["C2","G3","E4"]*37+["G2","D3","B4"]*23]`
        would play the C voicing for 37s and G voicing for 23s). 
    """
    def __init__(self, chord_sequence, length, pitch_binning='adaptive'):
        """
        Args:
         chord_sequence: (:obj:`str` or :obj:`list`): The chord or chord
    	  sequence used for the sonification. If a string, parse using
    	  :obj:`parse_chord_sequence`. If a :obj:`list`, each entry is
    	  a :obj:`list` of strings or floats, representing the notes of a
          chord. notes are represented as strings using scientific pitch
    	  notation, e.g. :obj:`[['C3','E3', 'G3'], ['C3', 'F3', 'A4']]`.
          If floats, take values as note frequency in Hz. NOTE: currently
    	  only supported in combination with the :obj:`Synthesiser`
    	  generator class.
         length: (:obj:`str` or :obj:`float`): the length of the
          sonification. If a string, parse minutes and seconds from
    	  format 'Xm Y.Zs'. If a float, read as seconds.
         pitch_binning (optional, :obj:`str`): pitch binning mode - choose
          from 'adaptive', where sources are binned by the pitch mapping
          such that each interval is represented the same fraction of the
          time, and 'uniform' where the pitch binning is based on uniform
          size bins in the mapped pitch parameter. 
        """
        # check types to handle score length correctly
        if isinstance(length, str):
            regex = "([0-9]*)m\s*([0-9]*.[0-9]*)s"
            reobj = re.match(regex, length, re.M | re.I)
            self.length = float(reobj.group(1))*60. + float(reobj.group(2))            
        else:
            self.length = length        
        
        # check types to handle different chord formats correctly
        if isinstance(chord_sequence, list):
            self.note_sequence = chord_sequence
        if isinstance(chord_sequence, str):
            self.note_sequence = parse_chord_sequence(chord_sequence)

        if pitch_binning in ['adaptive','uniform']:
            self.pitch_binning = pitch_binning
        else:
            raise Exception(
                f"\"{pitch_binning}\" is not a valid pitch_binning mode")        

        # number of chords in the sequence 
        self.nchords = len(self.note_sequence)
        self.nintervals = [len(c) for c in self.note_sequence]
        
        # For now, chords changes are just equally spaced in the timeline
        self.fracbins = np.linspace(0,1, self.nchords+1) 
        self.timebins = self.length * self.fracbins
        
def parse_chord_sequence(chord_sequence):
    """ parse a chord sequence from a string

    Args:
      chord_sequence (:obj:`str`): chord sequence to parse, with chord
    	names appended with '_N' where N is the root octave of the
    	chord, and each chord is separated by a pipe character, '|'.

    Returns:
      note_list (:obj:`list(list)`): the chord sequence represented as
      a list of lists, where each sub-list is a chord comprised of
      strings representing each note in scientific pitch notation
      (e.g. 'A4') 
    """
    chord_list = chord_sequence.split("|")
    note_list = []
    note_frqs = []
    for i in range(len(chord_list)):
        chord_list[i] = chord_list[i].strip()
        chord_list[i] = chord_list[i].split('_')
        chord_notes = notes.chord_notes(*chord_list[i])
        note_list.append(chord_notes)
    return note_list

if __name__ == '__main__':
    chords = "Am7_4 | G_3 | F_3 | E7b9_3 "
    # chords = "Am7_4"
    Score(chords)
