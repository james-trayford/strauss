from . import stream
from . import notes
import numpy as np
import pychord as chrd
import numbers
import re

# TO DO:
# - Ultimately have Synth and Sampler classes that own their own stream (stream.py) object
#   allowing ADSR volume and filter enveloping, LFO implementation etc.
# - Functions here will generally be called from a "Score" class that is provided with the
#   musical choices and uses these to generate sound, but can be interfaced with directly.

class Score:
    def __init__(self, chord_sequence, length):

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

        # number of chords in the sequence 
        self.nchords = len(self.note_sequence)
        self.nintervals = [len(c) for c in self.note_sequence]
        
        # For now, chords changes are just equally spaced in the timeline
        self.fracbins = np.linspace(0,1, self.nchords+1) 
        self.timebins = self.length * self.fracbins
        
def parse_chord_sequence(chord_sequence):
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
