import numpy as np
import pychord as chrd
import re

# Tune note system to A440 standard
# Equal temperement

tuneC0 = 440 * pow(2, (-9./12)-4) 

notecount = np.arange(12)
notesharps  = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
noteflats = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
semitone_dict = {**dict(zip(notesharps, notecount)),
                 **dict(zip(noteflats, notecount))}

def parse_note(notename):
    """ 
    Takes scientific pitch name and returns frequency in Hz.
    flat and sharp numbers supported
    """
    nsplit = re.findall("(\D+|\d+)", notename)
    semi = semitone_dict[nsplit[0]]/12.
    octv  = int(nsplit[1])
    return tuneC0*pow(2.,semi+octv)

def parse_chord(chordname, rootoct=3):
    chord = chrd.Chord(chordname)
    notes = chord.components_with_pitch(rootoct)
    frqs = []
    for n in notes:
        frqs.append(parse_note(n))
    return np.array(frqs)

def chord_notes(chordname, rootoct=3):
    chord = chrd.Chord(chordname)
    notes = chord.components_with_pitch(int(rootoct))
    return notes

def mkey_to_note(val):
    from strauss.notes import notesharps
    octv = val // 12 - 1
    semi = val % 12
    return f'{notesharps[semi]}{octv}'
