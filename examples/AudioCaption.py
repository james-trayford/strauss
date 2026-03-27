#!/usr/bin/env python
# coding: utf-8

# ### <u> Generate a sonification with an audio caption in `strauss` </u>
# Import the relevant modules:
# 
# ***Note***: you will need to have some form of python text-to-speech installed (`TTS` or `pyttsx3`) for these examples to work. See the error raised when trying to run the examples below for more info:
from strauss.sonification import Sonification
from strauss.sources import Events, Objects
from strauss import channels
from strauss.score import Score
from strauss.tts_caption import set_engine, render_caption, get_ttsMode
import numpy as np
from strauss.generator import Sampler, Synthesizer
import matplotlib.pyplot as plt
import os
from pathlib import Path
import strauss
import pyttsx3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--animate", action="store_true",
                    help="create an animation of plot")
args = parser.parse_args()

# Set TTS module to 'kokoro', 'coqui-tts' or 'pyttsx3'
set_engine('kokoro')
mode = get_ttsMode()

# What text to speech do we have?
print(f"Available text-to-speech (TTS) is: {mode}")

# Generate a placeholder sonification (a short sequence of glockenspiel notes) that we may want to add a caption to:

# platform agnostic absolute path for samples...
strauss_dir = Path(strauss.__file__).parents[2]
sample_path = Path(strauss_dir, 'data','samples','glockenspiels')
# setup used in stars appearing example
chords = [['Db3','Gb3', 'Ab3', 'Eb4','F4']]
length = 6
system = 'stereo'
score =  Score(chords, length)

maplims =  {'time': ('0%', '150%'),
            'pitch' : ('0%', '100%'),
           'phi':('0%','100%'),
            'theta':('0%','100%')}

events = Events(maplims.keys())

data = {'pitch':np.arange(5),
        'time':np.arange(5),
       'phi': np.arange(5),
       'theta': np.arange(5)}


generator = Sampler(Path(sample_path))

events.fromdict(data)
events.apply_mapping_functions(map_lims=maplims)


# Now, lets look at the avaialble voices for our TTS engine:
from strauss.tts_caption import getVoices
voices = getVoices(True)


# Generate text-to-speech (TTS) for the caption, using the default choice of voice (`"Jenny"` for the `coqui-tts` module, "bf-emma" for kokoro, OS default for `pyttsx3`)
caption_en = 'In the following audio, a glockenspiel is used to represent stars of varying colour.'

soni = Sonification(score, events, generator, system,
                    caption=caption_en)
soni.render()
soni.hear()

# We could also try an alternative model, if one's available
caption_en = 'In the following audio, a glockenspiel is used to represent stars of varying colour.'

set_engine('coqui-tts')
#set_engine('kokoro')
#set_engine('pyttsx3.init')
mode = get_ttsMode()

if mode == 'coqui-tts':
    soni = Sonification(score, events, generator, system,
                        caption=caption_en,
                       ttsmodel=Path('tts_models', 'eng', 'fairseq', 'vits'))
elif mode == 'kokoro':
    soni = Sonification(score, events, generator, system,
                        caption=caption_en,
                       ttsmodel=('am_michael'))
elif mode == 'pyttsx3':
    for v in voices[::-1]:
        #print(v.languages[0][:2])
        if v.languages[0][:2] == 'en':
            break
    print(f"Selected voice: {v.name}")
    soni = Sonification(score, events, generator, system,
                        caption=caption_en,
                       ttsmodel={'voice':v.id,
                                 # we can also set a rate for pyttsx3 (int16)...
                                'rate': 217})
soni.render()
soni.hear()


# Other TTS models are available in several languages. We can demonstrate a German voice, for example
caption_de = "In der folgenden Tonspur wird ein Glockenspiel verwendet um Sterne mit unterschiedlichen Farben zu repräsentieren."

set_engine('coqui-tts')
#set_engine('kokoro')
#set_engine('pyttsx3.init')
mode = get_ttsMode()

if mode == 'coqui-tts':
    language_index = 0 # or, pick a different index for another langauge
    iso_codes = ['deu', 'spa', 'ita', 'pol', 'hin']
    captions = [caption_de,
                "En el siguiente audio, se utiliza una campana para representar estrellas de diferentes colores.",
                "Nell'audio seguente, il suono di un campanello verra utilizzato per rappresentare stelle di diversi colori.",
                "W następującym nagraniu dźwiękowym dzwonek reprezentuje gwiazdy w różnych kolorach.",
                "आगे आने वाले ऑडियो में विभिन्न रंगों के तारों को दर्शाने के लिए अलग-अलग स्वरों का उपयोग किया गया है।"]
    models = [Path('tts_models', 'de', 'thorsten', 'vits'),
              Path('tts_models', iso_codes[1], 'fairseq', 'vits'),
              Path('tts_models', iso_codes[3], 'fairseq', 'vits'),
              Path('tts_models', iso_codes[4], 'fairseq', 'vits')]

    soni = Sonification(score, events, generator, system,
                        caption=captions[language_index],
                        ttsmodel=models[language_index])
elif mode == 'kokoro':
    print('Non-English voices are currently not available in Strauss')
    soni = Sonification(score, events, generator, system,
                       caption=caption_en,
                       ttsmodel=('af_nicole'))

elif mode == 'pyttsx3':
    # find a German-language voice...
    has_voice = 0
    for v in voices:
        if v.languages[0][:2] == 'de':
            has_voice = 1
            break
    if not has_voice:
        print('no language-compatible voice, using first available...')
        v = voices[0]
    soni = Sonification(score, events, generator, system,
                        caption=caption_de,
                        ttsmodel={'voice':v.id})

soni.render()
soni.hear()


# **Note**: the `TTS` can behave unpredictably when using unrecognised characters or terms. Sometimes these will be mispronounced by the TTS, other times they could be skipped entirely. This can be circumvented by writing out the how symbols should be pronounced, or spelling phonetically to improve pronunciation:

symbol_examples_en = 'The Lyman-α resonance is 1216 Å. The Lyman alpha resonance is twelve hundred and sixteen angstroms. '

set_engine('coqui-tts')
#set_engine('kokoro')
#set_engine('pyttsx3.init')
mode = get_ttsMode()

if mode == 'coqui-tts':
    soni = Sonification(score, events, generator, system,
                        caption=symbol_examples_en, 
                        ttsmodel=Path('tts_models', 'eng', 'fairseq', 'vits'))
    
elif mode == 'kokoro':
    soni = Sonification(score, events, generator, system,
                        caption=symbol_examples_en,
                        ttsmodel=('bf_emma'))

elif mode == 'pyttsx3':
    for v in voices[::-1]:
        #print(v.languages[0][:2])
        if v.languages[0][:2] == 'en':
            break
                       
    soni = Sonification(score, events, generator, system,
                        caption=symbol_examples_en,
                        ttsmodel={'voice':v.id})

soni.render()
soni.hear()


# Read in a quasar spectrum and use the Synthesizer to sonify it.

quasar = np.genfromtxt(Path('..', 'data', 'datasets', 'quasar_spectrum.csv'))
wavelength = quasar[0]
flux = quasar[1]
x = wavelength[np.argsort(wavelength)]
y = flux[np.argsort(wavelength)]

# Plot spectrum
fig, ax1 = plt.subplots()
ax1.plot(x,y)
ax1.set_xlabel('Wavelength (Å)')
ax1.set_ylabel('Flux')
ax2 = ax1.twinx()
ax2.set_ylim(0.0, 1.0)
ax2.set_ylabel('Pitch')
plt.title('SDSS Spectrum 3589-55186-0936')
plt.show()

# specify audio system (e.g. mono, stereo, 5.1, ...)
system = "stereo"

# length of the sonification in s
length = 20.

#set up synthesizer generator
generator = Synthesizer()
generator.load_preset('pitch_mapper')
generator.preset_details('pitch_mapper')

notes = [["A2"]]
score =  Score(notes, length)

lims = {'time_evo': ('0%','100%'),
        'pitch_shift': ('0%','100%')}

plims = {'pitch_shift': (0.1,6.)}

data = {'pitch':1.,
        'time_evo':x,
        'azimuth':(x*0.5+0.25) % 1,
        'polar':0.5,
        'pitch_shift':y}

# set up source
sources = Objects(data.keys())
sources.fromdict(data)
sources.apply_mapping_functions(map_lims=lims)

soni = Sonification(score, sources, generator, system)
soni.render()
soni.hear()

def animate(wavelength, flux, soni):
    '''Make frames for animating a plot showing changes in water fraction.
       Create a sequence to animate this with the sound overlaid.'''

    print("\n Creating animation frames. This may take a few minutes.")
    # Make frames for animation
    import warnings
    from pathlib import Path
    import shutil
    import tempfile

    from strauss.animation import Animate
    here = Path.cwd()
    # Define the final target directory
    target_dir_name = Path("figure_animations") / "AudioCaption"
    
    # Use a temporary directory for all intermediate files
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        print(f"Using temporary directory: {temp_dir}")

        pipe = Animate(temp_dir)

        pipe.register('pitch', sonification=soni, pre_caption=f'This is the spectrum of an S D S S Lyman Alpha quasar with broad absorption lines and emission peaks', post_caption='Thank you for listening!', stype='animation')
        xp = wavelength
        yp = flux
        nframe = int(soni.score.length*int(pipe.pars['fps']))
        xf = np.linspace(xp[0], xp[-1], nframe)
        yf = np.interp(xf, xp, yp)
        xp, yp = xf, yf
        for i in range(xp.size)[::1]:
            fig, ax1 = plt.subplots()
            plt.title("SDSS Spectrum 3589-55186-0276")
            ax1.set_xlabel("Wavelength (Å)") 
            ax2 = ax1.twinx() 
            ax1.plot(xp, yp)
            ax1.set_ylabel("Flux")
            ax1.tick_params(axis ='y')
            ax1.axvline(xp[i], ls ='--', c='C0',lw=1.5, alpha=0.55)
            ax1.axhline(yp[i], ls ='--', c='C0',lw=1.5, alpha=0.55)
            ax2.set_ylim(0.0, 1.0)
            ax2.set_ylabel('Pitch')
            ax2.tick_params(axis ='y')
            plt.savefig(pipe.frames["pitch"].parent / f"frame_{i:05d}.png", dpi=120)
            plt.close()
        print(f"Frames created in temporary directory!")
        pipe.render()
        temp_final_mp4 = temp_dir / "final.mp4" 
        target_dir_name.mkdir(parents=True, exist_ok=True)
        final_target_path = target_dir_name / temp_final_mp4.name
        
        if temp_final_mp4.exists():
            shutil.copy(temp_final_mp4, final_target_path)
            print(f"\nFinal animation copied to: {final_target_path}")
        else:
            warnings.warn(f"Could not find {temp_final_mp4} after rendering.")


if args.animate:
    animate(wavelength, flux, soni)
