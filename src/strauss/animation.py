""" The :obj:`animation` submodule: creates an audio-visual sequence including options for animating plots, mapping variables to selected sound properties. It also offers options to include audio captions, static slides and background videos.

"""

import glob
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
from IPython.display import Audio
from TTS.api import TTS
from pathlib import Path
import uuid
import subprocess as sp
import contextlib
import io
import sys
from scipy.io import wavfile
from strauss.utilities import resample
import wavio as wav

# ----------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------

# fix audio sample rate
SAMPRATE = 48000

# maximum absolute sample value for audio peak normalisation
MAXSAMP = (2**31)-1

# supported sequence types
seq_types = ['animation',
             'image',
             'text',
             'blank',
             'clip']

defaults = {'fps': '30', 		# frames per second
            'crf': '18',		# ffmpeg quality level
            'invert_colours': True,	# dark on light is true
            'transition_time': '30',	# sequence transition time [frames]
            'breathing_time': '6',	# minimum time between separate sounds [frames]
            #'dimensions': '4200x2100',
            'background_video': './example_media/starfield.mov',
            #'background_video': '/Users/jamestrayford/Downloads/stockvideo_01171.mov',
            'dimensions': '3840x2160',   # video dimensions (4k by standard)
            'orientation': 'vertical',
            'transition_type': 'fade',
            'slide_min_margin': '300',
            'slide_key_black': '0',
            'clip_override_duration': '1'
            }

# ----------------------------------------------------------------------
# Useful dictionaries
# ----------------------------------------------------------------------

orient = {'vertical': '3840x2160',
          'horizontal': '2160x3840'}

res = {'4k': (3840, 2160),
       '1080p': (1920, 1080),
       '720p' : (1280, 720)}

# ----------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------

class Animate:
    
    def __init__(self, topdir, pars={}):
        isdir = glob.glob(topdir)
        Path(topdir).mkdir(parents=True, exist_ok=True)
        if isdir:
            if glob.glob(f'{isdir[0]}/*'):
                warnings.warn(f"{topdir} is not empty, instead name "\
                              "an empty directory, or a new one.")
        self.topdir = topdir

        # handle parameters
        self.pars = defaults.copy()
        for k in pars.keys():
            self.pars[k] = pars[k]
        self.pars['spf'] = SAMPRATE/int(self.pars['fps'])
        self.spf = SAMPRATE/int(self.pars['fps'])

        if self.spf % 1:
            Exception("non integer samples-per-frame value, please use a standard video"\
                      "fps (30,25,20) and audio sample rate (48000, 44100)")
        
        self.sequences = {}
        self.frames = {}
        self.seqlist = []
        self.seqdx = 0

        # stereo audio ramps to prevent dropouts, 30ms hard coded
        self.aramplen = int(0.03*SAMPRATE)
        self.arampin = np.column_stack([np.linspace(0,1, self.aramplen)]*2)
        self.arampout = self.arampin[::-1]

        # audio padding for breathing time and transitions
        self.apadbreath = np.column_stack([np.zeros(int(self.pars['spf'] * float(self.pars['breathing_time'])))]*2)
        self.apadtrans = np.column_stack([np.zeros(int(self.pars['spf'] * float(self.pars['transition_time'])))]*2)
        self.halfbsamps = self.apadtrans.size // 2
        
    def register(self, name, duration, sonification=None, pre_caption='', post_caption='', stype='animation', infile=None, pars={}):
        if ((duration * (int(self.pars['fps']) + SAMPRATE)) % 1) and (stype != 'clip'):
            Exception(f"Duration {duration}s gives a non-integer number of frames and/or audio samples," \
                      f"please retry, for example with an integer number of seconds e.g. ({int(np.ceil(duration))}s)")

        inpars = self.pars.copy()
        for k in pars.keys():
            inpars[k] = pars[k]

        self.sequences[name] = Sequence(name, duration=duration, topdir=self.topdir, index=self.seqdx, sonification=sonification,
                                        pre_caption=pre_caption, post_caption=post_caption, pars=inpars, stype=stype,
                                        infile=infile)        
        self.seqlist.append(name)
        self.frames[name] = self.sequences[name].frame
        self.seqdx += 1

    def render(self):

        master = []
        flist = []
        fromfile = None

        print("First, process sequences.\n")
        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")    
        
        # first, render sequences
        for i in range(len(self.seqlist)):

            name = self.seqlist[i]
            seq = self.sequences[name]
            seqvid = f'{seq.path}/{seq.name}.mp4'

            print(f"Sequence {i}: \t {name}")
            subfs = []
            
            # render steps, per sequence
            seq.render_frames()
            seq.render_caption()
            seq.render_sonification()
            seq.render_caption_stills()
                            
            # compile sonification audio
            prepath = seq.path+'/pre.wav'
            if seq.pre:
                audio = force_stereo(prepath)
                audio *= MAXSAMP / abs(audio).max()
                audio = house_audio(audio, self.pars['spf'], self.pars['breathing_time'])
                master.append(audio)
                # append pre vid sequence
                subfs.append(prepath[:-3]+'mp4')
                
            # compile sonification audio
            if seq.sonification:
                audio = force_stereo(seq.audiofile)
                
                # ramp audio
                audio[:self.aramplen] *= self.arampin
                audio[-self.aramplen:] *= self.arampout

                audio *= MAXSAMP / abs(audio).max()
                audio = house_audio(audio, self.pars['spf'])

            elif seq.stype == 'clip':
                print('in')
                audio = force_stereo(seq.audiofile, do_resample=1)
                audio *= MAXSAMP / abs(audio).max()
                audio = house_audio(audio, self.pars['spf'])
                
            else:
                # stereo silence
                audio = np.zeros((int(seq.duration*SAMPRATE), 2))

            # append vid sequence
            subfs.append(seqvid)

            # append audio
            master.append(audio)
                
            postpath = seq.path+'/post.wav'
            if seq.post:
                audio = force_stereo(postpath)
                audio *= MAXSAMP / abs(audio).max()
                audio = house_audio(audio, self.pars['spf'], self.pars['breathing_time'])
                master.append(audio)
                # append post vid sequence
                subfs.append(postpath[:-3]+'mp4')
            
            # pad for transition out
            master.append(self.apadtrans.copy())

            if fromfile:
                render_transition(fromfile, seqvid, seq)
                subfs = [seq.path+'/transin.mp4'] + subfs

            flist += subfs
            fromfile = flist[-1]


        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")
        print("Now, compile material.")
        # remove final transition
        # for a in master:
        #     print('duration: ', a.shape[0]/int(SAMPRATE))
        outsamps = np.vstack(master[:-1]).astype('int32')

        self.master_wav = self.topdir+'/master.wav'
        self.master_mp3 = self.topdir+'/master.mp3'
        self.combined = self.topdir+'/combo.mp4'
        self.concat_file = self.topdir+'/concat_files.txt'
        self.final = self.topdir+'/final.mp4'
        
        wavfile.write(self.master_wav, SAMPRATE, outsamps)

        with open(self.concat_file, "w") as cfiles:
            for f in flist:
                cfiles.write(f"file '{f}'\n")
            
        # concatenate files!
        print(f"Concatenate files...")
        sp.check_call(["ffmpeg", '-y',
                         "-f", 'concat',
                         '-safe', '0',
                         '-i', self.concat_file,
                         '-c', 'copy',
                         self.combined],
                        stdout=sp.DEVNULL, stderr=sp.STDOUT)
        
        # make and convert master audio track (run python script)
        print(f"Convert audio...")
        sp.check_call(["ffmpeg", '-y',
                         "-i", self.master_wav,
                         '-vn', '-ar', str(SAMPRATE),
                         '-ac', '2', '-b:a', '192k',
                         self.master_mp3],
                        stdout=sp.DEVNULL, stderr=sp.STDOUT)

        bgfile = glob.glob(self.pars['background_video'])

        if bgfile:
            print(f"Combine sequences and chroma-key background video...")
            tempfile = self.topdir+'/overlay.mp4'
            sp.check_call(["ffmpeg", '-y',
                             '-stream_loop', '-1',
                             '-i', bgfile[0],
                             '-i', self.combined,
                             '-filter_complex',
                             '[1:v]colorkey=0x000000:0.01:0.01[ckout];[0:v][ckout]overlay=(W-w)/2:(H-h)/2:shortest=1[out]',
                             '-map', '[out]', tempfile],
                            stdout=sp.DEVNULL, stderr=sp.STDOUT)
            self.combined = tempfile

        else:
            print(f"Background video file '{self.pars['background_video']}' not found, skipping...")
            
        print(f"Dubbing final video...")
        sp.check_call(["ffmpeg", '-y',
                         '-i', self.combined,
                         '-itsoffset', '0.047', # TODO: investigate why extra 47ms padding is needed to sync?
                         '-i', self.master_mp3,
                         '-map', '0:v', '-map', '1:a',
                         '-c:v', 'copy', '-shortest',
                         self.final],
                        stdout=sp.DEVNULL, stderr=sp.STDOUT)

        print("Done!")
    
class Sequence:
    def __init__(self, name, duration, sonification=None, topdir='', index=None, stype='animation', custom_path=None,
                 invert_colours=True, pars=defaults, pre_caption='', post_caption='', infile=None):
        self.name = name
        if stype == "clip":
            dur = sp.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of",
                          "default=noprint_wrappers=1:nokey=1", infile],
                         stdout=sp.PIPE, stderr=sp.STDOUT)
            self.duration = float(dur.stdout)
        else:
            self.duration = duration
        self.pars = pars
        self.infile = infile
        
        self.index = index
        self.subs = {}
        self.stype = stype
        self.path = topdir+f'/{name}'
        self.frame = self.path + "/frame_{index:05d}.png"
        self.nframes = int(np.ceil(int(self.pars['fps']) * self.duration))
        self.sonification = sonification

        if sonification and (duration is not sonification.score.length):
            Exception(f"Provided sonification length ({sonification.score.length}s) != sequence duration ({duration}s)")
        
        self.audiofile = self.path + f"/{name}.wav"
        self.pars = pars
        self._torender_flags = {'pre': True, 'post': True,
                                'frames': True, 'sonification': bool(sonification)}
        
        self.lastdx = None
        self.length = None

        if pre_caption:
            self.pre = prep_caption(pre_caption)
        else:
            self.pre = ''

        if post_caption:
            self.post = prep_caption(post_caption)
        else:
            self.post = ''

        print(f"making {self.path}")
        Path(self.path).mkdir(parents=True, exist_ok=True)
        
    def caption(self, pre='', post=''):
        if self.pre != pre:
            self.pre = prep_caption(pre)
            self._torender_flags['pre'] = True
        if self.post != post:
            self.post = prep_caption(post)
            self._torender_flags['post'] = True
            
    def render_caption(self, notebook=True):
        print(f"\t Rendering {self.name} captions:")
        if self.pre:
            fpath = self.path + f'/pre.wav'
            print(f'\t\t pre-caption: "{self.pre}" to {fpath}')
            # with suppress_output():
            with contextlib.redirect_stdout(None):
                generate_caption(self.pre, fpath, notebook)
            self._torender_flags['pre'] = False
        if self.post:
            fpath = self.path + f'/post.wav'
            print(f'\t\t post-caption: "{self.post}" to {fpath}')
            # with suppress_output():
            with contextlib.redirect_stdout(None):
                generate_caption(self.post, fpath, notebook)
            self._torender_flags['post'] = False            

    def render_sonification(self):
        if self._torender_flags['sonification'] and self.sonification:
            # check it's been rendered 
            if not self.sonification.out_channels['0'].values.any():
                print(f"\t Rendering Sonification for {self.name} sequence...")
                with contextlib.redirect_stdout(None):
                    self.sonification.render()
            with contextlib.redirect_stdout(None):
                self.sonification.save(self.audiofile)
            
    def render_frames(self):
        if self._torender_flags['frames']:
            inv = ""
            if self.pars["invert_colours"]:
                inv = "-vf negate"
            outfile = f"{self.path}/{self.name}.mp4"

            # store the number of frames
            # nframes = len(glob.glob(f'{self.path}/frames*'))
            # self.lastdx = nframes-1
            # self.duration = nframes / int(self.pars["fps"])
            
            # make video from frames...
            print(f"\t Render video for sequence {self.name}...")

            if self.stype == 'animation':
                # TODO: decide how failure-permitted subprocesses should be run?
                sp.check_call(["ffmpeg", '-y',
                                 '-r', self.pars["fps"],
                                 '-i', f'{self.path}/frame_%05d.png',
                                 '-c:v', 'mpeg4',
                                 '-crf', self.pars["crf"]] +
                                inv.split() + [outfile],
                                stdout=sp.DEVNULL, stderr=sp.STDOUT)
                
            elif self.stype == 'slide':
                # make slide sequence
                generate_slide_video(self, self.infile, self.name, time=self.duration)

            elif self.stype == 'clip':
                # make slide sequence
                prepare_clip(self, self.infile, self.name)

            else:
                sp.check_call(["ffmpeg", '-y',
                                 '-f', 'lavfi',
                                 "-i", f"color=c=black:s={self.pars['dimensions']}",
                                 '-frames', str(self.duration * int(self.pars['fps'])),
                                 '-r', self.pars["fps"],
                                 '-c:v', 'mpeg4',
                                 '-crf', self.pars["crf"],
                                 outfile],
                                stdout=sp.DEVNULL, stderr=sp.STDOUT)
                
                
            # frames rendered for now...
            self._torender_flags['frames'] = False

    def render_caption_stills(self):
        print(f"\t Rendering {self.name} caption stills...")
        # iterate through existing captions
        video = f"{self.path}/{self.name}.mp4"
        pos = 0
        ctype = ["pre", "post"]
        for c in [self.pre, self.post]:
            if c:
                print(f"\t\t Making {ctype[pos]}-caption still for {self.name}...")
                clen = wav.read(self.path+f'/{ctype[pos]}.wav').data.shape[0]
                nframes = clen / self.pars['spf']
                nframes = -int(-nframes // 1) + int(self.pars['breathing_time'])
                if (self.stype == 'animation') and (glob.glob(self.frame.format(index=0))):
                    fnum = pos*(int(self.pars['fps'])*self.duration - 1)
                    frame = self.frame.format(index=int(fnum))
                    if self.pars["invert_colours"]:
                        inv = "-vf negate"

                    # make still sequence
                    sp.check_call(["ffmpeg", '-y',
                                   "-loop", "1",
                                   '-i', frame,
                                   '-r', self.pars["fps"],
                                   '-frames', str(nframes),
                                   '-c:v', 'mpeg4',
                                   '-crf', self.pars["crf"]] +
                                   inv.split() +
                                   [f"{self.path}/{ctype[pos]}.mp4"],
                                  stdout=sp.DEVNULL, stderr=sp.STDOUT)

                elif self.stype == 'slide':
                    generate_slide_video(self, self.infile, ctype[pos], nframes=nframes)

                elif self.stype == 'clip':
                    if pos:
                        frame = f'{self.path}/pre.png'
                        sp.check_call(["ffmpeg", '-y',
                                       "-sseof", '-0.1',
                                       '-i', f'{self.path}/{self.name}.mp4',
                                       '-update', '1',
                                       frame],
                                      stdout=sp.DEVNULL, stderr=sp.STDOUT)
                    else:
                        frame = f'{self.path}/post.png'
                        sp.check_call(["ffmpeg", '-y',
                                       '-i', f'{self.path}/{self.name}.mp4',
                                       '-vf', "select=eq(n\\,0)",
                                       frame],
                                      stdout=sp.DEVNULL, stderr=sp.STDOUT)
                   # make still sequence
                    sp.check_call(["ffmpeg", '-y',
                                   "-loop", "1",
                                   '-i', frame,
                                   '-r', self.pars["fps"],
                                   '-frames', str(nframes),
                                   '-c:v', 'mpeg4',
                                   '-crf', self.pars["crf"],
                                   f"{self.path}/{ctype[pos]}.mp4"],
                                  stdout=sp.DEVNULL, stderr=sp.STDOUT)
                    
                else:
                    # blank video
                    sp.check_call(["ffmpeg", '-y',
                                     '-f', 'lavfi',
                                     "-i", f"color=c=black:s={self.pars['dimensions']}",
                                     '-frames', str(nframes),
                                     '-r', self.pars["fps"],
                                     '-c:v', 'mpeg4',
                                     '-crf', self.pars["crf"],
                                    f"{self.path}/{ctype[pos]}.mp4"],
                                    stdout=sp.DEVNULL, stderr=sp.STDOUT)
                pos += 1
                
        #ffmpeg -i input.mp4 -vf "scale=iw*sar:ih,setsar=1" -vframes 1 filename.png
            
#================================================================================
#===== helper functions =========================================================
#================================================================================

@contextlib.contextmanager
def suppress_output():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout


def generate_caption(caption, path, notebook=True):
    tts = TTS(model_name='tts_models/en/jenny/jenny', progress_bar=False, gpu=False)
    tts.tts_to_file(text=caption, file_path=path)

def prep_caption(caption):
    if caption:
        sents = caption.split('.')
        return '.'.join(sents[:-1] + [sents[-1]+'.']) 
    return None

def force_stereo(audio_file, do_resample=False):
    sound = wav.read(audio_file)

    data = sound.data
    
    if do_resample:
        print(int(sound.rate), SAMPRATE)
        data = resample(int(sound.rate), SAMPRATE, data)
        
    if data.shape[1] == 1:
        audio = np.column_stack([data, data])
    else:
        audio = data[:,:2]
    return audio.astype(float)

def house_audio(audio, spf, fpad=0):
    fpad = int(fpad)
    spf = int(spf)
    halfpad = (spf*fpad // 2)
    zarr = np.zeros(((-int(-(audio.shape[0]/spf) // 1) + fpad) *
                     spf,2))
    # print(audio.shape, spf, fpad, halfpad, zarr.shape)
    zarr[halfpad:audio.shape[0]+halfpad] = audio
    return zarr


def render_transition(fromfile, tofile, toseq):
    # get transition frames
    # ffmpeg -i inputfile.mkv -vf "select=eq(n\,0)" -q:v 3 output_image.jpg
    inframe = '/'.join(fromfile.split('/')[:-1] + ['from.png'])
    outframe = '/'.join(tofile.split('/')[:-1] + ['to.png'])
    sp.check_call(["ffmpeg", '-y',
                     "-sseof", '-0.2',
                     '-i', fromfile,
                     '-update', '1',
                     inframe],
                    stdout=sp.DEVNULL, stderr=sp.STDOUT)

    sp.check_call(["ffmpeg", '-y',
                     '-i', tofile,
                     '-vf', "select=eq(n\\,0)",
                     outframe],
                    stdout=sp.DEVNULL, stderr=sp.STDOUT)
    
    # make transition video
    tdur = int(toseq.pars['transition_time'])/int(toseq.pars['fps'])
    toff = 0
    transfile = '/'.join(tofile.split('/')[:-1] + ["transin.mp4"])
    tvidpars = ['-r', toseq.pars['fps'],
                '-loop', '1',
                '-t', str(tdur)]

    print(f"\t Transition into sequence {toseq.name}...")
    sp.check_call(["ffmpeg", '-y',
                     *tvidpars,
                     '-i', inframe,
                     *tvidpars,
                     '-i', outframe,
                     '-filter_complex',
                     f"[0][1]xfade=transition={toseq.pars['transition_type']}:duration={tdur}:offset={toff}",#format=yuv420p",
                     '-bsf:v', 'h264_metadata=sample_aspect_ratio=1/1',
                     '-c:v', 'mpeg4',
                     '-crf', toseq.pars["crf"],
                     transfile],
                    stdout=sp.DEVNULL, stderr=sp.STDOUT)

def prepare_clip(seq, infile, outtype):
    dims = seq.pars['dimensions'].split('x')
    margin = int(seq.pars['slide_min_margin'])
    filts = []
    filts.append(f"scale=w={int(dims[0])-margin}:h={int(dims[1])-margin}:force_original_aspect_ratio=1")
    if not int(seq.pars['slide_key_black']):
        # this subtle brightening ensures all pixels are outside keyed range (above absolute black)
        filts.append(f"eq=brightness=0.04")
    filts.append(f"pad={dims[0]}:{dims[1]}:(ow-iw)/2:(oh-ih)/2")

    # extract audio
    sp.check_call(["ffmpeg", '-y',
                   '-i', infile,
                   f"{seq.path}/{outtype}.wav"],
                  stdout=sp.DEVNULL, stderr=sp.STDOUT)    

    # reencode video
    cmd = ["ffmpeg", '-y',
                   '-i', infile,
                   "-vf", ",".join(filts),
                   '-r', seq.pars["fps"],
                   '-c:v', 'mpeg4',
                   '-crf', seq.pars["crf"],
                   f"{seq.path}/{outtype}.mp4"]

    # print (' '.join(cmd))
    sp.check_call(cmd, stdout=sp.DEVNULL, stderr=sp.STDOUT)
    
def generate_slide_video(seq, still, outtype, time=None, nframes=None):
    dims = seq.pars['dimensions'].split('x')
    margin = int(seq.pars['slide_min_margin'])
    filts = []
    filts.append(f"scale=w={int(dims[0])-margin}:h={int(dims[1])-margin}:force_original_aspect_ratio=1")
    if not int(seq.pars['slide_key_black']):
        # this subtle brightening ensures all pixels are outside keyed range (above absolute black)
        filts.append(f"eq=brightness=0.04")
    filts.append(f"pad={dims[0]}:{dims[1]}:(ow-iw)/2:(oh-ih)/2")
    
    if time:
        dur = ['-t', str(time)]
    if nframes:
        dur = ['-frames', str(nframes)]

    cmd = ["ffmpeg", '-y',
                   "-loop", "1",
                   '-i', still,
                   "-vf", ",".join(filts),
                   '-r', seq.pars["fps"],
                   dur[0], dur[1],
                   '-c:v', 'mpeg4',
                   '-crf', seq.pars["crf"],
                   f"{seq.path}/{outtype}.mp4"]

    # print (' '.join(cmd))
    sp.check_call(cmd, stdout=sp.DEVNULL, stderr=sp.STDOUT)

def merge_audio(sequences_file="sequences.txt"):
    """ merge all audio files into a single master file

    Args:
      sequences_file (:obj:`str`): A text file containing a list of the 
      pathnames of all audio files to be merged.
    """

    SAMPRATE = 48000
    OUTTIME = 0.03

    fsamps = int(OUTTIME*SAMPRATE)
    rampin = np.linspace(0,1, fsamps)
    rampout = rampin[::-1]

    seq = pd.read_csv(sequences_file, header=None, delim_whitespace=True)

    t_tot = np.array(seq[1]).sum()
    s_tot = int(t_tot*SAMPRATE)

    print(f"length {t_tot} s, {s_tot} samples.")

    master = np.zeros((s_tot, 2)).astype('int32')
    tags = list(seq[0])
    ts = list(seq[1])

    print(tags, ts)
    exit

    sdx = 0
    edx = 0

    for i in range(len(seq[1])):
        edx += int(ts[i]*SAMPRATE)
        f = f"figure_animations/{tags[i]}/{tags[i]}.wav"
        if glob.glob(f):
            print(f)
            wave = wav.read(f)
            if wave.data.shape[1] == 1:
                svals = np.column_stack([wave.data]*2).astype(float)
            else:
                svals = wave.data.astype(float)
            svals *= (2**31) / abs(svals).max()
            svals[:fsamps] *= np.column_stack([rampin]*2)
            svals[-fsamps:] *= np.column_stack([rampout]*2)
            print(svals.max())
            plt.plot(svals[0])
            rms = np.sqrt(np.mean(svals**2))
            rrat = 0.8
            cstart = sdx+((edx-sdx-svals[:,1].size)//2)
            cend = cstart+svals[:,1].size
            master[cstart:cend] = svals*rrat
        sdx = edx
    wavfile.write(f'master.wav', SAMPRATE, master.astype('int32'))

def merge_animations():
    """ Join background movie, all animations and master sound file to make final
    animated output """

    # Helper function to run ffmpeg commands
    def run_ffmpeg(command):
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)

    # Loop BG (starfield.mov) to create a 50 second background
    run_ffmpeg([
        'ffmpeg', '-stream_loop', '5', '-i', 'starfield.mov', '-c',
        'copy', 'starfield_5loops.mov'
        ])

    # Make plot animations
    figures = ['fig1a', 'fig1b', 'fig1c']
    for figure in figures:
        run_ffmpeg([
            'ffmpeg', '-r', '30', '-i', f'figure_animations/{figure}/frames_%03d.png', 
            '-c:v', 'mpeg4', '-crf', '18', '-vf', 'negate', 
            f'figure_animations/{figure}/{figure}.mp4'
        ])

    # Create x-fade transitions
    run_ffmpeg([
        'ffmpeg', '-r', '30', '-loop', '1', '-t', '4', 
        '-i', 'figure_animations/fig1a/frames_089.png', 
        '-loop', '1', '-r', '30', '-t', '4', 
        '-i', 'figure_animations/fig1b/frames_000.png', 
        '-filter_complex',
        '[0][1]xfade=transition=fade:duration=1:offset=3,format=yuv420p[xfade];[xfade]negate[final]', 
        '-map', '[final]', '-c:v', 'mpeg4', '-crf', '18', 
        'figure_animations/fig1a2b/fig1a2b.mp4'
    ])

    run_ffmpeg([
        'ffmpeg', '-r', '30', '-loop', '1', '-t', '3', 
        '-i', 'figure_animations/fig1b/frames_089.png', 
        '-loop', '1', '-r', '30', '-t', '3', 
        '-i', 'figure_animations/fig1c/frames_000.png', 
        '-filter_complex', '[0][1]xfade=transition=fade:duration=1:offset=2,format=yuv420p[xfade];[xfade]negate[final]', 
        '-map', '[final]', '-c:v', 'mpeg4', '-crf', '18', 
        'figure_animations/fig1b2c/fig1b2c.mp4'
    ])

    # Make fade-in/out
    os.makedirs('figure_animations/fig1in', exist_ok=True)
    os.makedirs('figure_animations/fig1out', exist_ok=True)

    run_ffmpeg([
        'ffmpeg', '-loop', '1', '-i', 'figure_animations/fig1a/frames_000.png', 
        '-c:v', 'mpeg4', '-t', '10', '-pix_fmt', 'yuv420p', 
        '-crf', '18', '-r', '30', '-filter_complex', 
        '[0:v]negate[negate];[negate]fade=t=in:st=7:d=3[final]', 
        '-map', '[final]', 'figure_animations/fig1in/fig1in.mp4'
    ])

	run_ffmpeg([
		'ffmpeg', '-loop', '1', '-i', 'figure_animations/fig1c/frames_089.png', 
		'-c:v', 'mpeg4', '-t', '10', '-pix_fmt', 'yuv420p', 
		'-crf', '18', '-r', '30', '-filter_complex', 
		'[0:v]negate[negate];[negate]fade=t=out:st=3:d=3[final]', 
		'-map', '[final]', 'figure_animations/fig1out/fig1out.mp4'
	])

	# Create file list for concatenation
	with open('files.txt', 'w') as f:
		f.write("file 'figure_animations/fig1in/fig1in.mp4'\n")
		f.write("file 'figure_animations/fig1a/fig1a.mp4'\n")
		f.write("file 'figure_animations/fig1a2b/fig1a2b.mp4'\n")
		f.write("file 'figure_animations/fig1b/fig1b.mp4'\n")
		f.write("file 'figure_animations/fig1b2c/fig1b2c.mp4'\n")
		f.write("file 'figure_animations/fig1c/fig1c.mp4'\n")
		f.write("file 'figure_animations/fig1out/fig1out.mp4'\n")

	# Concatenate files into one video
	run_ffmpeg(['ffmpeg', '-f', 'concat', '-safe', '1', '-i', 'files.txt', '-c', 'copy', 'combo.mp4'])

	# Create and convert master audio track
	subprocess.run(['python3', 'merge_audio.py'], check=True)
	run_ffmpeg(['ffmpeg', '-i', 'master.wav', '-vn', '-ar', '48000', '-ac', '2', '-b:a', '192k', 'master.mp3'])

	# Combine video with master audio and key out black for transparency
	run_ffmpeg([
		'ffmpeg', '-i', 'starfield_5loops.mov', '-i', 'combo.mp4', 
		'-filter_complex', '[1:v]colorkey=0x000000:0.01:0.01[ckout];[0:v][ckout]overlay=(W-w)/2:(H-h)/2:shortest=1[out]', 
		'-map', '[out]', 'overlay_test.mp4'
	])

	run_ffmpeg([
		'ffmpeg', '-i', 'overlay_test.mp4', '-i', 'master.mp3', 
		'-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-shortest', 'dubbed.mp4'
	])

	print("Video creation complete!")



#if __name__ == '__main__':
#    Animate("/Users/jamestrayford/Documents/strauss/showreel")
