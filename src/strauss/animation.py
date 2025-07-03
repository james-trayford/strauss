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
        if topdir.exists() and any(topdir.iterdir()):
            warnings.warn(f"{topdir} is not empty, instead name "
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

        self.sequences[name] = Sequence(name, duration=duration,
                                        topdir=self.topdir, index=self.seqdx,
                                        sonification=sonification,
                                        pre_caption=pre_caption,
                                        post_caption=post_caption,
                                        pars=inpars, stype=stype, infile=infile)
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
            prepath = str(Path(seq.path)/'pre.wav')
            if seq.pre:
                audio = force_stereo(prepath)
                audio *= MAXSAMP / abs(audio).max()
                audio = house_audio(audio, self.pars['spf'], self.pars['breathing_time'])
                master.append(audio)
                # append pre vid sequence
                subfs.append(prepath[:-3]+'mp4')
                
            # compile sonification audio
            if seq.sonification:
                audio = force_stereo(str(seq.audiofile))
                
                # ramp audio
                audio[:self.aramplen] *= self.arampin
                audio[-self.aramplen:] *= self.arampout

                audio *= MAXSAMP / abs(audio).max()
                audio = house_audio(audio, self.pars['spf'])

            elif seq.stype == 'clip':
                print('in')
                audio = force_stereo(str(seq.audiofile), do_resample=1)
                audio *= MAXSAMP / abs(audio).max()
                audio = house_audio(audio, self.pars['spf'])
                
            else:
                # stereo silence
                audio = np.zeros((int(seq.duration*SAMPRATE), 2))

            # append vid sequence
            subfs.append(seqvid)

            # append audio
            master.append(audio)
                
            postpath = str(Path(seq.path)/'post.wav')
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

        self.master_wav = str(Path(self.topdir)/'master.wav')
        self.master_mp3 = str(Path(self.topdir)/'master.mp3')
        self.combined = str(Path(self.topdir)/'combo.mp4')
        self.concat_file = str(Path(self.topdir)/'concat_files.txt')
        self.final = str(Path(self.topdir)/'final.mp4')
        
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
            tempfile = str(Path(self.topdir)/'overlay.mp4')
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
        self.path = Path(topdir) / name
        self.frame = Path(self.path)  / f"frame_{index:05d}.png"
        self.nframes = int(np.ceil(int(self.pars['fps']) * self.duration))
        self.sonification = sonification

        if sonification and (duration is not sonification.score.length):
            Exception(f"Provided sonification length ({sonification.score.length}s) != sequence duration ({duration}s)")
        
        self.audiofile = Path(self.path) / f"{name}.wav"
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
            fpath = Path(self.path) / 'pre.wav'
            print(f'\t\t pre-caption: "{self.pre}" to {fpath}')
            # with suppress_output():
            with contextlib.redirect_stdout(None):
                generate_caption(self.pre, fpath, notebook)
            self._torender_flags['pre'] = False
        if self.post:
            fpath = Path(self.path) / 'post.wav'
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
            outfile = str(Path(self.path)/f"{self.name}.mp4")

            # store the number of frames
            # nframes = len(glob.glob(f'{self.path}/frames*'))
            # self.lastdx = nframes-1
            # self.duration = nframes / int(self.pars["fps"])
            
            # make video from frames...
            print(f"\t Render video for sequence {self.name} {self.stype}...")

            if self.stype == 'animation':
                print(str(Path(self.path) / f'{self.name}.png'))
                # TODO: decide how failure-permitted subprocesses should be run?
                #sp.check_call(["ffmpeg", '-y',
                #                 '-r', self.pars["fps"],
                #                 '-i', str(Path(self.path) / f'{self.name}.png'),
                #                 '-c:v', 'mpeg4',
                #                 '-crf', self.pars["crf"]] +
                #                inv.split() + [outfile],
                #                stdout=sp.DEVNULL, stderr=sp.STDOUT)
                sp.run(['ffmpeg', '-y',
                        '-r', self.pars["fps"],
                        '-i', str(Path(self.path) / f'frame_%05d.png'),
                        '-c:v', 'mpeg4',
                        '-crf', self.pars["crf"]] +
                        inv.split() + [outfile],
                        stdout=sp.DEVNULL, stderr=sp.STDOUT,
                        check=False)
                
            elif self.stype == 'slide':
                # make slide sequence
                generate_slide_video(self, self.infile, self.name, time=self.duration)

            elif self.stype == 'clip':
                # make slide sequence
                prepare_clip(self, self.infile, self.name)

            else:
                sp.check_call(['ffmpeg', '-y',
                               '-f', 'lavfi',
                               '-i', f'color=c=black:s={self.pars["dimensions"]}',
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
        video = str(Path(self.path)/f"{self.name}.mp4")
        pos = 0
        ctype = ["pre", "post"]
        for c in [self.pre, self.post]:
            if c:
                print(f"\t\t Making {ctype[pos]}-caption still for {self.name}...")
                clen = wav.read(str(Path(self.path) / f'{ctype[pos]}.wav')).data.shape[0]
                nframes = clen / self.pars['spf']
                nframes = -int(-nframes // 1) + int(self.pars['breathing_time'])
                if (self.stype == 'animation') and (glob.glob(str(self.frame).format(index=0))):
                    fnum = pos*(int(self.pars['fps'])*self.duration - 1)
                    frame = str(self.frame).format(index=int(fnum))
                    if self.pars["invert_colours"]:
                        inv = "-vf negate"

                    # make still sequence
                    sp.check_call(['ffmpeg', '-y',
                                   '-loop', '1',
                                   '-i', frame,
                                   '-r', self.pars["fps"],
                                   '-frames', str(nframes),
                                   '-c:v', 'mpeg4',
                                   '-crf', self.pars["crf"]] +
                                   inv.split() +
                                   [str(Path(self.path)/f"{ctype[pos]}.mp4")],
                                   stdout=sp.DEVNULL, stderr=sp.STDOUT)

                elif self.stype == 'slide':
                    generate_slide_video(self, self.infile, ctype[pos], nframes=nframes)

                elif self.stype == 'clip':
                    if pos:
                        frame = str(Path(self.path)/f'pre.png')
                        sp.check_call(['ffmpeg', '-y',
                                       '-sseof', '-0.1',
                                       '-i', Path(self.path)/f'{self.name}.mp4',
                                       '-update', '1',
                                       frame],
                                       stdout=sp.DEVNULL, stderr=sp.STDOUT)
                    else:
                        frame = str(Path(self.path)/f'post.png')
                        sp.check_call(['ffmpeg', '-y',
                                       '-i', str(Path(self.path)/f'{self.name}.mp4'),
                                       '-vf', "select=eq(n\\,0)",
                                       frame],
                                       stdout=sp.DEVNULL, stderr=sp.STDOUT)
                   # make still sequence
                    sp.check_call(['ffmpeg', '-y',
                                   '-loop', '1',
                                   '-i', frame,
                                   '-r', self.pars["fps"],
                                   '-frames', str(nframes),
                                   '-c:v', 'mpeg4',
                                   '-crf', self.pars["crf"],
                                   str(Path(self.path)/f"{ctype[pos]}.mp4")],
                                   stdout=sp.DEVNULL, stderr=sp.STDOUT)
                    
                else:
                    # blank video
                    sp.check_call(['ffmpeg', '-y',
                                   '-f', 'lavfi',
                                   '-i', f'color=c=black:s={self.pars["dimensions"]}',
                                   '-frames', str(nframes),
                                   '-r', self.pars["fps"],
                                   '-c:v', 'mpeg4',
                                   '-crf', self.pars["crf"],
                                   str(Path(self.path)/f"{ctype[pos]}.mp4")],
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

