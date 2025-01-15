"""  The :obj:`stream` submodule: representing the sound signal

Containing the ``Stream`` class to house the ``Sonificiation`` audio
signal for each channel in the ``Channels`` object. This can be
split into uniform segments or `buffers` via the ``Buffers`` object,
for processing.

Todo:
  * implement filter Q-parameter mapping
"""
import numpy as np
import wavio
import matplotlib.pyplot as plt
from scipy.signal.windows import hann

class Stream:
    """
    Stream object representing audio samples.

    Houses audio samples and associates metadata representing the
    actual audio signal produced by the `Generator` class and output
    via the `audio_channels` class.

    Attributes:
      samprate (:obj:`int`): Samples per second of audio stream (Hz)
      length (:obj:`float`): Duration of the stream in seconds
      values (:obj:`ndarray`): Values of individual samples
      samples (:obj:`ndarray`): Indices of each sample
      samptype (:obj:`ndarray`): Time in seconds each sample occurs
      buffers (:obj:`Buffers`): Buffered stream if generated
    """
    def __init__(self, length, samprate=44100, ltype='seconds'):
        """ 
        Args:
          length (numerical): Number representing the length of the
            stream either as an integer number of samples, or a value
            of seconds
          samprate (optional :obj:`int`): Samples per second of audio
            stream (Hz)
          ltype (optional :obj:`str`): quantity represented by
            `length`, either duration in 'seconds' or precise number 
            of 'samples'
        """
        # variables we want to keep constant
        self.samprate = samprate
        self._nyqfrq = 0.5*self.samprate
        
        if ltype == 'seconds':
            self.length = length
            self._nsamp_stream = int(samprate * length)
        elif ltype == 'samples':
            self._nsamp_stream = length
            self.length = length / samprate
            
            
        # sample values initialised to 0 (silence)
        self.values =  np.zeros(self._nsamp_stream)

        # private stream for keeping track of buffered stream
        self._bvalues =  np.zeros(self._nsamp_stream)
        
        # sample numbers for indexing
        self.samples = np.arange(self._nsamp_stream, dtype=int)

        # time at which each sample occurs
        self.samptime = self.samples / self.samprate
        
    def bufferize(self, bufflength=0.1):
        """Wrapper to initialise Buffers subclass

        Args:
          bufflength (optional, :obj:`float`): duration in seconds of
            each buffer to be generated
        """
        self.buffers = Buffers(self, bufflength)

    def consolidate_buffers(self):
        """
        Wrapper to reassign stream values to consolidated buffers

        See :func:`~stream.Buffers.buffers.to_stream`
        """
        self.values = self.buffers.to_stream()
        
    def filt_sweep(self, ffunc, fmap, qmap=lambda x:x*0 + 0.1,
                   flo=20, fhi=2.205e4, qlo=0.5, qhi=10):
        """
        Apply time varying filter to buffered stream

        Args:
          ffunc (function): function that applies filter 
          fmap (function): mapping function representing filter cutoff
            sweep
          qmap (optional, function): mapping function for a filters
            Q parameter
          flo (optional, :obj:`float`): lowest frequency of sweep in Hz,
            default 20
          fhi (optional, :obj:`float`): lowest frequency of sweep in Hz,
            default 22.05 kHz
          qlo (optional, :obj:`float`): lowest 'Q' value of sweep,
            default 0.5
          qhi (optional, :obj:`float`): lowest frequency of sweep,
            default 10
          
        """
        if not hasattr(self, "buffers"):
            Exception("needs bufferized stream, please run 'bufferize' method first.")
        buffers = self.buffers

        # array to regularly sample maps at each buffer
        x = np.linspace(0, 1, buffers._nbuffs_tot)

        lfhi = np.log10(fhi)
        lflo = np.log10(flo)
        
        # obtain buffer values from maps, with cutoff sweep in units
        # of the nyquist frequency
        svals = pow(10., fmap(x)*(lfhi-lflo)+lflo)/self._nyqfrq
        qvals = (qmap(x)*(qhi-qlo)+qlo)

        # loop over buffers, applying appropriate filtering to each 
        for i in range(buffers._nbuffs):
            i2 = 2*i
            buffers.buffs_tile[i] = ffunc(buffers.buffs_tile[i], svals[i2], qvals[i2])
        for i in range(buffers._nbuffs-1):
            i2 = 2*i+1
            buffers.buffs_olap[i] = ffunc(buffers.buffs_olap[i], svals[i2], qvals[i2])

        # finally, consolidate buffers to apply effect to stream
        self.consolidate_buffers()

    def get_sampfracs(self):
        """ Get fractional position of the sample in total stream duration
        """
        self.sampfracs = np.linspace(0, 1, self.values.size)
        
    def save_wav(self, filename):
        """Save audio stream to wav file, specified by filename

        Args:
          filename (:obj:`str`): name of output WAV file 
        """
        wavio.write(filename, self.values, self.samprate, sampwidth=3) 
        
    def reset(self):
        """Zero audio stream and buffers if present."""
        self.values *= 0.
        if hasattr(self, "buffers"):
            self.buffers.buffs_tile *= 0.
            self.buffers.buffs_olap *= 0.        
        
class Buffers:
    """Audio buffers split into uniform discrete chunks or 'buffers'.

    Audio ~:class:`stream.Stream` as a discrete sequence of individual
    'buffers' of fixed duration (number of samples). This allows time
    varying operations in frequency space, such as signal filtering.
    Buffers are tiled in a 'brickwork' fashion so they always overlap
    with another buffer.

    Attributes:
      fade (:obj:`ndarray`): Window function for recombining
        overlapping buffers
      nsamp_padstream (:obj:`int`): Number of samples needed to split
        the stream into discrete buffers of chosen length
      nsamp_pad (:obj:`int`): Number of additional samples needed to
        add to the original `Stream` size in this case
      buffs_tile (:obj:`ndarray`): 2d array of buffers completely
        enclosing the stream (number of buffers x samples per buffer)
      buffs_olap (:obj:`ndarray`) 2d array of overlap buffers,
        allowing for cross fading 
    """
    def __init__(self, stream, bufflength=0.1):
        """
        Args:
          stream (~:class:`stream.Stream`): Stream object to be
            represented using the buffers
          bufflength (optional, :obj:`float`): duration in seconds of
            each buffer to be generated
        """
        nbuff = stream.samprate*bufflength
        if nbuff < 20:
            Exception(f"Error: buffer length {nbuff} samples below "
                      "lower limit of 20, with specified bufflength "
                      "{bufflength} seconds and sample rate {self.samprate} Hz")

        # force buffer length to an even number of samples
        self._nsamp_halfbuff = int(stream.samprate*bufflength) // 2
        self._nsamp_buff = 2 * self._nsamp_halfbuff

        # minimum number of tiled buffers to completely enclose stream 
        self._nbuffs = 1+(stream._nsamp_stream//self._nsamp_buff)

        # total number buffers including overlaps
        self._nbuffs_tot = 2*self._nbuffs-1

        # tent function for linearly x-fading buffers on recombination
        # self.fade = 1.-abs(np.linspace(1,-1, self._nsamp_buff))
        self.fade = hann(self._nsamp_buff)

        # pad the stream up to an exact multiple of buffer sample length
        self.nsamp_padstream = self._nbuffs * self._nsamp_buff
        self.nsamp_pad = self.nsamp_padstream-stream._nsamp_stream
        self._olap_pad = self.nsamp_pad-self._nsamp_halfbuff
        self._olap_lim = min(stream._nsamp_stream, stream._nsamp_stream+self._olap_pad)
        
        # construct tile and overlap buffer arrays
        self.buffs_tile = np.pad(stream.values, (0,self.nsamp_pad)
                                 ).reshape((self._nbuffs, self._nsamp_buff))
        self.buffs_olap = np.pad(stream.values[self._nsamp_halfbuff:self._olap_lim],
                                 (0,max(0, self._olap_pad))
                                 ).reshape((self._nbuffs-1), self._nsamp_buff)

    def to_stream(self):
        """Reconstruct stream by cross-fading buffers

        Takes the `self.buffs_tile` and `self.buffs_olap` arrays and using
        the `self.fade` window function, add overlapping sample values
        together to yield a 1d array of samples.

        Returns:
          out (:obj:`ndarray`): 1d array of sample values representing the
            new audio signal for the parent `Stream`. 
        """
        # apply fades to buffers, first special edge cases...
        self.buffs_tile[0,self._nsamp_halfbuff:] *= self.fade[self._nsamp_halfbuff:]
        self.buffs_tile[-1,:self._nsamp_halfbuff] *= self.fade[:self._nsamp_halfbuff]

        # ...then to remaining buffers
        self.buffs_tile[1:-1,:] *= self.fade.T
        self.buffs_olap *= self.fade.T

        # reconstruct stream
        padded_stream = np.zeros(self.nsamp_padstream)
        padded_stream += self.buffs_tile.flatten()
        flat_olaps    = self.buffs_olap.flatten()
        padded_stream[self._nsamp_halfbuff:-self._nsamp_halfbuff] += flat_olaps

        # remove padding on returning reconstructed stream
        return padded_stream[:-self.nsamp_pad]
