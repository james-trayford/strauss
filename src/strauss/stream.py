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
import warnings
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.signal.windows import hann

# frequency limits a filter cutoff sweeps between, in Hz, from the bottom
# of hearing to just under the Nyquist frequency of a 44.1 kHz signal. The
# cutoff parameter is mapped logarithmically between them
filter_freq_lims = (20., 2.205e4)

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
                   flo=filter_freq_lims[0], fhi=filter_freq_lims[1],
                   qlo=0.5, qhi=10):
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

    def convolve_sweep(self, irfunc, apply=True):
        """Apply a time varying impulse response to the buffered stream

        The convolution analogue of :func:`~stream.Stream.filt_sweep`.
        Each buffer is convolved with the impulse response evaluated at
        that buffer's position through the stream, and the buffers are
        cross-faded back together in the same way. Every buffer is
        given the input samples preceding it, so that it carries the
        same tail it would have had were the whole stream convolved at
        once.

        An impulse response function may return several responses at
        once, in which case the stream is convolved with each of them
        and one output returned per response. This is how a binaural
        pair is produced, sharing the cost of transforming the input
        between the two ears.

        Args:
          irfunc (function): function returning the impulse response at
            a fractional position through the stream, of shape
            (`ntaps`,) or (`nout`, `ntaps`). All responses returned
            must be the same length.
          apply (optional, :obj:`bool`): where a single response is
            returned, also assign the result to the stream values, as
            :func:`~stream.Stream.filt_sweep` does

        Returns:
          out (:obj:`ndarray`): (`nout`, `nsamples`) array of the
            convolved stream, one row per impulse response
        """
        if not hasattr(self, "buffers"):
            raise Exception("needs bufferized stream, please run 'bufferize' method first.")
        buffers = self.buffers
        if buffers._consumed:
            raise Exception("buffers already consolidated, please run 'bufferize' method again.")

        # array to regularly sample the impulse response at each buffer,
        # matching the convention filt_sweep uses for its mapping functions
        x = np.linspace(0, 1, buffers._nbuffs_tot)

        h = np.atleast_2d(np.asarray(irfunc(x[0]), dtype=float))
        nout, ntaps = h.shape
        nsamp_buff = buffers._nsamp_buff

        if nsamp_buff < 4*ntaps:
            warnings.warn(f"buffer of {nsamp_buff} samples is short for a {ntaps} "
                          "tap impulse response, consider a longer 'bufflength'.")

        # samples of input history each buffer needs to reproduce the tail
        # it would have had within a convolution of the entire stream
        nsamp_pre = ntaps - 1

        tile = np.zeros((nout, buffers._nbuffs,   nsamp_buff))
        olap = np.zeros((nout, buffers._nbuffs-1, nsamp_buff))

        # loop over buffers, convolving each with its impulse response
        for j in range(buffers._nbuffs_tot):
            if j:
                h = np.atleast_2d(np.asarray(irfunc(x[j]), dtype=float))
            start = buffers._buff_start(j)
            seg = np.concatenate([buffers.lookbehind(j, nsamp_pre),
                                  buffers._src_padded[start:start+nsamp_buff]])

            # trim to the steady state part of the convolution, that is
            # exactly the samples belonging to this buffer's span
            y = fftconvolve(seg[None,:], h, mode='full',
                            axes=-1)[:, nsamp_pre:nsamp_pre+nsamp_buff]

            i, is_olap = divmod(j, 2)
            if is_olap:
                olap[:,i] = y
            else:
                tile[:,i] = y

        # finally, consolidate each set of buffers into an output stream
        out = np.array([buffers._consolidate(tile[k], olap[k]) for k in range(nout)])
        buffers._consumed = True

        if apply and (nout == 1):
            self.values = out[0]

        return out

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
            self.buffers._src_padded *= 0.
        
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
    # Note: buffers are indexed across the brickwork by a single index
    # `j` running from 0 to `_nbuffs_tot - 1`, where even `j` are tiled
    # buffers and odd `j` are the overlaps between them.

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
        
        # an un-windowed copy of the padded source, so that effects needing
        # the input preceding a buffer (e.g. convolution) can look behind it
        self._src_padded = np.pad(stream.values, (0,self.nsamp_pad))

        # flag set once the buffers have been consolidated, after which
        # they have been windowed in place and cannot be reused
        self._consumed = False

        # construct tile and overlap buffer arrays. Note the tiles must be a
        # copy rather than a reshaped view, as they are written to in place
        # by the effects that operate on them
        self.buffs_tile = self._src_padded.copy(
                                 ).reshape((self._nbuffs, self._nsamp_buff))
        self.buffs_olap = np.pad(stream.values[self._nsamp_halfbuff:self._olap_lim],
                                 (0,max(0, self._olap_pad))
                                 ).reshape((self._nbuffs-1), self._nsamp_buff)

    def _buff_start(self, j):
        """Index into the padded source at which a buffer begins

        Buffers are indexed across the brickwork as
        :func:`~stream.Stream.filt_sweep` maps them, with even `j`
        being ``buffs_tile[j//2]`` and odd `j` being
        ``buffs_olap[j//2]``. Both series advance by half a buffer, so
        the start index is simply half a buffer per step.

        Args:
          j (:obj:`int`): index of the buffer across the brickwork,
            from 0 to ``self._nbuffs_tot - 1``

        Returns:
          start (:obj:`int`): first sample of the buffer in the padded
            source stream
        """
        return j * self._nsamp_halfbuff

    def lookbehind(self, j, nsamp):
        """The un-windowed input samples preceding a buffer

        Used by effects that need the signal history to produce the
        same output a buffer would have had within a whole-stream
        operation. Zero-padded at the head for the first buffers of the
        stream, where no history exists.

        Args:
          j (:obj:`int`): index of the buffer across the brickwork
          nsamp (:obj:`int`): number of preceding samples wanted

        Returns:
          pre (:obj:`ndarray`): `nsamp` samples preceding buffer `j`
        """
        if nsamp <= 0:
            return np.zeros(0)
        start = self._buff_start(j)
        pre = self._src_padded[max(0, start-nsamp):start]
        if pre.size < nsamp:
            pre = np.pad(pre, (nsamp-pre.size, 0))
        return pre

    def _consolidate(self, buffs_tile, buffs_olap):
        """Cross-fade a tile and overlap buffer pair into a single stream

        Operates on the buffer arrays it is passed rather than those
        held by this object, so that one set of buffer geometry can
        consolidate several signals, such as the left and right ear of
        a binaural pair.

        Args:
          buffs_tile (:obj:`ndarray`): 2d array of tiled buffers
          buffs_olap (:obj:`ndarray`): 2d array of overlap buffers

        Returns:
          out (:obj:`ndarray`): 1d array of sample values representing
            the new audio signal
        """
        # work on copies, as the fades are applied in place
        bt = buffs_tile.copy()
        bo = buffs_olap.copy()

        # apply fades to buffers, first special edge cases...
        bt[0,self._nsamp_halfbuff:] *= self.fade[self._nsamp_halfbuff:]
        bt[-1,:self._nsamp_halfbuff] *= self.fade[:self._nsamp_halfbuff]

        # ...then to remaining buffers
        bt[1:-1,:] *= self.fade.T
        bo *= self.fade.T

        # reconstruct stream
        padded_stream = np.zeros(self.nsamp_padstream)
        padded_stream += bt.flatten()
        flat_olaps    = bo.flatten()
        padded_stream[self._nsamp_halfbuff:-self._nsamp_halfbuff] += flat_olaps

        # remove padding on returning reconstructed stream
        return padded_stream[:-self.nsamp_pad]

    def to_stream(self):
        """Reconstruct stream by cross-fading buffers

        Takes the `self.buffs_tile` and `self.buffs_olap` arrays and using
        the `self.fade` window function, add overlapping sample values
        together to yield a 1d array of samples.

        Returns:
          out (:obj:`ndarray`): 1d array of sample values representing the
            new audio signal for the parent `Stream`.
        """
        out = self._consolidate(self.buffs_tile, self.buffs_olap)
        self._consumed = True
        return out
