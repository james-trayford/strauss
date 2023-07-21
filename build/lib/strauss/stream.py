import numpy as np
import wavio
import matplotlib.pyplot as plt
# To Do
# - implement filter Q-parameter mapping

class Stream:
    """ Stream object representing audio samples"""
    def __init__(self, length, samprate=44100):
        # variables we want to keep constant
        self.length = length
        self.samprate = samprate
        self._nyqfrq = 0.5*self.samprate
        self._nsamp_stream = int(samprate * length)
        
        # sample values initialised to 0 (silence)
        self.values =  np.zeros(self._nsamp_stream)

        # private stream for keeping track of buffered stream
        self._bvalues =  np.zeros(self._nsamp_stream)
        
        # sample numbers for indexing
        self.samples = np.arange(self._nsamp_stream, dtype=int)

        # time at which each sample occurs
        self.samptime = self.samples / self.samprate
        
    def bufferize(self, bufflength=0.1):
        """ wrapper to initialise Buffers subclass """
        self.buffers = Buffers(self, bufflength)

    def consolidate_buffers(self):
        """ wrapper to reassign stream values to consolidated stream """
        self.values = self.buffers.to_stream()
        
    def filt_sweep(self, ffunc, fmap, qmap=lambda x:x*0 + 0.1,
                   flo=20, fhi=2.205e4, qlo=0.5, qhi=10):
        """
        ffunc: function that applies filter 
        fmap: mapping function representing filter cutoff sweep
        qmap: mapping function for a filters Q parameter, default: lambda:None
        flo: lowest frequency of sweep in Hz, default 20
        fhi: highest frequency of sweep in Hz, default 22.05 kHz
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
        self.sampfracs = np.linspace(0, 1, self.values.size)
        
    def save_wav(self, filename):
        """ save audio stream to wav file, specified by filename"""
        wavio.write(filename, self.values, self.samprate, sampwidth=3) 
        
    def reset(self):
        """ zero audio stream and buffers if present """
        self.values *= 0.
        if hasattr(self, "buffers"):
            self.buffers.buffs_tile *= 0.
            self.buffers.buffs_olap *= 0.        
        
class Buffers: 
    def __init__(self, stream, bufflength=0.1):
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
        self.fade = 1.-abs(np.linspace(1,-1, self._nsamp_buff))

        # pad the stream up to an exact multiple of buffer sample length
        self.nsamp_padstream = self._nbuffs * self._nsamp_buff
        self.nsamp_pad = self.nsamp_padstream-stream._nsamp_stream
        self.olap_pad = self.nsamp_pad-self._nsamp_halfbuff
        self.olap_lim = min(stream._nsamp_stream, stream._nsamp_stream+self.olap_pad)
        
        # construct tile and overlap buffer arrays
        self.buffs_tile = np.pad(stream.values, (0,self.nsamp_pad)
                                 ).reshape((self._nbuffs, self._nsamp_buff))
        self.buffs_olap = np.pad(stream.values[self._nsamp_halfbuff:self.olap_lim],
                                 (0,max(0, self.olap_pad))
                                 ).reshape((self._nbuffs-1), self._nsamp_buff)

    def to_stream(self):
        """ reconstruct stream by x-fading buffers """
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
