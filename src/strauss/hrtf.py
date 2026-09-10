"""The :obj:`hrtf` submodule: head-related transfer functions.

A head-related transfer function (HRTF) describes how the head, and to
a lesser extent the shoulders and the outer ear, filter a sound on its
way to each eardrum. Two cues dominate: the sound reaches the nearer
ear first (the *interaural time difference*, or ITD), and the head
casts an acoustic shadow that makes it quieter at the further ear (the
*interaural level difference*, or ILD). Convolving a source with the
pair of impulse responses for its direction places it around the
listener in a way that simple volume panning cannot, though the effect
is only correct over headphones.

This submodule provides HRTFs as impulse response pairs, which the
:obj:`"binaural"` audio setup of the :class:`~strauss.channels.audio_channels`
sweeps over a moving source using
:func:`~strauss.stream.Stream.convolve_sweep`.

Directions follow the STRAUSS convention throughout: an azimuth in
radians with 0 straight ahead and 0.5 pi to the listener's left, and a
polar angle in radians measured from directly overhead, so that 0.5 pi
is the horizontal plane.

Two providers are available: :class:`SphericalHeadHRTF`, modelled on a
rigid sphere and needing no data, and :class:`MeasuredHRTF`, which
interpolates a set of real measurements and so carries the outer ear
filtering that tells front from back.
"""

import numpy as np
import warnings
from pathlib import Path
from scipy.signal import bilinear, lfilter, fftconvolve, resample_poly
from scipy.spatial import ConvexHull

# speed of sound in air, m/s
sound_speed = 343.

# radius of an average human head in m, the standard value used with the
# Woodworth-Schlosberg formula below
head_radius = 0.0875


def direction(azi, polar):
    """Unit vector pointing towards a direction.

    In listener-centred cartesian coordinates, with `x` straight ahead,
    `y` to the listener's left and `z` upwards.

    Args:
      azi (:obj:`float`): azimuth in radians, 0 ahead and 0.5 pi left
      polar (:obj:`float`): polar angle in radians from directly overhead

    Returns:
      vec (:obj:`ndarray`): 3 element unit vector
    """
    sinp = np.sin(polar)
    return np.array([sinp*np.cos(azi), sinp*np.sin(azi), np.cos(polar)])


class HRTFProvider:
    """Maps a direction to a pair of head-related impulse responses.

    Base class defining the interface the binaural renderer uses.
    Subclasses implement :func:`_ir`, returning the left and right ear
    impulse responses for a direction; this class adds the caching and
    angle wrapping around it.

    Args:
      samprate (optional, :obj:`int`): sample rate the responses are
        generated for, in Hz
      ntaps (optional, :obj:`int`): length in samples of every response
        returned. Defaults to a value the subclass considers sufficient
      bufflength (optional, :obj:`float`): duration in seconds of the
        buffers the renderer should sweep this provider over
      cache_bins (optional, :obj:`int`): number of directions per full
        turn of azimuth that responses are computed at. Directions are
        quantised to this grid before lookup, so a static source costs a
        single response and a moving one reuses its neighbours

    Attributes:
      samprate (:obj:`int`): sample rate of the responses in Hz
      ntaps (:obj:`int`): length in samples of every response
      bufflength (:obj:`float`): buffer duration in seconds to sweep over
    """
    def __init__(self, samprate=48000, ntaps=None, bufflength=0.03,
                 cache_bins=720):
        self.samprate = int(samprate)
        self.bufflength = bufflength
        self.cache_bins = int(cache_bins)

        # polar angle covers half a turn, so it wants half as many bins
        self._polar_bins = max(1, self.cache_bins // 2)

        if ntaps is None:
            ntaps = self._default_ntaps()
        self.ntaps = int(ntaps)

        self._cache = {}
        self._nlookup = 0
        self._nbuild = 0

    # half width in samples of the fractional delay kernel
    _half = 16

    def _frac_delay(self, delay):
        """Windowed sinc kernel delaying a signal by a fractional sample count.

        Normalised to unit sum, which makes the gain at DC exactly one
        and the group delay exactly `delay`, and leaves the kernel
        varying smoothly as the delay does so that a moving source is
        free of zipper artefacts.

        Args:
          delay (:obj:`float`): delay in samples

        Returns:
          h (:obj:`ndarray`): `ntaps` length kernel
        """
        x = np.arange(self.ntaps) - delay
        half = self._half
        # Blackman window, zeroed outside its half width
        w = np.where(abs(x) <= half,
                     0.42 + 0.5*np.cos(np.pi*x/half)
                     + 0.08*np.cos(2*np.pi*x/half), 0.)
        h = np.sinc(x)*w
        return h/h.sum()

    def _default_ntaps(self):
        """Response length in samples, where the caller gives none."""
        return int(2**np.ceil(np.log2(0.003*self.samprate)))

    def _ir(self, azi, polar):
        """Left and right ear impulse responses for a direction.

        Implemented by subclasses, and called only for directions that
        are not already cached.

        Args:
          azi (:obj:`float`): azimuth in radians, wrapped to [0, 2 pi)
          polar (:obj:`float`): polar angle in radians, in [0, pi]

        Returns:
          ir (:obj:`ndarray`): (2, `ntaps`) array of left then right ear
            impulse responses
        """
        raise NotImplementedError("HRTFProvider subclasses must define '_ir'.")

    def ir(self, azi, polar):
        """Cached impulse response pair for a direction.

        The azimuth is wrapped before lookup. This matters: evolving
        azimuths are unwrapped by the :obj:`sources` submodule so that a
        source takes the shortest angular path between two points, which
        legitimately produces values outside a single turn.

        Args:
          azi (:obj:`float`): azimuth in radians, 0 ahead and 0.5 pi left
          polar (:obj:`float`): polar angle in radians from overhead

        Returns:
          ir (:obj:`ndarray`): (2, `ntaps`) array of left then right ear
            impulse responses
        """
        self._nlookup += 1
        abin = int(round((azi % (2*np.pi))/(2*np.pi)
                         * self.cache_bins)) % self.cache_bins
        pbin = int(round(np.clip(polar, 0., np.pi)/np.pi * self._polar_bins))
        key = (abin, pbin)

        h = self._cache.get(key)
        if h is None:
            # evaluate at the centre of the bin, so that a direction
            # always yields the same response however it was reached
            self._nbuild += 1
            h = self._cache[key] = self._ir(2*np.pi*abin/self.cache_bins,
                                            np.pi*pbin/self._polar_bins)
        return h

    def ir_pair_func(self, azifunc, polfunc):
        """Adapt a pair of angle functions to an impulse response function.

        Args:
          azifunc (function): azimuth in radians at a fractional
            position through a note
          polfunc (function): polar angle in radians at a fractional
            position through a note

        Returns:
          irfunc (function): impulse response pair at a fractional
            position, in the form :func:`~strauss.stream.Stream.convolve_sweep`
            expects
        """
        return lambda x: self.ir(float(azifunc(x)), float(polfunc(x)))


class SphericalHeadHRTF(HRTFProvider):
    """HRTFs modelled on a rigid sphere, needing no measured data.

    The head is treated as a sphere with an ear at each side. Three
    effects are combined:

    * an interaural time difference from the Woodworth-Schlosberg
      formula, which takes the direct path to the nearer ear and the
      wave creeping around the sphere to the further one,
    * the head shadow, as the one pole one zero shelving filter of
      Brown & Duda (1998), which leaves low frequencies alone and
      attenuates high ones at the shadowed ear,
    * an elevation cue, as a single reflection standing in for the
      outer ear, whose notch frequency rises as a source rises.

    The elevation cue is applied identically to both ears, so it can
    never disturb the interaural cues. Being modelled rather than
    measured, it gives a clear sense of left and right but only a weak
    sense of height, and no front-to-back discrimination.

    Args:
      radius (optional, :obj:`float`): head radius in metres
      alpha_min (optional, :obj:`float`): shelf gain at the most
        shadowed angle, setting the depth of the head shadow
      theta_min (optional, :obj:`float`): angle in degrees from the ear
        axis at which the shadow is deepest. Past this the shadow eases
        again, reproducing the bright spot directly behind the head
      notch_lims (optional, :obj:`tuple`): frequencies in Hz of the
        elevation notch for a source directly below and directly above
      pinna_gain (optional, :obj:`float`): strength of the reflection
        producing the elevation notch

    Raises:
      Exception: if `ntaps` is too short to contain the response
    """
    def __init__(self, radius=head_radius, alpha_min=0.1, theta_min=150.,
                 notch_lims=(6000., 10000.), pinna_gain=0.35, **kwargs):
        self.radius = radius
        self.alpha_min = alpha_min
        self.theta_min = theta_min
        self.pinna_gain = pinna_gain

        # time in seconds for sound to travel a head radius, which sets
        # the scale of every delay in the model
        self._tau = radius/sound_speed

        super().__init__(**kwargs)

        self._notch_lims = np.clip(notch_lims, 20., 0.45*self.samprate)

        # a constant offset making every delay causal, large enough that
        # the kernel for the shortest path still fits in front of it
        self._n0 = int(np.ceil(self._half + self._tau*self.samprate)) + 2

        # the shelf leaves the response flat at DC, so the pair of
        # kernels summing to one plus the reflection sets the overall
        # gain. Normalise so a frontal source reaches each ear at the
        # half amplitude the stereo setup would give it, keeping a
        # binaural render at the same loudness as a stereo one
        self._norm = (1. + self.pinna_gain)/0.5

        need = self._n0 + self._tau*(0.5*np.pi)*self.samprate \
            + self.samprate/(2*self._notch_lims[0]) + self._half \
            + 4*self._tau*self.samprate
        if self.ntaps < need:
            raise Exception(f"ntaps of {self.ntaps} too short for this model, "
                            f"which needs at least {int(np.ceil(need))} at "
                            f"{self.samprate} Hz.")

    def _default_ntaps(self):
        # enough for the longest path around the head, the elevation
        # reflection behind it and the tail of the shelving filter
        need = self._half + 2*self._tau*self.samprate \
            + 0.5*self.samprate/1000. + 4*self._tau*self.samprate + 8
        return int(2**np.ceil(np.log2(need)))

    def _shadow(self, h, gamma):
        """Apply the head shadow for an angle from the ear axis.

        The shelving filter of Brown & Duda (1998), whose gain falls
        from twice the free field value at the ear facing the source to
        `alpha_min` at `theta_min` away from it, easing again past that
        to leave a bright spot behind the head. The response is flat at
        DC whatever the angle, so the head shadow contributes no overall
        level change, only a spectral tilt above a few hundred Hz.

        Args:
          h (:obj:`ndarray`): impulse response to filter
          gamma (:obj:`float`): angle in radians between the direction
            and this ear's axis

        Returns:
          h (:obj:`ndarray`): filtered impulse response
        """
        alpha = (1 + self.alpha_min/2) \
            + (1 - self.alpha_min/2)*np.cos(np.degrees(gamma)/self.theta_min*np.pi)

        # the sphere's natural frequency, where the shadow sets in
        w0 = sound_speed/self.radius
        bz, az = bilinear([alpha/(2*w0), 1.], [1./(2*w0), 1.], fs=self.samprate)
        return lfilter(bz, az, h)

    def _ir(self, azi, polar):
        """Modelled impulse response pair, see :func:`HRTFProvider._ir`."""
        # the lateral cue, +1 at an ear and -1 at the opposite one, and
        # zero on the median plane and at the poles. Being a product of
        # sines it is smooth everywhere, so a source passing in front,
        # behind or overhead crosses the midline without a discontinuity
        lat = np.sin(polar)*np.sin(azi)

        # a source below the horizon reflects with a longer delay than
        # one above it, putting the notch it produces at a lower
        # frequency, which is the direction the ear reads as elevation
        elev = 0.5*np.pi - polar
        notch = np.interp(np.sin(elev), [-1., 1.], self._notch_lims)
        nsamp_pinna = self.samprate/(2*notch)

        ears = []
        for u in (lat, -lat):
            gamma = np.arccos(np.clip(u, -1., 1.))

            # Woodworth-Schlosberg: a direct ray to the near side of the
            # head, and a wave creeping around it to the far side. The
            # two branches meet smoothly where the source is side on
            if gamma <= 0.5*np.pi:
                delay = -self._tau*np.cos(gamma)
            else:
                delay = self._tau*(gamma - 0.5*np.pi)
            delay = self._n0 + delay*self.samprate

            # the direct sound, plus the reflection carrying the
            # elevation cue at the same offset in both ears
            h = self._frac_delay(delay) \
                + self.pinna_gain*self._frac_delay(delay + nsamp_pinna)

            ears.append(self._shadow(h, gamma))

        return np.array(ears)/self._norm


def _minimum_phase(h, nfft=None):
    """Minimum phase counterpart of an impulse response.

    Built from the real cepstrum, folding the non-causal half onto the
    causal one. The magnitude spectrum is preserved exactly; what is
    discarded is the excess phase, which for a head-related response is
    very nearly a pure delay and is handled separately.

    Args:
      h (:obj:`ndarray`): impulse response
      nfft (optional, :obj:`int`): transform length, defaulting to
        eight times the response length to keep wraparound negligible

    Returns:
      hmin (:obj:`ndarray`): minimum phase response, same length as `h`
    """
    n = h.size
    nfft = nfft or 8*n
    mag = np.abs(np.fft.rfft(h, nfft))
    # floor the magnitude so that its logarithm stays finite
    mag = np.maximum(mag, mag.max()*1e-8)
    cep = np.fft.irfft(np.log(mag), nfft)
    fold = np.zeros(nfft)
    fold[0] = 1.
    fold[1:nfft//2] = 2.
    fold[nfft//2] = 1.
    return np.fft.irfft(np.exp(np.fft.rfft(cep*fold, nfft)), nfft)[:n]


def _excess_delay(h, hmin, samprate, flo=200., fhi=1500., nfft=None):
    """Delay in samples between a response and its minimum phase form.

    Measured from the phase slope of the all-pass component the two
    differ by, over a band low enough that the response is smooth and
    strong. Cross-correlating the two instead is tempting but much less
    reliable: for a shadowed ear the response is diffuse, its
    correlation peak ambiguous, and the resulting delays vary sharply
    between neighbouring measurements, which then shows up as a
    stepping interaural time difference once they are interpolated.

    Args:
      h (:obj:`ndarray`): impulse response
      hmin (:obj:`ndarray`): its minimum phase counterpart
      samprate (:obj:`int`): sample rate of the response in Hz
      flo (optional, :obj:`float`): bottom of the fitting band in Hz
      fhi (optional, :obj:`float`): top of the fitting band in Hz
      nfft (optional, :obj:`int`): transform length

    Returns:
      delay (:obj:`float`): delay in samples
    """
    nfft = nfft or 8*h.size
    spec = np.fft.rfft(hmin, nfft)
    # the minimum phase form shares the magnitude, so their ratio is an
    # all-pass whose phase carries the delay and nothing else
    allpass = np.fft.rfft(h, nfft)/np.where(np.abs(spec) < 1e-12, 1e-12, spec)

    freq = np.fft.rfftfreq(nfft, 1./samprate)
    band = (freq >= flo) & (freq <= fhi)
    bins = np.arange(nfft//2 + 1)[band]
    phase = np.unwrap(np.angle(allpass))[band]

    slope = np.polyfit(bins, phase, 1)[0]
    return -slope*nfft/(2*np.pi)


def kemar_to_npz(kemar_dir, outfile):
    """Convert the MIT KEMAR compact measurements to a STRAUSS HRTF file.

    The compact set gives one stereo response per direction over half
    the sphere, the other half following by symmetry. Each response is
    split into its minimum phase part and a pure delay, so that
    responses for neighbouring directions can be blended without the
    comb filtering that averaging differently delayed copies would
    cause.

    Note the MIT set measures azimuth towards the listener's right,
    where STRAUSS measures it towards the left, so the azimuths are
    negated on the way in. This was established from the data rather
    than the documentation: at MIT azimuth 90 the second channel leads
    by 32 samples and is 14 dB the louder.

    Args:
      kemar_dir (:obj:`str` or :obj:`Path`): directory holding the
        `elev<N>` folders of the unpacked compact archive
      outfile (:obj:`str` or :obj:`Path`): `.npz` file to write

    Returns:
      outfile (:obj:`Path`): the file written
    """
    import glob
    import os
    import re
    from scipy.io import wavfile

    rows = []
    rate = None

    # the asset fetcher hands back the folder of the first file it
    # unpacked, which for this archive is one of the elevation folders
    root = Path(kemar_dir)
    if root.name.startswith('elev'):
        root = root.parent
    files = sorted(glob.glob(str(Path(root, '**', 'H*e*a.wav')), recursive=True))
    if not files:
        raise Exception(f"no KEMAR measurements found under \"{root}\".")

    for f in files:
        m = re.match(r'H(-?\d+)e(\d+)a\.wav', os.path.basename(f))
        elev, mit_azi = int(m.group(1)), int(m.group(2))
        rate, wav = wavfile.read(f)
        # (2, nsamples), as left and right ear
        ears = wav.astype(np.float64).T/32768.

        # elevation above the horizon becomes polar angle from overhead
        polar = np.radians(90. - elev)

        # the measured half of the sphere lies to the right in the
        # STRAUSS convention, and mirrors onto the left with the ears
        # exchanged
        entries = [((360 - mit_azi) % 360, ears)]
        if mit_azi not in (0, 180):
            entries.append((mit_azi, ears[::-1]))

        for azi, pair in entries:
            hmin = np.array([_minimum_phase(pair[e]) for e in (0, 1)])
            delay = np.array([_excess_delay(pair[e], hmin[e], rate) for e in (0, 1)])
            rows.append((np.radians(azi), polar, hmin, delay))

    azi = np.array([r[0] for r in rows])
    polar = np.array([r[1] for r in rows])
    hrir = np.array([r[2] for r in rows], dtype=np.float32)
    itd = np.array([r[3] for r in rows])

    # delays are only meaningful relative to one another, so referencing
    # them to the earliest keeps the responses as short as possible
    itd -= itd.min()

    outfile = Path(outfile)
    np.savez_compressed(outfile, azi=azi, polar=polar, hrir=hrir, itd=itd,
                        samprate=rate, convention='strauss-v1',
                        source='MIT KEMAR compact, Gardner & Martin, '
                               'Copyright 1994 MIT Media Laboratory')
    return outfile


class MeasuredHRTF(HRTFProvider):
    """HRTFs interpolated from a set of measured responses.

    Measured responses carry the filtering of a real outer ear, which
    the modelled :class:`SphericalHeadHRTF` cannot: a sphere is
    symmetric front to back, so it places a source ahead and the same
    source behind identically. The folds of a real pinna break that
    symmetry, giving front from back and a usable sense of elevation.

    Responses are stored as a minimum phase part and a pure delay.
    Blending the two separately is what makes interpolation safe:
    averaging responses that carry different delays would comb filter
    the result. The delay is put back with the same fractional delay
    kernel the modelled HRTFs use.

    A direction is looked up by finding the triangle of measured
    directions containing it and weighting its three responses
    barycentrically. The triangulation is the convex hull of the
    measured directions, which for points on a sphere is their
    Delaunay triangulation, and so copes with the irregular rings and
    the poles of a real measurement grid.

    Args:
      path (optional, :obj:`str`): `.npz` file of measurements to load.
        Defaults to the MIT KEMAR set, downloaded and converted the
        first time it is asked for
      samprate (optional, :obj:`int`): sample rate to provide responses
        at, resampling the measurements if they were made at another

    Raises:
      Exception: if the file does not carry a convention this
        understands, which would leave its directions ambiguous
    """
    def __init__(self, path=None, samprate=48000, **kwargs):
        if path is None:
            path = kemar_path()
        data = np.load(path, allow_pickle=False)

        convention = str(data['convention'])
        if convention != 'strauss-v1':
            raise Exception(f"HRTF file \"{path}\" uses convention "
                            f"\"{convention}\", which this version does not "
                            "know how to orient. Directions would be ambiguous.")

        self.path = path
        self.source = str(data['source']) if 'source' in data else 'unknown'
        self._hrir = data['hrir'].astype(np.float64)
        self._itd = data['itd'].astype(np.float64)
        file_rate = int(data['samprate'])

        samprate = int(samprate)
        if samprate != file_rate:
            # measurements are rarely made at the rate we render at
            gcd = np.gcd(samprate, file_rate)
            self._hrir = resample_poly(self._hrir, samprate//gcd,
                                       file_rate//gcd, axis=-1)
            self._itd *= samprate/file_rate

        # offset making every delay causal, leaving room for the kernel
        self._n0 = self._half + 2

        super().__init__(samprate=samprate, **kwargs)

        # unit vectors towards each measured direction, and the triangles
        # of the sphere they span
        self._xyz = np.asarray(direction(data['azi'], data['polar'])).T
        simplices = ConvexHull(self._xyz).simplices
        mats = self._xyz[simplices].transpose(0, 2, 1)

        # a triangle whose vertices lie on a great circle is edge on to
        # the origin and cannot be solved for, so drop any that appear
        keep = np.abs(np.linalg.det(mats)) > 1e-9
        self._simplices = simplices[keep]
        self._Tinv = np.linalg.inv(mats[keep])

        # match the gain the modelled HRTFs and the stereo setup give a
        # source straight ahead, so switching between them is level
        self._norm = 1.
        self._norm = np.mean(self._ir(0., 0.5*np.pi).sum(axis=1))/0.5

    def _default_ntaps(self):
        # the measurements, the delay put back, and the kernel doing it
        return int(self._hrir.shape[-1] + self._n0 + self._itd.max()
                   + self._half + 4)

    def _ir(self, azi, polar):
        """Interpolated impulse response pair, see :func:`HRTFProvider._ir`."""
        s = direction(azi % (2*np.pi), np.clip(polar, 0., np.pi))

        # barycentric weights of the ray through s in every triangle, of
        # which the containing one is that with no negative weight
        weights = self._Tinv @ s
        inside = np.all(weights >= -1e-9, axis=1)

        if inside.any():
            k = int(np.argmax(inside))
            idx = self._simplices[k]
            wts = weights[k]/weights[k].sum()
        else:
            # no triangle claims it, so fall back on the nearest
            # measurement rather than failing
            idx = np.array([int(np.argmax(self._xyz @ s))])
            wts = np.array([1.])

        hmin = (self._hrir[idx] * wts[:, None, None]).sum(axis=0)
        delay = (self._itd[idx] * wts[:, None]).sum(axis=0)

        ir = np.array([fftconvolve(hmin[e],
                                   self._frac_delay(self._n0 + delay[e]))[:self.ntaps]
                       for e in (0, 1)])
        return ir/self._norm


def kemar_path():
    """Path to the converted MIT KEMAR measurements, preparing them if needed.

    The archive is fetched from the MIT Media Laboratory and converted
    on first use, then cached alongside the other STRAUSS assets.

    Returns:
      path (:obj:`Path`): the `.npz` file of converted measurements
    """
    from .assets import assets, get_asset_path

    cached = Path(assets.path, 'kemar_strauss.npz')
    if not cached.exists():
        warnings.warn("preparing the MIT KEMAR measurements for first use, "
                      "this takes a few seconds...")
        kemar_to_npz(get_asset_path('kemar'), cached)
    return cached


hrtf_providers = {'spherical': SphericalHeadHRTF,
                  'measured': MeasuredHRTF,
                  'kemar': MeasuredHRTF}


def get_hrtf(name='spherical', **kwargs):
    """Get an HRTF provider by name.

    Args:
      name (optional, :obj:`str`): name of the provider, or an already
        constructed :class:`HRTFProvider` to pass straight through
      **kwargs: passed to the provider

    Returns:
      hrtf (:class:`HRTFProvider`): the requested provider

    Raises:
      Exception: if the named provider is unknown
    """
    if isinstance(name, HRTFProvider):
        return name
    if name not in hrtf_providers:
        raise Exception(f"HRTF provider \"{name}\" unknown, choose from "
                        f"{list(hrtf_providers)}.")
    return hrtf_providers[name](**kwargs)
