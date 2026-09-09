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

Todo:
  * Support measured HRTF sets alongside the modelled one
"""

import numpy as np
import warnings
from scipy.signal import bilinear, lfilter

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

        # half width in samples of the fractional delay kernel
        self._half = 16

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


hrtf_providers = {'spherical': SphericalHeadHRTF}


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
