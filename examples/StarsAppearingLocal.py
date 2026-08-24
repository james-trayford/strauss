#!/usr/bin/env python
# coding: utf-8

# ### <u> Generate a local "_Stars Appearing_" sequence: sonification + animation </u>
#
# This builds the "_Stars Appearing_" piece from the "_Audible Universe_" planetarium
# show for **any site and any night**, and renders a matching animation.
#
# The sky is computed with `skyfield` from the _Hipparcos_ catalogue, sonified with
# `strauss`, and animated as an **equirectangular** (360 x 180 degree) panorama. A
# planetarium dome master is produced by letting `ffmpeg`'s `v360` filter reproject
# the panorama to fisheye - we never render fisheye ourselves.
#
# The single most important idea here is that the animation does **not** re-derive
# when each star appears. It reads the timings straight out of the rendered
# sonification via `Sonification.event_table()`, so sound and picture cannot drift
# apart.
#
# Extra requirements beyond `strauss`:  `pip install skyfield`
# (and a working `ffmpeg` on your `PATH`).

import subprocess
import shutil
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import tqdm

from strauss.generator import Sampler
from strauss.score import Score
from strauss.sonification import Sonification
from strauss.sources import Events


# <u> __Settings:__ </u>
#
# Everything you would want to change lives in one place. The defaults describe
# the Sherwood Observatory / Sherwood Planetarium site looking south.

@dataclass
class Config:
    """Every knob for one run of the sequence."""

    # -- where and when --------------------------------------------------
    # latitude is +ve north; longitude is +ve *east*, so 1.22 degrees west
    # of Greenwich is -1.22
    latitude: float = 53.1143737
    longitude: float = -1.2219389
    date_time: str = "2026-03-25 18:45:00"      # local wall-clock, YYYY-MM-DD HH:mm:ss
    time_zone: str = "Europe/London"            # TZ identifier, e.g. 'Europe/London'

    # which way is the listener/viewer facing? a cardinal point, and the
    # centre of both the panorama and the stereo/surround image
    facing: str = "S"

    # faintest star to include; larger numbers mean more, dimmer stars
    mag_limit: float = 5.0

    # -- the sound -------------------------------------------------------
    duration: float = 60.0                      # seconds
    system: str = "5.1"                         # 'mono', 'stereo', '5.1', 'ambix2', ...
    instrument: str = "glockenspiel"            # a key of INSTRUMENTS, below

    # -- the picture -----------------------------------------------------
    width: int = 4096
    height: int = 2160
    fps: int = 30

    # peak radius in pixels of the brightest and faintest star pulses
    max_star_px: float = 15.0
    min_star_px: float = 1.0

    # a star flashes rather than persists: it swells over tau_in and decays
    # over tau_out, both in seconds
    tau_in: float = 0.06
    tau_out: float = 0.6

    # opacity of a star at full size
    star_alpha: float = 0.6

    # an equirectangular panorama over-samples the sky near the poles, so
    # pulses are stretched in azimuth to stay round on the dome. Clamped,
    # since the stretch diverges at the zenith itself.
    max_stretch: float = 40.0

    # optional panorama to lay the stars over: a 360x180 degree image with
    # `facing` at its centre. None gives a black sky.
    background: Path | None = None

    # -- outputs ---------------------------------------------------------
    outdir: Path = Path("stars_appearing_out")

    # also reproject the panorama to a fisheye dome master
    dome: bool = True
    dome_pitch: float = 90.0                    # degrees to tilt the dome view up
    dome_size: int | None = None                # square edge; defaults to `height`

    # -- reproducibility -------------------------------------------------
    # magnitudes are jittered to break ties, and notes detuned very slightly
    seed: int = 0

    # where skyfield keeps its downloaded catalogue and ephemeris
    cache: Path = Path("~/.strauss_skyfield").expanduser()

    def __post_init__(self):
        self.outdir = Path(self.outdir)
        if self.background is not None:
            self.background = Path(self.background)
        if self.dome_size is None:
            self.dome_size = self.height


# <u> __Instruments:__ </u>
#
# Each entry gives the chord the stars are drawn from, how to build the
# `Sampler`, and any preset changes. Adding an instrument means adding a key
# here and nothing else.

def _download(url, path):
    """Fetch `url` to `path` once, returning the local path."""
    path = Path(path)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {path.name} ...")
        with urllib.request.urlopen(url) as response:
            path.write_bytes(response.read())
    return path


# the long decay is what makes the piece shimmer: notes pile up rather than
# sounding one at a time
_CHIME_PRESET = {
    "note_length": 5,
    "volume_envelope": {
        "use": "on",
        "A": 0.02,    # fade in to full volume (s)
        "D": 4.5,     # fall from full volume to the sustain level (s)
        "S": 0.0,     # sustain level, as a fraction of full volume
        "R": 0.07,    # release once the note is let go (s)
    },
}

_PLUCK_PRESET = {
    "note_length": 5,
    "volume_envelope": {"use": "on", "A": 0.02, "D": 0.0, "S": 1.0, "R": 0.07},
}

INSTRUMENTS = {
    "glockenspiel": {
        "chords": [["Db3", "Gb3", "Ab3", "Eb4", "F4"]],
        "sampler": lambda cfg: Sampler(Path("..", "data", "samples", "glockenspiels")),
        "preset": _CHIME_PRESET,
    },
    "mallets": {
        # a fast-rendering stand-in while you are still choosing settings
        "chords": [["Db3", "Gb3", "Ab3", "Eb4", "F4"]],
        "sampler": lambda cfg: Sampler(Path("..", "data", "samples", "mallets")),
        "preset": _CHIME_PRESET,
    },
    "qanun": {
        "chords": [["A2", "E3", "A3", "B3", "C4", "D4", "E4", "A4", "B4", "C5",
                    "D5", "E5"]],
        "sampler": lambda cfg: Sampler(str(_download(
            "http://www.ozanyarman.com/files/QanunDrOz.sf2",
            cfg.cache / "qanun.sf2"))),
        "preset": _PLUCK_PRESET,
    },
}


# <u> __The sky:__ </u>
#
# `skyfield` gives us the altitude and azimuth of every catalogue star as seen
# from the chosen site at the chosen instant. We cut the catalogue down by
# magnitude *before* computing positions, which is the difference between
# transforming ~118,000 stars and ~1,500.

CARDINALS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def unit_scale(values):
    """Scale values onto 0-1.

    A handful of stars can share a magnitude or a colour exactly - a
    single constellation, say - which would otherwise divide by a zero
    range. Those degenerate cases land in the middle of the scale.
    """
    values = np.asarray(values, dtype=float)
    low, high = values.min(), values.max()
    if not np.isfinite(high - low) or high == low:
        return np.full(values.shape, 0.5)

    return (values - low) / (high - low)


def facing_degrees(facing):
    """Bearing of a cardinal point in degrees clockwise from north."""
    if facing not in CARDINALS:
        raise ValueError(f"'{facing}' is not a cardinal point. "
                         f"Choose from: {CARDINALS}")
    return 360.0 * CARDINALS.index(facing) / len(CARDINALS)


def load_catalogue(cfg, loader):
    """Read the Hipparcos catalogue, keeping stars we could plausibly show."""
    from skyfield.data import hipparcos

    with loader.open(hipparcos.URL) as f:
        cat = pd.read_csv(
            f, sep="|", names=hipparcos._COLUMN_NAMES, compression=None,
            usecols=["HIP", "Vmag", "RAdeg", "DEdeg", "Plx", "pmRA", "pmDE", "B-V"],
            na_values=["     ", "       ", "        ", "            ", "      "],
        )

    cat.columns = ("hip", "magnitude", "ra_degrees", "dec_degrees",
                   "parallax_mas", "ra_mas_per_year", "dec_mas_per_year", "bv")
    cat = cat.assign(ra_hours=cat["ra_degrees"] / 15.0, epoch_year=1991.25)
    cat = cat.set_index("hip")

    # cut down before doing any astrometry - a star we will never draw or
    # sound is not worth transforming
    keep = (cat["magnitude"] < cfg.mag_limit) & np.isfinite(cat["bv"])

    return cat[keep]


def observed_sky(cfg):
    """Stars above the horizon at the configured place and time.

    Returns:
      sky (:obj:`pandas.DataFrame`): a row per visible star, indexed by
        Hipparcos number, with its altitude and azimuth in degrees,
        `V` magnitude, and `B-V` colour.
    """
    from skyfield.api import Loader, Star, wgs84

    loader = Loader(str(cfg.cache), verbose=True)
    cat = load_catalogue(cfg, loader)

    when = datetime.strptime(cfg.date_time, "%Y-%m-%d %H:%M:%S")
    when = when.replace(tzinfo=ZoneInfo(cfg.time_zone))
    t = loader.timescale().from_datetime(when)

    earth = loader("de421.bsp")["earth"]
    observer = (earth + wgs84.latlon(cfg.latitude, cfg.longitude)).at(t)

    alt, az, _ = observer.observe(Star.from_dataframe(cat)).apparent().altaz()

    sky = pd.DataFrame({"alt": alt.degrees,
                        "az": az.degrees,
                        "magnitude": cat["magnitude"].to_numpy(float),
                        "bv": cat["bv"].to_numpy(float)},
                       index=cat.index)

    sky = sky[np.isfinite(sky["alt"]) & (sky["alt"] > 0)].copy()

    # break ties so that no two stars land on exactly the same instant
    rng = np.random.default_rng(cfg.seed)
    sky["magnitude"] += 1e-2 * rng.random(len(sky))

    return sky.sort_values("magnitude")


# <u> __The sonification:__ </u>
#
# Brightest stars sound first, so magnitude drives `time`. Colour drives
# `pitch`, picking a note from the chord: blue stars take high notes and red
# stars low ones, carrying the short-to-long wavelength of the light onto the
# short-to-long wavelength of the sound. Position drives the spatial angles.
#
# Note on angles: `strauss` measures azimuth **anticlockwise from straight
# ahead** (`stereo` puts L at 90 degrees and R at 270), while astronomical
# azimuth runs **clockwise from north**. That is why `facing - az` below is the
# right way round, and it is the same reason the panorama later uses
# `180 - azimuth` to get its horizontal pixel.
#
# We pass `angle_unit='degrees'` rather than giving `map_lims` for the angles.
# Both map the sound identically, but only `angle_unit` makes `event_table()`
# report real degrees - with `map_lims` the angles come back as raw 0-1
# fractions still labelled `[degrees]`, and the animation would be nonsense.

def build_sonification(sky, cfg):
    """Sonify the visible sky, returning the rendered `Sonification`."""
    if cfg.instrument not in INSTRUMENTS:
        raise ValueError(f"Unknown instrument '{cfg.instrument}'. "
                         f"Choose from: {list(INSTRUMENTS)}")
    voice = INSTRUMENTS[cfg.instrument]

    rng = np.random.default_rng(cfg.seed + 1)
    mag = sky["magnitude"].to_numpy(float)

    # normalised magnitude, 0 brightest to 1 faintest
    smag = unit_scale(mag)

    data = {
        "azimuth": (facing_degrees(cfg.facing) - sky["az"].to_numpy(float)) % 360,
        "polar": 90.0 - sky["alt"].to_numpy(float),   # 0 at the zenith
        "time": mag,
        "pitch": -sky["bv"].to_numpy(float),
        # dimmer stars are far more numerous, so quieten them to keep the
        # overall level roughly even through the piece
        "volume": (1 - smag) ** 0.5,
        "pitch_shift": 5e-3 * rng.random(len(sky)),
    }

    map_lims = {
        # '104%' leaves a tail of silence for the last notes to ring out in
        "time": ("0%", "104%"),
        "pitch": ("0%", "100%"),
        "volume": ("0%", "100%"),
        "pitch_shift": (0, 1),
    }

    events = Events(list(data))
    events.fromdict(data)
    # name each event for its star, so the animation can join back to the
    # catalogue from the event table
    events.names = [f"HIP{h}" for h in sky.index]
    events.apply_mapping_functions(map_lims=map_lims, angle_unit="degrees")

    sampler = voice["sampler"](cfg)
    sampler.modify_preset(voice["preset"])

    soni = Sonification(Score(voice["chords"], cfg.duration), events,
                        sampler, cfg.system)
    soni.render()

    return soni


def timings(soni, sky):
    """Join the sonification's event table to its stars.

    The event table is the single source of truth for *when* and *where*:
    the animation takes its timings and angles from here rather than
    recomputing them, so picture and sound cannot disagree.
    """
    table = soni.event_table()
    # the table carries units in a second column level, which is for reading
    # rather than for arithmetic
    table.columns = table.columns.get_level_values(0)

    events = pd.DataFrame({
        "time": table["Time"].to_numpy(float),
        "azimuth": table["Azimuthal Angle"].to_numpy(float),
        "polar": table["Polar Angle"].to_numpy(float),
    }, index=table["Source"].to_numpy())

    stars = sky[["magnitude", "bv"]].set_axis([f"HIP{h}" for h in sky.index])

    return events.join(stars).sort_values("time")


# <u> __The animation:__ </u>
#
# Each star is a pulse that swells and fades as its note sounds. Frames are
# generated as raw `RGBA` and piped straight into `ffmpeg` - no intermediate
# `PNG`s, so nothing touches the disk between here and the finished video.
#
# Two things keep this quick. A star is only drawn while it is bigger than half
# a pixel, which for these envelopes is around a second out of the whole piece,
# so a binary search over the (sorted) event times finds the handful of stars
# actually alive in each frame. And only the rows those stars touch are cleared
# and converted, rather than the whole 4K canvas.

def star_rgb(bv01):
    """Colour of a star from its normalised `B-V`, 0 bluest to 1 reddest."""
    return 1 - 0.3 * (np.array([1.0, 0.5, 0.0]) - bv01) ** 2


def render_frames(events, cfg):
    """Yield one raw `RGBA` frame per frame of the animation."""
    W, H = cfg.width, cfg.height
    n_frames = int(round(cfg.duration * cfg.fps))

    t_star = events["time"].to_numpy(float)

    # horizontal pixel: strauss azimuth runs anticlockwise from the facing
    # direction, the image runs left to right, and `facing` sits at the centre
    cx = ((180.0 - events["azimuth"].to_numpy(float)) % 360.0) * (W / 360.0)
    cy = events["polar"].to_numpy(float) * (H / 180.0)

    # peak radius: bright stars are big, faint ones small
    brightness = 1 - unit_scale(events["magnitude"].to_numpy(float))
    amp = 1.2 * (cfg.max_star_px * brightness + cfg.min_star_px)

    # colour, clipped to the bulk of the B-V range so a few outliers do not
    # flatten everything else
    bv = events["bv"].to_numpy(float)
    bv01 = unit_scale(np.clip(bv, *np.percentile(bv, [1, 99])))
    rgb = star_rgb(bv01[:, None])

    # azimuthal stretch, so a pulse stays round on the sky - and so stays
    # round on the dome once reprojected. Two things stretch it. A pulse at
    # polar angle `t` spans 1/sin(t) times as much azimuth as it does
    # altitude, diverging at the poles. And the canvas carries 360 degrees
    # across but only 180 down, so unless it is 2:1 its pixels are not square
    # in angle, by a further factor of W/2H.
    aspect = W / (2.0 * H)
    stretch = np.clip(aspect / np.sin(np.pi * np.clip(cy, 0.5, H - 0.5) / H),
                      aspect, cfg.max_stretch)

    # how long before and after its note a star is worth drawing at all
    reach = np.log(max(2.0 * amp.max(), np.e))
    lead, trail = cfg.tau_in * reach, cfg.tau_out * reach

    # premultiplied colour and coverage, reused between frames
    colour = np.zeros((H, W, 3), dtype=np.float32)
    alpha = np.zeros((H, W), dtype=np.float32)
    out = np.zeros((H, W, 4), dtype=np.uint8)

    dirty = (0, H)

    for frame in range(n_frames):
        now = frame / cfg.fps

        # forget the previous frame, but only where it drew something
        y0, y1 = dirty
        colour[y0:y1] = 0.0
        alpha[y0:y1] = 0.0

        lo = np.searchsorted(t_star, now - trail, side="left")
        hi = np.searchsorted(t_star, now + lead, side="right")

        touched_lo, touched_hi = H, 0

        for i in range(lo, hi):
            dt = now - t_star[i]
            env = np.exp(dt / cfg.tau_in) if dt < 0 else np.exp(-dt / cfg.tau_out)
            size = amp[i] * env
            if size < 0.5:
                continue

            sx = stretch[i]
            # a pulse near the zenith stretches a long way in azimuth. Keep it
            # under one full turn, so that no column appears twice once wrapped
            # - a repeat would silently drop one of its contributions.
            rx = min(size * sx, (W - 3) / 2)

            ys0 = max(int(np.floor(cy[i] - size)), 0)
            ys1 = min(int(np.ceil(cy[i] + size)) + 1, H)
            if ys1 <= ys0:
                continue

            xs = np.arange(int(np.floor(cx[i] - rx)),
                           int(np.ceil(cx[i] + rx)) + 1)[:W]
            ys = np.arange(ys0, ys1)

            # radius in units of the (unstretched) pulse
            dx = (xs - cx[i]) / sx
            dy = ys - cy[i]
            r = np.hypot(dx[None, :], dy[:, None])

            # a one-pixel soft edge, standing in for cairo's antialiasing
            cover = np.clip(size + 0.5 - r, 0.0, 1.0).astype(np.float32)
            if not cover.any():
                continue

            idx = np.ix_(ys, xs % W)          # wrap around the back of the sky

            # composite the pulse over whatever is already there
            have = alpha[idx]
            add = cover * cfg.star_alpha * (1.0 - have)
            colour[idx] += rgb[i] * add[..., None]
            alpha[idx] = have + add

            touched_lo, touched_hi = min(touched_lo, ys0), max(touched_hi, ys1)

        # everything the stars did not touch stays fully transparent, so the
        # background shows through untouched
        out[:] = 0

        if touched_hi > touched_lo:
            sl = slice(touched_lo, touched_hi)
            a = alpha[sl]
            # undo the premultiplication, since ffmpeg expects straight alpha
            np.multiply(colour[sl], 255.0 / np.maximum(a, 1e-6)[..., None],
                        out=colour[sl])
            out[sl, :, :3] = np.clip(colour[sl], 0, 255).astype(np.uint8)
            out[sl, :, 3] = (a * 255).astype(np.uint8)

        dirty = (touched_lo, touched_hi) if touched_hi > touched_lo else (0, 0)

        yield out.tobytes()


# <u> __Compositing:__ </u>
#
# One `ffmpeg` pass does the lot: lay the star frames over the background
# panorama, mux the rendered audio, and - for the dome master - reproject the
# equirectangular result to fisheye with the `v360` filter.

def encode(cfg, audio, out_path, dome=False):
    """Build the ffmpeg command for one output, and return it."""
    if cfg.background is not None:
        background = ["-loop", "1", "-framerate", str(cfg.fps),
                      "-i", str(cfg.background)]
    else:
        background = ["-f", "lavfi", "-i",
                      f"color=c=black:s={cfg.width}x{cfg.height}:r={cfg.fps}"]

    chain = (f"[0:v]scale={cfg.width}:{cfg.height},setsar=1,format=rgba[bg];"
             "[bg][1:v]overlay=shortest=1[comp];[comp]")
    if dome:
        # equirectangular in, fisheye out, tilted up so the zenith lands in
        # the middle of the dome
        chain += (f"v360=e:fisheye:h_fov=180:v_fov=180:pitch={cfg.dome_pitch}"
                  f":w={cfg.dome_size}:h={cfg.dome_size},")
    chain += "format=yuv420p[v]"

    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            *background,
            "-f", "rawvideo", "-pixel_format", "rgba",
            "-video_size", f"{cfg.width}x{cfg.height}",
            "-framerate", str(cfg.fps), "-i", "pipe:0",
            "-i", str(audio),
            "-filter_complex", chain,
            "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-crf", "16", "-preset", "medium",
            "-c:a", "aac", "-b:a", "320k",
            "-r", str(cfg.fps), "-shortest", str(out_path)]


def write_videos(cfg, events, audio, targets):
    """Render the frames once, into one ffmpeg process per output.

    The panorama and the dome master are the same pixels reprojected
    differently, so they share a single pass of the frame generator
    rather than drawing everything twice.

    Args:
      targets (:obj:`list`): `(path, dome)` pairs, one per output.

    Returns:
      paths (:obj:`list`): the paths written, in the order given.
    """
    n_frames = int(round(cfg.duration * cfg.fps))
    procs = [subprocess.Popen(encode(cfg, audio, path, dome=dome),
                              stdin=subprocess.PIPE)
             for path, dome in targets]

    label = ", ".join(path.name for path, _ in targets)
    try:
        for frame in tqdm.tqdm(render_frames(events, cfg), total=n_frames,
                               desc=label, unit="frame"):
            for proc in procs:
                proc.stdin.write(frame)
    finally:
        for proc in procs:
            proc.stdin.close()
        failed = [path for (path, _), proc in zip(targets, procs)
                  if proc.wait() != 0]

    if failed:
        raise RuntimeError("ffmpeg failed while writing "
                           f"{', '.join(str(p) for p in failed)}")

    return [path for path, _ in targets]


def write_video(cfg, events, audio, out_path, dome=False):
    """Render every frame into a single ffmpeg process."""
    return write_videos(cfg, events, audio, [(out_path, dome)])[0]


# <u> __Running it:__ </u>

def make_sequence(cfg=None):
    """Build the whole sequence, returning the paths written."""
    cfg = cfg or Config()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on your PATH.")

    cfg.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Finding stars over {cfg.latitude:.3f}, {cfg.longitude:.3f} "
          f"at {cfg.date_time} {cfg.time_zone} ...")
    sky = observed_sky(cfg)
    print(f"  {len(sky)} stars brighter than magnitude {cfg.mag_limit} "
          f"above the horizon")

    print(f"Rendering the '{cfg.instrument}' sonification in {cfg.system} ...")
    soni = build_sonification(sky, cfg)

    audio = cfg.outdir / f"stars_appearing_{cfg.instrument}.wav"
    # a caption is *prepended* to the audio, which would slide the whole track
    # against the picture, so the copy we mux into video goes without one
    soni.save(str(audio), embed_caption=False)

    events = timings(soni, sky)

    targets = [(cfg.outdir / "stars_appearing_panorama.mp4", False)]
    if cfg.dome:
        targets.append((cfg.outdir / "stars_appearing_dome.mp4", True))

    print("Rendering the animation ...")
    written = [audio] + write_videos(cfg, events, audio, targets)

    for path in written:
        print(f"  wrote {path}")

    return written


if __name__ == "__main__":
    make_sequence(Config())
