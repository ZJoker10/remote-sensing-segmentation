"""Building height from shadow length — a demo-grade estimate, not a survey.

A single nadir RGB image has no metric depth. Given the sun's position, the
height of a building above the ground can be estimated from the shadow it casts:

    H = shadow_length_px * GSD * tan(sun_elevation)

This module implements that, plus the solar-position maths needed to get the sun
elevation from a timestamp and coordinates (NOAA algorithm, no external deps).

ASSETS AND LIMITS — read before trusting a number
  * Assumes flat ground and a shadow that is not cast onto another building.
  * Assumes the image is north-up. Rotated/pitched imagery invalidates the result.
  * Assumes the sun is high enough for the shadow to be measurable (> ~10 deg);
    below that, shadow length explodes and self-shadowing dominates.
  * Assumes the GSD is known and uniform.
  * Reports the *roof-edge* height; sloped or tiered roofs yield one number.
  * Occlusion biases estimates low, so the per-building value uses a high
    percentile of the measured rays and the spread is reported, not hidden.

For real numbers you need LiDAR nDSM labels; this is the fallback for imagery
that has none, and the UI labels it as an estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from scipy import ndimage

MIN_ELEVATION_DEG = 10.0     # below this, shadow geometry is unusable
MAX_ELEVATION_DEG = 75.0     # near-zenith sun: shadows vanish and tan() explodes
MAX_SHADOW_PX = 400          # beyond this, the "shadow" is probably terrain
MIN_RAYS = 15                # fewer usable rays than this -> no estimate
MAX_PLAUSIBLE_HEIGHT_M = 200.0   # taller than this is a measurement artefact
RAY_MAX_GAP = 2              # tolerate this many bright pixels inside a shadow run
BAND_PX = 15                 # width of the sunlit / shadowed bands used for the gate

# Shadowless imagery must be refused, not measured. Real cast shadow makes the
# band behind a building clearly darker than the mirrored band in front of it.
# Tiles with no shadow sit near 1.0; genuine shadows measure 0.5-0.7 (see
# tools/test_height.py, which checks both cases against real WHU tiles).
MIN_SHADOW_CONTRAST = 0.78


@dataclass
class SunPosition:
    elevation_deg: float
    azimuth_deg: float        # clockwise from north

    @property
    def usable(self) -> bool:
        return self.elevation_deg >= MIN_ELEVATION_DEG


# ─────────────────────────────────────────────────────────────────────────────
# Solar position (NOAA solar calculator, Gregorian dates, UTC input)
# ─────────────────────────────────────────────────────────────────────────────

def solar_position(latitude_deg: float, longitude_deg: float,
                   when_utc: datetime) -> SunPosition:
    """Sun elevation and azimuth for a UTC instant and a WGS84 location."""
    when_utc = when_utc.replace(tzinfo=timezone.utc) if when_utc.tzinfo is None else when_utc

    doy = when_utc.timetuple().tm_yday
    hour = (when_utc.hour + when_utc.minute / 60.0
            + when_utc.second / 3600.0)

    gamma = 2.0 * math.pi / 365.0 * (doy - 1 + (hour - 12.0) / 24.0)

    # Equation of time (minutes) and solar declination (radians)
    eqtime = 229.18 * (0.000075
                       + 0.001868 * math.cos(gamma)
                       - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma)
                       - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918
            - 0.399912 * math.cos(gamma)
            + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma)
            + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma)
            + 0.001480 * math.sin(3 * gamma))

    # True solar time (minutes), then hour angle (degrees)
    time_offset = eqtime + 4.0 * longitude_deg      # tz = 0 because input is UTC
    tst = hour * 60.0 + time_offset
    ha = math.radians(tst / 4.0 - 180.0)

    lat = math.radians(latitude_deg)
    cos_zenith = (math.sin(lat) * math.sin(decl)
                  + math.cos(lat) * math.cos(decl) * math.cos(ha))
    cos_zenith = min(1.0, max(-1.0, cos_zenith))
    zenith = math.acos(cos_zenith)
    elevation = 90.0 - math.degrees(zenith)

    # Azimuth: clockwise from north.
    #   cos(A) = (sin δ − sin φ cos z) / (cos φ sin z), A in [0, 180]
    #   mirror around south for afternoon hour angles (H > 0)
    cos_az = ((math.sin(decl) - math.sin(lat) * cos_zenith)
              / (math.cos(lat) * math.sin(zenith) + 1e-12))
    cos_az = min(1.0, max(-1.0, cos_az))
    azimuth = math.degrees(math.acos(cos_az))
    if ha > 0:                       # afternoon: sun is in the west
        azimuth = 360.0 - azimuth

    return SunPosition(elevation_deg=elevation, azimuth_deg=azimuth % 360.0)


def shadow_vector(azimuth_deg: float) -> tuple[float, float]:
    """(dx, dy) unit vector of the shadow, in image coordinates (y down, north up)."""
    a = math.radians((azimuth_deg + 180.0) % 360.0)
    return math.sin(a), -math.cos(a)


# ─────────────────────────────────────────────────────────────────────────────
# Shadow mask
# ─────────────────────────────────────────────────────────────────────────────

def luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def shadow_mask(rgb: np.ndarray, exclude: np.ndarray | None = None,
                darkness: float = 0.55) -> np.ndarray:
    """Dark pixels, relative to the *sunlit* ground brightness of this image.

    The reference is the 90th percentile of the ground luminance, not the median.
    Using the median silently misses real shadows whenever a large part of the
    frame is already shaded: on WHU test tile 129 the sunlit ground reads ~200
    and cast shadow ~91, but the whole-image median is ~110, so a median-based
    threshold of 0.65 x 110 = 71 flagged nothing. Referencing the bright end
    keeps the threshold tied to actual sunlight.

    `darkness` is the fraction of sunlit brightness below which a pixel counts as
    shadowed. Purely relative, so it survives exposure differences.
    """
    lum = luminance(rgb.astype(np.float32))
    if exclude is not None and (~exclude).sum() > 100:
        sample = lum[~exclude]
    else:
        sample = lum
    sunlit = float(np.percentile(sample, 90))
    if sunlit <= 0:
        return np.zeros(lum.shape, dtype=bool)
    return lum < darkness * sunlit


# ─────────────────────────────────────────────────────────────────────────────
# Per-building height
# ─────────────────────────────────────────────────────────────────────────────

def _instance_boundaries(labels: np.ndarray, n_labels: int) -> dict[int, np.ndarray]:
    """Boundary pixels of every instance (mask minus its erosion)."""
    out = {}
    for idx in range(1, n_labels + 1):
        comp = labels == idx
        if not comp.any():
            continue
        eroded = ndimage.binary_erosion(comp, structure=np.ones((3, 3)))
        ys, xs = np.nonzero(comp & ~eroded)
        if len(ys):
            out[idx] = np.stack([ys, xs], axis=1)
    return out


def _ray_length(ys: np.ndarray, xs: np.ndarray, shadow: np.ndarray,
                buildings: np.ndarray, dx: float, dy: float,
                step: float = 1.0, max_gap: int = RAY_MAX_GAP) -> float:
    """Walk along (dx, dy) while pixels are shadowed; stop at buildings/image edge.

    A single bright pixel inside an otherwise continuous shadow (roof edge,
    texture, a parked car) would truncate the run far too early, so up to
    `max_gap` consecutive non-shadow pixels are tolerated before giving up.
    """
    h, w = shadow.shape
    length = 0.0
    gap = 0
    for t in range(2, MAX_SHADOW_PX):
        y = int(round(ys + dy * t))
        x = int(round(xs + dx * t))
        if y < 0 or y >= h or x < 0 or x >= w:
            break
        if buildings[y, x]:
            break
        if shadow[y, x]:
            gap = 0
            length = t * step
        else:
            gap += 1
            if gap > max_gap:
                break
    return length


def shadow_contrast(lum: np.ndarray, buildings: np.ndarray,
                    dx: float, dy: float, band: int = BAND_PX) -> float | None:
    """How much darker the anti-sunward band is than the mirrored sunward one.

    Returns `median(anti) / median(sunward)` measured on ground pixels only, or
    None when there is not enough ground to compare. Values near 1.0 mean the
    image has no measurable cast shadow, whatever the sun angle claims.
    """
    h, w = lum.shape
    behind = np.zeros_like(buildings)
    ahead = np.zeros_like(buildings)

    for t in range(2, band):
        y_shift, x_shift = dy * t, dx * t
        if abs(y_shift) >= h or abs(x_shift) >= w:
            continue
        behind |= ndimage.shift(buildings, (y_shift, x_shift), order=0,
                                mode="constant", cval=0) > 0
        ahead |= ndimage.shift(buildings, (-y_shift, -x_shift), order=0,
                               mode="constant", cval=0) > 0

    behind &= ~buildings
    ahead &= ~buildings
    if behind.sum() < 400 or ahead.sum() < 400:
        return None

    anti = float(np.median(lum[behind]))
    sunward = float(np.median(lum[ahead]))
    if sunward <= 1e-6:
        return None
    return anti / sunward


def estimate_heights(
    rgb: np.ndarray,
    building_mask: np.ndarray,
    gsd_m: float,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
    max_rays_per_building: int = 120,
    percentile: float = 80.0,
) -> dict:
    """Estimate each building's height above ground from its shadow.

    Returns a dict with per-instance heights (metres), a volume estimate and the
    assumptions used, so the caller can label the numbers honestly.
    """
    if gsd_m is None or gsd_m <= 0:
        return {"usable": False, "reason": "ground sample distance (m/pixel) is required",
                "instances": [], "volume_m3": None}

    if sun_elevation_deg < MIN_ELEVATION_DEG:
        return {"usable": False,
                "reason": f"sun elevation {sun_elevation_deg:.1f}° is below the "
                          f"{MIN_ELEVATION_DEG:.0f}° minimum; shadows are too long "
                          f"and self-shadowing dominates",
                "instances": [], "volume_m3": None}

    if sun_elevation_deg > MAX_ELEVATION_DEG:
        return {"usable": False,
                "reason": f"sun elevation {sun_elevation_deg:.1f}° is above the "
                          f"{MAX_ELEVATION_DEG:.0f}° maximum; shadows are too short "
                          f"(1 px of error becomes several pixels of height)",
                "instances": [], "volume_m3": None}

    buildings = building_mask.astype(bool)
    if not buildings.any():
        return {"usable": False, "reason": "no buildings detected",
                "instances": [], "volume_m3": None}

    labels, n = ndimage.label(buildings, structure=np.ones((3, 3)))
    lum = luminance(rgb.astype(np.float32))
    shadow = shadow_mask(rgb, exclude=buildings)
    dx, dy = shadow_vector(sun_azimuth_deg)
    boundaries = _instance_boundaries(labels, n)

    # Evidence gate: a high sun, overcast light or the wrong azimuth leave no
    # measurable cast shadow, and measuring one anyway produces confident
    # nonsense (WHU tile 106: no shadow, yet a naive run reported 0.5 m roofs).
    contrast = shadow_contrast(lum, buildings, dx, dy)
    if contrast is None:
        return {"usable": False,
                "reason": "not enough open ground around the buildings to check "
                          "for shadow; supply a wider crop or check the mask",
                "instances": [], "volume_m3": None, "shadow_contrast": None}
    if contrast >= MIN_SHADOW_CONTRAST:
        return {"usable": False,
                "reason": f"no measurable cast shadow at this sun azimuth "
                          f"(anti-sunward ground is {contrast:.2f}x the sunward "
                          f"brightness; {MIN_SHADOW_CONTRAST:.2f} or above means "
                          f"shadowless imagery). Check the azimuth, or use "
                          f"imagery with visible shadows.",
                "instances": [], "volume_m3": None,
                "shadow_contrast": round(contrast, 3)}

    tan_elev = math.tan(math.radians(sun_elevation_deg))
    px_area = gsd_m ** 2

    instances = []
    volume = 0.0
    heights = []
    for idx, pts in boundaries.items():
        if len(pts) < 8:
            continue
        if len(pts) > max_rays_per_building:
            sel = np.linspace(0, len(pts) - 1, max_rays_per_building).astype(int)
            pts = pts[sel]

        lengths = np.array([_ray_length(y, x, shadow, buildings, dx, dy)
                            for y, x in pts])
        valid = lengths[lengths > 0]
        if len(valid) < MIN_RAYS:
            continue

        shadow_px = float(np.percentile(valid, percentile))
        height_m = shadow_px * gsd_m * tan_elev
        if not math.isfinite(height_m) or height_m > MAX_PLAUSIBLE_HEIGHT_M:
            continue

        area_px = int((labels == idx).sum())
        area_m2 = area_px * px_area
        volume += area_m2 * height_m

        instances.append({
            "id": int(idx),
            "area_m2": round(area_m2, 1),
            "height_m": round(height_m, 2),
            "shadow_len_px": round(shadow_px, 1),
            "rays_used": int(len(valid)),
            "height_spread_m": round(
                float(np.percentile(valid, 95) - np.percentile(valid, 5))
                * gsd_m * tan_elev, 2),
        })
        heights.append(height_m)

    if not instances:
        return {"usable": False,
                "reason": "shadow detected but no building had enough measurable "
                          "shadows; try a higher sun angle or a different azimuth",
                "instances": [], "volume_m3": None,
                "shadow_contrast": round(contrast, 3)}

    return {
        "usable": True,
        "method": "shadow_length x GSD x tan(sun_elevation)",
        "assumptions": [
            "flat ground",
            "north-up, nadir imagery",
            "shadow not cast onto another building",
            f"height taken at the {percentile:.0f}th percentile of rays "
            "(occlusion biases raw rays low)",
        ],
        "sun_elevation_deg": round(sun_elevation_deg, 2),
        "sun_azimuth_deg": round(sun_azimuth_deg, 2),
        "gsd_m": gsd_m,
        "shadow_contrast": round(contrast, 3),
        "instances": instances,
        "mean_height_m": round(float(np.mean(heights)), 2),
        "max_height_m": round(float(np.max(heights)), 2),
        "volume_m3": round(volume, 0),
        "accuracy": "rough estimate; expect large error, not survey grade",
        "height_available": True,
    }
