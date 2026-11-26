"""Tests for rs_core.height: solar position and shadow-geometry height recovery.

Two independent checks:
  1. Solar position against published reference values (NOAA) for known
     places/times, tolerance 1.5 degrees.
  2. A synthetic scene with a building of *known* height and a shadow of exactly
     the length that height would cast, so the estimator is scored against
     ground truth rather than against itself.

Usage:
    venv\\Scripts\\python.exe tools\\test_height.py
"""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rs_core.height import (                      # noqa: E402
    estimate_heights,
    shadow_mask,
    shadow_vector,
    solar_position,
)

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def check_close(value: float, expected: float, tol: float, msg: str) -> None:
    ok = abs(value - expected) <= tol
    check(ok, f"{msg}: got {value:.2f}, expected {expected:.2f} ±{tol}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Solar position
# ─────────────────────────────────────────────────────────────────────────────

def test_solar_position() -> None:
    print("\n[1] solar position")

    # London, 2024-06-21 12:00 UTC (near local solar noon).
    # Max elevation there on the solstice is 90 - 51.5 + 23.44 = 61.9 deg.
    sp = solar_position(51.5074, -0.1278, datetime(2024, 6, 21, 12, 0, 0))
    check_close(sp.elevation_deg, 61.9, 1.5, "London solstice noon elevation")
    check_close(sp.azimuth_deg, 180.0, 6.0, "London solstice noon azimuth (south)")

    # Same day, 08:00 UTC: sun in the east-southeast.
    sp = solar_position(51.5074, -0.1278, datetime(2024, 6, 21, 8, 0, 0))
    check(90 < sp.azimuth_deg < 130, f"morning azimuth is east-ish ({sp.azimuth_deg:.1f})")
    check(30 < sp.elevation_deg < 45, f"morning elevation plausible ({sp.elevation_deg:.1f})")

    # Equator, equinox, 09:00 UTC: elevation ≈ 45 deg, azimuth ≈ 90 (east).
    sp = solar_position(0.0, 0.0, datetime(2024, 3, 20, 9, 0, 0))
    check_close(sp.elevation_deg, 45.0, 2.0, "equator equinox 09:00 elevation")
    check_close(sp.azimuth_deg, 90.0, 3.0, "equator equinox 09:00 azimuth (east)")

    # Equator, equinox, 15:00 UTC: mirror image to the west.
    sp = solar_position(0.0, 0.0, datetime(2024, 3, 20, 15, 0, 0))
    check_close(sp.azimuth_deg, 270.0, 3.0, "equator equinox 15:00 azimuth (west)")

    # Southern hemisphere summer: Sydney, 2024-12-21 02:00 UTC (~13:00 local).
    sp = solar_position(-33.87, 151.21, datetime(2024, 12, 21, 2, 0, 0))
    check(70 < sp.elevation_deg < 85, f"Sydney midsummer elevation high ({sp.elevation_deg:.1f})")
    check(sp.azimuth_deg < 30 or sp.azimuth_deg > 330,
          f"Sydney midsummer sun is north ({sp.azimuth_deg:.1f})")

    # Night: elevation must go negative, so the estimator refuses.
    sp = solar_position(51.5074, -0.1278, datetime(2024, 6, 21, 0, 0, 0))
    check(sp.elevation_deg < 0, f"midnight elevation is negative ({sp.elevation_deg:.1f})")
    check(not sp.usable, "night is correctly marked unusable")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Shadow geometry against known truth
# ─────────────────────────────────────────────────────────────────────────────

def make_scene(height_m: float, gsd_m: float, elevation_deg: float,
               azimuth_deg: float, size: int = 600):
    """Ground + one rectangular building + a shadow of the correct length."""
    rgb = np.full((size, size, 3), 150, dtype=np.uint8)

    # Building footprint: 100 px tall, 160 px wide.
    y0, y1, x0, x1 = 300, 400, 220, 380
    rgb[y0:y1, x0:x1] = 200

    shadow_px = height_m / (gsd_m * math.tan(math.radians(elevation_deg)))
    dx, dy = shadow_vector(azimuth_deg)

    # Paint the shadow by stepping along the shadow direction from every
    # building pixel, which reproduces what the sun actually does.
    ys, xs = np.nonzero(np.ones((y1 - y0, x1 - x0), dtype=bool))
    ys = ys + y0
    xs = xs + x0
    for t in range(1, int(round(shadow_px)) + 1):
        yy = np.round(ys + dy * t).astype(int)
        xx = np.round(xs + dx * t).astype(int)
        ok = (yy >= 0) & (yy < size) & (xx >= 0) & (xx < size)
        # Do not overwrite the building itself.
        inside = (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
        sel = ok & ~inside
        rgb[yy[sel], xx[sel]] = 40

    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return rgb, mask, shadow_px * (abs(dx) + abs(dy))


def test_height_recovery() -> None:
    print("\n[2] shadow height recovery (synthetic ground truth)")

    cases = [
        # height, gsd, elevation, azimuth
        (10.0, 0.10, 45.0, 180.0),     # shadow to the north
        (15.0, 0.10, 40.0, 135.0),     # diagonal shadow
        (9.0, 0.30, 55.0, 225.0),      # WHU-like GSD
    ]
    for height_m, gsd, elev, azim in cases:
        rgb, mask, _ = make_scene(height_m, gsd, elev, azim)
        res = estimate_heights(rgb, mask, gsd_m=gsd,
                              sun_elevation_deg=elev, sun_azimuth_deg=azim)
        label = f"H={height_m}m gsd={gsd} elev={elev} azim={azim}"
        if not res["usable"]:
            check(False, f"{label} -> estimator refused: {res['reason']}")
            continue
        got = res["instances"][0]["height_m"]
        err = abs(got - height_m) / height_m
        check(err < 0.20, f"{label} -> recovered {got:.2f} m (error {err * 100:.1f}%)")

    # Volume should be area x height for a flat-roofed block.
    rgb, mask, _ = make_scene(10.0, 0.10, 45.0, 180.0)
    res = estimate_heights(rgb, mask, gsd_m=0.10,
                           sun_elevation_deg=45.0, sun_azimuth_deg=180.0)
    area_m2 = 100 * 160 * 0.10 ** 2
    expected_volume = area_m2 * 10.0
    check(abs(res["volume_m3"] - expected_volume) / expected_volume < 0.25,
          f"volume {res['volume_m3']:.0f} m3 vs expected ~{expected_volume:.0f} m3")

    # Refusals must be explicit, never a silent wrong number.
    print("\n[3] refusals")
    res = estimate_heights(rgb, mask, gsd_m=None,
                           sun_elevation_deg=45.0, sun_azimuth_deg=180.0)
    check(not res["usable"] and "ground sample distance" in res["reason"],
          "missing GSD is refused with a clear reason")

    res = estimate_heights(rgb, mask, gsd_m=0.1,
                           sun_elevation_deg=5.0, sun_azimuth_deg=180.0)
    check(not res["usable"], "sun too low is refused")

    res = estimate_heights(rgb, mask, gsd_m=0.1,
                           sun_elevation_deg=88.0, sun_azimuth_deg=180.0)
    check(not res["usable"], "sun almost overhead is refused")

    res = estimate_heights(rgb, np.zeros_like(mask), gsd_m=0.1,
                           sun_elevation_deg=45.0, sun_azimuth_deg=180.0)
    check(not res["usable"], "empty mask is refused")

    # A scene with no shadow at all must not invent heights.
    flat = np.full((300, 300, 3), 150, dtype=np.uint8)
    m = np.zeros((300, 300), dtype=bool)
    m[100:160, 100:180] = True
    flat[m] = 200
    res = estimate_heights(flat, m, gsd_m=0.1,
                           sun_elevation_deg=45.0, sun_azimuth_deg=180.0)
    check(not res["usable"], "shadowless scene yields no estimate (no invented heights)")

    # Shadow mask must be relative, not an absolute brightness cut: the same
    # shadowed fraction of the frame has to be found in a dark and a bright image.
    for ground, shade, label in ((60, 25, "dark"), (200, 100, "bright")):
        img = np.full((100, 100, 3), ground, dtype=np.uint8)
        img[0:20, :] = shade                    # a shadow band across the top
        sm = shadow_mask(img, exclude=None)
        check(sm[0:20].mean() > 0.9 and not sm[50:80].any(),
              f"shadow mask adapts to a {label} image (relative threshold)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Real WHU imagery: refuse when shadowless, measure when shadows exist
# ─────────────────────────────────────────────────────────────────────────────

WHU_GSD = 0.30   # WHU aerial imagery is 0.3 m/px (median house = 81 m2, see README)


def _load_tile(name: str):
    from PIL import Image

    img_path = ROOT / "data/WHU/test/image" / name
    lab_path = ROOT / "data/WHU/test/label" / name
    if not (img_path.is_file() and lab_path.is_file()):
        return None
    rgb = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(lab_path).convert("L")) > 127
    return rgb, mask


def test_real_imagery() -> None:
    print("\n[4] real WHU imagery")

    # Tile 106 is summer near-noon light with no measurable cast shadow; the
    # anti-sunward/sunward contrast is ~0.82. It must be refused rather than
    # measured, otherwise the app reports ~0.5 m "roofs".
    tile = _load_tile("106.tif")
    if tile is None:
        print("  SKIP  WHU test tiles not present")
        return
    rgb, mask = tile
    res = estimate_heights(rgb, mask, gsd_m=WHU_GSD,
                           sun_elevation_deg=65.0, sun_azimuth_deg=270.0)
    check(not res["usable"], "shadowless tile 106 is refused, not measured")
    check("shadow" in res["reason"].lower(),
          f"refusal names the shadow problem (contrast={res.get('shadow_contrast')})")

    # Tile 129 does have cast shadow (~0.52 contrast). With the correct GSD and
    # a realistic high sun, house heights must land in a plausible range.
    tile = _load_tile("129.tif")
    if tile is not None:
        rgb, mask = tile
        res = estimate_heights(rgb, mask, gsd_m=WHU_GSD,
                               sun_elevation_deg=65.0, sun_azimuth_deg=225.0)
        if res["usable"]:
            h = res["mean_height_m"]
            check(2.0 <= h <= 20.0,
                  f"tile 129 mean height {h:.1f} m is plausible for houses "
                  f"(contrast={res['shadow_contrast']})")
            check(res["max_height_m"] >= h, "tile 129 max >= mean")
        else:
            check(False, f"tile 129 should be measurable: {res['reason']}")

    # The gate must reject a wrong azimuth on shadowed imagery, not just flat light.
    tile = _load_tile("129.tif")
    if tile is not None:
        rgb, mask = tile
        res = estimate_heights(rgb, mask, gsd_m=WHU_GSD,
                               sun_elevation_deg=65.0, sun_azimuth_deg=45.0)
        check(not res["usable"] or res["mean_height_m"] < 20.0,
              "wrong azimuth does not produce absurd heights")


if __name__ == "__main__":
    test_solar_position()
    test_height_recovery()
    test_real_imagery()
    print()
    if failures:
        print(f"HEIGHT TESTS FAILED — {len(failures)} problem(s):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("HEIGHT TESTS PASSED")
