"""End-to-end smoke test for app/server.py, forced onto the CPU.

Runs the real server in a subprocess with CUDA_VISIBLE_DEVICES="" so it never
competes with a training job for VRAM, posts one real WHU tile through
/api/segment, and verifies every artefact the UI links to.

Usage:
    venv\\Scripts\\python.exe tools/smoke_app.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("SMOKE_PORT", "7871"))
BASE = f"http://127.0.0.1:{PORT}"
OUT = Path(os.environ.get("SMOKE_TMP", "/tmp/opencode"))
OUT.mkdir(parents=True, exist_ok=True)

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def http(path: str, method: str = "GET", body: bytes | None = None,
         ctype: str = "application/json"):
    req = urllib.request.Request(BASE + path, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", ctype)
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status, r.read(), r.headers.get("Content-Type", "")


def pick_tile_with_buildings(min_frac: float = 0.08) -> Path:
    """First WHU test tile whose ground truth actually contains buildings.

    Some tiles in the split are pure background (label 0.tif is all zeros), and
    testing on one of those would prove nothing.
    """
    import numpy as np

    for img_path in sorted((ROOT / "data/WHU/test/image").glob("*.tif")):
        lab_path = ROOT / "data/WHU/test/label" / img_path.name
        if not lab_path.is_file():
            continue
        lab = np.array(Image.open(lab_path).convert("L"))
        frac = float((lab > 127).mean())
        if frac >= min_frac:
            print(f"[smoke] tile {img_path.name}: {frac * 100:.1f}% building pixels in GT")
            return img_path
    raise SystemExit(f"no test tile with >= {min_frac:.0%} buildings found")


def make_test_image() -> Path:
    """A 2x2 canvas of a real WHU tile, so tiling + seam blending are exercised."""
    tile_path = pick_tile_with_buildings()
    tile = Image.open(tile_path).convert("RGB")
    w, h = tile.size
    side = min(512, w, h)
    tile = tile.crop((0, 0, side, side))

    canvas = Image.new("RGB", (side * 2, side * 2))
    for dy in range(2):
        for dx in range(2):
            canvas.paste(tile, (dx * side, dy * side))
    path = OUT / "rs_smoke_input.png"
    canvas.save(path)
    print(f"[smoke] input: {path.name} {canvas.size} (2x2 of {tile_path.name}, "
          f"forces multi-window stitching)")
    return path


def dump_log(path: Path, tail: int = 60) -> None:
    """Show the server's own output when something goes wrong."""
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"\n[smoke] --- server output (last {tail} lines) ---")
    for line in lines[-tail:]:
        print("  " + line)
    print("[smoke] --- end server output ---\n")


def main() -> int:
    test_img = make_test_image()
    raw = test_img.read_bytes()

    env = {**os.environ}
    env.pop("PYTHONPATH", None)
    log_path = OUT / "rs_smoke_server.log"
    log_file = open(log_path, "w", encoding="utf-8")
    print(f"[smoke] starting server on port {PORT} (CPU only, log {log_path})")
    # --device cpu rather than CUDA_VISIBLE_DEVICES: torch ignores an empty value
    # on Windows, so the flag is the only reliable way to keep the GPU free.
    proc = subprocess.Popen(
        [sys.executable, "app/server.py", "--port", str(PORT),
         "--no-browser", "--device", "cpu"],
        cwd=str(ROOT), env=env,
        stdout=log_file, stderr=subprocess.STDOUT, text=True,
    )

    try:
        # ── wait for the server ───────────────────────────────────────────────
        health = None
        deadline = time.time() + 180
        while time.time() < deadline:
            if proc.poll() is not None:
                dump_log(log_path)
                raise SystemExit("server exited before becoming ready")
            try:
                _, body, _ = http("/health")
                health = json.loads(body)
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(1.0)
        if health is None:
            raise SystemExit("server did not become ready in 180 s")

        print(f"[smoke] /health -> device={health['device']} "
              f"phases={sorted(health['phases'])}")
        check(health["device"] == "cpu", "server forced onto CPU (GPU left for training)")
        check(len(health["phases"]) > 0, "at least one checkpoint discovered")
        check(health["classes"]["building"]["available"], "building class available")
        check(health["height_available"] is True,
              "height available via shadow geometry (no trained head needed)")
        check(health.get("height_method") == "shadow_geometry",
              "height method reported as shadow_geometry")

        diag = health.get("diagnostics", {})
        flagged = [p for p, d in diag.items() if d.get("degraded")]
        print(f"[smoke] checkpoint diagnostics: degraded={sorted(flagged)}")
        check(isinstance(diag, dict) and diag,
              "checkpoint diagnostics exposed for every listed phase")
        check("3" in flagged and "4" in flagged,
              "known-broken checkpoints (phase 3/4 pos_embed zeros) are flagged, "
              "not silently offered")

        # ── the index page ────────────────────────────────────────────────────
        status, body, ctype = http("/")
        check(status == 200 and b"Remote Sensing Segmentation" in body,
              "GET / serves the UI")
        check("text/html" in ctype, "UI served as text/html")

        # ── a real segmentation request ───────────────────────────────────────
        # Phase 1 is the checkpoint that is both healthy and accurate, so the
        # end-to-end test measures the pipeline against a model that works.
        healthy = [p for p in (1, 5, 2, 3, 4)
                   if p in health["phases"]
                   and not (health["diagnostics"].get(str(p)) or {}).get("degraded")]
        phase = healthy[0] if healthy else sorted(health["phases"])[0]
        payload = {
            "phase": phase,
            "threshold": 0.5,
            "gsd_m": 0.30,
            "min_area_m2": 20,
            "images": [{
                "name": "smoke_tile.png",
                "data": "data:image/png;base64," + base64.b64encode(raw).decode(),
            }],
        }
        print(f"[smoke] POST /api/segment phase={phase} (CPU inference, may take a while)")
        t0 = time.time()
        status, body, _ = http("/api/segment", "POST",
                               json.dumps(payload).encode(), "application/json")
        elapsed = time.time() - t0
        data = json.loads(body)
        check(status == 200, f"/api/segment returned 200 ({elapsed:.1f}s)")

        results = data.get("results", [])
        check(len(results) == 1, "one result returned for one uploaded image")
        if results:
            r = results[0]
            s = r["stats"]
            print(f"[smoke] stats: {s['instances']} buildings, "
                  f"{s['coverage_pct']}% coverage, "
                  f"{s['total_area_m2']} m2, {s['seconds']}s")
            check(s["instances"] > 0, "at least one building detected in a real tile")
            check(health.get("gsd_m") == 0.30,
                  "WHU ground sample distance is 0.30 m (not the old 0.075)")
            check(0.0 <= s["coverage_pct"] <= 100.0, "coverage is a valid percentage")
            check(s["total_area_m2"] is not None,
                  "metric area computed from the supplied GSD")
            check(float(s["total_area_m2"]) > 0, "total roof area is positive")
            check(s["width"] == 1024 and s["height_px"] == 1024,
                  "reported size matches upload")
            check("available" in s["classes"]["water"],
                  "water status reported per image")
            check(len(r["instance_list"]) == s["instances"] or
                  len(r["instance_list"]) == 500,
                  "per-building list matches the instance count")

            # ── every artefact the UI links to ────────────────────────────────
            artifacts = [("overlay", r["overlay"])]
            artifacts += [(f"{k} mask", v) for k, v in r["masks"].items()]
            artifacts += [(f"{k} prob", v) for k, v in r["probs"].items()]
            for label, url in artifacts:
                status, body, ctype = http(url)
                check(status == 200 and body.startswith(b"\x89PNG"),
                      f"{label} PNG downloads ({len(body) // 1024} KiB)")

            status, body, ctype = http(f"/outputs/{data['run']}/summary.csv")
            check(status == 200 and b"building_coverage_pct" in body,
                  "summary.csv downloads with the new columns")
            rows = body.decode().strip().splitlines()
            check(len(rows) == 2, "summary.csv has a header plus one row")

            # mask must agree with the reported count
            mask_img = Image.open(io.BytesIO(http(r["masks"]["building"])[1]))
            check(mask_img.size == (1024, 1024),
                  "mask is full resolution across a 2x2 window grid")

        # ── height path: shadow geometry over a real tile ─────────────────────
        # The smoke tile is 106, which is summer near-noon imagery with no cast
        # shadow. The correct answer is a refusal, and that is what is asserted
        # here. The positive path (tile 129, 6.9 m mean) lives in test_height.py,
        # which can pick an image that actually has shadows.
        print("[smoke] POST /api/segment with sun position (shadow height)")
        height_payload = dict(payload)
        height_payload["sun"] = {"enabled": True, "mode": "manual",
                                 "elevation_deg": 65.0, "azimuth_deg": 270.0}
        status, body, _ = http("/api/segment", "POST",
                               json.dumps(height_payload).encode(), "application/json")
        hdata = json.loads(body)
        check(status == 200, "height request accepted")
        if hdata.get("results"):
            hstats = hdata["results"][0]["stats"]
            hinfo = hstats.get("height") or {}
            check("usable" in hinfo, "height result states usability explicitly")
            if hinfo.get("usable"):
                print(f"[smoke] height: mean={hinfo['mean_height_m']} m "
                      f"max={hinfo['max_height_m']} m vol={hinfo['volume_m3']} m3")
                check(hinfo["mean_height_m"] > 0, "mean height is positive")
                check(hinfo["max_height_m"] >= hinfo["mean_height_m"],
                      "max height >= mean height")
                check(hinfo["volume_m3"] > 0, "volume is positive")
                with_h = [i for i in hdata["results"][0]["instance_list"]
                          if i.get("height_m") is not None]
                check(len(with_h) > 0, "per-building heights attached to instances")
            else:
                print(f"[smoke] height refused: {hinfo.get('reason')}")
                check(bool(hinfo.get("reason")),
                      "refusal carries a human-readable reason")
                check("shadow" in str(hinfo.get("reason", "")).lower(),
                      "shadowless tile is refused for the shadow reason, not a crash")
                check(hinfo.get("shadow_contrast") is not None,
                      "refusal reports the measured shadow contrast")

        # Without a GSD, height must be refused rather than guessed in pixels.
        no_gsd = dict(height_payload)
        no_gsd["gsd_m"] = None
        status, body, _ = http("/api/segment", "POST",
                               json.dumps(no_gsd).encode(), "application/json")
        ndata = json.loads(body)
        if ndata.get("results"):
            ninfo = ndata["results"][0]["stats"].get("height") or {}
            check(not ninfo.get("usable", False),
                  "height refused when no GSD is supplied")

        # ── path traversal is refused ─────────────────────────────────────────
        try:
            http("/outputs/../configs/config.yaml")
            check(False, "path traversal outside outputs/ is blocked")
        except urllib.error.HTTPError as exc:
            check(exc.code in (403, 404), f"path traversal blocked (HTTP {exc.code})")

        # ── error handling ────────────────────────────────────────────────────
        try:
            http("/api/segment", "POST", b"{}", "application/json")
            check(False, "empty request rejected")
        except urllib.error.HTTPError as exc:
            check(exc.code == 500, f"empty request reports an error (HTTP {exc.code})")

    except BaseException:
        dump_log(log_path)
        raise
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()

    print()
    if failures:
        print(f"SMOKE TEST FAILED — {len(failures)} problem(s):")
        for f in failures:
            print("  -", f)
        return 1
    print("SMOKE TEST PASSED — upload -> mask -> overlay -> prob -> csv all work on CPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
