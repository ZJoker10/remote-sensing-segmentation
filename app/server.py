"""Local upload-and-segment web app (standard library only, no extra deps).

Run:
    venv\\Scripts\\python.exe app\\server.py --port 7860
or double-click start_app.bat.

Design notes:
  * Uploads arrive as base64 JSON, so no multipart parser and no `cgi` module
    (removed in Python 3.13).
  * One model instance is cached and a lock serialises inference, which keeps
    VRAM use predictable on a laptop GPU.
  * Every run writes to outputs/<timestamp>/ so results survive a refresh and
    can be zipped and handed in.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rs_core.inference import CLASS_ORDER, predict_probs    # noqa: E402
from rs_core.model import (                                # noqa: E402
    CLASSES,
    HEIGHT_AVAILABLE,
    HEIGHT_METHOD,
    HEIGHT_NOTE,
    available_phases,
    checkpoint_diagnostics,
    class_status,
    load_config,
    load_model,
)
from rs_core.postprocess import analyse              # noqa: E402
from rs_core.visualize import mask_png, overlay, prob_png  # noqa: E402

MAX_BODY_BYTES = 160 * 1024 * 1024
APP_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"

CONFIG = load_config()
MODEL_LOCK = threading.Lock()
MODEL_CACHE: dict[int, torch.nn.Module] = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(phase: int) -> torch.nn.Module:
    model = MODEL_CACHE.get(phase)
    if model is None:
        print(f"[app] loading phase {phase} model on {DEVICE} ...")
        t0 = time.time()
        model = load_model(phase, CONFIG, DEVICE)
        model.eval()
        MODEL_CACHE[phase] = model
        print(f"[app] phase {phase} ready in {time.time() - t0:.1f}s")
    return model


def decode_upload(data_url: str) -> np.ndarray:
    if "," in data_url and data_url.startswith("data:"):
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(raw))
    return np.array(img.convert("RGB"), dtype=np.uint8)


def _resolve_sun(payload: dict) -> tuple[float, float, str] | None:
    """Sun elevation/azimuth for shadow height, from manual values or a timestamp."""
    sun = payload.get("sun") or {}
    if not sun.get("enabled"):
        return None

    if sun.get("mode", "manual") == "computed":
        from datetime import datetime

        from rs_core.height import solar_position

        try:
            when = datetime.fromisoformat(str(sun["when_utc"]).replace("Z", "+00:00"))
            lat = float(sun["lat"])
            lon = float(sun["lon"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid sun position inputs: {exc}") from exc

        pos = solar_position(lat, lon, when)
        if not pos.usable:
            raise ValueError(
                f"sun elevation is {pos.elevation_deg:.1f} deg at that time and place; "
                "shadows are not measurable"
            )
        return pos.elevation_deg, pos.azimuth_deg, "computed"

    try:
        return (float(sun["elevation_deg"]), float(sun["azimuth_deg"]), "manual")
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid sun elevation/azimuth: {exc}") from exc


def run_batch(payload: dict) -> dict:
    phase = int(payload.get("phase", 1))
    threshold = float(payload.get("threshold", 0.5))
    min_area = float(payload.get("min_area_m2", 20.0))
    gsd = payload.get("gsd_m")
    gsd = float(gsd) if gsd not in (None, "", 0, "0") else None
    images = payload.get("images") or []
    if not images:
        raise ValueError("no images in request")

    sun = _resolve_sun(payload)

    run_dir = OUTPUT_ROOT / time.strftime("%Y%m%d-%H%M%S")
    (run_dir / "masks").mkdir(parents=True, exist_ok=True)
    (run_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (run_dir / "prob").mkdir(parents=True, exist_ok=True)

    results = []
    with MODEL_LOCK:
        model = get_model(phase)
        for item in images:
            name = Path(item.get("name", "image.png")).name
            t0 = time.time()
            rgb = decode_upload(item["data"])
            h, w = rgb.shape[:2]
            probs = predict_probs(model, rgb, DEVICE)
            n_classes = probs.shape[0]

            # Channel order is fixed by rs_core.inference.CLASS_ORDER.
            analysed = {"building": analyse(probs[0], threshold=threshold,
                                            gsd_m=gsd, min_area_m2=min_area)}
            water_stats = None
            if n_classes > 1:
                analysed["water"] = analyse(probs[1], threshold=threshold,
                                            gsd_m=gsd, min_area_m2=min_area)
                water_stats = analysed["water"]["stats"]

            masks = {cls: res["mask"] for cls, res in analysed.items()}
            stats = dict(analysed["building"]["stats"])

            stem = Path(name).stem
            Image.fromarray(overlay(rgb, masks, CLASSES)).save(
                run_dir / "overlays" / f"{stem}_overlay.png")
            for cls, res in analysed.items():
                mask_png(res["mask"]).save(run_dir / "masks" / f"{stem}_{cls}_mask.png")
                prob_png(probs[CLASS_ORDER.index(cls)]
                         if cls in CLASS_ORDER and CLASS_ORDER.index(cls) < n_classes
                         else probs[0]).save(run_dir / "prob" / f"{stem}_{cls}_prob.png")

            # ── height (shadow geometry, estimate only) ──────────────────────
            height = None
            if sun is not None:
                from rs_core.height import estimate_heights

                elev, azim, source = sun
                try:
                    height = estimate_heights(rgb, analysed["building"]["mask"],
                                              gsd_m=gsd, sun_elevation_deg=elev,
                                              sun_azimuth_deg=azim)
                    height["sun_source"] = source
                except Exception as exc:  # noqa: BLE001 - never fail a whole batch
                    height = {"usable": False, "reason": f"{type(exc).__name__}: {exc}"}
                if height.get("usable"):
                    by_id = {i["id"]: i["height_m"] for i in height["instances"]}
                    for inst in analysed["building"]["instances"]:
                        inst["height_m"] = by_id.get(inst["id"])
            elif gsd is None:
                height = {"usable": False,
                          "reason": "height needs a ground sample distance"}

            stats["width"] = w
            stats["height_px"] = h
            stats["seconds"] = round(time.time() - t0, 2)
            stats["classes"] = {
                "building": {
                    "coverage_pct": stats["coverage_pct"],
                    "instances": stats["instances"],
                    "area_m2": stats["total_area_m2"],
                    "available": True,
                },
                "water": ({
                    "coverage_pct": water_stats["coverage_pct"],
                    "instances": water_stats["instances"],
                    "area_m2": water_stats["total_area_m2"],
                    "available": True,
                } if water_stats else {"available": False,
                                       "reason": "this checkpoint has no water head"}),
            }
            stats["height"] = height if height is not None else {
                "usable": False, "reason": "height not requested"}
            stats["height_available"] = bool(height and height.get("usable"))

            results.append({
                "name": name,
                "stats": stats,
                "instance_list": analysed["building"]["instances"][:500],
                "water_instance_list": (analysed["water"]["instances"][:200]
                                        if water_stats else []),
                "overlay": f"/outputs/{run_dir.name}/overlays/{stem}_overlay.png",
                "masks": {cls: f"/outputs/{run_dir.name}/masks/{stem}_{cls}_mask.png"
                          for cls in analysed},
                "probs": {cls: f"/outputs/{run_dir.name}/prob/{stem}_{cls}_prob.png"
                          for cls in analysed},
            })
            water_txt = (f", water {water_stats['coverage_pct']}%"
                         if water_stats else "")
            height_txt = (f", mean height {height['mean_height_m']} m"
                          if height and height.get("usable") else "")
            print(f"[app] {name} {w}x{h} -> {stats['instances']} buildings, "
                  f"{stats['coverage_pct']}%{water_txt}{height_txt} "
                  f"({stats['seconds']}s)")

    with open(run_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "width", "height", "building_instances",
                         "building_coverage_pct", "building_area_m2",
                         "water_coverage_pct", "water_area_m2",
                         "mean_height_m", "max_height_m", "volume_m3",
                         "mean_confidence", "seconds"])
        for r in results:
            s = r["stats"]
            water = s["classes"]["water"]
            hgt = s["height"] if s.get("height", {}).get("usable") else {}
            writer.writerow([
                r["name"], s["width"], s["height_px"], s["instances"],
                s["coverage_pct"], s["total_area_m2"],
                water.get("coverage_pct"), water.get("area_m2"),
                hgt.get("mean_height_m"), hgt.get("max_height_m"),
                hgt.get("volume_m3"), s["mean_confidence"], s["seconds"],
            ])

    return {"run": run_dir.name, "results": results,
            "device": str(DEVICE), "phase": phase}


class Handler(BaseHTTPRequestHandler):
    server_version = "RemoteSensingApp/1.0"

    def log_message(self, fmt, *args):  # quieter console
        if "/api/segment" not in self.path:
            return
        super().log_message(fmt, *args)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _send(self, status: int, body: bytes, ctype: str):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: dict):
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

    def _serve_file(self, path: Path, ctype: str | None = None):
        if not path.is_file():
            self._json(404, {"error": "not found"})
            return
        ctype = ctype or {
            ".html": "text/html; charset=utf-8",
            ".png": "image/png",
            ".csv": "text/csv",
            ".js": "application/javascript",
            ".css": "text/css",
        }.get(path.suffix.lower(), "application/octet-stream")
        self._send(200, path.read_bytes(), ctype)

    # ── routes ───────────────────────────────────────────────────────────────
    def do_GET(self):
        route = self.path.split("?")[0]
        if route in ("/", "/index.html"):
            self._serve_file(APP_DIR / "index.html")
        elif route == "/health":
            phases = available_phases(CONFIG)
            self._json(200, {
                "device": str(DEVICE),
                "phases": phases,
                "diagnostics": {str(p): checkpoint_diagnostics(p, CONFIG) for p in phases},
                "classes": class_status(CONFIG),
                "height_available": HEIGHT_AVAILABLE,
                "height_method": HEIGHT_METHOD,
                "height_note": HEIGHT_NOTE,
                "gsd_m": CONFIG.get("data", {}).get("gsd_m"),
            })
        elif route.startswith("/outputs/"):
            target = (ROOT / route.lstrip("/")).resolve()
            if not str(target).startswith(str(OUTPUT_ROOT.resolve())):
                self._json(403, {"error": "forbidden"})
                return
            self._serve_file(target)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0] != "/api/segment":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_BODY_BYTES:
                self._json(413, {"error": f"bad payload size: {length}"})
                return
            payload = json.loads(self.rfile.read(length))
            self._json(200, run_batch(payload))
        except Exception as exc:  # noqa: BLE001 - surface the error in the UI
            traceback.print_exc()
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main():
    ap = argparse.ArgumentParser(description="Local segmentation web app")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--warm", action="store_true",
                    help="Load the default model before serving")
    ap.add_argument("--no-browser", action="store_true",
                    help="Do not open the browser (used by tests and servers)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="auto = CUDA when available. cpu frees the GPU for training.")
    args = ap.parse_args()

    global DEVICE
    if args.device == "cpu":
        DEVICE = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is unavailable")
        DEVICE = torch.device("cuda")

    print(f"[app] device={DEVICE} | checkpoint dir={ROOT / 'checkpoints'}")
    print(f"[app] phases available: {sorted(available_phases(CONFIG))}")
    if args.warm:
        try:
            get_model(1)
        except Exception as exc:  # noqa: BLE001
            print(f"[app] warm-up skipped: {exc}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[app] serving {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.0), webbrowser.open(url)),
                         daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[app] stopped")


if __name__ == "__main__":
    main()
