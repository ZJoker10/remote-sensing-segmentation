"""rs_core — shared inference library for the research scripts and the app.

No UI code, no global state. Everything the app does goes through this package
so `train.py` / `predict.py` and the server can never drift apart.
"""

__all__ = ["model", "inference", "postprocess", "visualize"]
