from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["_root"] = Path(path).resolve().parent
    return cfg


def data_bbox(cfg: dict) -> tuple[float, float, float, float]:
    m = cfg["data_margin_deg"]
    lats = [cfg["start"]["lat"], cfg["end"]["lat"]]
    lons = [cfg["start"]["lon"], cfg["end"]["lon"]]
    return (min(lons) - m, min(lats) - m, max(lons) + m, max(lats) + m)
