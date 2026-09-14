"""Téléchargement des données brutes (Overture Maps + tuiles d'altitude) pour l'emprise configurée."""
from __future__ import annotations

import sys
from pathlib import Path

from .config import data_bbox, load_config
from .overture import extract

SEGMENT_COLS = ["id", "names", "subtype", "class", "subclass", "connectors", "road_surface",
                "road_flags", "level_rules", "access_restrictions", "geometry", "bbox"]
BUILDING_COLS = ["id", "geometry", "bbox", "subtype", "class", "height"]
LANDUSE_COLS = ["id", "geometry", "bbox", "subtype", "class", "names"]
WATER_COLS = ["id", "geometry", "bbox", "subtype", "class", "names"]
LAND_COLS = ["id", "geometry", "bbox", "subtype", "class", "names"]


def fetch_all(cfg: dict) -> dict[str, Path]:
    bbox = data_bbox(cfg)
    rel = cfg["overture_release"]
    ddir = cfg["_root"] / cfg["data_dir"] / f"overture_{rel}"
    out = {}
    for theme, type_, cols in [("transportation", "segment", SEGMENT_COLS),
                               ("base", "water", WATER_COLS),
                               ("base", "land_use", LANDUSE_COLS),
                               ("base", "land", LAND_COLS),
                               ("buildings", "building", BUILDING_COLS)]:
        dest = ddir / f"{type_}.parquet"
        extract(rel, theme, type_, bbox, cols, dest)
        out[type_] = dest
    from .elevation import fetch_dem
    out["dem"] = fetch_dem(cfg)
    return out


if __name__ == "__main__":
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    print(fetch_all(cfg))
