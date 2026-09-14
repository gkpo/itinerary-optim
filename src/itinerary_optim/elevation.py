"""Modèle numérique de terrain à partir des tuiles Terrarium (AWS Terrain Tiles, accès public).

Encodage Terrarium : altitude (m) = (R * 256 + G + B / 256) - 32768.
"""
from __future__ import annotations

import io
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import map_coordinates

from .config import data_bbox

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _get_tile(z, x, y):
    url = TILE_URL.format(z=z, x=x, y=y)
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            img = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.float64)
            return (img[:, :, 0] * 256.0 + img[:, :, 1] + img[:, :, 2] / 256.0) - 32768.0
        except Exception as e:
            if attempt == 3:
                raise
            print(f"  retry tile {z}/{x}/{y}: {e}", file=sys.stderr)
    raise RuntimeError


def fetch_dem(cfg: dict) -> Path:
    """Assemble une mosaïque de tuiles couvrant l'emprise ; sauvegarde en .npz (grille + géoréférencement)."""
    z = int(cfg["dem_zoom"])
    xmin, ymin, xmax, ymax = data_bbox(cfg)
    dest = cfg["_root"] / cfg["data_dir"] / f"dem_terrarium_z{z}.npz"
    if dest.exists():
        return dest
    tx0, ty1 = lonlat_to_tile(xmin, ymin, z)
    tx1, ty0 = lonlat_to_tile(xmax, ymax, z)
    tx0, ty0, tx1, ty1 = int(tx0), int(ty0), int(tx1), int(ty1)
    tiles = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]
    print(f"[dem] {len(tiles)} tuiles z{z}", file=sys.stderr)
    with ThreadPoolExecutor(8) as ex:
        arrays = list(ex.map(lambda t: _get_tile(z, t[0], t[1]), tiles))
    ncols, nrows = tx1 - tx0 + 1, ty1 - ty0 + 1
    grid = np.zeros((nrows * 256, ncols * 256))
    for (x, y), a in zip(tiles, arrays):
        grid[(y - ty0) * 256:(y - ty0 + 1) * 256, (x - tx0) * 256:(x - tx0 + 1) * 256] = a
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, grid=grid, z=z, tx0=tx0, ty0=ty0)
    return dest


class DEM:
    def __init__(self, path: Path):
        d = np.load(path)
        self.grid, self.z, self.tx0, self.ty0 = d["grid"], int(d["z"]), int(d["tx0"]), int(d["ty0"])

    def sample(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """Altitude par interpolation bilinéaire (les tuiles sont en Web Mercator)."""
        lons, lats = np.asarray(lons, float), np.asarray(lats, float)
        n = 2 ** self.z
        px = ((lons + 180.0) / 360.0 * n - self.tx0) * 256.0 - 0.5
        lat_r = np.radians(lats)
        py = ((1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / math.pi) / 2.0 * n - self.ty0) * 256.0 - 0.5
        return map_coordinates(self.grid, [py, px], order=1, mode="nearest")
