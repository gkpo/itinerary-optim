"""Petits utilitaires géographiques (projection locale en mètres, distances, azimuts)."""
from __future__ import annotations

import math

import numpy as np

R_EARTH = 6371008.8


class LocalProjection:
    """Projection équirectangulaire locale : (lon, lat) -> (x, y) en mètres. Suffisante pour ~20 km."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = math.radians(1) * R_EARTH * math.cos(math.radians(lat0))
        self.ky = math.radians(1) * R_EARTH

    def forward(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def inverse(self, x, y):
        return np.asarray(x) / self.kx + self.lon0, np.asarray(y) / self.ky + self.lat0


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (np.asarray(lon1), np.asarray(lat1), np.asarray(lon2), np.asarray(lat2)))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def polyline_length(lons, lats) -> float:
    lons, lats = np.asarray(lons), np.asarray(lats)
    if len(lons) < 2:
        return 0.0
    return float(haversine(lons[:-1], lats[:-1], lons[1:], lats[1:]).sum())


def bearing(lon1, lat1, lon2, lat2) -> float:
    """Azimut en degrés (0 = nord, 90 = est)."""
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def densify(lons, lats, step_m: float):
    """Rééchantillonne une polyligne tous les `step_m` mètres (conserve les sommets d'origine)."""
    lons, lats = np.asarray(lons, float), np.asarray(lats, float)
    out_lon, out_lat, out_d = [lons[0]], [lats[0]], [0.0]
    cum = 0.0
    for i in range(len(lons) - 1):
        seg = float(haversine(lons[i], lats[i], lons[i + 1], lats[i + 1]))
        n = max(1, int(math.ceil(seg / step_m)))
        for k in range(1, n + 1):
            f = k / n
            out_lon.append(lons[i] + f * (lons[i + 1] - lons[i]))
            out_lat.append(lats[i] + f * (lats[i + 1] - lats[i]))
            out_d.append(cum + f * seg)
        cum += seg
    return np.array(out_lon), np.array(out_lat), np.array(out_d)
