"""Analyse de profil altimétrique : pente max en montée sur fenêtre glissante, dénivelé, tronçons raides."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def smooth_elevation(dist: np.ndarray, elev: np.ndarray, smooth_m: float = 30.0) -> np.ndarray:
    """Lissage léger (moyenne glissante ~smooth_m) pour atténuer le bruit du MNT (bâti, pas de 25-30 m)."""
    if len(dist) < 3:
        return elev.copy()
    step = np.median(np.diff(dist)) if len(dist) > 1 else smooth_m
    k = max(1, int(round(smooth_m / max(step, 1e-6))))
    if k <= 1:
        return elev.copy()
    return uniform_filter1d(elev, size=k, mode="nearest")


def windowed_grades(dist: np.ndarray, elev: np.ndarray, window_m: float, min_len_m: float = 1.0) -> np.ndarray:
    """Pente (%) sur une fenêtre de `window_m` mètres centrée sur chaque point (fenêtre tronquée aux bords).

    Pente = (alt(d + w/2) - alt(d - w/2)) / max(longueur réelle de la fenêtre, min_len_m).
    `min_len_m` > 1 amortit le bruit du MNT sur les arêtes très courtes (utilisé pour le graphe).
    """
    if len(dist) < 2:
        return np.zeros_like(dist)
    half = window_m / 2.0
    d0 = np.clip(dist - half, dist[0], dist[-1])
    d1 = np.clip(dist + half, dist[0], dist[-1])
    e0 = np.interp(d0, dist, elev)
    e1 = np.interp(d1, dist, elev)
    length = np.maximum(d1 - d0, min_len_m)
    return (e1 - e0) / length * 100.0


def segment_grades(dist: np.ndarray, elev: np.ndarray) -> np.ndarray:
    """Pente (%) de chaque intervalle entre points consécutifs (len = n-1)."""
    dd = np.maximum(np.diff(dist), 1e-6)
    return np.diff(elev) / dd * 100.0


def steep_intervals(dist: np.ndarray, grades_pt: np.ndarray, threshold_pct: float):
    """Intervalles [d_start, d_end] où la pente (fenêtrée, en montée) dépasse le seuil."""
    steep = grades_pt > threshold_pct
    out, start = [], None
    for i, s in enumerate(steep):
        if s and start is None:
            start = i
        if (not s or i == len(steep) - 1) and start is not None:
            end = i if s else i - 1
            out.append((float(dist[start]), float(dist[end])))
            start = None
    return out


def elevation_gain(elev: np.ndarray) -> float:
    d = np.diff(elev)
    return float(d[d > 0].sum())


def elevation_loss(elev: np.ndarray) -> float:
    d = np.diff(elev)
    return float(-d[d < 0].sum())


def travel_time_s(dist: np.ndarray, elev: np.ndarray, rider: dict) -> float:
    """Temps estimé (s) par modèle de puissance : P = (m g (crr + pente) + 0.5 rho CdA v²) v, résolu pour v.

    Vitesse plafonnée en descente ; en dessous de `min_speed_kmh`, le cycliste pousse à cette vitesse.
    """
    g, rho = 9.81, 1.2
    P, m, crr, cda = rider["power_w"], rider["mass_kg"], rider["crr"], rider["cda"]
    vmax = rider["max_speed_kmh"] / 3.6
    vmin = rider["min_speed_kmh"] / 3.6
    total = 0.0
    dd = np.diff(dist)
    de = np.diff(elev)
    for L, dh in zip(dd, de):
        if L <= 0:
            continue
        s = dh / L
        # résolution de P = a v + b v^3 par Newton
        a = m * g * (crr + s)
        b = 0.5 * rho * cda
        v = 5.0
        for _ in range(30):
            f = a * v + b * v ** 3 - P
            fp = a + 3 * b * v * v
            if fp <= 0:
                break
            v_new = v - f / fp
            if v_new <= 0.1:
                v_new = 0.1
            if abs(v_new - v) < 1e-4:
                v = v_new
                break
            v = v_new
        if a < 0 and a * vmax + b * vmax ** 3 <= P:  # descente : on ne pédale pas plus vite que vmax
            v = vmax
        v = min(max(v, vmin), vmax)
        total += L / v
    return total
