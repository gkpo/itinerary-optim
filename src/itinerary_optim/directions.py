"""Feuille de route textuelle (type Google Maps) à partir d'un itinéraire."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geo import bearing
from .graph import Graph
from .routing import Route

CLASS_LABEL = {
    "cycleway": "piste cyclable", "service": "voie de service", "path": "chemin", "footway": "allée",
    "pedestrian": "voie piétonne", "track": "chemin", "living_street": "zone de rencontre",
    "residential": "rue sans nom", "unclassified": "voie sans nom", "tertiary": "voie sans nom",
    "secondary": "voie sans nom", "primary": "voie sans nom", "unknown": "voie sans nom",
}


@dataclass
class Step:
    instruction: str
    name: str
    length_m: float
    d_start: float
    d_end: float
    max_grade: float          # pente fenêtrée max en montée (%) sur l'étape
    climb_len_m: float        # longueur en montée (> 2 %)
    climb_gain_m: float       # dénivelé positif de l'étape
    descent_len_m: float      # longueur en descente (< -2 %)
    steep_len_m: float        # longueur > seuil raide
    lon: float = 0.0          # position du début de l'étape (pour l'étiquette)
    lat: float = 0.0
    idx_start: int = 0        # indices dans le profil de l'itinéraire
    idx_end: int = 0
    notes: list[str] = field(default_factory=list)


def _turn_word(delta: float) -> str:
    a = abs(delta)
    side = "droite" if delta > 0 else "gauche"
    if a < 25:
        return "Continuer tout droit sur"
    if a < 60:
        return f"Légèrement à {side} sur"
    if a < 135:
        return f"Tourner à {side} sur"
    return f"Faire demi-tour (virage serré à {side}) sur"


def _fmt_dist(m: float) -> str:
    return f"{m / 1000:.1f} km" if m >= 1000 else f"{m:.0f} m"


def build_steps(route: Route, graph: Graph, cfg: dict) -> list[Step]:
    E = graph.edges
    steep_thr = float(cfg["steep_threshold_pct"])
    # 1) regrouper les arcs consécutifs portant le même nom
    groups: list[dict] = []
    cum = 0.0
    for j, (ei, fwd) in enumerate(route.arcs):
        e = E[ei]
        name = e.name or CLASS_LABEL.get(e.road_class, "voie sans nom")
        if groups and groups[-1]["name"] == name:
            groups[-1]["arcs"].append(j)
            groups[-1]["d_end"] = cum + e.length
        else:
            groups.append({"name": name, "arcs": [j], "d_start": cum, "d_end": cum + e.length, "named": e.name is not None})
        cum += e.length
    # 2) fusionner les tronçons très courts (traversées de carrefour, bretelles) avec le précédent
    merged: list[dict] = []
    for g in groups:
        if merged and (g["d_end"] - g["d_start"] < 25 and not g["named"]):
            merged[-1]["d_end"] = g["d_end"]
            merged[-1]["arcs"] += g["arcs"]
        elif merged and merged[-1]["name"] == g["name"]:
            merged[-1]["d_end"] = g["d_end"]
            merged[-1]["arcs"] += g["arcs"]
        else:
            merged.append(g)
    # 3) instructions et statistiques de pente
    steps: list[Step] = []
    prev_bearing = None
    n = len(route.dist)
    for g in merged:
        i0 = int(np.searchsorted(route.dist, g["d_start"], side="left"))
        i1 = int(np.searchsorted(route.dist, g["d_end"], side="right")) - 1
        i0, i1 = max(0, min(i0, n - 1)), max(0, min(i1, n - 1))
        # azimut d'entrée dans l'étape (sur les ~30 premiers mètres)
        k = i0
        while k + 1 < n and route.dist[k + 1] - route.dist[i0] < 30 and k + 1 <= i1:
            k += 1
        k = max(k, min(i0 + 1, n - 1))
        b_in = bearing(route.lons[i0], route.lats[i0], route.lons[k], route.lats[k])
        if prev_bearing is None:
            instr = "Partir sur"
        else:
            delta = (b_in - prev_bearing + 540) % 360 - 180
            instr = _turn_word(delta)
        # azimut de sortie (30 derniers mètres)
        k2 = i1
        while k2 - 1 > i0 and route.dist[i1] - route.dist[k2 - 1] < 30:
            k2 -= 1
        k2 = min(k2, max(i1 - 1, 0))
        prev_bearing = bearing(route.lons[k2], route.lats[k2], route.lons[i1], route.lats[i1]) if i1 > k2 else b_in
        seg = slice(i0, i1 + 1)
        gr = route.grades[seg]
        dd = np.diff(route.dist[i0:i1 + 1]) if i1 > i0 else np.array([])
        gpt = gr[:-1] if len(gr) > 1 else np.array([])
        el = route.elev[seg]
        de = np.diff(el) if len(el) > 1 else np.array([])
        st = Step(
            instruction=instr, name=g["name"], length_m=g["d_end"] - g["d_start"], d_start=g["d_start"], d_end=g["d_end"],
            max_grade=float(gr.max()) if len(gr) else 0.0,
            climb_len_m=float(dd[gpt > 2.0].sum()) if len(dd) else 0.0,
            climb_gain_m=float(de[de > 0].sum()) if len(de) else 0.0,
            descent_len_m=float(dd[gpt < -2.0].sum()) if len(dd) else 0.0,
            steep_len_m=float(dd[gpt > steep_thr].sum()) if len(dd) else 0.0,
            lon=float(route.lons[i0]), lat=float(route.lats[i0]), idx_start=i0, idx_end=i1,
        )
        if st.climb_len_m >= 40 and st.max_grade >= 2.0:
            note = f"montée sur {_fmt_dist(st.climb_len_m)} (+{st.climb_gain_m:.0f} m), pente max {st.max_grade:.1f} %"
            if st.steep_len_m > 0:
                note += f", dont {st.steep_len_m:.0f} m à plus de {steep_thr:g} %"
            st.notes.append(note)
        if st.descent_len_m >= 40:
            st.notes.append(f"descente sur {_fmt_dist(st.descent_len_m)} (pente min {gr.min():.1f} %)")
        steps.append(st)
    return steps


def format_steps(steps: list[Step], route: Route, cfg: dict) -> str:
    lines = [f"Départ : {cfg['start']['name']}"]
    for i, s in enumerate(steps, 1):
        line = f"{i:2d}. {s.instruction} **{s.name}** — {_fmt_dist(s.length_m)}"
        if s.notes:
            line += "  \n    ⛰ " + " ; ".join(s.notes) if any("montée" in n for n in s.notes) else "  \n    ↘ " + " ; ".join(s.notes)
        lines.append(line)
    lines.append(f"{len(steps) + 1:2d}. Arrivée : {cfg['end']['name']} — total {_fmt_dist(route.length_m)}, "
                 f"{route.time_s / 60:.0f} min, D+ {route.gain_m:.0f} m")
    return "\n".join(lines)
