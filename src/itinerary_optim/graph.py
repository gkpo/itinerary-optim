"""Construction du graphe routable vélo à partir des segments Overture + MNT.

Chaque segment Overture est découpé en arêtes entre connecteurs consécutifs. Pour chaque arête on
échantillonne l'altitude tous les `sample_step_m` mètres et on calcule, dans chaque sens :
  - la longueur,
  - la pente max en montée sur fenêtre glissante (= critère d'optimisation),
  - la longueur cumulée à plus du seuil "raide",
  - le dénivelé positif.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from shapely import wkb
from shapely.ops import substring

from .elevation import DEM
from .geo import densify, polyline_length
from .profile import smooth_elevation, windowed_grades


@dataclass
class Edge:
    u: int
    v: int
    seg_id: str
    name: str | None
    road_class: str
    length: float
    lons: np.ndarray
    lats: np.ndarray
    dist: np.ndarray      # distance cumulée (m) le long des points échantillonnés (sens u -> v)
    elev: np.ndarray      # altitude lissée
    max_grade_fwd: float  # pente max montée (%) sens u->v, fenêtre glissante
    max_grade_bwd: float  # idem sens v->u
    steep_len_fwd: float
    steep_len_bwd: float
    gain_fwd: float
    gain_bwd: float
    allow_fwd: bool = True
    allow_bwd: bool = True
    pref: float = 1.0


@dataclass
class Graph:
    edges: list[Edge]
    node_xy: dict[int, tuple[float, float]]           # id -> (lon, lat)
    adj: dict[int, list[tuple[int, int, bool]]] = field(default_factory=dict)  # node -> [(edge_idx, other, forward)]

    def build_adj(self):
        self.adj = {}
        for i, e in enumerate(self.edges):
            if e.allow_fwd:
                self.adj.setdefault(e.u, []).append((i, e.v, True))
            if e.allow_bwd:
                self.adj.setdefault(e.v, []).append((i, e.u, False))

    def nearest_node(self, lon: float, lat: float) -> int:
        ids = np.array(list(self.node_xy.keys()))
        xy = np.array(list(self.node_xy.values()))
        kx = math.cos(math.radians(lat))
        d2 = ((xy[:, 0] - lon) * kx) ** 2 + (xy[:, 1] - lat) ** 2
        return int(ids[int(np.argmin(d2))])


def _bike_access(row) -> tuple[bool, bool, bool, bool]:
    """(autorisé, sens direct, sens inverse, vélo explicitement autorisé) d'après access_restrictions.

    Règles : les restrictions horaires sont ignorées ; les accès "privé", "sur autorisation",
    "clients"/"riverains" sont traités comme interdits ; une règle sans mode s'applique à tous ;
    une règle avec modes ne concerne le vélo que si "bicycle" y figure.
    """
    allow, fwd, bwd, explicit = True, True, True, False
    for r in row.get("access_restrictions") or []:
        when = r.get("when") or {}
        modes = when.get("mode") or []
        during = when.get("during")
        if during:
            continue
        if modes and "bicycle" not in modes:
            continue
        heading = when.get("heading")
        atype = r.get("access_type")
        if atype == "allowed" and (when.get("recognized") or when.get("using")):
            atype = "denied"  # privé / permis / clients / riverains seulement
        if atype == "denied":
            if heading == "forward":
                fwd = False
            elif heading == "backward":
                bwd = False
            else:
                allow = False
        elif atype in ("allowed", "designated"):
            if "bicycle" in modes:
                explicit = True
            if heading == "forward":
                fwd = True
            elif heading == "backward":
                bwd = True
            else:
                allow, fwd, bwd = True, True, True
    return allow, fwd, bwd, explicit


def _flag_covers(row, flag: str, a: float, b: float) -> bool:
    for f in row.get("road_flags") or []:
        if flag not in (f.get("values") or []):
            continue
        bt = f.get("between")
        if not bt or (bt[0] <= a + 1e-9 and bt[1] >= b - 1e-9):
            return True
    return False


def _primary_name(names) -> str | None:
    if not names:
        return None
    return names.get("primary")


def build_graph(cfg: dict, seg_path: Path, dem_path: Path, cache: Path | None = None) -> Graph:
    if cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    classes: dict = cfg["road_classes"]
    step = float(cfg["sample_step_m"])
    window = float(cfg["slope_window_m"])
    steep_thr = float(cfg["steep_threshold_pct"])
    min_len = 1.0  # fenêtre tronquée aux extrémités de l’arête (= pente moyenne pour les arêtes courtes)
    dem = DEM(dem_path)
    table = pq.read_table(seg_path)
    rows = table.to_pylist()
    print(f"[graph] {len(rows)} segments Overture", file=sys.stderr)

    node_ids: dict[str, int] = {}
    node_xy: dict[int, tuple[float, float]] = {}
    edges: list[Edge] = []

    def nid(cid: str, lon: float, lat: float) -> int:
        if cid not in node_ids:
            node_ids[cid] = len(node_ids)
            node_xy[node_ids[cid]] = (lon, lat)
        return node_ids[cid]

    n_skip = 0
    for row in rows:
        if row["subtype"] != "road" or row["class"] not in classes:
            n_skip += 1
            continue
        allow, allow_fwd, allow_bwd, explicit = _bike_access(row)
        if not allow or not (allow_fwd or allow_bwd):
            n_skip += 1
            continue
        if row["class"] == "footway" and (row.get("subclass") in ("sidewalk", "crosswalk") or not explicit):
            n_skip += 1  # trottoirs / passages piétons et chemins piétons sans autorisation vélo explicite
            continue
        geom = wkb.loads(row["geometry"])
        if geom.geom_type != "LineString" or geom.length == 0:
            continue
        conns = sorted(row["connectors"] or [], key=lambda c: c["at"])
        if len(conns) < 2:
            continue
        name = _primary_name(row.get("names"))
        rclass = row["class"]
        for a, b in zip(conns[:-1], conns[1:]):
            if b["at"] <= a["at"]:
                continue
            part = substring(geom, a["at"], b["at"], normalized=True)
            if part.geom_type != "LineString" or part.is_empty:
                continue
            xs, ys = np.array(part.coords)[:, 0], np.array(part.coords)[:, 1]
            lons, lats, dist = densify(xs, ys, step)
            length = float(dist[-1])
            if length <= 0:
                continue
            elev = smooth_elevation(dist, dem.sample(lons, lats))
            if _flag_covers(row, "is_tunnel", a["at"], b["at"]) or _flag_covers(row, "is_bridge", a["at"], b["at"]):
                # le MNT décrit la surface : sous un pont / dans un tunnel on interpole entre les extrémités
                elev = np.interp(dist, [dist[0], dist[-1]], [elev[0], elev[-1]])
            gf = windowed_grades(dist, elev, window, min_len)
            gb = -gf[::-1]
            dd = np.diff(dist)
            de = np.diff(elev)
            # longueur "raide" : intervalles dont la pente fenêtrée (au point de départ) dépasse le seuil
            steep_f = float(dd[gf[:-1] > steep_thr].sum())
            steep_b = float(dd[::-1][gb[:-1] > steep_thr].sum())
            u = nid(a["connector_id"], float(xs[0]), float(ys[0]))
            v = nid(b["connector_id"], float(xs[-1]), float(ys[-1]))
            edges.append(Edge(
                u=u, v=v, seg_id=row["id"], name=name, road_class=rclass, length=length,
                lons=lons, lats=lats, dist=dist, elev=elev,
                max_grade_fwd=float(gf.max()), max_grade_bwd=float(gb.max()),
                steep_len_fwd=steep_f, steep_len_bwd=steep_b,
                gain_fwd=float(de[de > 0].sum()), gain_bwd=float(-de[de < 0].sum()),
                allow_fwd=allow_fwd, allow_bwd=allow_bwd, pref=float(classes[rclass]),
            ))
    g = Graph(edges=edges, node_xy=node_xy)
    g.build_adj()
    print(f"[graph] {len(edges)} arêtes, {len(node_xy)} nœuds ({n_skip} segments exclus)", file=sys.stderr)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(g, f)
    return g
