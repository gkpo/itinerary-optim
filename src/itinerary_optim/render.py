"""Rendu cartographique PNG : fond de carte (Overture : occupation du sol, eau, bâti, voirie, noms de
rues) + tracé de l'itinéraire, tronçons raides surlignés, départ/arrivée, légende et profil incrustés.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import FancyBboxPatch
from shapely import wkb
from shapely.geometry import box

from .directions import Step
from .geo import LocalProjection
from .graph import Graph
from .routing import Route

# ------------------------------------------------------------------ style
C_LAND = "#f2efe9"
C_WATER = "#aad3df"
C_GREEN = "#cdebb0"
C_PARK = "#c8facc"
C_BUILDING = "#d9d0c9"
C_BUILDING_EDGE = "#c7bdb4"
C_INDUSTRIAL = "#ebdbe8"
C_CEMETERY = "#aacbaf"
C_SCHOOL = "#fff4d3"
C_RAIL = "#8a8a8a"
ROAD_STYLE = {  # classe -> (couleur, largeur (m) approx., ordre)
    "motorway": ("#e892a2", 22, 9), "trunk": ("#f9b29c", 18, 8), "primary": ("#fcd6a4", 15, 7),
    "secondary": ("#f7fabf", 13, 6), "tertiary": ("#ffffff", 11, 5), "residential": ("#ffffff", 8, 4),
    "unclassified": ("#ffffff", 8, 4), "living_street": ("#ededed", 7, 4), "service": ("#ffffff", 4, 3),
    "pedestrian": ("#dddde8", 6, 3), "cycleway": ("#9db8ff", 2.5, 3), "path": ("#b09a72", 2, 2),
    "footway": ("#f4a29b", 1.5, 2), "track": ("#a98b5e", 2, 2), "steps": ("#f4a29b", 2, 2), "unknown": ("#ffffff", 5, 3),
}
LABEL_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary", "residential", "unclassified", "living_street", "pedestrian", "cycleway"}
C_ROUTE = "#1f5fd6"
C_STEEP = "#e0161a"
C_START = "#1a9c3b"
C_END = "#c8102e"


def _poly_coords(geom, proj):
    polys = []
    if geom.geom_type == "Polygon":
        polys = [geom]
    elif geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
    out = []
    for p in polys:
        x, y = proj.forward(np.array(p.exterior.coords)[:, 0], np.array(p.exterior.coords)[:, 1])
        out.append(np.column_stack([x, y]))
    return out


def _lines_coords(geom, proj):
    lines = []
    if geom.geom_type == "LineString":
        lines = [geom]
    elif geom.geom_type == "MultiLineString":
        lines = list(geom.geoms)
    out = []
    for l in lines:
        c = np.array(l.coords)
        x, y = proj.forward(c[:, 0], c[:, 1])
        out.append(np.column_stack([x, y]))
    return out


class MapData:
    """Charge et projette une fois les couches de fond pour une emprise donnée (lon/lat)."""

    def __init__(self, data_dir: Path, proj: LocalProjection, extent_ll: tuple[float, float, float, float]):
        self.proj = proj
        bb = box(*extent_ll)
        self.landuse, self.water_poly, self.water_line, self.buildings, self.roads, self.rails = [], [], [], [], [], []
        self.road_names = []  # (name, class, coords)

        def rows(name, cols):
            t = pq.read_table(data_dir / f"{name}.parquet", columns=cols + ["bbox"])
            b = t.column("bbox").to_pylist()
            keep = [i for i, r in enumerate(b) if r["xmin"] <= extent_ll[2] and r["xmax"] >= extent_ll[0]
                    and r["ymin"] <= extent_ll[3] and r["ymax"] >= extent_ll[1]]
            return t.take(keep).to_pylist()

        for r in rows("land_use", ["geometry", "subtype", "class"]):
            g = wkb.loads(r["geometry"])
            st, cl = r["subtype"], r["class"]
            if st in ("park", "recreation", "agriculture", "golf") or cl in ("grass", "garden", "allotments", "meadow"):
                col = C_PARK if st == "park" else C_GREEN
            elif st == "cemetery":
                col = C_CEMETERY
            elif st in ("developed", "construction", "transportation") and cl in ("industrial", "railway", "brownfield", "construction"):
                col = C_INDUSTRIAL
            elif st == "education":
                col = C_SCHOOL
            else:
                continue
            for c in _poly_coords(g, proj):
                self.landuse.append((c, col))
        for r in rows("land", ["geometry", "subtype", "class"]):
            if r["subtype"] in ("forest", "grass", "shrub", "wetland"):
                for c in _poly_coords(wkb.loads(r["geometry"]), proj):
                    self.landuse.append((c, C_GREEN))
        for r in rows("water", ["geometry", "subtype", "class"]):
            g = wkb.loads(r["geometry"])
            self.water_poly += _poly_coords(g, proj)
            if r["subtype"] in ("river", "canal", "stream"):
                self.water_line += _lines_coords(g, proj)
        for r in rows("building", ["geometry"]):
            self.buildings += _poly_coords(wkb.loads(r["geometry"]), proj)
        for r in rows("segment", ["geometry", "subtype", "class", "names"]):
            g = wkb.loads(r["geometry"])
            if not g.intersects(bb):
                continue
            cs = _lines_coords(g, proj)
            if r["subtype"] == "rail":
                self.rails += cs
            elif r["subtype"] == "road" and r["class"] in ROAD_STYLE:
                for c in cs:
                    self.roads.append((c, r["class"]))
                    nm = (r.get("names") or {}).get("primary") if r.get("names") else None
                    if nm and r["class"] in LABEL_CLASSES:
                        self.road_names.append((nm, r["class"], c))


def _meters_per_point(ax, fig):
    """Combien de mètres (données) par point typographique, pour dimensionner les largeurs de voies."""
    x0, x1 = ax.get_xlim()
    w_in = ax.get_position().width * fig.get_size_inches()[0]
    return (x1 - x0) / (w_in * 72.0)


def _draw_basemap(ax, fig, md: MapData, extent_xy, unit_scale: float = 1.0):
    """`unit_scale` : nombre d'unités projetées par mètre réel (1 en projection locale, 1/cos φ en Web Mercator)."""
    ax.set_facecolor(C_LAND)
    x0, x1, y0, y1 = extent_xy
    if md.landuse:
        cols = {}
        for c, col in md.landuse:
            cols.setdefault(col, []).append(c)
        for col, cs in cols.items():
            ax.add_collection(PolyCollection(cs, facecolors=col, edgecolors="none", zorder=1))
    if md.water_poly:
        ax.add_collection(PolyCollection(md.water_poly, facecolors=C_WATER, edgecolors="none", zorder=2))
    if md.water_line:
        ax.add_collection(LineCollection(md.water_line, colors=C_WATER, linewidths=1.2, zorder=2))
    if md.buildings:
        ax.add_collection(PolyCollection(md.buildings, facecolors=C_BUILDING, edgecolors=C_BUILDING_EDGE, linewidths=0.2, zorder=3))
    mpp = _meters_per_point(ax, fig) / unit_scale
    # voirie : d'abord les bordures (casing) puis le remplissage, par ordre d'importance
    by_class = {}
    for c, cl in md.roads:
        by_class.setdefault(cl, []).append(c)
    order = sorted(by_class, key=lambda k: ROAD_STYLE[k][2])
    for cl in order:
        col, w_m, z = ROAD_STYLE[cl]
        lw = max(w_m / mpp, 0.6)
        if cl in ("cycleway", "path", "footway", "track", "steps"):
            ax.add_collection(LineCollection(by_class[cl], colors=col, linewidths=lw, linestyles=(0, (3, 2)) if cl != "cycleway" else "solid", zorder=4, capstyle="round"))
        else:
            ax.add_collection(LineCollection(by_class[cl], colors="#b4b4b4", linewidths=lw + 1.0, zorder=4, capstyle="round", joinstyle="round"))
    for cl in order:
        col, w_m, z = ROAD_STYLE[cl]
        if cl in ("cycleway", "path", "footway", "track", "steps"):
            continue
        lw = max(w_m / mpp, 0.6)
        ax.add_collection(LineCollection(by_class[cl], colors=col, linewidths=lw, zorder=5 + z * 0.01, capstyle="round", joinstyle="round"))
    if md.rails:
        ax.add_collection(LineCollection(md.rails, colors="#ffffff", linewidths=2.2, zorder=6))
        ax.add_collection(LineCollection(md.rails, colors=C_RAIL, linewidths=2.2, linestyles=(0, (6, 6)), zorder=6.01))


class _LabelPlacer:
    """Placement d'étiquettes le long de lignes avec test de recouvrement approximatif (rectangles orientés)."""

    def __init__(self, ax, fig, extent_xy, fontsize):
        self.ax, self.fontsize = ax, fontsize
        self.x0, self.x1, self.y0, self.y1 = extent_xy
        self.mpp = _meters_per_point(ax, fig)   # mètres par point typographique
        self.boxes = []  # (cx, cy, half_len, half_h, ang_rad, name)

    def text_size_m(self, name):
        return len(name) * self.fontsize * 0.58 * self.mpp, self.fontsize * 1.1 * self.mpp

    def _overlaps(self, cx, cy, hl, hh, ang, name):
        for bx, by, bhl, bhh, bang, bname in self.boxes:
            d = math.hypot(cx - bx, cy - by)
            if d > hl + bhl + hh:
                continue
            # projection sur les axes de chaque boîte (test grossier de séparation)
            ca, sa = math.cos(ang), math.sin(ang)
            u = abs((bx - cx) * ca + (by - cy) * sa); v = abs(-(bx - cx) * sa + (by - cy) * ca)
            if u > hl + bhl * abs(math.cos(bang - ang)) + bhh * abs(math.sin(bang - ang)) or \
               v > hh + bhl * abs(math.sin(bang - ang)) + bhh * abs(math.cos(bang - ang)):
                continue
            return True
        return False

    def place_on_line(self, name, cc, bold=False, color="#4a4a4a", max_per_name=1, min_sep_same=600.0, offset_m=0.0):
        """Tente de placer `name` au milieu (par longueur) de la polyligne `cc` (déjà projetée)."""
        seg = np.hypot(np.diff(cc[:, 0]), np.diff(cc[:, 1]))
        if len(seg) == 0:
            return False
        cum = np.concatenate([[0], np.cumsum(seg)])
        L = cum[-1]
        tl, th = self.text_size_m(name)
        if tl > L * 1.05:
            return False
        if sum(1 for b in self.boxes if b[5] == name) >= max_per_name:
            return False
        # candidats : milieu, puis 35 % / 65 %, 20 % / 80 %
        for frac in (0.5, 0.35, 0.65, 0.2, 0.8):
            mid = L * frac
            i = max(0, min(int(np.searchsorted(cum, mid)) - 1, len(cc) - 2))
            f = (mid - cum[i]) / max(seg[i], 1e-9)
            x = cc[i, 0] + f * (cc[i + 1, 0] - cc[i, 0]); y = cc[i, 1] + f * (cc[i + 1, 1] - cc[i, 1])
            j0 = max(0, int(np.searchsorted(cum, mid - tl / 2)) - 1); j1 = min(len(cc) - 1, int(np.searchsorted(cum, mid + tl / 2)))
            dx, dy = cc[j1, 0] - cc[j0, 0], cc[j1, 1] - cc[j0, 1]
            if math.hypot(dx, dy) < tl * 0.7:  # rue trop sinueuse à cet endroit
                continue
            ang = math.atan2(dy, dx)
            if ang > math.pi / 2: ang -= math.pi
            if ang < -math.pi / 2: ang += math.pi
            if offset_m:
                x += -math.sin(ang) * offset_m; y += math.cos(ang) * offset_m
            if not (self.x0 + tl / 2 < x < self.x1 - tl / 2 and self.y0 + th < y < self.y1 - th):
                continue
            if any(b[5] == name and math.hypot(x - b[0], y - b[1]) < min_sep_same for b in self.boxes):
                continue
            if self._overlaps(x, y, tl / 2, th / 2, ang, name):
                continue
            self.ax.text(x, y, name, fontsize=self.fontsize, rotation=math.degrees(ang), rotation_mode="anchor", ha="center", va="center",
                         color=color, zorder=20, fontweight="bold" if bold else "normal",
                         path_effects=[pe.withStroke(linewidth=2.4, foreground="white", alpha=0.92)])
            self.boxes.append((x, y, tl / 2, th / 2, ang, name))
            return True
        return False


def _merge_by_name(road_names):
    """Fusionne les tronçons Overture de même nom en polylignes continues (shapely linemerge)."""
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge
    by_name = {}
    for name, cl, c in road_names:
        by_name.setdefault(name, {"cls": cl, "parts": []})["parts"].append(c)
    out = []
    for name, d in by_name.items():
        merged = linemerge(MultiLineString([LineString(p) for p in d["parts"] if len(p) >= 2]))
        lines = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
        for l in lines:
            out.append((name, d["cls"], np.array(l.coords)))
    return out


def _label_streets(ax, fig, md: MapData, extent_xy, route_steps, route_xy, max_labels=160):
    """Étiquettes : d'abord les rues de l'itinéraire (sur le tracé lui-même), puis les voies visibles."""
    x0, x1, y0, y1 = extent_xy
    span = max(x1 - x0, y1 - y0)
    placer = _LabelPlacer(ax, fig, extent_xy, fontsize=6.4)
    route_names = set()
    if route_steps:
        rx, ry = route_xy
        for s in sorted(route_steps, key=lambda s: -s.length_m):
            if s.length_m < 60 or s.name.startswith(("voie", "piste", "chemin", "allée", "rue sans")):
                continue
            cc = np.column_stack([rx[s.idx_start:s.idx_end + 1], ry[s.idx_start:s.idx_end + 1]])
            placer.place_on_line(s.name, cc, bold=True, color="#111111", max_per_name=2, min_sep_same=900.0,
                                 offset_m=placer.fontsize * 1.05 * placer.mpp)
            route_names.add(s.name)
    cands = []
    for name, cl, c in _merge_by_name(md.road_names):
        inside = (c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1)
        if inside.sum() < 2:
            continue
        cc = c[inside]
        L = float(np.hypot(np.diff(cc[:, 0]), np.diff(cc[:, 1])).sum())
        if L < span * 0.025:
            continue
        cands.append((-ROAD_STYLE[cl][2], -L, name, cc))
    cands.sort(key=lambda t: (t[0], t[1]))
    n = len(placer.boxes)
    for _, negL, name, cc in cands:
        if n >= max_labels:
            break
        if name in route_names:
            continue
        if placer.place_on_line(name, cc, max_per_name=2 if -negL > span * 0.5 else 1, min_sep_same=span * 0.45):
            n += 1


def _save_png(fig, out_png: Path, dpi: int = 150):
    """Enregistre la figure puis réduit le PNG à une palette de 256 couleurs (octree, sans tramage) :
    ~5× plus léger sans perte visible pour une carte."""
    from PIL import Image
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    im = Image.open(out_png).convert("RGB")
    im.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(out_png, optimize=True)


def _fmt_time(s: float) -> str:
    m = int(round(s / 60))
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def render_route(route: Route, steps: list[Step], graph: Graph, cfg: dict, data_dir: Path, out_png: Path,
                 title: str, shortest_len: float, md: MapData | None = None, proj: LocalProjection | None = None,
                 extent_ll=None, base_extent_ll=None) -> Path:
    steep_thr = float(cfg["steep_threshold_pct"])
    window = float(cfg["slope_window_m"])
    if proj is None:
        proj = LocalProjection(float(route.lons.mean()), float(route.lats.mean()))
    if extent_ll is None:
        pad_lon = (route.lons.max() - route.lons.min()) * 0.12 + 0.004
        pad_lat = (route.lats.max() - route.lats.min()) * 0.12 + 0.003
        extent_ll = (route.lons.min() - pad_lon, route.lats.min() - pad_lat, route.lons.max() + pad_lon, route.lats.max() + pad_lat)
    if md is None:
        md = MapData(data_dir, proj, extent_ll)
    x0, y0 = proj.forward(extent_ll[0], extent_ll[1])
    x1, y1 = proj.forward(extent_ll[2], extent_ll[3])
    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    # figure : carte carrée-ish + bandeau profil en bas
    w_m, h_m = x1 - x0, y1 - y0
    fig_w = 13.0
    map_h = fig_w * h_m / w_m
    fig_h = map_h + 2.3
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    ax = fig.add_axes([0, 2.3 / fig_h, 1, map_h / fig_h])
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    _draw_basemap(ax, fig, md, (x0, x1, y0, y1))

    # tracé
    rx, ry = proj.forward(route.lons, route.lats)
    ax.plot(rx, ry, color="white", linewidth=7.5, zorder=10, solid_capstyle="round", solid_joinstyle="round", alpha=0.9)
    ax.plot(rx, ry, color=C_ROUTE, linewidth=4.2, zorder=11, solid_capstyle="round", solid_joinstyle="round")
    for a, b in route.steep_intervals:
        m = (route.dist >= a - 1e-6) & (route.dist <= b + 1e-6)
        if m.sum() >= 2:
            ax.plot(rx[m], ry[m], color=C_STEEP, linewidth=5.2, zorder=12, solid_capstyle="round", solid_joinstyle="round")
    # flèches de sens tous les ~800 m
    for d in np.arange(400, route.length_m - 200, 800):
        i = int(np.searchsorted(route.dist, d))
        if 0 < i < len(rx):
            ax.annotate("", xy=(rx[i], ry[i]), xytext=(rx[i - 1], ry[i - 1]),
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.2, mutation_scale=14), zorder=13)
    _label_streets(ax, fig, md, (x0, x1, y0, y1), steps, (rx, ry))
    # départ / arrivée
    sx, sy = proj.forward(cfg["start"]["lon"], cfg["start"]["lat"])
    ex, ey = proj.forward(cfg["end"]["lon"], cfg["end"]["lat"])
    for (px, py, col, lab) in [(sx, sy, C_START, "D"), (ex, ey, C_END, "A")]:
        ax.scatter([px], [py], s=330, c=col, edgecolors="white", linewidths=2, zorder=30, marker="o")
        ax.text(px, py, lab, ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=31)
    ax.annotate("Départ\n" + cfg["start"]["name"], (sx, sy), xytext=(12, 12), textcoords="offset points", fontsize=7,
                zorder=31, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_START, alpha=0.92))
    ax.annotate("Arrivée\n" + cfg["end"]["name"], (ex, ey), xytext=(12, -28), textcoords="offset points", fontsize=7,
                zorder=31, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_END, alpha=0.92))

    # échelle
    for L in (200, 500, 1000, 2000):
        if L <= w_m * 0.25:
            scale = L
    bx = x0 + w_m * 0.03; by = y0 + h_m * 0.025
    ax.plot([bx, bx + scale], [by, by], color="black", lw=3, zorder=40, solid_capstyle="butt")
    ax.plot([bx, bx + scale / 2], [by, by], color="white", lw=1.4, zorder=41, solid_capstyle="butt")
    ax.text(bx + scale / 2, by + h_m * 0.008, f"{scale} m" if scale < 1000 else f"{scale // 1000} km", ha="center", va="bottom", fontsize=7,
            zorder=41, path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    # nord
    nx_, ny_ = x1 - w_m * 0.035, y0 + h_m * 0.03
    ax.annotate("N", xy=(nx_, ny_ + h_m * 0.06), xytext=(nx_, ny_), ha="center", va="center", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5), zorder=41,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    # attribution
    ax.text(x1 - w_m * 0.005, y0 + h_m * 0.004, "Données : © Overture Maps Foundation, © OpenStreetMap contributors · Relief : AWS Terrain Tiles (Mapzen)",
            ha="right", va="bottom", fontsize=5.5, color="#333", zorder=41,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    # légende incrustée (en haut à gauche)
    detour = (route.length_m / shortest_len - 1) * 100
    lines = [
        (title, True),
        (f"Distance totale : {route.length_m / 1000:.2f} km", False),
        (f"Détour vs plus court : {detour:+.1f} %  (plus court : {shortest_len / 1000:.2f} km)", False),
        (f"Pente max en montée (fenêtre {window:.0f} m) : {route.max_grade_pct:.1f} %", False),
        (f"Tronçons > {steep_thr:g} % : {route.steep_len_m:.0f} m cumulés", False),
        (f"Dénivelé positif : +{route.gain_m:.0f} m   (négatif : −{route.loss_m:.0f} m)", False),
        (f"Temps estimé (vélo classique) : {_fmt_time(route.time_s)}", False),
    ]
    tx, ty = 0.012, 0.985
    lh = 0.031 * (10 / map_h)
    box_h = lh * (len(lines) + 3.4)
    ax.add_patch(FancyBboxPatch((tx - 0.004, ty - box_h), 0.40, box_h + 0.004, boxstyle="round,pad=0.006", transform=ax.transAxes,
                                fc="white", ec="#666", alpha=1.0, zorder=50))
    y = ty - lh * 0.8
    for txt, bold in lines:
        ax.text(tx + 0.006, y, txt, transform=ax.transAxes, fontsize=9.5 if bold else 8.2, fontweight="bold" if bold else "normal",
                va="center", ha="left", zorder=51)
        y -= lh
    # symboles
    y -= lh * 0.2
    ax.plot([tx + 0.012, tx + 0.045], [y, y], transform=ax.transAxes, color=C_ROUTE, lw=4, zorder=51)
    ax.text(tx + 0.055, y, "itinéraire", transform=ax.transAxes, fontsize=8, va="center", zorder=51)
    ax.plot([tx + 0.16, tx + 0.193], [y, y], transform=ax.transAxes, color=C_STEEP, lw=5, zorder=51)
    ax.text(tx + 0.203, y, f"pente > {steep_thr:g} %", transform=ax.transAxes, fontsize=8, va="center", zorder=51)
    y -= lh
    ax.scatter([tx + 0.028], [y], transform=ax.transAxes, s=90, c=C_START, edgecolors="white", zorder=51)
    ax.text(tx + 0.055, y, "départ (D)", transform=ax.transAxes, fontsize=8, va="center", zorder=51)
    ax.scatter([tx + 0.176], [y], transform=ax.transAxes, s=90, c=C_END, edgecolors="white", zorder=51)
    ax.text(tx + 0.203, y, "arrivée (A)", transform=ax.transAxes, fontsize=8, va="center", zorder=51)

    # profil altimétrique (bandeau bas)
    axp = fig.add_axes([0.06, 0.55 / fig_h, 0.92, 1.45 / fig_h])
    d_km = route.dist / 1000
    axp.fill_between(d_km, route.elev, route.elev.min() - 5, color="#dcd6cc", zorder=1)
    axp.plot(d_km, route.elev, color="#555", lw=1, zorder=3)
    up = route.grades > 2
    for a, b in route.steep_intervals:
        m = (route.dist >= a) & (route.dist <= b)
        axp.fill_between(d_km[m], route.elev[m], route.elev.min() - 5, color=C_STEEP, alpha=0.55, zorder=2)
    # montées modérées en orange clair
    m_up = (route.grades > 2) & (route.grades <= steep_thr)
    axp.fill_between(d_km, route.elev, route.elev.min() - 5, where=m_up, color="#f5b25e", alpha=0.5, zorder=2)
    axp.set_xlim(0, d_km[-1]); axp.set_ylim(route.elev.min() - 5, route.elev.max() + 8)
    axp.set_xlabel("distance (km)", fontsize=7, labelpad=1); axp.set_ylabel("altitude (m)", fontsize=7, labelpad=2)
    axp.tick_params(labelsize=6.5)
    axp.grid(True, lw=0.3, alpha=0.6)
    # repères des étapes principales (rues nommées de plus de 300 m)
    last_x = -1.0
    for s in steps:
        if s.length_m >= 300 and not s.name.startswith(("voie", "piste", "chemin", "allée", "rue sans")):
            xc = (s.d_start + s.length_m / 2) / 1000
            if xc - last_x < route.length_m / 1000 * 0.11:
                continue
            axp.axvline(s.d_start / 1000, color="#999", lw=0.4, zorder=2)
            axp.text(xc, route.elev.max() + 7, s.name, fontsize=5.2, ha="center", va="top", color="#333", clip_on=True)
            last_x = xc
    i = int(np.argmax(route.grades))
    axp.annotate(f"pente max {route.max_grade_pct:.1f} %", (d_km[i], route.elev[i]), xytext=(0, -22), textcoords="offset points",
                 fontsize=6.5, ha="center", color=C_STEEP if route.max_grade_pct > steep_thr else "#a05a00",
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#a05a00"))
    axp.text(0.995, 0.06, "profil : orange = montée 2–%g %%, rouge = > %g %% (fenêtre %.0f m)" % (steep_thr, steep_thr, window),
             transform=axp.transAxes, fontsize=6, ha="right", va="bottom", color="#444",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    _save_png(fig, out_png)
    plt.close(fig)
    return out_png


def render_overview(routes: list[Route], graph: Graph, cfg: dict, data_dir: Path, out_png: Path, shortest_len: float) -> Path:
    """Carte de synthèse : tous les tracés distincts superposés."""
    distinct = [r for r in routes if r.same_as is None]
    lons = np.concatenate([r.lons for r in distinct]); lats = np.concatenate([r.lats for r in distinct])
    proj = LocalProjection(float(lons.mean()), float(lats.mean()))
    pad_lon = (lons.max() - lons.min()) * 0.08 + 0.004; pad_lat = (lats.max() - lats.min()) * 0.08 + 0.003
    ext = (lons.min() - pad_lon, lats.min() - pad_lat, lons.max() + pad_lon, lats.max() + pad_lat)
    md = MapData(data_dir, proj, ext)
    x0, y0 = proj.forward(ext[0], ext[1]); x1, y1 = proj.forward(ext[2], ext[3])
    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    w_m, h_m = x1 - x0, y1 - y0
    fig_w = 13.0; fig_h = fig_w * h_m / w_m
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    _draw_basemap(ax, fig, md, (x0, x1, y0, y1))
    palette = ["#e6194b", "#1f5fd6", "#3cb44b", "#f58231", "#911eb4", "#469990", "#9a6324"]
    handles = []
    for i, r in enumerate(distinct):
        rx, ry = proj.forward(r.lons, r.lats)
        col = palette[i % len(palette)]
        ax.plot(rx, ry, color="white", lw=6.5, zorder=10 + i, alpha=0.8)
        tiers = [r.tier_pct] + [o.tier_pct for o in routes if o.same_as == r.tier_pct]
        lab = " = ".join(f"+{t:g} %" for t in tiers)
        h, = ax.plot(rx, ry, color=col, lw=3.6, zorder=11 + i, label=f"{lab} : {r.length_m / 1000:.2f} km, pente max {r.max_grade_pct:.1f} %, D+ {r.gain_m:.0f} m, {_fmt_time(r.time_s)}")
        handles.append(h)
    _label_streets(ax, fig, md, (x0, x1, y0, y1), None, None, max_labels=130)
    sx, sy = proj.forward(cfg["start"]["lon"], cfg["start"]["lat"]); ex, ey = proj.forward(cfg["end"]["lon"], cfg["end"]["lat"])
    for (px, py, col, lab) in [(sx, sy, C_START, "D"), (ex, ey, C_END, "A")]:
        ax.scatter([px], [py], s=330, c=col, edgecolors="white", linewidths=2, zorder=30)
        ax.text(px, py, lab, ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=31)
    leg = ax.legend(handles=handles, loc="upper left", fontsize=8, title="Variantes (palier de détour)", title_fontsize=9, framealpha=0.94)
    leg.set_zorder(50)
    ax.text(x1 - w_m * 0.005, y0 + h_m * 0.004, "Données : © Overture Maps Foundation, © OpenStreetMap contributors · Relief : AWS Terrain Tiles (Mapzen)",
            ha="right", va="bottom", fontsize=5.5, color="#333", zorder=41, path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
    _save_png(fig, out_png); plt.close(fig)
    return out_png


def render_basemap_mercator(data_dir: Path, extent_ll, out_img: Path, width_px: int = 3400, fontsize: float = 7.0) -> dict:
    """Fond de carte raster en Web Mercator (pour superposition exacte dans une carte web).

    Retourne les bornes lon/lat de l'image et sa taille en pixels.
    """
    from .geo import WebMercator
    proj = WebMercator()
    md = MapData(data_dir, proj, extent_ll)
    x0, y0 = proj.forward(extent_ll[0], extent_ll[1]); x1, y1 = proj.forward(extent_ll[2], extent_ll[3])
    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    w_m, h_m = x1 - x0, y1 - y0
    dpi = 150
    fig_w = width_px / dpi
    fig_h = fig_w * h_m / w_m
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    lat_mid = (extent_ll[1] + extent_ll[3]) / 2
    _draw_basemap(ax, fig, md, (x0, x1, y0, y1), unit_scale=WebMercator.scale(lat_mid))
    placer = _LabelPlacer(ax, fig, (x0, x1, y0, y1), fontsize=fontsize)
    span = max(w_m, h_m)
    cands = []
    for name, cl, c in _merge_by_name(md.road_names):
        inside = (c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1)
        if inside.sum() < 2:
            continue
        cc = c[inside]
        L = float(np.hypot(np.diff(cc[:, 0]), np.diff(cc[:, 1])).sum())
        if L < span * 0.012:
            continue
        cands.append((-ROAD_STYLE[cl][2], -L, name, cc))
    cands.sort(key=lambda t: (t[0], t[1]))
    n = 0
    for _, negL, name, cc in cands:
        if n >= 900:
            break
        if placer.place_on_line(name, cc, max_per_name=3 if -negL > span * 0.25 else 1, min_sep_same=span * 0.2):
            n += 1
    out_img.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_img.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=dpi, facecolor="white"); plt.close(fig)
    from PIL import Image
    im = Image.open(tmp).convert("RGB")
    if out_img.suffix.lower() == ".webp":
        im.save(out_img, "WEBP", quality=82, method=6)
    else:
        im.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(out_img, optimize=True)
    tmp.unlink()
    return {"west": extent_ll[0], "south": extent_ll[1], "east": extent_ll[2], "north": extent_ll[3], "width": im.width, "height": im.height}
