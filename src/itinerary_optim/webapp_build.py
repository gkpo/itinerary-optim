"""Précalcul pour la webapp : frontière détour → tracé optimal (pour plusieurs fenêtres de pente),
fond de carte raster Web Mercator, données JS.

Usage : PYTHONPATH=src python -m itinerary_optim.webapp_build [config.yaml]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .directions import build_steps
from .fetch import fetch_all
from .graph import build_graph
from .render import render_basemap_mercator
from .routing import Router


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cfg = load_config(argv[0] if argv else "config.yaml")
    root = cfg["_root"]
    wcfg = cfg["webapp"]
    out = root / "webapp"
    out.mkdir(exist_ok=True)
    paths = fetch_all(cfg)
    data_dir = paths["segment"].parent
    tiers = list(range(0, int(wcfg["max_detour_pct"]) + 1, int(wcfg["detour_step_pct"])))
    routes_rec: dict[tuple, dict] = {}
    frontier: dict[int, list] = {}
    t0 = time.time()
    for w in wcfg["slope_windows_m"]:
        cfg_w = dict(cfg); cfg_w["slope_window_m"] = w
        g = build_graph(cfg_w, paths["segment"], paths["dem"], cache=root / cfg["data_dir"] / f"graph_cache_w{w}.pkl")
        router = Router(g, cfg_w)
        src = g.nearest_node(cfg["start"]["lon"], cfg["start"]["lat"]); dst = g.nearest_node(cfg["end"]["lon"], cfg["end"]["lat"])
        routes = router.solve(src, dst, tiers)
        frontier[w] = []
        for r in routes:
            if r.key not in routes_rec:
                steps = build_steps(r, g, cfg_w)
                routes_rec[r.key] = {
                    "id": len(routes_rec),
                    "length_m": round(r.length_m, 1),
                    "lon": [round(float(v), 6) for v in r.lons], "lat": [round(float(v), 6) for v in r.lats],
                    "elev": [round(float(v), 1) for v in r.elev], "dist": [round(float(v), 1) for v in r.dist],
                    "steps": [{"i": s.instruction, "n": s.name, "L": round(s.length_m), "d0": round(s.d_start), "d1": round(s.d_end),
                               "a": int(s.idx_start), "b": int(s.idx_end)} for s in steps],
                }
            frontier[w].append({"pct": r.tier_pct, "id": routes_rec[r.key]["id"]})
        print(f"[webapp] fenêtre {w} m : {len(set(f['id'] for f in frontier[w]))} tracés distincts, {time.time() - t0:.0f}s", file=sys.stderr)
    recs = sorted(routes_rec.values(), key=lambda d: d["id"])
    lons = np.concatenate([np.array(d["lon"]) for d in recs]); lats = np.concatenate([np.array(d["lat"]) for d in recs])
    pad_lon = (lons.max() - lons.min()) * 0.06 + 0.004; pad_lat = (lats.max() - lats.min()) * 0.06 + 0.003
    ext = (float(lons.min() - pad_lon), float(lats.min() - pad_lat), float(lons.max() + pad_lon), float(lats.max() + pad_lat))
    print(f"[webapp] fond de carte {ext}", file=sys.stderr)
    bm = render_basemap_mercator(data_dir, ext, out / "basemap.webp", width_px=int(wcfg["basemap_width_px"]))
    data = {
        "start": cfg["start"], "end": cfg["end"], "steep_threshold_pct": cfg["steep_threshold_pct"],
        "rider": cfg["rider"], "windows": wcfg["slope_windows_m"], "frontier": {str(w): f for w, f in frontier.items()},
        "routes": recs, "basemap": bm, "generated": time.strftime("%Y-%m-%d"),
    }
    (out / "data.js").write_text("window.ROUTE_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    assemble_index(out)
    print(f"[webapp] {len(recs)} tracés distincts au total -> {out}/data.js ({(out / 'data.js').stat().st_size / 1e6:.1f} Mo), "
          f"basemap {bm['width']}x{bm['height']} ({(out / 'basemap.webp').stat().st_size / 1e6:.1f} Mo)", file=sys.stderr)


if __name__ == "__main__":
    main()


def assemble_index(webapp_dir: Path) -> Path:
    """Assemble webapp/index.html : index.src.html + feuille de style Leaflet inlinée (CSP des artefacts)."""
    src = (webapp_dir / "index.src.html").read_text(encoding="utf-8")
    css = (webapp_dir / "leaflet.css").read_text(encoding="utf-8")
    out = webapp_dir / "index.html"
    out.write_text(src.replace("/*LEAFLET_CSS*/", css), encoding="utf-8")
    return out
