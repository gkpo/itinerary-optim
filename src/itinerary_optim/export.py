"""Exports : GPX, GeoJSON, rapport Markdown, tableau récapitulatif."""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from .directions import Step, format_steps
from .routing import Route


def tier_label(r: Route) -> str:
    return "+0 % (trajet le plus court)" if r.tier_pct <= 0 else f"+{r.tier_pct:g} % de détour"


def slug(r: Route) -> str:
    return f"variante_+{int(round(r.tier_pct)):02d}pct"


def fmt_time(s: float) -> str:
    m = int(round(s / 60))
    return f"{m // 60} h {m % 60:02d}" if m >= 60 else f"{m} min"


def write_gpx(route: Route, path: Path, name: str):
    pts = "\n".join(f'      <trkpt lat="{la:.6f}" lon="{lo:.6f}"><ele>{el:.1f}</ele></trkpt>'
                    for lo, la, el in zip(route.lons, route.lats, route.elev))
    path.write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="itinerary-optim" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <name>{escape(name)}</name>
    <trkseg>
{pts}
    </trkseg>
  </trk>
</gpx>
''', encoding="utf-8")


def write_geojson(route: Route, steps: list[Step], path: Path, name: str):
    feats = [{
        "type": "Feature",
        "properties": {"name": name, "tier_pct": route.tier_pct, "length_m": round(route.length_m), "detour_pct": round(route.detour_pct, 1),
                       "max_grade_pct": round(route.max_grade_pct, 1), "steep_len_m": round(route.steep_len_m), "gain_m": round(route.gain_m),
                       "loss_m": round(route.loss_m), "time_s": round(route.time_s)},
        "geometry": {"type": "LineString", "coordinates": [[round(float(lo), 6), round(float(la), 6), round(float(el), 1)]
                                                            for lo, la, el in zip(route.lons, route.lats, route.elev)]},
    }]
    for a, b in route.steep_intervals:
        m = (route.dist >= a - 1e-6) & (route.dist <= b + 1e-6)
        if m.sum() >= 2:
            feats.append({"type": "Feature", "properties": {"kind": "steep", "from_m": round(a), "to_m": round(b)},
                          "geometry": {"type": "LineString", "coordinates": [[round(float(lo), 6), round(float(la), 6)] for lo, la in zip(route.lons[m], route.lats[m])]}})
    for i, s in enumerate(steps, 1):
        feats.append({"type": "Feature", "properties": {"kind": "step", "n": i, "instruction": s.instruction, "name": s.name, "length_m": round(s.length_m),
                                                        "max_grade_pct": round(s.max_grade, 1), "notes": s.notes},
                      "geometry": {"type": "Point", "coordinates": [round(s.lon, 6), round(s.lat, 6)]}})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False), encoding="utf-8")


def summary_table(routes: list[Route], cfg: dict, shortest_len: float) -> str:
    thr = cfg["steep_threshold_pct"]; win = cfg["slope_window_m"]
    head = (f"| Palier | Distance | Détour réel | Pente max montée ({win:g} m) | Longueur > {thr:g} % | D+ | D− | Temps estimé | Tracé |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for r in routes:
        trace = f"identique au palier +{r.same_as:g} %" if r.same_as is not None else f"[carte]({slug(r)}.png)"
        rows.append(f"| +{r.tier_pct:g} % | {r.length_m / 1000:.2f} km | {r.detour_pct:+.1f} % | **{r.max_grade_pct:.1f} %** | {r.steep_len_m:.0f} m | "
                    f"+{r.gain_m:.0f} m | −{r.loss_m:.0f} m | {fmt_time(r.time_s)} | {trace} |")
    return head + "\n".join(rows) + "\n"


def write_report(routes: list[Route], steps_by_tier: dict[float, list[Step]], cfg: dict, out_dir: Path, shortest_len: float,
                 sources_note: str) -> Path:
    thr = cfg["steep_threshold_pct"]; win = cfg["slope_window_m"]
    md = [f"# Itinéraires vélo à pente douce\n",
          f"**Départ :** {cfg['start']['name']} ({cfg['start']['lat']}, {cfg['start']['lon']})  ",
          f"**Arrivée :** {cfg['end']['name']} ({cfg['end']['lat']}, {cfg['end']['lon']})  ",
          f"**Mode :** vélo classique · **Paliers :** {', '.join(f'+{t:g} %' for t in cfg['detour_tiers_pct'])} · "
          f"**Pente max mesurée sur fenêtre de {win:g} m** · **Seuil « raide » : {thr:g} %**\n",
          "## Tableau récapitulatif\n", summary_table(routes, cfg, shortest_len),
          "\nPour chaque palier, l'itinéraire retenu est celui dont la pente maximale en montée (fenêtre glissante de "
          f"{win:g} m) est la plus faible parmi les tracés dont la longueur reste dans le budget du palier ; à pente max égale, "
          f"on privilégie le tracé qui minimise les tronçons > {thr:g} % puis les mètres proches de la pente max.\n",
          "![Synthèse](synthese.png)\n"]
    for r in routes:
        md.append(f"\n## Palier {tier_label(r)}\n")
        if r.same_as is not None:
            md.append(f"Le tracé optimal de ce palier est **identique à celui du palier +{r.same_as:g} %** "
                      f"({r.length_m / 1000:.2f} km, pente max {r.max_grade_pct:.1f} %) : le budget de distance supplémentaire "
                      "ne permet pas de trouver un tracé à pente maximale plus faible. Pas d'image dupliquée.\n")
            continue
        md.append(f"- Distance : **{r.length_m / 1000:.2f} km** (détour {r.detour_pct:+.1f} % par rapport au plus court, {shortest_len / 1000:.2f} km)\n"
                  f"- Pente max en montée (fenêtre {win:g} m) : **{r.max_grade_pct:.1f} %**\n"
                  f"- Tronçons > {thr:g} % : **{r.steep_len_m:.0f} m** cumulés\n"
                  f"- Dénivelé : **+{r.gain_m:.0f} m** / −{r.loss_m:.0f} m\n"
                  f"- Temps estimé : **{fmt_time(r.time_s)}**\n"
                  f"- Fichiers : [{slug(r)}.png]({slug(r)}.png) · [GPX]({slug(r)}.gpx) · [GeoJSON]({slug(r)}.geojson)\n\n"
                  f"![{tier_label(r)}]({slug(r)}.png)\n\n### Feuille de route\n")
        md.append(format_steps(steps_by_tier[r.tier_pct], r, cfg) + "\n")
    md.append("\n## Sources et méthode\n" + sources_note)
    path = out_dir / "rapport.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path
