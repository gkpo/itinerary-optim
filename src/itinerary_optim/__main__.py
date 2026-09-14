"""Point d'entrée : `python -m itinerary_optim [config.yaml]` — télécharge les données si besoin, calcule
les variantes, écrit cartes PNG, GPX, GeoJSON, rapport Markdown et résumé JSON dans `output_dir`."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .config import load_config
from .directions import build_steps
from .export import slug, tier_label, write_geojson, write_gpx, write_report
from .fetch import fetch_all
from .graph import build_graph
from .render import render_overview, render_route
from .routing import Router

SOURCES_NOTE = """\
- **Réseau viaire** : Overture Maps (thème `transportation`, dérivé d'OpenStreetMap), lecture directe des fichiers
  Parquet publics sur S3 pour l'emprise du calcul ; noms de rues, classes de voies, sens uniques et restrictions
  d'accès (voies privées, interdictions vélo) sont pris en compte. Les trottoirs et passages piétons sont exclus ;
  les chemins piétons ne sont utilisés que s'ils autorisent explicitement les vélos.
- **Relief** : tuiles d'altitude Terrarium (AWS Terrain Tiles / Mapzen, zoom 14, ≈ 6 m par pixel, données
  SRTM/EU-DEM ~25–30 m). Altitude échantillonnée tous les 10 m le long des voies, lissée sur 30 m ; sous les ponts
  et dans les tunnels l'altitude est interpolée entre les extrémités.
- **Moteur de routage** : moteur maison sensible au relief (les moteurs publics BRouter / OpenRouteService /
  GraphHopper ne sont pas joignables depuis l'environnement d'exécution). Pour chaque palier, recherche
  dichotomique de la plus petite pente max admissible : plus court chemin (Dijkstra) sur le sous-graphe des
  arcs dont la pente effective ≤ seuil, vérification de la pente fenêtrée sur le tracé complet, puis relèvement de
  la pente effective des arcs responsables et nouvel essai (contraintes paresseuses) jusqu'à conformité.
- **Temps estimé** : modèle de puissance (110 W soutenus, 90 kg cycliste + vélo, Crr 0,007, SCx 0,55, 32 km/h max
  en descente, marche à 3,5 km/h sous cette vitesse) + 8 s par changement de rue.
- **Limites** : la précision du MNT (~1 m d'altitude, résolution 25–30 m) rend incertaines les pentes sur des
  tronçons très courts ; les doubles-sens cyclables ne sont pas encodés dans Overture, les sens uniques sont donc
  respectés comme pour une voiture.
"""


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cfg = load_config(argv[0] if argv else "config.yaml")
    root = cfg["_root"]
    out = root / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    paths = fetch_all(cfg)
    data_dir = paths["segment"].parent
    graph = build_graph(cfg, paths["segment"], paths["dem"], cache=root / cfg["data_dir"] / "graph_cache.pkl")
    router = Router(graph, cfg)
    src = graph.nearest_node(cfg["start"]["lon"], cfg["start"]["lat"])
    dst = graph.nearest_node(cfg["end"]["lon"], cfg["end"]["lat"])
    routes = router.solve(src, dst, cfg["detour_tiers_pct"])
    shortest_len = routes[0].length_m if routes[0].tier_pct <= 0 else min(r.length_m for r in routes)
    steps_by_tier = {}
    summary = []
    for r in routes:
        steps = build_steps(r, graph, cfg)
        steps_by_tier[r.tier_pct] = steps
        rec = {"tier_pct": r.tier_pct, "length_m": round(r.length_m), "detour_pct": round(r.detour_pct, 1), "max_grade_pct": round(r.max_grade_pct, 1),
               "steep_len_m": round(r.steep_len_m), "gain_m": round(r.gain_m), "loss_m": round(r.loss_m), "time_s": round(r.time_s),
               "same_as_tier": r.same_as, "png": None}
        if r.same_as is None:
            name = f"Variante {tier_label(r)}"
            png = out / f"{slug(r)}.png"
            print(f"[render] {png.name}", file=sys.stderr)
            render_route(r, steps, graph, cfg, data_dir, png, name, shortest_len)
            write_gpx(r, out / f"{slug(r)}.gpx", name)
            write_geojson(r, steps, out / f"{slug(r)}.geojson", name)
            rec["png"] = png.name
        summary.append(rec)
    render_overview(routes, graph, cfg, data_dir, out / "synthese.png", shortest_len)
    write_report(routes, steps_by_tier, cfg, out, shortest_len, SOURCES_NOTE)
    (out / "resume.json").write_text(json.dumps({"start": cfg["start"], "end": cfg["end"], "variants": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {len(routes)} paliers, {sum(1 for r in routes if r.same_as is None)} tracés distincts, {time.time() - t0:.0f}s -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
