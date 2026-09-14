# itinerary-optim — itinéraires vélo à pente douce

Calcule, entre deux adresses, l'itinéraire vélo dont la **pente maximale en montée** est la plus faible
possible pour chaque **palier de détour** autorisé (0 %, +10 %, … par rapport au trajet le plus court),
puis produit pour chaque variante une carte PNG (fond de carte avec rues, tracé, tronçons raides
surlignés, légende incrustée, profil altimétrique), une feuille de route texte, un GPX, un GeoJSON et un
tableau récapitulatif.

**Résultats du cas configuré (Noisy-le-Sec → Montreuil) : voir [`output/rapport.md`](output/rapport.md).**

## Utilisation

```bash
pip install -r requirements.txt
# éditer config.yaml (adresses, paliers, fenêtre de pente, seuil "raide", modèle cycliste…)
PYTHONPATH=src python -m itinerary_optim config.yaml
```

Le premier lancement télécharge les données de l'emprise (≈ 60 Mo, quelques dizaines de secondes) dans
`data/` ; les lancements suivants réutilisent ce cache (supprimer `data/graph_cache.pkl` après un changement
de paramètre du graphe : classes de voies, fenêtre, seuil, pas d'échantillonnage).

Sorties dans `output/` :

| Fichier | Contenu |
|---|---|
| `rapport.md` | tableau récapitulatif + feuille de route détaillée de chaque variante |
| `variante_+XXpct.png` | carte de la variante (une seule image si plusieurs paliers donnent le même tracé) |
| `variante_+XXpct.gpx` / `.geojson` | tracé avec altitudes, tronçons raides, étapes |
| `synthese.png` | tous les tracés distincts superposés |
| `resume.json` | métriques de toutes les variantes (lisible par machine) |

## Webapp interactive

**En ligne : https://gkpo.github.io/itinerary-optim/** (déployée par GitHub Actions depuis `webapp/`).

`webapp/index.html` est une application autonome (mobile et bureau) : curseur de **détour maximal autorisé**
qui change le tracé en direct, choix de la **fenêtre de pente** (50 / 100 / 200 m, tracés recalculés pour
chacune), **seuil « raide »** et **puissance du cycliste** réglables, profil altimétrique interactif lié à la
carte, courbe « pente max atteignable selon le détour », feuille de route, superposition du plus court chemin
et de toutes les variantes.

```bash
PYTHONPATH=src python -m itinerary_optim.webapp_build config.yaml   # précalcul (≈ 15 min) + fond de carte
python -m http.server -d webapp 8000                                # puis ouvrir http://localhost:8000
```

Le précalcul résout le problème pour chaque détour de 0 à `webapp.max_detour_pct` (pas `detour_step_pct`)
et chaque fenêtre de `webapp.slope_windows_m`, dédoublonne les tracés et écrit `webapp/data.js` ; le fond
de carte `webapp/basemap.webp` est rendu en Web Mercator pour se superposer exactement à la carte Leaflet
(embarquée dans le dossier, aucun accès réseau nécessaire hormis les polices).

## Paramètres (`config.yaml`)

- `start`, `end` : coordonnées et libellés.
- `detour_tiers_pct` : paliers de détour (% de distance en plus du plus court chemin).
- `slope_window_m` : fenêtre glissante (m) de calcul de la pente max (100 m par défaut).
- `steep_threshold_pct` : seuil des tronçons « raides » surlignés et cumulés (6 % par défaut).
- `road_classes` : classes de voies Overture autorisées et leur facteur de préférence.
- `rider` : modèle cycliste pour le temps estimé.
- `secondary` : pondérations de l'affinage à pente max fixée.
- `data_margin_deg`, `overture_release`, `dem_zoom` : emprise et sources de données.

## Données et méthode

- **Voirie** : [Overture Maps](https://overturemaps.org) (thème `transportation`, dérivé d'OpenStreetMap),
  lu directement dans les fichiers Parquet publics sur S3 (`src/itinerary_optim/overture.py`, lecture des
  seuls row groups intersectant l'emprise). Classes de voies, noms, sens uniques et restrictions d'accès
  (voies privées, interdictions vélo) sont respectés ; trottoirs et passages piétons exclus.
- **Relief** : tuiles Terrarium des [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/)
  (zoom 14, ≈ 6 m/pixel, données sources SRTM / EU-DEM). Altitude échantillonnée tous les 10 m le long des
  voies et lissée sur 30 m ; interpolation entre extrémités sous les ponts et dans les tunnels.
- **Fond de carte** : rendu maison (matplotlib) à partir d'Overture : occupation du sol, eau, bâtiments,
  voirie stylée par classe, voies ferrées, noms de rues.
- **Routage** (`src/itinerary_optim/routing.py`) : le palier 0 % est le plus court chemin (Dijkstra sur la
  longueur). Pour un palier p, budget = L0 × (1 + p/100). Recherche dichotomique de la plus petite pente g
  telle qu'un chemin de longueur ≤ budget ait une pente fenêtrée max ≤ g : à chaque essai, plus court chemin
  sur les arcs de pente effective ≤ g, réévaluation de la pente sur fenêtre glissante le long du tracé
  assemblé, et si une fenêtre dépasse g, relèvement de la pente effective des arcs responsables puis nouvel
  essai (« contraintes paresseuses »). À g fixé, le budget restant sert à adoucir le profil (moins de mètres
  proches de la pente max, moins de tronçons > seuil, voies plus adaptées) si le gain est significatif.
  Deux paliers produisant le même tracé sont signalés au lieu d'être dupliqués.

### Pourquoi pas BRouter / OpenRouteService / GraphHopper ?

Ces services (ainsi que les serveurs de tuiles OSM et Overpass) ne sont pas joignables depuis l'environnement
d'exécution utilisé pour produire les résultats ; seuls les jeux de données ouverts hébergés sur S3 l'étaient.
Le moteur maison cible en revanche exactement le critère demandé (minimiser la pente max sur fenêtre de 100 m
sous contrainte de distance), ce qu'un moteur à coût pondéré n'optimise qu'indirectement.

### Limites

- Précision du MNT (~1 m en altitude, 25–30 m de résolution native) : pentes incertaines sur des tronçons
  très courts ; la fenêtre de 100 m et le lissage atténuent ce bruit.
- Les doubles-sens cyclables ne sont pas encodés dans Overture : les sens uniques sont respectés comme pour
  une voiture (les itinéraires restent valides, éventuellement un peu plus longs).
- Le temps estimé est un modèle physique simple (puissance constante), sans feux ni trafic au-delà d'une
  pénalité forfaitaire par changement de rue.
