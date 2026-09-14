"""Recherche d'itinéraires "pente la plus douce sous contrainte de distance".

Pour un palier de détour p, le budget vaut L0 × (1 + p/100) où L0 est la longueur du plus court chemin.
On cherche la plus petite pente g telle qu'il existe un chemin de longueur ≤ budget dont la pente max
en montée, mesurée sur une fenêtre glissante de `slope_window_m` le long du tracé complet, soit ≤ g.

Méthode : recherche dichotomique sur g. Pour un g donné, on calcule le plus court chemin en excluant les
arcs dont la "pente effective" dépasse g, on réévalue la pente fenêtrée sur le tracé assemblé, et si
une fenêtre dépasse g on relève la pente effective des arcs responsables (contraintes paresseuses) puis
on recommence, jusqu'à obtenir un tracé conforme ou dépasser le budget.

À g fixé, on affine ensuite en utilisant le budget restant pour adoucir le profil (moins de mètres
proches de la pente max, moins de tronçons raides, voies plus adaptées au vélo).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .graph import Graph
from .profile import elevation_gain, elevation_loss, steep_intervals, travel_time_s, windowed_grades


@dataclass
class Route:
    tier_pct: float
    arcs: list[tuple[int, bool]]        # (edge_idx, forward)
    nodes: list[int]
    lons: np.ndarray
    lats: np.ndarray
    dist: np.ndarray
    elev: np.ndarray
    grades: np.ndarray                  # pente fenêtrée (%) en chaque point
    sample_arc: np.ndarray              # indice (dans `arcs`) de l'arc contenant chaque point
    length_m: float
    max_grade_pct: float
    steep_len_m: float
    steep_intervals: list[tuple[float, float]]
    gain_m: float
    loss_m: float
    time_s: float
    threshold_pct: float                # pente g retenue par la recherche
    detour_pct: float = 0.0
    same_as: float | None = None        # palier dont le tracé est identique
    harshness: float = 0.0              # Σ longueur × (pente/seuil raide)^3 en montée

    @property
    def key(self) -> tuple:
        return tuple(self.nodes)


class Router:
    def __init__(self, graph: Graph, cfg: dict):
        self.g = graph
        self.cfg = cfg
        self.n = len(graph.node_xy)
        self.tol = float(cfg.get("grade_tolerance_pct", 0.1))
        arcs = []  # (u, v, edge_idx, forward)
        for i, e in enumerate(graph.edges):
            if e.allow_fwd:
                arcs.append((e.u, e.v, i, True))
            if e.allow_bwd:
                arcs.append((e.v, e.u, i, False))
        E = graph.edges
        self.arc_u = np.array([a[0] for a in arcs])
        self.arc_v = np.array([a[1] for a in arcs])
        self.arc_e = np.array([a[2] for a in arcs])
        self.arc_fwd = np.array([a[3] for a in arcs])
        self.arc_len = np.array([E[a[2]].length for a in arcs])
        self.arc_grade = np.array([E[a[2]].max_grade_fwd if a[3] else E[a[2]].max_grade_bwd for a in arcs])
        self.arc_mean = np.array([(E[a[2]].elev[-1] - E[a[2]].elev[0]) / E[a[2]].length * (100 if a[3] else -100) for a in arcs])
        self.arc_steep = np.array([E[a[2]].steep_len_fwd if a[3] else E[a[2]].steep_len_bwd for a in arcs])
        self.arc_pref = np.array([E[a[2]].pref for a in arcs])
        self.eff_grade = self.arc_grade.copy()   # pente effective, relevée par les contraintes paresseuses
        self.n_sp = 0

    # ------------------------------------------------------------------ plus court chemin générique
    def shortest(self, src: int, dst: int, cost: np.ndarray, allowed: np.ndarray) -> list[int] | None:
        """Retourne la liste d'arcs du chemin de coût minimal, ou None si inaccessible."""
        self.n_sp += 1
        ks = np.nonzero(allowed & np.isfinite(cost))[0]
        if len(ks) == 0:
            return None
        # doublons (u,v) : la matrice creuse additionnerait les poids ; on garde le meilleur arc
        order = np.lexsort((cost[ks], self.arc_v[ks], self.arc_u[ks]))
        ks = ks[order]
        key = self.arc_u[ks].astype(np.int64) * self.n + self.arc_v[ks]
        _, first = np.unique(key, return_index=True)
        ks = ks[first]
        key = key[first]
        m = csr_matrix((cost[ks] + 1e-9, (self.arc_u[ks], self.arc_v[ks])), shape=(self.n, self.n))
        d, pred = dijkstra(m, directed=True, indices=src, return_predecessors=True)
        if not np.isfinite(d[dst]):
            return None
        path_nodes = [dst]
        while path_nodes[-1] != src:
            path_nodes.append(int(pred[path_nodes[-1]]))
        path_nodes.reverse()
        pk = np.array(path_nodes[:-1], dtype=np.int64) * self.n + np.array(path_nodes[1:])
        return [int(ks[i]) for i in np.searchsorted(key, pk)]

    # ------------------------------------------------------------------ assemblage / métriques
    def assemble(self, arcs: list[int], tier_pct: float = 0.0, threshold: float = 0.0) -> Route:
        E = self.g.edges
        lons, lats, dist, elev, samp = [], [], [], [], []
        nodes = [int(self.arc_u[arcs[0]])]
        cum = 0.0
        for j, k in enumerate(arcs):
            e = E[self.arc_e[k]]
            fwd = bool(self.arc_fwd[k])
            lo, la, di, el = (e.lons, e.lats, e.dist, e.elev) if fwd else (e.lons[::-1], e.lats[::-1], e.dist[-1] - e.dist[::-1], e.elev[::-1])
            start = 0 if not lons else 1  # évite le doublon du nœud commun
            lons += list(lo[start:]); lats += list(la[start:]); dist += list(cum + di[start:]); elev += list(el[start:])
            samp += [j] * (len(lo) - start)
            cum += e.length
            nodes.append(int(self.arc_v[k]))
        lons, lats, dist, elev, samp = map(np.array, (lons, lats, dist, elev, samp))
        window = float(self.cfg["slope_window_m"])
        thr = float(self.cfg["steep_threshold_pct"])
        grades = windowed_grades(dist, elev, window)
        ivs = steep_intervals(dist, grades, thr)
        dd = np.diff(dist)
        rel = np.clip(grades[:-1], 0, None) / thr
        return Route(
            tier_pct=tier_pct, arcs=[(int(self.arc_e[k]), bool(self.arc_fwd[k])) for k in arcs], nodes=nodes,
            lons=lons, lats=lats, dist=dist, elev=elev, grades=grades, sample_arc=samp,
            length_m=float(dist[-1]), max_grade_pct=float(max(grades.max(), 0.0)),
            steep_len_m=float(sum(b - a for a, b in ivs)), steep_intervals=ivs,
            gain_m=elevation_gain(elev), loss_m=elevation_loss(elev),
            time_s=travel_time_s(dist, elev, self.cfg["rider"]), threshold_pct=threshold,
            harshness=float((dd * rel ** 3).sum()),
        )

    # ------------------------------------------------------------------ contraintes paresseuses
    def _mark(self, arcs: list[int], route: Route, g: float) -> int:
        """Relève la pente effective des arcs responsables des fenêtres > g. Retourne le nombre d'arcs marqués."""
        viol = route.grades > g + self.tol
        marked = 0
        i = 0
        n = len(viol)
        while i < n:
            if not viol[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and viol[j + 1]:
                j += 1
            gmax = float(route.grades[i:j + 1].max())
            local = sorted(set(int(a) for a in route.sample_arc[i:j + 1]))
            ks = [arcs[a] for a in local]
            steep_own = [k for k in ks if self.arc_mean[k] > g]
            targets = steep_own if steep_own else [max(ks, key=lambda k: self.arc_mean[k])]
            for k in targets:
                if self.eff_grade[k] < gmax:
                    self.eff_grade[k] = gmax
                    marked += 1
            i = j + 1
        return marked

    def constrained_path(self, src, dst, cost, g: float, budget: float, max_iter: int = 60) -> list[int] | None:
        """Chemin de coût minimal dont la pente fenêtrée max ≤ g (+tol) et la longueur ≤ budget, sinon None."""
        for _ in range(max_iter):
            path = self.shortest(src, dst, cost, self.eff_grade <= g + 1e-9)
            if path is None or self.arc_len[path].sum() > budget + 1e-6:
                return None
            route = self.assemble(path)
            if route.max_grade_pct <= g + self.tol:
                return path
            if self._mark(path, route, g) == 0:
                return None
        return None

    # ------------------------------------------------------------------ affinage à seuil fixé
    def _refine(self, src, dst, base_arcs, g, budget):
        """À pente max fixée, utilise le budget restant pour adoucir le profil.

        Coût d'un arc = longueur × (1 + α·(pente/g)^3 en montée) + λ·longueur raide + μ·longueur×(préf. − 1).
        On garde la pondération la plus forte qui respecte le budget ET améliore réellement le profil
        (moins de tronçons raides, ou critère de douceur réduit d'au moins `min_gain`).
        """
        sec = self.cfg.get("secondary", {})
        alphas = sec.get("alphas", [4.0, 2.0, 1.0, 0.5, 0.25, 0.0])
        lambdas = sec.get("lambdas", [20.0, 5.0, 1.0, 0.0])
        mus = sec.get("mus", [0.3, 0.0])
        min_gain = float(sec.get("min_gain", 0.05))
        base = self.assemble(base_arcs)
        best, best_r = base_arcs, base
        for lam in lambdas:
            for alpha in alphas:
                for mu in mus:
                    rel = np.clip(self.eff_grade, 0, None) / max(g, 0.5)
                    cost = self.arc_len * (1 + mu * (self.arc_pref - 1) + alpha * rel ** 3) + lam * self.arc_steep
                    path = self.constrained_path(src, dst, cost, g, budget)
                    if path is None:
                        continue
                    r = self.assemble(path)
                    if r.steep_len_m < best_r.steep_len_m - 1e-6 or r.harshness < best_r.harshness * (1 - min_gain):
                        best, best_r = path, r
                    break
        return best

    # ------------------------------------------------------------------ optimisation par palier
    def solve(self, src: int, dst: int, tiers: list[float]) -> list[Route]:
        base = self.shortest(src, dst, self.arc_len, np.ones(len(self.arc_len), bool))
        if base is None:
            raise RuntimeError("Arrivée inaccessible depuis le départ dans le graphe vélo.")
        base_route = self.assemble(base)
        L0 = base_route.length_m
        print(f"[route] plus court chemin : {L0:.0f} m, pente max {base_route.max_grade_pct:.1f}%", file=sys.stderr)
        results = []
        precision = float(self.cfg.get("grade_precision_pct", 0.05))
        for p in tiers:
            budget = L0 * (1 + p / 100.0)
            if p <= 0:
                r = self.assemble(base, p, base_route.max_grade_pct)
            else:
                # dichotomie sur la pente max g (le plus court chemin est toujours faisable à g = sa pente max)
                lo, hi = 0.0, base_route.max_grade_pct
                best_arcs, best_g = base, hi
                while hi - lo > precision:
                    mid = (lo + hi) / 2
                    path = self.constrained_path(src, dst, self.arc_len, mid, budget)
                    if path is not None:
                        best_arcs, best_g = path, mid
                        hi = mid
                    else:
                        lo = mid
                chosen = self._refine(src, dst, best_arcs, best_g, budget)
                r = self.assemble(chosen, p, best_g)
            r.detour_pct = (r.length_m / L0 - 1) * 100.0
            for prev in results:
                if prev.key == r.key:
                    r.same_as = prev.tier_pct
                    break
            print(f"[route] palier +{p:g}% : {r.length_m:.0f} m (+{r.detour_pct:.1f}%), pente max {r.max_grade_pct:.1f}%, "
                  f"raide {r.steep_len_m:.0f} m, D+ {r.gain_m:.0f} m, douceur {r.harshness:.0f}"
                  + (f" — identique au palier +{r.same_as:g}%" if r.same_as is not None else "")
                  + f"  [{self.n_sp} plus courts chemins]", file=sys.stderr)
            results.append(r)
        return results
