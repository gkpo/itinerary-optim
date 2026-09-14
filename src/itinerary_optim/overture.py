"""Extraction d'une emprise depuis les fichiers Parquet publics d'Overture Maps (S3, accès anonyme).

Les fichiers sont volumineux (plusieurs centaines de Mo chacun) mais triés spatialement : on lit
seulement les pieds de page (métadonnées) puis les row groups dont les statistiques bbox
intersectent l'emprise demandée.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

BUCKET = "https://overturemaps-us-west-2.s3.amazonaws.com"


def list_files(release: str, theme: str, type_: str) -> list[str]:
    prefix = f"release/{release}/theme={theme}/type={type_}/"
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        root = ElementTree.fromstring(r.content)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        keys += [c.find("s3:Key", ns).text for c in root.findall("s3:Contents", ns)]
        nxt = root.find("s3:NextContinuationToken", ns)
        if nxt is None:
            break
        token = nxt.text
    return [f"{BUCKET}/{k}" for k in keys if k.endswith(".parquet")]


def _matching_row_groups(md, bbox):
    xmin, ymin, xmax, ymax = bbox
    names = [md.row_group(0).column(j).path_in_schema for j in range(md.row_group(0).num_columns)]
    ix = {n: names.index(n) for n in ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax")}
    out = []
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        s = {n: rg.column(j).statistics for n, j in ix.items()}
        if any(v is None or not v.has_min_max for v in s.values()):
            out.append(i)
            continue
        if s["bbox.xmin"].min > xmax or s["bbox.xmax"].max < xmin:
            continue
        if s["bbox.ymin"].min > ymax or s["bbox.ymax"].max < ymin:
            continue
        out.append(i)
    return out


def _extract_file(url: str, bbox, columns):
    fs = fsspec.filesystem("https")
    xmin, ymin, xmax, ymax = bbox
    tables = []
    for attempt in range(4):
        try:
            with fs.open(url, "rb", block_size=4 << 20) as f:
                pf = pq.ParquetFile(f)
                rgs = _matching_row_groups(pf.metadata, bbox)
                for i in rgs:
                    t = pf.read_row_group(i, columns=columns)
                    b = t.column("bbox")
                    mask = pc.and_(
                        pc.and_(pc.less_equal(pc.struct_field(b, "xmin"), xmax),
                                pc.greater_equal(pc.struct_field(b, "xmax"), xmin)),
                        pc.and_(pc.less_equal(pc.struct_field(b, "ymin"), ymax),
                                pc.greater_equal(pc.struct_field(b, "ymax"), ymin)),
                    )
                    t = t.filter(mask)
                    if t.num_rows:
                        tables.append(t)
            return tables
        except Exception as e:  # réseau instable : on réessaie
            if attempt == 3:
                raise
            print(f"  retry {url.rsplit('/', 1)[-1][:20]}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
            tables = []
    return tables


def extract(release: str, theme: str, type_: str, bbox, columns, dest: Path, workers: int = 8) -> pa.Table:
    """Télécharge les entités de `theme/type_` intersectant `bbox` (xmin, ymin, xmax, ymax) vers `dest`."""
    if dest.exists():
        return pq.read_table(dest)
    urls = list_files(release, theme, type_)
    print(f"[overture] {theme}/{type_}: {len(urls)} fichiers, emprise {bbox}", file=sys.stderr)
    t0 = time.time()
    tables = []
    with ThreadPoolExecutor(workers) as ex:
        for res in ex.map(lambda u: _extract_file(u, bbox, columns), urls):
            tables += res
    table = pa.concat_tables(tables, promote_options="default") if tables else pa.table({c: [] for c in columns})
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dest, compression="zstd")
    print(f"[overture] {theme}/{type_}: {table.num_rows} entités en {time.time() - t0:.0f}s -> {dest}", file=sys.stderr)
    return table
