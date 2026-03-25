#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json, gzip
from pathlib import Path
from typing import Iterable
import numpy as np
from shapely.geometry import Polygon
import yaml, cv2
import concurrent.futures
from tqdm import tqdm
from ethogram_vis_functions import build_polygon, iou_polygons

# ----------------------------
# Config & FS
# ----------------------------

def read_config(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Geometry helpers
# ----------------------------

def build_overlap_groups_polys(polys: list[Polygon], iou_thresh: float, contain_frac: float = 0.5) -> list[list[int]]:
    n = len(polys)
    if n == 0:
        return []
    visited = [False] * n
    groups: list[list[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = [i]
        while stack:
            a = stack.pop()
            pa = polys[a]
            for j in range(n):
                if visited[j] or j == a:
                    continue
                pb = polys[j]
                # Treat containment as overlap: if intersection covers a large fraction of the smaller polygon
                try:
                    inter_area = pa.intersection(pb).area
                except Exception:
                    inter_area = 0.0
                smaller_area = min(pa.area if pa is not None else 0.0, pb.area if pb is not None else 0.0)
                contain_overlap = (smaller_area > 0) and (inter_area / smaller_area >= float(contain_frac))
                if iou_polygons(pa, pb) >= iou_thresh or contain_overlap:
                    visited[j] = True
                    stack.append(j)
                    comp.append(j)
        groups.append(comp)
    return groups

def merge_polygons(polys: list[Polygon]) -> Polygon | None:
    """Merge a list of polygons using union. Return the largest bounding merged result."""
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    merged = polys[0]
    for p in polys[1:]:
        merged = merged.union(p)
    return merged

# ----------------------------
# Threshold helpers
# ----------------------------

def get_conf_threshold(cfg: dict) -> float:
    if "class_thresholds" in cfg:
        thr_map = cfg["class_thresholds"]
        if not isinstance(thr_map, dict) or not thr_map:
            raise ValueError("class_thresholds must be a non-empty mapping")
        return float(next(iter(thr_map.values())))
    if "conf_threshold" in cfg:
        return float(cfg["conf_threshold"])
    raise ValueError("Config must define either class_thresholds or conf_threshold")

# ----------------------------
# IO helpers
# ----------------------------

def is_jsonl_gz_path(path: Path) -> bool:
    return path.name.endswith(".jsonl.gz")


def iter_jsonl(path: Path) -> Iterable[dict]:
    if not is_jsonl_gz_path(path):
        raise ValueError(f"Expected .jsonl.gz input, got {path}")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def write_jsonl_gz(path: Path, records: Iterable[dict]):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

def filtered_out_name(p: Path) -> str:
    if not is_jsonl_gz_path(p):
        raise ValueError(f"Expected .jsonl.gz input, got {p}")
    base = p.name[: -len(".jsonl.gz")]
    return f"{base}.filtered.jsonl.gz"

# ----------------------------
# Main filter function
# ----------------------------

def filter_one_file(in_path: Path, out_path: Path, cfg: dict):
    iou_thr = float(cfg["iou_threshold"])
    conf_thr = get_conf_threshold(cfg)
    contain_frac = float(cfg["containment_fraction"])
    if not is_jsonl_gz_path(in_path):
        raise ValueError(f"Expected .jsonl.gz input, got {in_path}")

    out_records = []

    frames_iter = list(iter_jsonl(in_path))
    for rec in tqdm(frames_iter, desc=f"Filtering {in_path.name}", unit="frame"):
        cids = list(map(int, rec["class_ids"]))
        confs = list(map(float, rec["confidences"]))
        polys_xy = rec["polygons"]
        frame_idx = int(rec["frame_idx"])

        raw_polys, raw_idx = [], []
        for idx, p_xy in enumerate(polys_xy):
            poly = build_polygon(p_xy)
            raw_polys.append(poly)
            raw_idx.append(idx)

        kept_global_idx, kept_conf, kept_polys_merged = [], [], []
        if raw_polys:
            confs_arr = np.array([confs[i] for i in raw_idx], dtype=float)
            groups = build_overlap_groups_polys(raw_polys, iou_thr, contain_frac=contain_frac)
            for comp in groups:
                high_conf_indices = [i for i in comp if float(confs_arr[i]) >= conf_thr]

                if not high_conf_indices:
                    continue

                if len(high_conf_indices) == 1:
                    i_local = high_conf_indices[0]
                    kept_global_idx.append(raw_idx[i_local])
                    kept_conf.append(float(confs_arr[i_local]))
                    kept_polys_merged.append(raw_polys[i_local])
                else:
                    polys_to_merge = [raw_polys[i] for i in high_conf_indices]
                    confs_to_consider = [confs_arr[i] for i in high_conf_indices]

                    merged_poly = merge_polygons(polys_to_merge)

                    sizes = [p.area for p in polys_to_merge]
                    largest_idx = np.argmax(sizes)
                    largest_conf = confs_to_consider[largest_idx]

                    kept_global_idx.append(raw_idx[high_conf_indices[largest_idx]])
                    kept_conf.append(float(largest_conf))
                    kept_polys_merged.append(merged_poly)

        kept_sorted = sorted(zip(kept_global_idx, kept_conf, kept_polys_merged), key=lambda x: x[0])
        out_class_ids = [cids[i] for i, _, _ in kept_sorted]
        out_confs    = [conf for _, conf, _ in kept_sorted]
        # Convert merged Shapely polygons back to xy coordinates
        out_polys = []
        for _, _, merged_poly in kept_sorted:
            if merged_poly is None or merged_poly.is_empty:
                raise ValueError("Merged polygon is empty")
            # If union produced a MultiPolygon or GeometryCollection, choose the largest polygon
            geom = merged_poly
            geom_type = getattr(geom, 'geom_type', '')
            if geom_type == 'MultiPolygon' or geom_type == 'GeometryCollection':
                parts = [p for p in getattr(geom, 'geoms', []) if getattr(p, 'area', 0) > 0]
                if not parts:
                    raise ValueError("Merged geometry contained no polygon parts")
                geom = max(parts, key=lambda p: p.area)
            # Now geom is a Polygon
            x, y = geom.exterior.xy
            out_polys.append(list(zip(x, y)))

        out_records.append({
            "frame_idx": frame_idx,
            "class_ids": out_class_ids,
            "confidences": out_confs,
            "polygons": out_polys,
        })

    ensure_dir(Path(cfg["paths"]["output_dir"]))
    write_jsonl_gz(out_path, out_records)

# ----------------------------
# Entry point
# ----------------------------

def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).with_name("config_filter.yaml")

    parser = argparse.ArgumentParser(
        description="Filter YOLO segmentation JSONL outputs using a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to filter config YAML (default: {default_config})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = read_config(args.config.expanduser().resolve())
    in_dir = Path(cfg["paths"]["prediction_dir"]).expanduser()
    out_dir = Path(cfg["paths"]["output_dir"]).expanduser()
    ensure_dir(out_dir)

    files = sorted([p for p in in_dir.iterdir() if p.name.endswith(".jsonl.gz")])

    if "checkpoint" in cfg:
        checkpoint = str(cfg["checkpoint"])
        found = False
        for i, p in enumerate(files):
            if checkpoint == p.name:
                files = files[i:]
                print(f"[INFO] Resuming from checkpoint '{checkpoint}' at index {i}: {files[0].name}")
                found = True
                break
        if not found:
            raise ValueError(f"checkpoint '{checkpoint}' not found in file list")
    if not files:
        print(f"[WARN] No JSONL files in {in_dir}")
        return

    # Parallel execution config
    parallel_cfg = cfg["parallel"]
    parallel_enabled = bool(parallel_cfg["enabled"])
    max_workers = int(parallel_cfg["workers"])

    if parallel_enabled and len(files) > 1:
        print(f"[INFO] Running filtering in parallel with {max_workers} workers")
        # Submit tasks to worker processes. Each worker will call filter_one_file.
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            fut_to_path = {
                ex.submit(filter_one_file, p, out_dir / filtered_out_name(p), cfg): p
                for p in files
            }
            for fut in tqdm(concurrent.futures.as_completed(fut_to_path), total=len(fut_to_path), desc="Filtering all videos (parallel)"):
                p = fut_to_path[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"[ERROR] Processing {p} failed: {e}")
    else:
        for p in tqdm(files, desc="Filtering all videos"):
            out_path = out_dir / filtered_out_name(p)
            filter_one_file(p, out_path, cfg)

if __name__ == "__main__":
    main()
