#!/usr/bin/env python3

## PATRICK'S BROODPILE STUFF USES THIS
from __future__ import annotations

import json, gzip
from pathlib import Path
from typing import Iterable
import numpy as np
from shapely.geometry import Polygon
import yaml, cv2
import os
import concurrent.futures
from tqdm import tqdm
from ethogram_vis_functions import draw_polys, build_polygon, iou_polygons

# ----------------------------
# Config & FS
# ----------------------------

def read_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

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
# Class/threshold helpers
# ----------------------------

def get_class_map(cfg: dict) -> dict[int, str]:
    if "class_names" in cfg and isinstance(cfg["class_names"], (list, tuple)):
        return {i: str(n) for i, n in enumerate(cfg["class_names"])}
    if "class_map" in cfg and isinstance(cfg["class_map"], dict):
        try:
            return {int(k): str(v) for k, v in cfg["class_map"].items()}
        except Exception:
            pass
    return {}

def resolve_classes_to_process(cfg: dict, class_map: dict[int, str]) -> set[int] | None:
    requested = cfg.get("classes_to_process")
    if not requested:
        return None
    name_to_id = {v: k for k, v in class_map.items()}
    selected: set[int] = set()
    for item in requested:
        if isinstance(item, int):
            selected.add(int(item))
        else:
            cid = name_to_id.get(str(item))
            if cid is not None:
                selected.add(int(cid))
            else:
                print(f"[WARN] Unknown class in classes_to_process: {item!r}")
    if not selected:
        print("[WARN] classes_to_process resolved to empty set; will process ALL classes.")
        return None
    return selected

def get_class_threshold(cid: int, class_map: dict[int, str], cfg: dict) -> float:
    default_thr = float(cfg.get("conf_threshold", 0.0))
    thr_map = cfg.get("class_thresholds", {}) or {}
    if cid in thr_map:
        return float(thr_map[cid])
    cname = class_map.get(cid)
    if cname is not None and cname in thr_map:
        return float(thr_map[cname])
    return default_thr

# ----------------------------
# IO helpers
# ----------------------------

def iter_jsonl(path: Path) -> Iterable[dict]:
    if path.name.startswith("."):
        return
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def write_jsonl_gz(path: Path, records: Iterable[dict]):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

def filtered_out_name(p: Path) -> str:
    if p.name.endswith(".jsonl.gz"):
        base = p.name[: -len(".jsonl.gz")]
    elif p.suffix == ".jsonl":
        base = p.stem
    else:
        base = p.stem
    return f"{base}.filtered.jsonl.gz"

# ----------------------------
# Visualization
# ----------------------------

DEFAULT_COLORS = [(0,255,0),(0,0,255),(255,0,0),(0,255,255),(255,0,255),(255,255,0)]

def find_video_for_stem(stem: str, video_dir: Path) -> Path | None:
    cand = video_dir/f"{stem}.mp4"
    return cand

def draw_prefilter(frame_bgr, polys_xy, class_ids, confs, color_map, edge_thickness):
    out = frame_bgr.copy()
    for p_xy, cid, conf in zip(polys_xy, class_ids, confs):
        if not p_xy or len(p_xy) < 3:
            continue
        color = color_map[int(cid) % len(color_map)]
        pts = np.array(p_xy, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=int(edge_thickness), lineType=cv2.LINE_AA)
        cx, cy = int(np.mean(pts[:,0,0])), int(np.mean(pts[:,0,1]))
        cv2.putText(out, f"{int(cid)}:{float(conf):.2f}", (cx, max(12, cy-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
    return out

def draw_postfilter(frame_bgr, kept_polys, kept_cls, color_map, tags, edge_thickness):
    out = frame_bgr.copy()
    for poly, cid in zip(kept_polys, kept_cls):
        if poly is None or poly.is_empty:
            continue
        color = color_map[int(cid) % len(color_map)]
        x, y = poly.exterior.xy
        pts = np.stack([x, y], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=int(edge_thickness), lineType=cv2.LINE_AA)
    for t in tags:
        cx, cy = int(round(t["cx"])), int(round(t["cy"]))
        cv2.circle(out, (cx, cy), 6, (255,255,255), -1, cv2.LINE_AA)
        cv2.putText(out,  str(t["tag_id"]), (cx+40, cy+40), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 12, cv2.LINE_AA)
    return out
# ----------------------------
# Main filter function
# ----------------------------

def filter_one_file(in_path: Path, out_path: Path, cfg: dict):
    class_map = get_class_map(cfg)
    allowed_cids = resolve_classes_to_process(cfg, class_map)
    iou_thr = float(cfg.get("iou_threshold", 0.1))
    mask_buffer_px = float(cfg.get("mask_buffer_px", 0.0))

    # Preview setup
    test_cfg = cfg.get("test", {}) or {}
    do_test = bool(test_cfg.get("enabled", False))
    video_dir = Path(test_cfg["video_dir"]).expanduser() if test_cfg.get("video_dir") else None
    fps = float(test_cfg.get("preview_fps", 10.0))
    edge_thickness = int(test_cfg.get("edge_thickness", 3))

    cap = None
    writer = None
    stem = in_path.name.replace(".jsonl.gz", "").replace(".jsonl", "")
    if do_test and video_dir:
        vid = find_video_for_stem(stem, video_dir)
        if vid:
            cap = cv2.VideoCapture(str(vid))
            if cap.isOpened():
                W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                prev_dir = Path(cfg["paths"]["output_dir"]) / "videos_filtered"
                ensure_dir(prev_dir)
                print(f"[INFO] Writing preview video to {prev_dir / f'{stem}_filtered.mp4'}")
                writer = cv2.VideoWriter(str(prev_dir / f"{stem}_filtered.mp4"), fourcc, fps, (W, H))
            else:
                print(f"[WARN] Could not open video for preview: {vid}")

    out_records = []
    images_dir = Path(cfg["paths"]["output_dir"]) / "images_filtered"
    if do_test:
        ensure_dir(images_dir)
    overlay_colors = cfg.get("overlay_colors", DEFAULT_COLORS)

    frames_iter = list(iter_jsonl(in_path))
    for rec in tqdm(frames_iter, desc=f"Filtering {in_path.name}", unit="frame"):
        cids = list(map(int, rec.get("class_ids", []) or []))
        confs = list(map(float, rec.get("confidences", []) or []))
        polys_xy = rec.get("polygons", []) or []
        frame_idx = int(rec.get("frame_idx", -1))

        raw_polys, raw_idx = [], []
        for idx, (p_xy, cid) in enumerate(zip(polys_xy, cids)):
            if allowed_cids is not None and cid not in allowed_cids:
                continue
            if not p_xy or len(p_xy) < 3:
                continue
            poly = build_polygon(p_xy, buffer_px=mask_buffer_px)
            if poly is None:
                continue
            raw_polys.append(poly)
            raw_idx.append(idx)

        kept_global_idx, kept_conf, kept_polys_merged = [], [], []
        if raw_polys:
            clsid_arr = np.array([cids[i] for i in raw_idx], dtype=int)
            confs_arr = np.array([confs[i] for i in raw_idx], dtype=float)
            for cid in np.unique(clsid_arr):
                idxs_local = np.where(clsid_arr == cid)[0]
                if idxs_local.size == 0:
                    continue
                polys_c = [raw_polys[k] for k in idxs_local]
                confs_c = confs_arr[idxs_local]
                # Allow containment to be treated as overlap; read from cfg if provided
                contain_frac = float(cfg.get("containment_fraction", cfg.get("containment_threshold", 0.5)))
                groups = build_overlap_groups_polys(polys_c, iou_thr, contain_frac=contain_frac)
                class_thr = get_class_threshold(int(cid), class_map, cfg)
                for comp in groups:
                    # Filter polygons in this group to only those above confidence threshold
                    high_conf_indices = [i for i in comp if float(confs_c[i]) >= class_thr]
                    
                    if not high_conf_indices:
                        continue
                    
                    if len(high_conf_indices) == 1:
                        # Only one polygon above threshold: keep it as-is
                        i_local = high_conf_indices[0]
                        kept_global_idx.append(raw_idx[idxs_local[i_local]])
                        kept_conf.append(float(confs_c[i_local]))
                        kept_polys_merged.append(raw_polys[idxs_local[i_local]])
                    else:
                        # Multiple polygons above threshold: merge them
                        polys_to_merge = [polys_c[i] for i in high_conf_indices]
                        confs_to_consider = [confs_c[i] for i in high_conf_indices]
                        
                        # Merge all high-confidence polygons
                        merged_poly = merge_polygons(polys_to_merge)
                        
                        # Use the confidence of the largest polygon
                        sizes = [p.area for p in polys_to_merge]
                        largest_idx = np.argmax(sizes)
                        largest_conf = confs_to_consider[largest_idx]
                        
                        # Create a pseudo-index for the merged polygon
                        kept_global_idx.append(raw_idx[idxs_local[high_conf_indices[largest_idx]]])
                        kept_conf.append(float(largest_conf))
                        kept_polys_merged.append(merged_poly)

        kept_sorted = sorted(zip(kept_global_idx, kept_conf, kept_polys_merged), key=lambda x: x[0])
        out_class_ids = [cids[i] for i, _, _ in kept_sorted]
        out_confs    = [conf for _, conf, _ in kept_sorted]
        # Convert merged Shapely polygons back to xy coordinates
        out_polys = []
        for _, _, merged_poly in kept_sorted:
            if merged_poly is None or merged_poly.is_empty:
                out_polys.append([])
                continue
            # If union produced a MultiPolygon or GeometryCollection, choose the largest polygon
            geom = merged_poly
            geom_type = getattr(geom, 'geom_type', '')
            if geom_type == 'MultiPolygon' or geom_type == 'GeometryCollection':
                parts = [p for p in getattr(geom, 'geoms', []) if getattr(p, 'area', 0) > 0]
                if not parts:
                    out_polys.append([])
                    continue
                geom = max(parts, key=lambda p: p.area)
            # Now geom is a Polygon
            try:
                x, y = geom.exterior.xy
                out_polys.append(list(zip(x, y)))
            except Exception:
                out_polys.append([])

        out_records.append({
            "frame_idx": frame_idx,
            "class_ids": out_class_ids,
            "confidences": out_confs,
            "polygons": out_polys,
        })

        if do_test and cap is not None and cap.isOpened():
            if frame_idx % 30 == 0:  # Only process every 300th frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, im = cap.read()
                if ok:
                    fr = draw_polys(im, out_polys, out_class_ids, out_confs, 
                                  color_map=overlay_colors,  # Pass the colors from config
                                  edge_thickness=edge_thickness)
                    if writer is not None:
                        writer.write(fr)
                    cv2.imwrite(str(images_dir / f"{stem}_{frame_idx:06d}.png"), fr)
                    pre_fr = draw_prefilter(im, polys_xy, cids, confs,color_map=overlay_colors, edge_thickness=edge_thickness)
                    cv2.imwrite(str(images_dir / f"{stem}_{frame_idx:06d}_prefilt.png"), pre_fr)

    ensure_dir(Path(cfg["paths"]["output_dir"]))
    write_jsonl_gz(out_path, out_records)
    if writer: writer.release()
    if cap: cap.release()

# ----------------------------
# Entry point
# ----------------------------

def main():
    cfg = read_config("/home/tracking/Dropbox (Dropbox @RU)/Jina_shared/scripts/broodpile/config_filter.yaml")
    in_dir = Path(cfg["paths"]["prediction_dir"]).expanduser()
    out_dir = Path(cfg["paths"]["output_dir"]).expanduser()
    ensure_dir(out_dir)

    files = sorted([p for p in in_dir.iterdir() if (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz"))])

    ## Not working??
    if cfg.get("checkpoint"):
        checkpoint = str(cfg.get("checkpoint"))
        # allow numeric checkpoint (index) or substring/stem match
        found = False
        for i, p in enumerate(files):
            if checkpoint == p.name:
                files = files[i:]
                print(f"[INFO] Resuming from checkpoint '{checkpoint}' at index {i}: {files[0].name}")
                found = True
                break
        if not found:
            print(f"[WARN] checkpoint '{checkpoint}' not found in file list; processing all files")
    if not files:
        print(f"[WARN] No JSONL files in {in_dir}")
        return

    # Parallel execution config
    parallel_cfg = cfg.get("parallel", {}) or {}
    parallel_enabled = bool(parallel_cfg.get("enabled", True))
    max_workers = int(parallel_cfg.get("workers", os.cpu_count() or 1))

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
