#!/usr/bin/env python3
"""
Combine filtered JSONL detections with colony masks.

For each colony, outputs a CSV with per-frame statistics:
  - frame_idx
  - jsonl_source (input filename)
  - row_in_jsonl (record index in that file)
  - polygon_count (# of polygons within colony mask)
  - polygon_areas (list of areas, one per polygon)
  - total_polygon_area (sum of all polygon areas)
  - convex_hull_area (area of outer convex hull enclosing all polygons)
  - hull_centroid_x, hull_centroid_y (center of hull)
"""

import json
import gzip
import pickle
import numpy as np
from pathlib import Path
from typing import Iterable
import csv
from collections import defaultdict
from shapely.geometry import Polygon as ShapelyPolygon
import cv2
from tqdm import tqdm

from process_colony_masks import Colony

DEFAULT_OVERLAY_COLORS = [(0,255,0),(0,0,255),(255,0,0),(0,255,255),(255,0,255),(255,255,0)]

def load_colonies(colonies_pkl_path: Path):
    """Load Colony objects from pickle."""
    with open(colonies_pkl_path, "rb") as f:
        colonies = pickle.load(f)
    return colonies


def iter_jsonl_with_source(jsonl_path: Path) -> Iterable[tuple[dict, int]]:
    """Iterate JSONL records with row index (0-based)."""
    if jsonl_path.suffixes != [".jsonl", ".gz"]:
        raise ValueError(f"Expected a .jsonl.gz file, got: {jsonl_path}")
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            s = line.strip()
            if s:
                yield json.loads(s), row_idx


def point_in_mask(pt: tuple, mask: np.ndarray) -> bool:
    """Check if (x, y) point is within mask (mask > 0)."""
    x, y = int(round(pt[0])), int(round(pt[1]))
    h, w = mask.shape
    if 0 <= x < w and 0 <= y < h:
        return mask[y, x] > 0
    return False


def polygon_center_in_mask(poly_xy: list, mask: np.ndarray) -> bool:
    """Check if polygon's centroid is within the mask."""
    if len(poly_xy) < 3:
        raise ValueError(f"Polygon must contain at least 3 points, got {len(poly_xy)}")
    cx = np.mean([p[0] for p in poly_xy])
    cy = np.mean([p[1] for p in poly_xy])
    return point_in_mask((cx, cy), mask)


def polygon_area_from_xy(poly_xy: list) -> float:
    """Compute area of polygon given list of (x, y) tuples using shoelace formula."""
    if len(poly_xy) < 3:
        raise ValueError(f"Polygon must contain at least 3 points, got {len(poly_xy)}")
    poly = ShapelyPolygon(poly_xy)
    if not poly.is_valid:
        raise ValueError("Encountered invalid polygon geometry")
    return poly.area


def compute_convex_hull_area(polys_xy: list) -> tuple[float, tuple[float, float]]:
    """
    Given list of polygons (each a list of (x, y) tuples),
    compute the area of their convex hull and the centroid.
    """
    all_points = []
    for poly in polys_xy:
        if poly and len(poly) >= 3:
            all_points.extend(poly)
    
    if len(all_points) < 3:
        return 0.0, (0.0, 0.0)
    
    pts = np.array(all_points, dtype=np.float32)
    hull = cv2.convexHull(pts)
    
    # Area via shoelace
    hull_area = cv2.contourArea(hull)
    
    # Centroid via moments
    M = cv2.moments(hull)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = 0.0, 0.0
    
    return float(hull_area), (float(cx), float(cy))


def process_frame_for_colony(rec: dict, colony, jsonl_source: str, row_idx: int, frame_offset: int = 0):
    """
    Extract polygons in this frame that fall within the colony mask.
    Return metrics dict or None if no polygons in mask.
    
    Args:
        rec: Frame record from JSONL
        colony: Colony object
        jsonl_source: Name of source JSONL file
        row_idx: Row index in source JSONL
        frame_offset: Cumulative frame offset from all previous files
    """
    frame_idx_local = int(rec["frame_idx"])
    frame_idx_absolute = frame_idx_local + frame_offset
    polys_xy = rec["polygons"]
    
    # Filter polygons within colony mask
    polys_in_mask = []
    for poly_xy in polys_xy:
        if polygon_center_in_mask(poly_xy, colony.mask):
            polys_in_mask.append(poly_xy)
    
    if not polys_in_mask:
        return None
    
    # Compute metrics
    poly_count = len(polys_in_mask)
    poly_areas = [polygon_area_from_xy(p) for p in polys_in_mask]
    total_area = sum(poly_areas)
    hull_area, (hull_cx, hull_cy) = compute_convex_hull_area(polys_in_mask)
    
    
    return {
        "frame_idx": int(frame_idx_absolute),
        "jsonl_source": str(jsonl_source),
        "row_in_jsonl": int(row_idx),
        "polygon_count": int(poly_count),
        "polygon_areas": poly_areas,
        "total_polygon_area": float(total_area),
        "convex_hull_area": float(hull_area),
        "hull_centroid_x": float(hull_cx),
        "hull_centroid_y": float(hull_cy),
    }


def compute_convex_hull_points(polys_xy: list):
    """Return convex hull points (Nx2 float32) for given list of polygons or None."""
    all_points = []
    for poly in polys_xy:
        if poly and len(poly) >= 3:
            all_points.extend(poly)
    if len(all_points) < 3:
        return None
    pts = np.array(all_points, dtype=np.float32)
    hull = cv2.convexHull(pts)
    return hull.reshape(-1, 2)


def get_polys_in_colony(rec: dict, colony) -> list:
    """Return list of polygons (xy lists) in this frame whose centroid lies inside colony.mask."""
    polys_xy = rec["polygons"]
    polys_in_mask = []
    for poly_xy in polys_xy:
        if polygon_center_in_mask(poly_xy, colony.mask):
            polys_in_mask.append(poly_xy)
    return polys_in_mask


def build_metrics_index(colony_metrics: dict[int, list[dict]]) -> dict[str, dict[int, dict[int, dict]]]:
    """
    Build lookup: jsonl_source -> row_in_jsonl -> colony_id -> metrics row.
    """
    metrics_index: dict[str, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for colony_id, metrics_list in colony_metrics.items():
        for metrics in metrics_list:
            jsonl_source = str(metrics["jsonl_source"])
            row_idx = int(metrics["row_in_jsonl"])
            metrics_index[jsonl_source][row_idx][int(colony_id)] = metrics
    return metrics_index


def assert_metrics_match_row(
    jsonl_source: str,
    row_idx: int,
    colony_id: int,
    polys_in_mask: list,
    metrics: dict | None,
    tol: float = 1e-4,
) -> None:
    """
    Validate that one combine row matches the polygons extracted from the filtered JSONL record.
    """
    if not polys_in_mask:
        if metrics is not None:
            raise ValueError(
                f"Unexpected combine row for {jsonl_source} row {row_idx}, colony {colony_id}: "
                "combine output exists but no polygons were found in the filtered record"
            )
        return

    if metrics is None:
        raise ValueError(
            f"Missing combine row for {jsonl_source} row {row_idx}, colony {colony_id}: "
            "polygons were found in the filtered record but no combine output row exists"
        )

    poly_count = len(polys_in_mask)
    poly_areas = [polygon_area_from_xy(p) for p in polys_in_mask]
    total_area = sum(poly_areas)
    hull_area, (hull_cx, hull_cy) = compute_convex_hull_area(polys_in_mask)

    if int(metrics["polygon_count"]) != poly_count:
        raise ValueError(
            f"polygon_count mismatch for {jsonl_source} row {row_idx}, colony {colony_id}: "
            f"combine={metrics['polygon_count']} filtered={poly_count}"
        )
    if abs(float(metrics["total_polygon_area"]) - float(total_area)) > tol:
        raise ValueError(
            f"total_polygon_area mismatch for {jsonl_source} row {row_idx}, colony {colony_id}: "
            f"combine={metrics['total_polygon_area']} filtered={total_area}"
        )
    if abs(float(metrics["convex_hull_area"]) - float(hull_area)) > tol:
        raise ValueError(
            f"convex_hull_area mismatch for {jsonl_source} row {row_idx}, colony {colony_id}: "
            f"combine={metrics['convex_hull_area']} filtered={hull_area}"
        )
    if abs(float(metrics["hull_centroid_x"]) - float(hull_cx)) > tol:
        raise ValueError(
            f"hull_centroid_x mismatch for {jsonl_source} row {row_idx}, colony {colony_id}: "
            f"combine={metrics['hull_centroid_x']} filtered={hull_cx}"
        )
    if abs(float(metrics["hull_centroid_y"]) - float(hull_cy)) > tol:
        raise ValueError(
            f"hull_centroid_y mismatch for {jsonl_source} row {row_idx}, colony {colony_id}: "
            f"combine={metrics['hull_centroid_y']} filtered={hull_cy}"
        )


def draw_overlay(frame, colonies, cols_polys_map, colors=DEFAULT_OVERLAY_COLORS, edge_thickness=3, show_counts=True):
    """Draw polygons and convex hulls for each colony onto frame (in-place) and return frame."""
    out = frame.copy()
    for i, colony in enumerate(colonies):
        cid = colony.colony_id
        polys = cols_polys_map[cid]
        color = colors[i % len(colors)]
        for p in polys:
            pts = np.array(p, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=edge_thickness, lineType=cv2.LINE_AA)
        hull_pts = compute_convex_hull_points(polys)
        if hull_pts is not None and len(hull_pts) >= 3:
            pts = np.array(hull_pts, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(out, [pts], isClosed=True, color=(255,255,255), thickness=max(2, edge_thickness+1), lineType=cv2.LINE_AA)
        if show_counts and polys:
            cx, cy = colony.center
            text = f"C{cid}:{len(polys)}"
            cv2.putText(out, text, (int(cx)+10, int(cy)+10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(out, text, (int(cx)+10, int(cy)+10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[i % len(colors)], 2, cv2.LINE_AA)
    return out


def write_colony_csv(colony, metrics_list: list, output_path: Path):
    """Write metrics for one colony to CSV."""
    if not metrics_list:
        print(f"  [INFO] No detections in colony {colony.colony_id}, skipping CSV.")
        return
    
    # Sort by frame_idx then row_in_jsonl for consistent ordering
    metrics_list = sorted(metrics_list, key=lambda m: (m["frame_idx"], m["row_in_jsonl"]))
    
    fieldnames = [
        "frame_idx",
        "jsonl_source",
        "row_in_jsonl",
        "polygon_count",
        "polygon_areas",
        "total_polygon_area",
        "convex_hull_area",
        "hull_centroid_x",
        "hull_centroid_y",
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            # Format polygon_areas as semicolon-separated values
            areas_str = ";".join(f"{a:.4f}" for a in m["polygon_areas"])
            row = {
                "frame_idx": m["frame_idx"],
                "jsonl_source": m["jsonl_source"],
                "row_in_jsonl": m["row_in_jsonl"],
                "polygon_count": m["polygon_count"],
                "polygon_areas": areas_str,
                "total_polygon_area": f"{m['total_polygon_area']:.4f}",
                "convex_hull_area": f"{m['convex_hull_area']:.4f}",
                "hull_centroid_x": f"{m['hull_centroid_x']:.2f}",
                "hull_centroid_y": f"{m['hull_centroid_y']:.2f}",
            }
            writer.writerow(row)
    
    print(f"  Wrote {len(metrics_list)} records to {output_path}")


def derive_video_path(video_dir: Path, jsonl_name: str) -> Path:
    if not jsonl_name.endswith(".filtered.jsonl.gz"):
        raise ValueError(f"Expected *.filtered.jsonl.gz input, got {jsonl_name}")
    stem = jsonl_name[: -len(".filtered.jsonl.gz")]
    return video_dir / f"{stem}.mp4"


def load_frame_row_map(jsonl_path: Path) -> dict[int, tuple[dict, int]]:
    frame_map: dict[int, tuple[dict, int]] = {}
    for rec, row_idx in iter_jsonl_with_source(jsonl_path):
        frame_idx = int(rec["frame_idx"])
        frame_map[frame_idx] = (rec, row_idx)
    return frame_map


def write_overlay_video(
    jsonl_path: Path,
    video_path: Path,
    output_path: Path,
    colonies,
    metrics_rows_for_source: dict[int, dict[int, dict]],
    edge_thickness: int,
) -> None:
    frame_map = load_frame_row_map(jsonl_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not determine video size for {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output writer: {output_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rec_info = frame_map.get(frame_idx)
            if rec_info is None:
                overlay = frame
            else:
                rec, row_idx = rec_info
                row_metrics = metrics_rows_for_source.get(row_idx, {})
                cols_polys_map = {}
                seen_colonies: set[int] = set()
                for colony in colonies:
                    cid = int(colony.colony_id)
                    polys_in_mask = get_polys_in_colony(rec, colony)
                    assert_metrics_match_row(
                        jsonl_path.name,
                        row_idx,
                        cid,
                        polys_in_mask,
                        row_metrics.get(cid),
                    )
                    if polys_in_mask:
                        seen_colonies.add(cid)
                    cols_polys_map[cid] = polys_in_mask

                extra_colonies = sorted(set(row_metrics) - seen_colonies)
                if extra_colonies:
                    raise ValueError(
                        f"Combine output has unexpected colony rows for {jsonl_path.name} row {row_idx}: "
                        f"{extra_colonies}"
                    )

                overlay = draw_overlay(
                    frame,
                    colonies,
                    cols_polys_map,
                    edge_thickness=edge_thickness,
                )

            writer.write(overlay)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Combine filtered JSONL detections with colony masks."
    )
    parser.add_argument(
        "jsonl_dir",
        type=Path,
        help="Directory containing *.filtered.jsonl.gz files"
    )
    parser.add_argument(
        "colonies_pkl",
        type=Path,
        help="Path to colonies.pkl (output from process_colony_masks.py)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        required=True,
        help="Output directory for CSVs"
    )
    parser.add_argument("--preview", action="store_true", help="Create overlay preview videos (one per input JSONL)")
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Create full overlay videos validated against the combine output rows",
    )
    parser.add_argument("--video_dir", type=Path, default=None, help="Directory that contains source videos")
    parser.add_argument("--preview_every", type=int, default=30, help="Write every Nth frame to preview video (default: 30)")
    parser.add_argument("--preview_fps", type=float, default=10.0, help="FPS for preview video (default: 10)")
    parser.add_argument("--edge_thickness", type=int, default=3, help="Edge thickness for overlay drawing")
    args = parser.parse_args()
    
    jsonl_dir = args.jsonl_dir.expanduser().resolve()
    colonies_pkl = args.colonies_pkl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    preview = bool(args.preview)
    overlay = bool(args.overlay)
    video_dir = args.video_dir.expanduser().resolve() if args.video_dir else None
    preview_every = int(args.preview_every)
    preview_fps = float(args.preview_fps)
    edge_thickness = int(args.edge_thickness)
    
    if not jsonl_dir.exists():
        print(f"[ERROR] JSONL directory not found: {jsonl_dir}")
        return
    
    if not colonies_pkl.exists():
        print(f"[ERROR] colonies.pkl not found: {colonies_pkl}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    if (preview or overlay) and video_dir is None:
        print("[ERROR] --video_dir is required when --preview or --overlay is enabled")
        return
    
    print(f"[INFO] Loading colonies from {colonies_pkl}")
    colonies = load_colonies(colonies_pkl)
    print(f"[INFO] Loaded {len(colonies)} colonies")
    
    # Find all filtered JSONL files
    jsonl_files = sorted([p for p in jsonl_dir.glob("*.filtered.jsonl.gz")])
    if not jsonl_files:
        print(f"[WARN] No *.filtered.jsonl.gz files found in {jsonl_dir}")
        return
    
    print(f"[INFO] Found {len(jsonl_files)} filtered JSONL files")
    
    # Initialize metrics dict: colony_id -> list of metric dicts
    colony_metrics = {c.colony_id: [] for c in colonies}
    
    # Process each JSONL file, tracking cumulative frame offset
    frame_offset = 0
    videos_preview_dir = output_dir / "videos_preview"
    if preview:
        videos_preview_dir.mkdir(parents=True, exist_ok=True)

    for jsonl_path in tqdm(jsonl_files, desc="Processing JSONL files"):
        jsonl_name = jsonl_path.name
        file_frame_count = 0
        cap = None
        writer = None

        if preview:
            if not jsonl_name.endswith(".filtered.jsonl.gz"):
                raise ValueError(f"Expected *.filtered.jsonl.gz input, got {jsonl_name}")
            stem = jsonl_name[: -len(".filtered.jsonl.gz")]
            cand = derive_video_path(video_dir, jsonl_name)
            if cand.exists():
                cap = cv2.VideoCapture(str(cand))
                if cap.isOpened():
                    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(videos_preview_dir / f"{stem}_preview.mp4"), fourcc, preview_fps, (W, H))
                    print(f"[INFO] Preview video -> {videos_preview_dir / f'{stem}_preview.mp4'}")
                else:
                    print(f"[WARN] Could not open video for preview: {cand}")
                    cap = None
            else:
                raise FileNotFoundError(f"Expected preview source video at {cand}")

        for rec, row_idx in iter_jsonl_with_source(jsonl_path):
            file_frame_count += 1
            for colony in colonies:
                metrics = process_frame_for_colony(rec, colony, jsonl_name, row_idx, frame_offset=frame_offset)
                if metrics is not None:
                    colony_metrics[colony.colony_id].append(metrics)

            if preview and writer is not None:
                local_idx = int(rec["frame_idx"])
                if local_idx % preview_every == 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        cols_polys_map = {}
                        for colony in colonies:
                            cols_polys_map[colony.colony_id] = get_polys_in_colony(rec, colony)
                        out_fr = draw_overlay(frame, colonies, cols_polys_map, edge_thickness=edge_thickness)
                        writer.write(out_fr)

        frame_offset += file_frame_count
        if writer:
            writer.release()
        if cap:
            cap.release()
    
    print(f"[INFO] Writing output CSVs...")
    for colony in colonies:
        metrics = colony_metrics[colony.colony_id]
        output_csv = output_dir / f"colony_{colony.colony_id:02d}_detections.csv"
        write_colony_csv(colony, metrics, output_csv)

    if overlay:
        metrics_index = build_metrics_index(colony_metrics)
        videos_overlay_dir = output_dir / "videos_overlay"
        videos_overlay_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Writing validated overlay videos...")
        for jsonl_path in tqdm(jsonl_files, desc="Writing overlay videos"):
            video_path = derive_video_path(video_dir, jsonl_path.name)
            if not video_path.exists():
                raise FileNotFoundError(f"Expected overlay source video at {video_path}")
            stem = jsonl_path.name[: -len(".filtered.jsonl.gz")]
            overlay_path = videos_overlay_dir / f"{stem}_overlay.mp4"
            write_overlay_video(
                jsonl_path=jsonl_path,
                video_path=video_path,
                output_path=overlay_path,
                colonies=colonies,
                metrics_rows_for_source=metrics_index.get(jsonl_path.name, {}),
                edge_thickness=edge_thickness,
            )
            print(f"[INFO] Overlay video -> {overlay_path}")
    
    print(f"[INFO] Done. Output CSVs in {output_dir}")


if __name__ == "__main__":
    main()
