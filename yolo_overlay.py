#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_COLORS = [
    (230, 139, 34)   # bright azure
]


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


def load_frame_map(path: Path) -> dict[int, dict]:
    frame_map: dict[int, dict] = {}
    for rec in iter_jsonl(path):
        frame_idx = int(rec["frame_idx"])
        frame_map[frame_idx] = rec
    return frame_map


def derive_output_path(jsonl_path: Path, output: Path | None) -> Path:
    if output is not None:
        return output

    name = jsonl_path.name
    if name.endswith(".filtered.jsonl.gz"):
        base = name[: -len(".filtered.jsonl.gz")]
    else:
        if not is_jsonl_gz_path(jsonl_path):
            raise ValueError(f"Expected .jsonl.gz input, got {jsonl_path}")
        base = name[: -len(".jsonl.gz")]
    return jsonl_path.with_name(f"{base}.overlay.mp4")


def draw_confidences(frame_bgr: np.ndarray, polys_xy, confidences) -> np.ndarray:
    out = frame_bgr.copy()
    for poly, conf in zip(polys_xy, confidences):
        if not poly or len(poly) < 3:
            continue
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cx = int(np.mean(pts[:, 0, 0]))
        cy = int(np.mean(pts[:, 0, 1]))
        label = f"{float(conf):.2f}"
        cv2.putText(
            out,
            label,
            (cx, max(12, cy - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            7,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            (cx, max(12, cy - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
    return out

def draw_polys(frame_bgr, polys_xy, class_ids, confidences, color_map=DEFAULT_COLORS, edge_thickness=3):
    out = frame_bgr.copy()
    for p_xy, cid, conf in zip(polys_xy, class_ids, confidences):
        if not p_xy or len(p_xy) < 3:
            continue
        color = color_map[int(cid) % len(color_map)]
        pts = np.array(p_xy, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=int(edge_thickness), lineType=cv2.LINE_AA)
    return out

def overlay_video(jsonl_path: Path, video_path: Path, output_path: Path) -> None:
    frame_map = load_frame_map(jsonl_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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
    progress = tqdm(
        total=(total_frames if total_frames > 0 else None),
        desc=f"Overlaying {video_path.name}",
        unit="frame",
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rec = frame_map.get(frame_idx)
            if rec is not None:
                polys_xy = rec.get("polygons", [])
                class_ids = rec.get("class_ids", [])
                confidences = rec.get("confidences", [])
                overlay = draw_polys(
                    frame,
                    polys_xy,
                    class_ids,
                    confidences,
                    edge_thickness=8,
                )
                overlay = draw_confidences(overlay, polys_xy, confidences)
            else:
                overlay = frame

            writer.write(overlay)
            frame_idx += 1
            progress.update(1)
    finally:
        progress.close()
        cap.release()
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay filtered YOLO polygons and confidences onto a source video."
    )
    parser.add_argument("filtered_jsonl", type=Path, help="Path to *.filtered.jsonl.gz output")
    parser.add_argument("video", type=Path, help="Path to the source video")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output overlay video path (default: beside the JSONL)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = derive_output_path(args.filtered_jsonl, args.output)
    overlay_video(args.filtered_jsonl, args.video, output_path)
    print(f"Wrote overlay video to {output_path}")


if __name__ == "__main__":
    main()
