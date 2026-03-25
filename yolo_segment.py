
#THIS WORKS WELL
#just didn't love the logging

#!/usr/bin/env python3
from __future__ import annotations
import sys, io, gzip, json, math
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO

# Optional deps
try:
    import torch
    torch.backends.cudnn.benchmark = True
except Exception:
    torch = None

# Optional: only needed to get a reliable total / render overlay
try:
    import cv2
except Exception:
    cv2 = None

def _probe_total_frames(video_path: str, vid_stride: int) -> int | None:
    """
    Return effective total frames after stride, or None if unknown.
    """
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if raw_total is None or raw_total <= 0:
        return None
    # Adjust for stride: ceil(raw_total / vid_stride)
    return math.ceil(raw_total / max(1, vid_stride))

def _is_readable_video(video_path: str) -> tuple[bool, str | None]:
    """
    Return whether the video can be opened and decoded enough to start processing.
    """
    vp = Path(video_path)
    if not vp.exists():
        return False, "file does not exist"
    if cv2 is None:
        return True, None

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        cap.release()
        return False, "OpenCV could not open the file"

    ok, _frame = cap.read()
    cap.release()
    if not ok:
        return False, "OpenCV could not decode the first frame"
    return True, None

def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)

def _format_fps(processed: int, elapsed_s: float) -> str:
    fps = processed / elapsed_s if elapsed_s > 0 else 0.0
    return f"{processed} frames @ {fps:.2f} fps"

def export_video_to_jsonl_gz(
    model_path: str,
    video_path: str,
    out_dir: str,
    *,
    imgsz: int = 1024,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str | int | None = 0,
    half: bool = True,
    vid_stride: int = 1,
    classes: list[int] | None = None,
    gzip_compresslevel: int = 1,
    write_buffer_size: int = 1 << 20,
    flush_every: int = 200,
    overlay_dir: str | None = None,
):
    vp = Path(video_path)
    op = Path(out_dir); op.mkdir(parents=True, exist_ok=True)
    out_file = op / f"{vp.stem}.jsonl.gz"
    ok, reason = _is_readable_video(str(vp))
    if not ok:
        _log(f"Skipping unreadable video {vp}: {reason}")
        return False

    # Try to get a deterministic total; OK if None
    total = _probe_total_frames(str(vp), vid_stride)
    _log(
        f"Starting segmentation for {vp}"
        + (f" ({total} frames after stride {vid_stride})" if total is not None else "")
        + f" on device {device}"
        + f" -> {out_file}"
    )

    if overlay_dir and cv2 is None:
        raise RuntimeError("OpenCV is required for overlay rendering but is not available.")

    overlay_writer = None
    overlay_path = None
    overlay_fps = None
    if overlay_dir:
        overlay_parent = Path(overlay_dir) if overlay_dir else op
        overlay_parent.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_parent / f"{vp.stem}_overlay.mp4"

        fps = 0.0
        cap_meta = cv2.VideoCapture(str(vp))
        if cap_meta.isOpened():
            fps = cap_meta.get(cv2.CAP_PROP_FPS) or 0.0
            cap_meta.release()
        else:
            cap_meta.release()
        overlay_fps = (fps if fps > 0 else 30.0) / max(1, vid_stride)
        _log(f"Overlay output enabled for {vp}: {overlay_path}")

    start_s = time.monotonic()
    try:
        model = YOLO(model_path)
        stream = model.predict(
            source=str(vp),
            stream=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            half=bool(half and (device is not None and device != "cpu")),
            vid_stride=vid_stride,
            classes=classes,
            verbose=False,
            agnostic_nms=False,
            max_det=300
        )

        with gzip.open(out_file, "wb", compresslevel=gzip_compresslevel) as gz_raw:
            with io.TextIOWrapper(gz_raw, encoding="utf-8", newline="\n", write_through=False) as gz:
                buffer = []
                for frame_idx, r in enumerate(stream):
                    processed = frame_idx + 1
                    boxes = getattr(r, "boxes", None)
                    masks = getattr(r, "masks", None)

                    if overlay_dir:
                        overlay_frame = r.plot(conf=False, boxes=boxes is not None, masks=True)
                        if overlay_writer is None:
                            h, w = overlay_frame.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, overlay_fps, (w, h))
                            if not overlay_writer.isOpened():
                                raise RuntimeError(f"Failed to open overlay writer for {overlay_path}")
                        overlay_writer.write(overlay_frame)

                    if boxes is None or masks is None:
                        buffer.append(json.dumps({
                            "frame_idx": frame_idx,
                            "class_ids": [],
                            "confidences": [],
                            "polygons": []
                        }) + "\n")
                    else:
                        try:
                            polys_xy = masks.xy  # list[np.ndarray (N_i x 2)]
                        except Exception:
                            polys_xy = []

                        if not polys_xy:
                            buffer.append(json.dumps({
                                "frame_idx": frame_idx,
                                "class_ids": [],
                                "confidences": [],
                                "polygons": []
                            }) + "\n")
                        else:
                            cls = boxes.cls.detach().cpu().numpy().astype(int).tolist()
                            confs = boxes.conf.detach().cpu().numpy().astype(float).tolist()
                            polygons = [p.astype(float).tolist() for p in polys_xy]
                            keep = [i for i, p in enumerate(polygons) if isinstance(p, list) and len(p) >= 3]
                            rec = {
                                "frame_idx": frame_idx,
                                "class_ids": [cls[i] for i in keep],
                                "confidences": [confs[i] for i in keep],
                                "polygons": [polygons[i] for i in keep],
                            }
                            buffer.append(json.dumps(rec) + "\n")

                    if len(buffer) >= flush_every:
                        gz.writelines(buffer); buffer.clear()

                    if processed % 1000 == 0 or processed == 1 or processed == total:
                        _log(f"[{vp.name}] {_format_fps(processed, time.monotonic() - start_s)}")

                if buffer:
                    gz.writelines(buffer)
    except Exception as exc:
        if overlay_writer is not None:
            overlay_writer.release()
        if out_file.exists():
            out_file.unlink()
        if overlay_path is not None and overlay_path.exists():
            overlay_path.unlink()
        _log(f"Skipping video {vp}: {exc}")
        return False

    if overlay_writer is not None:
        overlay_writer.release()
    elapsed_s = time.monotonic() - start_s
    _log(
        f"Finished segmentation for {vp}"
        + (f" ({total} frames)" if total is not None else "")
        + f" in {elapsed_s:.1f}s -> {out_file}"
    )
    return True

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <model_path> <video_path> <out_dir> [overlay_dir] [device]")
        sys.exit(1)

    overlay_dir = sys.argv[4] if len(sys.argv) > 4 else None
    device = sys.argv[5] if len(sys.argv) > 5 else 0

    export_video_to_jsonl_gz(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        imgsz=1024,
        conf=0.25,
        overlay_dir=overlay_dir,
        device=device,
    )

if __name__ == "__main__":
    main()

# model=/media/AntGate/user/jinb/shared/yolo_ethogram/runs/segment/train8/weights/best.pt
# out_dir=/media/AntGate/user/jinb/shared/ethogram/tk_temporalpolyethism_col03_20250501_115420

# for video in /media/AntGate/user/tomas/shared/ethogram_raws/tk_temporalpolyethism_col2_20250504_164229/*.mp4; do
#     /bin/python3 "/home/tracking/Dropbox (Dropbox @RU)/Jina_shared/scripts/myrepo-yolo/yolo_export-segmentation.py" \
#         "$model" "$video" "$out_dir"
# done

# parallel --lb -j6 'PYTHONUNBUFFERED=1 /bin/python3 "/home/tracking/Dropbox (Dropbox @RU)/Jina_shared/scripts/myrepo-yolo/yolo_export-segmentation.py" \
#     '"$model"' {} '"$out_dir"' '"$overlay_dir"'' \
#     ::: /media/AntGate/user/tomas/shared/ethogram_raws/tk_temporalpolyethism_col03_20250501_115420/*.mp4
