import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import pickle
import json
import argparse

DEBUG = False

class Colony:
    def __init__(self, colony_id: int, polygon_xy: np.ndarray, image_shape: Tuple[int, int, int]):
        """
        Colony represented by a polygon and a full-frame binary mask.

        Args:
            colony_id: Unique id (1-based).
            polygon_xy: (N, 2) array of (x, y) integer vertices.
            image_shape: (H, W, C) of the source image.
        """
        assert polygon_xy.ndim == 2 and polygon_xy.shape[1] == 2, "polygon must be (N,2)"
        self.colony_id = colony_id

        # OpenCV contour shape (N,1,2) int32
        self.polygon = polygon_xy.astype(np.int32).reshape(-1, 1, 2)

        # Bounding box from polygon
        x, y, w, h = cv2.boundingRect(self.polygon)
        self.mask_location = {'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h}

        # Full-frame mask
        H, W = image_shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [self.polygon], 255)
        self.mask = mask  # 0 or 255

    @property
    def center(self) -> tuple[int, int]:
        # Centroid from moments, fallback to bbox
        M = cv2.moments(self.polygon)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = (self.mask_location['x1'] + self.mask_location['x2']) // 2
            cy = (self.mask_location['y1'] + self.mask_location['y2']) // 2
        return (cx, cy)
    
    @staticmethod
    def load_colonies(colonies_pkl_path: str) -> List["Colony"]:
        with open(colonies_pkl_path, "rb") as f:
            colonies = pickle.load(f)
        return colonies

class ColonyMaker:
    def __init__(self, width: int | None = None, height: int | None = None,
                 orientation: str | None = None, use_otsu: bool = True, debug: bool = False):
        self.W = width
        self.H = height
        self.orientation = orientation
        self.use_otsu = use_otsu
        self.debug = debug

    @staticmethod
    def load_and_resize_image(img: np.ndarray, W: int, H: int) -> np.ndarray:
        target_w = W
        target_h = H
        if (img.shape[1], img.shape[0]) != (target_w, target_h):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return img

    def detect_colonies(self, image_path: str, info, orientation: str | None = None,
                        use_otsu: bool | None = None) -> List["Colony"]:
        """
        Detect white shapes (colonies) on a black background and save overlays on:
          1) the resized grayscale image
          2) the resized binary mask
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image from {image_path}")

        # Use instance defaults when not passed
        orientation = self.orientation if orientation is None else orientation
        use_otsu = self.use_otsu if use_otsu is None else use_otsu

        # Resize to info size (your chosen working coordinate system)
        W = int(info.get("width", img_bgr.shape[1])) if self.W is None else int(self.W)
        H = int(info.get("height", img_bgr.shape[0])) if self.H is None else int(self.H)
        img_bgr = self.load_and_resize_image(img_bgr, W, H)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Threshold (binary mask lives in the same resized space)
        if use_otsu:
            _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours on the resized binary
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        colonies: List["Colony"] = []
        for c in contours:
            poly_xy = c.reshape(-1, 2)
            colonies.append(Colony(len(colonies) + 1, poly_xy, (H, W, 3)))  # pass resized shape

        if orientation == "horizontal":
            colonies = self.sort_colonies_horizontal(colonies)

        if orientation == "clockwise":
            colonies = self.sort_colonies_clockwise(colonies, (H, W))

        # Save overlays (both share the same resized coordinates)
        # 1) On the resized grayscale image
        self.save_labeled_colonies(
            colonies,
            base_img=img_bgr,  # draw in color
            base_path=image_path,
            out_name="colonies_labeled_mask.png",
        )

        return colonies

    @staticmethod
    def sort_colonies_clockwise(colonies, image_shape):
        """Sort colonies clockwise around image center, starting from top-middle."""
        H, W = image_shape[:2]
        cx, cy = W / 2, H / 2

        # Compute angle (0 = top, increases clockwise)
        angles = [((np.arctan2(c.center[1] - cy, c.center[0] - cx) + np.pi/2) % (2*np.pi)) for c in colonies]
        colonies = [c for _, c in sorted(zip(angles, colonies), key=lambda x: x[0])]

        # Rotate so first colony is nearest top-middle (W/2, 0)
        start = np.argmin([(c.center[0] - W/2)**2 + (c.center[1])**2 for c in colonies])
        colonies = colonies[start:] + colonies[:start]

        # Reassign IDs
        for i, c in enumerate(colonies, 1):
            c.colony_id = i
        return colonies

    @staticmethod
    def sort_colonies_horizontal(colonies):
        from collections import defaultdict
        rows = defaultdict(list)
        for colony in colonies:
            rows[colony.center[1]].append(colony)

        ordered = []
        for y in sorted(rows):
            ordered.extend(sorted(rows[y], key=lambda c: c.center[0]))

        for idx, colony in enumerate(ordered, 1):
            colony.colony_id = idx
        return ordered

    @staticmethod
    def save_colonies(colonies: List["Colony"], filepath: Path):
        """Save Colony objects to pickle file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(colonies, f)
        print(f"Saved {len(colonies)} colonies to {filepath}")

    def save_labeled_colonies(
        self,
        colonies: List["Colony"],
        base_img: np.ndarray,
        base_path: str | Path,
        out_name: str = "colonies_labeled.png",
    ) -> str | None:
        """
        Draw polygons + IDs on any base image (resized grayscale, resized mask, etc.)
        """
        if base_img is None or len(colonies) == 0:
            return None

        # Ensure 3-channel for colored strokes/text
        if base_img.ndim == 2:
            labeled = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            labeled = base_img.copy()

        H = labeled.shape[0]
        font_scale = H / 500.0
        thickness = max(1, int(H / 300))
        font = cv2.FONT_HERSHEY_SIMPLEX

        for col in colonies:
            # polygon + id are already in the resized coordinate space
            cv2.polylines(labeled, [col.polygon], isClosed=True, color=(0, 255, 255), thickness=thickness)

            text = str(col.colony_id)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cx, cy = col.center
            tx, ty = cx - tw // 2, cy + th // 2
            cv2.putText(labeled, text, (tx, ty), font, font_scale, (0, 0, 0), thickness + 1)
            cv2.putText(labeled, text, (tx, ty), font, font_scale, (0, 0, 255), thickness)

        out_path = str(Path(base_path).with_name(out_name))
        cv2.imwrite(out_path, labeled)

        if self.debug:
            print("\nColony information:")
            for col in colonies:
                bb = col.mask_location
                print(f"\nColony {col.colony_id}:")
                print(f"  Center: {col.center}")
                print(f"  Polygon vertices: {col.polygon.reshape(-1,2).tolist()[:8]}{' ...' if len(col.polygon)>8 else ''}")
                print(f"  Mask shape/dtype: {col.mask.shape}/{col.mask.dtype} (0/255)")

        return out_path

def main():
    parser = argparse.ArgumentParser(description="Detect colony polygons from a mask image.")
    parser.add_argument("--base-dir", type=Path, default=None, help="Experiment directory containing mask.png and experiment_info.json")
    parser.add_argument("--image-path", type=Path, default=None, help="Path to the colony mask image")
    parser.add_argument("--info-path", type=Path, default=None, help="Path to experiment_info.json")
    parser.add_argument("--colonies-file", type=Path, default=None, help="Output path for colonies.pkl")
    parser.add_argument("--orientation", choices=["clockwise", "horizontal"], default=None, help="Optional colony ordering")
    parser.add_argument("--width", type=int, default=None, help="Override working width")
    parser.add_argument("--height", type=int, default=None, help="Override working height")
    parser.add_argument("--fixed-threshold", action="store_true", help="Use a fixed binary threshold instead of Otsu")
    parser.add_argument("--debug", action="store_true", help="Print extra colony details")
    args = parser.parse_args()

    base_dir = args.base_dir.expanduser().resolve() if args.base_dir else None
    image_path = args.image_path.expanduser().resolve() if args.image_path else None
    info_path = args.info_path.expanduser().resolve() if args.info_path else None
    colonies_file = args.colonies_file.expanduser().resolve() if args.colonies_file else None

    if base_dir is not None:
        image_path = image_path or (base_dir / "mask.png")
        info_path = info_path or (base_dir / "experiment_info.json")
        colonies_file = colonies_file or (base_dir / "colonies.pkl")

    if image_path is None or info_path is None or colonies_file is None:
        parser.error("Provide --base-dir or explicitly set --image-path, --info-path, and --colonies-file.")

    with open(info_path, "r") as f:
        info = json.load(f)

    cm = ColonyMaker(
        width=args.width,
        height=args.height,
        orientation=args.orientation,
        use_otsu=not args.fixed_threshold,
        debug=args.debug,
    )
    colonies = cm.detect_colonies(str(image_path), info, args.orientation)
    cm.save_colonies(colonies, colonies_file)
    print(f"Detected and saved {len(colonies)} colonies to {colonies_file}")



if __name__ == "__main__":
    main()
