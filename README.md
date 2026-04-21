# broodpile

Minimal scripts for a broodpile analysis workflow built around YOLO segmentation outputs.

The repository contains four main steps:
- `yolo_segment.py` and `yolo_segment.sh`: run YOLO segmentation on videos and write raw `*.jsonl.gz` detections.
- `process_colony_masks.py`: extract colony polygons from a colony mask image and write `colonies.pkl`.
- `yolo_filtering.py`: merge and filter raw detections into `*.filtered.jsonl.gz`.
- `combine_segmentation_colony.py`: assign filtered detections to colonies and write per-colony CSV summaries.

## Setup

Create the conda environment:

```bash
conda env create -f environment_full.yml
conda activate broodpile
```

## Usage

The quickest entry point is [broodpile_pipeline.ipynb](broodpile_pipeline.ipynb). The notebook now starts with an example configuration that uses placeholder paths such as `/path/to/broodpile_example`; replace those paths and run names with your local data layout before running the notebook.

`yolo_segment.sh` is a convenience wrapper for batch GPU runs and expects GNU `parallel` plus `nvidia-smi`. If those are not available, run `yolo_segment.py` directly.

## Reference
This analysis is part of [DOI] :) Go Patrick!