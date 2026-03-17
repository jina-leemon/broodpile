#!/usr/bin/env bash
set -euo pipefail

model=../shared/Patrick_broodpile/20250917_run/runs/segment/train/weights/best.pt 
out_dir=../shared/Patrick_broodpile/20250917_run/model_predictions/patrick_oct17_alexa647_20241017_162401.40264048/
root=/media/AntGate/user/patrick/shared/patrick_oct17_alexa647_20241017_162401.40264048/

log_path="$(realpath "$(dirname "$0")/yolo_export_$(date +'%Y%m%d_%H%M%S').log")"

nohup bash -lc "
python3 - <<'PY' | parallel -j10 -u '/bin/python3 \"/home/tracking/Dropbox (Dropbox @RU)/Jina_shared/scripts/ethogram/yolo_export_segmentation_tqdm.py\" \"'$model'\" {} \"'$out_dir'\"'
from pathlib import Path
root = Path(\"$root\")
for mp4 in sorted(root.glob('*.mp4')):
    print(mp4)
    stem = mp4.stem
PY
" >"$log_path" 2>&1 &

pid=$!
echo "Started export job (PID: $pid); logs: $log_path"

## pkill -f yolo_export_segmentation_copy.py
