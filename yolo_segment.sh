#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="$1"
input_dir="$2"
out_dir="$3"
jobs="$4"
overlay_dir="$5"
gpu_count="$(nvidia-smi --list-gpus | wc -l)"

mapfile -t inputs < <(find "$input_dir" -maxdepth 1 -type f -name '*.mp4' ! -name '._*' | sort)

run_one() {
    local video_path="$1"
    local device="$2"
    PYTHONUNBUFFERED=1 /bin/python3 "$script_dir/yolo_segment.py" "$model" "$video_path" "$out_dir" "$overlay_dir" "$device"
}

export script_dir model out_dir overlay_dir gpu_count
export -f run_one
parallel --lb -j "$jobs" 'run_one "$1" "$(( ($2 - 1) % gpu_count ))"' _ {} {%} ::: "${inputs[@]}"
