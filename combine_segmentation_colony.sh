#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <filtered_jsonl_dir> <colonies.pkl> -o <output_dir> [combine_segmentation_colony.py args...]" >&2
    exit 1
fi

exec /bin/python3 "$script_dir/combine_segmentation_colony.py" "$@"
