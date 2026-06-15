#!/usr/bin/env bash
#
# download_data.sh — fetch the WebShop and ALFWorld environment data.
#
# These datasets are public but large (the full WebShop catalog alone is
# ~5.2 GB), so they are NOT bundled with the repository. This script downloads
# them into the data root. Three small WebShop variant files
# (items_shuffle_1000.json, items_ins_v2_1000.json, items_human_ins.json) are
# already shipped and back the `webshop.use_small: true` code path — the full
# download is only needed for full-scale runs.
#
# Usage:
#   bash download_data.sh [--webshop] [--alfworld]   # default: both
#
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"
DO_WEBSHOP=1
DO_ALFWORLD=1

if [[ $# -gt 0 ]]; then
  DO_WEBSHOP=0; DO_ALFWORLD=0
  for arg in "$@"; do
    case "$arg" in
      --webshop)  DO_WEBSHOP=1 ;;
      --alfworld) DO_ALFWORLD=1 ;;
      *) echo "Unknown flag: $arg" >&2; exit 2 ;;
    esac
  done
fi

mkdir -p "${DATA_ROOT}"
echo "[download_data] data root = ${DATA_ROOT}"

# --------------------------------------------------------------------------- #
# WebShop
# --------------------------------------------------------------------------- #
if [[ "${DO_WEBSHOP}" -eq 1 ]]; then
  echo "[download_data] WebShop: all shipped configs use webshop.use_small=true,"
  echo "  backed by the three small data files already vendored under"
  echo "  third_party/verl-agent/.../webshop/data/ (items_shuffle_1000.json,"
  echo "  items_ins_v2_1000.json, items_human_ins.json) — no download is required"
  echo "  to reproduce the paper's WebShop results."
  echo "  For full-catalog (non-use_small) runs, fetch items_shuffle.json (~5.2GB)"
  echo "  and items_ins_v2.json (~178MB) from the Princeton NLP WebShop project"
  echo "  (github.com/princeton-nlp/WebShop) into that same data/ directory."
fi

# --------------------------------------------------------------------------- #
# ALFWorld
# --------------------------------------------------------------------------- #
if [[ "${DO_ALFWORLD}" -eq 1 ]]; then
  echo "[download_data] ALFWorld: downloading via the alfworld-download CLI ..."
  export ALFWORLD_DATA="${ALFWORLD_DATA:-${DATA_ROOT}/alfworld}"
  mkdir -p "${ALFWORLD_DATA}"
  if command -v alfworld-download >/dev/null 2>&1; then
    alfworld-download
    echo "  ALFWorld data -> ${ALFWORLD_DATA}"
  else
    echo "  ERROR: 'alfworld-download' not found. Activate the fedagent-alfworld" >&2
    echo "  conda env (it installs the alfworld package) first, then re-run." >&2
    exit 1
  fi
fi

echo "[download_data] done."
