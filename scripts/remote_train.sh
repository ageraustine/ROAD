#!/usr/bin/env bash
# remote_train.sh - Runs ON the pod (via SSH), not on your local machine.
# Installs deps, sanity-checks the dataset/config are actually present, then
# launches train.py fully detached (nohup + disown) so it survives your SSH
# session ending. This is the "don't rely on the notebook" path.
#
# Usage (from inside the pod, e.g. after `ssh <pod>` then running this):
#   bash remote_train.sh [config_filename]
#
# Default config_filename: config.yaml
# train.py's main() already does SCRIPT_DIR.joinpath("configs") / args.config
# internally - so --config takes just the bare filename, NOT a configs/
# prefix (passing one doubles it into configs/configs/...). Configs live at
# <repo>/src/qwen2vl/configs/<filename>.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/ROAD}"
SRC_DIR="${REPO_DIR}/src/qwen2vl"
CONFIG_NAME="${1:-config.yaml}"
CONFIG_PATH="${SRC_DIR}/configs/${CONFIG_NAME}"
LOG_PATH="/workspace/train_$(date +%Y%m%d_%H%M%S).log"

cd "${SRC_DIR}" || { echo "!! ${SRC_DIR} not found - has the repo been cloned to ${REPO_DIR}?"; exit 1; }

# --- 1. Install dependencies ---
echo ">> Installing dependencies..."
if [[ -f requirements.txt ]]; then
  pip install -r requirements.txt
else
  echo "!! No requirements.txt found at ${SRC_DIR} - skipping install. Verify this is expected."
fi

# --- 2. Download the dataset if it's not already there ---
echo ">> Checking dataset..."
DATASET_ZIP_URL="${DATASET_ZIP_URL:-https://storage.googleapis.com/road-handwriting/images.zip}"

if [[ ! -d "${REPO_DIR}/dataset/images" ]] || [[ -z "$(ls -A "${REPO_DIR}/dataset/images" 2>/dev/null)" ]]; then
  echo ">> Dataset not found at ${REPO_DIR}/dataset/images - downloading..."
  mkdir -p "${REPO_DIR}/dataset/images"

  TMP_ZIP=$(mktemp /tmp/images_XXXXXX.zip)
  echo ">> Fetching ${DATASET_ZIP_URL} ..."
  if ! curl -fL --progress-bar -o "${TMP_ZIP}" "${DATASET_ZIP_URL}"; then
    echo "!! Download failed. Check the URL is still valid and reachable from this pod."
    rm -f "${TMP_ZIP}"
    exit 1
  fi

  TMP_EXTRACT=$(mktemp -d)
  echo ">> Extracting..."
  if ! python3 -c "
import zipfile, sys
try:
    with zipfile.ZipFile('${TMP_ZIP}') as z:
        z.extractall('${TMP_EXTRACT}')
except zipfile.BadZipFile:
    print('!! Not a valid zip file', file=sys.stderr)
    sys.exit(1)
"; then
    echo "!! Extraction failed - is the download actually a valid zip? (partial download, wrong URL, etc.)"
    rm -f "${TMP_ZIP}"; rm -rf "${TMP_EXTRACT}"
    exit 1
  fi

  # Flatten regardless of the zip's internal folder structure - batched into as
  # few `mv` calls as possible (the `+` form groups many files per invocation,
  # instead of spawning one process per file, which is what was slow before).
  echo ">> Moving extracted images into place..."
  find "${TMP_EXTRACT}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
    -exec mv -t "${REPO_DIR}/dataset/images/" {} +

  rm -f "${TMP_ZIP}"
  rm -rf "${TMP_EXTRACT}"

  if [[ -z "$(ls -A "${REPO_DIR}/dataset/images" 2>/dev/null)" ]]; then
    echo "!! Extraction completed but no images ended up in ${REPO_DIR}/dataset/images."
    echo "   The zip's internal structure may be unexpected - inspect it manually:"
    echo "     curl -fL -o /tmp/images.zip ${DATASET_ZIP_URL} && unzip -l /tmp/images.zip | head -30"
    exit 1
  fi
  echo ">> Download and extraction complete."
fi

if [[ ! -f "${REPO_DIR}/dataset/Train.csv" ]] || [[ ! -f "${REPO_DIR}/dataset/Test.csv" ]]; then
  echo "!! Note: Train.csv/Test.csv not found at ${REPO_DIR}/dataset/ - only images.zip was downloaded here."
  echo "   These are assumed to already be committed in the repo. If they're missing, pull them separately."
fi
N_IMAGES=$(ls "${REPO_DIR}/dataset/images" | wc -l)
echo ">> Found ${N_IMAGES} images."

# --- 3. Verify the config is where train.py expects it ---
echo ">> Checking config..."
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "!! Config not found at ${CONFIG_PATH}"
  echo "   train.py loads configs from <repo>/src/qwen2vl/config/<name>, not just"
  echo "   anywhere in the repo. If you haven't pushed it up yet, from your LOCAL"
  echo "   machine run something like:"
  echo "     scp config.yaml <ssh-target>:${CONFIG_PATH}"
  echo "   (get <ssh-target> from: runpodctl ssh info <pod-id>)"
  exit 1
fi
echo ">> Config found: ${CONFIG_PATH}"

# --- 4. Launch training, fully detached ---
echo ">> Launching training in the background..."
echo "   Log file: ${LOG_PATH}"
nohup python3 -u train.py --config "${CONFIG_NAME}" > "${LOG_PATH}" 2>&1 &
TRAIN_PID=$!
disown

echo ">> Started. PID: ${TRAIN_PID}"
echo ">> This keeps running even if your SSH session disconnects."
echo ">> To check on it later:"
echo "     tail -f ${LOG_PATH}"
echo "     ps -p ${TRAIN_PID}          # confirm it's still alive"
echo "     pkill -f 'train.py'          # if you need to stop it"