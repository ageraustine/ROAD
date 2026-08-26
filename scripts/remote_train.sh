#!/usr/bin/env bash
# remote_train.sh - Runs ON the pod (via SSH), not on your local machine.
# Installs deps, sanity-checks the dataset/config are actually present, then
# launches train.py fully detached (nohup + disown) so it survives your SSH
# session ending. This is the "don't rely on the notebook" path.
#
# Usage (from inside the pod, e.g. after `ssh <pod>` then running this):
#   bash remote_train.sh [config_filename]
#
# Default config_filename: config_qwen3_8b.yaml
# IMPORTANT: train.py expects the config at <repo>/src/qwen2vl/config/<filename>
#   (SCRIPT_DIR/config/<name> internally) - NOT just anywhere in the repo.

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

# --- 2. Verify the dataset is actually there before burning GPU hours ---
echo ">> Checking dataset..."
if [[ ! -d "${REPO_DIR}/dataset/images" ]] || [[ -z "$(ls -A "${REPO_DIR}/dataset/images" 2>/dev/null)" ]]; then
  echo "!! Dataset not found at ${REPO_DIR}/dataset/images (or it's empty)."
  echo "   This was flagged before: the repo doesn't include an actual dataset"
  echo "   download command. Fetch it manually first, e.g.:"
  echo "     # gsutil -m cp -r gs://YOUR_BUCKET/road-dataset ${REPO_DIR}/dataset"
  echo "     # wget ... && tar -xzf ... -C ${REPO_DIR}/dataset"
  echo "   Re-run this script once the dataset is in place."
  exit 1
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
  echo "     scp config_qwen3_8b.yaml <ssh-target>:${CONFIG_PATH}"
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