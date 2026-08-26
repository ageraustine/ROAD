#!/usr/bin/env bash
# runpod_launch.sh
# Provisions a RunPod GPU pod, clones a GitHub repo, and prints the JupyterLab URL.
#
# Requirements:
#   1. Install the RunPod CLI:  wget -qO- https://raw.githubusercontent.com/runpod/runpodctl/main/install.sh | bash
#   2. Set your API key:        runpodctl config --apiKey <YOUR_RUNPOD_API_KEY>
#
# Usage:
#   ./runpod_launch.sh [github_repo_url] [gpu_type] [pod_name]
#
# Defaults to the ROAD repo if no repo URL is given.
#
# IMPORTANT: gpu_type must be the exact "gpuId" from `runpodctl gpu list`,
# e.g. "NVIDIA A100 80GB PCIe" — NOT the shorter "displayName" like "A100 PCIe".
# Run `runpodctl gpu list` yourself first if unsure what's currently available.
#
# Example:
#   ./runpod_launch.sh                                      # uses ROAD repo, A100 80GB PCIe
#   ./runpod_launch.sh https://github.com/ageraustine/ROAD.git "NVIDIA A100 80GB PCIe" road-run

set -euo pipefail

REPO_URL="${1:-https://github.com/ageraustine/ROAD.git}"
GPU_TYPE="${2:-NVIDIA A100 80GB PCIe}"
POD_NAME="${3:-notebook-pod}"
REPO_NAME=$(basename "$REPO_URL" .git)

# Sized for Qwen3-8B/14B LoRA/QLoRA fine-tuning: container disk holds the env only
# (ephemeral); volume holds model weights + checkpoints + dataset (persistent).
# Override via env vars if your model/workflow differs, e.g.:
#   CONTAINER_DISK_GB=25 VOLUME_GB=70 ./runpod_launch.sh ...
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-25}"
VOLUME_GB="${VOLUME_GB:-70}"

# --- Preflight: verify the requested GPU actually exists and has stock ---
echo ">> Checking availability for: ${GPU_TYPE}"
GPU_LIST_JSON=$(runpodctl gpu list)

AVAIL_CHECK=$(echo "${GPU_LIST_JSON}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
target = '${GPU_TYPE}'
match = next((g for g in data if g['gpuId'] == target), None)
if not match:
    print('NOT_FOUND')
    sys.exit(0)
any_stock = any(dc['stockStatus'] != 'none' for dc in match['dataCenterAvailability'])
price = match.get('securePricePerHr')
print(f\"{'OK' if any_stock else 'NO_STOCK'}|{price}\")
")

if [[ "${AVAIL_CHECK}" == "NOT_FOUND" ]]; then
  echo "!! GPU type '${GPU_TYPE}' not found in current inventory."
  echo ">> Cheapest available options right now:"
  echo "${GPU_LIST_JSON}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
avail = [g for g in data if any(dc['stockStatus'] != 'none' for dc in g['dataCenterAvailability'])]
avail.sort(key=lambda g: g['securePricePerHr'])
for g in avail[:8]:
    print(f\"   {g['gpuId']:<32} \${g['securePricePerHr']}/hr  ({g['memoryInGb']}GB)\")
"
  exit 1
fi

STOCK_STATUS="${AVAIL_CHECK%%|*}"
PRICE="${AVAIL_CHECK##*|}"

if [[ "${STOCK_STATUS}" == "NO_STOCK" ]]; then
  echo "!! '${GPU_TYPE}' exists but shows no stock in any data center right now. Try again shortly or pick another type."
  exit 1
fi

echo ">> Available (~\$${PRICE}/hr). Proceeding."

# Startup command: clone repo, install deps if present, launch on boot
START_CMD="bash -c 'git clone ${REPO_URL} /workspace/${REPO_NAME} && cd /workspace/${REPO_NAME} && (pip install -r requirements.txt || true) && jupyter lab --ip=0.0.0.0 --allow-root --no-browser'"

echo ">> Creating pod '${POD_NAME}' with GPU: ${GPU_TYPE}"

POD_ID=$(runpodctl create pod \
  --name "${POD_NAME}" \
  --gpuType "${GPU_TYPE}" \
  --imageName "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" \
  --containerDiskSize "${CONTAINER_DISK_GB}" \
  --volumeSize "${VOLUME_GB}" \
  --volumePath "/workspace" \
  --ports "8888/http,22/tcp" \
  --args "${START_CMD}" \
  | grep -oE 'pod "[a-zA-Z0-9]+"' | grep -oE '[a-zA-Z0-9]+$')

if [[ -z "${POD_ID}" ]]; then
  echo "Failed to create pod. Run 'runpodctl get pod' to check status manually."
  exit 1
fi

echo ">> Pod created: ${POD_ID}"
echo ">> Waiting for pod to become RUNNING..."

for i in $(seq 1 30); do
  STATUS=$(runpodctl get pod "${POD_ID}" | awk 'NR==2{print $3}')
  if [[ "${STATUS}" == "RUNNING" ]]; then
    echo ">> Pod is RUNNING."
    break
  fi
  sleep 5
done

echo ""
echo ">> Pod ID:      ${POD_ID}"
echo ">> Repo cloned: /workspace/${REPO_NAME}"
echo ">> Connect via the RunPod console 'Connect' button, or:"
echo "   runpodctl get pod ${POD_ID}"
echo ""
echo ">> When finished, STOP or TERMINATE to avoid idle billing:"
echo "   runpodctl stop pod ${POD_ID}"
echo "   runpodctl remove pod ${POD_ID}"