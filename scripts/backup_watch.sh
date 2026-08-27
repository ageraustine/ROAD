#!/usr/bin/env bash
# backup_watch.sh - Run on your LOCAL machine (not the pod) alongside a
# training run. Periodically rsyncs the outputs/ directory down, so progress
# survives even without a persistent Network Volume.
#
# HONEST LIMITATION: this is not equivalent to a Network Volume. If the pod
# is deleted between backup intervals, whatever changed since the last
# successful sync is lost. Shorter INTERVAL_SECONDS = less at risk, but more
# rsync overhead. This is a real trade-off, not a free substitute.
#
# RunPod SSH uses a non-standard port and a specific identity key file (see
# `runpodctl ssh info <pod-id>`'s "ssh_command" field) - plain `ssh user@host`
# won't work. This script takes those as explicit arguments rather than
# assuming defaults.
#
# Usage:
#   ./backup_watch.sh <user@host> <port> <identity_file> [remote_outputs_path] [local_dest] [interval_seconds]
#
# Example (from `runpodctl ssh info <pod-id>`'s ssh_command field):
#   ssh -i /Users/macbook/.runpod/ssh/RunPod-Key-Go root@194.68.245.115 -p 22075
# becomes:
#   ./backup_watch.sh root@194.68.245.115 22075 /Users/macbook/.runpod/ssh/RunPod-Key-Go

set -euo pipefail

REMOTE_HOST="${1:?Usage: $0 <user@host> <port> <identity_file> [remote_outputs_path] [local_dest] [interval_seconds]}"
SSH_PORT="${2:?Usage: $0 <user@host> <port> <identity_file> [remote_outputs_path] [local_dest] [interval_seconds]}"
IDENTITY_FILE="${3:?Usage: $0 <user@host> <port> <identity_file> [remote_outputs_path] [local_dest] [interval_seconds]}"
REMOTE_PATH="${4:-/workspace/ROAD/outputs}"
LOCAL_DEST="${5:-./backups}"
INTERVAL="${6:-300}"  # default: every 5 minutes

SSH_CMD="ssh -i ${IDENTITY_FILE} -p ${SSH_PORT}"

mkdir -p "${LOCAL_DEST}"

echo ">> Backing up ${REMOTE_HOST}:${REMOTE_PATH} -> ${LOCAL_DEST} every ${INTERVAL}s"
echo ">> Using: ${SSH_CMD}"
echo ">> Ctrl+C to stop. This does NOT stop or affect training on the pod."
echo ""

while true; do
  TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[${TIMESTAMP}] Syncing..."
  if rsync -avz --partial -e "${SSH_CMD}" "${REMOTE_HOST}:${REMOTE_PATH}/" "${LOCAL_DEST}/" 2>&1 | tail -5; then
    echo "[${TIMESTAMP}] OK"
  else
    echo "[${TIMESTAMP}] !! Sync failed - pod may be unreachable. Will retry next interval."
  fi
  echo ""
  sleep "${INTERVAL}"
done