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
# Usage:
#   ./backup_watch.sh <ssh-target> [remote_outputs_path] [local_dest] [interval_seconds]
#
# Example:
#   ./backup_watch.sh root@1.2.3.4 /workspace/ROAD/outputs ./backups 300

set -euo pipefail

SSH_TARGET="${1:?Usage: $0 <ssh-target> [remote_outputs_path] [local_dest] [interval_seconds]}"
REMOTE_PATH="${2:-/workspace/ROAD/outputs}"
LOCAL_DEST="${3:-./backups}"
INTERVAL="${4:-300}"  # default: every 5 minutes

mkdir -p "${LOCAL_DEST}"

echo ">> Backing up ${SSH_TARGET}:${REMOTE_PATH} -> ${LOCAL_DEST} every ${INTERVAL}s"
echo ">> Ctrl+C to stop. This does NOT stop or affect training on the pod."
echo ""

while true; do
  TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[${TIMESTAMP}] Syncing..."
  if rsync -avz --partial -e ssh "${SSH_TARGET}:${REMOTE_PATH}/" "${LOCAL_DEST}/" 2>&1 | tail -5; then
    echo "[${TIMESTAMP}] OK"
  else
    echo "[${TIMESTAMP}] !! Sync failed - pod may be unreachable. Will retry next interval."
  fi
  echo ""
  sleep "${INTERVAL}"
done