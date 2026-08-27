#!/usr/bin/env bash
# runpod.sh - Lifecycle manager for RunPod pods (ROAD training pipeline).
#
# Subcommands: create, resize, migrate, stop, start, destroy, status
#
# ARCHITECTURE: all persistent data lives on a dedicated Network Volume
# (a separate RunPod resource, billed independently of the pod), not on the
# pod's own inline disk. This is what makes resize/destroy safe to run
# repeatedly - pods are disposable, the volume is not. Container disk
# (CONTAINER_DISK_GB) is always ephemeral and wiped on delete.
#
# Requirements: runpodctl v2.x
#   curl -sSL https://cli.runpod.net | bash
#   runpodctl doctor
#
# UNVERIFIED ASSUMPTION: this script assumes `runpodctl pod list`, `pod get`,
# and `network-volume list` emit JSON (confirmed true for `gpu list` from your
# earlier paste, not yet confirmed for these). If pod/volume lookups return
# empty unexpectedly, run one of those commands manually and send me the
# output - the parsing functions below are isolated and easy to patch.

set -euo pipefail

DEFAULT_REPO="https://github.com/ageraustine/ROAD.git"
DEFAULT_GPU="NVIDIA A100 80GB PCIe"
DEFAULT_IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-25}"
VOLUME_GB="${VOLUME_GB:-70}"

# USE_NETWORK_VOLUME=false switches to a pod-local (inline) volume instead of
# a Network Volume. This sidesteps the volume-capable-datacenter constraint
# entirely - any datacenter with GPU stock works, opening up cheaper GPUs
# (A40, RTX A6000) that don't currently have stock in a volume-capable DC.
#
# REAL TRADE-OFF, not free: an inline volume survives pod STOP but is
# permanently destroyed on pod DELETE/TERMINATE. There is no separate,
# reusable storage resource anymore - so `resize` (which deletes and
# recreates the pod to change GPU) becomes DESTRUCTIVE in this mode, not
# safe like it is with a real Network Volume. This mode assumes you are
# periodically downloading checkpoints externally (see backup_watch.sh) as
# the actual safety net, not the pod's disk itself.
USE_NETWORK_VOLUME="${USE_NETWORK_VOLUME:-false}"

# ---------- shared helpers ----------

gpu_list_json() { runpodctl gpu list; }

gpu_price_and_stock() {  # $1=gpuId -> "OK|price" / "NO_STOCK|price" / "NOT_FOUND"
  local gpu_id="$1"
  gpu_list_json | python3 -c "
import json, sys
data = json.load(sys.stdin)
target = '${gpu_id}'
match = next((g for g in data if g['gpuId'] == target), None)
if not match:
    print('NOT_FOUND'); sys.exit(0)
any_stock = any(dc['stockStatus'] != 'none' for dc in match['dataCenterAvailability'])
print(f\"{'OK' if any_stock else 'NO_STOCK'}|{match['securePricePerHr']}\")
"
}

# Datacenters confirmed to support Network Volumes, per RunPod's own API error
# message (which enumerates the full supported list). GPU stock existing in a
# datacenter does NOT mean that datacenter supports volumes - CA-MTL-1 is a
# real example that has GPU stock but rejects volume creation entirely. This
# list may drift over time; if volume creation starts failing again with a
# similar "not found or does not support network volumes" error, update it
# from that error message's own list.
VOLUME_CAPABLE_DCS=(
  AP-IN-2 AP-JP-1 CA-MTL-3 CA-MTL-4 EU-FR-1 EU-NL-1 EU-RO-1 EUR-IS-1
  EUR-IS-3 EUR-IS-4 EUR-NO-1 EUR-NO-2 US-CA-2 US-CO-1 US-IL-1 US-KS-2
  US-MO-2 US-NC-2 US-NE-1 US-TX-3 US-WA-1
)

pick_datacenter_for_gpu() {  # $1=gpuId -> a volume-capable DC id with GPU stock, or empty
  local gpu_id="$1"
  local dc_list; dc_list=$(printf '%s\n' "${VOLUME_CAPABLE_DCS[@]}")
  gpu_list_json | python3 -c "
import json, sys
data = json.load(sys.stdin)
volume_capable = set('''$dc_list'''.split())
target = '${gpu_id}'
match = next((g for g in data if g['gpuId'] == target), None)
if not match:
    sys.exit(0)
for dc in match['dataCenterAvailability']:
    if dc['stockStatus'] != 'none' and dc['dataCenterId'] in volume_capable:
        print(dc['dataCenterId']); break
"
}

print_cheapest_alternatives() {
  gpu_list_json | python3 -c "
import json, sys
data = json.load(sys.stdin)
avail = [g for g in data if any(dc['stockStatus'] != 'none' for dc in g['dataCenterAvailability'])]
avail.sort(key=lambda g: g['securePricePerHr'])
for g in avail[:8]:
    print(f\"   {g['gpuId']:<32} \${g['securePricePerHr']}/hr  ({g['memoryInGb']}GB)\")
"
}

require_gpu_stock() {  # $1=gpuId, exits 1 with guidance if unusable
  local gpu_id="$1"
  echo ">> Checking availability for: ${gpu_id}"
  local check; check=$(gpu_price_and_stock "${gpu_id}")
  if [[ "${check}" == "NOT_FOUND" ]]; then
    echo "!! GPU type '${gpu_id}' not found. Cheapest available:"; print_cheapest_alternatives; exit 1
  fi
  local status="${check%%|*}" price="${check##*|}"
  if [[ "${status}" == "NO_STOCK" ]]; then
    echo "!! '${gpu_id}' has no stock anywhere right now."; print_cheapest_alternatives; exit 1
  fi
  echo ">> Available (~\$${price}/hr) as of this check. (Point-in-time only - see retry logic on create.)"
}

get_pod_id() {  # $1=pod name -> id or empty
  runpodctl pod list --all --name "$1" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d[0]['id'] if d else '')
except Exception:
    print('')
" 2>/dev/null || true
}

get_pod_status() {  # $1=pod id -> status string
  runpodctl pod get "$1" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('status', d.get('desiredStatus', '')))
except Exception:
    print('')
" 2>/dev/null || true
}

wait_for_running() {  # $1=pod id
  for i in $(seq 1 30); do
    local s; s=$(get_pod_status "$1")
    if [[ "$s" == "RUNNING" ]]; then echo ">> Pod is RUNNING."; return 0; fi
    sleep 5
  done
  echo "!! Pod did not reach RUNNING within 150s. Check manually: runpodctl pod get $1"
}

get_volume_id() {  # $1=volume name -> id or empty
  runpodctl network-volume list | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    match = next((v for v in d if v.get('name') == '$1'), None)
    print(match['id'] if match else '')
except Exception:
    print('')
" 2>/dev/null || true
}

ensure_volume() {  # $1=vol name $2=size GB $3=gpuId -> prints volume id (info goes to stderr)
  local vol_name="$1" size="$2" gpu_id="$3"
  local vol_id; vol_id=$(get_volume_id "${vol_name}")
  if [[ -n "${vol_id}" ]]; then echo "${vol_id}"; return 0; fi
  local dc; dc=$(pick_datacenter_for_gpu "${gpu_id}")
  if [[ -z "${dc}" ]]; then
    echo "!! Could not find a volume-capable data center with stock for '${gpu_id}'." >&2
    exit 1
  fi
  echo ">> Creating Network Volume '${vol_name}' (${size}GB) in ${dc}..." >&2
  local create_output create_exit
  create_output=$(runpodctl network-volume create --name "${vol_name}" --size "${size}" --data-center-id "${dc}" 2>&1)
  create_exit=$?
  echo "${create_output}" >&2

  # Don't trust the exit code alone - verify the volume genuinely exists now.
  # This is what silently failed before: an error response was printed but
  # never checked, and an empty vol_id got passed straight into pod create,
  # producing a pod with NO volume attached and no warning about it.
  vol_id=$(get_volume_id "${vol_name}")
  if [[ ${create_exit} -ne 0 ]] || [[ -z "${vol_id}" ]]; then
    echo "!! Volume creation failed or could not be verified. Refusing to proceed -" >&2
    echo "   creating a pod without a real volume would silently lose data on stop/delete." >&2
    exit 1
  fi
  echo "${vol_id}"
}

create_pod_attached() {  # $1=pod_name $2=gpu_id $3=volume_id_or_empty $4=docker_args
  if [[ "${USE_NETWORK_VOLUME}" == "true" ]]; then
    runpodctl pod create \
      --name "$1" \
      --gpu-id "$2" \
      --image "${DEFAULT_IMAGE}" \
      --container-disk-in-gb "${CONTAINER_DISK_GB}" \
      --network-volume-id "$3" \
      --volume-mount-path "/workspace" \
      --ports "8888/http,22/tcp" \
      --docker-args "$4"
  else
    # Inline volume - no --network-volume-id at all, works in any datacenter
    # with GPU stock. Dies with the pod on delete/terminate (survives stop).
    runpodctl pod create \
      --name "$1" \
      --gpu-id "$2" \
      --image "${DEFAULT_IMAGE}" \
      --container-disk-in-gb "${CONTAINER_DISK_GB}" \
      --volume-in-gb "${VOLUME_GB}" \
      --volume-mount-path "/workspace" \
      --ports "8888/http,22/tcp" \
      --docker-args "$4"
  fi
}

# ---------- subcommands ----------

cmd_create() {
  local repo_url="${1:-$DEFAULT_REPO}" gpu_id="${2:-$DEFAULT_GPU}" pod_name="${3:-notebook-pod}"
  local vol_name="${pod_name}-vol"
  local repo_name; repo_name=$(basename "${repo_url}" .git)

  require_gpu_stock "${gpu_id}"

  local existing; existing=$(get_pod_id "${pod_name}")
  if [[ -n "${existing}" ]]; then
    echo "!! A pod named '${pod_name}' already exists (${existing}). Use 'resize' or 'destroy' first."
    exit 1
  fi

  local vol_id=""
  if [[ "${USE_NETWORK_VOLUME}" == "true" ]]; then
    vol_id=$(ensure_volume "${vol_name}" "${VOLUME_GB}" "${gpu_id}")
    if [[ -z "${vol_id}" ]]; then
      echo "!! No volume ID returned - refusing to create a pod without persistent storage."
      exit 1
    fi
  else
    echo ">> USE_NETWORK_VOLUME=false: using an inline volume (no volume-capable-datacenter"
    echo "   constraint, but does NOT survive pod delete/terminate - only stop)."
    echo "   Back up checkpoints externally, e.g. via backup_watch.sh, not relying on this disk alone."
  fi
  # NOTE: dependency install skipped on purpose - Jupyter launches right after
  # clone. Run `pip install -r requirements.txt` manually from a Jupyter
  # terminal (or the notebook's own install cell) once you actually need it.
  local repo_path="/workspace/${repo_name}"
  local start_cmd="bash -c 'set -e; if [ -d \"${repo_path}/.git\" ]; then echo \"Repo already present, pulling latest...\"; cd \"${repo_path}\" && (git pull || echo \"git pull failed, continuing with existing checkout\"); elif [ -d \"${repo_path}\" ]; then echo \"Found non-git directory at ${repo_path}, replacing it...\"; rm -rf \"${repo_path}\" && git clone ${repo_url} \"${repo_path}\"; else git clone ${repo_url} \"${repo_path}\"; fi; jupyter lab --ip=0.0.0.0 --allow-root --no-browser --notebook-dir=/workspace'"

  echo ">> Creating pod '${pod_name}' with GPU: ${gpu_id}, volume: ${vol_name}"
  local out ok=0
  for attempt in 1 2 3; do
    set +e
    out=$(create_pod_attached "${pod_name}" "${gpu_id}" "${vol_id}" "${start_cmd}" 2>&1)
    exit_code=$?
    set -e
    if [[ ${exit_code} -eq 0 ]]; then ok=1; break; fi
    if echo "${out}" | grep -qi "no longer any instances available\|no instances available"; then
      echo "!! Attempt ${attempt}/3: out of stock. Retrying in 10s..."
      sleep 10
    else
      echo "!! Create failed: ${out}"; exit 1
    fi
  done
  if [[ ${ok} -ne 1 ]]; then
    echo "!! Out of retries. Alternatives:"; print_cheapest_alternatives; exit 1
  fi

  local pod_id; pod_id=$(get_pod_id "${pod_name}")
  echo ">> Pod created: ${pod_id}"
  wait_for_running "${pod_id}"
  echo ">> Repo: /workspace/${repo_name}   Volume: ${vol_name} (${vol_id})"
  echo ">> Connect: runpodctl pod get ${pod_id}"
}

cmd_resize() {
  # Upgrade or downgrade GPU - same mechanism either direction: stop+delete
  # the pod, keep the Network Volume, recreate on the new GPU (must be in the
  # same datacenter the volume already lives in).
  local pod_name="${1:?Usage: $0 resize <pod_name> <new_gpu_id>}"
  local new_gpu="${2:?Usage: $0 resize <pod_name> <new_gpu_id>}"
  local vol_name="${pod_name}-vol"

  if [[ "${USE_NETWORK_VOLUME}" != "true" ]]; then
    echo "!! resize is unsafe in USE_NETWORK_VOLUME=false mode: it deletes the pod,"
    echo "   which destroys the inline volume's data with no way to preserve it."
    echo "   Back up what you need first (backup_watch.sh / manual rsync), then use"
    echo "   'destroy' + 'create' explicitly - that makes the data loss a conscious"
    echo "   step instead of something resize does silently."
    exit 1
  fi

  local pod_id; pod_id=$(get_pod_id "${pod_name}")
  [[ -z "${pod_id}" ]] && { echo "!! No pod named '${pod_name}' found."; exit 1; }
  local vol_id; vol_id=$(get_volume_id "${vol_name}")
  [[ -z "${vol_id}" ]] && { echo "!! No Network Volume '${vol_name}' found - was this pod created via this script's 'create'?"; exit 1; }

  require_gpu_stock "${new_gpu}"

  echo ">> Volume '${vol_name}' will be preserved. Pod '${pod_name}' will be destroyed and recreated on ${new_gpu}."
  echo ">> Stopping and deleting old pod (${pod_id})..."
  runpodctl pod stop "${pod_id}" || true
  runpodctl pod delete "${pod_id}"

  echo ">> Recreating on ${new_gpu}..."
  # No repo re-clone needed - it's already on the volume. Just relaunch Jupyter.
  local start_cmd="bash -c 'jupyter lab --ip=0.0.0.0 --allow-root --no-browser --notebook-dir=/workspace'"
  local out ok=0
  for attempt in 1 2 3; do
    set +e
    out=$(create_pod_attached "${pod_name}" "${new_gpu}" "${vol_id}" "${start_cmd}" 2>&1)
    exit_code=$?
    set -e
    if [[ ${exit_code} -eq 0 ]]; then ok=1; break; fi
    if echo "${out}" | grep -qi "no instances available"; then
      echo "!! Out of stock, retrying in 10s (${attempt}/3)..."; sleep 10
    else
      echo "!! Recreate failed: ${out}"
      echo "!! Your data is safe on volume '${vol_name}' (${vol_id}) - retry manually with a different GPU:"
      echo "   $0 resize ${pod_name} \"<other gpuId>\""
      exit 1
    fi
  done
  if [[ ${ok} -ne 1 ]]; then
    echo "!! Out of retries. Data is safe on volume '${vol_name}'. Alternatives:"; print_cheapest_alternatives; exit 1
  fi

  pod_id=$(get_pod_id "${pod_name}")
  echo ">> New pod: ${pod_id}"
  wait_for_running "${pod_id}"
}

cmd_migrate() {
  # Cross-datacenter migration. Network Volumes are datacenter-locked, so this
  # is NOT a resize - it requires copying data to a fresh volume in the target
  # datacenter. NOT fully automated: this prints the exact steps rather than
  # running an unattended multi-hour rsync of model checkpoints blind.
  local pod_name="${1:?Usage: $0 migrate <pod_name> <new_gpu_id>}"
  local new_gpu="${2:?Usage: $0 migrate <pod_name> <new_gpu_id>}"
  local vol_name="${pod_name}-vol"

  local pod_id; pod_id=$(get_pod_id "${pod_name}")
  [[ -z "${pod_id}" ]] && { echo "!! No pod named '${pod_name}' found."; exit 1; }

  require_gpu_stock "${new_gpu}"
  local target_dc; target_dc=$(pick_datacenter_for_gpu "${new_gpu}")
  local new_vol_name="${vol_name}-${target_dc}"

  echo ">> Source pod SSH info:"
  runpodctl ssh info "${pod_id}"
  echo ""
  echo "== Migration is guided, not automatic. Steps: =="
  echo "1. Create the destination volume:"
  echo "   runpodctl network-volume create --name ${new_vol_name} --size ${VOLUME_GB} --data-center-id ${target_dc}"
  echo "2. Create a temporary pod attached to it (any cheap GPU in ${target_dc} works just to run rsync):"
  echo "   runpodctl pod create --name ${pod_name}-migrate-tmp --gpu-id \"${new_gpu}\" --image ${DEFAULT_IMAGE} \\"
  echo "     --container-disk-in-gb ${CONTAINER_DISK_GB} --network-volume-id <new-vol-id> --volume-mount-path /workspace --ports \"22/tcp\""
  echo "3. Get its SSH info: runpodctl ssh info <temp-pod-id>"
  echo "4. Copy data directly between pods (large checkpoints - budget real time for this):"
  echo "   rsync -avz -e ssh /workspace/ <temp-pod-ssh-target>:/workspace/"
  echo "5. Once confirmed synced, verify on the new pod, then:"
  echo "   $0 destroy ${pod_name}                 # removes old pod + prompts on old volume"
  echo "   runpodctl pod update <temp-pod-id> --name ${pod_name}"
  echo ""
  echo "Reminder: 'resize' is the right tool if ${new_gpu} is available in the SAME datacenter as the"
  echo "current volume - it's instant and doesn't need any of this. This path is only for a genuine DC move."
}

cmd_status() {
  echo "=== Pods ==="
  runpodctl pod list --all
  echo ""
  echo "=== Network Volumes ==="
  runpodctl network-volume list
}

cmd_stop() {
  local pod_name="${1:?Usage: $0 stop <pod_name>}"
  local pod_id; pod_id=$(get_pod_id "${pod_name}")
  [[ -z "${pod_id}" ]] && { echo "!! No pod named '${pod_name}' found."; exit 1; }
  runpodctl pod stop "${pod_id}"
  echo ">> Stopped ${pod_name} (${pod_id}). Network Volume data preserved; billing pauses for compute, volume still bills separately."
}

cmd_start() {
  local pod_name="${1:?Usage: $0 start <pod_name>}"
  local pod_id; pod_id=$(get_pod_id "${pod_name}")
  [[ -z "${pod_id}" ]] && { echo "!! No pod named '${pod_name}' found."; exit 1; }
  runpodctl pod start "${pod_id}"
  wait_for_running "${pod_id}"
}

cmd_destroy() {
  local pod_name="${1:?Usage: $0 destroy <pod_name> [--keep-volume]}"
  local keep_volume="${2:-}"
  local pod_id; pod_id=$(get_pod_id "${pod_name}")
  if [[ -n "${pod_id}" ]]; then
    echo ">> Stopping and deleting pod '${pod_name}' (${pod_id})..."
    runpodctl pod stop "${pod_id}" || true
    runpodctl pod delete "${pod_id}"
  else
    echo "!! No pod named '${pod_name}' found (may already be deleted)."
  fi

  local vol_name="${pod_name}-vol"
  local vol_id; vol_id=$(get_volume_id "${vol_name}")
  if [[ -n "${vol_id}" ]]; then
    if [[ "${keep_volume}" == "--keep-volume" ]]; then
      echo ">> Keeping volume '${vol_name}' (${vol_id}) - your data is safe, still billed separately."
    else
      echo "!! Volume '${vol_name}' (${vol_id}) still exists and is billed separately."
      echo "   This script never deletes volumes automatically - deleting is irreversible."
      echo "   To delete it: runpodctl network-volume delete ${vol_id}"
    fi
  fi
}

cmd_train() {
  # Not fully automated on purpose: SSH connection details are best confirmed
  # live (runpodctl ssh info format isn't something I've verified output for
  # here), and you'll want to see dataset/config checks pass before a
  # multi-hour run kicks off, rather than trust it blind. This prints the
  # exact commands to run.
  local pod_name="${1:?Usage: $0 train <pod_name> [config_filename]}"
  local config_name="${2:-config_qwen3_8b.yaml}"
  local pod_id; pod_id=$(get_pod_id "${pod_name}")
  [[ -z "${pod_id}" ]] && { echo "!! No pod named '${pod_name}' found."; exit 1; }

  echo ">> SSH connection info for '${pod_name}':"
  runpodctl ssh info "${pod_id}"
  echo ""
  echo "== To start training (fully detached - survives SSH disconnect): =="
  echo "1. Push remote_train.sh onto the pod (once):"
  echo "   scp remote_train.sh <ssh-target-from-above>:/workspace/remote_train.sh"
  echo "2. If the config isn't already in the repo, push it to where train.py expects it:"
  echo "   scp ${config_name} <ssh-target-from-above>:/workspace/${pod_name%%-*}*/src/qwen2vl/config/${config_name}"
  echo "   (adjust the repo folder name if it differs)"
  echo "3. SSH in and launch it:"
  echo "   ssh <ssh-target-from-above>"
  echo "   bash /workspace/remote_train.sh ${config_name}"
  echo "   # you'll get a PID and a log path, then you can safely disconnect"
  echo ""
  echo "Training keeps running on the pod after you close the SSH session - only"
  echo "'runpod.sh stop/destroy' or a crash inside train.py will stop it."
}

cmd_logs() {
  local pod_name="${1:?Usage: $0 logs <pod_name>}"
  local pod_id; pod_id=$(get_pod_id "${pod_name}")
  [[ -z "${pod_id}" ]] && { echo "!! No pod named '${pod_name}' found."; exit 1; }

  echo ">> Container-level logs (from the pod's boot process, e.g. Jupyter):"
  runpodctl pod logs "${pod_id}" || true
  echo ""
  echo "== Training progress lives in a SEPARATE log file (not shown above), =="
  echo "== since it runs detached via SSH, not as the container's main process. =="
  echo "SSH connection info:"
  runpodctl ssh info "${pod_id}"
  echo ""
  echo "Then tail the actual training log:"
  echo "   ssh <ssh-target-from-above> 'tail -f /workspace/train_*.log'"
}

# ---------- dispatch ----------

case "${1:-}" in
  create)  shift; cmd_create "$@" ;;
  resize)  shift; cmd_resize "$@" ;;
  migrate) shift; cmd_migrate "$@" ;;
  train)   shift; cmd_train "$@" ;;
  logs)    shift; cmd_logs "$@" ;;
  status)  shift; cmd_status "$@" ;;
  stop)    shift; cmd_stop "$@" ;;
  start)   shift; cmd_start "$@" ;;
  destroy) shift; cmd_destroy "$@" ;;
  *)
    cat <<USAGE
Usage: $0 <command> [args]

Commands:
  create  [repo_url] [gpu_id] [pod_name]   Create a pod + Network Volume, clone repo, start Jupyter
  train   <pod_name> [config_filename]     Print steps to push remote_train.sh + config and launch detached training
  logs    <pod_name>                       Show container logs + how to tail the live training log
  resize  <pod_name> <new_gpu_id>          Upgrade/downgrade GPU, same datacenter. Data preserved.
  migrate <pod_name> <new_gpu_id>          Cross-datacenter move. Prints guided rsync steps (not blind-automated).
  status                                   List all pods and Network Volumes
  stop    <pod_name>                       Stop pod (compute billing pauses, volume persists)
  start   <pod_name>                       Start a stopped pod
  destroy <pod_name> [--keep-volume]       Delete pod. Volume is NEVER auto-deleted - do that explicitly.

Environment overrides: CONTAINER_DISK_GB (default 25), VOLUME_GB (default 70)

Examples:
  $0 create
  $0 create https://github.com/ageraustine/ROAD.git "NVIDIA A100 80GB PCIe" road-run
  $0 train road-run config_qwen3_8b.yaml
  $0 logs road-run
  $0 resize road-run "NVIDIA A100-SXM4-80GB"
  $0 status
  $0 destroy road-run --keep-volume
USAGE
    exit 1
    ;;
esac