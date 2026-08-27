#!/usr/bin/env python3
"""
check_volume_capable_gpus.py - Find GPUs that have stock specifically in a
datacenter that supports Network Volumes. Stock alone isn't enough - see
runpod.sh's VOLUME_CAPABLE_DCS list for why.

Usage:
    runpodctl gpu list | python3 check_volume_capable_gpus.py [min_vram_gb]
"""
import json
import sys

# Keep this in sync with runpod.sh's VOLUME_CAPABLE_DCS - if this list goes
# stale, volume creation will start failing with a clear error naming the
# current correct list; update both places from that error message.
VOLUME_CAPABLE_DCS = {
    "AP-IN-2", "AP-JP-1", "CA-MTL-3", "CA-MTL-4", "EU-FR-1", "EU-NL-1",
    "EU-RO-1", "EUR-IS-1", "EUR-IS-3", "EUR-IS-4", "EUR-NO-1", "EUR-NO-2",
    "US-CA-2", "US-CO-1", "US-IL-1", "US-KS-2", "US-MO-2", "US-NC-2",
    "US-NE-1", "US-TX-3", "US-WA-1",
}

min_vram = int(sys.argv[1]) if len(sys.argv) > 1 else 0
data = json.load(sys.stdin)

results = []
for g in data:
    if g["memoryInGb"] < min_vram:
        continue
    good_dcs = [
        dc["dataCenterId"] for dc in g["dataCenterAvailability"]
        if dc["stockStatus"] != "none" and dc["dataCenterId"] in VOLUME_CAPABLE_DCS
    ]
    if good_dcs:
        results.append((g, good_dcs))

results.sort(key=lambda x: x[0]["securePricePerHr"])

print(f"=== GPUs with stock in a volume-capable datacenter (min {min_vram}GB) ===")
print(f"{'GPU':<20} {'gpuId':<38} {'$/hr':<8} {'VRAM':<6} Volume-capable DCs with stock")
for g, dcs in results:
    print(f"{g['displayName']:<20} {g['gpuId']:<38} ${g['securePricePerHr']:<7} {g['memoryInGb']}GB   {', '.join(dcs)}")

if not results:
    print("(none found - try a lower min_vram or check again shortly, stock shifts)")
