#!/usr/bin/env python3
"""
check_48gb.py - Find RunPod GPU options at the 48GB VRAM tier.

Usage:
    runpodctl gpu list | python3 check_48gb.py
"""
import json
import sys

TARGET_GB = 48

data = json.load(sys.stdin)

def stock_str(gpu):
    dcs = [dc for dc in gpu["dataCenterAvailability"] if dc["stockStatus"] != "none"]
    if not dcs:
        return "no stock anywhere"
    return ", ".join(f"{dc['dataCenterId']}:{dc['stockStatus']}" for dc in dcs)

matches = [g for g in data if g["memoryInGb"] == TARGET_GB]
matches.sort(key=lambda g: g["securePricePerHr"])

print(f"=== GPUs at exactly {TARGET_GB}GB VRAM ===")
print(f"{'GPU':<20} {'gpuId':<38} {'$/hr':<8} Stock")
for g in matches:
    print(f"{g['displayName']:<20} {g['gpuId']:<38} ${g['securePricePerHr']:<7} {stock_str(g)}")

if not matches:
    print("(none found at exactly 48GB - check nearby tiers, e.g. 40GB or 80GB)")