"""
Vulkan/SYCL Interop Matrix Test Runner

PURPOSE:
    Systematically tests all combinations of:
    - Data types, Channels, Dimensions
    - Tiling modes (OPTIMAL is default, LINEAR is opt-in)
    - Image access modes (Unsampled is default, Sampled is auto-tested where supported)

PHASES:
    Phase 1: Logic Qualification (no semaphores)
        - Tests basic functionality
        - Generates survivors.json and summary.csv
    
    Phase 2: Sync Verification (with semaphores)
        - Runs on Phase 1 survivors
        - Tests synchronization correctness
        - Generates summary_phase2.csv
        - On LINUX: Stops on first failure to prevent driver hangs.
        - On WINDOWS: Continues testing even after failures.

USAGE EXAMPLES:

    # Standard Run (Optimal Tiling -> Unsampled & Sampled)
    python run_matrix.py

    # Extended Run (Adds Linear Tiling -> Unsampled & Sampled)
    python run_matrix.py --linear

OUTPUT FILES:
    survivors.json      - Tests that passed Phase 1
    casualties.json     - Tests that failed Phase 1
    summary.csv         - Phase 1 Results
    summary_phase2.csv  - Phase 2 Results (including Semaphores)
"""


import subprocess
import sys
import os
import json
import csv
import argparse

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SURVIVORS_FILE = "survivors.json"
CASUALTIES_FILE = "casualties.json"
SUMMARY_FILE_P1 = "summary.csv"
SUMMARY_FILE_P2 = "summary_phase2.csv"

TESTS = [
    ("1D Read",       "./vsr_1d_test.bin",        "both"),
    ("1D Write",      "./vsu_1d_w_test.bin",      "unsampled_only"),
    ("2D Read",       "./vsr_2d_test.bin",        "both"),
    ("2D Write",      "./vsu_2d_w_test.bin",      "unsampled_only"), 
    ("2D Arithmetic", "./vs_2d_arith.bin",        "both"),
]

DIMENSIONS = [(32,32), (31,32), (32,33)]

TYPES = [
    "float", "half", "int32", "uint32", 
    "int16", "uint16", "uint8", "int8", "unorm8"
]

CHANNELS = ["1", "2", "4"]

# ---------------------------------------------------------
# VULKAN FORMAT MAPPING
# ---------------------------------------------------------
def get_vulkan_format(data_type, channels):
    """Maps SYCL-style types/channels to Vulkan Format strings."""
    prefix = {
        "1": "R",
        "2": "R8G8" if data_type == "uint8" else "R16G16" if "16" in data_type else "R32G32",
        "4": "R32G32B32A32" # Default to 32-bit for the R/G/B/A naming
    }
    
    # Specific overrides for 4-channel naming conventions
    if channels == "4":
        chan_prefix = "R32G32B32A32" if "32" in data_type or data_type == "float" else \
                      "R16G16B16A16" if "16" in data_type or data_type == "half" else \
                      "R8G8B8A8"
    elif channels == "2":
        chan_prefix = "R32G32" if "32" in data_type or data_type == "float" else \
                      "R16G16" if "16" in data_type or data_type == "half" else \
                      "R8G8"
    else: # channels == "1"
        chan_prefix = "R32" if "32" in data_type or data_type == "float" else \
                      "R16" if "16" in data_type or data_type == "half" else \
                      "R8"

    suffix = {
        "float":  "SFLOAT",
        "half":   "SFLOAT",
        "int32":  "SINT",
        "uint32": "UINT",
        "int16":  "SINT",
        "uint16": "UINT",
        "int8":   "SINT",
        "uint8":  "UINT",
        "unorm8": "UNORM"
    }

    return f"VK_FORMAT_{chan_prefix}_{suffix.get(data_type, 'UNKNOWN')}"

# ---------------------------------------------------------
# UTILS & PLATFORM HANDLING
# ---------------------------------------------------------
IS_WINDOWS = (os.name == 'nt')
if IS_WINDOWS: os.system('color') 

RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; RESET = "\033[0m"

def get_platform_binary(linux_style_path):
    if not IS_WINDOWS: return linux_style_path
    base_name = linux_style_path.replace("./", "").replace(".bin", ".exe")
    return f".\\{base_name}"

def run_cmd(cmd, timeout=30):
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return True, result.stdout.decode('utf-8', errors='replace')
    except subprocess.TimeoutExpired: return False, "TIMEOUT"
    except subprocess.CalledProcessError as e:
        return False, (e.stdout.decode('utf-8') if e.stdout else "") + (e.stderr.decode('utf-8') if e.stderr else "")

# ---------------------------------------------------------
# PHASE 1: LOGIC QUALIFICATION
# ---------------------------------------------------------
def run_phase_1(include_linear=False):
    print(f"\n{YELLOW}=== PHASE 1: LOGIC QUALIFICATION ==={RESET}")
    all_results = []; survivors = []; casualties = []; failures = 0

    configs = [(False, False), (False, True)] # Optimal Unsampled/Sampled
    if include_linear: configs += [(True, False), (True, True)]

    for label, raw_binary, test_support in TESTS:
        binary = get_platform_binary(raw_binary)
        if not os.path.exists(binary.strip(".\\").strip("./")): continue

        for is_linear, is_sampled in configs:
            if is_sampled and test_support == "unsampled_only": continue
            
            tiling_str = "LINEAR" if is_linear else "OPTIMAL"
            mode_str = "SAMPLED" if is_sampled else "UNSAMPLED"

            for type_name in TYPES:
                for ch in CHANNELS:
                    vk_format = get_vulkan_format(type_name, ch)
                    for w, h in DIMENSIONS:
                        size_str = f"{w}" if "1D" in label else f"{w}x{h}"
                        flags = f"--type {type_name} --channels {ch}"
                        if is_sampled: flags += " --sampled"
                        if is_linear: flags += " --linear"
                        
                        full_cmd = f"{binary} {flags} {size_str}"
                        success, output = run_cmd(full_cmd)
                        
                        status_clean = "PASS" if success else "FAIL"
                        print(f"{label:<15} | {vk_format:<25} | {tiling_str:<8} | {status_clean}")
                        
                        record = {
                            "label": label, "vk_format": vk_format, "mode": mode_str,
                            "tiling": tiling_str, "type": type_name, "ch": ch,
                            "size": size_str, "cmd": full_cmd
                        }

                        all_results.append({
                            "Vulkan Format": vk_format,
                            "Test Name": label,
                            "Mode": mode_str,
                            "Tiling": tiling_str,
                            "Data Type": type_name,
                            "Channels": ch,
                            "Dimensions": size_str,
                            "Phase 1 Result": status_clean,
                            "Command": full_cmd
                        })

                        if success: survivors.append(record)
                        else:
                            failures += 1
                            casualties.append({**record, "output": output[:500]})

    if all_results:
        with open(SUMMARY_FILE_P1, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
            
    return len(survivors) > 0

# ---------------------------------------------------------
# PHASE 2: SYNC VERIFICATION
# ---------------------------------------------------------
def run_phase_2():
    print(f"\n{YELLOW}=== PHASE 2: SYNC VERIFICATION ==={RESET}")
    if not os.path.exists(SURVIVORS_FILE): return

    with open(SURVIVORS_FILE, 'r') as f: survivors = json.load(f)
    phase2_results_csv = []; semaphore_alive = True
    
    for test in survivors:
        csv_record = {
            "Vulkan Format": test['vk_format'],
            "Test Name": test['label'],
            "Mode": test['mode'],
            "Tiling": test['tiling'],
            "Data Type": test['type'],
            "Channels": test['ch'],
            "Dimensions": test['size'],
            "Command": test['cmd'] + " --semaphores"
        }

        if not semaphore_alive and not IS_WINDOWS:
            csv_record["Phase 2 Result"] = "SKIPPED (System Unstable)"
            phase2_results_csv.append(csv_record)
            continue

        success, _ = run_cmd(test['cmd'] + " --semaphores")
        status_clean = "PASS" if success else "FAIL"
        print(f"{test['label']:<15} | {test['vk_format']:<25} | {status_clean}")
        
        csv_record["Phase 2 Result"] = status_clean
        phase2_results_csv.append(csv_record)
        
        if not success and not IS_WINDOWS: semaphore_alive = False

    if phase2_results_csv:
        with open(SUMMARY_FILE_P2, 'w', newline='') as f:
            fieldnames = ["Vulkan Format", "Test Name", "Mode", "Tiling", "Data Type", "Channels", "Dimensions", "Phase 2 Result", "Command"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(phase2_results_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', choices=['phase1', 'phase2', 'all'], default='all')
    parser.add_argument('--linear', action='store_true')
    args = parser.parse_args()

    if args.mode in ['phase1', 'all']:
        if not run_phase_1(include_linear=args.linear): sys.exit(1)
    if args.mode in ['phase2', 'all']:
        run_phase_2()