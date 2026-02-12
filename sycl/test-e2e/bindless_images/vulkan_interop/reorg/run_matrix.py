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

# Test Configuration: (Label, Binary Name, Test Type)
# Test Type:
#   "both"           -> Runs Unsampled AND Sampled
#   "unsampled_only" -> Runs Unsampled only (skips Sampled loop)
TESTS = [
    # Label                     Binary Name                 Test Type
    ("1D Read",                 "./vsr_1d_test.bin",        "both"),
    ("1D Write",                "./vsu_1d_w_test.bin",      "unsampled_only"),

    ("2D Read",                 "./vsr_2d_test.bin",        "both"),
    ("2D Write",                "./vsu_2d_w_test.bin",      "unsampled_only"), 
    
    # 3D tests commented out
    # ("3D Read",                 "./vsr_3d_test.bin",        "both"),
    # ("3D Write",                "./vsu_3d_w_test.bin",      "unsampled_only"), 

    ("2D Arithmetic",           "./vs_2d_arith.bin",        "both"),
]

DIMENSIONS = [
     (32,32),
     (31,32),
     (32,33),
]

TYPES = [
    "float", "half", "int32", "uint32", 
    "int16", "uint16", "uint8", "int8", "unorm8"
]

CHANNELS = ["1", "2", "4"]

# ---------------------------------------------------------
# UTILS & PLATFORM HANDLING
# ---------------------------------------------------------
IS_WINDOWS = (os.name == 'nt')

if IS_WINDOWS:
    os.system('color') 
    
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def get_platform_binary(linux_style_path):
    if not IS_WINDOWS:
        return linux_style_path
    base_name = linux_style_path.replace("./", "")
    base_name = base_name.replace(".bin", ".exe")
    return f".\\{base_name}"

def run_cmd(cmd, timeout=30):
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            timeout=timeout
        )
        return True, result.stdout.decode('utf-8', errors='replace')
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: Test hung"
    except subprocess.CalledProcessError as e:
        stdout = e.stdout.decode('utf-8', errors='replace') if e.stdout else ""
        stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
        return False, stdout + stderr

# ---------------------------------------------------------
# PHASE 1: LOGIC QUALIFICATION
# ---------------------------------------------------------
def run_phase_1(include_linear=False):
    print(f"\n{YELLOW}=== PHASE 1: LOGIC QUALIFICATION ==={RESET}")
    if include_linear:
        print("Modes: OPTIMAL + LINEAR")
    else:
        print("Modes: OPTIMAL only")
        
    print("-" * 110)
    print(f"{'TEST':<20} | {'MODE':<10} | {'TILING':<8} | {'TYPE':<8} | {'CH':<2} | {'SIZE':<8} | {'RESULT'}")
    print("-" * 110)
    
    survivors = []
    casualties = []
    all_results = []
    failures = 0

    configs = [
        (False, False), # Optimal, Unsampled
        (False, True),  # Optimal, Sampled
    ]
    
    if include_linear:
        configs.append((True, False)) # Linear, Unsampled
        configs.append((True, True))  # Linear, Sampled

    for label, raw_binary, test_support in TESTS:
        binary = get_platform_binary(raw_binary)
        
        check_path = binary.replace("./", "").replace(".\\", "")
        if not os.path.exists(check_path):
            print(f"Skipping {label} (Binary not found: {binary})")
            continue

        for is_linear, is_sampled in configs:
            if is_sampled and test_support == "unsampled_only":
                continue

            if "3D" in label and is_linear:
                continue

            tiling_str = "LINEAR" if is_linear else "OPTIMAL"
            mode_str = "SAMPLED" if is_sampled else "UNSAMPLED"

            for type_name in TYPES:
                for ch in CHANNELS:
                    for w, h in DIMENSIONS:
                        
                        if "1D" in label:
                            size_str = f"{w}"
                        elif "3D" in label:
                            size_str = f"{w}x{h}x{h}"
                        else:
                            size_str = f"{w}x{h}"
                        
                        flags = f"--type {type_name} --channels {ch}"
                        if is_sampled:
                            flags += " --sampled"
                        if is_linear:
                            flags += " --linear"
                        
                        full_cmd = f"{binary} {flags} {size_str}"
                        
                        # EXECUTE
                        success, output = run_cmd(full_cmd)
                        
                        status = f"{GREEN}PASS{RESET}" if success else f"{RED}FAIL{RESET}"
                        status_clean = "PASS" if success else "FAIL"
                        
                        print(f"{label:<20} | {mode_str:<10} | {tiling_str:<8} | {type_name:<8} | {ch:<2} | {size_str:<8} | {status}")
                        
                        record = {
                            "label": label,
                            "api": "Vulkan",
                            "mode": mode_str,
                            "tiling": tiling_str,
                            "type": type_name,
                            "ch": ch,
                            "size": size_str,
                            "cmd": full_cmd
                        }
                        
                        # Add to CSV list
                        all_results.append({
                            "API": "Vulkan",
                            "Test Name": label,
                            "Mode": mode_str,
                            "Tiling": tiling_str,
                            "Data Type": type_name,
                            "Channels": ch,
                            "Dimensions": size_str,
                            "Phase 1 Result": status_clean,
                            "Command": full_cmd
                        })

                        if success:
                            survivors.append(record)
                        else:
                            failures += 1
                            casualties.append({**record, "output": output[:500]})

    # Save JSONs
    with open(SURVIVORS_FILE, 'w') as f:
        json.dump(survivors, f, indent=2)

    with open(CASUALTIES_FILE, 'w') as f:
        json.dump(casualties, f, indent=2)

    # Save CSV
    if all_results:
        with open(SUMMARY_FILE_P1, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    
    print("-" * 110)
    print(f"Phase 1 Complete. {len(survivors)} passed. {failures} failed.")
    return len(survivors) > 0

# ---------------------------------------------------------
# PHASE 2: VERIFICATION (SYNC)
# ---------------------------------------------------------
def run_phase_2(survivors_file=SURVIVORS_FILE):
    print(f"\n{YELLOW}=== PHASE 2: SYNC VERIFICATION (With Semaphores) ==={RESET}")
    
    if not os.path.exists(survivors_file):
        print(f"{RED}Error: {survivors_file} not found.{RESET}")
        return

    with open(survivors_file, 'r') as f:
        survivors = json.load(f)

    print(f"Loaded {len(survivors)} qualified tests.")
    print("-" * 110)
    print(f"{'TEST':<20} | {'MODE':<10} | {'TILING':<8} | {'TYPE':<8} | {'CH':<2} | {'SIZE':<8} | {'RESULT'}")
    print("-" * 110)

    semaphore_alive = True
    phase2_results_csv = []
    
    for test in survivors:
        
        # Prepare CSV Record (Based on Phase 1 record)
        csv_record = {
            "API": "Vulkan",
            "Test Name": test['label'],
            "Mode": test['mode'],
            "Tiling": test['tiling'],
            "Data Type": test['type'],
            "Channels": test['ch'],
            "Dimensions": test['size'],
            "Command": test['cmd'] + " --semaphores"
        }

        # Check safety latch (LINUX ONLY)
        if not semaphore_alive and not IS_WINDOWS:
            # RECORD SKIPPED TESTS
            print(f"{test['label']:<20} | {test['mode']:<10} | {test['tiling']:<8} | {test['type']:<8} | {test['ch']:<2} | {test['size']:<8} | {YELLOW}SKIPPED{RESET}")
            csv_record["Phase 2 Result"] = "SKIPPED (System Unstable)"
            phase2_results_csv.append(csv_record)
            continue

        cmd_sem = f"{test['cmd']} --semaphores"
        
        success, output = run_cmd(cmd_sem, timeout=30)
        status = f"{GREEN}PASS{RESET}" if success else f"{RED}FAIL{RESET}"
        status_clean = "PASS" if success else "FAIL"
        
        print(f"{test['label']:<20} | {test['mode']:<10} | {test['tiling']:<8} | {test['type']:<8} | {test['ch']:<2} | {test['size']:<8} | {status}")
        
        csv_record["Phase 2 Result"] = status_clean
        phase2_results_csv.append(csv_record)
        
        if not success:
            if not IS_WINDOWS:
                print(f"\n{RED}!!! SEMAPHORE FAILURE (Stopping to preserve driver state) !!!")
                print(f"Cmd: {cmd_sem}")
                print(f"Out: {output[:300]}{RESET}\n")
                semaphore_alive = False # LINUX SAFETY LATCH
            else:
                # On Windows, we just complain but keep going
                print(f"{RED}   -> FAILED (Continuing run...){RESET}")

    # Save Phase 2 CSV
    if phase2_results_csv:
        with open(SUMMARY_FILE_P2, 'w', newline='') as f:
            # Ensure columns are in a nice order
            fieldnames = ["API", "Test Name", "Mode", "Tiling", "Data Type", "Channels", "Dimensions", "Phase 2 Result", "Command"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(phase2_results_csv)

    print(f"\nPhase 2 Complete. Results saved to {SUMMARY_FILE_P2}")

# ---------------------------------------------------------
# ENTRY
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', choices=['phase1', 'phase2', 'all'], default='all')
    parser.add_argument('--linear', action='store_true', help='Include LINEAR tiling tests')
    
    args = parser.parse_args()

    if args.mode in ['phase1', 'all']:
        success = run_phase_1(include_linear=args.linear)
        if not success:
            sys.exit(1)
            
    if args.mode in ['phase2', 'all']:
        run_phase_2()