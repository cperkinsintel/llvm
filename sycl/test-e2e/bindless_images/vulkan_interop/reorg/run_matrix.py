import subprocess
import sys
import os
import json
import argparse

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SURVIVORS_FILE = "survivors.json"

# Map your binary names here
TESTS = [
    # Label                  Binary Name               Default Flags
    ("1D Unsampled Read",   "./vsu_1d_test.bin",       ""),
    ("1D Unsampled Write",  "./vsu_1d_w_test.bin",     ""),
    ("1D Sampled Read",     "./vss_1d_test.bin",       ""),

    ("2D Unsampled Read",   "./vsu_2d_test.bin",       ""),
    ("2D Unsampled Write",  "./vsu_2d_w_test.bin",     ""), 
    ("2D Sampled Read",     "./vss_2d_test.bin",       ""),
    
    # ("3D Unsampled Read",   "./vsu_3d_test.bin",       ""),
    # ("3D Unsampled Write",  "./vsu_3d_w_test.bin",     ""), 
    # ("3D Sampled Read",     "./vss_3d_test.bin",       ""),

    ("2D Arithmetic",       "./vs_2d_arith.bin",       ""),
]

# (Width, Height) Tuples
DIMENSIONS = [
    (16, 16),       # Tiny Square
    (1024, 768),    # Classic Rect (4:3)
    (1920, 1080),   # Full HD (Stride Stress)
    (13, 17),       # Prime Rect (Alignment Stress)
    (1024, 1024),   # Power of Two
    # (3127, 123),  # The Cliff Hunter
]

TYPES = [
    "float", "half", "int32", "uint32", 
    "int16", "uint16", "uint8", "int8", "unorm8"
]

CHANNELS = ["1", "2", "4"]

# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stdout + e.stderr

# ---------------------------------------------------------
# PHASE 1: QUALIFICATION (LOGIC ONLY)
# ---------------------------------------------------------
def run_phase_1():
    print(f"\n{YELLOW}=== PHASE 1: LOGIC QUALIFICATION (No Semaphores) ==={RESET}")
    print(f"{'TEST':<25} | {'TYPE':<8} | {'CH':<2} | {'SIZE':<10} | {'RESULT'}")
    print("-" * 80)
    
    survivors = []
    failures = 0

    for label, binary, default_flags in TESTS:
        if not os.path.exists(binary.split()[0]):
            print(f"Skipping {label} (Binary not found)")
            continue

        for type_name in TYPES:
            for ch in CHANNELS:
                for w, h in DIMENSIONS:
                    
                    # 1D vs 2D Logic
                    is_1d = "1D" in label
                    size_str = f"{w}" if is_1d else f"{w}x{h}"
                    dim_args = f"{w}" if is_1d else f"{w} {h}"
                    
                    # Construct Basic Command
                    flags = f"--type {type_name} --channels {ch} {dim_args}"
                    if "arith" in binary: flags += f"x{size_str}" # Special case for arithmetic if needed, or rely on main() logic

                    full_cmd = f"{binary} {flags} {default_flags}"
                    
                    # Execute
                    success, output = run_cmd(full_cmd)
                    
                    status = f"{GREEN}PASS{RESET}" if success else f"{RED}FAIL{RESET}"
                    print(f"{label:<25} | {type_name:<8} | {ch:<2} | {size_str:<10} | {status}")
                    
                    if success:
                        # Save the command template for Phase 2
                        survivors.append({
                            "label": label,
                            "type": type_name,
                            "ch": ch,
                            "size": size_str,
                            "cmd": full_cmd
                        })
                    else:
                        failures += 1

    # Save Survivors
    with open(SURVIVORS_FILE, 'w') as f:
        json.dump(survivors, f, indent=2)
    
    print("-" * 80)
    print(f"Phase 1 Complete. {len(survivors)} passed. {failures} failed.")
    print(f"Survivors saved to {SURVIVORS_FILE}")
    return len(survivors) > 0

# ---------------------------------------------------------
# PHASE 2: VERIFICATION (SYNC STRESS)
# ---------------------------------------------------------
def run_phase_2():
    print(f"\n{YELLOW}=== PHASE 2: SYNC VERIFICATION (With Semaphores) ==={RESET}")
    
    if not os.path.exists(SURVIVORS_FILE):
        print(f"{RED}Error: {SURVIVORS_FILE} not found. Run Phase 1 first.{RESET}")
        return

    with open(SURVIVORS_FILE, 'r') as f:
        survivors = json.load(f)

    print(f"Loaded {len(survivors)} qualified tests.")
    print(f"{'TEST':<25} | {'TYPE':<8} | {'CH':<2} | {'SIZE':<10} | {'RESULT'}")
    print("-" * 80)

    semaphore_system_alive = True
    
    for test in survivors:
        if not semaphore_system_alive:
            print(f"{test['label']:<25} | {test['type']:<8} | {test['ch']:<2} | {test['size']:<10} | {RED}SKIP (Poisoned){RESET}")
            continue

        # Append Semaphore Flag
        cmd_sem = f"{test['cmd']} --semaphores"
        
        success, output = run_cmd(cmd_sem)
        status = f"{GREEN}PASS{RESET}" if success else f"{RED}FAIL{RESET}"
        
        print(f"{test['label']:<25} | {test['type']:<8} | {test['ch']:<2} | {test['size']:<10} | {status}")
        
        if not success:
            print(f"\n{RED}!!! CRITICAL SEMAPHORE FAILURE DETECTED !!!")
            print(f"Failed Command: {cmd_sem}")
            print(f"Stopping Phase 2 to preserve system state.{RESET}\n")
            semaphore_system_alive = False
            # We do NOT delete the survivors file, so you can retry specific ones if needed.

# ---------------------------------------------------------
# MAIN DISPATCHER
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vulkan/SYCL Matrix Runner")
    parser.add_argument('mode', nargs='?', choices=['phase1', 'phase2', 'all'], default='all', help='Phase to run')
    args = parser.parse_args()

    if args.mode == 'phase1' or args.mode == 'all':
        if not run_phase_1():
            print("No tests passed Phase 1. Aborting.")
            sys.exit(1)
            
    if args.mode == 'phase2' or args.mode == 'all':
        run_phase_2()