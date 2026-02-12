"""
Vulkan/SYCL Interop Matrix Test Runner

PURPOSE:
    Systematically tests all combinations of:
    - Test types (1D/2D/3D Read/Write)
    - Data types (float, half, int32, uint32, int16, uint16, uint8, int8, unorm8)
    - Channel counts (1, 2, 4)
    - Dimensions (various sizes including edge cases)
    - Tiling modes (OPTIMAL vs LINEAR)
    - Image access modes (sampled vs unsampled)

PHASES:
    Phase 1: Logic Qualification (no semaphores)
        - Tests basic functionality
        - Generates survivors.json for tests that pass
        - Safe to run (won't hang system)
    
    Phase 2: Sync Verification (with semaphores)
        - Only runs on Phase 1 survivors
        - Tests synchronization correctness
        - Can hang on bad configs (hence the filtering)

USAGE EXAMPLES:

    # Basic run (OPTIMAL tiling, unsampled images, both phases)
    python run_matrix.py

    # Run only Phase 1 with default settings
    python run_matrix.py phase1

    # Test LINEAR tiling (important for some platforms)
    python run_matrix.py phase1 --linear

    # Test sampled images (Read tests only, Write tests auto-skipped)
    python run_matrix.py phase1 --sampled

    # Test sampled images with LINEAR tiling
    python run_matrix.py phase1 --sampled --linear

    # Run Phase 2 on default survivors
    python run_matrix.py phase2

    # Run Phase 2 on specific survivors file
    python run_matrix.py phase2 --survivors survivors_linear.json

    # Run both phases with specific configuration
    python run_matrix.py all --linear

OUTPUT FILES:
    survivors[_MODE].json      - Tests that passed (input for Phase 2)
    casualties[_MODE].json     - Tests that failed (for bug reports)
    summary[_MODE].csv         - Spreadsheet-friendly results
    phase2[_MODE].json         - Phase 2 results with semaphores

NOTES:
    - Write tests are automatically skipped when using --sampled
      (they don't support sampled image writes and will error)
    - Phase 2 stops immediately on semaphore failure to prevent system hangs
    - Timeout protection prevents indefinite hangs (30s default)
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
SUMMARY_FILE = "summary.csv"

# Test Configuration: (Label, Binary Name, Test Type)
# Test Flags: "both" means test supports unsampled (default) and  --sampled,
#  "unsampled_only" means it does not support --sampled flag. 
TESTS = [
    # Label                          Binary Name                  Test Type
    ("1D Read",                     "./vsr_1d_test.bin",         "both"),
    ("1D Write",                    "./vsu_1d_w_test.bin",       "unsampled_only"),

    ("2D Read",                     "./vsr_2d_test.bin",         "both"),
    ("2D Write",                    "./vsu_2d_w_test.bin",       "unsampled_only"), 
    
    ("3D Read",                     "./vsr_3d_test.bin",         "both"),
    ("3D Write",                    "./vsu_3d_w_test.bin",       "unsampled_only"), 

    ("2D Arithmetic",               "./vs_2d_arith.bin",         "both"),
]

# (Width, Height) Tuples
# DIMENSIONS = [
#     (16, 16),        # Tiny Square
#     (1024, 768),     # Classic Rect (4:3)
#     #(1920, 1080),    # Full HD (Stride Stress) -- causes hang in unorm linux
#     (13, 17),        # Prime Rect (Alignment Stress)
#     (1024, 1024),    # Power of Two
#     # (3127, 123),   # The Cliff Hunter
# ]

DIMENSIONS = [(16,16)]

# DIMENSIONS = [
#      (15, 16),
#      (16, 16),
#      (16,17),
#      (31,32),
#      (32,32),
#      (32,33),
#      (63,64),
#      (64,64),
#      (64,65),
# ]

TYPES = [
    "float", "half", "int32", "uint32", 
    "int16", "uint16", "uint8", "int8", "unorm8"
]

CHANNELS = ["1", "2", "4"]

# ---------------------------------------------------------
# UTILS & PLATFORM HANDLING
# ---------------------------------------------------------
IS_WINDOWS = (os.name == 'nt')

# Colors
if IS_WINDOWS:
    os.system('color') 
    
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def get_platform_binary(linux_style_path):
    """
    Converts './app.bin' to '.\\app.exe' if running on Windows.
    Keeps it as './app.bin' on Linux.
    """
    if not IS_WINDOWS:
        return linux_style_path
    
    # Windows Conversion
    base_name = linux_style_path.replace("./", "")
    base_name = base_name.replace(".bin", ".exe")
    
    return f".\\{base_name}"

def run_cmd(cmd, timeout=30):
    """Execute command with timeout to prevent hangs"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            timeout=timeout
        )
        # Decode output ourselves with error handling
        output = result.stdout.decode('utf-8', errors='replace')
        return True, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: Test hung (likely semaphore/driver issue)"
    except subprocess.CalledProcessError as e:
        # Decode error output safely
        stdout = e.stdout.decode('utf-8', errors='replace') if e.stdout else ""
        stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
        return False, stdout + stderr

# ---------------------------------------------------------
# PHASE 1: QUALIFICATION (LOGIC ONLY)
# ---------------------------------------------------------
def run_phase_1(use_linear=False, test_sampled=False):
    mode_desc = []
    if use_linear:
        mode_desc.append("LINEAR tiling")
    if test_sampled:
        mode_desc.append("SAMPLED images")
    mode_str = " + ".join(mode_desc) if mode_desc else "Default (OPTIMAL, UNSAMPLED)"
    
    print(f"\n{YELLOW}=== PHASE 1: LOGIC QUALIFICATION ({mode_str}) ==={RESET}")
    print(f"{'TEST':<25} | {'MODE':<8} | {'TYPE':<8} | {'CH':<2} | {'SIZE':<10} | {'RESULT'}")
    print("-" * 90)
    
    survivors = []
    casualties = []
    all_results = []
    
    failures = 0

    for label, raw_binary, test_type in TESTS:
        # 1. Adapt Binary Name to OS
        binary = get_platform_binary(raw_binary)

        # 2. Check existence
        check_path = binary.replace("./", "").replace(".\\", "")
        if not os.path.exists(check_path):
            print(f"Skipping {label} (Binary not found: {binary})")
            continue
        
        # 3. Determine if this test should run in the current mode
        #    KEY FIX: Skip unsampled_only tests when --sampled is specified
        if test_type == "unsampled_only" and test_sampled:
            print(f"Skipping {label} (Does not support --sampled)")
            continue
        
        # 4. Determine the actual mode to report/use
        if test_type == "both":
            # This test supports both modes, use what was requested
            mode = "both" if test_sampled else "unsampled"
        else:
            # unsampled_only tests always run as unsampled
            mode = "unsampled"

        for type_name in TYPES:
            for ch in CHANNELS:
                for w, h in DIMENSIONS:
                    
                    # 1D vs 2D/3D Logic
                    is_1d = "1D" in label
                    is_3d = "3D" in label
                    
                    # Unified Size String
                    if is_1d:
                        size_str = f"{w}"
                    elif is_3d:
                        # For 3D, use w x h x h (cube-ish)
                        size_str = f"{w}x{h}x{h}"
                    else:
                        size_str = f"{w}x{h}"
                    
                    # Construct Command
                    flags = f"--type {type_name} --channels {ch}"
                    
                    # Only add --sampled if the test supports it AND we're testing sampled mode
                    if test_type == "both" and test_sampled:
                        flags += " --sampled"
                    
                    if use_linear:
                        flags += " --linear"
                    
                    full_cmd = f"{binary} {flags} {size_str}"
                    
                    # Execute
                    success, output = run_cmd(full_cmd)
                    
                    status = f"{GREEN}PASS{RESET}" if success else f"{RED}FAIL{RESET}"
                    status_clean = "PASS" if success else "FAIL"
                    
                    mode_display = f"{mode[:3].upper()}"  # "SAM" or "UNS"
                    print(f"{label:<25} | {mode_display:<8} | {type_name:<8} | {ch:<2} | {size_str:<10} | {status}")
                    
                    # Record Data
                    record = {
                        "label": label,
                        "mode": mode,
                        "type": type_name,
                        "ch": ch,
                        "size": size_str,
                        "linear": use_linear,
                        "cmd": full_cmd
                    }
                    
                    all_results.append({
                        "Test Name": label,
                        "Mode": mode,
                        "Data Type": type_name,
                        "Channels": ch,
                        "Dimensions": size_str,
                        "Linear": "Yes" if use_linear else "No",
                        "Phase 1 Result": status_clean,
                        "Command": full_cmd
                    })

                    if success:
                        survivors.append(record)
                    else:
                        failures += 1
                        casualties.append({**record, "output": output[:500]})  # Truncate error

    # Generate filenames with mode suffix
    suffix = []
    if use_linear:
        suffix.append("linear")
    if test_sampled:
        suffix.append("both")
    suffix_str = "_" + "_".join(suffix) if suffix else ""
    
    survivors_file = f"survivors{suffix_str}.json"
    casualties_file = f"casualties{suffix_str}.json"
    summary_file = f"summary{suffix_str}.csv"

    # 1. Save Survivors (JSON)
    with open(survivors_file, 'w') as f:
        json.dump(survivors, f, indent=2)

    # 2. Save Casualties (JSON)
    with open(casualties_file, 'w') as f:
        json.dump(casualties, f, indent=2)

    # 3. Save Summary (CSV)
    if all_results:
        keys = all_results[0].keys()
        with open(summary_file, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_results)
    
    print("-" * 90)
    print(f"Phase 1 Complete. {len(survivors)} passed. {failures} failed.")
    print(f"Files Generated:")
    print(f"  - {survivors_file} (Ready for Phase 2)")
    print(f"  - {casualties_file} (Failures for JIRA)")
    print(f"  - {summary_file}    (Report for Management)")
    
    return len(survivors) > 0, survivors_file

# ---------------------------------------------------------
# PHASE 2: VERIFICATION (SYNC STRESS)
# ---------------------------------------------------------
def run_phase_2(survivors_file=None):
    print(f"\n{YELLOW}=== PHASE 2: SYNC VERIFICATION (With Semaphores) ==={RESET}")
    
    # Auto-detect survivors file if not specified
    if survivors_file is None:
        survivors_file = SURVIVORS_FILE
    
    if not os.path.exists(survivors_file):
        print(f"{RED}Error: {survivors_file} not found. Run Phase 1 first.{RESET}")
        return

    with open(survivors_file, 'r') as f:
        survivors = json.load(f)

    print(f"Loaded {len(survivors)} qualified tests from {survivors_file}")
    print(f"{'TEST':<25} | {'MODE':<8} | {'TYPE':<8} | {'CH':<2} | {'SIZE':<10} | {'RESULT'}")
    print("-" * 90)

    semaphore_system_alive = True
    phase2_results = []
    
    for test in survivors:
        if not semaphore_system_alive:
            print(f"{test['label']:<25} | {test.get('mode', 'UNS')[:3].upper():<8} | {test['type']:<8} | {test['ch']:<2} | {test['size']:<10} | {RED}SKIP (Poisoned){RESET}")
            phase2_results.append({
                **test,
                "phase2_result": "SKIPPED",
                "phase2_reason": "Semaphore system poisoned by previous failure"
            })
            continue

        cmd_sem = f"{test['cmd']} --semaphores"
        
        success, output = run_cmd(cmd_sem, timeout=30)
        status = f"{GREEN}PASS{RESET}" if success else f"{RED}FAIL{RESET}"
        
        mode_display = test.get('mode', 'unsampled')[:3].upper()
        print(f"{test['label']:<25} | {mode_display:<8} | {test['type']:<8} | {test['ch']:<2} | {test['size']:<10} | {status}")
        
        phase2_results.append({
            **test,
            "phase2_result": "PASS" if success else "FAIL",
            "phase2_cmd": cmd_sem
        })
        
        if not success:
            print(f"\n{RED}!!! CRITICAL SEMAPHORE FAILURE DETECTED !!!")
            print(f"Failed Command: {cmd_sem}")
            print(f"Output: {output[:500]}")
            print(f"Stopping Phase 2 to preserve system state.{RESET}\n")
            semaphore_system_alive = False
    
    # Save Phase 2 results
    base_name = survivors_file.replace("survivors", "phase2").replace(".json", "")
    phase2_file = f"{base_name}.json"
    
    with open(phase2_file, 'w') as f:
        json.dump(phase2_results, f, indent=2)
    
    print(f"\nPhase 2 results saved to: {phase2_file}")

# ---------------------------------------------------------
# MAIN DISPATCHER
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vulkan/SYCL Matrix Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_matrix.py                           # Run both phases (default settings)
  python run_matrix.py phase1                    # Run Phase 1 only (OPTIMAL, unsampled)
  python run_matrix.py phase1 --linear           # Run Phase 1 with LINEAR tiling
  python run_matrix.py phase1 --sampled          # Run Phase 1 with sampled images (Read tests only)
  python run_matrix.py phase1 --linear --sampled # Run Phase 1 with both LINEAR and sampled
  python run_matrix.py phase2                    # Run Phase 2 (semaphores) on survivors
  python run_matrix.py all --linear              # Run both phases with LINEAR tiling
        """
    )
    
    parser.add_argument('mode', nargs='?', choices=['phase1', 'phase2', 'all'], 
                        default='all', help='Operation mode')
    parser.add_argument('--linear', action='store_true', 
                        help='Use LINEAR tiling (default: OPTIMAL)')
    parser.add_argument('--sampled', action='store_true', 
                        help='Test sampled image paths (Write tests auto-skipped)')
    parser.add_argument('--survivors', type=str, 
                        help='Specify survivors file for phase2 (auto-detected if not specified)')
    
    args = parser.parse_args()

    # Run phases
    survivors_file = None
    
    if args.mode == 'phase1' or args.mode == 'all':
        success, survivors_file = run_phase_1(use_linear=args.linear, test_sampled=args.sampled)
        if not success:
            print(f"{RED}No tests passed Phase 1. Aborting.{RESET}")
            sys.exit(1)
            
    if args.mode == 'phase2' or args.mode == 'all':
        # Use specified file, or the one from phase1, or default
        file_to_use = args.survivors or survivors_file or SURVIVORS_FILE
        run_phase_2(file_to_use)

    print(f"\n{GREEN}=== TEST RUN COMPLETE ==={RESET}")