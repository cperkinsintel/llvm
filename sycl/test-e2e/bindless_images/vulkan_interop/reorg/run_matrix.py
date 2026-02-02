import subprocess
import sys
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Map your binary names here
TESTS = [
    # Label                  Binary Name               Default Flags
    ("1D Unsampled Read",   "./vsu_1d_test.bin",       ""),
    ("1D Unsampled Write",  "./vsu_1d_w_test.bin",     ""),
    ("1D Sampled Read",     "./vss_1d_test.bin",       ""),

    ("2D Unsampled Read",   "./vsu_2d_test.bin",       ""),
    ("2D Unsampled Write",  "./vsu_2d_w_test.bin",     ""), 
    ("2D Sampled Read",     "./vss_2d_test.bin",       ""),
    
    #("3D Unsampled Read",   "./vsu_3d_test.bin",       ""),
    #("3D Unsampled Write",  "./vsu_3d_w_test.bin",     ""), 
    #("3D Sampled Read",     "./vss_3d_test.bin",       ""),

    ("2D Arithmetic",       "./vs_2d_arith.bin",       ""),
]

# (Width, Height) Tuples
# We now mix standard resolutions with "Stress" sizes
DIMENSIONS = [
    (16, 16),       # Tiny Square (Baseline)
    (1024, 768),    # Classic Rect 4:3 (Row Pitch != Width)
    (1920, 1080),   # Full HD (Common Stride padding issues)
    (13, 17),       # Prime Rect (Alignment Stress)
    (1024, 1024),   # Standard Power of Two
    # (3127, 123),  # "The Cliff" Hunter (Uncomment for deep stress)
]

TYPES = [
    "float", 
    "half", 
    "int32", 
    "uint32", 
    "int16", 
    "uint16", 
    "uint8", 
    "int8", 
    "unorm8"
]

CHANNELS = ["1", "2", "4"]

# ---------------------------------------------------------
# THE ENGINE
# ---------------------------------------------------------
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def run_cmd(cmd):
    try:
        # Run and capture output
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stdout + e.stderr

def main():
    print(f"{'TEST':<25} | {'TYPE':<8} | {'CH':<2} | {'SIZE':<10} | {'FLAGS':<20} | {'RESULT'}")
    print("-" * 105)
    
    failures = []

    semaphore_system_alive = True  # <--- Circuit Breaker Flag

    for label, binary, default_flags in TESTS:
        if not os.path.exists(binary.split()[0]):
            print(f"Skipping {label} (Binary not found)")
            continue

        for type_name in TYPES:
            for ch in CHANNELS:
                for w, h in DIMENSIONS:
                    
                    # Logic: Is this a 1D or 2D test?
                    is_1d = "1D" in label
                    
                    # Display String (e.g., "1024x768")
                    size_str = f"{w}" if is_1d else f"{w}x{h}"
                    
                    # Argument String (e.g., "1024 768")
                    # 1D takes just Width. 2D takes Width Height.
                    dim_args = f"{w}" if is_1d else f"{w} {h}"
                    
                    # Construct Command
                    flags = f"--type {type_name} --channels {ch} {dim_args}"
                    
                    # 1. Basic Run (Always run this)
                    cmd = f"{binary} {flags} {default_flags}"
                    success, output = run_cmd(cmd)
                    
                    status = f"{GREEN}PASS{RESET}" if success else f"{RED}FAIL{RESET}"
                    print(f"{label:<25} | {type_name:<8} | {ch:<2} | {size_str:<10} | {'Basic':<20} | {status}")
                    
                    if not success: failures.append(cmd)

                    # 2. Semaphore Run (Only if system is still alive)
                    if semaphore_system_alive:
                        cmd_sem = f"{cmd} --semaphores"
                        success, output = run_cmd(cmd_sem)
                        status = f"{GREEN}PASS{RESET}" if success else f"{RED}FAIL{RESET}"
                        print(f"{label:<25} | {type_name:<8} | {ch:<2} | {size_str:<10} | {'+Semaphores':<20} | {status}")
                        
                        if not success: 
                            failures.append(cmd_sem)
                            print(f"{RED}!!! SEMAPHORE FAILURE DETECTED. DISABLING FUTURE SEMAPHORE TESTS !!!{RESET}")
                            semaphore_system_alive = False
                    else:
                        print(f"{label:<25} | {type_name:<8} | {ch:<2} | {size_str:<10} | {'+Semaphores':<20} | {RED}SKIP (Poisoned){RESET}")

    print("-" * 105)
    if failures:
        print(f"{RED}FAILURES DETECTED:{RESET}")
        for f in failures:
            print(f"  {f}")
    else:
        print(f"{GREEN}ALL TESTS PASSED. FLAWLESS VICTORY.{RESET}")

if __name__ == "__main__":
    main()