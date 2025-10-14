import sys
import os

def normalize_path_for_comparison(full_path):
    """
    Normalizes a path to be relative to the 'build' directory.
    """
    normalized_path = os.path.normpath(full_path)
    anchor = "/build/"
    try:
        index = normalized_path.index(anchor)
        return normalized_path[index:]
    except ValueError:
        return normalized_path

def main():
    if len(sys.argv) != 4:
        print("Usage: python filter_manifest.py <glob_manifest> <deps_file> <output_manifest>")
        sys.exit(1)

    glob_manifest_path = sys.argv[1]
    deps_file_path = sys.argv[2]
    output_manifest_path = sys.argv[3]


    print(f"DEBUG: Reading glob_manifest from absolute path: {os.path.abspath(glob_manifest_path)}")

    # 1. Build the set of required dependencies.
    required_deps = set()
    with open(deps_file_path, 'r') as f:
        content = f.read().replace('\\\n', ' ')
        paths = [p for p in content.split() if os.path.isabs(p)]
        for path in paths:
            required_deps.add(normalize_path_for_comparison(path))
    
    print(f"Loaded {len(required_deps)} required dependencies (normalized).")

    print("\n--- BEGIN DUMP OF required_deps SET ---")
    # We sort the list for a clean, deterministic output
    for dep in sorted(list(required_deps)):
        print(dep)
    print("--- END DUMP OF required_deps SET ---\n")

    # 2. Filter the glob_manifest with aggressive debugging.
    final_manifest = []
    print("\n--- AGGRESSIVE DEBUGGING: GLOB MANIFEST RAW BYTES ---")
    search_string = "ext/intel/math.hpp"
    print(f"Searching for string with repr: {repr(search_string)}")
    print("i wish I understood this")
    
    with open(glob_manifest_path, 'r') as f:
        #for i, line in enumerate(f):
        for line in f:
            glob_path_raw = line.strip()
            if glob_path_raw:
                #print(glob_path_raw)
                # Check for the substring and print the byte representation if found
                #if search_string in glob_path_raw:
                    #print(f"\n--- FOUND POTENTIAL MATCH ON LINE {i+1} ---")
                    #print(f"  RAW LINE REPR: {repr(glob_path_raw)}")
                    #print("---------------------------------")
                
                #  keep .bc unconditionally.
                if glob_path_raw.endswith('.bc'):
                    final_manifest.append(glob_path_raw)
                    continue

                # The original filtering logic
                glob_path_normalized = normalize_path_for_comparison(glob_path_raw)
                if glob_path_normalized in required_deps:
                    final_manifest.append(glob_path_raw)
    
    print("--- END DEBUG ---\n")

    # 3. Write the final filtered manifest.
    with open(output_manifest_path, 'w') as f:
        for path in sorted(final_manifest):
            f.write(path + '\n')
    
    print(f"Wrote {len(final_manifest)} filtered paths to {output_manifest_path}.")

if __name__ == "__main__":
    main()