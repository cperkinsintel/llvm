import sys
import os
import re

def get_build_relative_path(full_path):
    """
    Normalizes a path to be relative to the 'build' directory using a regex.
    """
    normalized_path = os.path.normpath(full_path).replace(os.sep, '/')
    match = re.search(r'(/build/.*)', normalized_path)
    if match:
        return match.group(1)
    return normalized_path

def main():
    if len(sys.argv) != 4:
        print("Usage: python filter_manifest.py <glob_manifest> <deps_file> <output_manifest>")
        sys.exit(1)

    glob_manifest_path = sys.argv[1]
    deps_file_path = sys.argv[2]
    output_manifest_path = sys.argv[3]

    # Build the set of required dependencies, normalized to start with '/build/'.
    required_deps = set()
    with open(deps_file_path, 'r') as f:
        content = f.read().replace('\\\n', ' ')
        paths = [p for p in content.split() if os.path.isabs(p)]
        for path in paths:
            required_deps.add(get_build_relative_path(path))
    
    print(f"Loaded {len(required_deps)} required dependencies (normalized).")

    # Filter the glob_manifest.
    final_manifest = []
    with open(glob_manifest_path, 'r') as f:
        for line in f:
            glob_path_raw = line.strip()
            if glob_path_raw:
                # Keep all .bc files unconditionally. Their paths are absolute.
                if glob_path_raw.endswith('.bc'):
                    final_manifest.append(glob_path_raw)
                    continue

                # For headers, prepend '/build/' to the relative path from the
                # glob manifest to create a string that can be compared
                # against the required_deps set.
                glob_path_for_comparison = "/build/" + glob_path_raw.replace(os.sep, '/')
                
                if glob_path_for_comparison in required_deps:
                    # If it matches, add the ORIGINAL absolute path from the glob manifest
                    # We need the absolute path for the final #embed step.
                    final_manifest.append(glob_path_raw)

    # Write the final filtered manifest.
    with open(output_manifest_path, 'w') as f:
        for path in sorted(final_manifest):
            f.write(path + '\n')
    
    print(f"Wrote {len(final_manifest)} filtered paths to {output_manifest_path}.")

if __name__ == "__main__":
    main()