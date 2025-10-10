import os
import argparse
import sys
import fnmatch

def main():
    parser = argparse.ArgumentParser(
        description="Generate SYCL Headers Resource C++ file."
    )
    # These arguments are always required
    parser.add_argument("-o", "--output", type=str, required=True, help="Output C++ file")
    parser.add_argument("-i", "--toolchain-dir", type=str, required=True, help="Path to toolchain root directory.")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix for virtual file locations")
    
    # These two arguments control the mode and are mutually exclusive
    parser.add_argument("-m", "--manifest-input", type=str, help="Build from this manifest (read-only).")
    parser.add_argument("--manifest-output", type=str, help="Glob for files and write them to this manifest.")
    parser.add_argument(
        "--blacklist",
        type=str,
        help="Path to a file containing glob patterns of resources to exclude."
    )
    
    args = parser.parse_args()

    if args.manifest_input and args.manifest_output:
        print("Error: --manifest-input and --manifest-output are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    blacklist_patterns = set()
    if args.blacklist:
        print(f"Loading blacklist from: {args.blacklist}")
        with open(args.blacklist, "r") as f:
            for line in f:
                pattern = line.strip()
                if pattern and not pattern.startswith('#'): # Ignore blank lines and comments
                    blacklist_patterns.add(pattern)

    toolchain_dir = os.path.abspath(args.toolchain_dir)
    
    manifest_to_write = open(args.manifest_output, "w") if args.manifest_output else open(os.devnull, "w")

    with manifest_to_write as manifest_out, open(args.output, "w") as out:
        out.write(
            """
#include <Resource.h>
namespace jit_compiler::resource {
const resource_file ToolchainFiles[] = {"""
        )

        def process_file(file_path):
            for pattern in blacklist_patterns:
                if fnmatch.fnmatch(file_path, pattern):
                    print(f"  -> Skipping blacklisted file: {file_path}")
                    return # Skip this file
                    
            manifest_out.write(file_path + '\n')
            out.write(
                f"""
        {{
        {{"{args.prefix}{os.path.relpath(file_path, toolchain_dir).replace(os.sep, "/")}"}} ,
        []() {{
            static const char data[] = {{
            #embed "{file_path}" if_empty(0)
                , 0}};
            return resource_string_view{{data}};
        }}()
        }},"""
            )

        if args.manifest_input:
            # MODE 3: Read from manifest
            print(f"Reading resource list from manifest: {args.manifest_input}")
            with open(args.manifest_input, "r") as manifest_file:
                for line in manifest_file:
                    file_path = line.strip()
                    if file_path:
                        process_file(file_path)
        else:
            # MODE 1 (glob) or 2 (glob and output)
            if args.manifest_output:
                print(f"Globbing for resources and writing manifest to: {args.manifest_output}")
            else:
                print("Globbing for resources (no manifest output)...")

            def process_dir(dir):
                for root, _, files in os.walk(dir):
                    for file in files:
                        process_file(os.path.join(root, file))

            process_dir(os.path.join(args.toolchain_dir, "include/"))
            process_dir(os.path.join(args.toolchain_dir, "lib/clang/"))
            process_dir(os.path.join(args.toolchain_dir, "lib/clc/"))
            # ... any other globbing logic ...

        out.write(
            f"""
}};

unsigned long long NumToolchainFiles = size(ToolchainFiles);
resource_string_view ToolchainPrefix{{"{args.prefix}"}};
}} // namespace jit_compiler::resource
"""
        )

if __name__ == "__main__":
    main()