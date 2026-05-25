"""
build_nibble.py
Compiles nibble V4 + PPI extension into a shared library.

Windows (MinGW):
    cd box/nibble
    python build_nibble.py

Linux / macOS:
    cd box/nibble
    python build_nibble.py

The script auto-detects the platform and uses the right compiler/flags.
Output: nibble.dll (Windows) or libnibble.so (Linux/macOS)
"""
import subprocess
import sys
import os
import platform

HERE = os.path.dirname(os.path.abspath(__file__))

SOURCES = [
    os.path.join(HERE, "nibble.c"),
    os.path.join(HERE, "nibble_tier3.c"),
    os.path.join(HERE, "nibble_tier4.c"),
    os.path.join(HERE, "nibble_gaps.c"),
    os.path.join(HERE, "nibble_ppi.c"),      # PPI extension — new
]

# -----------------------------------------------------------------------
# Platform detection
# -----------------------------------------------------------------------
IS_WINDOWS = (os.name == 'nt')
IS_MAC     = (platform.system() == 'Darwin')
IS_LINUX   = (platform.system() == 'Linux')

if IS_WINDOWS:
    OUTPUT_LIB = os.path.join(HERE, "nibble.dll")
    COMPILER   = "x86_64-w64-mingw32-gcc"
    SHARED_FLAG = "-shared"
elif IS_MAC:
    OUTPUT_LIB  = os.path.join(HERE, "libnibble.so")
    COMPILER    = "gcc"
    SHARED_FLAG = "-dynamiclib"
else:  # Linux
    OUTPUT_LIB  = os.path.join(HERE, "libnibble.so")
    COMPILER    = "gcc"
    SHARED_FLAG = "-shared"

# -----------------------------------------------------------------------
# Compile command
# -----------------------------------------------------------------------
GCC_CMD = [
    COMPILER,
    "-O3",
    "-march=native",
    "-ffast-math",
    SHARED_FLAG,
    "-fPIC",            # position-independent code (required for .so)
    "-o", OUTPUT_LIB,
    f"-I{HERE}",
] + SOURCES + ["-lm"]

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Platform: {platform.system()}")
    print(f"Compiler: {COMPILER}")
    print(f"Output:   {OUTPUT_LIB}")
    print(f"Sources:  {[os.path.basename(s) for s in SOURCES]}")
    print()

    # Verify all source files exist
    missing = [s for s in SOURCES if not os.path.exists(s)]
    if missing:
        print("ERROR: missing source files:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print("Compiling nibble V4 + PPI extension...")
    result = subprocess.run(GCC_CMD, capture_output=True, text=True)

    if result.returncode != 0:
        print("COMPILE ERROR:")
        print(result.stderr)
        sys.exit(1)

    size_kb = os.path.getsize(OUTPUT_LIB) // 1024
    print(f"Built: {OUTPUT_LIB} ({size_kb} KB)")
    print()
    print("New PPI functions available:")
    print("  nibble_detect_interface()       — inter-residue contact detection")
    print("  nibble_load_ppi_surface()       — irregular surface patch loading")
    print("  nibble_sc_score()               — Lawrence-Coleman shape complementarity")
    print("  nibble_score_hotspots()         — alanine-scanning proxy hotspot detection")
    print("  nibble_ppi_affinity()           — composite PPI binding score")
    print("  nibble_load_ppi_inhibitor_site()— pharmacophore for inhibitor design")
    print()
    print("Done. nibble_bridge.py will load the updated library automatically.")
