"""
build_nibble.py
Compiles nibble V4 (Tier 3/4 + Gap Closure) into a 64-bit DLL via MinGW.

Run once:
    cd f:\\.life_sciences\\keybox\\box\\nibble
    python build_nibble.py
"""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SOURCES = [
    os.path.join(HERE, "nibble.c"),
    os.path.join(HERE, "nibble_tier3.c"),
    os.path.join(HERE, "nibble_tier4.c"),
    os.path.join(HERE, "nibble_gaps.c"),
]

OUTPUT_DLL = os.path.join(HERE, "nibble.dll")

GCC_CMD = [
    "x86_64-w64-mingw32-gcc",
    "-O3",
    "-march=native",
    "-ffast-math",
    "-shared",
    "-o", OUTPUT_DLL,
    f"-I{HERE}",
] + SOURCES + ["-lm"]

if __name__ == "__main__":
    print("Compiling nibble V4 (Tier 3/4 + Gap Closure)...")
    print("Sources:", [os.path.basename(s) for s in SOURCES])
    result = subprocess.run(GCC_CMD, capture_output=True, text=True)
    if result.returncode != 0:
        print("COMPILE ERROR:")
        print(result.stderr)
        sys.exit(1)
    print(f"Built: {OUTPUT_DLL}")
    size_kb = os.path.getsize(OUTPUT_DLL) // 1024
    print(f"DLL size: {size_kb} KB")
    print("Done. nibble_bridge.py will load this DLL automatically.")
