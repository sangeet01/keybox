"""
build_nibble.py
Compiles nibble.c into a 64-bit Python extension (_nibble_ext) using cffi.
cffi uses the same MSVC 64-bit toolchain that CPython was compiled with,
so the resulting .pyd is guaranteed to be compatible.

Run once:
    cd f:\\.life_sciences\\keybox\\box\\nibble
    python build_nibble.py
"""
import os
import sys
from cffi import FFI

HERE = os.path.dirname(os.path.abspath(__file__))

ffi = FFI()

# Declare the public C API for Python consumption.
ffi.cdef("""
#define N_CHANNELS 11

typedef struct {
    int dim_x;
    int dim_y;
    int dim_z;
    float resolution;
    float *data;
} NibbleGrid;

typedef enum {
    CH_STERIC_DEMAND = 0,
    CH_ELEC_DEMAND   = 1,
    CH_HBA_DEMAND    = 2,
    CH_HBD_DEMAND    = 3,
    CH_LIPO_DEMAND   = 4,
    CH_AROM_DEMAND   = 5,
    CH_METAL_DEMAND  = 6,
    CH_CATION_DEMAND = 7,
    CH_ANION_DEMAND  = 8,
    CH_PHOBIC_CORE   = 9,
    CH_SOLVENT_EXPO  = 10
} NibbleChannel;

NibbleGrid* nibble_create_grid(int dx, int dy, int dz, float res);
void        nibble_free_grid(NibbleGrid *g);
void        nibble_reset_steric(NibbleGrid *g);

int   nibble_load_pdb_pocket(NibbleGrid *g, const char *pdb_path,
                              float cx, float cy, float cz,
                              float range, float blur_radius);

float nibble_compute_affinity(const NibbleGrid *pocket, const NibbleGrid *drug);

void  nibble_update_local(NibbleGrid *g, int cx, int cy, int cz,
                           int radius, const float *diff_data);

void  nibble_project_atom(NibbleGrid *g, float x, float y, float z,
                           float radius, const float *channel_values);
""")

# Point cffi at the C source so it compiles it directly.
ffi.set_source(
    "_nibble_ext",
    '#include "nibble.h"',
    sources=[os.path.join(HERE, "nibble.c")],
    include_dirs=[HERE],
    libraries=["m"] if sys.platform != "win32" else [],
)

if __name__ == "__main__":
    print("Compiling nibble C engine via cffi (64-bit MSVC)...")
    out = ffi.compile(tmpdir=HERE, verbose=True)
    print(f"Built: {out}")
    print("Done. nibble_bridge.py will now load _nibble_ext instead of nibble.dll")
