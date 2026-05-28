"""
nibble_protac_bridge.py
-----------------------
Python ctypes bindings for the Nibble PROTAC ternary complex extension.

Plugs into the existing NibbleEngine / NibblePPIEngine architecture.
Loads the same shared library. Zero changes to existing files.

Full PROTAC workflow:
  1. Load target protein pocket  → NibbleGrid  (existing nibble_bridge)
  2. Load E3 ligase pocket       → NibbleGrid  (existing nibble_bridge)
  3. Create PROTAC molecule      → PROTAC_Molecule (this file)
  4. Define ternary geometry     → PROTAC_TernaryGeometry
  5. Run ternary trajectory      → nibble_protac_trajectory()
  6. Score and rank              → PROTACResult

Usage
-----
  from nibble_protac_bridge import make_protac_engine, PROTACMolecule

  engine = make_protac_engine()

  # Define PROTAC (warhead_a | linker | warhead_b split by atom indices)
  protac = PROTACMolecule(
      pdb_path="protac_001.pdb",
      split_a=18,        # warhead A = atoms 0–17
      split_b=28,        # linker   = atoms 18–27, warhead B = 28+
      linker_smiles="CCOCCOCCN",
  )

  # Score against target (BRD4) and E3 (CRBN)
  result = engine.score_ternary(
      target_pdb="brd4.pdb",
      e3_pdb="crbn.pdb",
      target_centre=(12.3, 8.1, 22.4),
      e3_centre=(4.5, 31.2, 9.8),
      protac=protac,
      run_trajectory=True,
  )
  print(result)

  # PROTAC design: scan linker lengths
  results = engine.scan_linker_lengths(
      target_pdb="brd4.pdb",
      e3_pdb="crbn.pdb",
      target_centre=(12.3, 8.1, 22.4),
      e3_centre=(4.5, 31.2, 9.8),
      warhead_a_atoms=18,
      warhead_b_atoms=12,
      linker_atoms_range=range(4, 20, 2),
  )

Author: Khukuri / KeyBox project (sangeet01)
License: Apache 2.0 with Commons Clause
"""

import ctypes
import os
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from nibble_bridge import (
    NibbleEngine,
    NibbleGridStruct,
    NibbleMolStruct,
    N_CHANNELS,
)
from nibble_ppi_bridge import (
    PPIInterfaceStruct,
    PPIScoreStruct,
    NibblePPIEngine,
)


# ---------------------------------------------------------------------------
# ctypes mirrors of PROTAC C structs
# ---------------------------------------------------------------------------

_PROTAC_MAX_LINKER_ATOMS = 80

class PROTACLinkerStruct(ctypes.Structure):
    _fields_ = [
        ("n_atoms",           ctypes.c_int),
        ("contour_length",    ctypes.c_float),
        ("persistence_length",ctypes.c_float),
        ("flexibility",       ctypes.c_float),
        ("coords",            ctypes.c_float * (_PROTAC_MAX_LINKER_ATOMS * 3)),
        ("ch_vals",           ctypes.c_float * (_PROTAC_MAX_LINKER_ATOMS * 16)),
    ]


class PROTACMoleculeStruct(ctypes.Structure):
    _fields_ = [
        ("warhead_a",  ctypes.POINTER(NibbleMolStruct)),
        ("warhead_b",  ctypes.POINTER(NibbleMolStruct)),
        ("linker",     PROTACLinkerStruct),
        ("sep_x",      ctypes.c_float),
        ("sep_y",      ctypes.c_float),
        ("sep_z",      ctypes.c_float),
        ("separation", ctypes.c_float),
    ]


class PROTACTernaryGeometryStruct(ctypes.Structure):
    _fields_ = [
        ("target_cx",          ctypes.c_float),
        ("target_cy",          ctypes.c_float),
        ("target_cz",          ctypes.c_float),
        ("e3_cx",              ctypes.c_float),
        ("e3_cy",              ctypes.c_float),
        ("e3_cz",              ctypes.c_float),
        ("pocket_sep_x",       ctypes.c_float),
        ("pocket_sep_y",       ctypes.c_float),
        ("pocket_sep_z",       ctypes.c_float),
        ("pocket_separation",  ctypes.c_float),
        ("required_linker_ete",ctypes.c_float),
        ("has_neo_ppi",        ctypes.c_int),
        ("neo_ppi_score",      ctypes.c_float),
    ]


class PROTACTernaryScoreStruct(ctypes.Structure):
    _fields_ = [
        ("score_warhead_a",    ctypes.c_float),
        ("score_warhead_b",    ctypes.c_float),
        ("linker_penalty",     ctypes.c_float),
        ("linker_end_to_end",  ctypes.c_float),
        ("linker_feasibility", ctypes.c_float),
        ("alpha",              ctypes.c_float),
        ("hook_effect",        ctypes.c_int),
        ("neo_ppi_score",      ctypes.c_float),
        ("deg_efficiency",     ctypes.c_float),
        ("composite_score",    ctypes.c_float),
        ("w_a",                ctypes.c_float),
        ("w_b",                ctypes.c_float),
        ("w_linker",           ctypes.c_float),
        ("w_coop",             ctypes.c_float),
        ("w_neo_ppi",          ctypes.c_float),
    ]


# ---------------------------------------------------------------------------
# Python result container
# ---------------------------------------------------------------------------

@dataclass
class PROTACResult:
    """Full result of a PROTAC ternary complex scoring run."""

    # Composite
    composite_score:      float = 0.0
    deg_efficiency:       float = 0.0   # degradation efficiency proxy 0–1

    # Warhead scores
    score_warhead_a:      float = 0.0   # vs target pocket
    score_warhead_b:      float = 0.0   # vs E3 ligase pocket

    # Linker geometry
    linker_feasibility:   float = 0.0
    linker_penalty:       float = 0.0
    linker_end_to_end:    float = 0.0   # Å — computed mean ETE distance
    required_ete:         float = 0.0   # Å — geometry-required ETE

    # Cooperativity
    alpha:                float = 1.0   # Hook et al. cooperativity parameter
    hook_effect:          bool  = False # True = Hook effect (too-tight binding)

    # Neo-PPI (induced ternary interface)
    neo_ppi_score:        float = 0.0

    # Geometry
    pocket_separation:    float = 0.0   # Å — target–E3 pocket centre distance
    target_centre:        tuple = (0.0, 0.0, 0.0)
    e3_centre:            tuple = (0.0, 0.0, 0.0)

    # Metadata
    n_trajectory_steps:   int   = 0
    backend:              str   = "unknown"

    def __str__(self) -> str:
        hook_str = " ⚠ Hook effect" if self.hook_effect else ""
        alpha_interp = (
            "positive cooperativity" if self.alpha > 2.0 else
            "neutral"                if self.alpha > 0.5 else
            "anticooperative (Hook risk)"
        )
        lines = [
            "─── PROTAC Ternary Score ───────────────────",
            f"  Composite score:        {self.composite_score:.4f}",
            f"  Degradation efficiency: {self.deg_efficiency:.4f}",
            f"─── Warheads ───────────────────────────────",
            f"  Warhead A (target):     {self.score_warhead_a:.4f}",
            f"  Warhead B (E3 ligase):  {self.score_warhead_b:.4f}",
            f"─── Linker Geometry ────────────────────────",
            f"  Feasibility:            {self.linker_feasibility:.4f}",
            f"  Required ETE:           {self.required_ete:.1f} Å",
            f"  Computed ETE:           {self.linker_end_to_end:.1f} Å",
            f"  Pocket separation:      {self.pocket_separation:.1f} Å",
            f"─── Cooperativity ──────────────────────────",
            f"  Alpha (α):              {self.alpha:.4f}  [{alpha_interp}]{hook_str}",
            f"─── Neo-PPI Interface ──────────────────────",
            f"  Neo-PPI score:          {self.neo_ppi_score:.4f}",
            f"─── Backend ────────────────────────────────",
            f"  Backend:                {self.backend}",
            f"  Trajectory steps:       {self.n_trajectory_steps}",
        ]
        return "\n".join(lines)

    @property
    def is_productive(self) -> bool:
        """Quick heuristic: productive PROTAC if composite > 0.5 and no Hook."""
        return self.composite_score > 0.5 and not self.hook_effect

    @property
    def degradation_class(self) -> str:
        d = self.deg_efficiency
        if d >= 0.75: return "High degrader"
        if d >= 0.50: return "Moderate degrader"
        if d >= 0.25: return "Weak degrader"
        return "Non-degrader"


# ---------------------------------------------------------------------------
# PROTACMolecule — Python wrapper for PROTAC definition
# ---------------------------------------------------------------------------

class PROTACMolecule:
    """
    Defines a PROTAC bifunctional molecule by splitting atom indices.

    The PROTAC atoms must be ordered:
        [warhead_A atoms | linker atoms | warhead_B atoms]

    Args:
        coords:         numpy or list of [n_atoms, 3] coordinates (Å)
        ch_vals:        [n_atoms, N_CHANNELS] channel values
        split_a:        exclusive end index of warhead A atoms
        split_b:        inclusive start index of warhead B atoms
        linker_smiles:  SMILES of the linker (optional, improves geometry)
        n_atoms:        total atom count (inferred if coords provided)
    """

    def __init__(
        self,
        coords,
        ch_vals,
        split_a:       int,
        split_b:       int,
        linker_smiles: str = None,
    ):
        import numpy as np
        self.coords       = np.array(coords, dtype=np.float32).flatten()
        self.ch_vals      = np.array(ch_vals, dtype=np.float32).flatten()
        self.n_atoms      = len(self.coords) // 3
        self.split_a      = split_a
        self.split_b      = split_b
        self.linker_smiles = linker_smiles

        # Validate
        assert 0 < split_a < split_b < self.n_atoms, (
            f"Invalid split indices: split_a={split_a}, split_b={split_b}, "
            f"n_atoms={self.n_atoms}"
        )

    @property
    def n_warhead_a(self) -> int:
        return self.split_a

    @property
    def n_linker(self) -> int:
        return self.split_b - self.split_a

    @property
    def n_warhead_b(self) -> int:
        return self.n_atoms - self.split_b

    @classmethod
    def from_uniform_channels(
        cls,
        coords,
        split_a:       int,
        split_b:       int,
        linker_smiles: str = None,
    ) -> "PROTACMolecule":
        """
        Construct with default uniform channel values (0.5 for all channels).
        Useful for geometry-only scoring when chemical data isn't available.
        """
        import numpy as np
        n_atoms = len(coords) // 3 if hasattr(coords, '__len__') else len(coords)
        ch_vals = [[0.5] * N_CHANNELS] * n_atoms
        return cls(coords, ch_vals, split_a, split_b, linker_smiles)


# ---------------------------------------------------------------------------
# NumPy fallback PROTAC scorer
# ---------------------------------------------------------------------------

class _NumPyPROTACScorer:
    """
    Simplified NumPy-based PROTAC scoring for when C library is unavailable.
    Geometry-accurate, physics-approximate.
    """

    def score(
        self,
        target_centre: tuple,
        e3_centre:     tuple,
        protac:        PROTACMolecule,
        n_steps:       int = 0,
    ) -> PROTACResult:
        import numpy as np

        tc = np.array(target_centre)
        ec = np.array(e3_centre)
        pocket_sep = float(np.linalg.norm(ec - tc))

        # Linker geometry
        n_linker = protac.n_linker
        b = 1.54 * 1.1
        contour  = n_linker * b
        r2_mean  = n_linker * (3.8 ** 2)
        r_rms    = math.sqrt(r2_mean)

        # Approximate warhead radii
        r_a = math.sqrt(protac.n_warhead_a) * 1.5
        r_b = math.sqrt(protac.n_warhead_b) * 1.5
        required_ete = max(0.0, pocket_sep - r_a - r_b)

        # Linker penalty
        if required_ete > contour:
            lp = 1.0
        elif r_rms < 1e-6:
            lp = 0.5
        else:
            exp_val = -3.0 * required_ete ** 2 / (2.0 * r2_mean)
            lp = float(1.0 - math.exp(exp_val))
        lp = max(0.0, min(1.0, lp))

        # Approximate warhead scores from size (proxy for affinity potential)
        sc_a = min(1.0, protac.n_warhead_a / 30.0)
        sc_b = min(1.0, protac.n_warhead_b / 30.0)

        # Cooperativity
        alpha = math.exp(math.log(max(sc_a * sc_b, 1e-6) + 1e-6) * -1.0 / 0.593)
        alpha = max(0.01, min(100.0, alpha))
        hook  = (alpha < 0.5)

        neo_ppi = min(1.0, (sc_a + sc_b) * 0.3)

        composite = (0.30 * sc_a + 0.30 * sc_b + 0.15 * (1.0 - lp) +
                     0.15 * min(1.0, math.log(alpha + 1) / math.log(101)) +
                     0.10 * neo_ppi)
        if hook:
            composite -= 0.30
        composite = max(0.0, min(1.0, composite))

        deg = min(1.0, 0.5 * composite + 0.3 * min(1.0, (alpha - 1) / 99) +
                  0.2 * neo_ppi)

        return PROTACResult(
            composite_score=composite,
            deg_efficiency=deg,
            score_warhead_a=sc_a,
            score_warhead_b=sc_b,
            linker_feasibility=1.0 - lp,
            linker_penalty=lp,
            linker_end_to_end=r_rms,
            required_ete=required_ete,
            alpha=alpha,
            hook_effect=hook,
            neo_ppi_score=neo_ppi,
            pocket_separation=pocket_sep,
            target_centre=tuple(target_centre),
            e3_centre=tuple(e3_centre),
            n_trajectory_steps=n_steps,
            backend="NumPy-Fallback",
        )


# ---------------------------------------------------------------------------
# NibblePROTACEngine — main engine
# ---------------------------------------------------------------------------

class NibblePROTACEngine:
    """
    PROTAC ternary complex scoring and optimisation engine.

    Wraps nibble_protac.c via ctypes with NumPy fallback.
    Shares the loaded library with NibbleEngine and NibblePPIEngine.

    Quick start
    -----------
    engine = NibblePROTACEngine()

    # Score PROTAC dARD-1 (BRD4 degrader)
    result = engine.score_ternary(
        target_pdb     = "brd4_bd1.pdb",
        e3_pdb         = "crbn.pdb",
        target_centre  = (12.3, 8.1, 22.4),
        e3_centre      = (4.5, 31.2, 9.8),
        protac         = my_protac,
        run_trajectory = True,
        n_steps        = 300,
    )
    print(result)
    print(result.degradation_class)
    """

    # Grid parameters for ternary complex (standard small-molecule pocket size)
    _DIM    = 60
    _RES    = 0.5   # Å
    _RADIUS = 12.0  # Å pocket radius

    def __init__(self):
        self._lib      = NibbleEngine._lib
        self._fallback = _NumPyPROTACScorer()

        if self._lib is not None:
            self._register_protac_functions()
            self._mode = "C-Native"
        else:
            self._mode = "NumPy-Fallback"

    def _register_protac_functions(self):
        lib = self._lib
        _G   = ctypes.POINTER(NibbleGridStruct)
        _F   = ctypes.c_float
        _PM  = ctypes.POINTER(PROTACMoleculeStruct)
        _PG  = ctypes.POINTER(PROTACTernaryGeometryStruct)
        _PS  = ctypes.POINTER(PROTACTernaryScoreStruct)
        _PL  = ctypes.POINTER(PROTACLinkerStruct)
        _PI  = ctypes.POINTER(PPIInterfaceStruct)
        _INT = ctypes.c_int
        _STR = ctypes.c_char_p

        lib.nibble_protac_create.argtypes = [
            _INT, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), _INT, _INT
        ]
        lib.nibble_protac_create.restype  = _PM

        lib.nibble_protac_free.argtypes   = [_PM]
        lib.nibble_protac_free.restype    = None

        lib.nibble_protac_linker_penalty.argtypes = [
            _PL, _F, ctypes.POINTER(_F)
        ]
        lib.nibble_protac_linker_penalty.restype  = _F

        lib.nibble_protac_cooperativity.argtypes = [_F, _F, _F, _F]
        lib.nibble_protac_cooperativity.restype  = _F

        lib.nibble_protac_neo_ppi.argtypes = [_G, _G, _PM, _PI, _F]
        lib.nibble_protac_neo_ppi.restype  = _F

        lib.nibble_protac_score.argtypes = [
            _G, _G, _PM, _PG, _F, _F, _PS
        ]
        lib.nibble_protac_score.restype  = _F

        lib.nibble_protac_trajectory.argtypes = [
            _G, _G, _PM, _PG, _INT, _F, _F, _F, _PS
        ]
        lib.nibble_protac_trajectory.restype  = _F

        lib.nibble_protac_infer_linker.argtypes = [_PL, _INT, _STR]
        lib.nibble_protac_infer_linker.restype  = None

        lib.nibble_protac_print_score.argtypes  = [_PS]
        lib.nibble_protac_print_score.restype   = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def score_ternary(
        self,
        target_pdb:     str,
        e3_pdb:         str,
        target_centre:  tuple,
        e3_centre:      tuple,
        protac:         PROTACMolecule,
        run_trajectory: bool  = True,
        n_steps:        int   = 300,
        pocket_radius:  float = 12.0,
        delta_G:        float = 2.0,
        kT:             float = 0.593,
        verbose:        bool  = False,
    ) -> PROTACResult:
        """
        Score a PROTAC ternary complex.

        Args:
            target_pdb:     Path to target protein PDB.
            e3_pdb:         Path to E3 ligase PDB.
            target_centre:  (x, y, z) of target binding pocket centre (Å).
            e3_centre:      (x, y, z) of E3 ligase binding pocket centre (Å).
            protac:         PROTACMolecule (warhead_a | linker | warhead_b).
            run_trajectory: If True, run dumbbell Langevin optimisation.
            n_steps:        Trajectory steps (default 300).
            pocket_radius:  Pocket loading radius in Å.
            delta_G:        Water desolvation energy (kcal/mol).
            kT:             Thermal energy (kcal/mol, 0.593 at 298K).
            verbose:        Print C-level summary to stdout.

        Returns:
            PROTACResult with all scoring components.
        """
        if self._mode == "NumPy-Fallback":
            return self._fallback.score(
                target_centre, e3_centre, protac, n_steps
            )
        return self._score_native(
            target_pdb, e3_pdb, target_centre, e3_centre,
            protac, run_trajectory, n_steps,
            pocket_radius, delta_G, kT, verbose,
        )

    def scan_linker_lengths(
        self,
        target_pdb:         str,
        e3_pdb:             str,
        target_centre:      tuple,
        e3_centre:          tuple,
        warhead_a_coords,
        warhead_a_chvals,
        warhead_b_coords,
        warhead_b_chvals,
        linker_atoms_range: range = range(4, 20, 2),
        kT:                 float = 0.593,
    ) -> List[Tuple[int, PROTACResult]]:
        """
        Scan across different linker lengths.
        For each linker length, creates a uniform-channel linker,
        scores the ternary complex, and returns results sorted by
        degradation efficiency.

        Useful for PROTAC linker optimisation in the design loop.

        Returns:
            List of (n_linker_atoms, PROTACResult) sorted by deg_efficiency.
        """
        import numpy as np
        results = []

        for n_linker in linker_atoms_range:
            # Build uniform linker coords (straight line between warheads)
            tc = np.array(target_centre)
            ec = np.array(e3_centre)
            direction = ec - tc
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                continue
            direction /= dist

            n_a = len(warhead_a_coords) // 3
            n_b = len(warhead_b_coords) // 3

            linker_coords = []
            for i in range(n_linker):
                t = (i + 1) / (n_linker + 1)
                pt = tc + direction * dist * t
                linker_coords.extend(pt.tolist())

            # Combine: warhead_a | linker | warhead_b
            all_coords = (
                list(warhead_a_coords) +
                linker_coords +
                list(warhead_b_coords)
            )
            n_total = n_a + n_linker + n_b
            all_chvals = [[0.5] * N_CHANNELS] * n_total

            protac = PROTACMolecule(
                coords=all_coords,
                ch_vals=[v for row in all_chvals for v in row],
                split_a=n_a,
                split_b=n_a + n_linker,
            )

            result = self.score_ternary(
                target_pdb, e3_pdb,
                target_centre, e3_centre,
                protac,
                run_trajectory=False,
                kT=kT,
            )
            results.append((n_linker, result))

        results.sort(key=lambda x: x[1].deg_efficiency, reverse=True)
        return results

    def compute_optimal_linker_length(
        self,
        target_centre: tuple,
        e3_centre:     tuple,
        warhead_a_radius: float = 5.0,
        warhead_b_radius: float = 5.0,
        flexibility:   float = 0.8,
    ) -> dict:
        """
        Compute the optimal linker length for a given target–E3 pair
        without running full scoring.

        Uses the Gaussian chain model to find the linker atom count
        whose most probable ETE matches the required separation.

        Returns dict with:
            optimal_n_atoms:    recommended linker heavy atom count
            required_ete:       Å — required end-to-end distance
            pocket_separation:  Å — centre-to-centre distance
            contour_length:     Å — resulting contour length
        """
        import numpy as np
        tc = np.array(target_centre)
        ec = np.array(e3_centre)
        pocket_sep = float(np.linalg.norm(ec - tc))
        required_ete = max(0.0, pocket_sep - warhead_a_radius - warhead_b_radius)

        # Find n such that sqrt(n * b²) ≈ required_ete
        b = 3.8  # Kuhn length
        n_optimal = max(3, int(math.ceil((required_ete / b) ** 2)))

        # Contour length at this n
        contour = n_optimal * 1.54 * 1.1

        return {
            "optimal_n_atoms":   n_optimal,
            "required_ete":      required_ete,
            "pocket_separation": pocket_sep,
            "contour_length":    contour,
            "suggested_linker":  f"PEG{n_optimal // 2}-like  (~{n_optimal} heavy atoms)",
        }

    @property
    def mode(self) -> str:
        return self._mode

    # -----------------------------------------------------------------------
    # Internal C-native implementation
    # -----------------------------------------------------------------------

    def _score_native(
        self, target_pdb, e3_pdb, target_centre, e3_centre,
        protac, run_trajectory, n_steps,
        pocket_radius, delta_G, kT, verbose,
    ) -> PROTACResult:
        import numpy as np
        lib = self._lib

        # --- Load target pocket ---
        dim = self._DIM
        res = self._RES
        grid_target = lib.nibble_create_grid(dim, dim, dim, ctypes.c_float(res))
        grid_e3     = lib.nibble_create_grid(dim, dim, dim, ctypes.c_float(res))

        tc = target_centre
        ec = e3_centre
        blur = 1.2

        lib.nibble_load_pdb_pocket(
            grid_target,
            target_pdb.encode(),
            ctypes.c_float(tc[0]), ctypes.c_float(tc[1]), ctypes.c_float(tc[2]),
            ctypes.c_float(pocket_radius),
            ctypes.c_float(blur),
        )
        lib.nibble_load_pdb_pocket(
            grid_e3,
            e3_pdb.encode(),
            ctypes.c_float(ec[0]), ctypes.c_float(ec[1]), ctypes.c_float(ec[2]),
            ctypes.c_float(pocket_radius),
            ctypes.c_float(blur),
        )

        # --- Create PROTAC molecule ---
        coords_c  = (ctypes.c_float * len(protac.coords))(*protac.coords)
        chvals_c  = (ctypes.c_float * len(protac.ch_vals))(*protac.ch_vals)
        protac_ptr = lib.nibble_protac_create(
            ctypes.c_int(protac.n_atoms),
            coords_c, chvals_c,
            ctypes.c_int(protac.split_a),
            ctypes.c_int(protac.split_b),
        )
        if not protac_ptr:
            lib.nibble_free_grid(grid_target)
            lib.nibble_free_grid(grid_e3)
            return PROTACResult(backend="C-Native-Error-PROTAC")

        # Infer linker if SMILES provided
        if protac.linker_smiles:
            lib.nibble_protac_infer_linker(
                ctypes.byref(protac_ptr.contents.linker),
                ctypes.c_int(protac.n_linker),
                protac.linker_smiles.encode(),
            )

        # Set warhead initial positions at their respective pocket centres
        protac_ptr.contents.warhead_a.contents.cx = tc[0]
        protac_ptr.contents.warhead_a.contents.cy = tc[1]
        protac_ptr.contents.warhead_a.contents.cz = tc[2]
        protac_ptr.contents.warhead_b.contents.cx = ec[0]
        protac_ptr.contents.warhead_b.contents.cy = ec[1]
        protac_ptr.contents.warhead_b.contents.cz = ec[2]

        # --- Build ternary geometry ---
        geom = PROTACTernaryGeometryStruct()
        geom.target_cx = ctypes.c_float(tc[0])
        geom.target_cy = ctypes.c_float(tc[1])
        geom.target_cz = ctypes.c_float(tc[2])
        geom.e3_cx = ctypes.c_float(ec[0])
        geom.e3_cy = ctypes.c_float(ec[1])
        geom.e3_cz = ctypes.c_float(ec[2])

        dx = ec[0] - tc[0]
        dy = ec[1] - tc[1]
        dz = ec[2] - tc[2]
        pocket_sep = math.sqrt(dx*dx + dy*dy + dz*dz)
        geom.pocket_sep_x      = ctypes.c_float(dx)
        geom.pocket_sep_y      = ctypes.c_float(dy)
        geom.pocket_sep_z      = ctypes.c_float(dz)
        geom.pocket_separation = ctypes.c_float(pocket_sep)
        # Required linker ETE: subtract approximate warhead radii
        r_a = math.sqrt(protac.n_warhead_a) * 1.5
        r_b = math.sqrt(protac.n_warhead_b) * 1.5
        geom.required_linker_ete = ctypes.c_float(
            max(0.0, pocket_sep - r_a - r_b)
        )
        geom.has_neo_ppi  = ctypes.c_int(0)
        geom.neo_ppi_score = ctypes.c_float(0.0)

        # --- Score or trajectory ---
        score_struct = PROTACTernaryScoreStruct()

        if run_trajectory:
            composite = lib.nibble_protac_trajectory(
                grid_target, grid_e3,
                protac_ptr,
                ctypes.byref(geom),
                ctypes.c_int(n_steps),
                ctypes.c_float(0.002),
                ctypes.c_float(1.0),
                ctypes.c_float(kT),
                ctypes.byref(score_struct),
            )
        else:
            composite = lib.nibble_protac_score(
                grid_target, grid_e3,
                protac_ptr,
                ctypes.byref(geom),
                ctypes.c_float(delta_G),
                ctypes.c_float(kT),
                ctypes.byref(score_struct),
            )

        if verbose:
            lib.nibble_protac_print_score(ctypes.byref(score_struct))

        result = PROTACResult(
            composite_score=float(score_struct.composite_score),
            deg_efficiency=float(score_struct.deg_efficiency),
            score_warhead_a=float(score_struct.score_warhead_a),
            score_warhead_b=float(score_struct.score_warhead_b),
            linker_feasibility=float(score_struct.linker_feasibility),
            linker_penalty=float(score_struct.linker_penalty),
            linker_end_to_end=float(score_struct.linker_end_to_end),
            required_ete=float(geom.required_linker_ete),
            alpha=float(score_struct.alpha),
            hook_effect=bool(score_struct.hook_effect),
            neo_ppi_score=float(score_struct.neo_ppi_score),
            pocket_separation=pocket_sep,
            target_centre=tuple(tc),
            e3_centre=tuple(ec),
            n_trajectory_steps=n_steps if run_trajectory else 0,
            backend="C-Native",
        )

        # Cleanup
        lib.nibble_protac_free(protac_ptr)
        lib.nibble_free_grid(grid_target)
        lib.nibble_free_grid(grid_e3)

        return result


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_protac_engine() -> NibblePROTACEngine:
    """
    Factory mirroring Khukuri conventions.

    Example — ARV-110 (BRD4 degrader targeting CRBN):
    --------------------------------------------------
    from nibble_protac_bridge import make_protac_engine, PROTACMolecule
    import numpy as np

    engine = make_protac_engine()

    # Quick geometry check before full scoring
    linker_rec = engine.compute_optimal_linker_length(
        target_centre=(12.3, 8.1, 22.4),   # BRD4 BD1 pocket
        e3_centre=(4.5, 31.2, 9.8),         # CRBN thalidomide pocket
    )
    print(f"Recommended linker: {linker_rec['suggested_linker']}")
    print(f"Required ETE: {linker_rec['required_ete']:.1f} Å")

    # Full ternary scoring with trajectory
    result = engine.score_ternary(
        target_pdb     = "brd4_bd1.pdb",
        e3_pdb         = "crbn_thalidomide.pdb",
        target_centre  = (12.3, 8.1, 22.4),
        e3_centre      = (4.5, 31.2, 9.8),
        protac         = my_protac_molecule,
        run_trajectory = True,
        n_steps        = 300,
        verbose        = True,
    )
    print(result)
    print(f"Class: {result.degradation_class}")

    # Linker length scan
    scan = engine.scan_linker_lengths(
        target_pdb      = "brd4_bd1.pdb",
        e3_pdb          = "crbn_thalidomide.pdb",
        target_centre   = (12.3, 8.1, 22.4),
        e3_centre       = (4.5, 31.2, 9.8),
        warhead_a_coords= wh_a_coords.flatten(),
        warhead_a_chvals= wh_a_chvals.flatten(),
        warhead_b_coords= wh_b_coords.flatten(),
        warhead_b_chvals= wh_b_chvals.flatten(),
        linker_atoms_range=range(4, 22, 2),
    )
    print("\\nLinker scan results:")
    for n, r in scan[:5]:
        print(f"  n={n:2d} atoms  deg={r.deg_efficiency:.3f}  "
              f"alpha={r.alpha:.2f}  hook={r.hook_effect}")
    """
    return NibblePROTACEngine()
