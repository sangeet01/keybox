"""
nibble_antibody_bridge.py
--------------------------
Python ctypes bindings for the Nibble antibody CDR-antigen docking extension.

Plugs into the existing NibbleEngine architecture.
Shares the same loaded shared library. Zero changes to existing files.

Full antibody docking pipeline:
  1. Parse antibody Fv from PDB  → AB_Antibody (CDR loops + framework)
     OR build from CDR sequences → AB_Antibody (de novo design mode)
  2. Build paratope demand grid  → NibbleGrid (composite of 6 CDRs)
  3. Load antigen surface grid   → NibbleGrid (existing nibble_bridge)
  4. Scan epitope patches        → ranked list of complementary surface patches
  5. CDR-H3 flexible trajectory  → optimise most variable loop
  6. Full composite score        → AntibodyDockingResult

Usage
-----
  from nibble_antibody_bridge import make_antibody_engine, AntibodyMolecule

  engine = make_antibody_engine()

  # From PDB (Fv or Fab format)
  ab = AntibodyMolecule.from_pdb("trastuzumab_fv.pdb", vh_chain='H', vl_chain='L')
  result = engine.dock(
      antigen_pdb    = "her2_ecd.pdb",
      antigen_centre = (45.2, 31.8, 22.4),
      antibody       = ab,
      verbose        = True,
  )
  print(result)

  # De novo / design mode: specify CDR sequences directly
  ab = AntibodyMolecule.from_sequences(
      L1="RASQSIS", L2="AASSLQS", L3="QQSYTTP",
      H1="GYTFTSYW", H2="IYPGDGDT", H3="ARWGGDYFDY",
  )
  result = engine.dock("antigen.pdb", (10.0, 10.0, 10.0), ab)
  print(result.binding_class)

Author: Khukuri / KeyBox project (sangeet01)
License: Apache 2.0 with Commons Clause
"""

import ctypes
import math
from dataclasses import dataclass, field
from typing import Optional, List

from nibble_bridge import (
    NibbleEngine,
    NibbleGridStruct,
    N_CHANNELS,
)
from nibble_peptide_bridge import (
    PEPResidueStruct,
    PEPSegmentStruct,
    PEPConstraintStruct,
    PEPPeptideStruct,
    _PEP_MAX_RESIDUES,
    _PEP_MAX_SEGMENTS,
    _PEP_MAX_CONSTRAINTS,
)
from nibble_ppi_bridge import PPIInterfaceStruct, PPIScoreStruct


# ---------------------------------------------------------------------------
# CDR index constants
# ---------------------------------------------------------------------------
CDR_L1, CDR_L2, CDR_L3 = 0, 1, 2
CDR_H1, CDR_H2, CDR_H3 = 3, 4, 5
CDR_NAMES  = ["L1", "L2", "L3", "H1", "H2", "H3"]
CDR_WEIGHTS = [0.08, 0.06, 0.14, 0.14, 0.14, 0.44]

_AB_N_CDR_LOOPS       = 6
_AB_MAX_CDR_LEN       = 30
_AB_MAX_EPITOPE_PATCHES = 20


# ---------------------------------------------------------------------------
# ctypes mirrors of antibody C structs
# ---------------------------------------------------------------------------

class ABCDRLoopStruct(ctypes.Structure):
    _fields_ = [
        ("cdr_idx",       ctypes.c_int),
        ("sequence",      ctypes.c_char * (_AB_MAX_CDR_LEN + 1)),
        ("n_residues",    ctypes.c_int),
        ("ca_x",          ctypes.c_float * _AB_MAX_CDR_LEN),
        ("ca_y",          ctypes.c_float * _AB_MAX_CDR_LEN),
        ("ca_z",          ctypes.c_float * _AB_MAX_CDR_LEN),
        ("anchor_n_x",    ctypes.c_float),
        ("anchor_n_y",    ctypes.c_float),
        ("anchor_n_z",    ctypes.c_float),
        ("anchor_c_x",    ctypes.c_float),
        ("anchor_c_y",    ctypes.c_float),
        ("anchor_c_z",    ctypes.c_float),
        ("canonical_class",ctypes.c_int),
        ("is_h3",         ctypes.c_int),
        ("pep",           PEPPeptideStruct),
    ]


class ABFvFrameworkStruct(ctypes.Structure):
    _fields_ = [
        ("vh_cx",    ctypes.c_float),
        ("vh_cy",    ctypes.c_float),
        ("vh_cz",    ctypes.c_float),
        ("vl_cx",    ctypes.c_float),
        ("vl_cy",    ctypes.c_float),
        ("vl_cz",    ctypes.c_float),
        ("elbow_angle", ctypes.c_float),
        ("qw",       ctypes.c_float),
        ("qx",       ctypes.c_float),
        ("qy",       ctypes.c_float),
        ("qz",       ctypes.c_float),
        ("vx",       ctypes.c_float),
        ("vy",       ctypes.c_float),
        ("vz",       ctypes.c_float),
        ("wx",       ctypes.c_float),
        ("wy",       ctypes.c_float),
        ("wz",       ctypes.c_float),
        ("mass",     ctypes.c_float),
    ]


class ABEpitopePatchStruct(ctypes.Structure):
    _fields_ = [
        ("cx",               ctypes.c_float),
        ("cy",               ctypes.c_float),
        ("cz",               ctypes.c_float),
        ("normal_x",         ctypes.c_float),
        ("normal_y",         ctypes.c_float),
        ("normal_z",         ctypes.c_float),
        ("area",             ctypes.c_float),
        ("complementarity",  ctypes.c_float),
        ("sc_score",         ctypes.c_float),
        ("buried_area",      ctypes.c_float),
        ("n_hbond_sites",    ctypes.c_int),
        ("n_hydrophobic",    ctypes.c_int),
        ("n_charged",        ctypes.c_int),
        ("is_linear_epitope",ctypes.c_int),
        ("is_conformational",ctypes.c_int),
    ]


class ABDockingResultStruct(ctypes.Structure):
    _fields_ = [
        ("composite_score",           ctypes.c_float),
        ("paratope_epitope_sc",       ctypes.c_float),
        ("h3_affinity",               ctypes.c_float),
        ("canonical_cdr_score",       ctypes.c_float),
        ("buried_surface_area",       ctypes.c_float),
        ("vhvl_orientation_penalty",  ctypes.c_float),
        ("cdr_scores",                ctypes.c_float * _AB_N_CDR_LOOPS),
        ("best_epitope_idx",          ctypes.c_int),
        ("best_epitope_complementarity", ctypes.c_float),
        ("n_epitope_patches",         ctypes.c_int),
        ("kd_class",                  ctypes.c_int),
    ]


class ABAntibodyStruct(ctypes.Structure):
    _fields_ = [
        ("cdrs",           ABCDRLoopStruct * _AB_N_CDR_LOOPS),
        ("framework",      ABFvFrameworkStruct),
        ("paratope_grid",  ctypes.POINTER(NibbleGridStruct)),
        ("paratope_built", ctypes.c_int),
        ("name",           ctypes.c_char * 32),
    ]


# ---------------------------------------------------------------------------
# Python result container
# ---------------------------------------------------------------------------

_KD_LABELS = {0: "nM-class binder", 1: "µM-class binder", 2: "Weak/non-binder"}


@dataclass
class AntibodyDockingResult:
    """Full result of antibody-antigen docking."""

    composite_score:              float = 0.0
    paratope_epitope_sc:          float = 0.0  # Lawrence-Coleman Sc
    h3_affinity:                  float = 0.0  # CDR-H3 specific
    canonical_cdr_score:          float = 0.0  # L1-L2-L3-H1-H2 combined
    buried_surface_area:          float = 0.0  # Å²
    vhvl_orientation_penalty:     float = 0.0

    cdr_scores:     list = field(default_factory=lambda: [0.0] * 6)
    n_epitope_patches:   int  = 0
    best_epitope_score:  float = 0.0
    kd_class:            int  = 2   # 0=nM, 1=µM, 2=weak

    # Epitope patches found
    epitope_patches: list = field(default_factory=list)

    # Antibody info
    cdr_sequences:  dict  = field(default_factory=dict)
    elbow_angle:    float = 160.0
    source:         str   = ""

    backend:        str   = "unknown"

    def __str__(self) -> str:
        cdr_lines = "\n".join(
            f"    {CDR_NAMES[i]}: {self.cdr_scores[i]:.4f}"
            f"  (w={CDR_WEIGHTS[i]:.2f})"
            f"  seq={self.cdr_sequences.get(CDR_NAMES[i], '?')}"
            for i in range(6)
        )
        lines = [
            "─── Antibody-Antigen Docking Result ────────",
            f"  Composite score:          {self.composite_score:.4f}"
            f"  [{_KD_LABELS[self.kd_class]}]",
            f"─── Scoring Components ─────────────────────",
            f"  Paratope-epitope Sc:      {self.paratope_epitope_sc:.4f}",
            f"  CDR-H3 affinity:          {self.h3_affinity:.4f}  (w=0.44)",
            f"  Canonical CDR score:      {self.canonical_cdr_score:.4f}",
            f"  Buried surface area:      {self.buried_surface_area:.1f} Å²",
            f"  VH/VL orientation pen:    {self.vhvl_orientation_penalty:.4f}",
            f"  VH/VL elbow angle:        {self.elbow_angle:.1f}°",
            f"─── Per-CDR Scores ─────────────────────────",
            cdr_lines,
            f"─── Epitope ────────────────────────────────",
            f"  Patches detected:         {self.n_epitope_patches}",
            f"  Best patch score:         {self.best_epitope_score:.4f}",
            f"─── Backend ────────────────────────────────",
            f"  Backend:                  {self.backend}",
            f"  Source:                   {self.source}",
        ]
        return "\n".join(lines)

    @property
    def binding_class(self) -> str:
        return _KD_LABELS[self.kd_class]

    @property
    def is_productive(self) -> bool:
        return (self.composite_score > 0.40 and
                self.h3_affinity > 0.20 and
                self.kd_class <= 1)

    @property
    def h3_dominated(self) -> bool:
        """True if CDR-H3 contributes > 50% of composite score."""
        return self.h3_affinity * 0.44 > self.composite_score * 0.5


# ---------------------------------------------------------------------------
# AntibodyMolecule — Python-side antibody definition
# ---------------------------------------------------------------------------

class AntibodyMolecule:
    """
    Defines an antibody Fv for docking.

    Two construction modes:
      1. from_pdb()        — parse CDRs from an existing PDB file
      2. from_sequences()  — specify CDR sequences directly (design mode)
    """

    def __init__(
        self,
        cdr_sequences:  dict,   # {name: sequence} for all six CDRs
        vh_chain:       str  = 'H',
        vl_chain:       str  = 'L',
        pdb_path:       str  = None,
        name:           str  = "antibody",
    ):
        self.cdr_sequences = cdr_sequences  # {"L1": seq, ..., "H3": seq}
        self.vh_chain      = vh_chain
        self.vl_chain      = vl_chain
        self.pdb_path      = pdb_path
        self.name          = name

    @classmethod
    def from_pdb(
        cls,
        pdb_path:  str,
        vh_chain:  str = 'H',
        vl_chain:  str = 'L',
        name:      str = None,
    ) -> "AntibodyMolecule":
        """
        Construct from a PDB file.
        CDR sequences will be extracted by the C engine at dock time.
        """
        import os
        n = name or os.path.splitext(os.path.basename(pdb_path))[0]
        return cls(
            cdr_sequences={},
            vh_chain=vh_chain,
            vl_chain=vl_chain,
            pdb_path=pdb_path,
            name=n,
        )

    @classmethod
    def from_sequences(
        cls,
        L1: str = "", L2: str = "", L3: str = "",
        H1: str = "", H2: str = "", H3: str = "",
        name: str = "designed_ab",
    ) -> "AntibodyMolecule":
        """
        Construct from explicit CDR sequences.
        Used for de novo antibody design and virtual screening.

        Example — trastuzumab CDRs (approximate):
            ab = AntibodyMolecule.from_sequences(
                L1="RASQDVNTAVA", L2="SASFLYS",   L3="QQHYTTPPT",
                H1="GFTFSSYW",   H2="IKDKSGAT",  H3="VFYGSKTDYFDY",
            )
        """
        return cls(
            cdr_sequences={"L1":L1,"L2":L2,"L3":L3,"H1":H1,"H2":H2,"H3":H3},
            name=name,
        )

    @property
    def from_pdb_mode(self) -> bool:
        return self.pdb_path is not None

    @property
    def h3_length(self) -> int:
        return len(self.cdr_sequences.get("H3", ""))

    def total_cdr_residues(self) -> int:
        return sum(len(v) for v in self.cdr_sequences.values())


# ---------------------------------------------------------------------------
# NumPy fallback antibody scorer
# ---------------------------------------------------------------------------

class _NumPyAntibodyScorer:
    """
    Simplified NumPy-based antibody scoring fallback.
    """

    _AA_HYDRO = {
        'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
        'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
        'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
        'S': -0.18, 'T': -0.05, 'W': -0.90, 'Y': 0.26, 'V': 1.08,
    }
    _AA_CHARGE = {'R':1.0,'K':1.0,'H':0.1,'D':-1.0,'E':-1.0}

    def score(self, ab: AntibodyMolecule) -> AntibodyDockingResult:
        seqs = ab.cdr_sequences

        # Per-CDR scores: hydrophobic + charged residue proxy
        cdr_scores = []
        for name in CDR_NAMES:
            seq = seqs.get(name, "")
            if not seq:
                cdr_scores.append(0.0)
                continue
            n_hydro  = sum(1 for aa in seq if self._AA_HYDRO.get(aa, 0) > 0.5)
            n_hbond  = sum(1 for aa in seq if aa in "STNQHRKDEY")
            n_arom   = sum(1 for aa in seq if aa in "FYWH")
            score = min(1.0, (n_hydro * 0.05 + n_hbond * 0.04 + n_arom * 0.06))
            cdr_scores.append(score)

        # H3-weighted composite
        h3_score = cdr_scores[CDR_H3]
        canon = sum(
            cdr_scores[i] * CDR_WEIGHTS[i]
            for i in range(5)  # L1-H2
        ) / sum(CDR_WEIGHTS[:5])

        composite = (
            0.30 * min(1.0, (h3_score + canon) * 0.5) +
            0.35 * h3_score +
            0.20 * canon +
            0.10 * 0.5 +   # BSA proxy
            0.05 * 0.0      # no VHVL penalty
        )
        composite = min(1.0, max(0.0, composite))
        kd = 0 if composite >= 0.60 else (1 if composite >= 0.35 else 2)

        return AntibodyDockingResult(
            composite_score=composite,
            paratope_epitope_sc=min(1.0, composite * 0.8),
            h3_affinity=h3_score,
            canonical_cdr_score=canon,
            buried_surface_area=1200.0,
            vhvl_orientation_penalty=0.0,
            cdr_scores=cdr_scores,
            n_epitope_patches=1,
            best_epitope_score=composite,
            kd_class=kd,
            cdr_sequences=seqs,
            elbow_angle=160.0,
            source=ab.name,
            backend="NumPy-Fallback",
        )


# ---------------------------------------------------------------------------
# NibbleAntibodyEngine — main engine
# ---------------------------------------------------------------------------

class NibbleAntibodyEngine:
    """
    Antibody CDR-antigen docking engine.

    Wraps nibble_antibody.c via ctypes with NumPy fallback.
    Shares the loaded library with all other Nibble engines.

    Covers the final interaction class:
      Antibody CDR loops → antigen epitope surface

    Combined with the other engines, Nibble now covers:
      Small molecule → protein pocket     (nibble.c)
      Protein → protein interface         (nibble_ppi.c)
      PPI → small mol inhibitor           (nibble_ppi.c Phase VI)
      PROTAC → ternary complex            (nibble_protac.c)
      Peptide → protein groove/surface    (nibble_peptide.c)
      Antibody CDR → antigen epitope      (nibble_antibody.c)

    Quick start
    -----------
    engine = NibbleAntibodyEngine()

    ab = AntibodyMolecule.from_sequences(
        H3="ARWGGDYFDY", H1="GYTFTSYW", H2="IYPGDGDT",
        L1="RASQSIS",   L2="AASSLQS", L3="QQSYTTP",
    )
    result = engine.dock("antigen.pdb", (10.0, 10.0, 10.0), ab)
    print(result)
    """

    _DIM    = 80
    _RES    = 0.5
    _RADIUS = 25.0  # largest loading radius — antibody footprint ~800–1200 Å²

    def __init__(self):
        self._lib      = NibbleEngine._lib
        self._fallback = _NumPyAntibodyScorer()

        if self._lib is not None:
            self._register_functions()
            self._mode = "C-Native"
        else:
            self._mode = "NumPy-Fallback"

    def _register_functions(self):
        lib  = self._lib
        _G   = ctypes.POINTER(NibbleGridStruct)
        _F   = ctypes.c_float
        _AB  = ctypes.POINTER(ABAntibodyStruct)
        _EP  = ctypes.POINTER(ABEpitopePatchStruct)
        _DR  = ctypes.POINTER(ABDockingResultStruct)
        _PI  = ctypes.POINTER(PPIInterfaceStruct)
        _INT = ctypes.c_int
        _STR = ctypes.c_char_p
        _PC  = ctypes.POINTER(ctypes.c_char_p)

        lib.nibble_antibody_parse.argtypes = [_STR, ctypes.c_char, ctypes.c_char, _AB]
        lib.nibble_antibody_parse.restype  = _INT

        lib.nibble_antibody_build_from_sequences.argtypes = [_PC, _AB]
        lib.nibble_antibody_build_from_sequences.restype  = _INT

        lib.nibble_antibody_build_paratope.argtypes = [_AB, _INT, _F]
        lib.nibble_antibody_build_paratope.restype  = _G

        lib.nibble_antibody_scan_epitope.argtypes = [_G, _G, _EP, _F]
        lib.nibble_antibody_scan_epitope.restype  = _INT

        lib.nibble_antibody_h3_trajectory.argtypes = [
            _G, _AB, _EP, _INT, _F, ctypes.POINTER(_F)
        ]
        lib.nibble_antibody_h3_trajectory.restype  = _F

        lib.nibble_antibody_score.argtypes = [_G, _AB, _INT, _F, _F, _DR]
        lib.nibble_antibody_score.restype  = _F

        lib.nibble_antibody_free.argtypes  = [_AB]
        lib.nibble_antibody_free.restype   = None

        lib.nibble_antibody_print_result.argtypes = [_DR]
        lib.nibble_antibody_print_result.restype  = None

        lib.nibble_antibody_print.argtypes = [_AB]
        lib.nibble_antibody_print.restype  = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def dock(
        self,
        antigen_pdb:    str,
        antigen_centre: tuple,
        antibody:       AntibodyMolecule,
        n_steps:        int   = 300,
        pocket_radius:  float = 25.0,
        blur:           float = 1.5,
        delta_G:        float = 2.0,
        kT:             float = 0.593,
        verbose:        bool  = False,
    ) -> AntibodyDockingResult:
        """
        Dock an antibody against an antigen surface.

        Args:
            antigen_pdb:    Path to antigen PDB file.
            antigen_centre: (x, y, z) of antigen binding region centre (Å).
            antibody:       AntibodyMolecule (from PDB or sequences).
            n_steps:        CDR-H3 Langevin trajectory steps (default 300).
            pocket_radius:  Surface loading radius (Å, default 25 — large for Ab).
            blur:           Gaussian blur for surface loading.
            delta_G:        Water desolvation energy (kcal/mol).
            kT:             Thermal energy (kcal/mol, 0.593 at 298K).
            verbose:        Print C-level summary to stdout.

        Returns:
            AntibodyDockingResult with all scoring components.
        """
        if self._mode == "NumPy-Fallback":
            return self._fallback.score(antibody)

        return self._dock_native(
            antigen_pdb, antigen_centre, antibody,
            n_steps, pocket_radius, blur, delta_G, kT, verbose,
        )

    def scan_epitopes(
        self,
        antigen_pdb:    str,
        antigen_centre: tuple,
        antibody:       AntibodyMolecule,
        threshold:      float = 0.05,
    ) -> List[dict]:
        """
        Scan for epitope patches on the antigen surface without full docking.
        Faster than dock() — useful for epitope mapping.

        Returns list of patch dicts sorted by complementarity.
        """
        if self._mode == "NumPy-Fallback":
            return []

        lib = self._lib
        dim = self._DIM
        res = self._RES
        pc  = antigen_centre

        # Load antigen grid
        grid = lib.nibble_create_grid(dim, dim, dim, ctypes.c_float(res))
        lib.nibble_load_pdb_pocket(
            grid, antigen_pdb.encode(),
            ctypes.c_float(pc[0]), ctypes.c_float(pc[1]), ctypes.c_float(pc[2]),
            ctypes.c_float(25.0), ctypes.c_float(1.5),
        )

        # Build antibody and paratope
        ab_struct = ABAntibodyStruct()
        self._build_ab_struct(antibody, ab_struct)

        pgrid = lib.nibble_antibody_build_paratope(
            ctypes.byref(ab_struct),
            ctypes.c_int(60),
            ctypes.c_float(0.5),
        )

        patches_arr = (ABEpitopePatchStruct * _AB_MAX_EPITOPE_PATCHES)()
        n = lib.nibble_antibody_scan_epitope(
            grid, pgrid, patches_arr, ctypes.c_float(threshold)
        )

        lib.nibble_antibody_free(ctypes.byref(ab_struct))
        lib.nibble_free_grid(grid)

        result = []
        for i in range(n):
            p = patches_arr[i]
            result.append({
                "index":         i,
                "centre":        (p.cx, p.cy, p.cz),
                "normal":        (p.normal_x, p.normal_y, p.normal_z),
                "area":          p.area,
                "complementarity": p.complementarity,
                "sc_score":      p.sc_score,
                "buried_area":   p.buried_area,
                "n_hbonds":      p.n_hbond_sites,
                "n_hydrophobic": p.n_hydrophobic,
                "n_charged":     p.n_charged,
                "linear":        bool(p.is_linear_epitope),
                "conformational":bool(p.is_conformational),
            })
        return result

    @property
    def mode(self) -> str:
        return self._mode

    # -----------------------------------------------------------------------
    # Internal C-native implementation
    # -----------------------------------------------------------------------

    def _dock_native(
        self, antigen_pdb, antigen_centre, antibody,
        n_steps, pocket_radius, blur, delta_G, kT, verbose,
    ) -> AntibodyDockingResult:
        lib = self._lib
        dim = self._DIM
        res = self._RES
        pc  = antigen_centre

        # Load antigen surface grid
        grid = lib.nibble_create_grid(dim, dim, dim, ctypes.c_float(res))
        lib.nibble_load_pdb_pocket(
            grid, antigen_pdb.encode(),
            ctypes.c_float(pc[0]), ctypes.c_float(pc[1]), ctypes.c_float(pc[2]),
            ctypes.c_float(pocket_radius), ctypes.c_float(blur),
        )

        # Build antibody struct
        ab_struct = ABAntibodyStruct()
        n_cdr_res = self._build_ab_struct(antibody, ab_struct)

        if n_cdr_res <= 0:
            lib.nibble_free_grid(grid)
            return AntibodyDockingResult(
                backend="C-Native-Error-Build",
                source=antibody.name,
            )

        # Score
        dock_result = ABDockingResultStruct()
        composite = lib.nibble_antibody_score(
            grid,
            ctypes.byref(ab_struct),
            ctypes.c_int(n_steps),
            ctypes.c_float(delta_G),
            ctypes.c_float(kT),
            ctypes.byref(dock_result),
        )

        if verbose:
            lib.nibble_antibody_print(ctypes.byref(ab_struct))
            lib.nibble_antibody_print_result(ctypes.byref(dock_result))

        # Extract CDR sequences for display
        cdr_seqs = {}
        for i in range(_AB_N_CDR_LOOPS):
            seq = ab_struct.cdrs[i].sequence.decode().strip('\x00')
            if seq:
                cdr_seqs[CDR_NAMES[i]] = seq

        result = AntibodyDockingResult(
            composite_score=float(dock_result.composite_score),
            paratope_epitope_sc=float(dock_result.paratope_epitope_sc),
            h3_affinity=float(dock_result.h3_affinity),
            canonical_cdr_score=float(dock_result.canonical_cdr_score),
            buried_surface_area=float(dock_result.buried_surface_area),
            vhvl_orientation_penalty=float(dock_result.vhvl_orientation_penalty),
            cdr_scores=[float(dock_result.cdr_scores[i]) for i in range(6)],
            n_epitope_patches=int(dock_result.n_epitope_patches),
            best_epitope_score=float(dock_result.best_epitope_complementarity),
            kd_class=int(dock_result.kd_class),
            cdr_sequences=cdr_seqs,
            elbow_angle=float(ab_struct.framework.elbow_angle),
            source=antibody.name,
            backend="C-Native",
        )

        lib.nibble_antibody_free(ctypes.byref(ab_struct))
        lib.nibble_free_grid(grid)

        return result

    def _build_ab_struct(
        self, antibody: AntibodyMolecule, ab_struct: ABAntibodyStruct
    ) -> int:
        """Build C struct from Python AntibodyMolecule. Returns CDR residue count."""
        lib = self._lib

        if antibody.from_pdb_mode:
            # Parse from PDB
            n = lib.nibble_antibody_parse(
                antibody.pdb_path.encode(),
                antibody.vh_chain.encode()[0:1],
                antibody.vl_chain.encode()[0:1],
                ctypes.byref(ab_struct),
            )
            return n
        else:
            # Build from sequences
            seqs = antibody.cdr_sequences
            seq_ptrs = []
            seq_bytes = []
            for name in CDR_NAMES:
                s = seqs.get(name, "")
                b = s.encode() if s else b""
                seq_bytes.append(b)
                seq_ptrs.append(b if b else None)

            c_seq_array = (ctypes.c_char_p * _AB_N_CDR_LOOPS)(*seq_ptrs)
            n = lib.nibble_antibody_build_from_sequences(
                c_seq_array,
                ctypes.byref(ab_struct),
            )
            return n


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_antibody_engine() -> NibbleAntibodyEngine:
    """
    Factory mirroring Khukuri conventions.

    Examples
    --------
    from nibble_antibody_bridge import make_antibody_engine, AntibodyMolecule

    engine = make_antibody_engine()
    print(f"Backend: {engine.mode}")

    # 1. From PDB (trastuzumab vs HER2)
    ab = AntibodyMolecule.from_pdb("trastuzumab_fv.pdb", vh_chain='H', vl_chain='L')
    result = engine.dock("her2_ecd.pdb", (45.2, 31.8, 22.4), ab, verbose=True)
    print(result)
    print(f"Binding: {result.binding_class}")
    print(f"H3-dominated: {result.h3_dominated}")

    # 2. De novo design (specify CDR sequences)
    ab = AntibodyMolecule.from_sequences(
        L1="RASQDVNTAVA", L2="SASFLYS",   L3="QQHYTTPPT",
        H1="GFTFSSYW",   H2="IKDKSGAT",  H3="VFYGSKTDYFDY",
        name="trastuzumab_approx",
    )
    result = engine.dock("her2_ecd.pdb", (45.2, 31.8, 22.4), ab)
    print(result)

    # 3. Epitope mapping only (fast, no H3 trajectory)
    patches = engine.scan_epitopes("antigen.pdb", (10.0, 10.0, 10.0), ab)
    print(f"Found {len(patches)} epitope patches:")
    for p in patches[:3]:
        print(f"  score={p['complementarity']:.3f}"
              f"  area={p['area']:.1f}Å²"
              f"  hbonds={p['n_hbonds']}")

    # 4. CDR-H3 length scan for design
    h3_lengths = [6, 8, 10, 12, 14]
    for n in h3_lengths:
        import random, string
        h3 = ''.join(random.choices('ACDEFGHIKLMNPQRSTVWY', k=n))
        ab_trial = AntibodyMolecule.from_sequences(
            H3=h3, H1="GYTFTSYW", H2="IYPGDGDT",
            L1="RASQSIS", L2="AASSLQS", L3="QQSYTTP",
        )
        r = engine.dock("antigen.pdb", (10.0, 10.0, 10.0), ab_trial)
        print(f"  H3 len={n:2d}  score={r.composite_score:.4f}"
              f"  [{r.binding_class}]")
    """
    return NibbleAntibodyEngine()
