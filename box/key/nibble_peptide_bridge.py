"""
nibble_peptide_bridge.py
------------------------
Python ctypes bindings for the Nibble peptide-protein docking extension.

Plugs into the existing NibbleEngine architecture.
Shares the same loaded shared library. Zero changes to existing files.

Full peptide docking pipeline:
  1. Build peptide from sequence + SS string  → PeptideMolecule
  2. Add constraints (staple, disulfide etc.) → optional
  3. Load protein surface grid               → NibbleGrid (existing)
  4. Run full docking pipeline               → PeptideDockingResult
  5. Extract pharmacophore for design        → optional NibbleGrid

Usage
-----
  from nibble_peptide_bridge import make_peptide_engine, PeptideMolecule

  engine = make_peptide_engine()

  # Stapled helix (MDM2 p53 activation domain mimic)
  pep = PeptideMolecule(
      sequence  = "ETFSDLWKLLPEN",
      ss_string = "HHHHHHHHHHHH",
      staple    = (2, 6),        # i, i+4 staple
  )

  result = engine.dock(
      protein_pdb    = "mdm2.pdb",
      pocket_centre  = (13.2, 9.8, 21.4),
      peptide        = pep,
      n_steps        = 500,
      verbose        = True,
  )
  print(result)
  print(f"Class: {result.binding_class}")

Author: Khukuri / KeyBox project (sangeet01)
License: Apache 2.0 with Commons Clause
"""

import ctypes
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from nibble_bridge import (
    NibbleEngine,
    NibbleGridStruct,
    N_CHANNELS,
)


# ---------------------------------------------------------------------------
# ctypes mirrors
# ---------------------------------------------------------------------------

_PEP_MAX_RESIDUES   = 64
_PEP_MAX_SEGMENTS   = 16
_PEP_MAX_CONSTRAINTS= 32
_PEP_MAX_GROOVES    = 20


class PEPResidueStruct(ctypes.Structure):
    _fields_ = [
        ("resname",        ctypes.c_char * 4),
        ("one_letter",     ctypes.c_char),
        ("ca_x",           ctypes.c_float),
        ("ca_y",           ctypes.c_float),
        ("ca_z",           ctypes.c_float),
        ("phi",            ctypes.c_float),
        ("psi",            ctypes.c_float),
        ("ss",             ctypes.c_int),
        ("hydrophobicity", ctypes.c_float),
        ("charge",         ctypes.c_float),
        ("hbd_count",      ctypes.c_int),
        ("hba_count",      ctypes.c_int),
        ("sasa",           ctypes.c_float),
        ("is_constrained", ctypes.c_int),
        ("d",              ctypes.c_float * N_CHANNELS),
    ]


class PEPSegmentStruct(ctypes.Structure):
    _fields_ = [
        ("res_start",    ctypes.c_int),
        ("res_end",      ctypes.c_int),
        ("cx",           ctypes.c_float),
        ("cy",           ctypes.c_float),
        ("cz",           ctypes.c_float),
        ("qw",           ctypes.c_float),
        ("qx",           ctypes.c_float),
        ("qy",           ctypes.c_float),
        ("qz",           ctypes.c_float),
        ("vx",           ctypes.c_float),
        ("vy",           ctypes.c_float),
        ("vz",           ctypes.c_float),
        ("wx",           ctypes.c_float),
        ("wy",           ctypes.c_float),
        ("wz",           ctypes.c_float),
        ("mass",         ctypes.c_float),
        ("inertia_xx",   ctypes.c_float),
        ("inertia_yy",   ctypes.c_float),
        ("inertia_zz",   ctypes.c_float),
        ("d",            ctypes.c_float * N_CHANNELS),
    ]


class PEPConstraintStruct(ctypes.Structure):
    _fields_ = [
        ("type",         ctypes.c_int),
        ("res_i",        ctypes.c_int),
        ("res_j",        ctypes.c_int),
        ("eq_distance",  ctypes.c_float),
        ("spring_k",     ctypes.c_float),
    ]


class PEPGrooveStruct(ctypes.Structure):
    _fields_ = [
        ("entry_x",               ctypes.c_float),
        ("entry_y",               ctypes.c_float),
        ("entry_z",               ctypes.c_float),
        ("axis_x",                ctypes.c_float),
        ("axis_y",                ctypes.c_float),
        ("axis_z",                ctypes.c_float),
        ("length",                ctypes.c_float),
        ("width",                 ctypes.c_float),
        ("depth",                 ctypes.c_float),
        ("affinity_score",        ctypes.c_float),
        ("n_hbond_sites",         ctypes.c_int),
        ("n_hydrophobic_patches", ctypes.c_int),
    ]


class PEPPeptideStruct(ctypes.Structure):
    _fields_ = [
        ("residues",         PEPResidueStruct  * _PEP_MAX_RESIDUES),
        ("n_residues",       ctypes.c_int),
        ("segments",         PEPSegmentStruct  * _PEP_MAX_SEGMENTS),
        ("n_segments",       ctypes.c_int),
        ("constraints",      PEPConstraintStruct * _PEP_MAX_CONSTRAINTS),
        ("n_constraints",    ctypes.c_int),
        ("dominant_ss",      ctypes.c_int),
        ("is_cyclic",        ctypes.c_int),
        ("is_stapled",       ctypes.c_int),
        ("net_charge",       ctypes.c_float),
        ("hydrophobic_moment",ctypes.c_float),
        ("com_x",            ctypes.c_float),
        ("com_y",            ctypes.c_float),
        ("com_z",            ctypes.c_float),
        ("length_angstrom",  ctypes.c_float),
    ]


class PEPDockingResultStruct(ctypes.Structure):
    _fields_ = [
        ("composite_score",    ctypes.c_float),
        ("affinity_score",     ctypes.c_float),
        ("ss_bias_score",      ctypes.c_float),
        ("groove_match_score", ctypes.c_float),
        ("constraint_penalty", ctypes.c_float),
        ("amphipathic_score",  ctypes.c_float),
        ("n_hbonds_formed",    ctypes.c_float),
        ("ca_coords",          ctypes.c_float * (_PEP_MAX_RESIDUES * 3)),
        ("groove_index",       ctypes.c_int),
        ("rmsd_from_start",    ctypes.c_float),
    ]


# ---------------------------------------------------------------------------
# Python result container
# ---------------------------------------------------------------------------

_SS_NAMES = {0: "Coil", 1: "Helix", 2: "Strand", 3: "Turn"}

_CONSTRAINT_TYPES = {
    "staple":    0,
    "cyclic":    1,
    "disulfide": 2,
    "lactam":    3,
}


@dataclass
class PeptideDockingResult:
    """Full result of a peptide-protein docking run."""

    composite_score:    float = 0.0
    affinity_score:     float = 0.0
    ss_bias_score:      float = 0.0
    groove_match_score: float = 0.0
    constraint_penalty: float = 0.0
    amphipathic_score:  float = 0.0
    n_hbonds_formed:    float = 0.0
    groove_index:       int   = -1
    rmsd_from_start:    float = 0.0

    # Peptide properties
    sequence:           str   = ""
    dominant_ss:        str   = "Coil"
    n_residues:         int   = 0
    net_charge:         float = 0.0
    hydrophobic_moment: float = 0.0
    is_cyclic:          bool  = False
    is_stapled:         bool  = False

    # Grooves detected
    grooves_detected:   int   = 0

    # Best pose Cα coordinates
    ca_coords:          list  = field(default_factory=list)

    backend:            str   = "unknown"

    def __str__(self) -> str:
        lines = [
            "─── Peptide Docking Result ─────────────────",
            f"  Composite score:      {self.composite_score:.4f}",
            f"  Affinity score:       {self.affinity_score:.4f}",
            f"─── Scoring Components ─────────────────────",
            f"  SS bias score:        {self.ss_bias_score:.4f}",
            f"  Groove match:         {self.groove_match_score:.4f}"
            f"  (groove {self.groove_index})",
            f"  Amphipathic score:    {self.amphipathic_score:.4f}",
            f"  Constraint penalty:   {self.constraint_penalty:.4f}",
            f"  Est. H-bonds:         {self.n_hbonds_formed:.1f}",
            f"─── Peptide Properties ─────────────────────",
            f"  Sequence:             {self.sequence}",
            f"  Length:               {self.n_residues} residues",
            f"  Dominant SS:          {self.dominant_ss}",
            f"  Net charge:           {self.net_charge:+.1f}",
            f"  Hydrophobic moment:   {self.hydrophobic_moment:.4f}"
            f"  {'(amphipathic)' if self.hydrophobic_moment > 0.4 else ''}",
            f"  Cyclic:               {'yes' if self.is_cyclic else 'no'}",
            f"  Stapled:              {'yes' if self.is_stapled else 'no'}",
            f"  Grooves detected:     {self.grooves_detected}",
            f"  Backend:              {self.backend}",
        ]
        return "\n".join(lines)

    @property
    def binding_class(self) -> str:
        s = self.composite_score
        if s >= 0.70: return "Strong binder"
        if s >= 0.50: return "Moderate binder"
        if s >= 0.30: return "Weak binder"
        return "Non-binder"

    @property
    def is_productive(self) -> bool:
        return (self.composite_score > 0.45 and
                self.constraint_penalty < 0.3)


# ---------------------------------------------------------------------------
# PeptideMolecule — Python-side peptide definition
# ---------------------------------------------------------------------------

class PeptideMolecule:
    """
    Defines a peptide for docking.

    Args:
        sequence:    One-letter amino acid sequence (e.g. "ACDEFGHIK")
        ss_string:   Per-residue SS: 'H'=helix, 'E'=strand, 'C'=coil
                     (same length as sequence, or None for all-coil)
        is_cyclic:   True if N-C cyclised
        staple:      (i, j) tuple for hydrocarbon staple between residues i and j
                     (i+4 for alpha-helix staple, i+7 for longer span)
        disulfide:   (i, j) tuple for Cys-Cys disulfide
        lactam:      (i, j) tuple for side-chain lactam bridge
    """

    def __init__(
        self,
        sequence:   str,
        ss_string:  str  = None,
        is_cyclic:  bool = False,
        staple:     tuple = None,
        disulfide:  tuple = None,
        lactam:     tuple = None,
    ):
        self.sequence  = sequence.upper().strip()
        self.ss_string = ss_string
        self.is_cyclic = is_cyclic
        self.staple    = staple
        self.disulfide = disulfide
        self.lactam    = lactam

        # Validate sequence
        valid = set("ACDEFGHIKLMNPQRSTVWY")
        for aa in self.sequence:
            if aa not in valid:
                raise ValueError(f"Unknown amino acid '{aa}' in sequence.")

        if ss_string and len(ss_string) != len(sequence):
            raise ValueError(
                f"ss_string length ({len(ss_string)}) must match "
                f"sequence length ({len(sequence)})."
            )

    @property
    def n_residues(self) -> int:
        return len(self.sequence)

    @property
    def has_constraints(self) -> bool:
        return any([
            self.staple, self.disulfide, self.lactam, self.is_cyclic
        ])

    @classmethod
    def stapled_helix(
        cls,
        sequence:  str,
        staple_i:  int = None,
        staple_j:  int = None,
    ) -> "PeptideMolecule":
        """
        Convenience constructor for a stapled alpha-helix.
        If staple_i/j not given, places i,i+4 staple at centre.
        """
        n = len(sequence)
        ss = "H" * n
        if staple_i is None:
            staple_i = n // 2 - 2
        if staple_j is None:
            staple_j = staple_i + 4
        return cls(sequence, ss_string=ss, staple=(staple_i, staple_j))

    @classmethod
    def cyclic_peptide(cls, sequence: str) -> "PeptideMolecule":
        """Convenience constructor for a cyclic peptide."""
        return cls(sequence, is_cyclic=True)


# ---------------------------------------------------------------------------
# NumPy fallback peptide scorer
# ---------------------------------------------------------------------------

class _NumPyPeptideScorer:
    """
    Simplified NumPy-based peptide scoring for when C library is unavailable.
    """

    _AA_HYDRO = {
        'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
        'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
        'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
        'S': -0.18, 'T': -0.05, 'W': -0.90, 'Y': 0.26, 'V': 1.08,
    }
    _AA_CHARGE = {
        'R': 1.0, 'K': 1.0, 'H': 0.1,
        'D': -1.0, 'E': -1.0,
    }

    def score(self, peptide: PeptideMolecule) -> PeptideDockingResult:
        seq = peptide.sequence
        n   = len(seq)

        # Hydrophobic moment
        angle_per = 2 * math.pi / 3.6
        sum_x = sum_y = 0.0
        for i, aa in enumerate(seq):
            h = self._AA_HYDRO.get(aa, 0.0)
            sum_x += h * math.cos(angle_per * i)
            sum_y += h * math.sin(angle_per * i)
        moment = math.sqrt(sum_x**2 + sum_y**2) / n

        # Net charge
        charge = sum(self._AA_CHARGE.get(aa, 0.0) for aa in seq)

        # Rough affinity proxy
        n_hydro = sum(1 for aa in seq if self._AA_HYDRO.get(aa, 0) > 0.5)
        n_hbond = sum(1 for aa in seq if aa in "STNQHRKDEY")
        affinity = min(1.0, (n_hydro * 0.04 + n_hbond * 0.03))

        ss = peptide.ss_string
        ss_score = 0.3
        if ss:
            n_helix  = ss.count('H') / n
            n_strand = ss.count('E') / n
            ss_score = max(n_helix, n_strand) * 0.5

        constraint_pen = 0.0
        if peptide.staple:
            i, j = peptide.staple
            ideal = 8.0 if (j - i) == 4 else 11.0
            actual = abs(j - i) * 1.5
            constraint_pen = abs(actual - ideal) / ideal * 0.5

        composite = min(1.0, max(0.0,
            0.50 * affinity + 0.15 * ss_score +
            0.10 * min(1.0, moment * 2) - 0.10 * constraint_pen
        ))

        dominant_ss = "Helix" if (ss and ss.count('H') > n/2) else \
                      "Strand" if (ss and ss.count('E') > n/3) else "Coil"

        return PeptideDockingResult(
            composite_score=composite,
            affinity_score=affinity,
            ss_bias_score=ss_score,
            groove_match_score=0.3,
            constraint_penalty=constraint_pen,
            amphipathic_score=min(1.0, moment * 2),
            n_hbonds_formed=n_hbond * 0.4,
            sequence=peptide.sequence,
            dominant_ss=dominant_ss,
            n_residues=n,
            net_charge=charge,
            hydrophobic_moment=moment,
            is_cyclic=peptide.is_cyclic,
            is_stapled=peptide.staple is not None,
            backend="NumPy-Fallback",
        )


# ---------------------------------------------------------------------------
# NibblePeptideEngine — main engine
# ---------------------------------------------------------------------------

class NibblePeptideEngine:
    """
    Peptide-protein docking engine.

    Wraps nibble_peptide.c via ctypes with NumPy fallback.
    Shares the loaded library with NibbleEngine, NibblePPIEngine,
    NibblePROTACEngine.

    Supports:
      - Linear peptides (all SS types)
      - Stapled alpha-helices (hydrocarbon staple)
      - Cyclic peptides (N-C cyclisation)
      - Disulfide-constrained peptides
      - Lactam-bridged peptides
      - Groove-seeded trajectory optimisation
      - Amphipathic helix recognition

    Quick start
    -----------
    engine = NibblePeptideEngine()

    # Stapled p53 helix vs MDM2
    pep    = PeptideMolecule.stapled_helix("ETFSDLWKLLPEN")
    result = engine.dock("mdm2.pdb", (13.2, 9.8, 21.4), pep)
    print(result)
    """

    _DIM    = 60
    _RES    = 0.5
    _RADIUS = 20.0   # larger than small-molecule — peptides need more surface

    def __init__(self):
        self._lib      = NibbleEngine._lib
        self._fallback = _NumPyPeptideScorer()

        if self._lib is not None:
            self._register_functions()
            self._mode = "C-Native"
        else:
            self._mode = "NumPy-Fallback"

    def _register_functions(self):
        lib = self._lib
        _G   = ctypes.POINTER(NibbleGridStruct)
        _F   = ctypes.c_float
        _PP  = ctypes.POINTER(PEPPeptideStruct)
        _PG  = ctypes.POINTER(PEPGrooveStruct)
        _PR  = ctypes.POINTER(PEPDockingResultStruct)
        _INT = ctypes.c_int
        _STR = ctypes.c_char_p

        lib.nibble_peptide_build.argtypes = [_STR, _STR, _INT, _PP]
        lib.nibble_peptide_build.restype  = _INT

        lib.nibble_peptide_add_constraint.argtypes = [_PP, _INT, _INT, _INT, _F]
        lib.nibble_peptide_add_constraint.restype  = _INT

        lib.nibble_peptide_scan_grooves.argtypes = [_G, _PG, _F]
        lib.nibble_peptide_scan_grooves.restype  = _INT

        lib.nibble_peptide_ss_bias.argtypes = [_G, _PP, _PG, _F]
        lib.nibble_peptide_ss_bias.restype  = None

        lib.nibble_peptide_project.argtypes = [_G, _PP, _INT, _F]
        lib.nibble_peptide_project.restype  = None

        lib.nibble_peptide_trajectory.argtypes = [
            _G, _PP, _PG, _INT, _F, _F, _F, _PR
        ]
        lib.nibble_peptide_trajectory.restype  = _F

        lib.nibble_peptide_hydrophobic_moment.argtypes = [_PP, _INT]
        lib.nibble_peptide_hydrophobic_moment.restype  = _F

        lib.nibble_peptide_dock.argtypes = [_G, _PP, _F, _F, _INT, _PR]
        lib.nibble_peptide_dock.restype  = _F

        lib.nibble_peptide_print_result.argtypes = [_PR]
        lib.nibble_peptide_print_result.restype  = None

        lib.nibble_peptide_print.argtypes = [_PP]
        lib.nibble_peptide_print.restype  = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def dock(
        self,
        protein_pdb:   str,
        pocket_centre: tuple,
        peptide:       PeptideMolecule,
        n_steps:       int   = 500,
        pocket_radius: float = 20.0,
        blur:          float = 1.5,
        delta_G:       float = 2.0,
        kT:            float = 0.593,
        verbose:       bool  = False,
    ) -> PeptideDockingResult:
        """
        Dock a peptide against a protein surface.

        Args:
            protein_pdb:   Path to protein PDB file.
            pocket_centre: (x, y, z) of binding region centre (Å).
            peptide:       PeptideMolecule definition.
            n_steps:       Langevin trajectory steps (default 500).
            pocket_radius: Surface loading radius (Å, default 20 — larger for peptides).
            blur:          Gaussian blur for surface loading.
            delta_G:       Water desolvation energy (kcal/mol).
            kT:            Thermal energy (kcal/mol, 0.593 at 298K).
            verbose:       Print C-level summary to stdout.

        Returns:
            PeptideDockingResult with all scoring components.
        """
        if self._mode == "NumPy-Fallback":
            result = self._fallback.score(peptide)
            result.grooves_detected = 0
            return result

        return self._dock_native(
            protein_pdb, pocket_centre, peptide,
            n_steps, pocket_radius, blur, delta_G, kT, verbose,
        )

    def scan_grooves(
        self,
        protein_pdb:   str,
        pocket_centre: tuple,
        pocket_radius: float = 20.0,
        min_length:    float = 8.0,
    ) -> List[dict]:
        """
        Scan for peptide binding grooves on a protein surface.
        Returns list of groove dicts sorted by affinity score.

        Useful for identifying binding sites before designing a peptide.
        """
        if self._mode == "NumPy-Fallback":
            return []

        lib = self._lib
        dim = self._DIM
        res = self._RES
        pc  = pocket_centre

        grid = lib.nibble_create_grid(dim, dim, dim, ctypes.c_float(res))
        lib.nibble_load_pdb_pocket(
            grid, protein_pdb.encode(),
            ctypes.c_float(pc[0]), ctypes.c_float(pc[1]), ctypes.c_float(pc[2]),
            ctypes.c_float(pocket_radius), ctypes.c_float(1.5),
        )

        groove_arr = (PEPGrooveStruct * _PEP_MAX_GROOVES)()
        n = lib.nibble_peptide_scan_grooves(
            grid, groove_arr, ctypes.c_float(min_length)
        )
        lib.nibble_free_grid(grid)

        grooves = []
        for i in range(n):
            g = groove_arr[i]
            grooves.append({
                "index":      i,
                "entry":      (g.entry_x, g.entry_y, g.entry_z),
                "axis":       (g.axis_x, g.axis_y, g.axis_z),
                "length":     g.length,
                "width":      g.width,
                "depth":      g.depth,
                "affinity":   g.affinity_score,
                "n_hbonds":   g.n_hbond_sites,
                "n_hydro":    g.n_hydrophobic_patches,
            })
        return grooves

    def compute_hydrophobic_moment(self, peptide: PeptideMolecule) -> float:
        """Compute Eisenberg hydrophobic moment for a peptide."""
        if self._mode == "NumPy-Fallback":
            return self._fallback.score(peptide).hydrophobic_moment

        lib  = self._lib
        pep_struct = PEPPeptideStruct()
        ss_bytes = peptide.ss_string.encode() if peptide.ss_string else None
        lib.nibble_peptide_build(
            peptide.sequence.encode(),
            ss_bytes,
            ctypes.c_int(1 if peptide.is_cyclic else 0),
            ctypes.byref(pep_struct),
        )
        ss_val = self._dominant_ss_int(pep_struct.dominant_ss)
        return lib.nibble_peptide_hydrophobic_moment(
            ctypes.byref(pep_struct), ctypes.c_int(ss_val)
        )

    @property
    def mode(self) -> str:
        return self._mode

    # -----------------------------------------------------------------------
    # Internal C-native implementation
    # -----------------------------------------------------------------------

    def _dock_native(
        self, protein_pdb, pocket_centre, peptide,
        n_steps, pocket_radius, blur, delta_G, kT, verbose,
    ) -> PeptideDockingResult:
        lib = self._lib
        dim = self._DIM
        res = self._RES
        pc  = pocket_centre

        # Load protein surface grid
        grid = lib.nibble_create_grid(dim, dim, dim, ctypes.c_float(res))
        lib.nibble_load_pdb_pocket(
            grid, protein_pdb.encode(),
            ctypes.c_float(pc[0]), ctypes.c_float(pc[1]), ctypes.c_float(pc[2]),
            ctypes.c_float(pocket_radius), ctypes.c_float(blur),
        )

        # Build peptide struct
        pep_struct = PEPPeptideStruct()
        ss_bytes   = peptide.ss_string.encode() if peptide.ss_string else None
        n_res = lib.nibble_peptide_build(
            peptide.sequence.encode(),
            ss_bytes,
            ctypes.c_int(1 if peptide.is_cyclic else 0),
            ctypes.byref(pep_struct),
        )

        if n_res <= 0:
            lib.nibble_free_grid(grid)
            return PeptideDockingResult(
                backend="C-Native-Error-Build",
                sequence=peptide.sequence,
            )

        # Add constraints
        if peptide.staple:
            lib.nibble_peptide_add_constraint(
                ctypes.byref(pep_struct),
                ctypes.c_int(_CONSTRAINT_TYPES["staple"]),
                ctypes.c_int(peptide.staple[0]),
                ctypes.c_int(peptide.staple[1]),
                ctypes.c_float(10.0),
            )
        if peptide.disulfide:
            lib.nibble_peptide_add_constraint(
                ctypes.byref(pep_struct),
                ctypes.c_int(_CONSTRAINT_TYPES["disulfide"]),
                ctypes.c_int(peptide.disulfide[0]),
                ctypes.c_int(peptide.disulfide[1]),
                ctypes.c_float(15.0),
            )
        if peptide.lactam:
            lib.nibble_peptide_add_constraint(
                ctypes.byref(pep_struct),
                ctypes.c_int(_CONSTRAINT_TYPES["lactam"]),
                ctypes.c_int(peptide.lactam[0]),
                ctypes.c_int(peptide.lactam[1]),
                ctypes.c_float(8.0),
            )
        if peptide.is_cyclic and not peptide.staple:
            lib.nibble_peptide_add_constraint(
                ctypes.byref(pep_struct),
                ctypes.c_int(_CONSTRAINT_TYPES["cyclic"]),
                ctypes.c_int(0),
                ctypes.c_int(n_res - 1),
                ctypes.c_float(12.0),
            )

        # Scan grooves first to get count
        groove_arr   = (PEPGrooveStruct * _PEP_MAX_GROOVES)()
        n_grooves = lib.nibble_peptide_scan_grooves(
            grid, groove_arr, ctypes.c_float(8.0)
        )

        # Run full docking pipeline
        dock_result = PEPDockingResultStruct()
        composite = lib.nibble_peptide_dock(
            grid,
            ctypes.byref(pep_struct),
            ctypes.c_float(delta_G),
            ctypes.c_float(kT),
            ctypes.c_int(n_steps),
            ctypes.byref(dock_result),
        )

        if verbose:
            lib.nibble_peptide_print(ctypes.byref(pep_struct))
            lib.nibble_peptide_print_result(ctypes.byref(dock_result))

        # Extract Cα coordinates
        ca_coords = [
            (dock_result.ca_coords[i*3],
             dock_result.ca_coords[i*3+1],
             dock_result.ca_coords[i*3+2])
            for i in range(n_res)
        ]

        result = PeptideDockingResult(
            composite_score=float(dock_result.composite_score),
            affinity_score=float(dock_result.affinity_score),
            ss_bias_score=float(dock_result.ss_bias_score),
            groove_match_score=float(dock_result.groove_match_score),
            constraint_penalty=float(dock_result.constraint_penalty),
            amphipathic_score=float(dock_result.amphipathic_score),
            n_hbonds_formed=float(dock_result.n_hbonds_formed),
            groove_index=int(dock_result.groove_index),
            rmsd_from_start=float(dock_result.rmsd_from_start),
            sequence=peptide.sequence,
            dominant_ss=_SS_NAMES.get(pep_struct.dominant_ss, "Coil"),
            n_residues=n_res,
            net_charge=float(pep_struct.net_charge),
            hydrophobic_moment=float(pep_struct.hydrophobic_moment),
            is_cyclic=bool(pep_struct.is_cyclic),
            is_stapled=bool(pep_struct.is_stapled),
            grooves_detected=n_grooves,
            ca_coords=ca_coords,
            backend="C-Native",
        )

        lib.nibble_free_grid(grid)
        return result

    @staticmethod
    def _dominant_ss_int(ss: int) -> int:
        return ss


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_peptide_engine() -> NibblePeptideEngine:
    """
    Factory mirroring Khukuri conventions.

    Examples
    --------
    from nibble_peptide_bridge import make_peptide_engine, PeptideMolecule

    engine = make_peptide_engine()

    # 1. Stapled p53 helix vs MDM2 (classic PPI disruption)
    pep = PeptideMolecule.stapled_helix(
        sequence  = "ETFSDLWKLLPEN",
        staple_i  = 2,
        staple_j  = 6,
    )
    result = engine.dock("mdm2.pdb", (13.2, 9.8, 21.4), pep, verbose=True)
    print(result)
    print(f"Class: {result.binding_class}")

    # 2. Cyclic peptide vs integrin αvβ3
    pep = PeptideMolecule.cyclic_peptide("CRGDFC")   # RGD motif
    result = engine.dock("integrin_avb3.pdb", (8.5, 22.1, 14.3), pep)
    print(result)

    # 3. Scan grooves first then design peptide
    grooves = engine.scan_grooves("target.pdb", (10.0, 10.0, 10.0))
    print(f"Found {len(grooves)} grooves:")
    for g in grooves:
        print(f"  length={g['length']:.1f}Å  affinity={g['affinity']:.3f}")

    # 4. Amphipathic helix (membrane-active)
    pep = PeptideMolecule(
        sequence  = "KLALKLALKALKAALK",
        ss_string = "H" * 16,
    )
    moment = engine.compute_hydrophobic_moment(pep)
    print(f"Hydrophobic moment: {moment:.4f}")

    # 5. Disulfide-constrained peptide
    pep = PeptideMolecule(
        sequence   = "ACDEFGHICKLM",
        disulfide  = (1, 8),    # Cys1-Cys8 bridge
    )
    result = engine.dock("receptor.pdb", (5.0, 15.0, 8.0), pep)
    print(result)
    """
    return NibblePeptideEngine()
