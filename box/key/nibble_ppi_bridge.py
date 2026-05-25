"""
nibble_ppi_bridge.py
--------------------
Python ctypes bindings for the Nibble PPI extension (nibble_ppi.c).

Plugs into the existing NibbleEngine architecture from nibble_bridge.py
without modifying it. Loads the same shared library and registers the
six new PPI functions.

Six-phase PPI pipeline:
  1. detect_interface()        — find residues at the PPI contact zone
  2. load_ppi_surface()        — load irregular surface patch into NibbleGrid
  3. sc_score()                — Lawrence-Coleman shape complementarity
  4. score_hotspots()          — alanine-scanning proxy hotspot detection
  5. ppi_affinity()            — composite PPI binding score
  6. load_ppi_inhibitor_site() — pharmacophore extraction for inhibitor design

Usage
-----
  from nibble_ppi_bridge import NibblePPIEngine

  ppi = NibblePPIEngine()

  # Score a protein-protein interface
  result = ppi.score_interface("proteinA.pdb", "proteinB.pdb", chain_a='A', chain_b='B')
  print(f"PPI composite score: {result.composite_score:.4f}")
  print(f"Shape complementarity: {result.sc_score:.4f}")
  print(f"Hotspots: {result.n_hotspots}")

  # Extract pharmacophore for inhibitor design
  pharm = ppi.extract_inhibitor_pharmacophore("proteinA.pdb", "proteinB.pdb")
  # pharm is a NibbleGrid — pass directly to nibble_trajectory() for docking

Author: Khukuri / KeyBox project (sangeet01)
License: Apache 2.0 with Commons Clause
"""

import ctypes
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# Import from existing nibble_bridge (same package)
from nibble_bridge import (
    NibbleGridStruct,
    NibbleMolStruct,
    NativeNibbleEngine,
    NumPyNibbleEngine,
    NibbleEngine,
    N_CHANNELS,
    CH_STERIC_DEMAND,
    CH_ELEC_DEMAND,
    CH_HBA_DEMAND,
    CH_HBD_DEMAND,
    CH_LIPO_DEMAND,
    CH_AROM_DEMAND,
    CH_METAL_DEMAND,
    CH_PHOBIC_CORE,
    CH_FLEXIBILITY,
    CH_NUCLEOPHILICITY,
    CH_ELECTROPHILICITY,
    CH_WATER_CONSERVED,
    CH_WATER_DYNAMIC,
)

# ---------------------------------------------------------------------------
# ctypes mirror of PPI_Residue C struct
# ---------------------------------------------------------------------------
class PPIResidueStruct(ctypes.Structure):
    _fields_ = [
        ("chain",          ctypes.c_char),
        ("resnum",         ctypes.c_int),
        ("resname",        ctypes.c_char * 4),
        ("cx",             ctypes.c_float),
        ("cy",             ctypes.c_float),
        ("cz",             ctypes.c_float),
        ("n_atoms",        ctypes.c_int),
        ("is_hotspot",     ctypes.c_int),
        ("hotspot_weight", ctypes.c_float),
    ]


# ---------------------------------------------------------------------------
# ctypes mirror of PPI_Interface C struct
# ---------------------------------------------------------------------------
_PPI_MAX_RESIDUES = 2000

class PPIInterfaceStruct(ctypes.Structure):
    _fields_ = [
        ("centre_x",           ctypes.c_float),
        ("centre_y",           ctypes.c_float),
        ("centre_z",           ctypes.c_float),
        ("extent_x",           ctypes.c_float),
        ("extent_y",           ctypes.c_float),
        ("extent_z",           ctypes.c_float),
        ("residues_a",         PPIResidueStruct * _PPI_MAX_RESIDUES),
        ("n_residues_a",       ctypes.c_int),
        ("residues_b",         PPIResidueStruct * _PPI_MAX_RESIDUES),
        ("n_residues_b",       ctypes.c_int),
        ("buried_surface_area",ctypes.c_float),
        ("n_hbonds",           ctypes.c_int),
        ("n_hydrophobic",      ctypes.c_int),
        ("n_bridging_waters",  ctypes.c_int),
        ("sc_score",           ctypes.c_float),
        ("mean_hotspot_weight",ctypes.c_float),
    ]


# ---------------------------------------------------------------------------
# ctypes mirror of PPI_Score C struct
# ---------------------------------------------------------------------------
class PPIScoreStruct(ctypes.Structure):
    _fields_ = [
        ("raw_affinity",         ctypes.c_float),
        ("sc_score",             ctypes.c_float),
        ("hotspot_score",        ctypes.c_float),
        ("water_bonus",          ctypes.c_float),
        ("desolvation_penalty",  ctypes.c_float),
        ("composite_score",      ctypes.c_float),
        ("n_hotspots",           ctypes.c_int),
        ("buried_area_a",        ctypes.c_float),
        ("buried_area_b",        ctypes.c_float),
    ]


# ---------------------------------------------------------------------------
# Python result container
# ---------------------------------------------------------------------------
@dataclass
class PPIResult:
    """
    Full result of a PPI scoring run.
    All scores normalised to [0, 1] unless noted.
    """
    # Scores
    composite_score:      float = 0.0
    raw_affinity:         float = 0.0
    sc_score:             float = 0.0   # Lawrence-Coleman shape complementarity
    hotspot_score:        float = 0.0
    water_bonus:          float = 0.0
    desolvation_penalty:  float = 0.0

    # Interface geometry
    n_hotspots:           int   = 0
    n_residues_a:         int   = 0
    n_residues_b:         int   = 0
    n_hbonds:             int   = 0
    n_hydrophobic:        int   = 0
    n_bridging_waters:    int   = 0
    buried_surface_area:  float = 0.0
    buried_area_a:        float = 0.0
    buried_area_b:        float = 0.0

    # Interface centre
    centre: tuple = (0.0, 0.0, 0.0)

    # Hotspot residues (protein A side)
    hotspot_residues_a: list = field(default_factory=list)
    # Hotspot residues (protein B side)
    hotspot_residues_b: list = field(default_factory=list)

    # Backend used
    backend: str = "unknown"

    def __str__(self) -> str:
        lines = [
            "─── PPI Score ─────────────────────────────",
            f"  Composite score:       {self.composite_score:.4f}",
            f"  Shape complementarity: {self.sc_score:.4f}  "
            f"({'good' if self.sc_score >= 0.70 else 'moderate' if self.sc_score >= 0.50 else 'weak'})",
            f"  Raw affinity:          {self.raw_affinity:.2f}",
            f"  Hotspot score:         {self.hotspot_score:.4f}  ({self.n_hotspots} hotspots)",
            f"  Water bonus:           {self.water_bonus:.4f}  ({self.n_bridging_waters} bridging waters)",
            f"  Desolvation penalty:   {self.desolvation_penalty:.4f}",
            f"─── Interface ─────────────────────────────",
            f"  Buried surface area:   {self.buried_surface_area:.1f} Å²",
            f"  Residues (A/B):        {self.n_residues_a} / {self.n_residues_b}",
            f"  H-bonds:               {self.n_hbonds}",
            f"  Hydrophobic contacts:  {self.n_hydrophobic}",
            f"  Centre:                ({self.centre[0]:.2f}, {self.centre[1]:.2f}, {self.centre[2]:.2f})",
            f"  Backend:               {self.backend}",
        ]
        if self.hotspot_residues_a:
            top = sorted(self.hotspot_residues_a, key=lambda r: r['weight'], reverse=True)[:5]
            lines.append("─── Top Hotspot Residues (A) ───────────────")
            for r in top:
                lines.append(f"  {r['resname']}{r['resnum']} (chain {r['chain']})  weight={r['weight']:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# NumPy fallback PPI scorer
# (used when C library is unavailable — same graceful degradation pattern
#  as nibble_bridge.py)
# ---------------------------------------------------------------------------
class _NumPyPPIScorer:
    """
    Pure NumPy PPI scoring fallback.
    Accuracy is lower than the C engine but the API is identical.
    """

    def detect_interface(self, pdb_a, pdb_b, chain_a, chain_b, cutoff):
        """Parse PDB files and find interface residues."""
        atoms_a = self._parse_pdb(pdb_a, chain_a)
        atoms_b = self._parse_pdb(pdb_b, chain_b)

        if not atoms_a or not atoms_b:
            return None, []

        cutoff2 = cutoff * cutoff
        interface_a, interface_b = set(), set()
        n_hbonds = n_hydro = n_waters = 0
        cx = cy = cz = 0.0
        n_pairs = 0

        for a in atoms_a:
            if a['water']:
                continue
            for b in atoms_b:
                if b['water']:
                    continue
                d2 = ((a['x']-b['x'])**2 + (a['y']-b['y'])**2 +
                      (a['z']-b['z'])**2)
                if d2 > cutoff2:
                    continue
                interface_a.add((a['chain'], a['resnum'], a['resname']))
                interface_b.add((b['chain'], b['resnum'], b['resname']))
                d = d2 ** 0.5
                ea, eb = a['element'], b['element']
                if d <= 3.5 and ((ea=='N' and eb=='O') or (ea=='O' and eb=='N')):
                    n_hbonds += 1
                if d <= 4.5 and ea == 'C' and eb == 'C':
                    n_hydro += 1
                cx += a['x']; cy += a['y']; cz += a['z']
                n_pairs += 1

        centre = (cx/max(n_pairs,1), cy/max(n_pairs,1), cz/max(n_pairs,1))
        bsa = (len(interface_a) + len(interface_b)) * 15.0

        iface_info = {
            'centre':               centre,
            'n_residues_a':         len(interface_a),
            'n_residues_b':         len(interface_b),
            'n_hbonds':             n_hbonds,
            'n_hydrophobic':        n_hydro,
            'n_bridging_waters':    n_waters,
            'buried_surface_area':  bsa,
            'residues_a':           list(interface_a),
            'residues_b':           list(interface_b),
        }
        return iface_info, list(interface_a)

    def score(self, pdb_a, pdb_b, chain_a, chain_b, cutoff) -> PPIResult:
        """Simplified NumPy-based PPI score."""
        iface, residues_a = self.detect_interface(
            pdb_a, pdb_b, chain_a, chain_b, cutoff
        )
        if iface is None:
            return PPIResult(backend="NumPy-Fallback-Error")

        atoms_a = self._parse_pdb(pdb_a, chain_a)
        atoms_b = self._parse_pdb(pdb_b, chain_b)

        # Rough complementarity: normalised contact count
        n_contacts = iface['n_hbonds'] + iface['n_hydrophobic']
        sc_approx = min(1.0, n_contacts / 30.0)

        # Hotspot proxy: fraction of interface residues in core
        n_hot = max(1, int(len(residues_a) * 0.3))
        hot_score = min(1.0, n_hot / max(len(residues_a), 1))

        # Water bonus
        water_bonus = iface['n_bridging_waters'] * 0.05

        # Desolvation
        desolv = iface['buried_surface_area'] * 0.0025

        composite = (0.35 * sc_approx + 0.25 * sc_approx +
                     0.20 * hot_score + 0.10 * min(1.0, water_bonus) -
                     0.10 * min(1.0, desolv / 10.0))
        composite = max(0.0, min(1.0, composite))

        return PPIResult(
            composite_score=composite,
            raw_affinity=float(n_contacts * 10.0),
            sc_score=sc_approx,
            hotspot_score=hot_score,
            water_bonus=water_bonus,
            desolvation_penalty=desolv,
            n_hotspots=n_hot,
            n_residues_a=iface['n_residues_a'],
            n_residues_b=iface['n_residues_b'],
            n_hbonds=iface['n_hbonds'],
            n_hydrophobic=iface['n_hydrophobic'],
            n_bridging_waters=iface['n_bridging_waters'],
            buried_surface_area=iface['buried_surface_area'],
            centre=iface['centre'],
            hotspot_residues_a=[
                {'resname': r[2], 'resnum': r[1], 'chain': chr(r[0])
                 if isinstance(r[0], int) else r[0], 'weight': 0.5}
                for r in residues_a[:n_hot]
            ],
            backend="NumPy-Fallback",
        )

    @staticmethod
    def _parse_pdb(path, chain_filter):
        atoms = []
        if not os.path.exists(path):
            return atoms
        with open(path) as fh:
            for line in fh:
                if not line.startswith(('ATOM  ', 'HETATM')):
                    continue
                if len(line) < 54:
                    continue
                chain = line[21] if len(line) > 21 else ' '
                if chain_filter and chain != chain_filter:
                    continue
                resname = line[17:20].strip()
                is_water = resname in ('HOH', 'WAT')
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                elem = line[76:78].strip() if len(line) > 77 else line[13:14].strip()
                if elem and elem[0].islower():
                    elem = elem[0].upper() + elem[1:]
                atoms.append({
                    'chain': chain, 'resnum': int(line[22:26].strip() or 0),
                    'resname': resname, 'element': elem[:1] if elem else 'C',
                    'x': x, 'y': y, 'z': z, 'water': is_water,
                })
        return atoms


# ---------------------------------------------------------------------------
# Main PPI Engine
# ---------------------------------------------------------------------------
class NibblePPIEngine:
    """
    Protein-Protein Interface scoring engine.

    Wraps nibble_ppi.c via ctypes, with NumPy fallback.
    Follows the same NibbleEngine pattern from nibble_bridge.py.

    Quick start
    -----------
    ppi = NibblePPIEngine()
    result = ppi.score_interface("A.pdb", "B.pdb", chain_a='A', chain_b='B')
    print(result)

    # For inhibitor design:
    grid = ppi.extract_inhibitor_pharmacophore("A.pdb", "B.pdb")
    # Pass grid to standard Nibble Langevin docking workflow
    """

    # Grid dimensions for PPI (larger than small-molecule default 30×30×30)
    _PPI_DIM    = 80
    _PPI_RES    = 0.5   # Å — finer resolution for surface detail

    def __init__(self):
        self._lib     = NibbleEngine._lib   # share the already-loaded library
        self._fallback = _NumPyPPIScorer()

        if self._lib is not None:
            self._register_ppi_functions()
            self._mode = "C-Native"
        else:
            self._mode = "NumPy-Fallback"

    def _register_ppi_functions(self):
        """Register ctypes signatures for all six PPI functions."""
        lib = self._lib
        _G   = ctypes.POINTER(NibbleGridStruct)
        _F   = ctypes.c_float
        _PI  = ctypes.POINTER(PPIInterfaceStruct)
        _PS  = ctypes.POINTER(PPIScoreStruct)
        _STR = ctypes.c_char_p
        _INT = ctypes.c_int

        # Phase I: detect interface
        lib.nibble_detect_interface.argtypes = [
            _STR, _STR, ctypes.c_char, ctypes.c_char, _F, _PI
        ]
        lib.nibble_detect_interface.restype = ctypes.c_int

        # Phase II: load surface
        lib.nibble_load_ppi_surface.argtypes = [_G, _STR, _PI, _INT, _F]
        lib.nibble_load_ppi_surface.restype  = ctypes.c_int

        # Phase III: shape complementarity
        lib.nibble_sc_score.argtypes = [_G, _G]
        lib.nibble_sc_score.restype  = ctypes.c_float

        # Phase IV: hotspot scoring
        lib.nibble_score_hotspots.argtypes = [_G, _G, _PI]
        lib.nibble_score_hotspots.restype  = ctypes.c_int

        # Phase V: full PPI affinity
        lib.nibble_ppi_affinity.argtypes = [_G, _G, _PI, _F, _F, _PS]
        lib.nibble_ppi_affinity.restype  = ctypes.c_float

        # Phase VI: inhibitor site
        lib.nibble_load_ppi_inhibitor_site.argtypes = [_G, _G, _PI, _G, _F]
        lib.nibble_load_ppi_inhibitor_site.restype  = ctypes.c_int

        # Utility printers
        lib.nibble_ppi_print_score.argtypes     = [_PS]
        lib.nibble_ppi_print_score.restype      = None
        lib.nibble_ppi_print_interface.argtypes = [_PI]
        lib.nibble_ppi_print_interface.restype  = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def score_interface(
        self,
        pdb_a:     str,
        pdb_b:     str,
        chain_a:   str = '',
        chain_b:   str = '',
        cutoff:    float = 6.0,
        delta_G:   float = 2.0,
        kT:        float = 0.593,
        blur:      float = 1.5,
        verbose:   bool  = False,
    ) -> PPIResult:
        """
        Score the protein-protein interface between two PDB structures.

        Args:
            pdb_a:    Path to PDB file for protein A.
            pdb_b:    Path to PDB file for protein B.
                      Can be the same file as pdb_a if chains differ.
            chain_a:  Chain ID for protein A (empty = any chain).
            chain_b:  Chain ID for protein B (empty = any chain).
            cutoff:   Inter-atom contact distance cutoff in Å (default 6.0).
            delta_G:  Water desolvation free energy (kcal/mol, default 2.0).
            kT:       Thermal energy (kcal/mol, default 0.593 at 298K).
            blur:     Gaussian blur radius for surface loading (Å).
            verbose:  If True, print C-level summary to stdout.

        Returns:
            PPIResult with all scoring components.
        """
        if self._mode == "NumPy-Fallback":
            return self._fallback.score(pdb_a, pdb_b,
                                        chain_a or '', chain_b or '', cutoff)
        return self._score_native(pdb_a, pdb_b, chain_a, chain_b,
                                  cutoff, delta_G, kT, blur, verbose)

    def extract_inhibitor_pharmacophore(
        self,
        pdb_a:              str,
        pdb_b:              str,
        chain_a:            str   = '',
        chain_b:            str   = '',
        cutoff:             float = 6.0,
        blur:               float = 1.5,
        min_hotspot_weight: float = 0.3,
    ) -> Optional[object]:
        """
        Extract the PPI inhibitor pharmacophore as a NibbleGrid.

        The returned grid encodes what a small molecule must present
        to displace protein B from the protein A interface.
        Pass it directly to nibble_trajectory() for Langevin docking.

        Args:
            pdb_a:              Protein A PDB path (the target surface).
            pdb_b:              Protein B PDB path (the partner to displace).
            chain_a/chain_b:    Chain identifiers.
            cutoff:             Contact distance cutoff (Å).
            blur:               Surface loading blur radius.
            min_hotspot_weight: Minimum hotspot weight to include in pharmacophore.

        Returns:
            NibbleGrid pharmacophore pocket (compatible with NibbleEngine),
            or None if interface detection fails.
        """
        if self._mode == "NumPy-Fallback":
            print("[NibblePPIEngine] Warning: inhibitor pharmacophore "
                  "extraction requires C-Native backend. "
                  "Returning None.")
            return None
        return self._extract_pharmacophore_native(
            pdb_a, pdb_b, chain_a, chain_b, cutoff, blur, min_hotspot_weight
        )

    def score_hotspots_only(
        self,
        pdb_a:   str,
        pdb_b:   str,
        chain_a: str = '',
        chain_b: str = '',
        cutoff:  float = 6.0,
        blur:    float = 1.5,
        top_n:   int   = 10,
    ) -> list:
        """
        Return only the ranked hotspot residues without full scoring.
        Faster than score_interface() when you only need hotspot positions
        (e.g. for multi-target threat identification in Khukuri's PINCER).

        Returns list of dicts: [{resname, resnum, chain, weight, side}, ...]
        sorted by weight descending.
        """
        result = self.score_interface(
            pdb_a, pdb_b, chain_a, chain_b, cutoff, blur=blur
        )
        all_hotspots = (
            [dict(r, side='A') for r in result.hotspot_residues_a] +
            [dict(r, side='B') for r in result.hotspot_residues_b]
        )
        return sorted(all_hotspots, key=lambda r: r.get('weight', 0), reverse=True)[:top_n]

    @property
    def mode(self) -> str:
        return self._mode

    # -----------------------------------------------------------------------
    # Internal C-native implementation
    # -----------------------------------------------------------------------

    def _score_native(
        self, pdb_a, pdb_b, chain_a, chain_b,
        cutoff, delta_G, kT, blur, verbose
    ) -> PPIResult:
        lib = self._lib

        # --- Phase I: detect interface ---
        iface = PPIInterfaceStruct()
        n_pairs = lib.nibble_detect_interface(
            pdb_a.encode(),
            pdb_b.encode(),
            chain_a.encode()[0:1] or b'\x00',
            chain_b.encode()[0:1] or b'\x00',
            ctypes.c_float(cutoff),
            ctypes.byref(iface),
        )
        if n_pairs <= 0:
            return PPIResult(backend="C-Native-Error-Interface")

        # --- Phase II: load surfaces ---
        dim = self._PPI_DIM
        res = self._PPI_RES

        grid_a_ptr = lib.nibble_create_grid(dim, dim, dim, ctypes.c_float(res))
        grid_b_ptr = lib.nibble_create_grid(dim, dim, dim, ctypes.c_float(res))

        lib.nibble_load_ppi_surface(
            grid_a_ptr, pdb_a.encode(),
            ctypes.byref(iface), ctypes.c_int(1), ctypes.c_float(blur)
        )
        lib.nibble_load_ppi_surface(
            grid_b_ptr, pdb_b.encode(),
            ctypes.byref(iface), ctypes.c_int(0), ctypes.c_float(blur)
        )

        # --- Phase V: full PPI affinity (includes III and IV internally) ---
        score_struct = PPIScoreStruct()
        composite = lib.nibble_ppi_affinity(
            grid_a_ptr, grid_b_ptr,
            ctypes.byref(iface),
            ctypes.c_float(delta_G),
            ctypes.c_float(kT),
            ctypes.byref(score_struct),
        )

        if verbose:
            lib.nibble_ppi_print_interface(ctypes.byref(iface))
            lib.nibble_ppi_print_score(ctypes.byref(score_struct))

        # Extract hotspot residues
        hotspots_a = [
            {
                'resname': iface.residues_a[r].resname.decode().strip(),
                'resnum':  iface.residues_a[r].resnum,
                'chain':   iface.residues_a[r].chain.decode(),
                'weight':  iface.residues_a[r].hotspot_weight,
            }
            for r in range(iface.n_residues_a)
            if iface.residues_a[r].is_hotspot
        ]
        hotspots_b = [
            {
                'resname': iface.residues_b[r].resname.decode().strip(),
                'resnum':  iface.residues_b[r].resnum,
                'chain':   iface.residues_b[r].chain.decode(),
                'weight':  iface.residues_b[r].hotspot_weight,
            }
            for r in range(iface.n_residues_b)
            if iface.residues_b[r].is_hotspot
        ]

        result = PPIResult(
            composite_score=float(score_struct.composite_score),
            raw_affinity=float(score_struct.raw_affinity),
            sc_score=float(score_struct.sc_score),
            hotspot_score=float(score_struct.hotspot_score),
            water_bonus=float(score_struct.water_bonus),
            desolvation_penalty=float(score_struct.desolvation_penalty),
            n_hotspots=int(score_struct.n_hotspots),
            n_residues_a=int(iface.n_residues_a),
            n_residues_b=int(iface.n_residues_b),
            n_hbonds=int(iface.n_hbonds),
            n_hydrophobic=int(iface.n_hydrophobic),
            n_bridging_waters=int(iface.n_bridging_waters),
            buried_surface_area=float(iface.buried_surface_area),
            buried_area_a=float(score_struct.buried_area_a),
            buried_area_b=float(score_struct.buried_area_b),
            centre=(iface.centre_x, iface.centre_y, iface.centre_z),
            hotspot_residues_a=hotspots_a,
            hotspot_residues_b=hotspots_b,
            backend="C-Native",
        )

        # Free grids
        lib.nibble_free_grid(grid_a_ptr)
        lib.nibble_free_grid(grid_b_ptr)

        return result

    def _extract_pharmacophore_native(
        self, pdb_a, pdb_b, chain_a, chain_b,
        cutoff, blur, min_hotspot_weight
    ):
        lib = self._lib
        dim = self._PPI_DIM
        res = self._PPI_RES

        iface = PPIInterfaceStruct()
        n_pairs = lib.nibble_detect_interface(
            pdb_a.encode(), pdb_b.encode(),
            chain_a.encode()[0:1] or b'\x00',
            chain_b.encode()[0:1] or b'\x00',
            ctypes.c_float(cutoff),
            ctypes.byref(iface),
        )
        if n_pairs <= 0:
            return None

        grid_a_ptr = lib.nibble_create_grid(dim, dim, dim, ctypes.c_float(res))
        grid_b_ptr = lib.nibble_create_grid(dim, dim, dim, ctypes.c_float(res))
        inhib_ptr  = lib.nibble_create_grid(dim, dim, dim, ctypes.c_float(res))

        lib.nibble_load_ppi_surface(
            grid_a_ptr, pdb_a.encode(),
            ctypes.byref(iface), ctypes.c_int(1), ctypes.c_float(blur)
        )
        lib.nibble_load_ppi_surface(
            grid_b_ptr, pdb_b.encode(),
            ctypes.byref(iface), ctypes.c_int(0), ctypes.c_float(blur)
        )

        # Run hotspot scoring first (populates hotspot_weight in iface)
        lib.nibble_score_hotspots(grid_a_ptr, grid_b_ptr, ctypes.byref(iface))

        n_voxels = lib.nibble_load_ppi_inhibitor_site(
            grid_a_ptr, grid_b_ptr,
            ctypes.byref(iface),
            inhib_ptr,
            ctypes.c_float(min_hotspot_weight),
        )

        lib.nibble_free_grid(grid_a_ptr)
        lib.nibble_free_grid(grid_b_ptr)

        if n_voxels <= 0:
            lib.nibble_free_grid(inhib_ptr)
            return None

        # Wrap in NativeNibbleEngine for compatibility with existing docking code
        engine = NativeNibbleEngine.__new__(NativeNibbleEngine)
        engine._lib     = lib
        engine.grid_ptr = inhib_ptr
        engine.dim_x    = dim
        engine.dim_y    = dim
        engine.dim_z    = dim
        engine.resolution = res

        return engine


# ---------------------------------------------------------------------------
# Convenience factory — mirrors Khukuri conventions
# ---------------------------------------------------------------------------
def make_ppi_engine() -> NibblePPIEngine:
    """
    Factory function mirroring Khukuri conventions.

    Example:
    --------
    from nibble_ppi_bridge import make_ppi_engine

    ppi = make_ppi_engine()

    # Score a complex (two chains in same PDB)
    result = ppi.score_interface(
        "data/1CRN.pdb", "data/1CRN.pdb",
        chain_a='A', chain_b='B',
        verbose=True,
    )
    print(result)

    # Score separate PDB files
    result = ppi.score_interface("bcl2.pdb", "bax.pdb")
    print(f"BCL2-BAX interface score: {result.composite_score:.4f}")
    print(f"Sc={result.sc_score:.3f}, hotspots={result.n_hotspots}")

    # Get hotspot residues for PINCER multi-target threat
    hotspots = ppi.score_hotspots_only("bcl2.pdb", "bax.pdb", top_n=5)
    for h in hotspots:
        print(f"  {h['resname']}{h['resnum']} chain {h['chain']}  w={h['weight']:.3f}")

    # Extract pharmacophore for inhibitor design
    pocket = ppi.extract_inhibitor_pharmacophore("bcl2.pdb", "bax.pdb")
    if pocket:
        # pocket is a NibbleGrid — use with standard Nibble docking
        from nibble_bridge import NibbleMolecule
        mol = NibbleMolecule(pocket, smiles="CCO", ...)
        score, pose = mol.run_trajectory(n_steps=100)
        print(f"Inhibitor docking score: {score:.4f}")
    """
    return NibblePPIEngine()
