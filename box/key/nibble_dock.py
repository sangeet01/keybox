"""
box/key/nibble_dock.py
----------------------
Unified dispatcher for all Nibble molecular docking interaction classes.

Provides a single entry point — dock() — that routes to the correct
engine based on interaction type. Wraps:

  NibbleEngine          (existing)  — small molecule → protein pocket
  NibblePPIEngine       (new)       — protein → protein interface
  NibblePROTACEngine    (new)       — PROTAC → ternary complex
  NibblePeptideEngine   (new)       — peptide → protein groove/surface
  NibbleAntibodyEngine  (new)       — antibody CDR → antigen epitope

All engines share the same loaded C shared library via NibbleEngine._lib.
All fall back to NumPy-based scoring if the C library is unavailable.

Design philosophy:
  - One import, one function call, all interaction classes
  - Consistent result interface across all types
  - Progressive disclosure: simple calls for simple cases,
    full parameter access for advanced use
  - Mirrors NibbleEngine's existing orchestrator pattern exactly

Usage
-----
  from nibble_dock import dock, NibbleDock

  # Small molecule (existing behaviour, unchanged)
  result = dock("small_molecule",
      pdb      = "target.pdb",
      centre   = (12.3, 8.1, 22.4),
      smiles   = "CCO",
      n_steps  = 100,
  )

  # Protein-protein interface
  result = dock("ppi",
      pdb_a    = "protein_a.pdb",
      pdb_b    = "protein_b.pdb",
      chain_a  = 'A',
      chain_b  = 'B',
  )

  # PROTAC ternary complex
  result = dock("protac",
      target_pdb    = "brd4.pdb",
      e3_pdb        = "crbn.pdb",
      target_centre = (12.3, 8.1, 22.4),
      e3_centre     = (4.5, 31.2, 9.8),
      protac        = my_protac,
      run_trajectory= True,
  )

  # Peptide
  result = dock("peptide",
      pdb     = "mdm2.pdb",
      centre  = (13.2, 9.8, 21.4),
      sequence= "ETFSDLWKLLPEN",
      ss      = "HHHHHHHHHHHHH",
      staple  = (2, 6),
  )

  # Antibody
  result = dock("antibody",
      pdb        = "her2_ecd.pdb",
      centre     = (45.2, 31.8, 22.4),
      H3         = "ARWGGDYFDY",
      H1         = "GYTFTSYW",  H2 = "IYPGDGDT",
      L1         = "RASQSIS",   L2 = "AASSLQS",  L3 = "QQSYTTP",
  )

  # Or use the class directly for multi-run workflows
  nd = NibbleDock()
  for smiles in compound_library:
      r = nd.dock_small_molecule("target.pdb", centre, smiles)
      results.append(r)

Author: Khukuri / KeyBox project (sangeet01)
License: Apache 2.0 with Commons Clause
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from nibble_bridge import NibbleEngine, NibbleMolecule

from nibble_ppi_bridge import (
    NibblePPIEngine,
    make_ppi_engine,
    PPIResult,
)
from nibble_protac_bridge import (
    NibblePROTACEngine,
    make_protac_engine,
    PROTACMolecule,
    PROTACResult,
)
from nibble_peptide_bridge import (
    NibblePeptideEngine,
    make_peptide_engine,
    PeptideMolecule,
    PeptideDockingResult,
)
from nibble_antibody_bridge import (
    NibbleAntibodyEngine,
    make_antibody_engine,
    AntibodyMolecule,
    AntibodyDockingResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interaction type enum
# ---------------------------------------------------------------------------

class InteractionType(str, Enum):
    SMALL_MOLECULE = "small_molecule"
    PPI            = "ppi"
    PROTAC         = "protac"
    PEPTIDE        = "peptide"
    ANTIBODY       = "antibody"

    @classmethod
    def from_string(cls, s: str) -> "InteractionType":
        s = s.lower().strip().replace("-", "_").replace(" ", "_")
        aliases = {
            "mol":            cls.SMALL_MOLECULE,
            "molecule":       cls.SMALL_MOLECULE,
            "small_mol":      cls.SMALL_MOLECULE,
            "ligand":         cls.SMALL_MOLECULE,
            "drug":           cls.SMALL_MOLECULE,
            "protein_protein":cls.PPI,
            "protein-protein":cls.PPI,
            "ppi":            cls.PPI,
            "protac":         cls.PROTAC,
            "ternary":        cls.PROTAC,
            "bifunctional":   cls.PROTAC,
            "peptide":        cls.PEPTIDE,
            "pep":            cls.PEPTIDE,
            "stapled":        cls.PEPTIDE,
            "cyclic_peptide": cls.PEPTIDE,
            "antibody":       cls.ANTIBODY,
            "ab":             cls.ANTIBODY,
            "mab":            cls.ANTIBODY,
            "fab":            cls.ANTIBODY,
            "fv":             cls.ANTIBODY,
            "nanobody":       cls.ANTIBODY,
        }
        if s in aliases:
            return aliases[s]
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(
            f"Unknown interaction type '{s}'. "
            f"Valid: {[m.value for m in cls]}"
        )


# ---------------------------------------------------------------------------
# Unified result wrapper
# ---------------------------------------------------------------------------

@dataclass
class DockingResult:
    """
    Unified result container for all interaction types.

    The raw result from the specific engine is preserved in `.raw`
    for full access to type-specific fields. The unified fields
    provide cross-type comparison.
    """
    interaction_type: str
    composite_score:  float
    backend:          str
    elapsed_s:        float

    # Unified cross-type fields
    binding_class:    str   = "unknown"
    n_hbonds:         float = 0.0
    buried_area:      float = 0.0
    selectivity_flag: bool  = False

    # Raw engine result (type-specific)
    raw: Any = None

    def __str__(self) -> str:
        lines = [
            f"─── DockingResult [{self.interaction_type.upper()}] ──────────",
            f"  Composite score: {self.composite_score:.4f}",
            f"  Binding class:   {self.binding_class}",
            f"  Backend:         {self.backend}",
            f"  Elapsed:         {self.elapsed_s:.3f}s",
        ]
        if self.raw is not None:
            lines.append(f"─── Detail ──────────────────────────────────")
            lines.append(str(self.raw))
        return "\n".join(lines)

    @property
    def is_productive(self) -> bool:
        return self.composite_score > 0.45

    def to_dict(self) -> dict:
        return {
            "interaction_type": self.interaction_type,
            "composite_score":  self.composite_score,
            "binding_class":    self.binding_class,
            "backend":          self.backend,
            "elapsed_s":        self.elapsed_s,
            "buried_area":      self.buried_area,
            "n_hbonds":         self.n_hbonds,
            "selectivity_flag": self.selectivity_flag,
        }


# ---------------------------------------------------------------------------
# NibbleDock — main unified class
# ---------------------------------------------------------------------------

class NibbleDock:
    """
    Unified docking engine dispatcher.

    Lazily instantiates each sub-engine on first use.
    All engines share the same loaded C library via NibbleEngine._lib.

    Thread safety: each call creates its own result objects.
    The engines themselves are stateless between calls.
    """

    def __init__(self) -> None:
        # Engines instantiated lazily
        self._small_mol_engine: Optional[NibbleEngine] = None
        self._ppi_engine:       Optional[NibblePPIEngine]     = None
        self._protac_engine:    Optional[NibblePROTACEngine]  = None
        self._peptide_engine:   Optional[NibblePeptideEngine] = None
        self._antibody_engine:  Optional[NibbleAntibodyEngine]= None

    # -----------------------------------------------------------------------
    # Sub-engine accessors (lazy init)
    # -----------------------------------------------------------------------

    @property
    def small_mol(self) -> NibbleEngine:
        if self._small_mol_engine is None:
            self._small_mol_engine = NibbleEngine()
        return self._small_mol_engine

    @property
    def ppi(self) -> NibblePPIEngine:
        if self._ppi_engine is None:
            self._ppi_engine = make_ppi_engine()
        return self._ppi_engine

    @property
    def protac(self) -> NibblePROTACEngine:
        if self._protac_engine is None:
            self._protac_engine = make_protac_engine()
        return self._protac_engine

    @property
    def peptide(self) -> NibblePeptideEngine:
        if self._peptide_engine is None:
            self._peptide_engine = make_peptide_engine()
        return self._peptide_engine

    @property
    def antibody(self) -> NibbleAntibodyEngine:
        if self._antibody_engine is None:
            self._antibody_engine = make_antibody_engine()
        return self._antibody_engine

    # -----------------------------------------------------------------------
    # Typed docking methods
    # -----------------------------------------------------------------------

    def dock_small_molecule(
        self,
        pdb:           str,
        centre:        tuple,
        smiles:        str         = None,
        coords:        list        = None,
        ch_vals:       list        = None,
        n_steps:       int         = 100,
        pocket_radius: float       = 10.0,
        blur:          float       = 1.5,
        dt:            float       = 0.002,
        gamma:         float       = 1.0,
        kT:            float       = 0.593,
        verbose:       bool        = False,
    ) -> DockingResult:
        """
        Dock a small molecule against a protein pocket.
        Wraps the existing NibbleEngine / NibbleMolecule workflow.
        """
        t0 = time.time()

        engine = self.small_mol
        engine.load_pdb_pocket(pdb, centre, pocket_radius, blur)

        if smiles or (coords and ch_vals):
            mol = NibbleMolecule(
                engine,
                smiles=smiles,
                coords=coords,
                ch_vals=ch_vals,
                centre_coords=centre,
            )
            score, pose = mol.run_trajectory(n_steps, dt, gamma, kT)
        else:
            score = 0.0
            pose  = centre

        raw_score = float(score)
        norm = min(1.0, max(0.0, raw_score / 2000.0))

        return DockingResult(
            interaction_type="small_molecule",
            composite_score=norm,
            binding_class=(
                "Strong" if norm >= 0.65 else
                "Moderate" if norm >= 0.45 else
                "Weak"
            ),
            backend=engine.mode,
            elapsed_s=time.time() - t0,
            buried_area=0.0,
            n_hbonds=0.0,
            raw={"score": raw_score, "pose": pose},
        )

    def dock_ppi(
        self,
        pdb_a:          str,
        pdb_b:          str,
        chain_a:        str   = '',
        chain_b:        str   = '',
        cutoff:         float = 6.0,
        delta_G:        float = 2.0,
        kT:             float = 0.593,
        verbose:        bool  = False,
    ) -> DockingResult:
        """Score a protein-protein interface."""
        t0 = time.time()
        result: PPIResult = self.ppi.score_interface(
            pdb_a, pdb_b, chain_a, chain_b, cutoff, delta_G, kT, verbose=verbose
        )
        return DockingResult(
            interaction_type="ppi",
            composite_score=result.composite_score,
            binding_class=(
                "Strong (Sc≥0.70)" if result.sc_score >= 0.70 else
                "Moderate"         if result.sc_score >= 0.50 else
                "Weak"
            ),
            backend=result.backend,
            elapsed_s=time.time() - t0,
            buried_area=result.buried_surface_area,
            n_hbonds=float(result.n_hbonds),
            raw=result,
        )

    def dock_protac(
        self,
        target_pdb:     str,
        e3_pdb:         str,
        target_centre:  tuple,
        e3_centre:      tuple,
        protac:         PROTACMolecule,
        run_trajectory: bool  = True,
        n_steps:        int   = 300,
        pocket_radius:  float = 12.0,
        kT:             float = 0.593,
        verbose:        bool  = False,
    ) -> DockingResult:
        """Score a PROTAC ternary complex."""
        t0 = time.time()
        result: PROTACResult = self.protac.score_ternary(
            target_pdb, e3_pdb, target_centre, e3_centre,
            protac, run_trajectory, n_steps, pocket_radius, kT=kT, verbose=verbose,
        )
        return DockingResult(
            interaction_type="protac",
            composite_score=result.composite_score,
            binding_class=result.degradation_class,
            backend=result.backend,
            elapsed_s=time.time() - t0,
            buried_area=0.0,
            n_hbonds=0.0,
            selectivity_flag=result.hook_effect,
            raw=result,
        )

    def dock_peptide(
        self,
        pdb:            str,
        centre:         tuple,
        sequence:       str,
        ss:             str   = None,
        is_cyclic:      bool  = False,
        staple:         tuple = None,
        disulfide:      tuple = None,
        lactam:         tuple = None,
        n_steps:        int   = 500,
        pocket_radius:  float = 20.0,
        kT:             float = 0.593,
        verbose:        bool  = False,
    ) -> DockingResult:
        """Dock a peptide against a protein surface."""
        t0 = time.time()
        pep = PeptideMolecule(
            sequence=sequence, ss_string=ss,
            is_cyclic=is_cyclic, staple=staple,
            disulfide=disulfide, lactam=lactam,
        )
        result: PeptideDockingResult = self.peptide.dock(
            pdb, centre, pep, n_steps,
            pocket_radius=pocket_radius, kT=kT, verbose=verbose,
        )
        return DockingResult(
            interaction_type="peptide",
            composite_score=result.composite_score,
            binding_class=result.binding_class,
            backend=result.backend,
            elapsed_s=time.time() - t0,
            n_hbonds=result.n_hbonds_formed,
            raw=result,
        )

    def dock_antibody(
        self,
        pdb:            str,
        centre:         tuple,
        antibody:       Union[AntibodyMolecule, str] = None,
        ab_pdb:         str   = None,
        vh_chain:       str   = 'H',
        vl_chain:       str   = 'L',
        L1: str="", L2: str="", L3: str="",
        H1: str="", H2: str="", H3: str="",
        n_steps:        int   = 300,
        pocket_radius:  float = 25.0,
        kT:             float = 0.593,
        verbose:        bool  = False,
    ) -> DockingResult:
        """
        Dock an antibody against an antigen surface.

        Accepts either:
          - A pre-built AntibodyMolecule object via `antibody` parameter
          - A PDB path via `ab_pdb` + `vh_chain`/`vl_chain`
          - CDR sequences directly via L1/L2/L3/H1/H2/H3 kwargs
        """
        t0 = time.time()

        # Resolve antibody input
        if isinstance(antibody, AntibodyMolecule):
            ab = antibody
        elif ab_pdb:
            ab = AntibodyMolecule.from_pdb(ab_pdb, vh_chain, vl_chain)
        else:
            ab = AntibodyMolecule.from_sequences(
                L1=L1, L2=L2, L3=L3,
                H1=H1, H2=H2, H3=H3,
            )

        result: AntibodyDockingResult = self.antibody.dock(
            pdb, centre, ab, n_steps,
            pocket_radius=pocket_radius, kT=kT, verbose=verbose,
        )
        return DockingResult(
            interaction_type="antibody",
            composite_score=result.composite_score,
            binding_class=result.binding_class,
            backend=result.backend,
            elapsed_s=time.time() - t0,
            buried_area=result.buried_surface_area,
            raw=result,
        )

    # -----------------------------------------------------------------------
    # Status and diagnostics
    # -----------------------------------------------------------------------

    def status(self) -> dict:
        """
        Return backend status for all engines.
        Instantiates all engines to check library loading.
        """
        return {
            "small_molecule": self.small_mol.mode,
            "ppi":            self.ppi.mode,
            "protac":         self.protac.mode,
            "peptide":        self.peptide.mode,
            "antibody":       self.antibody.mode,
            "shared_library": "loaded" if NibbleEngine._lib else "not loaded",
        }

    def benchmark(
        self,
        pdb: str,
        centre: tuple = (10.0, 10.0, 10.0),
        n_steps: int = 50,
    ) -> dict:
        """
        Run a quick benchmark across all interaction types.
        Returns timing in seconds per interaction type.
        Uses the provided PDB as both target and query (self-docking).
        """
        timings = {}

        # Small molecule — simple uniform channel molecule
        import numpy as np
        n_atoms = 20
        coords  = np.random.randn(n_atoms, 3).flatten().tolist()
        ch_vals = [[0.5] * 16] * n_atoms
        ch_flat = [v for row in ch_vals for v in row]
        r = self.dock_small_molecule(pdb, centre, coords=coords,
                                     ch_vals=ch_flat, n_steps=n_steps)
        timings["small_molecule"] = r.elapsed_s

        # PPI — same PDB vs itself
        r = self.dock_ppi(pdb, pdb)
        timings["ppi"] = r.elapsed_s

        # PROTAC — synthetic test
        n_tot = 37
        pr_coords = np.zeros(n_tot * 3).tolist()
        pr_ch     = [0.5] * (n_tot * 16)
        ec = (centre[0] + 12.0, centre[1], centre[2])
        pr = PROTACMolecule(pr_coords, pr_ch, split_a=15, split_b=22)
        r = self.dock_protac(pdb, pdb, centre, ec, pr,
                             run_trajectory=False)
        timings["protac"] = r.elapsed_s

        # Peptide
        r = self.dock_peptide(pdb, centre, "ACDEFGHIKL", n_steps=n_steps)
        timings["peptide"] = r.elapsed_s

        # Antibody
        r = self.dock_antibody(pdb, centre,
                               H3="ARWGGDYFDY", H1="GYTFTS",
                               L1="RASQSIS", n_steps=n_steps)
        timings["antibody"] = r.elapsed_s

        timings["total"] = sum(timings.values())
        return timings


# ---------------------------------------------------------------------------
# Module-level dock() function — the primary public API
# ---------------------------------------------------------------------------

_default_dock = None


def dock(interaction_type: str, **kwargs) -> DockingResult:
    """
    Unified docking function. Routes to the correct engine.

    Args:
        interaction_type: One of "small_molecule", "ppi", "protac",
                          "peptide", "antibody" (and aliases).
        **kwargs:         Arguments for the specific interaction type.
                          See NibbleDock.dock_* methods for full signatures.

    Returns:
        DockingResult with composite_score, binding_class, backend,
        elapsed_s, and raw engine-specific result.

    Examples
    --------
    from nibble_dock import dock

    # Small molecule
    r = dock("small_molecule", pdb="brd4.pdb", centre=(12.3, 8.1, 22.4),
             smiles="c1ccc(cc1)N", n_steps=100)
    print(f"Score: {r.composite_score:.4f}")

    # PPI
    r = dock("ppi", pdb_a="bcl2.pdb", pdb_b="bax.pdb",
             chain_a='A', chain_b='B')
    print(f"Sc: {r.raw.sc_score:.4f}  hotspots: {r.raw.n_hotspots}")

    # PROTAC (ARV-110 / BRD4-CRBN)
    protac = PROTACMolecule(coords, ch_vals, split_a=18, split_b=28)
    r = dock("protac",
             target_pdb="brd4.pdb", e3_pdb="crbn.pdb",
             target_centre=(12.3, 8.1, 22.4), e3_centre=(4.5, 31.2, 9.8),
             protac=protac, run_trajectory=True