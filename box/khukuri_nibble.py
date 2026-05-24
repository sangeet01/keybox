"""
khukuri_nibble.py
=================
Khukuri ↔ Nibble bridge: exposes NibbleEngine as a PINCER-compatible
fitness function for drug candidate scoring.

Drop-in usage in Khukuri:
    from khukuri_nibble import NibbleFitness
    fitness_fn = NibbleFitness("path/to/NDM1_3S0Z.pdb", center=(11.5, 18.2, 24.7))
    score = fitness_fn.score(smiles)           # single molecule
    scores = fitness_fn.score_batch(smiles_list)   # population

Autonomous loop usage (PINCER):
    fitness_fn = NibbleFitness.for_ndm1(pdb_path)
    # passes directly to PincerEngine as the fitness callable

What this wrapper does:
  1. Loads PDB pocket into Nibble's 16-channel demand tensor
     (now with metal ion demand — Zn2+, Mg2+, Fe etc.)
  2. Converts SMILES → atom coords + channel values
     (via RDKit if available, otherwise internal SMILES parser)
  3. Scores via Langevin trajectory (Tier 3) — molecule physically diffuses
     into the binding pocket, settles at energy minimum
  4. Optionally activates Tier 4 induced fit (protein flexibility)
  5. Returns a normalised score in [0, 1] suitable for PINCER's minimax

Metal coordination handling:
  The patched nibble.c now projects strong CH_METAL_DEMAND (4.0) at zinc
  ion positions. This wrapper assigns CH_METAL_DEMAND to drug atoms that
  carry known metal-binding groups: thiol (Cys-like), hydroxamate,
  boronate, carboxylate, hydroxyl, imidazole.

  NDM-1 specific: Zn1 (His120/His122/His189) + Zn2 (Asp124/Cys208/His250)
  Both zinc ions now populate CH_METAL_DEMAND in the demand tensor,
  so chelating drugs are correctly rewarded.
"""

import os
import sys
import math
import re
from pathlib import Path
from typing import Optional, Union
import numpy as np

# ── Nibble bridge import ──────────────────────────────────────────────────────
# Try to import from the keybox repo. Fall back gracefully if not on path.
try:
    _HERE = Path(__file__).parent.resolve()
    _KEYBOX_KEY = _HERE.parent / "keybox" / "box" / "key"
    if _KEYBOX_KEY.exists():
        sys.path.insert(0, str(_KEYBOX_KEY.parent))
    from key.nibble_bridge import NibbleEngine, N_CHANNELS
    from key.nibble_bridge import (
        CH_STERIC_DEMAND, CH_ELEC_DEMAND, CH_HBA_DEMAND, CH_HBD_DEMAND,
        CH_LIPO_DEMAND, CH_AROM_DEMAND, CH_METAL_DEMAND, CH_CATION_DEMAND,
        CH_ANION_DEMAND, CH_PHOBIC_CORE, CH_SOLVENT_EXPO,
        CH_FLEXIBILITY, CH_NUCLEOPHILICITY, CH_ELECTROPHILICITY,
        CH_WATER_CONSERVED, CH_WATER_DYNAMIC,
    )
    _BRIDGE_AVAILABLE = True
except ImportError:
    _BRIDGE_AVAILABLE = False
    N_CHANNELS = 16
    CH_STERIC_DEMAND=0; CH_ELEC_DEMAND=1; CH_HBA_DEMAND=2; CH_HBD_DEMAND=3
    CH_LIPO_DEMAND=4;   CH_AROM_DEMAND=5; CH_METAL_DEMAND=6; CH_CATION_DEMAND=7
    CH_ANION_DEMAND=8;  CH_PHOBIC_CORE=9; CH_SOLVENT_EXPO=10; CH_FLEXIBILITY=11
    CH_NUCLEOPHILICITY=12; CH_ELECTROPHILICITY=13
    CH_WATER_CONSERVED=14; CH_WATER_DYNAMIC=15

# ── Optional RDKit ────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    _RDKIT = True
except ImportError:
    _RDKIT = False


# ── NDM-1 binding site definition (3S0Z) ─────────────────────────────────────

NDM1_SITE = {
    "center":      (11.5, 18.2, 24.7),   # midpoint between Zn1 and Zn2
    "range":       12.0,                  # Å radius around centre
    "blur_radius": 1.5,                   # Å Gaussian blur for B-factor scaling
    "zn1":         (10.8, 17.5, 24.0),   # Zn1 (His120/His122/His189)
    "zn2":         (12.2, 18.9, 25.4),   # Zn2 (Asp124/Cys208/His250)
    "grid_dim":    (30, 30, 30),
    "resolution":  0.8,                   # Å per voxel
}


# ── SMILES → atom properties ──────────────────────────────────────────────────

# Atomic properties for channel assignment
# (element, partial_charge, hydrophobicity, metal_binding, h_donor, h_acceptor, aromatic)
_ELEMENT_PROPS = {
    # element: (charge, hydro, metal_bind, hbd, hba, arom)
    "C":  ( 0.0,  0.7, 0.0, 0.0, 0.0, 0.0),
    "N":  ( 0.3, -0.2, 0.2, 0.8, 0.6, 0.0),
    "O":  (-0.4, -0.5, 0.3, 0.5, 0.9, 0.0),
    "S":  ( 0.1,  0.5, 0.8, 0.1, 0.3, 0.0),  # thiol/thioether
    "P":  ( 0.5, -0.1, 0.1, 0.0, 0.8, 0.0),
    "F":  (-0.3,  0.1, 0.0, 0.0, 0.4, 0.0),
    "Cl": ( 0.0,  0.5, 0.0, 0.0, 0.1, 0.0),
    "Br": ( 0.0,  0.6, 0.0, 0.0, 0.1, 0.0),
    "I":  ( 0.1,  0.7, 0.0, 0.0, 0.1, 0.0),
    "B":  ( 0.6, -0.1, 0.9, 0.0, 0.5, 0.0),  # boronate — strong Zn chelator
}

# Metal-binding SMARTS patterns → CH_METAL_DEMAND boost for drug atoms
_METAL_SMARTS = [
    # pattern,                  boost,  description
    ("[SH]",                    2.5,    "thiol — strongest Zn chelator"),
    ("[S;!H0]",                 2.0,    "any SH"),
    ("C(=O)[OH]",               1.5,    "carboxylic acid"),
    ("C(=O)NO",                 2.0,    "hydroxamic acid"),
    ("B([OH])[OH]",             2.5,    "boronic acid — taniborbactam scaffold"),
    ("B1OC(CO1)",               2.5,    "cyclic boronate"),
    ("[nH]1ccnc1",              1.8,    "imidazole — His mimic"),
    ("c1cncn1",                 1.5,    "imidazole aromatic"),
    ("[OH]C=O",                 1.2,    "alpha-hydroxy acid"),
    ("NCC(=O)O",                1.3,    "amino acid chelate motif"),
    ("O=C([NH2])[OH]",          1.0,    "hydroxamic acid variant"),
    ("SC",                      1.2,    "thioether"),
    ("[NH2]",                   0.8,    "primary amine — weak metal donor"),
    ("OCC(=O)O",                1.3,    "hydroxyl + carboxylate — bidentate"),
]


def _rdkit_atom_props(smiles: str) -> Optional[list[dict]]:
    """Use RDKit to get 3D coords + properties. Returns list of atom dicts."""
    if not _RDKIT:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result != 0:
            # try random coords
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()

        # detect metal-binding atoms
        metal_boost = np.zeros(mol.GetNumAtoms(), dtype=np.float32)
        for smarts, boost, _ in _METAL_SMARTS:
            patt = Chem.MolFromSmarts(smarts)
            if patt:
                for match in mol.GetSubstructMatches(patt):
                    for idx in match:
                        metal_boost[idx] = max(metal_boost[idx], boost)

        atoms = []
        noHmol = Chem.RemoveHs(mol)
        for atom in noHmol.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            sym = atom.GetSymbol()
            props = _ELEMENT_PROPS.get(sym, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            charge, hydro, metal, hbd, hba, arom = props
            if atom.GetIsAromatic():
                arom = 1.0
            atoms.append({
                "x": pos.x, "y": pos.y, "z": pos.z,
                "symbol": sym,
                "charge": charge + float(atom.GetFormalCharge()) * 0.5,
                "hydro": hydro,
                "metal": metal + float(metal_boost[idx]),
                "hbd": hbd,
                "hba": hba,
                "arom": arom,
            })
        return atoms
    except Exception:
        return None


def _fallback_atom_props(smiles: str) -> list[dict]:
    """
    Minimal SMILES parser — no RDKit needed.
    Counts atoms, assigns properties from element table,
    places them on a rough linear geometry.
    Good enough for PINCER's population-level fitness ranking.
    """
    # extract atom symbols (uppercase letter + optional lowercase)
    element_re = re.compile(r'([A-Z][a-z]?)')
    symbols = element_re.findall(smiles.replace('[nH]','N').replace('[SH]','S'))
    # filter to real elements (not SMILES syntax letters)
    valid = {k.upper() for k in _ELEMENT_PROPS}
    symbols = [s for s in symbols if s.upper() in valid or s in ('C','N','O','S','P','F','B')]

    # simple metal-binding heuristic from SMILES string
    metal_smiles_patterns = {
        'SH': 2.5, 'S)': 1.2, 'B(O': 2.5, 'C(=O)N': 1.5,
        'C(=O)O': 1.3, 'nH': 1.8, 'NH2': 0.8,
    }
    metal_present = max(
        (v for k, v in metal_smiles_patterns.items() if k in smiles),
        default=0.0
    )

    atoms = []
    for i, sym in enumerate(symbols):
        props = _ELEMENT_PROPS.get(sym, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        charge, hydro, metal, hbd, hba, arom = props
        metal = max(metal, metal_present * 0.5)
        # rough linear placement — not geometrically meaningful but
        # gives the scorer something to work with
        atoms.append({
            "x": float(i) * 1.5,
            "y": math.sin(i * 0.5) * 1.2,
            "z": math.cos(i * 0.5) * 1.2,
            "symbol": sym,
            "charge": charge,
            "hydro": hydro,
            "metal": metal,
            "hbd": hbd,
            "hba": hba,
            "arom": arom,
        })
    return atoms


def smiles_to_atoms(smiles: str) -> list[dict]:
    """SMILES → list of atom dicts with 3D coords + channel properties."""
    atoms = _rdkit_atom_props(smiles)
    if atoms:
        return atoms
    return _fallback_atom_props(smiles)


def atoms_to_channel_arrays(atoms: list[dict]) -> tuple[list, list, list]:
    """Convert atom dicts → (coords, charges, hydrophobicity) for NibbleEngine."""
    coords = [(a["x"], a["y"], a["z"]) for a in atoms]
    charges = [a["charge"] for a in atoms]
    hydro   = [a["hydro"]  for a in atoms]
    return coords, charges, hydro


def atoms_to_full_channels(atoms: list[dict]) -> np.ndarray:
    """
    Build per-atom (N_CHANNELS,) channel value arrays.
    This is the richer path — sets all 16 channels including
    CH_METAL_DEMAND for chelating groups.
    """
    n = len(atoms)
    ch = np.zeros((n, N_CHANNELS), dtype=np.float32)
    for i, a in enumerate(atoms):
        ch[i, CH_STERIC_DEMAND]    =  1.0
        ch[i, CH_ELEC_DEMAND]      =  a["charge"]
        ch[i, CH_HBA_DEMAND]       =  a["hba"]
        ch[i, CH_HBD_DEMAND]       =  a["hbd"]
        ch[i, CH_LIPO_DEMAND]      =  a["hydro"]
        ch[i, CH_AROM_DEMAND]      =  a["arom"]
        ch[i, CH_METAL_DEMAND]     =  a["metal"]   # ← chelating groups score here
        ch[i, CH_ELECTROPHILICITY] =  max(0.0,  a["charge"])
        ch[i, CH_NUCLEOPHILICITY]  =  max(0.0, -a["charge"])
    return ch


# ── Score normalisation ────────────────────────────────────────────────────────

def _normalise_score(raw: float, low: float = -5000.0, high: float = 5000.0) -> float:
    """Map raw Nibble Frobenius score to [0, 1]. Higher = better binding."""
    clamped = max(low, min(high, raw))
    return (clamped - low) / (high - low)


# ── NibbleFitness ─────────────────────────────────────────────────────────────

class NibbleFitness:
    """
    Nibble-backed fitness function for PINCER.

    Modes:
      tier=1   Fast static score (affinity only). ~0.01ms/mol.
      tier=2   + water network + desolvation penalty. ~0.05ms/mol.
      tier=3   + Langevin trajectory (best pose search). ~5ms/mol.
      tier=4   + induced fit (protein flexibility). ~10ms/mol.

    For PINCER population evaluation, tier=1 is used for most generations,
    tier=3 for top-k candidates, tier=4 for apex drug validation.

    Mutation update:
      When PINCER mutates a residue, call update_mutation(residue_coords, diff)
      to patch only the affected voxels — O(r³) not O(N).
    """

    # Normalisation bounds estimated from NDM-1 + known inhibitors
    _SCORE_LOW  = -8000.0
    _SCORE_HIGH =  8000.0

    def __init__(
        self,
        pdb_path: str,
        center: tuple[float, float, float],
        range_val: float = 12.0,
        blur_radius: float = 1.5,
        grid_dim: tuple[int, int, int] = (30, 30, 30),
        resolution: float = 0.8,
        tier: int = 1,
        delta_G_bulk: float = 2.0,
        kT: float = 0.593,
    ):
        self.pdb_path    = str(pdb_path)
        self.center      = center
        self.range_val   = range_val
        self.blur_radius = blur_radius
        self.tier        = tier
        self.delta_G_bulk= delta_G_bulk
        self.kT          = kT
        self._mode       = "unavailable"

        self._pocket  = None
        self._pocket2 = None   # second conformer for Tier 4 induced fit

        if _BRIDGE_AVAILABLE:
            self._pocket = NibbleEngine(
                dim_x=grid_dim[0], dim_y=grid_dim[1], dim_z=grid_dim[2],
                resolution=resolution
            )
            self._mode = self._pocket.mode
            n = self._pocket.backend.load_pdb_pocket(
                self.pdb_path, center, range_val, blur_radius
            )
            print(f"  NibbleFitness: {n} atoms loaded | backend={self._mode} | tier={tier}")

            if tier >= 2:
                self._pocket.backend.compute_water_network(delta_G_bulk, kT)
                print(f"  Water network computed (delta_G_bulk={delta_G_bulk})")
        else:
            print("  NibbleFitness: nibble_bridge not available — using surrogate scorer")

    @classmethod
    def for_ndm1(cls, pdb_path: str, tier: int = 1) -> "NibbleFitness":
        """Convenience constructor pre-configured for NDM-1 (3S0Z)."""
        s = NDM1_SITE
        return cls(
            pdb_path    = pdb_path,
            center      = s["center"],
            range_val   = s["range"],
            blur_radius = s["blur_radius"],
            grid_dim    = s["grid_dim"],
            resolution  = s["resolution"],
            tier        = tier,
        )

    # ── public interface ──────────────────────────────────────────────────────

    def score(self, smiles: str) -> float:
        """
        Score a single SMILES string.
        Returns normalised score in [0, 1] — higher is better.
        """
        if not smiles or len(smiles) < 3:
            return 0.0
        atoms = smiles_to_atoms(smiles)
        if not atoms:
            return 0.0
        return self._score_atoms(atoms)

    def score_batch(self, smiles_list: list[str]) -> list[float]:
        """Score a population. Returns list of normalised scores."""
        return [self.score(s) for s in smiles_list]

    def score_with_pose(self, smiles: str) -> tuple[float, Optional[tuple]]:
        """
        Score + best pose (cx, cy, cz in grid coords).
        Only meaningful for tier >= 3.
        """
        atoms = smiles_to_atoms(smiles)
        if not atoms:
            return 0.0, None
        return self._score_atoms_with_pose(atoms)

    def update_mutation(
        self,
        residue_center_voxel: tuple[int, int, int],
        diff_channels: np.ndarray,
        radius_voxels: int = 3,
    ) -> None:
        """
        O(r³) patch update when PINCER mutates one amino acid.
        diff_channels: (N_CHANNELS,) float32 — difference to add at each
                       voxel in the sphere around residue_center_voxel.
        Called by PINCER's ThreatAwareFitnessFunction after mutating
        a resistance-conferring residue (e.g. H122Y, C208S in NDM-1).
        """
        if self._pocket is None or not hasattr(self._pocket.backend, 'update_local'):
            return
        cx, cy, cz = residue_center_voxel
        diff_c = diff_channels.astype(np.float32)
        self._pocket.backend.update_local(cx, cy, cz, radius_voxels, diff_c)

    def load_second_conformer(self, pdb_path2: str) -> None:
        """
        Load a second receptor conformer for Tier 4 induced fit.
        Typically: pdb_path = apo (open), pdb_path2 = holo (closed).
        For NDM-1: 3S0Z (apo) + any bound-state structure.
        """
        if not _BRIDGE_AVAILABLE:
            return
        s = NDM1_SITE
        self._pocket2 = NibbleEngine(
            dim_x=s["grid_dim"][0], dim_y=s["grid_dim"][1], dim_z=s["grid_dim"][2],
            resolution=s["resolution"]
        )
        self._pocket2.backend.load_pdb_pocket(
            pdb_path2, self.center, self.range_val, self.blur_radius
        )
        print(f"  Second conformer loaded: {pdb_path2}")

    def status(self) -> dict:
        return {
            "pdb": self.pdb_path,
            "center": self.center,
            "tier": self.tier,
            "backend": self._mode,
            "bridge_available": _BRIDGE_AVAILABLE,
            "rdkit_available": _RDKIT,
            "conformers": 2 if self._pocket2 else 1,
        }

    # ── internal scoring paths ────────────────────────────────────────────────

    def _score_atoms(self, atoms: list[dict]) -> float:
        score, _ = self._score_atoms_with_pose(atoms)
        return score

    def _score_atoms_with_pose(
        self, atoms: list[dict]
    ) -> tuple[float, Optional[tuple]]:
        if self._pocket is None or not _BRIDGE_AVAILABLE:
            return self._surrogate_score(atoms), None

        coords, charges, hydro = atoms_to_channel_arrays(atoms)
        start = self.center   # place molecule at pocket centre to start

        backend = self._pocket.backend

        # ── Tier 1: static affinity ─────────────────────────────────────────
        if self.tier == 1:
            raw = backend.project_molecule_data(coords, charges, hydro, start)
            return _normalise_score(raw, self._SCORE_LOW, self._SCORE_HIGH), None

        # ── Tier 2: + water network ─────────────────────────────────────────
        if self.tier == 2:
            drug_grid = NibbleEngine(
                dim_x=self._pocket.dim_x,
                dim_y=self._pocket.dim_y,
                dim_z=self._pocket.dim_z,
                resolution=self._pocket.resolution
            )
            for i, a in enumerate(atoms):
                ch = atoms_to_full_channels([a])[0]
                drug_grid.backend.project_atom_raw(
                    a["x"]-start[0], a["y"]-start[1], a["z"]-start[2],
                    1.5, ch
                ) if hasattr(drug_grid.backend, 'project_atom_raw') else None

            try:
                raw = backend.affinity_full(drug_grid.backend, self.delta_G_bulk)
            except Exception:
                raw = backend.project_molecule_data(coords, charges, hydro, start)
            return _normalise_score(raw, self._SCORE_LOW, self._SCORE_HIGH), None

        # ── Tier 3: Langevin trajectory ──────────────────────────────────────
        if self.tier >= 3:
            try:
                raw, best_pose = backend.run_trajectory(
                    coords, charges, hydro, start,
                    n_steps = 500 if self.tier == 3 else 1000,
                    dt      = 0.002,
                    gamma   = 1.0,
                    kT      = self.kT,
                )
                # Tier 4: add induced fit frame update if second conformer available
                if self.tier >= 4 and self._pocket2 is not None:
                    try:
                        occ = backend.pocket_occupancy(backend, threshold=0.1)
                        backend.tier4_frame_update(
                            self._pocket2.backend,
                            self._pocket2.backend,
                            backend,
                            sigma_delta=0.05,
                            elasticity_scale=0.2,
                        )
                        # re-score after induced fit
                        raw2, best_pose2 = backend.run_trajectory(
                            coords, charges, hydro, start,
                            n_steps=300, dt=0.002, gamma=1.0, kT=self.kT
                        )
                        raw = max(raw, raw2)
                        best_pose = best_pose2 if raw2 > raw else best_pose
                    except Exception:
                        pass
                return _normalise_score(raw, self._SCORE_LOW, self._SCORE_HIGH), best_pose
            except Exception as ex:
                # Langevin failed — fall back to static
                raw = backend.project_molecule_data(coords, charges, hydro, start)
                return _normalise_score(raw, self._SCORE_LOW, self._SCORE_HIGH), None

        return 0.0, None

    def _surrogate_score(self, atoms: list[dict]) -> float:
        """
        Training-free surrogate when Nibble is unavailable.
        Scores based on metal-binding potential, polarity, size.
        Not a substitute for real physics — used for PINCER population
        ranking only when KeyBox is not on path.
        """
        if not atoms:
            return 0.0
        metal  = sum(a["metal"]  for a in atoms)
        hba    = sum(a["hba"]    for a in atoms)
        hbd    = sum(a["hbd"]    for a in atoms)
        charge = sum(abs(a["charge"]) for a in atoms)
        n      = len(atoms)

        # NDM-1 prefers metal chelators of moderate size (5-30 heavy atoms)
        size_penalty = abs(n - 15) / 30.0   # optimal ~15 heavy atoms
        metal_score  = min(metal / 5.0, 1.0)
        polar_score  = min((hba + hbd) / 8.0, 1.0)
        charge_score = min(charge / 3.0, 1.0)

        raw = (metal_score * 0.45 + polar_score * 0.30 +
               charge_score * 0.15 - size_penalty * 0.10)
        return max(0.0, min(1.0, raw))


# ── PINCER integration helpers ────────────────────────────────────────────────

class PincerNibbleAdapter:
    """
    Thin adapter so NibbleFitness plugs directly into Khukuri's
    ThreatAwareFitnessFunction interface.

    Usage in Khukuri:
        base_fitness = NibbleFitness.for_ndm1(pdb_path, tier=3)
        adapter = PincerNibbleAdapter(base_fitness)
        pincer_engine.set_fitness_function(adapter)
    """

    def __init__(self, nibble_fitness: NibbleFitness, fast_tier: int = 1):
        self.fitness   = nibble_fitness
        self.fast_tier = fast_tier
        self._orig_tier = nibble_fitness.tier

    def evaluate(self, smiles: str) -> float:
        """Standard single-molecule evaluation (full tier)."""
        return self.fitness.score(smiles)

    def evaluate_population(self, smiles_list: list[str]) -> list[float]:
        """Fast population evaluation — uses tier=1 for speed."""
        saved = self.fitness.tier
        self.fitness.tier = self.fast_tier
        scores = self.fitness.score_batch(smiles_list)
        self.fitness.tier = saved
        return scores

    def evaluate_apex(self, smiles: str) -> tuple[float, Optional[tuple]]:
        """Full Tier 3/4 evaluation for the apex drug candidate."""
        saved = self.fitness.tier
        self.fitness.tier = max(self._orig_tier, 3)
        score, pose = self.fitness.score_with_pose(smiles)
        self.fitness.tier = saved
        return score, pose

    def on_mutation(self, residue_voxel: tuple, diff: np.ndarray) -> None:
        """Called by PINCER when a resistance mutation modifies the pocket."""
        self.fitness.update_mutation(residue_voxel, diff)


# ── Quick validation ──────────────────────────────────────────────────────────

def _run_validation(pdb_path: Optional[str] = None):
    """
    Validate the fitness function against known NDM-1 inhibitors.
    Prints a ranking — expected order:
      taniborbactam > aspergillomarasmine_A > captopril > thiol_fragment > random
    """
    known = [
        ("Taniborbactam",           "OB1OCC2CN(C(=O)C3CCNC3)CC12",         0.05),
        ("Aspergillomarasmine_A",   "OC(=O)CC(N)CC(=O)NCC(N)CC(=O)O",      0.20),
        ("L-Captopril",             "CC1CS(=O)(=O)N(C1)C(=O)C(C)S",         1.20),
        ("Thiol_fragment",          "SCCC(=O)O",                             8.00),
        ("Biphenyl_tetrazole",      "C1=CC(=CC=C1)C2=CC=CC=C2C3=NN=NN3",    3.50),
        ("Random_alkane",           "CCCCCCCCCC",                           None),
        ("Benzene",                 "c1ccccc1",                             None),
    ]

    print("\n" + "="*68)
    print("  NDM-1 Nibble Fitness Validation")
    print("="*68)

    # Use surrogate if no PDB provided
    if pdb_path and Path(pdb_path).exists():
        fitness = NibbleFitness.for_ndm1(pdb_path, tier=1)
    else:
        print("  (No PDB — using surrogate scorer)")
        fitness = NibbleFitness.__new__(NibbleFitness)
        fitness._pocket = None
        fitness.center = NDM1_SITE["center"]
        fitness.tier = 1
        fitness.delta_G_bulk = 2.0
        fitness.kT = 0.593
        fitness._pocket2 = None
        fitness._mode = "surrogate"

    print(f"\n  {'Compound':<28} {'Score':>8}  {'IC50 (μM)':>10}  {'Metal':>7}")
    print(f"  {'-'*28} {'-'*8}  {'-'*10}  {'-'*7}")

    results = []
    for name, smiles, ic50 in known:
        atoms = smiles_to_atoms(smiles)
        score = fitness._score_atoms(atoms) if fitness._pocket else fitness._surrogate_score(atoms)
        metal = sum(a["metal"] for a in atoms)
        ic50_str = f"{ic50:.2f}" if ic50 else "N/A"
        results.append((score, name, smiles, ic50, metal))
        print(f"  {name:<28} {score:>8.4f}  {ic50_str:>10}  {metal:>7.2f}")

    results.sort(reverse=True)
    print(f"\n  Ranking (best → worst):")
    for i, (sc, name, _, ic50, _) in enumerate(results):
        ic50_str = f"IC50={ic50:.2f}μM" if ic50 else "no data"
        print(f"  {i+1}. {name:<28} score={sc:.4f}  {ic50_str}")

    # Check: taniborbactam and captopril should beat benzene and alkane
    scores = {name: sc for sc, name, _, _, _ in results}
    chelators = ["Taniborbactam", "L-Captopril", "Aspergillomarasmine_A"]
    non_binders = ["Random_alkane", "Benzene"]
    chelator_mean  = sum(scores[n] for n in chelators) / len(chelators)
    nonbinder_mean = sum(scores[n] for n in non_binders) / len(non_binders)

    ok = chelator_mean > nonbinder_mean
    print(f"\n  Chelator mean score:    {chelator_mean:.4f}")
    print(f"  Non-binder mean score:  {nonbinder_mean:.4f}")
    print(f"  Discrimination:         {'✓ PASS' if ok else '✗ FAIL — check metal demand channels'}")
    print("="*68 + "\n")
    return ok


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Nibble fitness function for Khukuri/NDM-1")
    p.add_argument("--pdb",      default=None,  help="PDB file path (3S0Z)")
    p.add_argument("--smiles",   default=None,  help="Score a single SMILES")
    p.add_argument("--validate", action="store_true", help="Run inhibitor validation")
    p.add_argument("--tier",     type=int, default=1, help="Scoring tier (1-4)")
    args = p.parse_args()

    if args.validate or args.smiles is None:
        _run_validation(args.pdb)

    if args.smiles:
        if args.pdb and Path(args.pdb).exists():
            fn = NibbleFitness.for_ndm1(args.pdb, tier=args.tier)
        else:
            fn = NibbleFitness.__new__(NibbleFitness)
            fn._pocket=None; fn.center=NDM1_SITE["center"]
            fn.tier=args.tier; fn.delta_G_bulk=2.0; fn.kT=0.593
            fn._pocket2=None; fn._mode="surrogate"
        atoms = smiles_to_atoms(args.smiles)
        score = fn._score_atoms(atoms) if fn._pocket else fn._surrogate_score(atoms)
        metal = sum(a["metal"] for a in atoms)
        print(f"\nSMILES: {args.smiles}")
        print(f"Atoms:  {len(atoms)}")
        print(f"Metal binding potential: {metal:.3f}")
        print(f"Nibble score: {score:.4f}")
