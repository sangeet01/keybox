"""
KeyBox — box package
====================
Top-level package entry for the box/ directory.

Standard KeyBox usage (formulation compatibility):
    from key import KeyBoxSystem, NibbleEngine

Khukuri docking fitness (drug discovery):
    from khukuri_nibble import NibbleFitness, PincerNibbleAdapter
    # or after key import:
    from key import NibbleFitness, PincerNibbleAdapter
"""

from key import (
    KeyBoxSystem,
    NibbleEngine,
    NibbleFitness,
    PincerNibbleAdapter,
)

__all__ = [
    "KeyBoxSystem",
    "NibbleEngine",
    "NibbleFitness",
    "PincerNibbleAdapter",
]
