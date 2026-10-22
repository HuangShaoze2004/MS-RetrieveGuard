"""Conservative, version-recorded two-dimensional chemistry primitives.

No neutralization, largest-fragment selection, or fingerprint-based identity labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

CHEMISTRY_VERSION = "normalize-preserve-charge-reject-fragments-no-stereo-v1"
FP_BITS = 4096
FP_BITORDER = "little"
_NORMALIZER = rdMolStandardize.Normalizer()
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS, includeChirality=False)


@dataclass(frozen=True)
class Structure:
    identity: str
    smiles: str
    formula: str
    exact_mass: float

    def to_dict(self) -> dict:
        return asdict(self)


def standardize(smiles: str) -> Structure:
    """Return sanitized, normalized 2-D structure or a descriptive ValueError."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("empty_smiles")
    # Expected invalid records are counted by the audit; do not emit millions of
    # parser/InChI warning lines into experiment logs.
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid_smiles")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("multicomponent")
    mol = _NORMALIZER.normalize(mol)
    Chem.SanitizeMol(mol)
    Chem.RemoveStereochemistry(mol)
    with rdBase.BlockLogs():
        identity = Chem.MolToInchiKey(mol).split("-")[0]
    if len(identity) != 14:
        raise ValueError("invalid_inchi_identity")
    return Structure(identity, Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False),
                     rdMolDescriptors.CalcMolFormula(mol), float(Descriptors.ExactMolWt(mol)))


def fingerprint(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid_smiles")
    values = np.zeros(FP_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(_MORGAN.GetFingerprint(mol), values)
    return np.packbits(values, bitorder=FP_BITORDER)


def chemistry_metadata() -> dict:
    return {"standardization": CHEMISTRY_VERSION, "rdkit_version": rdBase.rdkitVersion,
            "identity": "first14_standard_inchikey_recomputed_after_normalization_and_stereo_removal",
            "fingerprint": {"kind": "Morgan", "radius": 2, "bits": FP_BITS,
                            "include_chirality": False, "packed_bitorder": FP_BITORDER},
            "neutralize": False, "largest_fragment": False, "multicomponent": "quarantine"}
