"""
RDKit chemistry helpers for generated SMILES. Provided — do not modify.

Canonical means RDKit's canonical SMILES (Chem.MolToSmiles), so two
strings for the same molecule compare equal. Uniqueness and novelty are
computed on canonical strings of valid samples.
"""

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED, Crippen, Descriptors, Draw
except ImportError as e:
    raise ImportError('provided/chem.py needs RDKit: pip install rdkit') from e

# Invalid samples are expected, so RDKit's per-string parse errors are noise
RDLogger.DisableLog('rdApp.*')

GRID_CELL_SIZE = (250, 250)


def is_valid(smiles: str) -> bool:
    """True if RDKit parses the string into a molecule. '' is not one."""
    return bool(smiles) and Chem.MolFromSmiles(smiles) is not None


def canonical(smiles: str) -> str:
    """RDKit canonical SMILES, or None if the string does not parse."""
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def validity(samples: list) -> float:
    """Fraction of samples that parse. samples: list of SMILES strings."""
    return sum(is_valid(s) for s in samples) / len(samples)


def uniqueness(samples: list) -> float:
    """
    Distinct canonical molecules / valid samples.

    Args:
        samples: List of SMILES strings

    Returns:
        Fraction in [0, 1], 0.0 if no sample is valid
    """
    valid = [c for c in (canonical(s) for s in samples) if c is not None]
    return len(set(valid)) / len(valid) if valid else 0.0


def novelty(samples: list, train_canonical: set) -> float:
    """
    Fraction of distinct valid molecules absent from the training set.

    Args:
        samples: List of SMILES strings
        train_canonical: Set of canonical training SMILES
            (load_canonical_set)

    Returns:
        Fraction in [0, 1], 0.0 if no sample is valid
    """
    distinct = {c for c in (canonical(s) for s in samples) if c is not None}
    if not distinct:
        return 0.0
    return len(distinct - train_canonical) / len(distinct)


def load_canonical_set(path: str) -> set:
    """Canonical SMILES of every line in a SMILES file (one per line)."""
    with open(path) as f:
        return {canonical(line.strip()) for line in f if line.strip()}


def properties(smiles: str) -> dict:
    """
    Drug-likeness, lipophilicity, and mass of one molecule.

    Args:
        smiles: SMILES string

    Returns:
        Dict with qed (QED in [0, 1]), logp (Crippen logP), mol_weight
        (Da), or None if the string does not parse
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'qed': QED.qed(mol),
        'logp': Crippen.MolLogP(mol),
        'mol_weight': Descriptors.MolWt(mol),
    }


def draw_grid(smiles_list: list, save_path: str, n_cols: int = 4,
              legends: list = None):
    """
    Render molecules as a PNG grid. Invalid strings leave a blank cell.

    Args:
        smiles_list: List of SMILES strings
        save_path: Output PNG path
        n_cols: Molecules per row
        legends: Optional list of captions, same length as smiles_list
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    image = Draw.MolsToGridImage(mols, molsPerRow=n_cols,
                                 subImgSize=GRID_CELL_SIZE, legends=legends)
    image.save(save_path)
