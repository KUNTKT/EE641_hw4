"""
Regex SMILES tokenizer and vocabulary. Provided — do not modify.

Characters are the wrong unit for SMILES: Cl and Br are single atoms, and
a bracket atom such as [nH] or [C@@H] is one atom with its charge,
hydrogens, and chirality. The pattern (Schwaller et al. 2019) matches
these as single tokens, so joining the tokens of a string reproduces the
string exactly.

Ids: PAD = 0, BOS = 1, EOS = 2, then the chemistry tokens in sorted order.
"""

import json
import re

SMILES_PATTERN = re.compile(
    r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/'
    r'|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])')

PAD_TOKEN = '<pad>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


def tokenize(smiles: str) -> list:
    """Split a SMILES string into tokens. ''.join(tokens) == smiles."""
    return SMILES_PATTERN.findall(smiles)


class SmilesVocab:
    """Token-id mapping with PAD, BOS, EOS ahead of the sorted tokens."""

    def __init__(self, tokens):
        """
        Args:
            tokens: Iterable of chemistry tokens (duplicates and order
                ignored), e.g. every token of the training set
        """
        self.itos = SPECIAL_TOKENS + sorted(set(tokens))
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, smiles: str) -> list:
        """
        Args:
            smiles: SMILES string whose tokens are all in the vocabulary

        Returns:
            List of ids [BOS, t_1, ..., t_n, EOS], length n + 2
        """
        return ([BOS_ID] + [self.stoi[token] for token in tokenize(smiles)]
                + [EOS_ID])

    def decode(self, ids) -> str:
        """
        Inverse of encode. Stops at the first EOS, drops PAD and BOS.

        Args:
            ids: List or 1D tensor of ids

        Returns:
            SMILES string (empty if EOS comes first)
        """
        tokens = []
        for i in [int(i) for i in ids]:
            if i == EOS_ID:
                break
            if i not in (PAD_ID, BOS_ID):
                tokens.append(self.itos[i])
        return ''.join(tokens)

    def save(self, path: str):
        """Write the full id-ordered token list as JSON."""
        with open(path, 'w') as f:
            json.dump(self.itos, f, indent=1)

    @classmethod
    def load(cls, path: str) -> 'SmilesVocab':
        """Read a vocabulary written by save."""
        with open(path) as f:
            itos = json.load(f)
        return cls(itos[len(SPECIAL_TOKENS):])
