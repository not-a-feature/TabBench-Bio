"""Network- and model-free tests for protein embedding curation invariants."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "generate_protein_embeddings.py"
    spec = importlib.util.spec_from_file_location("generate_protein_embeddings", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


def test_chunk_sequences_preserves_every_residue_and_row_mapping():
    sequences = ["A" * 5, "BCDEFGH"]
    chunks = SCRIPT._chunk_sequences(sequences, chunk_residues=3)
    reconstructed = ["", ""]
    for row_index, chunk in chunks:
        reconstructed[row_index] += chunk
    assert reconstructed == sequences
    assert [len(chunk) for _, chunk in chunks] == [3, 2, 3, 3, 1]


def test_beta_groups_are_the_single_mutated_residue_positions():
    reference = "A" * 286
    sequences = []
    for position in range(20):
        sequence = list(reference)
        sequence[position] = "C"
        sequences.append("".join(sequence))

    consensus, positions = SCRIPT._mutation_positions(sequences)
    assert consensus == reference
    np.testing.assert_array_equal(positions, np.arange(1, 21))


def test_beta_group_derivation_rejects_multiple_mutations():
    reference = "A" * 286
    sequences = []
    for position in range(20):
        sequence = list(reference)
        sequence[position] = "C"
        sequences.append("".join(sequence))
    sequences[0] = "CC" + reference[2:]

    try:
        SCRIPT._mutation_positions(sequences)
    except ValueError as error:
        assert "expected exactly one" in str(error)
    else:  # pragma: no cover - makes an invariant regression explicit
        raise AssertionError("multiple mutations were accepted")
