"""Network-free tests for the FusionAI middle-NT loader."""

from __future__ import annotations

import numpy as np
import pytest

from tabbench_bio.bio.loaders.fusionai import (
    ARCHIVE_BYTES,
    MEMBERS,
    NT_WIDTH_PER_PARTNER,
    exact_partner_components,
)


def test_fusionai_source_identities_are_pinned():
    assert ARCHIVE_BYTES == 3_574_557_943
    assert {member.basename for member in MEMBERS} == {
        "nt_train_seq1.csv",
        "nt_train_seq2.csv",
        "nt_test_seq1.csv",
        "nt_test_seq2.csv",
        "fusionai_train_target.csv",
        "fusionai_test_target.csv",
    }
    assert all(member.n_bytes > 0 and member.crc32 > 0 for member in MEMBERS)


def test_exact_partner_components_close_transitive_duplicate_partners():
    matrix = np.empty((4, 2 * NT_WIDTH_PER_PARTNER), dtype=np.float32)
    matrix[0, :NT_WIDTH_PER_PARTNER] = 1
    matrix[0, NT_WIDTH_PER_PARTNER:] = 2
    matrix[1, :NT_WIDTH_PER_PARTNER] = 1
    matrix[1, NT_WIDTH_PER_PARTNER:] = 3
    matrix[2, :NT_WIDTH_PER_PARTNER] = 4
    matrix[2, NT_WIDTH_PER_PARTNER:] = 3
    matrix[3, :NT_WIDTH_PER_PARTNER] = 5
    matrix[3, NT_WIDTH_PER_PARTNER:] = 6

    groups = exact_partner_components(matrix)

    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_exact_partner_components_rejects_wrong_width():
    with pytest.raises(ValueError, match="2048 columns"):
        exact_partner_components(np.zeros((2, 10), dtype=np.float32))
