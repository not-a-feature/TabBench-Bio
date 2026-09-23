"""Regression test for the live-curated MGYS00001255 target."""

from tabbench_bio.bio.datasets import load_specs


def test_mgys00001255_uses_the_current_valid_binary_target():
    spec = load_specs()["MGYS00001255"]
    assert spec.target == "host sex"
    assert spec.problem_type == "binary"
