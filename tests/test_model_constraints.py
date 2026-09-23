from tabbench_bio.model_constraints import REGULAR_MAX_FEATURES


def test_regular_feature_limits_match_the_benchmarked_adapters():
    assert REGULAR_MAX_FEATURES == {
        "MITRA": 500,
        "REALTABPFN-V2": 500,
        "REALTABPFN-V2.5": 2_000,
        "TABDPT": 2_500,
        "TABFM": 2_000,
        "TABICL": 2_000,
        "TABPFN-V3": 10_000,
    }
