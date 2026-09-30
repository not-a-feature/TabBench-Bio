"""Split indices for benchmark runs."""


def get_seeds(config):
    """Return the list of split indices for the run.

    The returned integers are the per-split ``random_state`` values the pipeline iterates
    over — each one both seeds the split and names its ``seed_<n>/`` output directory.

    * ``cv_folds`` set to ``k`` — one k-fold partition: yields ``[0, 1, ..., k - 1]``,
      the fold indices (see :meth:`tabbench_bio.benchmark.TabBenchBio._kfold_split`).
      ``n_repetitions`` must then be null or 1.
    * ``cv_folds`` null, ``n_repetitions`` set — repeated random holdout: yields
      ``[0, 1, ..., n_repetitions-1]``.
    * both null — the single ``random_state`` seed.

    All three keys are required in the config (see
    :data:`tabbench_bio.config.REQUIRED_KEYS`) — none defaults.

    Parameters
    ----------
    config : dict
        Benchmark configuration dictionary.

    Returns
    -------
    list[int]
        Ordered list of split indices to iterate over.
    """
    n_repetitions = config["n_repetitions"]
    cv_folds = config["cv_folds"]
    if cv_folds is not None:
        assert n_repetitions in (None, 1), "Repeated k-fold CV is not supported"
        return list(range(cv_folds))
    if n_repetitions is not None:
        return list(range(n_repetitions))
    return [config["random_state"]]
