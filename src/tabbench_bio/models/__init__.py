"""Custom model wrappers for the TabBench-Bio pipeline.

AutoGluon model classes that aren't in AutoGluon's own registry live here, one
module per model, so they're grouped and discoverable rather than scattered at the
package root. (Mirrors the original RamanBench ``models/`` package; flattened — no
``custom/`` sublevel.)

Currently:

- :class:`~tabbench_bio.models.tabpfn_wide.TabPFNWideModel` — TabPFN-Wide
  (wide-dataset TabPFN variant) wrapped for AutoGluon, selectable via the model key
  ``"TABPFN-WIDE"``. Classification only.
- :class:`~tabbench_bio.models.tabfm.TabFMModel` — TabFM (Google Research's zero-shot
  tabular foundation model) wrapped for AutoGluon, selectable via the model key
  ``"TABFM"``. Supports both classification and regression.
- :class:`~tabbench_bio.models.tabpfn_v3.TabPFNV3Model` — explicit TabPFN-3 defaults,
  selectable via ``"TABPFN-V3"``. Supports both classification and regression.

Import the wrapper from its submodule (``from tabbench_bio.models.tabpfn_wide import
TabPFNWideModel``) so AutoGluon is only imported when the model is actually used.
"""
