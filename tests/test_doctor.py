import json
from argparse import Namespace
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from tabbench_bio import doctor
from tabbench_bio.bio.cache import save_cached_raw
from tabbench_bio.bio.datasets import BioDatasetSpec
from tabbench_bio.bio.fingerprint import spec_fingerprint
from tabbench_bio.bio.loaders.base import BioRawDataset
from tabbench_bio.model_registry import MODEL_REGISTRY


def test_doctor_reports_stale_raw_and_processed_caches_without_writing(tmp_path, monkeypatch):
    spec = BioDatasetSpec("toy", "local", "x.csv", target="y", problem_type="binary")
    monkeypatch.setattr(doctor, "load_specs", lambda: {"toy": spec})
    fingerprint = spec_fingerprint(spec)
    raw = BioRawDataset(
        "toy",
        pd.DataFrame({"x": [1]}),
        pd.Series([0]),
        "binary",
        "",
        "",
        "",
        {"spec_sha256": fingerprint},
    )
    raw_path = save_cached_raw(tmp_path / "bio", raw)
    split = tmp_path / "datasets_processed/seed_0_hash/split_params.json"
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps({"dataset_specs": {"toy": fingerprint}}))
    before = {p: p.read_bytes() for p in (raw_path, split)}
    assert doctor.check_cache(tmp_path) == []
    spec = replace(spec, target="changed")
    problems = doctor.check_cache(tmp_path)
    assert any("stale dataset cache" in issue for issue in problems)
    assert any("stale split specification" in issue for issue in problems)
    assert before == {p: p.read_bytes() for p in before}


def test_skill_copy_license_drift_is_reported(tmp_path):
    canonical = tmp_path / "skills" / doctor.SKILL_NAME / "SKILL.md"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("---\nlicense: EUPL-1.2\n---\n")
    site = tmp_path / "site"
    site.mkdir()
    (site / "skill.md").write_text("license: Apache-2.0\n")
    problems = doctor.check_skill(tmp_path, site)
    assert any("Stale skill copy" in issue for issue in problems)
    assert any("Missing generated skill asset" in issue for issue in problems)
    assert "Apache" in (site / "skill.md").read_text()


def test_missing_profile_adapter_and_map_drift_are_reported(tmp_path, monkeypatch):
    spec = replace(MODEL_REGISTRY["RF"], adapter="tabbench_bio.models.absent:Model")
    monkeypatch.setattr(doctor, "MODEL_REGISTRY", {"RF": spec})
    monkeypatch.setattr(doctor.site, "MODEL_CATEGORY", {"RF": "wrong"})
    problems = doctor.check_metadata(tmp_path)
    assert any("missing environment profile" in issue for issue in problems)
    assert any("missing adapter module" in issue for issue in problems)
    assert any("site category differs" in issue for issue in problems)


def test_import_probe_uses_the_model_environment(monkeypatch):
    monkeypatch.setattr(doctor, "model_python", lambda profile: f"/{profile}/python")

    def probe(command, **kwargs):
        assert command[0] == "/tabpfn35/python"
        assert command[-1] == "TABPFN-V3.5"
        assert "_resolve_hyperparameters" in command[2]
        return SimpleNamespace(returncode=1, stderr="ImportError: missing dependency")

    monkeypatch.setattr(doctor.subprocess, "run", probe)
    assert "missing dependency" in doctor.check_adapter("TABPFNV35")[0]


@pytest.mark.parametrize("problems,code", [([], 0), (["stale cache"], 1)])
def test_command_status_reflects_failures(tmp_path, monkeypatch, capsys, problems, code):
    (tmp_path / "configs/models").mkdir(parents=True)
    monkeypatch.setattr(doctor, "CHECKOUT", tmp_path)
    monkeypatch.setattr(doctor, "check_metadata", lambda _: [])
    monkeypatch.setattr(doctor, "check_skill", lambda *a: [])
    monkeypatch.setattr(doctor, "check_cache", lambda _: problems)
    with pytest.raises(SystemExit) as error:
        doctor.cmd_doctor(Namespace(cache_dir=tmp_path, site_dir=None, model=[]))
    assert error.value.code == code
    assert "Adapter imports not tested" in capsys.readouterr().out
