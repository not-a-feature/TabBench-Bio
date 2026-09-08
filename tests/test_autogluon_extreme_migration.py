import json

from scripts.migrate_frozen_autogluon_extreme import CONFIG_FILES, TARGET_CELLS, migrate


def _payload(preset: str) -> dict:
    return {
        "models": ["RF", "AUTOGLUON"],
        "autogluon_time_limit": 3600,
        "model_overrides": {
            "AUTOGLUON": {
                "presets": preset,
                "ensemble": True,
            }
        },
    }


def test_migration_updates_frozen_configs_with_immutable_backup(tmp_path):
    root = tmp_path / "results"
    for cell_name in TARGET_CELLS:
        cell = root / cell_name
        cell.mkdir(parents=True)
        for name in CONFIG_FILES:
            (cell / name).write_text(json.dumps(_payload("best_quality")), encoding="utf-8")

    migrate(root, "20260828T120000Z")

    for cell_name in TARGET_CELLS:
        cell = root / cell_name
        backup = cell / "immutable_backups" / "pre_autogluon_extreme_20260828T120000Z"
        assert (
            json.loads((backup / "config.json").read_text())["model_overrides"]["AUTOGLUON"][
                "presets"
            ]
            == "best_quality"
        )
        assert (backup / "sha256_manifest.json").is_file()
        for name in CONFIG_FILES:
            assert (
                json.loads((cell / name).read_text())["model_overrides"]["AUTOGLUON"]["presets"]
                == "extreme"
            )

    migrate(root, "20260828T120001Z")
    for cell_name in TARGET_CELLS:
        assert not (
            root / cell_name / "immutable_backups" / "pre_autogluon_extreme_20260828T120001Z"
        ).exists()
