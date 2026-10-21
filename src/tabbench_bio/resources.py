"""Locate bundled definitions and create an editable benchmark workspace."""

import shutil
from pathlib import Path

from tabbench_bio import __version__


def resource_root() -> Path:
    source = Path(__file__).resolve().parents[2]
    if (source / "configs/grid_sweep_all.json").is_file():
        return source
    return Path(__file__).parent / "_resources"


def init_workspace(destination: Path) -> None:
    destination = destination.resolve()
    assert not destination.exists(), f"Choose a new workspace directory: {destination}"
    source = resource_root()
    for directory in ("configs", "environments"):
        shutil.copytree(source / directory, destination / directory)
    for path in (destination / "environments").glob("*.txt"):
        lines = path.read_text(encoding="utf-8").splitlines()
        lines = [
            f"tabbench-bio{line.removeprefix('-e .')}=={__version__}"
            if line.startswith("-e .[")
            else line
            for line in lines
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(
        source / "requirements-autogluon-fork.txt", destination / "requirements-autogluon-fork.txt"
    )
    shutil.copy2(source / "benchmark_my_model/my_model.py", destination / "my_model.py")
    print(f"Workspace: {destination}")
    print(f'cd "{destination}"')
    print("tabbench-bio evaluate --model my_model.py --task overall")


def cmd_init(args) -> None:
    init_workspace(args.directory)
