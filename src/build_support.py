"""Include runtime definitions from their canonical source directories in wheels."""

from pathlib import Path
from shutil import copy2, copytree, ignore_patterns, rmtree

from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        source = Path(__file__).parent.parent
        target = Path(self.build_lib) / "tabbench_bio" / "_resources"
        assert target.resolve().is_relative_to(Path(self.build_lib).resolve())
        if target.exists():
            rmtree(target)
        for directory in ("configs", "environments", "benchmark_my_model", "skills"):
            copytree(
                source / directory,
                target / directory,
                ignore=ignore_patterns("__pycache__"),
            )
        copy2(
            source / "requirements-autogluon-fork.txt", target / "requirements-autogluon-fork.txt"
        )
