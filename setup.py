"""Bundle the canonical skill without keeping a second source copy."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        source = Path(__file__).parent / "skills"
        destination = Path(self.build_lib) / "tabbench_bio" / "skills"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)


setup(cmdclass={"build_py": BuildPy})
