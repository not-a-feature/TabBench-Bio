"""Generate the TabBench-Bio social preview from the current dashboard leaders."""

import argparse
from pathlib import Path

from tabbench_bio.social_preview import generate_social_preview

DEFAULT_SITE_DIR = Path(__file__).resolve().parents[1] / "website"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_SITE_DIR / "data/dashboard.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_SITE_DIR / "assets/og.png")
    args = parser.parse_args()
    generate_social_preview(args.dashboard, args.output)
