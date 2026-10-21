"""Run the installed custom-model helper with the adjacent model template."""

import sys
from pathlib import Path

from tabbench_bio.evaluate import main

if __name__ == "__main__":
    main(["--model", str(Path(__file__).with_name("my_model.py")), *sys.argv[1:]])
