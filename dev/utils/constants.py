import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
AUTOMLBENCHMARK_CONFIG_DIR = (
    Path(os.environ["HOME"]) / ".config/automlbenchmark/benchmarks"
)