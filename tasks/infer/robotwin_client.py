"""Run in the RoboTwin environment; no VeOmni/Torch installation is required."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tasks.infer.robotwin.simulator import main


if __name__ == "__main__":
    main()
