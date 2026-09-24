"""Entry point that works without installing the package: ``python src/main.py run ...``.

The submission zip ships ``src/`` without packaging metadata, so this shim puts
``src/`` on the import path and hands over to the CLI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ber.cli import main

if __name__ == "__main__":
    sys.exit(main())
