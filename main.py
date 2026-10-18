#!/usr/bin/env python3
"""Launcher for the PNG to Gerber PCB-art application.

Run with:  python main.py [optional_image.png]
"""

import sys
from pathlib import Path

# Allow running straight from the repository without installing the package.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from png_to_gerber.gui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
