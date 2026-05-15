"""
main.py — Project-root entry point
====================================
Allows starting the proxy from the project root directory:

    python main.py

Equivalent to:  python src/main.py
"""

import sys
from pathlib import Path

# Add project root to sys.path so all 'src.*' imports resolve correctly.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Delegate to the real entry point.
from src.main import main

if __name__ == "__main__":
    main()
