"""Launcher script for Fraud-Spike Detector Web UI.

Usage:
  python scripts/run_ui.py
"""

import sys
import time
import webbrowser
from pathlib import Path

# Add project root to python import path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.web.server import run_server


def main():
    port = 8000
    url = f"http://127.0.0.1:{port}"
    print(f"🚀 Starting Fraud-Spike Detector Web Operations Console...")
    print(f"📡 Server running at: {url}")
    print(f"🛡️ Detector Version: v1.1.0 (Frozen StatisticalDeviationScorer)")
    
    # Open browser automatically after short delay
    try:
        webbrowser.open(url)
    except Exception:
        pass

    run_server(port=port)


if __name__ == "__main__":
    main()
