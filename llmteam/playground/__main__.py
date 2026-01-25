"""
Run playground as module: python -m playground
"""

import subprocess
import sys
import os


def main():
    """Launch Streamlit app."""
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


if __name__ == "__main__":
    main()
