"""Launcher:  python main.py   (equivalent to: streamlit run ui/app.py)"""
import subprocess, sys
from pathlib import Path

if __name__ == "__main__":
    app = Path(__file__).resolve().parent / "ui" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=False)
