import os
import sys

# Get the repo root directory
repo_root = os.path.dirname(os.path.abspath(__file__))

# Add src directory to Python path if not already there
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
