"""Ensure the repository root is importable as ``src`` during pytest runs."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
