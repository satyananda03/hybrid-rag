"""Regulation detectors, each tunable on its own, joined by ``pipeline``."""
from .pipeline import Regulation, analyze_regulation, remove_regulation_headers

__all__ = ["Regulation", "analyze_regulation", "remove_regulation_headers"]
