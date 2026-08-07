"""Utilities for collecting pipeline results and rendering reports."""

from .generator import generate_reports
from .writer import write_stage_result

__all__ = ["generate_reports", "write_stage_result"]
