"""Leakproof: catch data leakage, label noise and broken CV splits before you train."""

from .engine import audit, AuditConfig

__version__ = "0.1.0"
__all__ = ["audit", "AuditConfig", "__version__"]
