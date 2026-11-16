"""Compatibility imports; implementation lives in session.py."""

from .session import CausalInferenceSession, build_adapter


CausalPolicy = CausalInferenceSession

__all__ = ["CausalInferenceSession", "CausalPolicy", "build_adapter"]
