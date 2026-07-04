"""HOLA — a Hippocampus for Linear Attention.

Build-with-Paper reproduction of:
  "A Hippocampus for Linear Attention: An Exact Memory for What the
   Recurrent State Forgets" (Wanyun Cui, arXiv:2607.02303).

Pure-PyTorch, CPU/MPS friendly (no Triton/CUDA kernels required).
"""

from .backbones import GatedDeltaNet, DeltaNet, GLA, BACKBONES, l2norm
from .cache import HOLACache
from .model import HOLALM

__all__ = [
    "GatedDeltaNet",
    "DeltaNet",
    "GLA",
    "BACKBONES",
    "HOLACache",
    "HOLALM",
    "l2norm",
]
