"""Constraint-Aware Meta-Optimizer package."""

from .agent import PrimalDualAgent
from .meta_optimizer import MetaOptimizerRNN
from .meta_trainer import MetaTrainer, main
from .policy import GaussianPolicy

__all__ = [
    "MetaOptimizerRNN",
    "GaussianPolicy",
    "PrimalDualAgent",
    "main",
    "MetaTrainer",
]
__version__ = "0.1.0"
__author__ = "Misagh Soltani"
