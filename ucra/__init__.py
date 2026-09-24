"""UCRA - Uncertainty-to-Reservation Transformation Algorithm.

Pipeline stages (see docs/METHODOLOGY.md):
  1. Uncertainty Estimation            -> ucra.models.quantile_lstm
  2. Uncertainty-to-Reservation (Phi)  -> ucra.core.transform
  3. Risk-Adaptive Allocation          -> ucra.core.allocate
  4. Reservation Update & Release      -> ucra.core.transform
  5. Self-Evolving Learning            -> ucra.core.evolve
"""

__version__ = "0.1.0"
