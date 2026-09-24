"""UCRA - Uncertainty-to-Reservation Transformation Algorithm.

A self-evolving, risk-adaptive resource-allocation framework for 5G/B5G
network slicing, built on:

  Dataset 1  CTTC B5G Network Slicing (Zenodo 10610616, CC-BY-4.0)
  Dataset 2  Performance Management Counters from Live 5G/4G/2G RAN
             (Zenodo 17815388, CC-BY-4.0; Scientific Data, 2026)
  Optional   Liverpool 5G High-Density Demand Dataset (failsafe / stress)

Pipeline stages (see docs/METHODOLOGY.md):
  1. Uncertainty Estimation            -> ucra.models.quantile_lstm
  2. Uncertainty-to-Reservation (Phi)  -> ucra.core.transform
  3. Risk-Adaptive Allocation          -> ucra.core.allocate
  4. Reservation Update & Release      -> ucra.core.transform / evolve
  5. Self-Evolving Learning            -> ucra.core.evolve
"""

__version__ = "0.1.0"
