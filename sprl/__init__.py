"""SPRL-v2: Sparse-Patch Recurrent Latent.

Experimental decoder-only LM architecture targeting a single RTX 5080.
This is an alpha-stage research scaffold. See README.md for invariants
and `docs/architecture.md` for the central equation.
"""

from sprl.config import (  # noqa: F401
    SPRLConfig,
    default_full_500m,
    default_pilot_100m,
    default_pilot_200m,
)

__version__ = "0.1.0-alpha"
