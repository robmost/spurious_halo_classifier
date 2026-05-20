"""
plotting.py: shared plotting utilities for the spurious halo classifier.

Loads the project matplotlib style and exposes the style path so notebooks
can apply the same style without hardcoding a path.

Usage in a notebook or script:
    from src.utils.plotting import apply_style
    apply_style()

Or directly:
    import matplotlib.pyplot as plt
    plt.style.use(str(STYLE_PATH))
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style path
# ---------------------------------------------------------------------------

# Resolved relative to this file: src/utils/plotting.py -> reports/
STYLE_PATH: Path = (
    Path(__file__).parent.parent.parent / "reports" / "spurious_halo_classifier.mplstyle"
).resolve()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def apply_style() -> None:
    """
    Apply the project matplotlib style.

    Falls back to the default matplotlib style if the style file is not found,
    logging a warning rather than raising. This is done to prevent CI and environments
    without the full project structure from failing on import.
    """
    if not STYLE_PATH.exists():
        log.warning("Style file not found at '%s'; using matplotlib defaults.", STYLE_PATH)
        return
    plt.style.use(str(STYLE_PATH))
