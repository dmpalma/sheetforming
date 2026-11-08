"""
plot_style.py
=============

Shared matplotlib styling for every plotting module in this project
(constitutive_laws.py, yield_criteria_plane_stress.py, triaxiality.py, material.py,
yield_functions.py): applies graph.mplstyle once on import, and exposes the standard journal
figure widths (1-column/1.5-column/2-column, per Elsevier's artwork
sizing guide -- https://www.elsevier.com/authors/author-schemas/artwork-and-media-instructions/artwork-sizing)
plus a savefig() helper for writing a manuscript-ready vector format
alongside a quick-look PNG.

There is deliberately only ONE style here, not a separate "screen" and
"publication" pair: a figure created at its true final print width (see
figsize()/width_1column & co below) already renders sharp on screen too
(matplotlib figure size is in inches, fonts in points -- both absolute
-- so nothing about the style depends on where the figure is viewed).
What genuinely differs between a quick look and a manuscript figure is
the output FILE FORMAT (PNG raster is fine for a quick look; a journal
wants vector PDF/SVG for line art) -- see savefig().

Usage: `import plot_style` before creating any figure (the import alone
applies the style; pull in width_1column/width_column_and_a_half/
width_2columns/figsize()/savefig() as needed). This replaces the
rcParams block that used to be hand-duplicated in yield_criteria_plane_stress.py and
triaxiality.py.

Note there is no image_dpi constant: graph.mplstyle already sets
savefig.dpi, so plain fig.savefig(path) (no dpi= argument) already saves
at 300 dpi -- passing dpi explicitly everywhere would just repeat that
same value from a second place.

Author: Domingo Morales Palma <dmpalma@us.es>
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt


plt.style.use(str(Path(__file__).with_name("graph.mplstyle")))


MM_PER_INCH = 25.4
# 1-column, 1.5-column and 2-column print widths, in mm, for a 2-column
# journal page (Elsevier's artwork sizing guide).
TEXTWIDTH_1COLUMN_MM = 90.0
TEXTWIDTH_COLUMN_AND_A_HALF_MM = 140.0
TEXTWIDTH_2COLUMNS_MM = 190.0

def figwidth(width_mm: float, scale: float = 1.5) -> float:
    """
    Figure width in inches for a target print width in mm.

    scale enlarges the figure beyond its true print size while keeping
    fonts/linewidths at their absolute (point-based) size -- i.e. it
    shrinks them relative to the figure -- which is what the default of
    1.5 here has been used for throughout this project: a bit more
    headroom for tick labels/legends on typically-busy plots than the
    true 90/140/190 mm widths allow. Pass scale=1.0 for the true
    physical print size.
    """
    return width_mm / MM_PER_INCH * scale


def figsize(width_mm: float, aspect_ratio: float = 1.0, scale: float = 1.5) -> Tuple[float, float]:
    """(width, height) in inches -- see figwidth() for width_mm/scale; height = width * aspect_ratio."""
    w = figwidth(width_mm, scale)
    return w, w * aspect_ratio


# Ready-to-use widths (inches, already scaled -- see figwidth()), for the
# figwidth=... parameters already threaded through this project's plot_*
# functions.
width_1column = figwidth(TEXTWIDTH_1COLUMN_MM)
width_column_and_a_half = figwidth(TEXTWIDTH_COLUMN_AND_A_HALF_MM)
width_2columns = figwidth(TEXTWIDTH_2COLUMNS_MM)


def savefig(fig, path: str, formats: Iterable[str] = ("png",), dpi: int = None) -> None:
    """
    Saves `fig` under `path` (without extension), once per entry in
    `formats` -- e.g. formats=("pdf", "png") for a manuscript-ready
    vector file plus a PNG for a quick look, both from the very same
    (already print-sized) figure -- see the module docstring. dpi=None
    (default) uses graph.mplstyle's savefig.dpi (300); pass dpi to
    override it for this call only.
    """
    for ext in formats:
        out_path = f"{path}.{ext}"
        fig.savefig(out_path, dpi=dpi)
        print(f"Figure saved in {out_path}")
