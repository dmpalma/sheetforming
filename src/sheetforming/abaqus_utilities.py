"""
abaqus_utilities.py
====================

Shared helpers for reading Abaqus .rpt field-output reports (wrapped
header, elemental-nodal rows) -- used by triaxiality.py and
yield_functions.py to drive their models from a real/simulated strain
path instead of a synthetic one.

Author: Domingo Morales Palma <dmpalma@us.es>
"""

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .yield_criteria_plane_stress import PointState, YieldCriterionPlaneStress, VonMises_PS
from .plot_style import width_1column, width_2columns

# Column names are read from each report's own (wrapped, fixed-width)
# header -- see read_abaqus_report -- and mapped to these canonical
# names; any other variable keeps its Abaqus name, sanitised (e.g.
# "PE:PE11" -> "PE11", "S:S12" -> "S12", "STH" -> "STH").
#
# Principal values keep the historical names S_Max/S_Mid/S_Min (and
# PE_*), whatever the element type:
#   - 3D-element reports (e.g. 50-050-24_*.rpt): "Max/Mid/Min
#     Principal", algebraically sorted, none of them pinned to a
#     physical direction and sigma3 = 0 NOT enforced -- what makes them
#     ground truth for analyze_plane_stress_consistency.
#   - SHELL reports (HF-SPIF-*.rpt): "Max In-Plane", "Min In-Plane" and
#     "Out-of-Plane Principal" are stored as S_Max, S_Mid and S_Min
#     respectively (so S_Min is the out-of-plane one, ~0, NOT the
#     algebraic minimum). df.attrs["element_type"] tells them apart.
#
# Tensor components (S11, S22, S33, S12, S13, S23 and PE11, ...) are in
# the axes Abaqus reports them in: the element's local axes, i.e. the
# material axes when a local *ORIENTATION is defined. PE12, PE13, PE23
# are ENGINEERING shear strains (gamma_12 = 2*eps_12, as Abaqus outputs
# them -- checked against the reported in-plane principal values); S12,
# S13, S23 are the usual (tensor) shear stresses.
_PRINCIPAL_NAMES = {
    "Max Principal": "Max", "Mid Principal": "Mid", "Min Principal": "Min",
    "Max In-Plane Principal": "Max", "Min In-Plane Principal": "Mid",
    "Out-of-Plane Principal": "Min",
}

def _resolve_yield_criterion(
    yield_criterion: Optional[YieldCriterionPlaneStress],
) -> Optional[YieldCriterionPlaneStress]:
    """
    Validates yield_criterion (if given): must be an already-built,
    already-calibrated YieldCriterionPlaneStress instance -- no string-
    name shortcut (that would let an anisotropic criterion, e.g. YLD91_PS,
    be constructed zero-arg with every calibration field silently left
    None, which then makes solve() quietly return None for alpha/
    sigmaEQ/etc. instead of raising, surfacing later as a confusing
    arithmetic TypeError far from the real cause.
    """
    if yield_criterion is None:
        return None
    if not isinstance(yield_criterion, YieldCriterionPlaneStress):
        raise TypeError(
            f"yield_criterion must be an already-built YieldCriterionPlaneStress "
            f"instance (e.g. VonMises_PS()), or None "
            f"(got {type(yield_criterion)})"
        )
    # Exercise it on a trivial test point rather than inspecting its raw
    # fields for None: an anisotropic criterion (e.g. YLD91_PS) has several
    # alternative, mutually-substitutable calibration paths (r-values,
    # or C-coefficients, or experimental sigma0/45/90), so some fields
    # being None by design is normal and not itself a problem -- what
    # matters is whether _ensure_calibration can actually derive a
    # usable shape from whatever WAS given.
    if yield_criterion.solve(sigma1=1.0, sigma2=0.0).sigmaEQ is None:
        raise ValueError(
            f"{yield_criterion} does not appear to be calibrated -- solving a "
            f"trivial test point (sigma1=1, sigma2=0) failed to resolve sigmaEQ. "
            f"Pass a fully calibrated instance instead."
        )
    return yield_criterion


def _report_column_name(title: str) -> str:
    """
    Canonical DataFrame column name for one reassembled Abaqus column
    title, e.g. "S:Max In-Plane Principal (Avg: 75p) SP:1 PI: SHEET-1
    N: 19853" -> "S_Max", "PE:PE12 (Avg: 75p) ..." -> "PE12",
    "PEEQ (Avg: 75p) ..." -> "PEEQ", "X" -> "X" (see _PRINCIPAL_NAMES).
    """
    var = re.split(r" \(Avg| \(Not averaged| SP:| PI:| N:", title, maxsplit=1)[0].strip()
    if ":" in var:
        field, comp = var.split(":", 1)
        if comp in _PRINCIPAL_NAMES:
            return f"{field}_{_PRINCIPAL_NAMES[comp]}"
        return comp if comp.startswith(field) else f"{field}_{comp}".replace(" ", "_")
    return var.replace(" ", "_")


def _parse_report_value(token: str) -> float:
    """Float value of one data token: Fortran D exponents accepted, 'NoValue' -> NaN."""
    try:
        return float(token.replace("D", "E").replace("d", "E"))
    except ValueError:
        return np.nan


def read_abaqus_report(
    filepath: str,
    yield_criterion: Optional[YieldCriterionPlaneStress] = VonMises_PS(),
    repeated_times: str = "error",
) -> pd.DataFrame:
    '''
    Reads an Abaqus report and returns a DataFrame with one column per
    report column, named from the report's own header (see
    _report_column_name and the notes above _PRINCIPAL_NAMES). Report
    format: an (optionally blank-led) header block whose column titles
    are wrapped over several lines in fixed-width fields, e.g.

                                 PE:Max In-Plane P  ...  S:S11 (Avg: 75p)
                                 rincipal (Avg: 75  ...  SP:1 PI: SHEET-1
                                 p) SP:1 PI: SHEET  ...  N: 19853
                      X             -1 N: 19853     ...

    followed by a blank line, then the data rows. The first field (X) is
    wider than the others (19 characters each); titles are bottom-
    aligned, so a column's title is the concatenation of its field in
    every header line. Fields Abaqus has no value for ("NoValue", e.g.
    S33 in a shell, or PE before yielding) become NaN, keeping every row
    aligned with the header -- the column COUNT is taken from the header,
    and rows with a different number of fields are skipped.

    Export the report from Abaqus as UNIQUE NODAL: one row per instant X
    for the node. An Elemental Nodal export gives one row per element
    sharing the node at each instant, so X repeats; see repeated_times.

    Parameters:
      filepath : path to the (.rpt) file
      repeated_times : what to do if X repeats (Elemental Nodal export):
        "error" (default) raises ValueError asking for a Unique Nodal
        report; "average" collapses the rows of each instant into their
        mean (average_by_time), reproducing the nodal average of the
        components; "keep" returns every row as it is (one per element
        when the report lists the element values of the node), for
        calculations that must be done element by element before
        averaging (triaxiality.calc_average_eta_and_lode).
      yield_criterion : an already-built, already-calibrated
        YieldCriterionPlaneStress instance (e.g. VonMises_PS() or
        YLD91_PS(...)), or None. Attached to the
        returned DataFrame as df.attrs["yield_criterion"] -- metadata
        about the SIMULATION this report came from (which material
        model generated it), not a new column, so downstream code
        (analyze_plane_stress_consistency, or anything else needing to
        know "what criterion applies to this data") does not need it
        threaded through as a separate argument everywhere.

    Returns:
      pandas.DataFrame of floats. df.attrs also holds "element_type"
      ("shell" if the principal columns are In-Plane/Out-of-Plane, else
      "solid") and "column_titles" (canonical name -> full Abaqus title).
    '''
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = [line.rstrip('\r\n') for line in f]

    # Skip the header block explicitly (leading blank lines, then the
    # wrapped header text itself, then the blank line separating it from
    # the data): fragments of wrapped titles such as "-1" and "19853"
    # parse as valid floats on their own, so a header line must never be
    # mistaken for a data row.
    i, n = 0, len(lines)
    while i < n and lines[i].strip() == '':          # leading blank line(s)
        i += 1
    header = []
    while i < n and lines[i].strip() != '':           # the header block itself
        header.append(lines[i])
        i += 1
    if not header:
        raise ValueError(f"No header block found in {filepath}")

    # Fixed-width fields: the first (X) one is FIRST_WIDTH characters
    # wide, every other one FIELD_WIDTH, the first 2 of which are padding.
    FIRST_WIDTH, FIELD_WIDTH = 27, 19
    width = max(len(h) for h in header)
    ncols = 1 + round((width - FIRST_WIDTH) / FIELD_WIDTH)
    header = [h.ljust(width) for h in header]
    titles = ["".join(h[:FIRST_WIDTH] for h in header)]
    for k in range(ncols - 1):
        a = FIRST_WIDTH + FIELD_WIDTH * k
        titles.append("".join(h[a + 2:a + FIELD_WIDTH] for h in header))
    titles = [" ".join(t.split()) for t in titles]
    columns = [_report_column_name(t) for t in titles]
    if len(set(columns)) != len(columns):
        raise ValueError(f"Duplicate column names {columns} parsed from the header of {filepath}")

    rows = []
    for line in lines[i:]:
        toks = line.split()
        if len(toks) == ncols:
            row = [_parse_report_value(t) for t in toks]
            if not np.isnan(row[0]):
                rows.append(row)

    if not rows:
        raise ValueError(f"No data rows with {ncols} fields found in {filepath}")

    df = pd.DataFrame(rows, columns=columns)
    if repeated_times not in ("error", "average", "keep"):
        raise ValueError(f"repeated_times must be 'error', 'average' or 'keep', got {repeated_times!r}")
    averaged = False
    if not df["X"].is_unique and repeated_times != "keep":
        if repeated_times == "error":
            raise ValueError(
                f"{filepath}: {len(df)} rows for {df['X'].nunique()} distinct times: the report has several "
                f"values per instant (an Elemental Nodal export, or XY data of a node that Abaqus "
                f"returns once per element/region). Export one row per instant (Unique Nodal, "
                f"node selected by label), or pass repeated_times='average' to average the rows "
                f"of each instant.")
        df.attrs = {}
        df = average_by_time(df)
        averaged = True
    df.attrs["column_titles"] = dict(zip(columns, titles))
    # Shell elements: titles carry the section point (" SP:1") or in-plane principal values.
    shell = any("In-Plane" in t or re.search(r" SP:\d", t) for t in titles)
    df.attrs["element_type"] = "shell" if shell else "solid"
    if not shell:
        _add_solid_principal_columns(df, overwrite=averaged)
    if shell:
        # Out-of-plane stress: zero by the shell formulation (plane stress), but Abaqus writes
        # NoValue in most rows of a node. Filling it keeps every element usable.
        for c in ("S33", "S_Min"):
            if c in df.columns:
                df[c] = df[c].fillna(0.0)
        _add_shell_principal_columns(df, overwrite=averaged)
    df.attrs["yield_criterion"] = _resolve_yield_criterion(yield_criterion)
    return df


def _add_shell_principal_columns(df: pd.DataFrame, overwrite: bool = False) -> None:
    """
    Shell report exported with the tensor components only: derive the principal values the
    rest of the module uses, with the shell convention of _PRINCIPAL_NAMES (S_Max / S_Mid =
    max / min IN-PLANE principal, S_Min = out-of-plane, S33; likewise PE_*, with PE12
    ENGINEERING shear, so the tensor shear is PE12 / 2). Columns already present are kept unless
    overwrite (used after averaging rows: the principal values of the MEAN tensor are not the mean of
    the principal values).
    """
    for field, shear_factor in (("S", 1.0), ("PE", 0.5)):
        names = (f"{field}_Max", f"{field}_Mid", f"{field}_Min")
        comps = (f"{field}11", f"{field}22", f"{field}12")
        if (all(c in df.columns for c in names) and not overwrite) or not all(c in df.columns for c in comps):
            continue
        a, b, c = (df[k].to_numpy(float) for k in comps)
        c = shear_factor * c
        mean, radius = 0.5 * (a + b), np.hypot(0.5 * (a - b), c)
        df[names[0]] = mean + radius
        df[names[1]] = mean - radius
        df[names[2]] = df[f"{field}33"] if f"{field}33" in df.columns else 0.0


def _add_solid_principal_columns(df: pd.DataFrame, overwrite: bool = False) -> None:
    """
    Solid-element report exported with the FULL tensor components (S11, S22, S33, S12, S13, S23 and,
    likewise, PE11, PE22, PE33, PE12, PE13, PE23 with ENGINEERING shear): derive the algebraically
    sorted principal values S_Max / S_Mid / S_Min (PE_*). Without S13 and S23 the principal values
    cannot be obtained, and nothing is added. Columns already present are kept unless overwrite.
    """
    for field, shear_factor in (("S", 1.0), ("PE", 0.5)):
        names = (f"{field}_Max", f"{field}_Mid", f"{field}_Min")
        comps = tuple(f"{field}{c}" for c in ("11", "22", "33", "12", "13", "23"))
        if (all(c in df.columns for c in names) and not overwrite) or not all(c in df.columns for c in comps):
            continue
        a, b, c, t12, t13, t23 = (df[k].to_numpy(float) for k in comps)
        n = len(df)
        tensor = np.empty((n, 3, 3))
        tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2] = a, b, c
        tensor[:, 0, 1] = tensor[:, 1, 0] = shear_factor * t12
        tensor[:, 0, 2] = tensor[:, 2, 0] = shear_factor * t13
        tensor[:, 1, 2] = tensor[:, 2, 1] = shear_factor * t23
        ok = np.isfinite(tensor).all(axis=(1, 2))
        values = np.full((n, 3), np.nan)
        if ok.any():
            values[ok] = np.linalg.eigvalsh(tensor[ok])[:, ::-1]
        for k, name in enumerate(names):
            df[name] = values[:, k]


def add_principal_values(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    """
    Adds (in place, and returns df) the principal values S_Max / S_Mid / S_Min and PE_Max / PE_Mid / PE_Min
    derived from the tensor components of the report (see _PRINCIPAL_NAMES for the conventions of shell and
    solid reports). With overwrite=True the principal values already in the report (e.g. those Abaqus
    computes element by element and then averages) are replaced by the principal values of the tensor
    in the report, so that stresses and strains come from the same tensor.
    """
    if df.attrs.get("element_type") == "shell":
        _add_shell_principal_columns(df, overwrite=overwrite)
    else:
        _add_solid_principal_columns(df, overwrite=overwrite)
    return df


def average_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses the rows that share the same instant 'X' into a single row,
    averaging the remaining columns. read_abaqus_report(...,
    repeated_times="average") applies it; new reports should be exported
    as Unique Nodal instead, so that it is not needed.

    The Abaqus report was exported with 'Elemental Nodal' position (not
    'Unique Nodal'): for the same node and the same instant, a row appears
    for each element sharing it (up to 6 in this case), each with its own
    PE/PEEQ/S value. This function reproduces the nodal average ('Unique
    Nodal') that Abaqus would have given, so that each time instant is
    represented by a single point along the strain path.
    """
    out = df.groupby('X', as_index=False).mean()
    out.attrs = dict(df.attrs)  # groupby/mean does not reliably propagate .attrs
    return out


def split_in_plane_principal_stresses(s_max: float, s_mid: float, s_min: float) -> Tuple[float, float, float]:
    """
    Given one row's 3 principal stresses (algebraically Max >= Mid >=
    Min, NOT tied to any fixed physical direction), picks which one is
    OUT-OF-PLANE and returns (sigma1, sigma2, sigma3) -- the other two,
    larger first, then the out-of-plane one.

    At a genuine free surface (no direct contact there), continuum
    mechanics guarantees the surface normal IS a principal direction
    with principal value 0 (the traction vector on that face is zero,
    the definition of an eigenvector with eigenvalue 0) -- so the
    out-of-plane one should read ~0, and the natural guess is
    "whichever of the 3 has the smallest |value|". But that signal is
    contaminated by per-row dynamic/contact noise in an explicit-
    dynamics SPIF simulation (the same noise documented in
    triaxiality.calc_average_eta_and_lode) -- and, more importantly,
    picking on |value| ALONE can silently produce a physically
    implausible in-plane state: BIAXIAL in-plane compression, which
    does not happen in sheet forming (a stretched flange, in
    particular, is under tension by definition). So here that greedy
    pick is REJECTED whenever it would leave both remaining stresses
    <= 0, falling back to the next-smallest-|value| candidate instead
    -- i.e. "closest to zero, but only among choices that leave at
    least one in-plane stress tensile". Still only a heuristic -- not a
    substitute for reading the real fixed-frame out-of-plane component
    once reports export it directly (components S11, S22, S12, ...) -- but a
    materially better one than |value| alone.
    """
    labeled = sorted([("max", s_max), ("mid", s_mid), ("min", s_min)], key=lambda kv: abs(kv[1]))
    for name, value in labeled:
        remaining = [v for n, v in labeled if n != name]
        if any(v > 0 for v in remaining):
            sigma1, sigma2 = sorted(remaining, reverse=True)
            return sigma1, sigma2, value
    # No exclusion leaves a tensile in-plane stress (all 3 <= 0) -- fall
    # back to the plain smallest-|value| guess.
    name, value = labeled[0]
    sigma1, sigma2 = sorted([v for n, v in labeled if n != name], reverse=True)
    return sigma1, sigma2, value


def split_in_plane_principal_strains(eps_max: float, eps_mid: float, eps_min: float) -> Tuple[float, float, float]:
    """
    Given one row's 3 principal (plastic) strains, picks which one is
    OUT-OF-PLANE (thickness) and returns (eps1, eps2, eps3) -- the
    other two, larger first, then the out-of-plane one.

    Unlike the stress side (split_in_plane_principal_stresses), this
    does not need a magnitude-based guess: under plastic incompress-
    ibility (deps1 + deps2 + deps3 = 0) and stretching-dominated sheet
    forming, the two IN-PLANE strains are non-negative (stretching)
    while the THICKNESS strain is negative (thinning) -- so it is the
    UNIQUE negative one among the three (confirmed directly against
    real report data: ~98% of rows have exactly one negative value).
    Falls back to the smallest-|value| one if that pattern does not
    hold for a given row (zero or 2-3 negative values).
    """
    labeled = [("max", eps_max), ("mid", eps_mid), ("min", eps_min)]
    negatives = [(n, v) for n, v in labeled if v < 0]
    if len(negatives) == 1:
        name, value = negatives[0]
    else:
        name, value = min(labeled, key=lambda kv: abs(kv[1]))
    eps1, eps2 = sorted([v for n, v in labeled if n != name], reverse=True)
    return eps1, eps2, value


def _activity_blocks(peeq: np.ndarray):
    """
    (start, end) row indices of each ACTIVITY BLOCK: a maximal run of
    rows over which PEEQ keeps increasing, plus the row right before it
    as the baseline (see analyze_plane_stress_consistency).
    """
    growing = np.diff(peeq, prepend=peeq[0]) > 0
    blocks = []
    start = None
    for i, g in enumerate(growing):
        if g and start is None:
            start = i - 1
        elif not g and start is not None:
            blocks.append((start, i - 1))
            start = None
    if start is not None:
        blocks.append((start, len(peeq) - 1))
    return blocks


def _analyze_flow_consistency_components(
    x_time: np.ndarray,
    peeq: np.ndarray,
    s11: np.ndarray, s22: np.ndarray, s12: np.ndarray,
    pe11: np.ndarray, pe22: np.ndarray, pe12: np.ndarray,
    criterion: YieldCriterionPlaneStress,
) -> pd.DataFrame:
    """
    Component-based version of _analyze_flow_consistency, for reports
    that export the in-plane tensors in MATERIAL axes (S11, S22, S12 and
    PE11, PE22, PE12, engineering shear). Per activity block:

      - stress side: criterion.solve(sigma_xx=S11, sigma_yy=S22,
        tau_xy=S12) at the block's last row, so the anisotropic
        criterion sees the true ORIENTATION of the stress state
        (theta_stress = angle from axis 1 to sigma1), not just its
        principal values -> alpha_stress, beta_stress, sigmaEQ_stress,
        and psi_stress, the non-coaxiality the flow rule predicts;
      - strain side: the block's plastic strain INCREMENTS dPE11, dPE22,
        dPE12 -- fixed-frame components, so no principal-axis ranking
        is involved -- projected on the stress principal axes:

            deps1 = dPE11 c^2 + dPE22 s^2 + dPE12 s c
            deps2 = dPE11 s^2 + dPE22 c^2 - dPE12 s c    (c, s of theta_stress)

        -> beta_strain = deps2/deps1 (the same definition of beta the
        criterion uses off-axis, see yield_criteria_plane_stress),
        alpha_strain = calc_alpha_from_beta at that theta_stress, and
        psi_strain = angle from sigma1 to the major principal direction
        of the measured strain increment.
    """
    wrap = YieldCriterionPlaneStress._wrap_half_turn
    rows = []
    for start, end in _activity_blocks(peeq):
        stress_state = criterion.solve(sigma_xx=s11[end], sigma_yy=s22[end], tau_xy=s12[end])
        theta = stress_state.theta_stress
        c, s = np.cos(theta), np.sin(theta)

        d11, d22, d12 = pe11[end] - pe11[start], pe22[end] - pe22[start], pe12[end] - pe12[start]
        d1 = d11 * c * c + d22 * s * s + d12 * s * c
        d2 = d11 * s * s + d22 * c * c - d12 * s * c
        beta_strain = d2 / d1 if d1 != 0 else float("nan")
        alpha_strain = None
        if np.isfinite(beta_strain):
            alpha_strain = criterion.calc_alpha_from_beta(PointState(beta=beta_strain, theta_stress=theta))
        psi_stress = criterion.calc_psi_noncoax_from_theta_stress(
            PointState(theta_stress=theta, alpha=stress_state.alpha))

        def nan(v):
            return float("nan") if v is None else v

        rows.append({
            "X": x_time[end],
            "PEEQ": peeq[end],
            "theta_stress": theta,
            "alpha_stress": nan(stress_state.alpha),
            "beta_stress": nan(stress_state.beta),
            "beta_strain": beta_strain,
            "alpha_strain": nan(alpha_strain),
            "beta_diff": nan(stress_state.beta) - beta_strain,
            "alpha_diff": nan(stress_state.alpha) - nan(alpha_strain),
            "psi_stress": nan(psi_stress),
            "psi_strain": wrap(0.5 * float(np.arctan2(d12, d11 - d22)) - theta),
            "sigmaEQ_stress": nan(stress_state.sigmaEQ),
        })
    return pd.DataFrame(rows)


def _analyze_flow_consistency(
    x_time: np.ndarray,
    peeq: np.ndarray,
    sigma1_arr: np.ndarray,
    sigma2_arr: np.ndarray,
    eps1_arr: np.ndarray,
    eps2_arr: np.ndarray,
    criterion: YieldCriterionPlaneStress,
) -> pd.DataFrame:
    """
    Shared core of analyze_plane_stress_consistency (3D reports) and
    analyze_shell_flow_consistency (SHELL reports): given the two
    IN-PLANE sigma1/sigma2 (per row) and eps1/eps2 (per row,
    CUMULATIVE) arrays already identified by the caller -- by
    whichever method fits that report type -- groups rows into
    ACTIVITY BLOCKS and resolves the stress-based vs strain-based
    alpha/beta reconstruction for each one. See
    analyze_plane_stress_consistency's docstring for the full
    reasoning (activity blocks, increments vs cumulative values, what
    alpha_diff/beta_diff mean).
    """
    rows = []
    for start, end in _activity_blocks(peeq):
        stress_state = criterion.solve(sigma1=sigma1_arr[end], sigma2=sigma2_arr[end])

        d1, d2 = eps1_arr[end] - eps1_arr[start], eps2_arr[end] - eps2_arr[start]
        valid = d1 != 0
        beta_strain = (d2 / d1) if valid else float("nan")
        strain_state = criterion.solve(beta=beta_strain) if valid else None

        alpha_stress = stress_state.alpha
        beta_stress = stress_state.beta
        alpha_strain = strain_state.alpha if strain_state is not None else float("nan")

        rows.append({
            "X": x_time[end],
            "PEEQ": peeq[end],
            "alpha_stress": alpha_stress,
            "beta_stress": beta_stress,
            "beta_strain": beta_strain,
            "alpha_strain": alpha_strain,
            "beta_diff": (beta_stress - beta_strain) if beta_stress is not None else float("nan"),
            "alpha_diff": (alpha_stress - alpha_strain) if alpha_stress is not None else float("nan"),
            "sigmaEQ_stress": stress_state.sigmaEQ,
        })
    return pd.DataFrame(rows)


def analyze_plane_stress_consistency(
    df: pd.DataFrame,
    yield_criterion: Optional[YieldCriterionPlaneStress] = None,
) -> pd.DataFrame:
    """
    Row-wise check of the plane-stress assumption against a 3D-element
    Abaqus report (a df from read_abaqus_report on one
    of the 3D reports, e.g. 50-050-24_*.rpt -- NOT the HF-SPIF-*.rpt
    ones, which come from SHELL elements and so already enforce
    sigma3 = 0 by element formulation; there is nothing to check there
    -- see analyze_shell_flow_consistency instead for those).

    Which of the 3 principal stresses (and, independently, which of the
    3 principal strains) is treated as OUT-OF-PLANE is not assumed
    fixed (S_Min/PE_Min) -- it's picked PER ROW via
    split_in_plane_principal_stresses/split_in_plane_principal_strains;
    see their docstrings for why this matters and how each decides.

    For each row, resolves TWO INDEPENDENT PointStates from disjoint
    minimal subsets of that row's own (unconstrained, 3D) simulated
    data, via the same criterion's solve() -- using the two IN-PLANE
    columns identified above:

      - stress-only:  sigma1, sigma2 = the 2 in-plane principal stresses
            -> alpha_stress = sigma2/sigma1 (measured)
            -> beta_stress  = the flow direction the criterion's own
               associated flow rule PREDICTS from this stress state
      - strain-only:  beta_strain = deps2 / deps1 -- deps1, deps2 the
               INCREMENTS of the 2 in-plane principal strains OVER ONE
               ACTIVITY BLOCK (see below), NOT the cumulative
               PE_Max/PE_Mid/PE_Min values themselves -- beta is a
               ratio of increments by definition; using the cumulative
               values directly would silently assume the strain ratio
               had stayed constant since t = 0, wrong in general for a
               path with local bending oscillation, e.g. as a SPIF tool
               passes underneath a node
            -> alpha_strain = what stress ratio the criterion's
               alpha(beta) relation says WOULD produce that flow
               direction

    Rows are grouped into ACTIVITY BLOCKS before computing deps1/deps2,
    instead of differencing every pair of adjacent report rows: PEEQ
    only accumulates while the tool is actually near this node (the
    "peaks" in a plot of any field vs time), and is flat everywhere
    else -- in this report, PEEQ is unchanged (deps1 == 0 exactly)
    across 453/732 consecutive-row pairs. Dividing by that would blow
    up beta_strain (and, on a genuine 0/0, drop the row as NaN) far
    more often than it produces a real, usable value. A block is a
    maximal run of rows over which PEEQ keeps increasing (plus the one
    row right before it starts, as the baseline); deps1/deps2 are
    summed over the WHOLE block, giving one (alpha_strain, beta_strain)
    per peak instead of one per report row -- the returned DataFrame
    therefore has fewer rows than df, one per active block, not one per
    original row (the flat stretches between peaks carry no strain-side
    information to report). The stress side (alpha_stress, beta_stress,
    sigmaEQ_stress) is read directly from the block's LAST row (no
    aggregation needed there -- stress isn't cumulative).

    If the real (3D, not plane-stress-constrained) material actually
    followed this criterion with an associated flow rule, both
    reconstructions should agree closely; systematic disagreement
    (alpha_diff, beta_diff) flags where the plane-stress criterion is a
    poor fit to the real 3D response.

    Caveat: within a block, split_in_plane_principal_strains picks
    "in-plane 1"/"in-plane 2" independently per row (not a fixed
    physical axis) -- summing deps1/deps2 across a block implicitly
    assumes that pick stays the same rank throughout it. This is rare
    to violate in practice (the split is stable at ~98% of rows) but
    not impossible; a future fixed-frame strain export (see
    components PE11, PE22, PE12) removes the ambiguity entirely.

    Parameters
    ----------
    df : DataFrame from read_abaqus_report, one of
        the 3D-element reports.
    yield_criterion : overrides df.attrs["yield_criterion"] (the one
        read_abaqus_report attached) if given; required if neither is set.
    """
    criterion = yield_criterion or df.attrs.get("yield_criterion")
    if criterion is None:
        raise ValueError(
            "No yield_criterion given and none found in df.attrs -- pass one here, "
            "or read the report via read_abaqus_report(..., yield_criterion=...)."
        )
    criterion = _resolve_yield_criterion(criterion)

    # In-plane pair for every row (stress used as-is; strain kept
    # CUMULATIVE here -- summed over each activity block by
    # _analyze_flow_consistency, not differenced row-to-row).
    sigma_pairs = [split_in_plane_principal_stresses(r["S_Max"], r["S_Mid"], r["S_Min"]) for _, r in df.iterrows()]
    eps_pairs = [split_in_plane_principal_strains(r["PE_Max"], r["PE_Mid"], r["PE_Min"]) for _, r in df.iterrows()]

    return _analyze_flow_consistency(
        df["X"].to_numpy(), df["PEEQ"].to_numpy(),
        np.array([s[0] for s in sigma_pairs]), np.array([s[1] for s in sigma_pairs]),
        np.array([e[0] for e in eps_pairs]), np.array([e[1] for e in eps_pairs]),
        criterion,
    )


# In-plane tensor components a SHELL report needs for the component-based
# (orientation-aware) flow consistency check.
_SHELL_COMPONENTS = ("S11", "S22", "S12", "PE11", "PE22", "PE12")


def analyze_shell_flow_consistency(
    df: pd.DataFrame,
    yield_criterion: Optional[YieldCriterionPlaneStress] = None,
    use_components: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Same stress-based vs strain-based alpha/beta consistency check as
    analyze_plane_stress_consistency, but for a SHELL-element report
    (e.g. HF-SPIF-*.rpt) -- checking a DIFFERENT thing than the 3D
    version: not whether plane stress holds (sigma3 = 0 already exactly
    by element formulation, nothing to guess or verify there), but
    whether the chosen criterion's associated flow rule correctly
    predicts the measured strain direction from the measured stress
    direction -- i.e. how good a fit the criterion's SHAPE (isotropic
    Von Mises vs the real anisotropic material) is, independent of the
    plane-stress question.

    Unlike analyze_plane_stress_consistency, the two IN-PLANE columns
    are read directly and structurally here -- PE_Max/S_Max = "Max
    In-Plane Principal", PE_Mid/S_Mid = "Min In-Plane Principal" in the
    source report (PE_Min/S_Min = "Out-of-Plane Principal", unused) --
    rather than guessed via split_in_plane_principal_stresses/strains,
    since for a SHELL report which one is out-of-plane is already known
    from the report's own column order, not something to infer from
    magnitude or sign.

    df MUST come from read_abaqus_report on an actual
    SHELL report -- reading a 3D one here would misinterpret its
    algebraically max/mid/min-sorted columns as if they were already
    In-Plane/In-Plane/Out-of-Plane (use analyze_plane_stress_consistency
    for those instead). See that function's docstring for what
    alpha_stress/beta_stress/alpha_strain/beta_strain/*_diff mean and
    how the activity-block grouping works.

    If the report also exports the in-plane tensors in material axes
    (S11, S22, S12, PE11, PE22, PE12), they are used instead (see
    _analyze_flow_consistency_components): an anisotropic criterion then
    sees the true orientation of the stress state, and the strain
    increments are fixed-frame components rather than rotating
    principal values. The result then has extra columns theta_stress,
    psi_stress and psi_strain.

    Parameters
    ----------
    df : DataFrame from read_abaqus_report, a
        SHELL-element report.
    yield_criterion : overrides df.attrs["yield_criterion"] (the one
        read_abaqus_report attached) if given; required if neither is set.
    use_components : force (True) or disable (False) the component-based
        check; by default it is used whenever the columns are present.
    """
    criterion = yield_criterion or df.attrs.get("yield_criterion")
    if criterion is None:
        raise ValueError(
            "No yield_criterion given and none found in df.attrs -- pass one here, "
            "or read the report via read_abaqus_report(..., yield_criterion=...)."
        )
    criterion = _resolve_yield_criterion(criterion)

    if use_components is None:
        use_components = set(_SHELL_COMPONENTS) <= set(df.columns)
    if use_components:
        return _analyze_flow_consistency_components(
            df["X"].to_numpy(), df["PEEQ"].to_numpy(),
            *(df[c].to_numpy() for c in _SHELL_COMPONENTS),
            criterion,
        )
    return _analyze_flow_consistency(
        df["X"].to_numpy(), df["PEEQ"].to_numpy(),
        df["S_Max"].to_numpy(), df["S_Mid"].to_numpy(),
        df["PE_Max"].to_numpy(), df["PE_Mid"].to_numpy(),
        criterion,
    )


def plot_flow_consistency(result: pd.DataFrame, title: str, alpha_zoom=(-10.0, 5.0), beta_zoom=(-5.0, 5.0),
                          savefile=None):
    """
    Plots alpha and beta (stress- vs strain-based) against PEEQ for the
    DataFrame returned by analyze_plane_stress_consistency or
    analyze_shell_flow_consistency, in two columns:

    - general view: y-limits from the 2nd-98th percentiles of both
      curves (clips their outliers: ratios that blow up where sigma1 or
      deps1 crosses zero);
    - zoom: fixed y-limits alpha_zoom, beta_zoom, to read the distance
      between the two curves where they are closest. alpha_zoom = beta_zoom
      = None draws the general view only.

    If savefile is given, the figure is saved there. Returns the Figure.
    """
    zoom = alpha_zoom is not None or beta_zoom is not None
    ncols = 2 if zoom else 1
    width = width_2columns if zoom else width_1column
    fig, axes = plt.subplots(2, ncols, figsize=(width, width_1column * 1.3), sharex=True, squeeze=False)

    def general_limits(stress, strain):
        # 2nd-98th percentile band of each curve (both can have outliers: alpha and
        # beta are ratios), joined so that neither curve is clipped off-canvas
        lo_s, hi_s = np.nanpercentile(stress, [2, 98])
        lo_e, hi_e = np.nanpercentile(strain, [2, 98])
        lo, hi = min(lo_s, lo_e), max(hi_s, hi_e)
        margin = 0.2 * (hi - lo)
        return lo - margin, hi + margin

    panels = (("alpha", r"$\alpha = \sigma_2/\sigma_1$", r"$\alpha$ (from stress)", r"$\alpha$ (from strain)", alpha_zoom),
              ("beta", r"$\beta = d\varepsilon_2/d\varepsilon_1$", r"$\beta$ (predicted from stress)",
               r"$\beta$ (measured from strain)", beta_zoom))
    for row, (var, ylabel, label_stress, label_strain, zoom_limits) in enumerate(panels):
        stress, strain = result[f"{var}_stress"], result[f"{var}_strain"]
        for col in range(ncols):
            ax = axes[row, col]
            ax.plot(result["PEEQ"], stress, lw=0.8, label=label_stress)
            ax.plot(result["PEEQ"], strain, lw=1.5, label=label_strain)
            ax.set_ylim(general_limits(stress, strain) if col == 0 or zoom_limits is None else zoom_limits)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title("General view" if col == 0 else "Zoom", fontsize=9)
            if row == 1:
                ax.set_xlabel(r"$\varepsilon_{eq}$ (PEEQ)")
        axes[row, 0].legend(fontsize=8)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    if savefile is not None:
        fig.savefig(savefile)
        print(f"Figure saved in {savefile}")
    return fig
