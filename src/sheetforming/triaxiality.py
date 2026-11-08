    #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Domingo Morales Palma <dmpalma@us.es>
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .yield_criteria_plane_stress import YieldCriterionPlaneStress, VonMises_PS, Hill48_PS, YLD91_PS, PointState
from .plot_style import width_1column, width_column_and_a_half, width_2columns
from .abaqus_utilities import read_abaqus_report



# Equivalent stress via yield_criteria_plane_stress.py: its classes (VonMises_PS, Hill48_PS,
# Hill79_PS, YLD91_PS, ...) assume plane stress (sigma3 = 0), but every
# criterion in the module is insensitive to hydrostatic pressure, as is any
# Hill/Hosford/Barlat-type criterion -- so shifting the real triaxial state
# (s1, s2, s3) by subtracting s3 (giving sigma1 = s1-s3, sigma2 = s2-s3,
# sigma3' = 0) keeps its equivalent stress. With principal values ALONE,
# though, the orientation of the state relative to the material axes is
# lost: sigma1 is taken along RD, which is exact only for isotropic
# criteria (VonMises_PS) or when s3 is the ND principal value and the
# in-plane principal axes are RD/TD. For anisotropic criteria use the
# stress components in material axes (sigma_eq_from_report for shell
# reports; yield_criteria_3d for full 3D tensors).
def sigma_eq_from_yield_criterion(s1, s2, s3, criterion: YieldCriterionPlaneStress):
    """
    Equivalent stress for a general triaxial state (s1, s2, s3), using
    an already-calibrated criterion from yield_criteria_plane_stress.py.

    Parameters:
      s1, s2, s3 : principal stresses (array-like), without assuming
        plane stress (s3 may be nonzero).
      criterion : YieldCriterionPlaneStress instance (VonMises_PS, Hill48_PS,
        Hill79_PS, YLD91_PS, ...), already calibrated.

    Returns:
      np.ndarray with the pointwise sigma_eq (NaN where the criterion
      cannot be evaluated, e.g. if anisotropy parameters are missing).
    """
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    s3 = np.asarray(s3, dtype=float)

    def _sigma_eq_one(a, b, c):
        v = criterion.calc_equivalent_stress(PointState(sigma1=a - c, sigma2=b - c))
        return v if v is not None else np.nan

    return np.vectorize(_sigma_eq_one, otypes=[float])(s1, s2, s3)


def _criterion_3d(criterion: YieldCriterionPlaneStress):
    """3D counterpart (yield_criteria_3d, full tensor in material axes) of a plane-stress criterion, or None."""
    from .yield_criteria_3d import Hill48_3D, VonMises_3D, YLD91_3D
    from .yield_criteria_plane_stress import Hill48_PS
    for ps_class, d3_class in ((VonMises_PS, VonMises_3D), (Hill48_PS, Hill48_3D), (YLD91_PS, YLD91_3D)):
        if isinstance(criterion, ps_class):
            return d3_class.from_plane_stress(criterion)
    return None


def sigma_eq_from_report(df: pd.DataFrame, criterion: YieldCriterionPlaneStress) -> np.ndarray:
    """
    Pointwise equivalent stress for a report DataFrame (read_abaqus_report
    ). For a SHELL report that exports the in-plane
    stress components in material axes (S11, S22, S12), they are used
    directly, so an anisotropic criterion sees the true orientation of
    the stress state. Otherwise falls back to the principal values
    (sigma_eq_from_yield_criterion), which is exact only for isotropic
    criteria or when the in-plane principal axes are the material axes.

    A SOLID report with the full stress tensor (S11, S22, S33, S12, S13, S23) is evaluated with the 3D
    version of the criterion (yield_criteria_3d), which takes the components as given: they must be in
    the material axes (Abaqus: the solid elements have a local *ORIENTATION). Von Mises, Hill 1948 and
    Yld91 have a 3D version; for another criterion the principal values are used.
    """
    if df.attrs.get("element_type") == "shell" and {"S11", "S22", "S12"} <= set(df.columns):
        criterion._ensure_calibration()

        def _one(sxx, syy, sxy):
            v = criterion._equivalent_stress_xy(sxx, syy, sxy)   # calc_equivalent_stress needs principal values
            return v if v is not None else np.nan
        return np.vectorize(_one, otypes=[float])(
            df["S11"].to_numpy(float), df["S22"].to_numpy(float), df["S12"].to_numpy(float))
    voigt_columns = ["S11", "S22", "S33", "S12", "S13", "S23"]
    if df.attrs.get("element_type") == "solid" and set(voigt_columns) <= set(df.columns):
        criterion._ensure_calibration()
        criterion_3d = _criterion_3d(criterion)
        if criterion_3d is not None:
            voigt = df[voigt_columns].to_numpy(float)
            out = np.full(len(df), np.nan)
            ok = np.isfinite(voigt).all(axis=1)
            if ok.any():
                out[ok] = criterion_3d.equivalent_stress(voigt[ok])
            return out
    return sigma_eq_from_yield_criterion(df["S_Max"], df["S_Mid"], df["S_Min"], criterion)


# Default criterion for the functions below: von Mises (isotropic).
_DEFAULT_CRITERION = VonMises_PS()




def _read_rows(abaqus_report, repeated_times: str, averaging: str) -> pd.DataFrame:
    """
    Report rows for the triaxiality calculations. averaging="elements" keeps the
    several rows of an instant (one per element of the node) when the user allows
    repeated times (repeated_times="average"), so that sigma_eq, eta and theta are
    computed element by element; "nodal" averages the rows first (the stress
    components of the node) and computes everything from that mean tensor.
    """
    if averaging not in ("elements", "nodal"):
        raise ValueError(f"averaging must be 'elements' or 'nodal', got {averaging!r}")
    if averaging == "elements" and repeated_times == "average":
        repeated_times = "keep"
    return read_abaqus_report(abaqus_report, repeated_times=repeated_times)


def _mean_by_time(time, **columns):
    """Mean of each column over the rows sharing the same time (valid rows only, NaN-free input)."""
    g = pd.DataFrame({"t": time, **columns}).groupby("t", sort=True).mean()
    return g.index.to_numpy(), {k: g[k].to_numpy() for k in columns}


def _pointwise_eta_theta(df: pd.DataFrame, criterion: YieldCriterionPlaneStress):
    """(time, peeq, eta, theta), one value per row of the report DataFrame (NaN where undefined)."""
    missing = [c for c in ("S_Max", "S_Mid", "S_Min", "PEEQ") if c not in df.columns]
    if missing:
        raise ValueError(
            f"The report has no {missing}. A shell report needs the in-plane components (S11, S22, S12) or the "
            f"principal values; a solid report needs the principal values or the FULL stress tensor "
            f"(S11, S22, S33, S12, S13, S23): without S13 and S23 the principal values cannot be obtained. "
            f"Columns found: {list(df.columns)}.")
    s1, s2, s3, peeq, time = df["S_Max"], df["S_Mid"], df["S_Min"], df['PEEQ'], df['X']
#    print(peeq)

    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    s3 = np.asarray(s3, dtype=float)
    peeq = np.asarray(peeq, dtype=float)
    time = np.asarray(time, dtype=float)

    sigma_m = (s1 + s2 + s3) / 3.0
    sigma_eq = sigma_eq_from_report(df, criterion)

    # pointwise triaxiality (NaN where sigma_eq == 0 or undefined)
    with np.errstate(divide='ignore', invalid='ignore'):
        eta = sigma_m / sigma_eq
    eta = np.where(np.isfinite(sigma_eq) & (sigma_eq == 0), np.nan, eta)
    eta = np.where(~np.isfinite(eta), np.nan, eta)
#    print(eta)

    # pointwise normalized Lode angle (NaN where sigma_eq_vm == 0 or undefined)
    d1, d2, d3 = s1 - sigma_m, s2 - sigma_m, s3 - sigma_m
    J2 = 0.5 * (d1**2 + d2**2 + d3**2)
    J3 = d1 * d2 * d3
    sigma_eq_vm = np.sqrt(3.0 * J2)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos3theta = 27.0 * J3 / (2.0 * sigma_eq_vm**3)
    cos3theta = np.clip(cos3theta, -1.0, 1.0)
    theta = 1.0 - (2.0 / np.pi) * np.arccos(cos3theta)
    theta = np.where(np.isfinite(sigma_eq_vm) & (sigma_eq_vm == 0), np.nan, theta)
    theta = np.where(~np.isfinite(theta), np.nan, theta)
#    print(theta)

    # Before yielding starts (peeq <= 0), the reported stresses are not a
    # genuine plastic stress state -- at these points they are dominated by
    # elastic/dynamic-contact noise (e.g. inertial oscillations right as
    # the tool makes contact, in an explicit-dynamics SPIF simulation),
    # so eta/theta computed from them are meaningless noise rather than a
    # real triaxiality/Lode angle, and are set to NaN here for the same
    # reason as the sigma_eq == 0 / sigma_eq_vm == 0 cases above.
    eta = np.where(peeq <= 0, np.nan, eta)
    theta = np.where(peeq <= 0, np.nan, theta)
    return time, peeq, eta, theta


_AVERAGE_COLUMNS = ['peeq', 'eta', 'avg_eta', 'theta', 'avg_theta', 'time', 'avg_eta_time', 'avg_theta_time']


def _cumulative_averages(time, peeq, eta, theta) -> pd.DataFrame:
    """Cumulative PEEQ-weighted (and time) averages of eta and theta along one history (see calc_average_eta_and_lode)."""
    # cumulative averages with respect to PEEQ (and, for diagnosis, time)
    columns = _AVERAGE_COLUMNS

    # mask of valid points (PEEQ, time, eta and theta all finite)
    mask = np.isfinite(peeq) & np.isfinite(time) & np.isfinite(eta) & np.isfinite(theta)
    if np.sum(mask) < 2:
        return pd.DataFrame(columns=columns)    # not enough valid points

    # sort by TIME; kind='stable' preserves the original record order between any exact ties
    order = np.argsort(time[mask], kind='stable')
    t = time[mask][order]
    y_eta = eta[mask][order]
    y_theta = theta[mask][order]
    p = peeq[mask][order]

    def cumulative_trapezoid(y, x):
        """Cumulative integral of y dx (trapezoidal rule), 0 at the first point."""
        return np.concatenate(([0.0], np.cumsum(np.diff(x) * 0.5 * (y[1:] + y[:-1]))))

    # PEEQ: increments clipped at 0 (PEEQ never decreases; averaging by time can
    # give tiny negative steps); from 0 to the first valid point, its own eta/theta
    dp = np.concatenate(([p[0]], np.clip(np.diff(p), 0.0, None)))
    p_acc = np.cumsum(dp)
    with np.errstate(invalid='ignore', divide='ignore'):
        avg_eta = (y_eta[0] * p[0] + cumulative_trapezoid(y_eta, p_acc)) / p_acc
        avg_theta = (y_theta[0] * p[0] + cumulative_trapezoid(y_theta, p_acc)) / p_acc
        # time: from the first valid point (average undefined there)
        avg_eta_time = cumulative_trapezoid(y_eta, t) / (t - t[0])
        avg_theta_time = cumulative_trapezoid(y_theta, t) / (t - t[0])
    avg_eta_time[0] = avg_theta_time[0] = np.nan

    df_out = pd.DataFrame(np.array([p, y_eta, avg_eta, y_theta, avg_theta, t,
                                    avg_eta_time, avg_theta_time]).transpose(), columns=columns)
    return df_out




def calc_average_eta_and_lode(abaqus_report: str,
                               criterion: YieldCriterionPlaneStress = _DEFAULT_CRITERION,
                               repeated_times: str = "error",
                               averaging: str = "elements") -> pd.DataFrame:
    """
    Computes sigma_m, sigma_eq (according to the given yield criterion),
    the stress triaxiality (eta) and the normalized Lode angle (theta),
    plus their cumulative averages with respect to the equivalent plastic
    strain (avg_eta, avg_theta; Bao & Wierzbicki).

    Parameters:
      abaqus_report : path (str or Path) to the .rpt report (Unique
        Nodal export, see read_abaqus_report).
      criterion : already-calibrated YieldCriterionPlaneStress instance
        used to compute sigma_eq -- e.g. VonMises_PS(), Hill48_PS(...),
        YLD91_PS(...) (see yield_criteria_plane_stress.py). Defaults to von Mises.
      repeated_times : "error" (default) raises if the report has several rows
        per instant; "average" accepts them (see read_abaqus_report).
      averaging : what to do with those several rows (one per element sharing the
        node), analogous to Abaqus' "Compute scalars before/after averaging":
        "elements" (default) computes sigma_eq, eta and theta in each element and
        then takes their mean at each instant (before averaging: each element's
        own stress state, evaluated with the criterion in its own axes);
        "nodal" averages the stress components first and computes eta, theta and
        sigma_eq from the mean tensor (after averaging). Only elements that
        have yielded (finite PEEQ > 0) enter the mean. The two differ: eta of the
        mean tensor is not the mean of eta. Ignored if the report has one row
        per instant.

    Returns:
      DataFrame with columns 'peeq', 'eta', 'avg_eta', 'theta', 'avg_theta',
      'time', 'avg_eta_time', 'avg_theta_time', sorted by TIME (X).

    - eta: pointwise stress triaxiality = sigma_m / sigma_eq (sigma_eq
      taken from the given, possibly anisotropic, yield criterion).
    - theta: pointwise normalized Lode angle parameter, in [-1, 1]:
        theta = 1 - (2/pi) * arccos(27 * J3 / (2 * sigma_eq_vm**3))
      computed from the deviatoric invariants J2, J3 of (s1, s2, s3) using
      the isotropic von Mises equivalent stress sigma_eq_vm = sqrt(3*J2).
      Unlike eta, theta is a purely geometric descriptor of the stress
      state (its "shape"), so it does not depend on the given criterion:
        theta = +1 : axisymmetric tension (e.g. uniaxial tension)
        theta =  0 : generalized shear / plane strain
        theta = -1 : axisymmetric compression (e.g. equibiaxial tension)
      eta and theta are both NaN wherever sigma_eq/sigma_eq_vm is zero
      (undefined ratio) or peeq <= 0 (before yielding starts, where the
      reported stresses are elastic/dynamic-contact noise rather than a
      genuine plastic stress state).
    - avg_eta, avg_theta: cumulative averages with respect to the
      equivalent plastic strain,
        avg_eta(peeq) = (1/peeq) * integral_0^peeq eta d(peeq)
      (trapezoidal rule on the PEEQ increments; from 0 to the first valid
      point, eta of that point). This is the average of the ductile
      fracture criteria. Written in time, d(peeq) = (dpeeq/dt) dt, it is a
      time average WEIGHTED with the plastic strain rate: only the stress
      states while the material deforms plastically count.
    - avg_eta_time, avg_theta_time: plain time averages from the first
      valid point, (1/(t - t0)) * integral eta dt -- diagnostic only.
      They equal avg_eta/avg_theta only if the plastic strain rate is
      constant; in SPIF a point deforms plastically only while the tool
      passes near it, so they are dominated by the elastic (unloading)
      periods, which do not contribute to damage.

    The PEEQ-weighted average needs a sampling dense enough to resolve
    eta during each plastic increment: eta can swing through a full cycle
    while peeq barely grows (cyclic local tool contact), and a sparse
    field output may then give a non-converged avg_eta (compare reports
    exported at different frame counts). The remedy is denser output
    (history output) or integrating at every increment (VUSDFLD), not
    changing the averaging measure.
    """
    df = _read_rows(abaqus_report, repeated_times, averaging)
    time, peeq, eta, theta = _pointwise_eta_theta(df, criterion)

    # one row per element: mean over the yielded elements of each instant
    if averaging == "elements" and not pd.Series(time).is_unique:
        ok = np.isfinite(peeq) & np.isfinite(time) & np.isfinite(eta) & np.isfinite(theta)
        if not ok.any():
            return pd.DataFrame(columns=_AVERAGE_COLUMNS)
        time, cols = _mean_by_time(time[ok], peeq=peeq[ok], eta=eta[ok], theta=theta[ok])
        peeq, eta, theta = cols["peeq"], cols["eta"], cols["theta"]

    return _cumulative_averages(time, peeq, eta, theta)


def calc_average_eta_and_lode_by_element(abaqus_report: str,
                                         criterion: YieldCriterionPlaneStress = _DEFAULT_CRITERION,
                                         ) -> dict:
    """
    calc_average_eta_and_lode for EACH element sharing the node, to see how representative their
    mean (averaging="elements") is. Returns {k: DataFrame} for k = 1..n (same columns).

    The report lists, in every instant, one row per element of the node, and the order of those
    rows is stable along the report (checked on the HF-SPIF node 19853 reports: the plastic strain
    of every row is continuous and never decreases), so row k is element k. The reports carry no
    element label, so this is an assumption of the order; a RuntimeWarning is raised if the PEEQ
    of a row decreases in more than 1 % of the steps (the order would not be stable).
    """
    df = read_abaqus_report(abaqus_report, repeated_times="keep")
    time, peeq, eta, theta = _pointwise_eta_theta(df, criterion)
    slot = pd.Series(time).groupby(time).cumcount().to_numpy()      # k-th row of its instant
    out = {}
    for k in range(slot.max() + 1):
        rows = slot == k
        p = peeq[rows]
        steps = np.diff(p[np.isfinite(p)])
        if steps.size and np.mean(steps < -1e-6) > 0.01:
            warnings.warn(f"row {k + 1} of each instant: PEEQ decreases in {100 * np.mean(steps < -1e-6):.1f} % of "
                          f"the steps; the rows are probably not in a stable element order", RuntimeWarning)
        out[k + 1] = _cumulative_averages(time[rows], p, eta[rows], theta[rows])
    return out


def plot_avgeta_vs_peeq(paths, labels=None, peeq_min: float = 0.0, figwidth=width_1column, savefile: str = None):
    """
    Plots average stress triaxiality vs. equivalent plastic strain for a
    list of paths, one line per path, and saves the figure.

    Parameters:
      paths : list of (avg_eta, peeq) array pairs, e.g. the 'avg_eta' and
        'peeq' columns of calc_average_eta_and_lode's returned DataFrame.
      labels : legend label per path; defaults to none (unlabelled lines).
      peeq_min : drops points with peeq < peeq_min before plotting (but the
        underlying data is untouched -- see calc_average_eta_and_lode).
        avg_eta is a running average with respect to PEEQ, so right after
        yielding starts it is still averaging over very little accumulated
        deformation and swings a lot before settling into a representative
        value; raising peeq_min trims that startup transient from the plot
        without discarding it from the data itself. 0.0 (default): no
        trimming.
      figwidth : figure size passed to plt.figure.
      savefile : output path for the figure; the figure is not saved if
        None.
    """
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X']
    if labels is None:
        labels = [None] * len(paths)

    aspect_ratio = 0.6  # height / width
    figsize = (figwidth, figwidth * aspect_ratio)
    plt.figure(figsize=figsize)
    for i, ((avg_eta, peeq), label) in enumerate(zip(paths, labels)):
        avg_eta, peeq = np.asarray(avg_eta, dtype=float), np.asarray(peeq, dtype=float)
        mask = peeq >= peeq_min
        marker = markers[i % len(markers)]
        plt.plot(avg_eta[mask], peeq[mask], marker=marker, linestyle='-.', lw=0.2, ms=1, label=label)

    plt.xlabel(r'Average stress triaxiality $\bar{\eta}$')   # $\sigma_m / \sigma_{eq}$
    plt.ylabel(r'Equivalent plastic strain $\varepsilon_{eq}$')    # $\varepsilon_{eq}$
#    plt.title('Triaxiality vs PEEQ')
#    plt.axis([-0.6, 0.6, 0, 2.5])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile)
        print(f"Figura guardada en '{savefile}'.")


def plot_avgeta_vs_avglode_vs_peeq(paths, labels=None, peeq_min: float = 0.0, figwidth=width_2columns, savefile: str = None):
    """
    2x2 panel comparing average stress triaxiality (avg_eta), average
    normalized Lode angle (avg_theta) and equivalent plastic strain (peeq)
    for a list of paths: the three variables plotted pairwise in 2D, plus
    all three together in a 3D plot.

    Parameters:
      paths : list of (avg_eta, avg_theta, peeq) array triples, e.g. the
        'avg_eta', 'avg_theta' and 'peeq' columns of
        calc_average_eta_and_lode's returned DataFrame.
      labels : legend label per path; defaults to none (unlabelled lines).
      peeq_min : drops points with peeq < peeq_min before plotting -- see
        plot_avgeta_vs_peeq for why (the same running-average startup
        transient affects avg_theta too). 0.0 (default): no trimming.
      figwidth : figure size passed to plt.figure.
      savefile : output path for the figure; if None, the figure is not
        saved.

    Returns:
      fig -- the matplotlib Figure, with 4 subplots:
        (0,0) avg_eta   vs peeq
        (0,1) avg_theta vs peeq
        (1,0) avg_theta vs avg_eta
        (1,1) avg_eta, avg_theta, peeq together, in 3D
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

    markers = ['o', 's', '^', 'v', 'D', 'P', 'X']
    if labels is None:
        labels = [None] * len(paths)

    aspect_ratio = 0.6  # height / width
    figsize = (figwidth, figwidth * aspect_ratio)
    fig = plt.figure(figsize=figsize)
    ax_eta_peeq = fig.add_subplot(2, 2, 1)
    ax_theta_peeq = fig.add_subplot(2, 2, 2)
    ax_theta_eta = fig.add_subplot(2, 2, 3)
    ax_3d = fig.add_subplot(2, 2, 4, projection='3d')

    for i, ((avg_eta, avg_theta, peeq), label) in enumerate(zip(paths, labels)):
        avg_eta = np.asarray(avg_eta, dtype=float)
        avg_theta = np.asarray(avg_theta, dtype=float)
        peeq = np.asarray(peeq, dtype=float)
        mask = peeq >= peeq_min
        avg_eta, avg_theta, peeq = avg_eta[mask], avg_theta[mask], peeq[mask]
        marker = markers[i % len(markers)]
        style = dict(marker=marker, linestyle='-.', lw=0.2, ms=1, label=label)
        ax_eta_peeq.plot(avg_eta, peeq, **style)
        ax_theta_peeq.plot(avg_theta, peeq, **style)
        ax_theta_eta.plot(avg_eta, avg_theta, **style)
        ax_3d.plot(avg_eta, avg_theta, peeq, **style)

    ax_eta_peeq.set_xlabel(r'Average stress triaxiality $\bar{\eta}$')
    ax_eta_peeq.set_ylabel(r'Equivalent plastic strain $\varepsilon_{eq}$')

    ax_theta_peeq.set_xlabel(r'Average Lode angle $\bar{\theta}$')
    ax_theta_peeq.set_ylabel(r'Equivalent plastic strain $\varepsilon_{eq}$')

    ax_theta_eta.set_xlabel(r'Average stress triaxiality $\bar{\eta}$')
    ax_theta_eta.set_ylabel(r'Average Lode angle $\bar{\theta}$')

    ax_3d.set_xlabel(r'Average stress triaxiality $\bar{\eta}$')
    ax_3d.set_ylabel(r'Average Lode angle $\bar{\theta}$')
    ax_3d.set_zlabel(r'Equivalent plastic strain $\varepsilon_{eq}$')

    for ax in (ax_eta_peeq, ax_theta_peeq, ax_theta_eta):
        ax.grid(True, alpha=0.3)
    ax_eta_peeq.legend(fontsize=8)

    fig.tight_layout()
    if savefile is not None:
        fig.savefig(savefile)
        print(f"Figura guardada en '{savefile}'.")

    return fig


def plot_eta_vs_time(report_files, labels=None, zoom_t=(0.0, 2.0), figwidth=width_2columns,
                      criterion: YieldCriterionPlaneStress = _DEFAULT_CRITERION, savefile: str = None,
                      repeated_times: str = "error", averaging: str = "elements"):
    """
    Plots the pointwise triaxiality eta against TIME (X, not PEEQ) for one
    or more Abaqus reports -- a diagnostic to check whether two exports of
    the same simulation (e.g. at different frame counts) trace the same
    underlying eta(time) signal, independently of how avg_eta later
    integrates it (see calc_average_eta_and_lode: eta can oscillate much faster
    than PEEQ accumulates, e.g. under cyclic local tool contact in SPIF,
    which is exactly what makes a PEEQ-integrated average unreliable to
    compare across different frame counts).

    Produces two figures:
      - the full time range;
      - a zoom on zoom_t[0] <= time <= zoom_t[1], at DOUBLE the horizontal
        size (100% wider than the full-range figure) so closely overlapping
        curves can be told apart.

    Parameters:
      report_files : list of paths (str or Path) to the .rpt reports.
      labels : legend label per file; defaults to the file names.
      zoom_t : (tmin, tmax) time range for the zoomed figure.
      figwidth : figure width for the full-range figure; the zoomed one
        uses (2*figwidth, figwidth * 0.6) -- 100% wider.
      criterion : already-calibrated YieldCriterionPlaneStress instance
        used to compute sigma_eq (see sigma_eq_from_yield_criterion).
        Defaults to von Mises.
      savefile : if given, a path prefix (without extension); saves
        '<savefile>_full.png' and '<savefile>_zoom.png'.
      repeated_times, averaging : as in calc_average_eta_and_lode; with
        averaging="elements" the curve is the mean over the elements of each
        instant of their own eta.

    Returns:
      (fig_full, fig_zoom) -- the two matplotlib Figures.
    """
    if labels is None:
        labels = [Path(f).name for f in report_files]

    curves = []
    for report_file in report_files:
        df = _read_rows(report_file, repeated_times, averaging)

        s1 = np.asarray(df['S_Max'], dtype=float)
        s2 = np.asarray(df['S_Mid'], dtype=float)
        s3 = np.asarray(df['S_Min'], dtype=float)
        x = np.asarray(df['X'], dtype=float)

        sigma_m = (s1 + s2 + s3) / 3.0
        sigma_eq = sigma_eq_from_report(df, criterion)
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = sigma_m / sigma_eq

        if averaging == "elements" and not pd.Series(x).is_unique:   # mean of the elements' eta
            ok = np.isfinite(x) & np.isfinite(eta)
            x, cols = _mean_by_time(x[ok], eta=eta[ok])
            eta = cols["eta"]

        order = np.argsort(x, kind='stable')
        curves.append((x[order], eta[order]))

    aspect_ratio = 0.4  # height / width
    figsize = (figwidth, figwidth * aspect_ratio)
    fig_full, ax_full = plt.subplots(figsize=figsize)
    for (x, eta), label in zip(curves, labels):
        ax_full.plot(x, eta, '.-', ms=2, lw=0.5, label=label)
    ax_full.set_xlabel(r'Time $X$')
    ax_full.set_ylabel(r'Triaxiality $\eta$ (pointwise)')
    ax_full.set_title(r'Pointwise triaxiality vs time -- full range')
    ax_full.legend()
    ax_full.grid(True, alpha=0.3)
    fig_full.tight_layout()

    zoom_tmin, zoom_tmax = zoom_t
    fig_zoom, ax_zoom = plt.subplots(figsize=figsize)
    for (x, eta), label in zip(curves, labels):
        mask = (x >= zoom_tmin) & (x <= zoom_tmax)
        ax_zoom.plot(x[mask], eta[mask], '.-', ms=3, lw=0.8, label=label)
    ax_zoom.set_xlabel(r'Time $X$')
    ax_zoom.set_ylabel(r'Triaxiality $\eta$ (pointwise)')
    ax_zoom.set_title(rf'Pointwise triaxiality vs time -- zoom (${zoom_tmin} \leq X \leq {zoom_tmax}$)')
    ax_zoom.legend()
    ax_zoom.grid(True, alpha=0.3)
    fig_zoom.tight_layout()

    if savefile is not None:
        full_path = f"{savefile}_full.png"
        zoom_path = f"{savefile}_zoom.png"
        fig_full.savefig(full_path)
        fig_zoom.savefig(zoom_path)
        print(f"Figuras guardadas en '{full_path}' y '{zoom_path}'.")

    return fig_full, fig_zoom
