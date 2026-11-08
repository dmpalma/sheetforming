"""
yield_functions.py
==================

Hardening models -- isotropic, kinematic, and mixed -- that combine with
a yield_criteria_plane_stress.YieldCriterionPlaneStress into an evolving yield
function:

    f(state) = sigma_eq_criterion(sigma - backstress) - sigma_Y(peeq)

where sigma_eq_criterion(...) is the criterion's own (possibly
anisotropic) equivalent stress (yield_criteria_plane_stress.py -- unchanged: this
module only ever reads it, never redefines the surface's SHAPE),
sigma_Y(peeq) is the isotropic hardening curve (a
constitutive_laws.ConstitutiveLaw -- unchanged, same role as in
material.Material), and backstress is the yield surface's translation
in stress space, evolved here by a new KinematicHardeningLaw.

YieldFunction composes these three ingredients; which of the two
hardening contributions are active is just which of its two law slots
are set (both None: no hardening -- perfectly plastic; isotropic_law
only: pure isotropic hardening; kinematic_law only: pure kinematic
hardening; both: mixed hardening) -- there is no need for separate
IsotropicHardening/KinematicHardening/MixedHardening subclasses.

Scope of this first pass: standalone building blocks (the hardening
laws and YieldFunction itself), not yet wired into material.Material/
MaterialPoint -- that integration (advancing backstress alongside
peeq/eps1/eps2 as a MaterialPoint evolves, and combining with necking/
fracture criteria) is deferred to when those failure criteria are
implemented, per the point-state-evolution design already sketched
there.

Author: Domingo Morales Palma <dmpalma@us.es>
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .yield_criteria_plane_stress import YieldCriterionPlaneStress, PointState, VonMises_PS, get_yield_loci
from .constitutive_laws import ConstitutiveLaw, Swift, TensileData
from .plot_style import width_1column, width_2columns
from .abaqus_utilities import read_abaqus_report


class StrainIncrement(NamedTuple):
    """
    Plastic strain increment of one step of a strain path, in FIXED axes
    (the material axes): the in-plane tensor components deps_xx, deps_yy,
    deps_xy (TENSOR shear, deps_xy = gamma_xy / 2) and the equivalent
    plastic strain increment depsEQ >= 0. depsEQ is stored explicitly
    (rather than re-derived from the components) because for a real path
    it is the PEEQ increment Abaqus already computed.
    """
    deps_xx: float
    deps_yy: float
    deps_xy: float
    depsEQ: float


# Example strain path: a list of StrainIncrement applied step by step.
# Uniaxial tension along axis 1 (deps_yy = -deps_xx/2, isotropic
# plasticity, no shear), so depsEQ = deps_xx exactly at every step.
DEFAULT_PATH = [StrainIncrement(0.02, -0.01, 0.0, 0.02)] * 5

_PATH_COLUMNS = ("PEEQ", "PE11", "PE22", "PE12")


def _check_path_columns(df, abaqus_report):
    missing = [c for c in _PATH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{abaqus_report}: columns {missing} not found (found {list(df.columns)}). The strain "
            f"path needs the plastic strain tensor components in the local axes: export "
            f"PE11, PE22, PE12 and PEEQ.")


def _cut(df, until_time, until_peeq):
    """Rows of df (sorted by time) up to until_time, and up to the first row with PEEQ >= until_peeq (included)."""
    if until_time is not None:
        df = df[df["X"] <= until_time]
    if until_peeq is not None and len(df):
        reached = np.nonzero(df["PEEQ"].to_numpy(float) >= until_peeq)[0]
        if len(reached):
            df = df.iloc[:reached[0] + 1]
    return df


def _increments(df) -> List[StrainIncrement]:
    d = {c: np.diff(df[c].to_numpy(float)) for c in _PATH_COLUMNS}
    return [StrainIncrement(float(dxx), float(dyy), 0.5 * float(dxy), float(dp))
            for dxx, dyy, dxy, dp in zip(d["PE11"], d["PE22"], d["PE12"], d["PEEQ"])]


def _element_frames(abaqus_report):
    """{k: rows of the k-th element of the node} (k from 1), NaN plastic strain (not yielded yet) as 0."""
    import warnings
    df = read_abaqus_report(abaqus_report, repeated_times="keep")
    _check_path_columns(df, abaqus_report)
    df = df.assign(_element=df.groupby("X").cumcount() + 1)
    frames = {}
    for k, rows in df.groupby("_element"):
        rows = rows.copy()
        rows[list(_PATH_COLUMNS)] = rows[list(_PATH_COLUMNS)].fillna(0.0)
        steps = np.diff(rows["PEEQ"].to_numpy(float))
        if steps.size and np.mean(steps < -1e-6) > 0.01:
            warnings.warn(f"row {k} of each instant: PEEQ decreases in {100 * np.mean(steps < -1e-6):.1f} % of the "
                          f"steps; the rows are probably not in a stable element order", RuntimeWarning)
        frames[int(k)] = rows
    return frames


def path_from_abaqus_report(abaqus_report, repeated_times: str = "error", element: Optional[int] = None,
                            until_time: Optional[float] = None, until_peeq: Optional[float] = None
                            ) -> List[StrainIncrement]:
    """
    Strain path (list of StrainIncrement, see DEFAULT_PATH) from a real or
    simulated plastic strain history of ONE node, read from an Abaqus
    report with abaqus_utilities.read_abaqus_report.

    The report must contain the plastic strain TENSOR components in the
    element's local axes (Abaqus variables PE11, PE22, PE12 and PEEQ). Those
    axes start as the material axes when a local *ORIENTATION is defined and,
    without it, as the default ones (shells: the projection of the global X
    axis on the surface; solids: the global axes). In both cases Abaqus
    reports the components in a COROTATIONAL basis that rotates with the
    element's rigid-body rotation: this is the frame in which Abaqus itself
    integrates the stress and strain tensors, and the right one for a
    backstress (a tensor that must not rotate with respect to the material;
    the principal strains do rotate and cannot drive it). The path is only
    fixed with respect to the orthotropy axes (RD, TD) if the basis is
    aligned with them at the start, which matters for an anisotropic yield
    criterion and not for von Mises. Abaqus reports PE12 as the ENGINEERING
    shear strain (gamma_12), so deps_xy = dPE12 / 2 here. A report that only
    has principal values raises ValueError: re-export it with PE11, PE22, PE12
    and PEEQ.

    Parameters:
      abaqus_report : path of the (.rpt) file.
      repeated_times : passed to read_abaqus_report when element is None
        ("error": one row per instant, Unique Nodal; "average": the rows of an
        instant are averaged, i.e. the path of the MEAN tensor of the node).
      element : None (default) for the path of the node as the report gives it,
        or k = 1, 2, ... for the path of the k-th element of the node. The report
        must then list one row per element and instant (Abaqus "Not averaged",
        or an Averaging threshold that leaves the element values apart). The rows
        of an instant keep the same element order along the report (checked on
        the HF-SPIF and clover reports; a RuntimeWarning is raised if the PEEQ of
        a row decreases in more than 1 % of the steps), so row k is element k.
        A plastic strain that Abaqus does not write yet (the element has not
        yielded) counts as 0. The mean tensor of the elements of a node is NOT the
        path of any of them, and a kinematic law that depends on the path
        (Armstrong-Frederick) gives different results.
      until_time : stop the path at this time X (the last instant used is the
        last one with X <= until_time), e.g. the instant of the experimental
        fracture: what a simulation without damage computes afterwards is not
        physical.
      until_peeq : stop the path at the first instant with PEEQ >= until_peeq
        (that instant is included). If both limits are given the shorter path wins.

    depsEQ is the PEEQ increment computed by Abaqus, so it is exactly 0
    across an elastic segment (e.g. while the SPIF tool moves away),
    whatever the strain components do there.
    """
    if element is not None:
        frames = _element_frames(abaqus_report)
        if element not in frames:
            raise ValueError(f"{abaqus_report}: element {element} not found; the report lists "
                             f"{max(frames)} row(s) per instant at most")
        df = frames[element]
    else:
        df = read_abaqus_report(abaqus_report, repeated_times=repeated_times)
        _check_path_columns(df, abaqus_report)
    return _increments(_cut(df, until_time, until_peeq))


def paths_by_element(abaqus_report, until_time: Optional[float] = None,
                     until_peeq: Optional[float] = None) -> Dict[int, List[StrainIncrement]]:
    """
    {k: path_from_abaqus_report(abaqus_report, element=k, ...)} for every element of the node (k from 1),
    reading the report once. See path_from_abaqus_report for the element order and the cut.
    """
    return {k: _increments(_cut(df, until_time, until_peeq)) for k, df in _element_frames(abaqus_report).items()}


def downsample_path(path, n_points: int):
    """
    Collapses `path` to about n_points steps by SUMMING every component
    (deps_xx, deps_yy, deps_xy, depsEQ) within each block, rather than
    discarding the skipped steps (path[::stride] would). A real report has
    hundreds of near-zero rows (nothing moves between two Abaqus frames
    while the SPIF tool is elsewhere) alongside the few rows where it's
    actually in contact -- simple striding overwhelmingly picks the
    near-zero rows, both massively undercounting the true accumulated
    peeq (e.g. 1.46 collapsed to 0.01 over a 731-step SPIF report
    downsampled to 6 points this way) AND making the surviving
    deps_yy/deps_xx ratio (beta, see beta_from_path) numerically
    unstable (dividing by whatever near-zero deps_xx a lone kept row
    happens to have). Summing each block first fixes both: the total
    peeq is preserved exactly, and beta reflects the NET direction
    accumulated over that block instead of one arbitrary noisy row.
    """
    n = len(path)
    block = max(1, -(-n // n_points))   # ceil: at most n_points steps
    out = []
    for i in range(0, n, block):
        chunk = path[i:i + block]
        out.append(StrainIncrement(*(sum(c[k] for c in chunk) for k in range(4))))
    return out


def solve_state_point(criterion: YieldCriterionPlaneStress, sigmaY: float, beta: float) -> Tuple[float, float]:
    """
    (sigma1, sigma2) on criterion's locus (radius sigmaY) whose
    associated flow direction matches the given beta = deps2/deps1 --
    the general form of "the uniaxial-tension reference point (sigmaY,
    0)" used throughout this module's demos. Only valid for
    DEFAULT_PATH's constant beta = -0.5 (von Mises' own uniaxial-
    tension direction) does that hardcoded point coincide with the
    true state; for any other (e.g. a real, generally non-uniaxial)
    path, the actual state must be solved for via the criterion's own
    alpha(beta) relation, exactly like get_alpha_vs_beta does.
    """
    state = criterion.solve(beta=beta, sigmaEQ=sigmaY)
    return state.sigma1, state.sigma2


def beta_from_path(path) -> List[float]:
    """
    beta = deps_yy/deps_xx driving each step of a strain path (list of
    StrainIncrement, e.g. DEFAULT_PATH or path_from_abaqus_report),
    one value per state of the path, len(path) + 1 (including the
    INITIAL one, which reuses path's first step -- there being no
    earlier increment to derive a direction from). Falls back to the
    previous step's beta where deps_xx == 0 (e.g. an elastic segment in
    a real path -- no direction is actually imposed there, so it does
    not matter which beta is carried through).
    """
    betas = []
    last = -0.5
    for deps_xx, deps_yy, _, _ in path:
        last = (deps_yy / deps_xx) if deps_xx != 0 else last
        betas.append(last)
    return [betas[0]] + betas


@dataclass
class KinematicHardeningLaw(ABC):
    """
    Generic interface for a kinematic hardening rule: how the backstress
    (the yield surface's centre in stress space) evolves with the
    plastic strain increment.

    Deliberately independent of PointState (like ConstitutiveLaw.
    stress_from_strain, which takes a plain peeq rather than a whole
    state): backstress_increment takes the plastic strain increment
    TENSOR components explicitly (deps_xx, deps_yy, deps_xy -- the
    general, fixed-frame components, not deps1/deps2 in the stress
    principal axes, since those axes can rotate step to step and are
    generally NOT the true strain-increment principal axes either, see
    yield_criteria_plane_stress.calc_psi_noncoax_from_theta_stress) plus the CURRENT
    backstress (needed by saturating laws, e.g. Armstrong-Frederick).

    NOTE: only the in-plane components (xx, yy, xy) are tracked -- there
    is no backstress_zz, i.e. X3 is implicitly assumed to be 0. Unlike
    sigma3 = 0 (an actual physical boundary condition of plane stress,
    see yield_criteria_plane_stress.PointState), this is a simplification: a truly
    3D deviatoric backstress evolving under the same proportional law
    would generally pick up a nonzero X3 = -(X_xx + X_yy), since deps3
    = -(deps_xx + deps_yy) is itself generally nonzero (plastic
    incompressibility). Keeping X3 = 0 is what lets the SHIFTED state
    (sigma - X) stay a valid plane-stress state too, so the existing
    plane-stress sigma_eq formulas (VonMises_PS, Hill48_PS, ...) can still be
    reused unchanged for it -- a fully rigorous treatment would need a
    3D yield criterion instead.
    """

    @abstractmethod
    def backstress_increment(
        self,
        deps_xx: float, deps_yy: float, deps_xy: float,
        backstress_xx: float = 0.0, backstress_yy: float = 0.0, backstress_xy: float = 0.0,
        depsEQ: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Returns (dbackstress_xx, dbackstress_yy, dbackstress_xy) for this step."""


@dataclass
class Prager(KinematicHardeningLaw):
    """
    Linear kinematic hardening (Prager, 1956):

        dbackstress = C * deps_p

    proportional to the plastic strain increment tensor, with no
    saturation -- the backstress can grow without bound under
    monotonic loading. Simplest kinematic model; a reasonable first
    approximation, but does not reproduce the saturating Bauschinger
    effect seen in real reversed-loading tests as well as
    Armstrong-Frederick does.
    """
    C: float = 0.0

    def backstress_increment(
        self,
        deps_xx: float, deps_yy: float, deps_xy: float,
        backstress_xx: float = 0.0, backstress_yy: float = 0.0, backstress_xy: float = 0.0,
        depsEQ: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        return self.C * deps_xx, self.C * deps_yy, self.C * deps_xy


@dataclass
class ArmstrongFrederick(KinematicHardeningLaw):
    """
    Nonlinear kinematic hardening (Armstrong & Frederick, 1966):

        dbackstress = C * deps_p - gamma * backstress * dp

    where dp = depsEQ (equivalent plastic strain increment, >= 0). The
    recovery term (-gamma * backstress * dp) makes the backstress
    saturate towards a limiting magnitude of C / gamma under monotonic
    loading, rather than growing without bound as in Prager's law --
    this is what lets it reproduce the Bauschinger effect (rapid
    re-yielding on load reversal, followed by a gradual approach to a
    new, lower-than-monotonic flow stress) seen in real reversed and
    cyclic loading, which matters for path reversals in incremental
    forming (SPIF) and for springback prediction.

    depsEQ is required (unlike Prager, which does not use it) --
    without it, the recovery term has no rate to scale by.
    """
    C: float = 0.0
    gamma: float = 0.0

    def backstress_increment(
        self,
        deps_xx: float, deps_yy: float, deps_xy: float,
        backstress_xx: float = 0.0, backstress_yy: float = 0.0, backstress_xy: float = 0.0,
        depsEQ: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        dp = depsEQ if depsEQ is not None else 0.0
        recovery = self.gamma * dp
        return (
            self.C * deps_xx - recovery * backstress_xx,
            self.C * deps_yy - recovery * backstress_yy,
            self.C * deps_xy - recovery * backstress_xy,
        )


@dataclass
class YieldFunction:
    """
    Combines a yield criterion (SHAPE) with isotropic (SIZE) and/or
    kinematic (TRANSLATION) hardening into the actual evolving yield
    function f(state) = sigma_eq(sigma - backstress) - sigma_Y(peeq).

    Parameters
    ----------
    criterion : YieldCriterionPlaneStress
        Already-calibrated criterion (yield_criteria_plane_stress.py) supplying the
        surface's shape -- VonMises_PS, Hill48_PS, Hill79_PS, YLD91_PS.
    isotropic_law : ConstitutiveLaw, optional
        Hardening curve sigma_Y(peeq) (constitutive_laws.py -- Hollomon,
        Swift, Voce, Voce2, VoceSwift). None means no isotropic
        hardening (sigma_Y taken as 0, i.e. calc_yield_stress returns
        None -- there is no separate "initial yield stress" field here:
        a real law's stress_from_strain(0) already gives it).
    kinematic_law : KinematicHardeningLaw, optional
        Backstress evolution rule (Prager, ArmstrongFrederick). None
        means no kinematic hardening (backstress stays at 0).
    """

    criterion: YieldCriterionPlaneStress
    isotropic_law: Optional[ConstitutiveLaw] = None
    kinematic_law: Optional[KinematicHardeningLaw] = None

    def calc_yield_stress(self, state: PointState) -> Optional[float]:
        """Current size of the yield surface, sigma_Y(peeq), from the isotropic hardening curve."""
        if self.isotropic_law is None or state.peeq is None:
            return None
        return self.isotropic_law.stress_from_strain(state.peeq)

    def calc_shifted_equivalent_stress(self, state: PointState) -> Optional[float]:
        """
        Equivalent stress relative to the current backstress (the
        yield surface's shifted centre): the criterion's own
        calc_equivalent_stress, evaluated at (sigma - backstress)
        rather than at the raw stress state. Requires sigma_xx, sigma_yy
        (tau_xy defaults to 0); backstress_xx/yy/xy default to 0 (no
        kinematic hardening yet), so this also works for a purely
        isotropic (or unhardened) yield function.
        """
        if state.sigma_xx is None or state.sigma_yy is None:
            return None
        bxx = state.backstress_xx if state.backstress_xx is not None else 0.0
        byy = state.backstress_yy if state.backstress_yy is not None else 0.0
        bxy = state.backstress_xy if state.backstress_xy is not None else 0.0
        tau_xy = state.tau_xy if state.tau_xy is not None else 0.0

        shifted = self.criterion.solve(
            sigma_xx=state.sigma_xx - bxx,
            sigma_yy=state.sigma_yy - byy,
            tau_xy=tau_xy - bxy,
        )
        return shifted.sigmaEQ

    def evaluate(self, state: PointState) -> Optional[float]:
        """
        Yield function value f(state) = sigma_eq(sigma - backstress) -
        sigma_Y(peeq): negative inside the current surface (elastic),
        zero on it (yielding), positive outside (only transiently valid
        mid-increment, before a return-mapping step brings it back to
        zero). None if either ingredient cannot be evaluated (e.g. no
        isotropic_law and no peeq yet).
        """
        sigma_eq = self.calc_shifted_equivalent_stress(state)
        sigma_y = self.calc_yield_stress(state)
        if sigma_eq is None or sigma_y is None:
            return None
        return sigma_eq - sigma_y

    def calc_backstress_increment(
        self, state: PointState, deps_xx: float, deps_yy: float, deps_xy: float,
    ) -> Optional[Tuple[float, float, float]]:
        """
        (dbackstress_xx, dbackstress_yy, dbackstress_xy) for this step,
        from state's CURRENT backstress and depsEQ plus the given
        plastic strain increment tensor (see KinematicHardeningLaw --
        deps_xx/yy/xy are passed in explicitly rather than read off
        state, since PointState does not yet carry a general,
        fixed-frame plastic strain increment field; only the per-step
        principal deps1/deps2, in the stress-aligned axes). None if no
        kinematic_law is set.
        """
        if self.kinematic_law is None:
            return None
        bxx = state.backstress_xx if state.backstress_xx is not None else 0.0
        byy = state.backstress_yy if state.backstress_yy is not None else 0.0
        bxy = state.backstress_xy if state.backstress_xy is not None else 0.0
        return self.kinematic_law.backstress_increment(
            deps_xx, deps_yy, deps_xy,
            backstress_xx=bxx, backstress_yy=byy, backstress_xy=bxy,
            depsEQ=state.depsEQ,
        )
