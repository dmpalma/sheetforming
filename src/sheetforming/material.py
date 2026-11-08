"""
material.py
===========

`Material`: the central object representing one real sheet material at a
point, composing:

  - a yield criterion (yield_criteria_plane_stress.py -- VonMises_PS, Hill48_PS, Hill79_PS,
    YLD91_PS), for anisotropic yielding, and
  - a hardening law (constitutive_laws.py -- Hollomon, Swift, Voce,
    Voce2, VoceSwift), for the flow-stress-vs-plastic-strain relation,

and exposing "point behaviour" methods that delegate to them, instead of
scripts wiring a criterion and a law together by hand each time.

`yield_criterion` is calibrated ONCE (see __post_init__) and then reused
for every evaluation via its own non-mutating `solve()` (see
yield_criteria_plane_stress.YieldCriterionPlaneStress.solve): each call returns a
fresh PointState without touching the calibration held here.

Necking and ductile-fracture criteria are NOT implemented yet --
`necking_criterion`/`fracture_criterion` are left as optional, unused
extension points for now. triaxiality_and_lode() already covers the
stress-state descriptors (triaxiality, Lode angle) those future
criteria would consume; see triaxiality.py for the same computation
applied along a whole simulated/experimental strain path rather than a
single point.

This module also holds `MaterialPoint` (PointState itself lives in
yield_criteria_plane_stress.py, imported from there): tracking the evolving
stress/strain state of ONE physical point at a Material -- a
node/integration point extracted from an Abaqus simulation, a point
measured in a Nakajima/deep-drawing/SPIF test (e.g. via DIC), or a
synthetic loading path -- as a HISTORY of resolved snapshots, so that
necking/ductile-fracture criteria can eventually be evaluated over the
WHOLE path, not just its last point (the same reason triaxiality.py's
avg_eta is a cumulative quantity, integrated over the plastic strain,
not just the final eta).

Author: Domingo Morales Palma <dmpalma@us.es>
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
from math import acos, pi, sqrt
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from .yield_criteria_plane_stress import YieldCriterionPlaneStress, PointState, YLD91_PS
from .constitutive_laws import ConstitutiveLaw, TensileData, Swift


@dataclass
class Material:
    """
    Parameters
    ----------
    name : str
        Material/batch identifier, e.g. 'AA7075-O (batch 2026)'.
    yield_criterion : YieldCriterionPlaneStress
        Already-calibrated criterion instance (any subclass -- VonMises_PS,
        Hill48_PS, Hill79_PS, YLD91_PS). Only ONE per Material: __post_init__
        calibrates it once (see _ensure_calibration) and every subsequent
        evaluation reuses that same instance via its own solve(), never
        rebuilding or refitting it.
    hardening_law : ConstitutiveLaw, optional
        Already-fitted hardening law instance (Hollomon, Swift, Voce,
        Voce2, VoceSwift). None if not (yet) characterised.
    E, nu : float, optional
        Elastic constants (Young's modulus, Poisson's ratio) -- not used
        by any method here yet, but every point-behaviour model this
        class will eventually host (elastic unloading, springback, ...)
        needs them, so they belong on Material rather than being
        threaded through separately.
    necking_criterion, fracture_criterion : optional
        Not implemented yet. Reserved slots for a future forming-limit
        (necking) model and a ductile-fracture model (e.g. built on
        triaxiality_and_lode() below, analogous to what triaxiality.py
        already computes from Abaqus/experimental strain paths).
    """

    name: str
    yield_criterion: YieldCriterionPlaneStress
    hardening_law: Optional[ConstitutiveLaw] = None
    E: Optional[float] = None
    nu: Optional[float] = None
    necking_criterion: Optional[object] = None
    fracture_criterion: Optional[object] = None

    def __post_init__(self) -> None:
        # Calibrate once, eagerly, so it is cached on this persistent
        # instance rather than being refit from scratch on every future
        # solve() call (each of which works on its own throwaway copy --
        # see YieldCriterionPlaneStress.solve and _ensure_calibration).
        self.yield_criterion._ensure_calibration()

    # ------------------------------------------------------------------
    # Point behaviour: yielding
    # ------------------------------------------------------------------

    def equivalent_stress(self, sigma1: float, sigma2: float, sigma3: float = 0.0,
                          theta_stress: Optional[float] = None) -> Optional[float]:
        """
        Equivalent stress for a general triaxial state (sigma1, sigma2,
        sigma3), via this material's yield criterion, with sigma3 the
        principal value along ND and sigma1, sigma2 the in-plane ones,
        sigma1 at theta_stress (radians) from RD (None = along RD). Shifted
        to plane stress internally (subtracting sigma3), valid because
        every criterion in yield_criteria_plane_stress.py is insensitive
        to hydrostatic pressure. theta_stress only matters for
        anisotropic criteria, for which leaving it None is exact only
        when sigma1 is along RD -- see triaxiality.sigma_eq_from_report.
        """
        resolved = self.yield_criterion.solve(sigma1=sigma1 - sigma3, sigma2=sigma2 - sigma3,
                                              theta_stress=theta_stress)
        return resolved.sigmaEQ

    def triaxiality_and_lode(self, sigma1: float, sigma2: float, sigma3: float = 0.0,
                              theta_stress: Optional[float] = None
                              ) -> Tuple[Optional[float], Optional[float]]:
        """
        (eta, theta_bar) for a general triaxial state:
          - eta = sigma_m / sigma_eq: stress triaxiality, using THIS
            material's own (possibly anisotropic) equivalent_stress.
          - theta_bar: normalised Lode angle in [-1, 1], from the
            deviatoric stress invariants alone (von Mises-based), same
            formula as triaxiality.compute_triaxiality_and_lode -- it is
            a purely geometric descriptor of the stress state, so it
            does NOT depend on the yield criterion (see that function's
            docstring).
        theta_stress: orientation of sigma1 from RD, see equivalent_stress.
        """
        sigma_m = (sigma1 + sigma2 + sigma3) / 3.0
        sigma_eq = self.equivalent_stress(sigma1, sigma2, sigma3, theta_stress)
        if sigma_eq is None or sigma_eq == 0:
            return None, None
        eta = sigma_m / sigma_eq

        d1, d2, d3 = sigma1 - sigma_m, sigma2 - sigma_m, sigma3 - sigma_m
        J2 = 0.5 * (d1 ** 2 + d2 ** 2 + d3 ** 2)
        J3 = d1 * d2 * d3
        sigma_eq_vm = sqrt(3.0 * J2)
        if sigma_eq_vm == 0:
            return eta, None
        cos3theta = max(-1.0, min(1.0, 27.0 * J3 / (2.0 * sigma_eq_vm ** 3)))
        theta_bar = 1.0 - (2.0 / pi) * acos(cos3theta)
        return eta, theta_bar

    # ------------------------------------------------------------------
    # Point behaviour: hardening
    # ------------------------------------------------------------------

    def flow_stress(self, plastic_strain: float) -> Optional[float]:
        """Flow stress at the given cumulative plastic strain, via hardening_law. None if hardening_law isn't set."""
        if self.hardening_law is None:
            return None
        return self.hardening_law.stress_from_strain(plastic_strain)


@dataclass
class MaterialPoint:
    """
    Parameters
    ----------
    material : Material
        Supplies the (already-calibrated) yield_criterion used to
        resolve each new state, and the hardening_law against which
        peeq can later be checked (material.flow_stress(peeq)).
    history : list[PointState]
        Resolved/loaded states in order. Empty for a freshly created
        point; grows by one each time solve() or append() is called.
    """
    material: Material
    history: List[PointState] = field(default_factory=list)

    @property
    def current(self) -> Optional[PointState]:
        """The most recent state, or None if history is still empty."""
        return self.history[-1] if self.history else None

    def append(self, state: PointState) -> PointState:
        """
        Appends an already-fully-known state as-is -- no resolving, no
        accumulation. For bulk-loading externally measured/simulated
        data (e.g. one row of an Abaqus .rpt, already averaged by time
        via abaqus_utilities.read_abaqus_report) where peeq,
        eps1, eps2, ... are direct data, not something to integrate
        ourselves.
        """
        self.history.append(state)
        return state

    def solve(self, max_iter: int = 15, time: Optional[float] = None, **known) -> PointState:
        """
        Resolves a new stress state from `known` (any subset of the
        stress-side fields yield_criteria_plane_stress.YieldCriterionPlaneStress.solve()
        understands -- sigma_xx/sigma_yy/tau_xy, sigma1/sigma2, sigmaEQ,
        alpha, beta, theta_stress, or a hardening law's depsEQ), appends
        it to history, and returns it.

        Integrates the strain-side cumulative fields (peeq, eps1, eps2)
        from the PREVIOUS state (0 if this is the first) plus this
        step's resolved deps1/deps2/depsEQ -- i.e. each solve() call
        advances the point by one increment, the way an FE solver
        advances one integration point by one time step. eps_xx, eps_yy,
        eps_xy and theta_eps are NOT derived here (that needs eps1/eps2
        rotated through theta_stress, not yet implemented -- see the
        module docstring); set them on the returned state directly if
        you have them (e.g. from Abaqus/DIC), or leave them None.
        """
        state = self.material.yield_criterion.solve(max_iter=max_iter, **known)
        state.time = time

        previous = self.current
        prev_peeq = previous.peeq if previous and previous.peeq is not None else 0.0
        prev_eps1 = previous.eps1 if previous and previous.eps1 is not None else 0.0
        prev_eps2 = previous.eps2 if previous and previous.eps2 is not None else 0.0
        state.peeq = prev_peeq + (state.depsEQ or 0.0)
        state.eps1 = prev_eps1 + (state.deps1 or 0.0)
        state.eps2 = prev_eps2 + (state.deps2 or 0.0)

        self.history.append(state)
        return state

    def to_frame(self) -> pd.DataFrame:
        """
        The whole history as a pandas DataFrame (one row per state, one
        column per PointState field) -- for interop with triaxiality.py
        (which already works with Abaqus report data in this form) and
        for plotting/exporting.
        """
        columns = [f.name for f in dataclass_fields(PointState)]
        return pd.DataFrame([{c: getattr(s, c) for c in columns} for s in self.history], columns=columns)

    @classmethod
    def from_states(cls, material: Material, states: Iterable[PointState]) -> "MaterialPoint":
        """Builds a MaterialPoint whose history is exactly the given, already-resolved states (see append())."""
        point = cls(material=material)
        for state in states:
            point.append(state)
        return point
