"""
yield_criteria_plane_stress.py
===========================

Module for analysing the plastic behaviour of metals using different
yield criteria (Von Mises, Hill 1948, Hill 1979, Barlat 1991 and
Barlat et al. 2003, Yld2000-2d).
For general 3D stress states (e.g. Abaqus reports from solid-element
simulations) see yield_criteria_3d.py.
All classes inherit from the generic `YieldCriterionPlaneStress` class, which
assumes plane stress throughout (as is appropriate for sheet metal).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, replace
from math import sqrt
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .plot_style import width_1column, width_column_and_a_half, width_2columns






@dataclass
class PointState:
    """
    One resolved instant of stress/strain state at a point (e.g. a
    node/integration point from an Abaqus simulation, a point measured
    in a Nakajima/deep-drawing/SPIF test, or a synthetic loading path).

    Every yield criterion in this module is CALIBRATION ONLY (see
    YieldCriterionPlaneStress) -- it holds none of this itself. A state
    is always a separate, explicit PointState, produced by solve() (one
    point) or accumulated externally over many points (material.MaterialPoint,
    which imports this same class rather than defining its own).

    theta_eps and psi_noncoax are related but kept independent: under
    the associated flow rule, psi_noncoax = theta_eps - theta_stress
    (see calc_psi_noncoax_from_theta_stress) -- but when theta_eps comes
    from an independent source (DIC, Abaqus), storing it separately lets
    the model's own psi_noncoax prediction be checked against the
    measured one, rather than always deriving one from the other.
    """
    time: Optional[float] = None

    # --- stress state ---
    sigma_xx: Optional[float] = None
    sigma_yy: Optional[float] = None
    tau_xy: Optional[float] = None
    sigma1: Optional[float] = None
    sigma2: Optional[float] = None
    theta_stress: Optional[float] = None
    sigmaEQ: Optional[float] = None
    alpha: Optional[float] = None

    # --- strain state (cumulative) ---
    eps_xx: Optional[float] = None
    eps_yy: Optional[float] = None
    eps_xy: Optional[float] = None
    eps1: Optional[float] = None
    eps2: Optional[float] = None
    theta_eps: Optional[float] = None
    peeq: Optional[float] = None

    # --- kinematic hardening state (backstress, cumulative) ---
    # Shifts the yield surface's centre in stress space (Bauschinger
    # effect); tracked in the general (fixed) frame like sigma_xx/yy/tau_xy,
    # not in the (possibly rotating) stress principal axes, so it stays
    # consistent across a loading path whose principal directions rotate
    # (e.g. SPIF). Evolved by a yield_function.KinematicHardeningLaw.
    backstress_xx: Optional[float] = None
    backstress_yy: Optional[float] = None
    backstress_xy: Optional[float] = None

    # --- plastic strain increments (this step only, not cumulative) ---
    # beta and psi_noncoax relate stress to strain-INCREMENT direction
    # (beta = deps2/deps1; psi_noncoax comes from the flow rule's
    # increment direction, not the cumulative strain), so they belong
    # here rather than in the cumulative strain-state group above.
    beta: Optional[float] = None
    psi_noncoax: Optional[float] = None
    deps1: Optional[float] = None
    deps2: Optional[float] = None
    deps3: Optional[float] = None
    depsEQ: Optional[float] = None


@dataclass
class YieldCriterionPlaneStress(ABC):
    """Generic base class for metal yield criteria, under plane stress
    (sigma3 = 0) throughout -- the state of a sheet metal loaded in its own
    plane, which is what this module is concerned with.

    CALIBRATION ONLY: no stress/strain state fields live here (see
    PointState) -- concrete subclasses add only calibration fields
    (Hill's r0/r45/r90/F/G/H/N; Barlat's a/C1..C6/sigma0/sigma45/sigma90;
    ...). Every calc_* method takes the state it operates on as an
    explicit `state: PointState` argument and reads state.X, so the same
    calibrated instance can be reused across any number of evaluations
    (e.g. a Material's persistent yield_criterion) without one
    evaluation's state ever leaking into another's, or into the
    criterion itself.

    label: optional name for legends and tables (see name), e.g. to tell
    apart several calibrations of the same criterion; keyword-only.
    """
    label: Optional[str] = field(default=None, kw_only=True)

    # ------------------------------------------------------------------
    # Generic calculation methods (basic kinematic and kinetic relations)
    # ------------------------------------------------------------------

    def calc_alpha_from_stresses(self, state: PointState) -> Optional[float]:
        if state.sigma1 not in (None, 0) and state.sigma2 is not None:
            return state.sigma2 / state.sigma1
        return None

    def calc_sigma2_from_alpha(self, state: PointState) -> Optional[float]:
        if state.sigma1 is not None and state.alpha is not None:
            return state.alpha * state.sigma1
        return None

    def calc_sigma1_from_sigma2_alpha(self, state: PointState) -> Optional[float]:
        if state.sigma2 is not None and state.alpha not in (None, 0):
            return state.sigma2 / state.alpha
        return None

    def calc_principal_stresses_from_xy(self, state: PointState) -> Optional[Tuple[float, float, float]]:
        """
        Principal stresses (sigma1 >= sigma2) and the angle theta_stress (in
        radians) from the x axis to principal direction 1, for the general
        in-plane state (sigma_xx, sigma_yy, tau_xy), via Mohr's circle:

            sigma1, sigma2 = (sigma_xx + sigma_yy)/2
                             +- sqrt(((sigma_xx - sigma_yy)/2)^2 + tau_xy^2)
            theta_stress = 0.5 * atan2(2*tau_xy, sigma_xx - sigma_yy)

        tau_xy = None is treated as 0 (sigma_xx, sigma_yy already principal).
        """
        if None not in (state.sigma_xx, state.sigma_yy):
            tau_xy = state.tau_xy if state.tau_xy is not None else 0.0
            center = (state.sigma_xx + state.sigma_yy) / 2.0
            radius = sqrt(((state.sigma_xx - state.sigma_yy) / 2.0) ** 2 + tau_xy ** 2)
            theta_stress = 0.5 * float(np.arctan2(2.0 * tau_xy, state.sigma_xx - state.sigma_yy))
            return center + radius, center - radius, theta_stress
        return None

    # ------------------------------------------------------------------
    # Material-axes (component) evaluation
    #
    # Anisotropic criteria depend on the ORIENTATION of the stress state
    # relative to the material axes (1 = RD, 2 = TD), not only on its
    # principal values: they must be evaluated on the components
    # (sigma_xx, sigma_yy, tau_xy) in those axes. A state given only by
    # (sigma1, sigma2) is placed at angle theta_stress from RD (0 if
    # unset, i.e. sigma1 along RD -- the convention every closed-form
    # alpha/beta relation in this module assumes).
    #
    # Off-axis (theta_stress != 0), deps1 and deps2 are the NORMAL
    # components of the plastic strain increment along the sigma1 and
    # sigma2 directions (the stress principal axes), so beta = deps2/deps1
    # is still a ratio in those axes and the work conjugacy
    # sigma1*deps1 + sigma2*deps2 = sigmaEQ*depsEQ stays exact; the
    # strain increment then also has a shear component in those axes,
    # whose effect on the principal strain directions is psi_noncoax.
    # ------------------------------------------------------------------
    @staticmethod
    def _is_rd_aligned(theta_stress: Optional[float]) -> bool:
        """True if sigma1 lies along RD (theta_stress unset, or 0 modulo pi)."""
        return theta_stress is None or abs(np.sin(theta_stress)) < 1e-12

    @staticmethod
    def _components_from_principal(sigma1: float, sigma2: float,
                                   theta_stress: float) -> Tuple[float, float, float]:
        """(sigma_xx, sigma_yy, tau_xy) in material axes of (sigma1, sigma2) at theta_stress from RD."""
        c, s = np.cos(theta_stress), np.sin(theta_stress)
        return (float(sigma1 * c * c + sigma2 * s * s),
                float(sigma1 * s * s + sigma2 * c * c),
                float((sigma1 - sigma2) * s * c))

    def _stress_components(self, state: PointState) -> Optional[Tuple[float, float, float]]:
        """
        (sigma_xx, sigma_yy, tau_xy) in material axes: taken directly from
        state if given, otherwise rebuilt from (sigma1, sigma2,
        theta_stress), with theta_stress = 0 if unset.
        """
        if None not in (state.sigma_xx, state.sigma_yy):
            return state.sigma_xx, state.sigma_yy, (state.tau_xy if state.tau_xy is not None else 0.0)
        if None not in (state.sigma1, state.sigma2):
            theta = state.theta_stress if state.theta_stress is not None else 0.0
            return self._components_from_principal(state.sigma1, state.sigma2, theta)
        return None

    def _equivalent_stress_xy(self, sigma_xx: float, sigma_yy: float, tau_xy: float) -> Optional[float]:
        """Equivalent stress from material-axes components. Implemented by each subclass."""
        return None

    def _flow_gradient_xy(self, sigma_xx: float, sigma_yy: float,
                          tau_xy: float) -> Optional[Tuple[float, float, float]]:
        """
        Gradient (gx, gy, gxy) of sigmaEQ w.r.t. (sigma_xx, sigma_yy, tau_xy),
        up to a positive factor. Implemented by each subclass.
        """
        return None

    def _flow_in_stress_axes(self, alpha: float, theta_stress: float) -> Optional[Tuple[float, float]]:
        """
        (g1, g2) = d(sigmaEQ)/d(sigma1, sigma2) at the unit state (1, alpha)
        placed at theta_stress from RD, i.e. the normal components of the
        associated flow direction along the stress principal axes, scaled
        so that g1 + alpha*g2 = sigmaEQ(1, alpha) (Euler's theorem; the
        exact gradient, not just its direction).
        """
        sxx, syy, txy = self._components_from_principal(1.0, alpha, theta_stress)
        grad = self._flow_gradient_xy(sxx, syy, txy)
        seq = self._equivalent_stress_xy(sxx, syy, txy)
        if grad is None or seq is None or seq == 0:
            return None
        gx, gy, gxy = grad
        c, s = np.cos(theta_stress), np.sin(theta_stress)
        g1 = gx * c * c + gy * s * s + gxy * s * c
        g2 = gx * s * s + gy * c * c - gxy * s * c
        euler = g1 + alpha * g2
        if euler == 0:
            return None
        k = seq / euler
        return float(k * g1), float(k * g2)

    def _beta_from_alpha_theta(self, alpha: float, theta_stress: float) -> Optional[float]:
        """beta = deps2/deps1 (stress principal axes) for stress ratio alpha at theta_stress."""
        g = self._flow_in_stress_axes(alpha, theta_stress)
        if g is None or g[0] == 0:
            return None
        return g[1] / g[0]

    def _alpha_from_beta_theta(self, beta: float, theta_stress: float) -> Optional[float]:
        """
        Inverse of _beta_from_alpha_theta, by bisection on the flow-
        direction angle atan2(g2, g1) rather than on beta itself (which
        blows up where g1 = 0). At fixed theta_stress, (g1, g2) is the
        gradient of the section sigmaEQ(sigma1, sigma2) of the yield
        function -- convex and centrally symmetric -- so that angle grows
        monotonically through pi as alpha = tan(phi) sweeps the real line
        (the same argument as YLD91_PS.calc_alpha_from_beta).
        """
        def flow_angle(phi: float) -> Optional[float]:
            g = self._flow_in_stress_axes(float(np.tan(phi)), theta_stress)
            return None if g is None else float(np.arctan2(g[1], g[0]))

        eps = 1e-9
        lo, hi = -0.5 * np.pi + eps, 0.5 * np.pi - eps
        f_lo, f_hi = flow_angle(lo), flow_angle(hi)
        if f_lo is None or f_hi is None:
            return None
        target = float(np.arctan2(beta, 1.0))
        while target < f_lo:
            target += np.pi
        while target > f_hi:
            target -= np.pi
        f_lo, f_hi = f_lo - target, f_hi - target
        if (f_lo > 0) == (f_hi > 0):
            return None
        for _ in range(60):
            if hi - lo < 1e-10:
                break
            mid = 0.5 * (lo + hi)
            f_mid = flow_angle(mid)
            if f_mid is None:
                return None
            f_mid -= target
            if (f_mid > 0) == (f_lo > 0):
                lo, f_lo = mid, f_mid
            else:
                hi, f_hi = mid, f_mid
        return float(np.tan(0.5 * (lo + hi)))

    def _sigma1_from_equivalent_stress_alpha_theta(self, sigmaEQ: float, alpha: float,
                                                   theta_stress: float) -> Optional[float]:
        """sigma1 = sigmaEQ / sigmaEQ(1, alpha) at theta_stress (sigmaEQ is homogeneous of degree 1)."""
        sxx, syy, txy = self._components_from_principal(1.0, alpha, theta_stress)
        unit = self._equivalent_stress_xy(sxx, syy, txy)
        if unit is None or unit == 0:
            return None
        return sigmaEQ / unit

    def calc_beta_from_deviatoric_stresses(self, state: PointState) -> Optional[float]:
        """General relation, valid for any plane-stress state (sigma3 = 0)."""
        if None not in (state.sigma1, state.sigma2):
            s1, s2 = state.sigma1, state.sigma2
            numerator = 2 * s2 - s1
            denominator = 2 * s1 - s2
            if denominator != 0:
                return numerator / denominator
        return None

    def calc_beta_from_strain_increments(self, state: PointState) -> Optional[float]:
        if state.deps1 not in (None, 0) and state.deps2 is not None:
            return state.deps2 / state.deps1
        return None

    def calc_deps2_from_deps1_beta(self, state: PointState) -> Optional[float]:
        if state.deps1 is not None and state.beta is not None:
            return state.beta * state.deps1
        return None

    def calc_deps1_from_deps2_beta(self, state: PointState) -> Optional[float]:
        if state.deps2 is not None and state.beta not in (None, 0):
            return state.deps2 / state.beta
        return None

    def calc_deps3_from_incompressibility(self, state: PointState) -> Optional[float]:
        if state.deps1 is not None and state.deps2 is not None:
            return -(state.deps1 + state.deps2)
        return None

    def calc_equivalent_plastic_strain_increment(self, state: PointState) -> Optional[float]:
        """
        Equivalent plastic strain increment, from the plastic-work
        conjugacy sigmaEQ*depsEQ = sigma1*deps1 + sigma2*deps2 (+
        sigma3*deps3, but sigma3 = 0 in plane stress). This is the
        criterion-agnostic definition: it holds for ANY convex yield
        criterion together with its associated (normality) flow rule --
        not just von Mises -- because it is exactly what makes depsEQ
        consistent with the hardening law's flow-stress-vs-cumulative-
        plastic-strain curve regardless of which yield surface is used
        (sigma1, sigma2 and sigmaEQ already encode the criterion's own
        anisotropy through calc_alpha_from_beta / calc_equivalent_stress).

        Returns None while sigma1/sigma2/sigmaEQ are not yet resolved --
        solve() keeps retrying this every iteration and only degrades to
        the isotropic sqrt((2/3)*sum(deps_i^2)) formula (see
        calc_equivalent_plastic_strain_increment_isotropic) as a last
        resort after its fixed-point loop finishes, so that a stress
        state which only becomes available a few iterations in is never
        pre-empted by the isotropic shortcut.
        """
        if None not in (state.sigma1, state.sigma2, state.sigmaEQ,
                        state.deps1, state.deps2) and state.sigmaEQ != 0:
            return (state.sigma1 * state.deps1 + state.sigma2 * state.deps2) / state.sigmaEQ
        return None

    def calc_equivalent_plastic_strain_increment_isotropic(self, state: PointState) -> Optional[float]:
        """
        Isotropic (von Mises) equivalent plastic strain increment, from
        strain increments alone. Only correct when the flow rule is
        actually isotropic; used by solve() strictly as a last-resort
        fallback, when no stress state is available at all to evaluate
        the general calc_equivalent_plastic_strain_increment.
        """
        if None not in (state.deps1, state.deps2, state.deps3):
            return sqrt((2 / 3) * (state.deps1 ** 2 + state.deps2 ** 2 + state.deps3 ** 2))
        return None

    def calc_strain_increments_from_flow_rule(self, state: PointState):
        """
        Associated flow rule (plane stress, sigma3 = 0): returns
        (deps1, deps2, deps3) from sigma1, sigma2 (at theta_stress from RD,
        0 if unset) and depsEQ, as

            deps1, deps2 = depsEQ * d(sigmaEQ)/d(sigma1, sigma2),
            deps3 = -(deps1 + deps2)

        using THIS criterion's gradient (the Levy-Mises relations for
        VonMises_PS). deps1, deps2 are the normal components along the
        stress principal axes (see _flow_in_stress_axes). By Euler's
        theorem sigma1*deps1 + sigma2*deps2 = sigmaEQ*depsEQ.
        """
        if None in (state.sigma1, state.sigma2, state.depsEQ) or state.sigma1 == 0:
            return None
        theta = state.theta_stress if state.theta_stress is not None else 0.0
        g = self._flow_in_stress_axes(state.sigma2 / state.sigma1, theta)
        if g is None:
            return None
        # (g1, g2) were computed at the unit state (1, alpha): the gradient
        # is homogeneous of degree 0, but for sigma1 < 0 the state is the
        # opposite point of the yield surface, whose gradient is reversed.
        sign = 1.0 if state.sigma1 > 0 else -1.0
        d1 = sign * state.depsEQ * g[0]
        d2 = sign * state.depsEQ * g[1]
        return d1, d2, -(d1 + d2)

    def calc_sigma1_from_equivalent_stress_and_alpha(self, state: PointState) -> Optional[float]:
        """To be redefined in subclasses according to the specific yield function."""
        return None

    @abstractmethod
    def calc_equivalent_stress(self, state: PointState) -> Optional[float]:
        """Generic interface to compute the equivalent stress. Each subclass must implement it."""

    @abstractmethod
    def calc_alpha_from_beta(self, state: PointState) -> Optional[float]:
        """Generic interface: alpha = sigma2/sigma1 from a known beta. Each subclass must implement it."""

    @abstractmethod
    def calc_beta_from_alpha(self, state: PointState) -> Optional[float]:
        """Generic interface: beta = deps2/deps1 from a known alpha. Each subclass must implement it."""

    @abstractmethod
    def calc_sigma_from_theta_stress(self, state: PointState) -> Optional[float]:
        """
        Generic interface: yield stress for a uniaxial tension test at
        angle theta_stress from principal direction 1 (off-axis loading). Each
        subclass must implement it, returning None where not supported
        (a sheared-state extension of the criterion is needed).
        """

    @abstractmethod
    def calc_lankford_from_theta_stress(self, state: PointState) -> Optional[float]:
        """
        Generic interface: Lankford coefficient r for a uniaxial tension
        test at angle theta_stress from principal direction 1 (off-axis
        loading). Each subclass must implement it, returning None where
        not supported (a sheared-state extension of the criterion is
        needed).
        """

    @abstractmethod
    def calc_psi_noncoax_from_theta_stress(self, state: PointState) -> Optional[float]:
        """
        Generic interface: angle (in radians) between the principal
        direction of the applied stress and the true principal strain-
        increment direction, for a general stress ratio alpha =
        sigma2/sigma1 (defaults to 0, i.e. uniaxial tension, if not set)
        applied at angle theta_stress from principal direction 1 (off-axis
        loading). Even though that stress state has no shear in its own
        (principal-1, principal-2) axes, an anisotropic material
        generally responds with a nonzero shear strain increment there
        too, so those axes are not the true principal strain directions
        except where theta_stress is a material symmetry direction (0 or 90
        degrees) or the state is isotropic in-plane (e.g. alpha = 1,
        equibiaxial). Each subclass must implement it, returning None
        where not supported (a sheared-state extension of the criterion
        is needed).
        """

    @staticmethod
    def _signed_power(x: float, p: float) -> float:
        """x**p preserving the sign of x (for non-integer exponents p)."""
        if x == 0:
            return 0.0
        return (1.0 if x > 0 else -1.0) * abs(x) ** p

    @staticmethod
    def _wrap_half_turn(angle: float) -> float:
        """Wraps an angle (radians) representing a principal direction (period pi, not 2*pi) into [-pi/2, pi/2)."""
        return float((angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi)

    def _ensure_calibration(self) -> None:
        """
        Hook for subclasses whose calibration coefficients (e.g. Hill's
        F/G/H/N, Barlat's C1..C6) can be derived lazily from Lankford/
        stress data -- see HillLankfordCriterion/YLD91_PS's overrides. A
        no-op here (VonMises_PS has no such derived calibration).

        solve() calls this once, eagerly, so that for a criterion reused
        across many solve() calls -- e.g. a Material's persistent
        yield_criterion -- the fit runs once and is cached on self,
        rather than being recomputed on every single evaluation.
        """

    # ------------------------------------------------------------------
    # Unified iterative solver
    # ------------------------------------------------------------------

    def solve(self, max_iter: int = 15, **known) -> PointState:
        """
        Iteratively computes every attribute of a NEW PointState that can
        be derived from **known (and from each other), using this
        criterion's physical relations under plane stress (sigma3 = 0).
        Stops when no further progress can be made or after max_iter
        iterations. Returns that PointState -- self is never touched
        except for _ensure_calibration() (calibration, not state; see
        its docstring), so the same calibrated criterion can be solved
        at any number of different stress states, independently, via
        repeated `criterion.solve(sigma1=..., sigma2=...)` calls.
        """
        self._ensure_calibration()
        state = PointState(**known)

        if state.sigma1 is None or state.sigma2 is None or state.theta_stress is None:
            principal = self.calc_principal_stresses_from_xy(state)
            if principal is not None:
                s1, s2, theta_stress = principal
                if state.sigma1 is None:
                    state.sigma1 = s1
                if state.sigma2 is None:
                    state.sigma2 = s2
                if state.theta_stress is None:
                    state.theta_stress = theta_stress

        for _ in range(max_iter):
            changed = False

            # --- stress block ---
            if state.alpha is None:
                v = self.calc_alpha_from_stresses(state)
                if v is not None:
                    state.alpha = v
                    changed = True

            if state.sigma2 is None:
                v = self.calc_sigma2_from_alpha(state)
                if v is not None:
                    state.sigma2 = v
                    changed = True

            if state.sigma1 is None:
                v = self.calc_sigma1_from_sigma2_alpha(state)
                if v is not None:
                    state.sigma1 = v
                    changed = True
                else:
                    v = self.calc_sigma1_from_equivalent_stress_and_alpha(state)
                    if v is not None:
                        state.sigma1 = v
                        changed = True

            if state.sigmaEQ is None:
                v = self.calc_equivalent_stress(state)
                if v is not None:
                    state.sigmaEQ = v
                    changed = True

            # --- alpha <-> beta relation ---
            if state.beta is None:
                v = self.calc_beta_from_deviatoric_stresses(state)
                if v is not None:
                    state.beta = v
                    changed = True
                else:
                    v = self.calc_beta_from_strain_increments(state)
                    if v is not None:
                        state.beta = v
                        changed = True

            if state.alpha is None:
                v = self.calc_alpha_from_beta(state)
                if v is not None:
                    state.alpha = v
                    changed = True

            # --- stress/strain rotation offset ---
            if state.psi_noncoax is None and state.theta_stress is not None:
                v = self.calc_psi_noncoax_from_theta_stress(state)
                if v is not None:
                    state.psi_noncoax = v
                    changed = True

            # --- strain increment block ---
            if None in (state.deps1, state.deps2, state.deps3):
                v = self.calc_strain_increments_from_flow_rule(state)
                if v is not None:
                    d1, d2, d3 = v
                    if state.deps1 is None:
                        state.deps1, changed = d1, True
                    if state.deps2 is None:
                        state.deps2, changed = d2, True
                    if state.deps3 is None:
                        state.deps3, changed = d3, True

            if state.deps2 is None:
                v = self.calc_deps2_from_deps1_beta(state)
                if v is not None:
                    state.deps2 = v
                    changed = True

            if state.deps1 is None:
                v = self.calc_deps1_from_deps2_beta(state)
                if v is not None:
                    state.deps1 = v
                    changed = True

            if state.deps3 is None:
                v = self.calc_deps3_from_incompressibility(state)
                if v is not None:
                    state.deps3 = v
                    changed = True

            if state.depsEQ is None:
                v = self.calc_equivalent_plastic_strain_increment(state)
                if v is not None:
                    state.depsEQ = v
                    changed = True

            if not changed:
                break

        # Last resort: if the stress-based (criterion-specific) formula
        # never had enough to work with (e.g. no sigmaEQ/beta was ever
        # supplied), fall back to the isotropic estimate rather than
        # leaving depsEQ unset.
        if state.depsEQ is None:
            v = self.calc_equivalent_plastic_strain_increment_isotropic(state)
            if v is not None:
                state.depsEQ = v

        return state

    # ---- generic utilities, valid for any subclass -------
    @property
    def name(self) -> str:
        """
        Human-readable name of the criterion, for legends and tables: label
        if given, otherwise the concrete class name without the "_PS"
        (plane stress) suffix, e.g. "Hill48" for Hill48_PS.
        """
        if self.label:
            return self.label
        return self.__class__.__name__.removesuffix("_PS")

    def locus_points(self, n_points: int = 361, sigmaY: float = 1.0):
        """
        Returns the (sigma1, sigma2) points of the yield locus (tau_xy = 0),
        sweeping directions in the plane, scaled to sigmaY.

        Parameters
        ----------
        n_points : int
            Number of angular directions sampled between 0 and 2*pi.
        sigmaY : float
            Reference stress the locus is scaled to (default 1.0, i.e.
            already normalised -- with normalize=True downstream, any
            positive value gives the same normalised shape; pass the
            material's real yield stress for an UNnormalised plot).

        Returns
        -------
        sigma1, sigma2 : np.ndarray
        """
        theta = np.linspace(0.0, 2.0 * np.pi, n_points)
        c, s = np.cos(theta), np.sin(theta)

        self._ensure_calibration()
        seq_dir = np.array(
            [self.calc_equivalent_stress(PointState(sigma1=ci, sigma2=si)) for ci, si in zip(c, s)],
            dtype=float,
        )

        # avoid division by zero in degenerate directions
        r = np.divide(sigmaY, seq_dir,
                       out=np.zeros_like(seq_dir), where=seq_dir > 1e-12)

        sigma1 = r * c
        sigma2 = r * s
        return sigma1, sigma2

    def plot_locus(self, ax=None, n_points: int = 361, sigmaY: float = 1.0, **plot_kwargs):
        """Plots only this criterion's locus. Returns the axis."""
        sigma1, sigma2 = self.locus_points(n_points, sigmaY=sigmaY)
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        label = plot_kwargs.pop("label", self.name)
        ax.plot(sigma2, sigma1, label=label, **plot_kwargs)     # sigma1 vertical, as plot_yield_loci
        ax.set_xlabel(r"$\sigma_2$")
        ax.set_ylabel(r"$\sigma_1$")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return ax

    # ------------------------------------------------------------------
    # Field groups for summary(): general (xx/yy/xy) stress, strain
    # increments, and principal stress/orientation state are the fixed
    # first 3 columns of the grid (common to every criterion); the
    # Lankford coefficients (r0, r45, r90) get their own comma-
    # separated row instead, since they read more naturally that way than
    # split across column cells. Everything else the concrete class
    # defines -- Hill's F, G, H, N; Barlat's a, C1..C6; ... -- goes
    # into a 4th grid column, computed dynamically per subclass.
    _SUMMARY_COLUMNS = (
        ("sigma_xx", "sigma_yy", "tau_xy", "sigma1", "sigma2", "sigmaEQ"),
        ("deps1", "deps2", "deps3", "depsEQ"),
        ("alpha", "beta", "theta_stress", "psi_noncoax"),
    )
    _SUMMARY_LANKFORD_FIELDS = ("r0", "r45", "r90")
    _SUMMARY_STRESS_FIELDS = ("sigma0", "sigma45", "sigma90")
    _SUMMARY_ANGLE_FIELDS = ("theta_stress", "psi_noncoax")
    _SUMMARY_CELL_WIDTH = 24

    @classmethod
    def _summary_value(cls, obj, name: str) -> str:
        """Formatted value of one field on `obj` (self or a state), e.g. '12.3' or '0.4 rad (22.9 deg)'."""
        value = getattr(obj, name)
        if value is None:
            return "None"
        if name in cls._SUMMARY_ANGLE_FIELDS and isinstance(value, float):
            return f"{value:.4g} rad ({np.degrees(value):.3g} deg)"
        if isinstance(value, float):
            return f"{value:.4g}"
        return f"{value}"

    @classmethod
    def _summary_cell(cls, obj, name: str, show_unset: bool) -> str:
        """'name = value' for one field on `obj` (self or a state), padded for column alignment, or
        '' if unset and show_unset is False."""
        if getattr(obj, name) is None and not show_unset:
            return ""
        return f"{name:9s} = {cls._summary_value(obj, name)}"

    def summary(self, state: Optional[PointState] = None, show_unset: bool = False) -> str:
        """
        Human-readable summary of this criterion's calibration, plus
        `state`'s stress/strain values if a state is given:

        - A first row with the Lankford coefficients (r0, r45, r90),
          comma-separated, if this criterion defines them.
        - A second row with the experimental uniaxial yield stresses
          (sigma0, sigma45, sigma90), comma-separated, if this criterion
          defines them (e.g. Yld91, for the stress-based
          calibration -- see calc_barlat_coefficients_from_stress).
        - A grid (see `_SUMMARY_COLUMNS`) of `state`'s stress/strain
          values (general (sigma_xx, sigma_yy, tau_xy) and principal
          (theta_stress, sigma1, sigma2, alpha) stress, the equivalent
          stress/strain pair (sigmaEQ, depsEQ), and the strain increments
          (psi_noncoax, deps1, deps2, deps3, beta) -- left entirely blank
          if state is None) -- plus a 4th column with every calibration
          parameter this criterion defines (Hill's F, G, H, N; Barlat's
          a, C1..C6; ...), read from self.

        By default fields that are still unset (None) are omitted (the
        Lankford row) or left blank (grid cells); pass show_unset=True
        to also show them (as None).
        """
        w = self._SUMMARY_CELL_WIDTH
        lines = [self.name]

        lankford_parts = []
        for name in self._SUMMARY_LANKFORD_FIELDS:
            if not hasattr(self, name):
                continue
            if getattr(self, name) is None and not show_unset:
                continue
            lankford_parts.append(f"{name} = {self._summary_value(self, name)}")
        if lankford_parts:
            lines.append(", ".join(lankford_parts))

        stress_parts = []
        for name in self._SUMMARY_STRESS_FIELDS:
            if not hasattr(self, name):
                continue
            if getattr(self, name) is None and not show_unset:
                continue
            stress_parts.append(f"{name} = {self._summary_value(self, name)}")
        if stress_parts:
            lines.append(", ".join(stress_parts))

        already_shown = ({name for col in self._SUMMARY_COLUMNS for name in col}
                          | set(self._SUMMARY_LANKFORD_FIELDS) | set(self._SUMMARY_STRESS_FIELDS) | {"label"})
        calib_col = tuple(f.name for f in fields(self) if f.name not in already_shown)
        columns = self._SUMMARY_COLUMNS + (calib_col,) if calib_col else self._SUMMARY_COLUMNS
        column_objs = (state,) * len(self._SUMMARY_COLUMNS) + (self,)

        n_rows = max(len(col) for col in columns)
        for row in range(n_rows):
            cells = []
            for col, obj in zip(columns, column_objs):
                if row >= len(col) or obj is None:
                    cells.append(" " * w)
                    continue
                name = col[row]
                cells.append(self._summary_cell(obj, name, show_unset).ljust(w))
            line = "".join(cells).rstrip()
            if line:
                lines.append(line)

        return "\n".join(lines)

    def print_summary(self, state: Optional[PointState] = None, show_unset: bool = False) -> None:
        """Print `summary()` (see its docstring) to stdout."""
        print(self.summary(state=state, show_unset=show_unset))
        print()

    def __str__(self) -> str:
        return self.summary()


# ======================================================================
# Specific criteria deriving directly from YieldCriterionPlaneStress
# ======================================================================

@dataclass
class VonMises_PS(YieldCriterionPlaneStress):
    """Isotropic von Mises yield criterion."""

    def calc_equivalent_stress(self, state: PointState) -> Optional[float]:
        """Equivalent stress according to von Mises (plane stress, sigma3 = 0)."""
        if None not in (state.sigma1, state.sigma2):
            s1, s2 = state.sigma1, state.sigma2
            return sqrt(0.5 * ((s1 - s2) ** 2 + s2 ** 2 + s1 ** 2))
        return None

    def _equivalent_stress_xy(self, sigma_xx: float, sigma_yy: float, tau_xy: float) -> Optional[float]:
        return sqrt(max(sigma_xx ** 2 - sigma_xx * sigma_yy + sigma_yy ** 2 + 3.0 * tau_xy ** 2, 0.0))

    def _flow_gradient_xy(self, sigma_xx: float, sigma_yy: float,
                          tau_xy: float) -> Optional[Tuple[float, float, float]]:
        """Gradient of sigmaEQ^2 (a positive multiple of that of sigmaEQ)."""
        return 2.0 * sigma_xx - sigma_yy, 2.0 * sigma_yy - sigma_xx, 6.0 * tau_xy

    def calc_alpha_from_beta(self, state: PointState) -> Optional[float]:
        if state.beta is not None and (2 + state.beta) != 0:
            return (2 * state.beta + 1) / (2 + state.beta)
        return None

    def calc_beta_from_alpha(self, state: PointState) -> Optional[float]:
        """Inverse of calc_alpha_from_beta."""
        if state.alpha is not None and (2 - state.alpha) != 0:
            return (2 * state.alpha - 1) / (2 - state.alpha)
        return None

    def calc_sigma1_from_equivalent_stress_and_alpha(self, state: PointState) -> Optional[float]:
        if state.sigmaEQ is not None and state.alpha is not None:
            denom = sqrt(1 - state.alpha + state.alpha ** 2)
            if denom != 0:
                return state.sigmaEQ / denom
        return None

    def calc_sigma_from_theta_stress(self, state: PointState) -> Optional[float]:
        """
        Von Mises is isotropic: the (equivalent) stress does not depend
        on the loading angle. Scaled to state.sigmaEQ if given, otherwise
        left normalised to 1.0 (matching Hill48_PS/YLD91_PS's own
        calc_sigma_from_theta_stress convention).
        """
        if state.theta_stress is None:
            return None
        return state.sigmaEQ if state.sigmaEQ is not None else 1.0

    def calc_lankford_from_theta_stress(self, state: PointState) -> Optional[float]:
        """
        Von Mises is isotropic: deps2 = deps3 in magnitude for a uniaxial
        tension test at any angle (deps1:deps2:deps3 = 1:-0.5:-0.5), so
        r = deps2/deps3 = 1 regardless of theta_stress.
        """
        if state.theta_stress is None:
            return None
        return 1.0

    def calc_psi_noncoax_from_theta_stress(self, state: PointState) -> Optional[float]:
        """
        Von Mises is isotropic: the strain-increment tensor's principal
        directions always coincide with the stress tensor's (no induced
        shear), so the loading direction is always a principal strain
        direction too: psi_noncoax = 0 for any theta_stress.
        """
        if state.theta_stress is None:
            return None
        return 0.0


@dataclass
class HillLankfordCriterion(YieldCriterionPlaneStress):
    """
    Shared base for the Hill 1948 and Hill 1979 anisotropic criteria: both
    are calibrated from the same Lankford ratios (r0, r45, r90) via the
    same F, G, H, N formulas (Hosford-Caddell), and share the same
    stress-based flow-ratio relation. The actual yield/flow physics
    (quadratic vs. non-quadratic) is left to each subclass -- this class
    is not meant to be instantiated directly (it inherits the unimplemented
    abstract methods from YieldCriterionPlaneStress).
    """

    r0: Optional[float] = None
    r45: Optional[float] = None
    r90: Optional[float] = None
    F: Optional[float] = None
    G: Optional[float] = None
    H: Optional[float] = None
    N: Optional[float] = None

    def calc_hill_coefficients_from_lankford(self) -> Optional[Tuple[float, float, float, float]]:
        if None not in (self.r0, self.r45, self.r90):
            if self.r0 == -1 or self.r90 == -1:
                return None

            G = 1.0 / (1.0 + self.r0)
            H = self.r0 / (1.0 + self.r0)
            F = H / self.r90 if self.r90 != 0 else None
            # Standard Hosford-Caddell formula: reproduces r45 exactly in
            # r(theta) = [H + (2N-F-G-4H) sin^2(theta) cos^2(theta)] / [F sin^2(theta) + G cos^2(theta)].
            N = ((self.r0 + self.r90) * (1.0 + 2.0 * self.r45)
                 / (2.0 * self.r90 * (1.0 + self.r0))) if self.r90 != 0 else None

            if F is None or N is None:
                return None
            return F, G, H, N
        return None

    def ensure_hill_coefficients(self) -> None:
        if None in (self.F, self.G, self.H, self.N):
            coeffs = self.calc_hill_coefficients_from_lankford()
            if coeffs is not None:
                self.F, self.G, self.H, self.N = coeffs

    def _ensure_calibration(self) -> None:
        self.ensure_hill_coefficients()

    def calc_beta_from_deviatoric_stresses(self, state: PointState) -> Optional[float]:
        """Hill flow ratio computed from the principal stress state."""
        self.ensure_hill_coefficients()
        if None not in (state.sigma1, state.sigma2):
            if state.sigma1 == 0:
                return None
            return self.calc_beta_from_alpha(state)
        return None


@dataclass
class Hill48_PS(HillLankfordCriterion):
    """Anisotropic quadratic Hill 1948 yield criterion."""

    def calc_equivalent_stress(self, state: PointState) -> Optional[float]:
        """
        Equivalent stress for Hill 1948 (plane stress, sigma3 = 0), from
        the material-axes components (see _stress_components):

            sigmaEQ^2 = G*sigma_xx^2 + F*sigma_yy^2 + H*(sigma_xx - sigma_yy)^2
                        + 2*N*tau_xy^2
        """
        comps = self._stress_components(state)
        if comps is None:
            return None
        return self._equivalent_stress_xy(*comps)

    def _equivalent_stress_xy(self, sigma_xx: float, sigma_yy: float, tau_xy: float) -> Optional[float]:
        self.ensure_hill_coefficients()
        if None in (self.F, self.G, self.H, self.N):
            return None
        q = (self.G * sigma_xx ** 2 + self.F * sigma_yy ** 2
             + self.H * (sigma_xx - sigma_yy) ** 2 + 2.0 * self.N * tau_xy ** 2)
        return sqrt(q) if q >= 0 else None

    def calc_alpha_from_beta(self, state: PointState) -> Optional[float]:
        """Inverse relation for alpha = sigma2 / sigma1 from a known beta."""
        self.ensure_hill_coefficients()
        if state.beta is not None and not self._is_rd_aligned(state.theta_stress):
            return self._alpha_from_beta_theta(state.beta, state.theta_stress)
        if None not in (state.beta, self.F, self.G, self.H):
            denom = self.F + self.H + state.beta * self.H
            if denom != 0:
                return (self.H + state.beta * (self.G + self.H)) / denom
        return None

    def calc_beta_from_alpha(self, state: PointState) -> Optional[float]:
        """Strain increment ratio beta = deps2 / deps1 from the Hill flow law."""
        self.ensure_hill_coefficients()
        if state.alpha is not None and not self._is_rd_aligned(state.theta_stress):
            return self._beta_from_alpha_theta(state.alpha, state.theta_stress)
        if None not in (state.alpha, self.F, self.G, self.H):
            denom = self.G + self.H - self.H * state.alpha
            if denom != 0:
                return (self.F * state.alpha + self.H * (state.alpha - 1.0)) / denom
        return None

    def calc_sigma1_from_equivalent_stress_and_alpha(self, state: PointState) -> Optional[float]:
        """Hill 1948 conversion: sigma1 from sigmaEQ and alpha."""
        self.ensure_hill_coefficients()
        if None not in (state.sigmaEQ, state.alpha) and not self._is_rd_aligned(state.theta_stress):
            return self._sigma1_from_equivalent_stress_alpha_theta(state.sigmaEQ, state.alpha, state.theta_stress)
        if state.sigmaEQ is not None and state.alpha is not None:
            if None not in (self.F, self.G, self.H):
                denom = sqrt(self.G + self.H * (1.0 - state.alpha) ** 2 + self.F * state.alpha ** 2)
                if denom != 0:
                    return state.sigmaEQ / denom
        return None

    def calc_sigma_from_theta_stress(self, state: PointState) -> Optional[float]:
        """
        Stress required to yield in a uniaxial tension test at angle
        theta_stress from principal direction 1 (e.g. the rolling direction).

        In the (1, 2) axes, such a test is the sheared stress state
            sigma1' = sigma cos^2(theta_stress), sigma2' = sigma sin^2(theta_stress),
            tau12'  = sigma sin(theta_stress) cos(theta_stress).
        Substituting into Hill's quadratic yield function gives the
        standard result (normalised so sigma(0) = 1, since G + H = 1):

            sigma(theta_stress)^-2 = F sin^4(theta_stress) + G cos^4(theta_stress)
                               + H cos^2(2 theta_stress) + 2 N sin^2(theta_stress) cos^2(theta_stress)

        If state.sigmaEQ is already set (e.g. from a previous solve() at
        theta_stress = 0), the result is scaled to that reference stress;
        otherwise it is left normalised to sigma(0) = 1.
        """
        self.ensure_hill_coefficients()
        if state.theta_stress is None or None in (self.F, self.G, self.H, self.N):
            return None

        s2, c2 = np.sin(state.theta_stress) ** 2, np.cos(state.theta_stress) ** 2
        denom = self.F * s2 ** 2 + self.G * c2 ** 2 + self.H * (c2 - s2) ** 2 + 2.0 * self.N * s2 * c2
        if denom <= 0:
            return None

        scale = state.sigmaEQ if state.sigmaEQ is not None else 1.0
        return scale / sqrt(denom)

    def calc_lankford_from_theta_stress(self, state: PointState) -> Optional[float]:
        """
        Lankford coefficient r for a uniaxial tension test at angle
        theta_stress from principal direction 1. Standard closed-form result
        (Hosford & Caddell), derived from the associated flow rule at the
        same sheared stress state used in calc_sigma_from_theta_stress:

            r(theta_stress) = [H + (2N - F - G - 4H) sin^2(theta_stress) cos^2(theta_stress)]
                        / [F sin^2(theta_stress) + G cos^2(theta_stress)]

        which reproduces r(0) = r0, r(45 deg) = r45 and r(90 deg) = r90
        exactly.
        """
        self.ensure_hill_coefficients()
        if state.theta_stress is None or None in (self.F, self.G, self.H, self.N):
            return None

        s2, c2 = np.sin(state.theta_stress) ** 2, np.cos(state.theta_stress) ** 2
        denominator = self.F * s2 + self.G * c2
        if denominator == 0:
            return None

        numerator = self.H + (2.0 * self.N - self.F - self.G - 4.0 * self.H) * s2 * c2
        return numerator / denominator

    def _flow_gradient_xy(self, sigma_xx: float, sigma_yy: float,
                           tau_xy: float) -> Optional[Tuple[float, float, float]]:
        """
        Gradient (gx, gy, gxy) = d(sigmaEQ)/d(sigma_xx, sigma_yy, tau_xy)
        of the Hill 1948 associated flow rule at a GENERAL in-plane stress
        state (sigma_xx, sigma_yy, tau_xy need not be principal, unlike
        calc_equivalent_stress). Since

            sigmaEQ^2 = F*sigma_yy^2 + G*sigma_xx^2 + H*(sigma_xx-sigma_yy)^2
                         + 2*N*tau_xy^2

        (the shear-extended Hill 1948 form used by calc_sigma_from_theta_stress)
        is quadratic, its gradient is LINEAR in the stress components. A
        common positive factor (2*sigmaEQ) is dropped, as elsewhere in
        this module.
        """
        self.ensure_hill_coefficients()
        if None in (self.F, self.G, self.H, self.N):
            return None
        gx = self.G * sigma_xx + self.H * (sigma_xx - sigma_yy)
        gy = self.F * sigma_yy - self.H * (sigma_xx - sigma_yy)
        gxy = 2.0 * self.N * tau_xy
        return gx, gy, gxy

    def calc_psi_noncoax_from_theta_stress(self, state: PointState) -> Optional[float]:
        """
        Angle (radians) between the loading direction (the applied
        stress's own principal direction) and the true principal strain-
        increment direction, for a general stress ratio alpha =
        sigma2/sigma1 (defaults to 0, i.e. uniaxial tension, if not set)
        applied at angle theta_stress from principal material direction 1:

            psi_noncoax = wrap[0.5 * atan2(gxy, gx - gy) - theta_stress]

        where (gx, gy, gxy) = _flow_gradient_xy(sigma_xx, sigma_yy, tau_xy)
        at the rotated stress state

            sigma_xx = cos^2(theta_stress) + alpha*sin^2(theta_stress)
            sigma_yy = sin^2(theta_stress) + alpha*cos^2(theta_stress)
            tau_xy   = (1 - alpha)*sin(theta_stress)*cos(theta_stress)

        (the standard principal-stress rotation formula; alpha = 0
        reduces to the uniaxial state used by calc_sigma_from_theta_stress /
        calc_lankford_from_theta_stress). sigma_xx, sigma_yy, tau_xy are given
        in the MATERIAL axes (1, 2), so (gx, gy, gxy) -- and hence the raw
        atan2 -- is also expressed there, i.e. it is the strain-increment
        principal direction measured from material axis 1, not from the
        loading direction; subtracting theta_stress re-references it to the
        loading direction, and wrap[] (_wrap_half_turn) brings the period
        -pi principal-direction ambiguity back into [-pi/2, pi/2).
        """
        if state.theta_stress is None:
            return None
        alpha = state.alpha if state.alpha is not None else 0.0

        s, c = np.sin(state.theta_stress), np.cos(state.theta_stress)
        s2, c2, cs = s ** 2, c ** 2, s * c
        sigma_xx = c2 + alpha * s2
        sigma_yy = s2 + alpha * c2
        tau_xy = (1.0 - alpha) * cs

        grad = self._flow_gradient_xy(sigma_xx, sigma_yy, tau_xy)
        if grad is None:
            return None
        gx, gy, gxy = grad
        psi_noncoax_from_axis1 = 0.5 * float(np.arctan2(gxy, gx - gy))
        return self._wrap_half_turn(psi_noncoax_from_axis1 - state.theta_stress)


@dataclass
class Hill79_PS(HillLankfordCriterion):
    """Anisotropic non-quadratic Hill 1979 yield criterion."""

    m: float = 6.0

    def calc_equivalent_stress(self, state: PointState) -> Optional[float]:
        """
        Equivalent stress for Hill 1979 (plane stress, sigma3 = 0), from
        the material-axes components (see _stress_components). Only
        defined when the stress principal axes are the material axes
        (tau_xy = 0, i.e. sigma1 along RD or TD): Hill 1979 has no shear
        terms, so None is returned for any other orientation.
        """
        comps = self._stress_components(state)
        if comps is None:
            return None
        return self._equivalent_stress_xy(*comps)

    @staticmethod
    def _is_shear_free(sigma_xx: float, sigma_yy: float, tau_xy: float) -> bool:
        return abs(tau_xy) <= 1e-12 * max(abs(sigma_xx), abs(sigma_yy), 1e-300)

    def _equivalent_stress_xy(self, sigma_xx: float, sigma_yy: float, tau_xy: float) -> Optional[float]:
        self.ensure_hill_coefficients()
        if None in (self.F, self.G, self.H) or not self._is_shear_free(sigma_xx, sigma_yy, tau_xy):
            return None
        m = self.m
        inner = (self.G * abs(sigma_xx) ** m + self.F * abs(sigma_yy) ** m
                 + self.H * abs(sigma_xx - sigma_yy) ** m)
        return inner ** (1.0 / m) if inner >= 0 else None

    def _flow_gradient_xy(self, sigma_xx: float, sigma_yy: float,
                          tau_xy: float) -> Optional[Tuple[float, float, float]]:
        """Gradient of the Hill 1979 potential (positive multiple of that of sigmaEQ); shear-free states only."""
        self.ensure_hill_coefficients()
        if None in (self.F, self.G, self.H) or not self._is_shear_free(sigma_xx, sigma_yy, tau_xy):
            return None
        p = self.m - 1.0
        d = self._signed_power(sigma_xx - sigma_yy, p)
        return (self.G * self._signed_power(sigma_xx, p) + self.H * d,
                self.F * self._signed_power(sigma_yy, p) - self.H * d,
                0.0)

    def calc_sigma1_from_equivalent_stress_and_alpha(self, state: PointState) -> Optional[float]:
        self.ensure_hill_coefficients()
        if None not in (state.sigmaEQ, state.alpha) and not self._is_rd_aligned(state.theta_stress):
            return self._sigma1_from_equivalent_stress_alpha_theta(state.sigmaEQ, state.alpha, state.theta_stress)
        if state.sigmaEQ is not None and state.alpha is not None:
            if None not in (self.F, self.G, self.H):
                m = self.m
                denom = (self.G + self.F * abs(state.alpha) ** m
                         + self.H * abs(1.0 - state.alpha) ** m)
                if denom > 0:
                    return state.sigmaEQ / denom ** (1.0 / m)
        return None

    def calc_alpha_from_beta(self, state: PointState) -> Optional[float]:
        self.ensure_hill_coefficients()
        if state.beta is None or None in (self.F, self.G, self.H):
            return None
        if not self._is_rd_aligned(state.theta_stress):
            return self._alpha_from_beta_theta(state.beta, state.theta_stress)

        def residual(a: float) -> Optional[float]:
            # replace(): a throwaway clone of `state` with alpha=a, rather
            # than mutating state.alpha in place -- also safer than a
            # save/restore dance, which would leave state.alpha corrupted
            # if calc_beta_from_alpha() ever raised.
            b = self.calc_beta_from_alpha(replace(state, alpha=a))
            if b is None:
                return None
            return b - state.beta

        lo, hi = -0.999, 1.999
        # calc_beta_from_alpha's slope dpsi_d1 = G + H*(1-alpha)^(m-1) is a
        # sum of positive terms for alpha <= 1, so it never vanishes there;
        # but for alpha > 1, dpsi_d1 = G - H*(alpha-1)^(m-1) has a pole at
        # alpha_pole = 1 + (G/H)^(1/(m-1)), where beta(alpha) shoots to
        # +-infinity and re-enters a different (unphysical, for this
        # search) branch. If that pole falls inside (1, hi), clamp hi to
        # just below it -- otherwise the plain endpoint bracket check below
        # can miss a real root that lies well inside [lo, hi], because
        # beta(hi) ends up back on the wrong side of zero past the pole.
        if self.H not in (None, 0) and self.m > 1:
            ratio = self.G / self.H
            if ratio > 0:
                alpha_pole = 1.0 + ratio ** (1.0 / (self.m - 1.0))
                hi = min(hi, 1.0 + 0.999 * (alpha_pole - 1.0))

        f_lo, f_hi = residual(lo), residual(hi)
        if f_lo is None or f_hi is None or f_lo == 0 or f_hi == 0 or (f_lo > 0) == (f_hi > 0):
            return lo if f_lo == 0 else (hi if f_hi == 0 else None)

        # 40 iterations (down from a fixed 60 with no early exit) plus a bracket-
        # width tolerance: alpha is resolved to ~1e-8 well before either limit is
        # reached, which is far tighter than this model needs.
        for _ in range(40):
            if hi - lo < 1e-8:
                break
            mid = 0.5 * (lo + hi)
            f_mid = residual(mid)
            if f_mid is None:
                return None
            if f_mid == 0:
                return mid
            if (f_mid > 0) == (f_lo > 0):
                lo, f_lo = mid, f_mid
            else:
                hi, f_hi = mid, f_mid
        return 0.5 * (lo + hi)

    def calc_beta_from_alpha(self, state: PointState) -> Optional[float]:
        self.ensure_hill_coefficients()
        if state.alpha is not None and not self._is_rd_aligned(state.theta_stress):
            return self._beta_from_alpha_theta(state.alpha, state.theta_stress)
        if None not in (state.alpha, self.F, self.G, self.H):
            m = self.m
            alpha = state.alpha
            dpsi_d1 = (self.G * self._signed_power(1.0, m - 1)
                       - self.H * self._signed_power(alpha - 1.0, m - 1))
            dpsi_d2 = (self.F * self._signed_power(alpha, m - 1)
                       + self.H * self._signed_power(alpha - 1.0, m - 1))
            if dpsi_d1 != 0:
                return dpsi_d2 / dpsi_d1
        return None

    def calc_sigma_from_theta_stress(self, state: PointState) -> Optional[float]:
        """Not implemented: Hill 1979's non-quadratic form has no standard off-axis (sheared) extension."""
        return None

    def calc_lankford_from_theta_stress(self, state: PointState) -> Optional[float]:
        """Not implemented: Hill 1979's non-quadratic form has no standard off-axis (sheared) extension."""
        return None

    def calc_psi_noncoax_from_theta_stress(self, state: PointState) -> Optional[float]:
        """Not implemented: Hill 1979's non-quadratic form has no standard off-axis (sheared) extension."""
        return None


@dataclass
class YLD91_PS(YieldCriterionPlaneStress):
    """Anisotropic Barlat 1991 (Yld91) yield criterion."""

    r0: Optional[float] = None
    r45: Optional[float] = None
    r90: Optional[float] = None
    a: float = 8.0
    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: Optional[float] = None
    C5: Optional[float] = None
    C6: Optional[float] = None
    # Experimental uniaxial yield stresses at 0/45/90 degrees -- used by
    # the stress-based calibration strategies (see ensure_barlat_
    # coefficients for the full priority order): together with r0, r45,
    # r90 by calc_barlat_coefficients_from_stress_and_lankford_full (all
    # 6) or calc_barlat_coefficients_from_stress_and_lankford (r0, r90
    # only; r45 not used there), or alone (no r-values at all) by
    # calc_barlat_coefficients_from_stress. The flow-rule-only calibration
    # (calc_barlat_coefficients_from_lankford) needs just r0, r45, r90 and
    # ignores these. Not to be confused with sigmaEQ (the CURRENT
    # equivalent-stress state): these are fixed material calibration data.
    sigma0: Optional[float] = None
    sigma45: Optional[float] = None
    sigma90: Optional[float] = None

    def calc_barlat_coefficients_from_lankford(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        if None not in (self.r0, self.r45, self.r90):
            if min(self.r0, self.r45, self.r90) <= -1.0 or self.r90 == 0:
                return None

            solved = self._solve_c2_c3_c6(self.r0, self.r45, self.r90, self.a)
            if solved is None:
                return None
            C2, C3, C6 = solved
            return self._normalize_to_unit_RD_tension(C2, C3, C6, self.a)
        return None

    @staticmethod
    def _normalize_to_unit_RD_tension(C2: float, C3: float, C6: float, a: float
                                       ) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Rescales (C1=1, C2, C3, C4=1, C5=1, C6) by a single common factor
        so that calc_equivalent_stress(sigma1=sigma0, sigma2=0) = sigma0
        for ANY sigma0, not just some implicit unit scale.

        r0, r45, r90 alone only pin the SHAPE of phi (they come from the
        associated flow rule, i.e. the GRADIENT DIRECTION of phi, which a
        uniform positive scaling of phi leaves unchanged -- see
        _flow_gradient_xy) -- they say nothing about its absolute scale,
        which is why _solve_c2_c3_c6 has to pin C1 = 1 arbitrarily just to
        make that root-finding problem well posed. Left as is, that makes
        calc_equivalent_stress(sigma1, 0) come out different from sigma1
        in general (unlike Hill48_PS/Hill79_PS, whose F, G, H, N formulas satisfy
        G + H = 1 identically, so they get this normalisation for free).

        Since phi(S1,S2,S3) is homogeneous of degree `a` in stress (S is a
        LINEAR transform of sigma), sigmaEQ = (phi/2)^(1/a) is homogeneous
        of degree 1: scaling (C1..C6) together by any lambda > 0 scales
        sigmaEQ by that same lambda while leaving the flow direction (and
        hence r0, r45, r90) untouched. So fixing sigmaEQ at ONE stress
        state (unit RD tension: sigma_xx=1, sigma_yy=tau_xy=0) fixes it at
        every RD tensile stress -- matching the calibration convention
        already used by every other criterion in this module.
        """
        S1, S2, S3 = (C2 + C3) / 3.0, -C3 / 3.0, -C2 / 3.0  # sigma_xx=1, sigma_yy=tau_xy=0
        inner = abs(S1 - S2) ** a + abs(S2 - S3) ** a + abs(S3 - S1) ** a
        raw_sigmaEQ = (0.5 * inner) ** (1.0 / a)
        if raw_sigmaEQ == 0:
            return None
        scale = 1.0 / raw_sigmaEQ
        return scale, C2 * scale, C3 * scale, scale, scale, C6 * scale

    @classmethod
    def _solve_c2_c3_c6(cls, r0: float, r45: float, r90: float, a: float,
                         tol: float = 1e-10, max_iter: int = 60) -> Optional[Tuple[float, float, float]]:
        """
        Newton-Raphson (Jacobian by finite differences) for (C2, C3, C6),
        with C1 = 1 fixed (a normalisation: matching r-values is invariant
        to any common scaling of (C1, C2, C3) together -- see
        calc_equivalent_stress -- so one of them must be pinned) and
        C4 = C5 = 1 (they multiply the out-of-plane shears tau_yz, tau_zx
        in the full 3D Yld91 transformation -- see
        _transformed_principal_stresses -- which are always zero in plane
        stress, so C4, C5 have NO EFFECT on anything computed here), so
        that the ASSOCIATED FLOW RULE reproduces r0, r45 and r90 exactly
        at theta_stress = 0, 45 and 90 degrees (via
        calc_lankford_from_theta_stress) -- not just the yield locus at
        the RD/TD states.

        C6 is the in-plane shear coefficient (couples tau_xy); it is
        needed to match r45 (an off-axis test, which is a sheared state
        in material axes) but has NO effect on r0 or r90 (pure axis-
        aligned tension, no shear) -- with C1 fixed, the 3 remaining
        conditions (r0, r45, r90) exactly determine the 3 free
        coefficients (C2, C3, C6).
        """
        probe = cls(a=a)
        probe.C1, probe.C4, probe.C5 = 1.0, 1.0, 1.0
        targets = ((0.0, r0), (45.0, r45), (90.0, r90))

        def residuals(x):
            probe.C2, probe.C3, probe.C6 = x
            out = []
            for deg, target in targets:
                r = probe.calc_lankford_from_theta_stress(PointState(theta_stress=np.deg2rad(deg)))
                if r is None:
                    return None
                out.append(r - target)
            return np.array(out)

        # Initial guess (C2=1, C3=0, C6=1): avoids the degenerate point
        # C2=C3=0 (with C1=1), where S1=S2=S3=0 at theta=0 (Mohr-circle
        # radius R=0, where calc_lankford_from_theta_stress is undefined).
        x = np.array([1.0, 0.0, 1.0])

        eps = 1e-6
        for _ in range(max_iter):
            f = residuals(x)
            if f is None:
                return None
            if np.max(np.abs(f)) < tol:
                return float(x[0]), float(x[1]), float(x[2])

            J = np.zeros((3, 3))
            for j in range(3):
                xp = x.copy()
                xp[j] += eps
                fp = residuals(xp)
                if fp is None:
                    return None
                J[:, j] = (fp - f) / eps

            try:
                dx = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                return None
            step = np.max(np.abs(dx))
            damp = min(1.0, 0.5 / step) if step > 0.5 else 1.0
            x = x + damp * dx
        return None

    @classmethod
    def _solve_c1_c2_c3(cls, target0: float, target45: float, target90: float,
                         C6: float, a: float,
                         tol: float = 1e-10, max_iter: int = 60) -> Optional[Tuple[float, float, float]]:
        """
        Newton-Raphson for (C1, C2, C3), with C4 = C5 = 1 and C6 fixed, so
        that calc_equivalent_stress's UNIT value (what calc_sigma_from_
        theta_stress computes before applying any external scale) matches
        target0/target45/target90 at theta_stress = 0/45/90 degrees.

        The targets are sigma0/sigma_theta ratios from experimental
        uniaxial tension tests at 0/45/90 degrees (target0 = sigma0/sigma0
        = 1 always -- but still a genuine equation here, since C1 is NOT
        fixed at 1 as in _solve_c2_c3_c6: it is what pins the absolute
        scale of (C1, C2, C3) relative to sigma0). This fits the
        criterion's predicted STRESS anisotropy directly, unlike
        _solve_c2_c3_c6, which instead fits the flow direction (Lankford
        ratios).
        """
        probe = cls(a=a)
        probe.C4, probe.C5, probe.C6 = 1.0, 1.0, C6
        targets = ((0.0, target0), (45.0, target45), (90.0, target90))

        def residuals(x):
            probe.C1, probe.C2, probe.C3 = x
            out = []
            for deg, target in targets:
                th = np.deg2rad(deg)
                c2, s2, cs = np.cos(th) ** 2, np.sin(th) ** 2, np.sin(th) * np.cos(th)
                principal = probe._transformed_principal_stresses(c2, s2, cs)
                if principal is None:
                    return None
                s1, s2v, s3v = principal
                inner = abs(s1 - s2v) ** a + abs(s2v - s3v) ** a + abs(s3v - s1) ** a
                out.append((0.5 * inner) ** (1.0 / a) - target)
            return np.array(out)

        # Initial guess (C1=1, C2=0, C3=0): avoids starting at the fully
        # symmetric point C1=C2=C3(=C4=1), where S1=S2=S3 at theta=0 (a
        # degenerate, poorly-conditioned point for Newton's method here).
        x = np.array([1.0, 0.0, 0.0])
        eps = 1e-6
        for _ in range(max_iter):
            f = residuals(x)
            if f is None:
                return None
            if np.max(np.abs(f)) < tol:
                return float(x[0]), float(x[1]), float(x[2])

            J = np.zeros((3, 3))
            for j in range(3):
                xp = x.copy()
                xp[j] += eps
                fp = residuals(xp)
                if fp is None:
                    return None
                J[:, j] = (fp - f) / eps

            try:
                dx = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                return None
            step = np.max(np.abs(dx))
            damp = min(1.0, 0.5 / step) if step > 0.5 else 1.0
            x = x + damp * dx
        return None

    @classmethod
    def _solve_c1_c2_c3_c6(cls, target0: float, target45: float, target90: float,
                            r_target: float, r_theta_deg: float, a: float,
                            tol: float = 1e-10, max_iter: int = 80
                            ) -> Optional[Tuple[float, float, float, float]]:
        """
        Newton-Raphson for (C1, C2, C3, C6), with C4 = C5 = 1, matching the
        3 stress-ratio targets (as in _solve_c1_c2_c3) PLUS the Lankford
        coefficient r_target at theta_stress = r_theta_deg (0 or 90
        degrees -- a principal/material axis, so no shear is involved
        there; only the 45-degree stress equation constrains C6 here).

        Used only to obtain a C6 ESTIMATE (see
        calc_barlat_coefficients_from_stress_and_lankford): a single
        r-value (r0 OR r90, not both at once) does not pin down C6
        reliably on its own -- the two estimates from r0 and r90 are
        averaged there instead, then C1, C2, C3 are re-solved
        (_solve_c1_c2_c3) with that averaged C6 held fixed.
        """
        probe = cls(a=a)
        probe.C4, probe.C5 = 1.0, 1.0
        stress_targets = ((0.0, target0), (45.0, target45), (90.0, target90))

        def residuals(x):
            probe.C1, probe.C2, probe.C3, probe.C6 = x
            out = []
            for deg, target in stress_targets:
                th = np.deg2rad(deg)
                c2, s2, cs = np.cos(th) ** 2, np.sin(th) ** 2, np.sin(th) * np.cos(th)
                principal = probe._transformed_principal_stresses(c2, s2, cs)
                if principal is None:
                    return None
                s1, s2v, s3v = principal
                inner = abs(s1 - s2v) ** a + abs(s2v - s3v) ** a + abs(s3v - s1) ** a
                out.append((0.5 * inner) ** (1.0 / a) - target)

            r = probe.calc_lankford_from_theta_stress(PointState(theta_stress=np.deg2rad(r_theta_deg)))
            if r is None:
                return None
            out.append(r - r_target)
            return np.array(out)

        # Initial guess (C1=1, C2=1, C3=0.9, C6=1): avoids the degenerate
        # point C1=C2=C3, where S1=S2=S3 at theta=0 (Mohr-circle radius
        # R=0, where calc_lankford_from_theta_stress is undefined).
        x = np.array([1.0, 1.0, 0.9, 1.0])
        eps = 1e-6
        for _ in range(max_iter):
            f = residuals(x)
            if f is None:
                return None
            if np.max(np.abs(f)) < tol:
                return float(x[0]), float(x[1]), float(x[2]), float(x[3])

            J = np.zeros((4, 4))
            for j in range(4):
                xp = x.copy()
                xp[j] += eps
                fp = residuals(xp)
                if fp is None:
                    return None
                J[:, j] = (fp - f) / eps

            try:
                dx = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                return None
            step = np.max(np.abs(dx))
            damp = min(1.0, 0.5 / step) if step > 0.5 else 1.0
            x = x + damp * dx
        return None

    @classmethod
    def _solve_c1_c2_c3_c6_full(cls, target0: float, target45: float, target90: float,
                                 r0: float, r45: float, r90: float, a: float,
                                 tol: float = 1e-10, max_iter: int = 100
                                 ) -> Optional[Tuple[float, float, float, float]]:
        """
        Gauss-Newton least-squares fit for (C1, C2, C3, C6), with
        C4 = C5 = 1, matching all 3 stress-ratio targets (as in
        _solve_c1_c2_c3) AND all 3 Lankford coefficients r0, r45, r90 at once -- 6 target conditions for 4
        free coefficients (C1 is free here, unlike the r-only calibration
        in _solve_c2_c3_c6, since the stress ratios pin the absolute
        scale). This is overdetermined, so it finds the least-squares-
        optimal compromise across all 6 experimental data points, rather
        than _solve_c1_c2_c3_c6's approach of using only ONE r-value (r0
        XOR r90) and averaging two separate exactly-determined solves.
        """
        probe = cls(a=a)
        probe.C4, probe.C5 = 1.0, 1.0
        stress_targets = ((0.0, target0), (45.0, target45), (90.0, target90))
        lankford_targets = ((0.0, r0), (45.0, r45), (90.0, r90))

        def residuals(x):
            probe.C1, probe.C2, probe.C3, probe.C6 = x
            out = []
            for deg, target in stress_targets:
                th = np.deg2rad(deg)
                c2, s2, cs = np.cos(th) ** 2, np.sin(th) ** 2, np.sin(th) * np.cos(th)
                principal = probe._transformed_principal_stresses(c2, s2, cs)
                if principal is None:
                    return None
                s1, s2v, s3v = principal
                inner = abs(s1 - s2v) ** a + abs(s2v - s3v) ** a + abs(s3v - s1) ** a
                out.append((0.5 * inner) ** (1.0 / a) - target)
            for deg, target in lankford_targets:
                r = probe.calc_lankford_from_theta_stress(PointState(theta_stress=np.deg2rad(deg)))
                if r is None:
                    return None
                out.append(r - target)
            return np.array(out)

        # Initial guess (C1=1, C2=1, C3=0.9, C6=1): same starting point as
        # _solve_c1_c2_c3_c6, for the same reason (avoids the degenerate
        # C1=C2=C3 point).
        x = np.array([1.0, 1.0, 0.9, 1.0])
        eps = 1e-6
        for _ in range(max_iter):
            f = residuals(x)
            if f is None:
                return None

            J = np.zeros((6, 4))
            for j in range(4):
                xp = x.copy()
                xp[j] += eps
                fp = residuals(xp)
                if fp is None:
                    return None
                J[:, j] = (fp - f) / eps

            dx, *_ = np.linalg.lstsq(J, -f, rcond=None)
            step = np.max(np.abs(dx))
            if step < tol:
                return float(x[0]), float(x[1]), float(x[2]), float(x[3])
            damp = min(1.0, 0.5 / step) if step > 0.5 else 1.0
            x = x + damp * dx
        return float(x[0]), float(x[1]), float(x[2]), float(x[3])

    def calc_barlat_coefficients_from_stress_and_lankford_full(
            self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Combined calibration using all 6 experimental data points at once
        (r0, r45, r90, sigma0, sigma45, sigma90): a least-squares fit of
        (C1, C2, C3, C6) via _solve_c1_c2_c3_c6_full. Preferred over
        calc_barlat_coefficients_from_stress_and_lankford (which only uses
        r0 and r90, ignoring r45, and averages two separate partial
        solves) whenever r45 is also available -- see
        ensure_barlat_coefficients for the priority order between the
        different calibration strategies.
        """
        if None in (self.r0, self.r45, self.r90, self.sigma0, self.sigma45, self.sigma90):
            return None
        if self.sigma0 == 0 or self.sigma45 == 0 or self.sigma90 == 0:
            return None

        target0 = 1.0
        target45 = self.sigma0 / self.sigma45
        target90 = self.sigma0 / self.sigma90

        solved = self._solve_c1_c2_c3_c6_full(target0, target45, target90,
                                               self.r0, self.r45, self.r90, self.a)
        if solved is None:
            return None
        C1, C2, C3, C6 = solved
        return C1, C2, C3, 1.0, 1.0, C6

    def calc_barlat_coefficients_from_stress_and_lankford(
            self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Alternative calibration (stress-based) to
        calc_barlat_coefficients_from_lankford (flow-rule-based, C1 = 1
        fixed): needs r0, r90 AND the 3 experimental uniaxial yield
        stresses sigma0, sigma45, sigma90 (r45 is not used). Reproduces a
        hand calibration procedure (a = 8):

          1. Two 4-equation/4-unknown solves for (C1, C2, C3, C6), with
             C4 = C5 = 1, each matching the 3 stress ratios (sigma0/
             sigma0 = 1, sigma0/sigma45, sigma0/sigma90) plus ONE Lankford
             coefficient: r0 (at theta_stress = 0) in the first, r90 (at
             theta_stress = 90) in the second (_solve_c1_c2_c3_c6). Each
             solve gives its own (C1, C2, C3, C6), slightly different
             from the other.
          2. C6 is taken as the AVERAGE of the two resulting C6 estimates
             -- neither r0 alone nor r90 alone pins it down reliably
             (each also gives a different, imprecise C1, C2, C3), but
             the mean C6 gives a serviceable compromise.
          3. With that C6 fixed (and C4 = C5 = 1), a final 3-equation/
             3-unknown solve (_solve_c1_c2_c3) matches the 3 stress
             ratios exactly; r0 and r90 are no longer matched exactly, but
             come out reasonably close to the experimental values -- worth
             checking after calibration (e.g. via
             calc_lankford_from_theta_stress at theta_stress = 0 and 90).
        """
        if None in (self.r0, self.r90, self.sigma0, self.sigma45, self.sigma90):
            return None
        if self.sigma0 == 0 or self.sigma45 == 0 or self.sigma90 == 0:
            return None

        target0 = 1.0
        target45 = self.sigma0 / self.sigma45
        target90 = self.sigma0 / self.sigma90

        solved_r0 = self._solve_c1_c2_c3_c6(target0, target45, target90, self.r0, 0.0, self.a)
        solved_r90 = self._solve_c1_c2_c3_c6(target0, target45, target90, self.r90, 90.0, self.a)
        if solved_r0 is None or solved_r90 is None:
            return None
        C6 = 0.5 * (solved_r0[3] + solved_r90[3])

        solved = self._solve_c1_c2_c3(target0, target45, target90, C6, self.a)
        if solved is None:
            return None
        C1, C2, C3 = solved
        return C1, C2, C3, 1.0, 1.0, C6

    def calc_barlat_coefficients_from_stress(
            self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Calibration from the 3 experimental uniaxial yield stresses alone
        (sigma0, sigma45, sigma90), with NO Lankford data at all: last
        resort among the stress-based strategies, for when r0/r45/r90
        aren't available (or not enough of them are).

        Only 3 stress-ratio equations (sigma0/sigma0 = 1, sigma0/sigma45,
        sigma0/sigma90) are available for 4 free coefficients (C1, C2,
        C3, C6), so C6 has to be pinned externally to make the remaining
        3x3 system (_solve_c1_c2_c3) well posed -- unlike
        calc_barlat_coefficients_from_stress_and_lankford(_full), which
        gets that 4th equation from r0 (and/or r90, r45). Uses self.C6 if
        already supplied (an externally chosen or separately calibrated
        shear coefficient -- e.g. from a shear test, or carried over from
        a previous calibration of the same material), otherwise defaults
        to the isotropic C6 = 1.0.
        """
        if None in (self.sigma0, self.sigma45, self.sigma90):
            return None
        if self.sigma0 == 0 or self.sigma45 == 0 or self.sigma90 == 0:
            return None

        C6 = self.C6 if self.C6 is not None else 1.0

        target0 = 1.0
        target45 = self.sigma0 / self.sigma45
        target90 = self.sigma0 / self.sigma90

        solved = self._solve_c1_c2_c3(target0, target45, target90, C6, self.a)
        if solved is None:
            return None
        C1, C2, C3 = solved
        return C1, C2, C3, 1.0, 1.0, C6

    def ensure_barlat_coefficients(self) -> None:
        """
        Fill missing C1..C6 values if required, trying each available
        calibration strategy in turn and keeping the first that succeeds
        (each one already checks internally whether ITS OWN required
        fields are all supplied, returning None otherwise -- see each
        method's docstring), in order of how much experimental data it
        uses, most informed first:

          1. calc_barlat_coefficients_from_stress_and_lankford_full
             (r0, r45, r90, sigma0, sigma45, sigma90 -- all 6)
          2. calc_barlat_coefficients_from_stress_and_lankford
             (r0, r90, sigma0, sigma45, sigma90 -- r45 not used)
          3. calc_barlat_coefficients_from_stress
             (sigma0, sigma45, sigma90 only; C6 pinned to self.C6, or 1.0)
          4. calc_barlat_coefficients_from_lankford
             (r0, r45, r90 only; the shape-only, unnormalised-scale fit)
        """
        if any(v is None for v in (self.C1, self.C2, self.C3, self.C4, self.C5, self.C6)):
            strategies = (
                self.calc_barlat_coefficients_from_stress_and_lankford_full,
                self.calc_barlat_coefficients_from_stress_and_lankford,
                self.calc_barlat_coefficients_from_stress,
                self.calc_barlat_coefficients_from_lankford,
            )
            coeffs = None
            for strategy in strategies:
                coeffs = strategy()
                if coeffs is not None:
                    break
            if coeffs is not None:
                self.C1 = self.C1 if self.C1 is not None else coeffs[0]
                self.C2 = self.C2 if self.C2 is not None else coeffs[1]
                self.C3 = self.C3 if self.C3 is not None else coeffs[2]
                self.C4 = self.C4 if self.C4 is not None else coeffs[3]
                self.C5 = self.C5 if self.C5 is not None else coeffs[4]
                self.C6 = self.C6 if self.C6 is not None else coeffs[5]

    def _ensure_calibration(self) -> None:
        self.ensure_barlat_coefficients()

    def _transformed_principal_stresses(self, sigma_xx: float, sigma_yy: float,
                                         tau_xy: float = 0.0) -> Optional[Tuple[float, float, float]]:
        """
        Principal values (S1, S2, S3) of the Yld91 linearly-transformed
        stress tensor, per Barlat et al. (1991): S = L*sigma, with sigma
        the full (3D) Cauchy stress tensor in Voigt notation (sigma_xx,
        sigma_yy, sigma_zz, tau_yz, tau_zx, tau_xy) and

            L = [[ (C2+C3)/3, -C3/3,     -C2/3,     0,  0,  0 ],
                 [ -C3/3,     (C1+C3)/3, -C1/3,      0,  0,  0 ],
                 [ -C2/3,     -C1/3,     (C1+C2)/3,  0,  0,  0 ],
                 [ 0, 0, 0, C4, 0, 0 ],
                 [ 0, 0, 0, 0, C5, 0 ],
                 [ 0, 0, 0, 0, 0, C6 ]]

        Under plane stress (sigma_zz = tau_yz = tau_zx = 0, as everywhere
        in this module), C4 and C5 multiply components that are ALWAYS
        ZERO -- they have NO EFFECT on anything computed here (hence
        being fixed to 1 everywhere they are used), and only C1, C2, C3
        (in-plane normal coefficients) and C6 (in-plane shear, via tau_xy)
        actually matter. S_zz is generally NONZERO even though sigma_zz
        itself is 0, since L couples all 3 normal directions together;
        this is the term the OLD (pre-fix) version of this method missed
        entirely, along with using C5 rather than C6 for shear.

        Given that, S_xx, S_yy, S_xy are diagonalised via Mohr's circle
        (S_zz is already principal, not coupled to in-plane shear) to
        give S1 >= S2, and S3 = S_zz -- the ORDER does not matter, since
        calc_equivalent_stress's phi is symmetric in permutations of
        (S1, S2, S3).
        """
        self.ensure_barlat_coefficients()
        if None in (self.C1, self.C2, self.C3):
            return None
        C6 = self.C6 if self.C6 is not None else 0.0

        S_xx = (self.C2 + self.C3) / 3.0 * sigma_xx - self.C3 / 3.0 * sigma_yy
        S_yy = -self.C3 / 3.0 * sigma_xx + (self.C1 + self.C3) / 3.0 * sigma_yy
        S_zz = -self.C2 / 3.0 * sigma_xx - self.C1 / 3.0 * sigma_yy
        S_xy = C6 * tau_xy

        center = (S_xx + S_yy) / 2.0
        radius = sqrt(((S_xx - S_yy) / 2.0) ** 2 + S_xy ** 2)
        return center + radius, center - radius, S_zz

    def calc_equivalent_stress(self, state: PointState) -> Optional[float]:
        """
        Equivalent stress for Barlat 1991 (Yld91):

            phi(S1, S2, S3) = |S1-S2|^a + |S2-S3|^a + |S3-S1|^a = 2 * sigmaEQ^a

        S1, S2, S3 are a LINEAR transformation of the material-axes
        components (sigma_xx, sigma_yy, tau_xy) (see _stress_components and
        _transformed_principal_stresses): since phi is convex and
        isotropic, this composition is convex for any C1, C2, C3, C6 (a
        necessary condition of every admissible yield criterion).
        """
        comps = self._stress_components(state)
        if comps is None:
            return None
        return self._equivalent_stress_xy(*comps)

    def _equivalent_stress_xy(self, sigma_xx: float, sigma_yy: float, tau_xy: float) -> Optional[float]:
        principal = self._transformed_principal_stresses(sigma_xx, sigma_yy, tau_xy)
        if principal is None:
            return None
        s1, s2, s3 = principal
        inner = abs(s1 - s2) ** self.a + abs(s2 - s3) ** self.a + abs(s3 - s1) ** self.a
        return (0.5 * inner) ** (1.0 / self.a)

    def _flow_gradient_xy(self, sigma_xx: float, sigma_yy: float,
                           tau_xy: float) -> Optional[Tuple[float, float, float]]:
        """
        Gradient (gx, gy, gxy) = d(sigmaEQ)/d(sigma_xx, sigma_yy, tau_xy)
        of the Yld91 associated flow rule at a general (possibly sheared)
        in-plane stress state, via _transformed_principal_stresses (S1,
        S2, S3, chain-ruled through S_xx, S_yy, S_zz, S_xy). A common
        positive factor is dropped (it does not affect beta = gy/gx nor
        any flow-direction angle computed from it).
        """
        self.ensure_barlat_coefficients()
        if None in (self.C1, self.C2, self.C3):
            return None
        C6 = self.C6 if self.C6 is not None else 0.0

        S_xx = (self.C2 + self.C3) / 3.0 * sigma_xx - self.C3 / 3.0 * sigma_yy
        S_yy = -self.C3 / 3.0 * sigma_xx + (self.C1 + self.C3) / 3.0 * sigma_yy
        S_zz = -self.C2 / 3.0 * sigma_xx - self.C1 / 3.0 * sigma_yy
        S_xy = C6 * tau_xy

        center = (S_xx + S_yy) / 2.0
        u = (S_xx - S_yy) / 2.0
        R = sqrt(u ** 2 + S_xy ** 2)
        if R == 0:
            return None
        S1, S2, S3 = center + R, center - R, S_zz

        # d(|S1-S2|^a + |S2-S3|^a + |S3-S1|^a) / dS1, dS2, dS3.
        dphi_dS1 = self._signed_power(S1 - S2, self.a - 1) - self._signed_power(S3 - S1, self.a - 1)
        dphi_dS2 = -self._signed_power(S1 - S2, self.a - 1) + self._signed_power(S2 - S3, self.a - 1)
        dphi_dS3 = -self._signed_power(S2 - S3, self.a - 1) + self._signed_power(S3 - S1, self.a - 1)

        # S1, S2 = center(S_xx, S_yy) +- sqrt(u(S_xx, S_yy)^2 + S_xy^2), by
        # Mohr's circle; S3 = S_zz directly (not coupled to S_xy).
        dS1_dSxx, dS2_dSxx = 0.5 + u / (2.0 * R), 0.5 - u / (2.0 * R)
        dS1_dSyy, dS2_dSyy = 0.5 - u / (2.0 * R), 0.5 + u / (2.0 * R)
        dS1_dSxy, dS2_dSxy = S_xy / R, -S_xy / R

        # Chain rule through S_xx, S_yy, S_zz (linear in sigma_xx, sigma_yy)
        # and S_xy = C6*tau_xy.
        dSxx_dxx, dSxx_dyy = (self.C2 + self.C3) / 3.0, -self.C3 / 3.0
        dSyy_dxx, dSyy_dyy = -self.C3 / 3.0, (self.C1 + self.C3) / 3.0
        dSzz_dxx, dSzz_dyy = -self.C2 / 3.0, -self.C1 / 3.0

        gx = dphi_dS1 * (dS1_dSxx * dSxx_dxx + dS1_dSyy * dSyy_dxx) \
            + dphi_dS2 * (dS2_dSxx * dSxx_dxx + dS2_dSyy * dSyy_dxx) \
            + dphi_dS3 * dSzz_dxx
        gy = dphi_dS1 * (dS1_dSxx * dSxx_dyy + dS1_dSyy * dSyy_dyy) \
            + dphi_dS2 * (dS2_dSxx * dSxx_dyy + dS2_dSyy * dSyy_dyy) \
            + dphi_dS3 * dSzz_dyy
        gxy = (dphi_dS1 * dS1_dSxy + dphi_dS2 * dS2_dSxy) * C6
        return gx, gy, gxy

    def _flow_gradient(self, alpha: float) -> Optional[Tuple[float, float]]:
        """
        Gradient (g1, g2) = d(sigmaEQ)/d(sigma1, sigma2) of the Yld91
        associated flow rule, evaluated at (sigma1=1, sigma2=alpha,
        tau_xy=0) -- just _flow_gradient_xy at that (principal, shear-
        free) state, valid because S1, S2, S3 (and therefore the yield
        function) are homogeneous of degree 1 in (sigma1, sigma2), so the
        gradient direction does not depend on scale.
        """
        grad = self._flow_gradient_xy(1.0, alpha, 0.0)
        if grad is None:
            return None
        return grad[0], grad[1]

    def calc_alpha_from_beta(self, state: PointState) -> Optional[float]:
        """
        Invert the Yld91 flow rule numerically: alpha = sigma2/sigma1 from a
        known beta.

        beta = g2/g1 blows up to +-infinity wherever the flow direction
        crosses g1 = 0 (deps1 = 0), so a plain bisection on beta is not
        reliable. Instead, the search is carried out on the flow-DIRECTION
        angle atan2(g2, g1): as alpha sweeps every real value (equivalently,
        theta = atan(alpha) sweeps (-pi/2, pi/2)), this angle increases
        continuously and monotonically through exactly pi radians, since the
        Yld91 potential is centrally symmetric. That makes the angle safe to
        bisect, unlike the ratio itself.
        """
        self.ensure_barlat_coefficients()
        if state.beta is None or None in (self.C1, self.C2, self.C3):
            return None
        if not self._is_rd_aligned(state.theta_stress):
            return self._alpha_from_beta_theta(state.beta, state.theta_stress)

        def flow_angle(theta: float) -> Optional[float]:
            grad = self._flow_gradient(np.tan(theta))
            if grad is None:
                return None
            g1, g2 = grad
            return np.arctan2(g2, g1)

        eps = 1e-9
        theta_lo, theta_hi = -0.5 * np.pi + eps, 0.5 * np.pi - eps
        phi_lo, phi_hi = flow_angle(theta_lo), flow_angle(theta_hi)
        if phi_lo is None or phi_hi is None:
            return None

        # Bring the target angle into the (phi_lo, phi_hi) branch (they span pi).
        target_phi = np.arctan2(state.beta, 1.0)
        while target_phi < phi_lo:
            target_phi += np.pi
        while target_phi > phi_hi:
            target_phi -= np.pi

        lo, hi = theta_lo, theta_hi
        f_lo, f_hi = phi_lo - target_phi, phi_hi - target_phi
        if f_lo == 0:
            return float(np.tan(lo))
        if f_hi == 0:
            return float(np.tan(hi))
        if (f_lo > 0) == (f_hi > 0):
            return None

        # 40 iterations (down from a fixed 80 with no early exit) plus a bracket-
        # width tolerance: theta is resolved to ~1e-8 rad well before either
        # limit is reached, which is far tighter than this model needs.
        for _ in range(40):
            if hi - lo < 1e-8:
                break
            mid = 0.5 * (lo + hi)
            f_mid = flow_angle(mid)
            if f_mid is None:
                return None
            f_mid -= target_phi
            if f_mid == 0:
                return float(np.tan(mid))
            if (f_mid > 0) == (f_lo > 0):
                lo, f_lo = mid, f_mid
            else:
                hi, f_hi = mid, f_mid
        return float(np.tan(0.5 * (lo + hi)))

    def calc_beta_from_deviatoric_stresses(self, state: PointState) -> Optional[float]:
        """
        Yld91 flow ratio computed from the principal stress state, via
        calc_beta_from_alpha.

        The base class's calc_beta_from_deviatoric_stresses implements the
        ISOTROPIC (von Mises) flow rule beta = (2*s2-s1)/(2*s1-s2); left
        unoverridden, solve() would resolve beta from sigma1/sigma2 using
        that isotropic formula BEFORE the anisotropic calc_beta_from_alpha
        ever gets a chance to run (both need only sigma1, sigma2 -- and
        this one is tried first), silently discarding the Yld91 anisotropy
        for beta (and everything solve() derives from it: deps1, deps2,
        deps3, depsEQ). Hill48_PS/Hill79_PS avoid this the same way, via
        HillLankfordCriterion.calc_beta_from_deviatoric_stresses.
        """
        self.ensure_barlat_coefficients()
        if None not in (state.sigma1, state.sigma2):
            if state.sigma1 == 0:
                return None
            return self.calc_beta_from_alpha(state)
        return None

    def calc_beta_from_alpha(self, state: PointState) -> Optional[float]:
        """Strain increment ratio beta = deps2 / deps1 from the Yld91 associated flow rule."""
        if state.alpha is None:
            return None
        if not self._is_rd_aligned(state.theta_stress):
            return self._beta_from_alpha_theta(state.alpha, state.theta_stress)
        grad = self._flow_gradient(state.alpha)
        if grad is None:
            return None
        g1, g2 = grad
        return g2 / g1 if g1 != 0 else None

    def calc_sigma1_from_equivalent_stress_and_alpha(self, state: PointState) -> Optional[float]:
        """Inverse plane-stress relation for sigma1 using the Yld91 transformed-stress form."""
        self.ensure_barlat_coefficients()
        if None not in (state.sigmaEQ, state.alpha) and not self._is_rd_aligned(state.theta_stress):
            return self._sigma1_from_equivalent_stress_alpha_theta(state.sigmaEQ, state.alpha, state.theta_stress)
        if state.sigmaEQ is not None and state.alpha is not None:
            principal = self._transformed_principal_stresses(1.0, state.alpha)
            if principal is not None:
                s1, s2, s3 = principal
                inner = abs(s1 - s2) ** self.a + abs(s2 - s3) ** self.a + abs(s3 - s1) ** self.a
                scale = (0.5 * inner) ** (1.0 / self.a)
                if scale != 0:
                    return state.sigmaEQ / scale
        return None

    def calc_sigma_from_theta_stress(self, state: PointState) -> Optional[float]:
        """
        Stress required to yield in a uniaxial tension test at angle
        theta_stress from principal direction 1, generalising
        calc_equivalent_stress to a sheared stress state via
        _transformed_principal_stresses. In the (1, 2) axes such a test is
        the unit-magnitude sheared state
            sigma1' = cos^2(theta_stress), sigma2' = sin^2(theta_stress),
            tau12'  = sin(theta_stress) cos(theta_stress),
        which reduces exactly to calc_equivalent_stress's principal-stress
        formula at theta_stress = 0 or 90 degrees (tau12' = 0).

        If state.sigmaEQ is already set (e.g. from a previous solve() at
        theta_stress = 0), the result is scaled to that reference stress;
        otherwise it is left normalised to sigma(0) = 1.
        """
        if state.theta_stress is None:
            return None

        c2, s2 = np.cos(state.theta_stress) ** 2, np.sin(state.theta_stress) ** 2
        cs = np.sin(state.theta_stress) * np.cos(state.theta_stress)
        principal = self._transformed_principal_stresses(c2, s2, cs)
        if principal is None:
            return None
        S1, S2, S3 = principal

        inner = abs(S1 - S2) ** self.a + abs(S2 - S3) ** self.a + abs(S3 - S1) ** self.a
        sigma_eq_unit = (0.5 * inner) ** (1.0 / self.a)
        if sigma_eq_unit == 0:
            return None

        scale = state.sigmaEQ if state.sigmaEQ is not None else 1.0
        return scale / sigma_eq_unit

    def calc_lankford_from_theta_stress(self, state: PointState) -> Optional[float]:
        """
        Lankford coefficient r for a uniaxial tension test at angle
        theta_stress from principal direction 1, from the associated flow rule
        (_flow_gradient_xy) at the same sheared stress state used in
        calc_sigma_from_theta_stress. See Hill48_PS.calc_lankford_from_theta_stress
        for the general derivation of this formula from (gx, gy, gxy).
        """
        if state.theta_stress is None:
            return None

        c, s = np.cos(state.theta_stress), np.sin(state.theta_stress)
        c2, s2, cs = c ** 2, s ** 2, s * c
        grad = self._flow_gradient_xy(c2, s2, cs)
        if grad is None:
            return None
        gx, gy, gxy = grad

        denominator = -(gx + gy)
        if denominator == 0:
            return None
        numerator = gx * s2 + gy * c2 - gxy * cs
        return numerator / denominator

    def calc_psi_noncoax_from_theta_stress(self, state: PointState) -> Optional[float]:
        """
        Angle (radians) between the loading direction (the applied
        stress's own principal direction) and the true principal strain-
        increment direction, for a general stress ratio alpha =
        sigma2/sigma1 (defaults to 0, i.e. uniaxial tension, if not set)
        applied at angle theta_stress from principal direction 1. See
        Hill48_PS.calc_psi_noncoax_from_theta_stress for the general derivation and the
        rotated-state formula; this uses the same _flow_gradient_xy
        already used by calc_lankford_from_theta_stress:

            psi_noncoax = wrap[0.5 * atan2(gxy, gx - gy) - theta_stress]

        (the raw atan2 term is the strain-increment principal direction
        measured from material axis 1, since sigma_xx, sigma_yy, tau_xy
        are given in that frame; subtracting theta_stress re-references it to
        the loading direction, and wrap = _wrap_half_turn).
        """
        if state.theta_stress is None:
            return None
        alpha = state.alpha if state.alpha is not None else 0.0

        s, c = np.sin(state.theta_stress), np.cos(state.theta_stress)
        s2, c2, cs = s ** 2, c ** 2, s * c
        sigma_xx = c2 + alpha * s2
        sigma_yy = s2 + alpha * c2
        tau_xy = (1.0 - alpha) * cs

        grad = self._flow_gradient_xy(sigma_xx, sigma_yy, tau_xy)
        if grad is None:
            return None
        gx, gy, gxy = grad
        psi_noncoax_from_axis1 = 0.5 * float(np.arctan2(gxy, gx - gy))
        return self._wrap_half_turn(psi_noncoax_from_axis1 - state.theta_stress)


@dataclass
class YLD2000_PS(YieldCriterionPlaneStress):
    """
    Anisotropic Barlat et al. (2003) Yld2000-2d yield criterion (plane
    stress), Int. J. Plasticity 19, 1297-1319:

        phi = phi' + phi'' = 2 * sigmaEQ^a
        phi'  = |X'_1 - X'_2|^a
        phi'' = |2 X''_2 + X''_1|^a + |2 X''_1 + X''_2|^a

    X'_i, X''_i are the principal values of the two linearly transformed
    stress tensors X' = L' sigma, X'' = L'' sigma (sigma = (sigma_xx,
    sigma_yy, tau_xy) in material axes, 1 = RD, 2 = TD):

        L'  = [[ 2*a1/3, -a1/3,   0  ],
               [ -a2/3,  2*a2/3,  0  ],
               [  0,      0,      a7 ]]
        L'' = 1/9 * [[ -2*a3 + 2*a4 + 8*a5 - 2*a6,  a3 - 4*a4 - 4*a5 + 4*a6,  0     ],
                     [ 4*a3 - 4*a4 - 4*a5 + a6,  -2*a3 + 8*a4 + 2*a5 - 2*a6,  0     ],
                     [ 0,                         0,                          9*a8  ]]

    (a1..a8 = alpha1..alpha8). With all alphas = 1, L' = L'' is the
    deviator and phi reduces to Hosford's |s1-s2|^a + |s1|^a + |s2|^a
    (von Mises for a = 2). a = 6 for BCC, 8 for FCC metals.

    Calibration (calc_yld2000_coefficients, run lazily by solve() and the
    other methods): the 8 alphas from 8 experimental values, the uniaxial
    yield stresses and Lankford coefficients at 0/45/90 degrees plus the
    equibiaxial yield stress sigmab and the biaxial anisotropy rb =
    deps_yy/deps_xx at equibiaxial tension (e.g. from a bulge or disk
    compression test). sigmaEQ is normalised to sigma0 (unit RD tension
    gives sigmaEQ = 1), as in the other criteria of this module.
    Alternatively, the alphas can be given directly.
    """

    a: float = 8.0
    alpha1: Optional[float] = None
    alpha2: Optional[float] = None
    alpha3: Optional[float] = None
    alpha4: Optional[float] = None
    alpha5: Optional[float] = None
    alpha6: Optional[float] = None
    alpha7: Optional[float] = None
    alpha8: Optional[float] = None
    # experimental calibration data (fixed material data, not a state)
    r0: Optional[float] = None
    r45: Optional[float] = None
    r90: Optional[float] = None
    rb: Optional[float] = None
    sigma0: Optional[float] = None
    sigma45: Optional[float] = None
    sigma90: Optional[float] = None
    sigmab: Optional[float] = None

    _ALPHAS = ("alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6", "alpha7", "alpha8")

    # ---- alternative parameterisation (Bressan et al. 2025) -------------
    # The coefficients a1..a8 of Bressan et al., Int. J. Mater. Form. 18:64 (2025), eq. (13):
    #   phi = |a1 sxx + a2 syy + D''|^a / 2^a + |a1 sxx + a2 syy - D''|^a / 2^a + D'^a
    #   D'' = sqrt((a3 sxx - a4 syy)^2 + (2 a8 txy)^2),  D' = sqrt((a5 sxx - a6 syy)^2 + (2 a7 txy)^2)
    # are combinations of the entries of L', L'' (eqs. 11-12):
    #   a1 = 3 (L''11 + L''21), a2 = 3 (L''22 + L''12), a3 = L''11 - L''21, a4 = L''22 - L''12,
    #   a5 = L'11 - L'21, a6 = L'22 - L'12, a7 = L'66, a8 = L''66   (all ones = isotropic).
    @classmethod
    def from_bressan_coefficients(cls, coeffs, **kwargs) -> "YLD2000_PS":
        """
        Criterion from the coefficients (a1, ..., a8) of Bressan et al. (2025), eq. (13) (see
        above); kwargs are the other fields (exponent a, calibration data, label).
        """
        b1, b2, b3, b4, b5, b6, b7, b8 = (float(c) for c in coeffs)
        alpha1, alpha2 = 2 * b5 - b6, 2 * b6 - b5          # from L'11 - L'21 and L'22 - L'12
        # L''11 + L''21 = b1/3, L''11 - L''21 = b3; L''22 + L''12 = b2/3, L''22 - L''12 = b4
        L11, L21 = (b1 / 3 + b3) / 2, (b1 / 3 - b3) / 2
        L22, L12 = (b2 / 3 + b4) / 2, (b2 / 3 - b4) / 2
        M = np.array([[-2, 2, 8, -2], [1, -4, -4, 4], [4, -4, -4, 1], [-2, 8, 2, -2]]) / 9.0
        alpha3, alpha4, alpha5, alpha6 = np.linalg.solve(M, [L11, L12, L21, L22])
        return cls(alpha1=alpha1, alpha2=alpha2, alpha3=float(alpha3), alpha4=float(alpha4),
                   alpha5=float(alpha5), alpha6=float(alpha6), alpha7=b7, alpha8=b8, **kwargs)

    def bressan_coefficients(self) -> Optional[Tuple[float, ...]]:
        """The coefficients (a1, ..., a8) of Bressan et al. (2025), eq. (13), of this criterion."""
        L = self._linear_transforms()
        if L is None:
            return None
        L1, L2 = L
        return (3 * (L2[0, 0] + L2[1, 0]), 3 * (L2[1, 1] + L2[0, 1]), L2[0, 0] - L2[1, 0], L2[1, 1] - L2[0, 1],
                L1[0, 0] - L1[1, 0], L1[1, 1] - L1[0, 1], L1[2, 2], L2[2, 2])

    # ---- calibration ----------------------------------------------------
    def _alphas(self) -> Optional[Tuple[float, ...]]:
        values = tuple(getattr(self, n) for n in self._ALPHAS)
        return None if None in values else values

    def calc_yld2000_coefficients(self, tol: float = 1e-10, max_iter: int = 100
                                  ) -> Optional[Tuple[float, ...]]:
        """
        alpha1..alpha8 from (sigma0, sigma45, sigma90, sigmab, r0, r45,
        r90, rb) by Newton-Raphson (Jacobian by finite differences),
        starting from the isotropic point (all alphas = 1). Equations:

            sigmaEQ(unit tension at theta) = sigma0 / sigma_theta,  theta = 0, 45, 90
            sigmaEQ(1, 1, 0)               = sigma0 / sigmab
            r_theta (associated flow rule) = r0, r45, r90
            deps_yy/deps_xx at (1, 1, 0)   = rb

        Returns None if any of the 8 data is missing or Newton fails.
        """
        data = (self.sigma0, self.sigma45, self.sigma90, self.sigmab, self.r0, self.r45, self.r90, self.rb)
        if None in data or 0 in (self.sigma0, self.sigma45, self.sigma90, self.sigmab):
            return None
        targets = np.array([1.0, self.sigma0 / self.sigma45, self.sigma0 / self.sigma90,
                            self.sigma0 / self.sigmab, self.r0, self.r45, self.r90, self.rb])
        probe = YLD2000_PS(a=self.a)

        def residuals(x):
            for name, value in zip(self._ALPHAS, x):
                setattr(probe, name, float(value))
            out = []
            for deg in (0.0, 45.0, 90.0):
                th = np.deg2rad(deg)
                out.append(probe._equivalent_stress_xy(np.cos(th) ** 2, np.sin(th) ** 2,
                                                       np.sin(th) * np.cos(th)))
            out.append(probe._equivalent_stress_xy(1.0, 1.0, 0.0))
            for deg in (0.0, 45.0, 90.0):
                out.append(probe.calc_lankford_from_theta_stress(PointState(theta_stress=np.deg2rad(deg))))
            out.append(probe.calc_rb())
            if None in out:
                return None
            return np.array(out, dtype=float) - targets

        x = np.ones(8)
        eps = 1e-7
        for _ in range(max_iter):
            f = residuals(x)
            if f is None:
                return None
            if np.max(np.abs(f)) < tol:
                return tuple(float(v) for v in x)
            J = np.zeros((8, 8))
            for j in range(8):
                xp = x.copy()
                xp[j] += eps
                fp = residuals(xp)
                if fp is None:
                    return None
                J[:, j] = (fp - f) / eps
            try:
                dx = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                return None
            step = np.max(np.abs(dx))
            x = x + (min(1.0, 0.5 / step) if step > 0.5 else 1.0) * dx
        return None

    def ensure_yld2000_coefficients(self) -> None:
        """Fills the missing alphas from the experimental data (calc_yld2000_coefficients), if possible."""
        if self._alphas() is None:
            coeffs = self.calc_yld2000_coefficients()
            if coeffs is not None:
                for name, value in zip(self._ALPHAS, coeffs):
                    if getattr(self, name) is None:
                        setattr(self, name, value)

    def _ensure_calibration(self) -> None:
        self.ensure_yld2000_coefficients()

    def summary(self, state: Optional[PointState] = None, show_unset: bool = False) -> str:
        """As YieldCriterionPlaneStress.summary, calibrating first so that the alphas are shown."""
        self._ensure_calibration()
        return super().summary(state=state, show_unset=show_unset)

    # ---- yield function and its gradient ---------------------------------
    def _linear_transforms(self):
        """(L', L'') as 3x3 arrays acting on (sigma_xx, sigma_yy, tau_xy); None if not calibrated."""
        self.ensure_yld2000_coefficients()
        alphas = self._alphas()
        if alphas is None:
            return None
        a1, a2, a3, a4, a5, a6, a7, a8 = alphas
        L1 = np.array([[2 * a1 / 3, -a1 / 3, 0.0],
                       [-a2 / 3, 2 * a2 / 3, 0.0],
                       [0.0, 0.0, a7]])
        L2 = np.array([[-2 * a3 + 2 * a4 + 8 * a5 - 2 * a6, a3 - 4 * a4 - 4 * a5 + 4 * a6, 0.0],
                       [4 * a3 - 4 * a4 - 4 * a5 + a6, -2 * a3 + 8 * a4 + 2 * a5 - 2 * a6, 0.0],
                       [0.0, 0.0, 9 * a8]]) / 9.0
        return L1, L2

    def _phi_and_gradient(self, sigma_xx: float, sigma_yy: float, tau_xy: float):
        """
        phi = phi' + phi'' and its gradient d(phi)/d(sigma_xx, sigma_yy,
        tau_xy). For each transformed tensor X (components Xxx, Xyy, Xxy),
        the principal values are X1,2 = c +- R with c = (Xxx + Xyy)/2,
        R = sqrt(((Xxx - Xyy)/2)^2 + Xxy^2):
            phi'  = (2 R')^a
            phi'' = |3 c'' - R''|^a + |3 c'' + R''|^a
        """
        L = self._linear_transforms()
        if L is None:
            return None
        L1, L2 = L
        sigma = np.array([sigma_xx, sigma_yy, tau_xy], dtype=float)
        a = self.a
        sp = self._signed_power

        def mohr(X):
            c = 0.5 * (X[0] + X[1])
            u = 0.5 * (X[0] - X[1])
            R = sqrt(u * u + X[2] * X[2])
            # dc/dX and dR/dX (dR/dX = 0 at R = 0: its coefficients vanish there)
            dc = np.array([0.5, 0.5, 0.0])
            dR = np.array([0.5 * u / R, -0.5 * u / R, X[2] / R]) if R > 0 else np.zeros(3)
            return c, R, dc, dR

        X1 = L1 @ sigma
        c1, R1, _, dR1 = mohr(X1)
        phi1 = (2.0 * R1) ** a
        dphi1_dX = a * (2.0 * R1) ** (a - 1) * 2.0 * dR1

        X2 = L2 @ sigma
        c2, R2, dc2, dR2 = mohr(X2)
        m, p = 3.0 * c2 - R2, 3.0 * c2 + R2
        phi2 = abs(m) ** a + abs(p) ** a
        dphi2_dc = 3.0 * a * (sp(m, a - 1) + sp(p, a - 1))
        dphi2_dR = a * (sp(p, a - 1) - sp(m, a - 1))
        dphi2_dX = dphi2_dc * dc2 + dphi2_dR * dR2

        grad = L1.T @ dphi1_dX + L2.T @ dphi2_dX
        return phi1 + phi2, grad

    def calc_equivalent_stress(self, state: PointState) -> Optional[float]:
        """sigmaEQ = (phi/2)^(1/a), from the material-axes components of the state (see _stress_components)."""
        comps = self._stress_components(state)
        if comps is None:
            return None
        return self._equivalent_stress_xy(*comps)

    def _equivalent_stress_xy(self, sigma_xx: float, sigma_yy: float, tau_xy: float) -> Optional[float]:
        out = self._phi_and_gradient(sigma_xx, sigma_yy, tau_xy)
        if out is None:
            return None
        return (0.5 * out[0]) ** (1.0 / self.a)

    def _flow_gradient_xy(self, sigma_xx: float, sigma_yy: float,
                          tau_xy: float) -> Optional[Tuple[float, float, float]]:
        """Exact gradient d(sigmaEQ)/d(sigma_xx, sigma_yy, tau_xy) (associated flow rule)."""
        out = self._phi_and_gradient(sigma_xx, sigma_yy, tau_xy)
        if out is None or out[0] == 0:
            return None
        phi, grad = out
        factor = (0.5 * phi) ** (1.0 / self.a - 1.0) / (2.0 * self.a)   # d(sigmaEQ)/d(phi)
        return tuple(float(g) for g in factor * grad)

    # ---- flow rule: alpha <-> beta, sigma1 from sigmaEQ -------------------
    def calc_rb(self) -> Optional[float]:
        """Biaxial anisotropy coefficient rb = deps_yy/deps_xx at equibiaxial tension (1, 1, 0)."""
        grad = self._flow_gradient_xy(1.0, 1.0, 0.0)
        if grad is None or grad[0] == 0:
            return None
        return grad[1] / grad[0]

    def calc_beta_from_deviatoric_stresses(self, state: PointState) -> Optional[float]:
        """Anisotropic beta from (sigma1, sigma2), via calc_beta_from_alpha (not the isotropic base formula)."""
        if None in (state.sigma1, state.sigma2) or state.sigma1 == 0:
            return None
        return self.calc_beta_from_alpha(replace(state, alpha=state.sigma2 / state.sigma1))

    def calc_beta_from_alpha(self, state: PointState) -> Optional[float]:
        if state.alpha is None:
            return None
        theta = state.theta_stress if state.theta_stress is not None else 0.0
        return self._beta_from_alpha_theta(state.alpha, theta)

    def calc_alpha_from_beta(self, state: PointState) -> Optional[float]:
        if state.beta is None or self._linear_transforms() is None:
            return None
        theta = state.theta_stress if state.theta_stress is not None else 0.0
        return self._alpha_from_beta_theta(state.beta, theta)

    def calc_sigma1_from_equivalent_stress_and_alpha(self, state: PointState) -> Optional[float]:
        if None in (state.sigmaEQ, state.alpha):
            return None
        theta = state.theta_stress if state.theta_stress is not None else 0.0
        return self._sigma1_from_equivalent_stress_alpha_theta(state.sigmaEQ, state.alpha, theta)

    # ---- off-axis uniaxial tension ---------------------------------------
    def calc_sigma_from_theta_stress(self, state: PointState) -> Optional[float]:
        """
        Uniaxial yield stress at theta_stress from RD: the unit state
        (cos^2, sin^2, sin*cos) in material axes. Scaled to state.sigmaEQ
        if set, otherwise normalised to sigma(0) = 1.
        """
        if state.theta_stress is None:
            return None
        c, s = np.cos(state.theta_stress), np.sin(state.theta_stress)
        unit = self._equivalent_stress_xy(c * c, s * s, s * c)
        if unit is None or unit == 0:
            return None
        return (state.sigmaEQ if state.sigmaEQ is not None else 1.0) / unit

    def calc_lankford_from_theta_stress(self, state: PointState) -> Optional[float]:
        """Lankford coefficient at theta_stress from RD (see Hill48_PS.calc_lankford_from_theta_stress)."""
        if state.theta_stress is None:
            return None
        c, s = np.cos(state.theta_stress), np.sin(state.theta_stress)
        c2, s2, cs = c * c, s * s, s * c
        grad = self._flow_gradient_xy(c2, s2, cs)
        if grad is None:
            return None
        gx, gy, gxy = grad
        denominator = -(gx + gy)
        if denominator == 0:
            return None
        return (gx * s2 + gy * c2 - gxy * cs) / denominator

    def calc_psi_noncoax_from_theta_stress(self, state: PointState) -> Optional[float]:
        """Angle between loading and strain-increment principal directions (see YLD91_PS)."""
        if state.theta_stress is None:
            return None
        alpha = state.alpha if state.alpha is not None else 0.0
        sxx, syy, txy = self._components_from_principal(1.0, alpha, state.theta_stress)
        grad = self._flow_gradient_xy(sxx, syy, txy)
        if grad is None:
            return None
        gx, gy, gxy = grad
        return self._wrap_half_turn(0.5 * float(np.arctan2(gxy, gx - gy)) - state.theta_stress)


# Calibrated criteria of real materials are kept by the users (research notebooks), not here.


# --------------------------------------------------------------------------
# Data (get_*) / plotting (plot_*) functions
#
# Each get_* function below computes and returns the data for one panel,
# for a single `criterion` -- never touching matplotlib. Each plot_*
# function takes `criteria` (plural, a list) and calls the matching get_*
# once per criterion, drawing the result (ax.plot/scatter, labels,
# title), building a whole standalone figure and saving it; the
# lower-level, panel-only drawing (e.g. for combining several panels into
# one mosaic figure) is also done by calling get_* once per criterion --
# see plot_mosaic.
# --------------------------------------------------------------------------
def get_yield_loci(criterion, sigmaY: float = 1.0, n_points: int = 361, normalize: bool = False):
    """
    Computes the yield locus (sigma_1 vs sigma_2, plane stress sigma_3=0)
    of one yield criterion.

    Parameters
    ----------
    criterion : YieldCriterionPlaneStress
        A bare (calibration-only) criterion -- never solved.
    sigmaY : float
        Reference stress the locus is scaled to. With normalize=True
        this is a no-op scale-wise (any positive value gives the same
        normalised shape); with normalize=False, pass the criterion's
        real reference stress to plot the locus in real stress units.
    n_points : int
        Angular resolution of the sweep.
    normalize : bool
        If True, the locus is normalised by dividing by sigmaY (useful
        for comparing the SHAPE of different criteria even if they have
        different reference stresses).

    Returns
    -------
    (s1, s2) : np.ndarray pair.
    """
    s1, s2 = criterion.locus_points(n_points, sigmaY=sigmaY)
    if normalize:
        s1, s2 = s1 / sigmaY, s2 / sigmaY
    return s1, s2


def get_alpha_vs_beta(criterion, n_points: int = 361):
    """
    Computes alpha = sigma2 / sigma1 vs the flow ratio beta = deps2 / deps1
    for one yield criterion.

    The curve is built from two sweeps combined, since alpha can diverge
    to +-infinity at some beta (and vice versa): beta is swept over
    [beta_min, beta_max] and calc_alpha_from_beta gives the matching
    alpha, while alpha is swept over the WHOLE real line (via a tan(theta)
    substitution, theta in (-pi/2, pi/2)) and calc_beta_from_alpha gives
    the matching beta. beta only approaches beta_min/beta_max asymptotically
    as |alpha| grows, so this second sweep needs the full range -- reusing
    the (much narrower) [beta_min, beta_max] interval as an alpha range,
    as an earlier version of this function did, leaves the curve broken
    below the beta some intermediate |alpha| already reaches (e.g. Hill
    1979 stalls just past beta = -1). The wide sweep is then trimmed to
    the points actually inside [beta_min, beta_max], since calc_beta_from_alpha
    also picks up unrelated branches past its own asymptotes (e.g. where
    deps1 = 0). Combining both sweeps keeps the curve well-behaved and
    unbroken near the asymptotic branches.

    Neither bound of the beta sweep is an arbitrary number: the lower
    bound is the beta reached at alpha = -infinity, i.e. at the physical
    state sigma1 = 0, sigma2 < 0, sigma3 = 0 (approximated numerically by
    evaluating calc_beta_from_alpha at a very large negative alpha); the
    upper bound is the larger of 1 and the beta reached at alpha = 1 (the
    balanced-biaxial tension state), since beta(alpha=1) can fall short of
    the isotropic beta = 1 for an anisotropic criterion.

    Parameters
    ----------
    criterion : YieldCriterionPlaneStress
        A bare (calibration-only) criterion.
    n_points : int
        Number of points sampled for the beta sweep (the alpha sweep uses
        4x as many, since most of it is trimmed away afterwards).

    Returns
    -------
    (beta_vals, alpha_vals) : np.ndarray pair.
    """
    # Numerically stands in for alpha = -infinity (sigma1 -> 0, sigma2 < 0).
    alpha_minus_inf = -1.0e8

    # tan(theta) sweeps the whole real line as theta covers (-pi/2, pi/2),
    # so alpha's asymptotic branches (|alpha| -> infinity) are reached too.
    eps = 1e-6
    theta = np.linspace(-0.5 * np.pi + eps, 0.5 * np.pi - eps, 4 * n_points)
    alpha_wide = np.tan(theta)

    beta_min = criterion.calc_beta_from_alpha(PointState(alpha=alpha_minus_inf))
    beta_max = max(1.0, criterion.calc_beta_from_alpha(PointState(alpha=1.0)))
    param = np.linspace(beta_min, beta_max, n_points)

    alpha_from_beta = [criterion.calc_alpha_from_beta(PointState(beta=b)) for b in param]
    beta_from_alpha = [criterion.calc_beta_from_alpha(PointState(alpha=a)) for a in alpha_wide]

    beta_from_alpha = np.array(beta_from_alpha, dtype=float)
    in_range = (beta_from_alpha >= beta_min) & (beta_from_alpha <= beta_max)

    beta_vals = np.concatenate([param, beta_from_alpha[in_range]])
    alpha_vals = np.concatenate([np.array(alpha_from_beta, dtype=float), alpha_wide[in_range]])

    valid = ~(np.isnan(alpha_vals) | np.isnan(beta_vals))
    alpha_vals, beta_vals = alpha_vals[valid], beta_vals[valid]
    order = np.argsort(beta_vals)
    return beta_vals[order], alpha_vals[order]


def get_yield_stress_vs_angle(criterion, n_points: int = 181):
    """
    Computes the dimensionless uniaxial yield stress sigma(theta)/sigma(0)
    against the angle theta (in degrees) between the loading direction
    and principal direction 1 (e.g. the rolling direction), by sweeping
    the criterion's theta_stress and reading off
    calc_sigma_from_theta_stress() at a bare (state.sigmaEQ = None) state.
    Normalising by the curve's own theta = 0 value keeps the axis
    unit-independent and puts every criterion on the same scale for
    comparison. Only Hill48_PS implements it
    for now (see calc_sigma_from_theta_stress): it is the only criterion in
    this module whose equivalent stress has a standard closed-form
    extension to off-axis (sheared) stress states -- the others only
    accept principal stresses already aligned with the anisotropy axes,
    and raise TypeError here if passed in.

    Parameters
    ----------
    criterion : YieldCriterionPlaneStress
        Criterion to compute.
    n_points : int
        Number of angles sampled between 0 and 90 degrees.

    Returns
    -------
    (theta_deg, values) : np.ndarray pair.
    """
    theta_deg = np.linspace(0.0, 90.0, n_points)
    theta = np.deg2rad(theta_deg)

    values = np.array(
        [criterion.calc_sigma_from_theta_stress(PointState(theta_stress=t)) for t in theta],
        dtype=float,
    )
    if np.all(np.isnan(values)):
        raise TypeError(
            f"{criterion.name}: get_yield_stress_vs_angle not supported "
            "(calc_sigma_from_theta_stress is not implemented for this criterion)"
        )
    values = values / values[0]  # dimensionless: sigmaY(theta) / sigmaY(0)
    return theta_deg, values


def get_lankford_vs_angle(criterion, n_points: int = 181):
    """
    Computes the Lankford coefficient r(theta) = (width strain increment) /
    (thickness strain increment) for a uniaxial tension test at angle
    theta from principal direction 1 (e.g. the rolling direction), against
    theta (in degrees), by sweeping the criterion's theta_stress and
    reading off calc_lankford_from_theta_stress(). Only
    Hill48_PS implements it for now (see calc_lankford_from_theta_stress for why,
    and for the closed-form used); the others raise TypeError here if
    passed in.

    Parameters
    ----------
    criterion : YieldCriterionPlaneStress
        Criterion to compute.
    n_points : int
        Number of angles sampled between 0 and 90 degrees.

    Returns
    -------
    (theta_deg, values) : np.ndarray pair.
    """
    theta_deg = np.linspace(0.0, 90.0, n_points)
    theta = np.deg2rad(theta_deg)

    values = np.array(
        [criterion.calc_lankford_from_theta_stress(PointState(theta_stress=t)) for t in theta],
        dtype=float,
    )
    if np.all(np.isnan(values)):
        raise TypeError(
            f"{criterion.name}: get_lankford_vs_angle not supported "
            "(calc_lankford_from_theta_stress is not implemented for this criterion)"
        )
    return theta_deg, values


def get_beta_vs_angle(criterion, n_points: int = 181):
    """
    Computes beta(theta) = deps2/deps1 -- the ratio of the strain increments
    along the width and loading directions, IN THOSE (TEST) AXES, not
    necessarily the true principal strain directions (see
    calc_psi_noncoax_from_theta_stress) -- for a uniaxial tension test (alpha = 0) at
    angle theta from principal direction 1 (e.g. the rolling direction),
    against theta (in degrees).

    Derived from calc_lankford_from_theta_stress via the algebraic identity
    r = -beta/(1+beta) (so beta = -r/(1+r)): only criteria that implement
    calc_lankford_from_theta_stress support this; others raise TypeError.

    Parameters
    ----------
    criterion : YieldCriterionPlaneStress
        Criterion to compute.
    n_points : int
        Number of angles sampled between 0 and 90 degrees.

    Returns
    -------
    (theta_deg, values) : np.ndarray pair.
    """
    theta_deg = np.linspace(0.0, 90.0, n_points)
    theta = np.deg2rad(theta_deg)

    values = []
    for t in theta:
        r = criterion.calc_lankford_from_theta_stress(PointState(alpha=0.0, theta_stress=t))
        values.append(-r / (1.0 + r) if r is not None and r != -1.0 else None)

    values = np.array(values, dtype=float)
    if np.all(np.isnan(values)):
        raise TypeError(
            f"{criterion.name}: get_beta_vs_angle not supported "
            "(calc_lankford_from_theta_stress is not implemented for this criterion)"
        )
    return theta_deg, values


def get_psi_noncoax_vs_angle(criterion, n_points: int = 181):
    """
    Computes psi_noncoax(theta) (in degrees) -- the angle between the loading
    direction and the true principal strain-increment direction (see
    calc_psi_noncoax_from_theta_stress) -- for a uniaxial tension test (alpha = 0) at
    angle theta from principal direction 1 (e.g. the rolling direction),
    against theta (in degrees). Only criteria that
    implement calc_psi_noncoax_from_theta_stress support this; others raise TypeError.

    Parameters
    ----------
    criterion : YieldCriterionPlaneStress
        Criterion to compute.
    n_points : int
        Number of angles sampled between 0 and 90 degrees.

    Returns
    -------
    (theta_deg, values) : np.ndarray pair.
    """
    theta_deg = np.linspace(0.0, 90.0, n_points)
    theta = np.deg2rad(theta_deg)

    values = []
    for t in theta:
        psi_noncoax = criterion.calc_psi_noncoax_from_theta_stress(PointState(alpha=0.0, theta_stress=t))
        values.append(np.rad2deg(psi_noncoax) if psi_noncoax is not None else None)

    values = np.array(values, dtype=float)
    if np.all(np.isnan(values)):
        raise TypeError(
            f"{criterion.name}: get_psi_noncoax_vs_angle not supported "
            "(calc_psi_noncoax_from_theta_stress is not implemented for this criterion)"
        )
    return theta_deg, values


def build_flow_states_table(criteria):
    """
    Builds (without printing or plotting) the table of alpha/beta values
    at a set of characteristic stress/strain states, one row per state
    and one column per criterion.

        - BBT : balanced-biaxial tension    (alpha = 1)
        - BBS : balanced-biaxial stretching (beta = 1)
        - PS  : plane strain                (beta = 0)
        - UT  : uniaxial tension            (alpha = 0)
        - SH  : pure shear                  (alpha = -1)
        - UC  : uniaxial compression        (alpha = -infinity, sigma1 = 0, sigma2 < 0)

    Returns
    -------
    col_labels : list[str]
        ["State", "", ""] followed by one abbreviated criterion name per
        entry in `criteria`. The first blank header is a spacer column;
        the second carries, unlabelled, the alpha/beta symbol that the
        criterion columns report for that row (the opposite of the one
        fixed in the "State" column).
    rows : list[list[str]]
        One row per characteristic state; each row is
        [state abbreviation + fixed alpha/beta, "", reported symbol,
        one computed value per criterion].
    """

    # Abbreviated criterion names for the table header.
    _CRITERION_ABBR = {
        "VonMises": "Mises",
        "Hill48": "Hill48",
        "Hill79": "Hill79",
        "YLD91": "YLD91",
    }

    # Characteristic stress/strain states: (abbreviation, driving attribute, value).
    _FLOW_STATES = [
        ("BBT", "alpha", 1.0),           # balanced-biaxial tension
        ("BBS", "beta", 1.0),            # balanced-biaxial stretching
        ("PS", "beta", 0.0),             # plane strain
        ("UT", "alpha", 0.0),           # uniaxial tension
        ("SH", "alpha", -1.0),           # pure shear
        ("UC", "alpha", float("-inf")),  # uniaxial compression
    ]

    # Numerically stands in for alpha = -infinity (sigma1 -> 0, sigma2 < 0).
    _ALPHA_MINUS_INF = -1.0e8

    def _fmt(x: Optional[float]) -> str:
        if x is None:
            return "—"  # em dash, for an undefined/unreachable state
        if x == float("-inf"):
            return "-inf"
        if x == float("inf"):
            return "inf"
        return f"{x:.3g}"

    def _state_label(driver: str, value: float) -> str:
        symbol = "a" if driver == "alpha" else "b"  # alpha or beta
        return f"{symbol} = {_fmt(value)}"

    col_labels = ["State", "", ""] + [
        _CRITERION_ABBR.get(criterion.name, criterion.name) for criterion in criteria
    ]

    rows = []
    for label, driver, value in _FLOW_STATES:
        state_cell = f"{label}: {_state_label(driver, value)}"
        reported_symbol = "b" if driver == "alpha" else "a"
        row = [state_cell, "", reported_symbol]
        for criterion in criteria:
            if driver == "alpha":
                a = _ALPHA_MINUS_INF if value == float("-inf") else value
                result = criterion.calc_beta_from_alpha(PointState(alpha=a))
            else:
                result = criterion.calc_alpha_from_beta(PointState(beta=value))
            row.append(_fmt(result))
        rows.append(row)

    return col_labels, rows


def print_flow_states_table(criteria):
    """Prints the table built by build_flow_states_table to the console."""
    col_labels, rows = build_flow_states_table(criteria)
    widths = [max(len(col_labels[i]), *(len(row[i]) for row in rows))
              for i in range(len(col_labels))]

    def fmt_row(cells):
        first = cells[0].ljust(widths[0])
        rest = [c.rjust(w) for c, w in zip(cells[1:], widths[1:])]
        return "  ".join([first] + rest)

    print(fmt_row(col_labels))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def add_flow_states_table(ax, criteria, bbox=(0.0, -0.85, 1.0, 0.55)):
    """Adds the table built by build_flow_states_table below the given axes."""
    col_labels, rows = build_flow_states_table(criteria)
    n_crit = len(criteria)
    state_w, spacer_w, symbol_w = 0.30, 0.04, 0.06
    crit_w = (1.0 - state_w - spacer_w - symbol_w) / n_crit
    col_widths = [state_w, spacer_w, symbol_w] + [crit_w] * n_crit

    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center",
                      colWidths=col_widths, loc="bottom", bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if col == 0:
            cell.set_text_props(ha="left")
    return table



def plot_yield_loci(criteria, name='demo', 
                    experimental_sigma=None, 
                    ax=None, 
                    figwidth=width_1column,
                    normalize:bool = False,
                    labels=None,
                    sigmaY=None,
                    savefile=None):
    """
    Parameters
    ----------
    criteria : list[YieldCriterionPlaneStress]
        Bare (calibration-only) criteria to plot (e.g. VonMises_PS, Hill48_PS,
        Hill79_PS, YLD91_PS instances -- get_yield_loci supports all of
        them, unlike the angle-based get_*_vs_angle functions). With
        normalize=False, each locus is scaled to its criterion's own
        sigma0 (real stress units); criteria without sigma0 stay at 1.0
        unless sigmaY is given.
    sigmaY : float, optional
        Reference stress (unit RD tension) for every locus when
        normalize=False, e.g. the material's measured sigma0: use it when
        mixing criteria with and without their own sigma0 (VonMises_PS,
        Hill48_PS have none), otherwise those would be drawn at 1.0.
    name : str
        Used in the title / legend title.
    savefile : str or Path, optional
        If given, the figure is saved there (nothing is saved otherwise).
    normalize : bool
        If True, plot sigma/sigmaY (shape only); if False, real stresses.
    labels : list[str], optional
        Legend labels, one per criterion (defaults to criterion.name).
    experimental_sigma : sequence of (sigma1, sigma2), optional
        Extra experimental points to mark on top of the loci, e.g. a
        uniaxial tension calibration point sigma0 -> (sigma0, 0) or
        sigma90 -> (0, sigma90) -- both DO have an exact, unambiguous
        position in this (sigma1, sigma2, tau_xy=0) plane, since a
        uniaxial tension test along the rolling or transverse direction
        is exactly such a state. Not every experimental datum has an
        honest position here -- e.g. a 45-degree off-axis uniaxial test
        (sigma45) has nonzero shear in material axes
        (sigma_xx=sigma_yy=tau_xy=sigma45/2), so it does NOT reduce to a
        point in this tau_xy=0 plane; that kind of data belongs instead
        on plot_yield_stress_and_lankford_vs_angle. Drawn as a single
        scatter series with one legend entry; in the same units as the
        loci (normalised if normalize=True, e.g. sigma0 / sigmaY; real
        stresses otherwise).
    """
    aspect_ratio = 1.0
    if ax is None:
        _, ax = plt.subplots(figsize=(figwidth, figwidth * aspect_ratio))

    if labels is None:
        labels = [criterion.name for criterion in criteria]

    for criterion, label in zip(criteria, labels):
        ref = sigmaY if sigmaY is not None else (getattr(criterion, "sigma0", None) or 1.0)
        s1, s2 = get_yield_loci(criterion, sigmaY=ref, normalize=normalize)
        ax.plot(s2, s1, label=label)

    if experimental_sigma:
        sigma1_pts, sigma2_pts = zip(*experimental_sigma)
        # Same (x=sigma2, y=sigma1) orientation as the loci above
        ax.scatter(sigma2_pts, sigma1_pts, marker="o", s=40, facecolor="none",
                   edgecolor="black", zorder=5, label="Experimental data")

    xlabel = r"$\sigma_2 / \sigma_Y$" if normalize else r"$\sigma_2$ (MPa)"
    ylabel = r"$\sigma_1 / \sigma_Y$" if normalize else r"$\sigma_1$ (MPa)"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, title=f"{name}")
    #ax.set_title(rf"Yield locus ($\sigma_3=0$) - {name}", fontsize=12)

    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile)
        print(f"Figure saved in {savefile}")
    return ax


def plot_alpha_vs_beta(criteria, name='demo', figwidth=width_column_and_a_half, savefile=None):
    """
    Parameters
    ----------
    criteria : list[YieldCriterionPlaneStress]
        Bare (calibration-only) criteria to plot.
    name : str
        Not used in the figure (kept for symmetry with the other plot_* functions).
    savefile : str or Path, optional
        If given, the figure is saved there (nothing is saved otherwise).
    """
    print_flow_states_table(criteria)

    aspect_ratio = 1.0
    data = [get_alpha_vs_beta(criterion) for criterion in criteria]
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(figwidth, figwidth * aspect_ratio))

    for ax, ylim in ((ax1, (-3, 1.2)), (ax2, (-100, 5))):
        for criterion, (beta_vals, alpha_vals) in zip(criteria, data):
            ax.plot(beta_vals, alpha_vals, label=criterion.name)
        ax.set_xlabel(r"$\beta = d\varepsilon_2 / d\varepsilon_1$", fontsize=12)
        ax.set_ylabel(r"$\alpha = \sigma_2 / \sigma_1$", fontsize=12)
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_title(r"Stress ratio $\alpha$ vs. flow ratio $\beta$", fontsize=12)

    add_flow_states_table(ax2, criteria)
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile)
        print(f"Figure saved in {savefile}")


def plot_yield_stress_and_lankford_vs_angle(criteria, name='demo',
                                             experimental_sigma=None,
                                             experimental_lankford=None,
                                             figwidth=width_column_and_a_half,
                                             savefile=None):
    """
    sigmaY(theta) and r(theta) on two side-by-side subplots, one colour
    per criterion, plus experimental_sigma and experimental_lankford on
    their respective panels.

    Note: criteria must implement calc_sigma_from_theta_stress /
    calc_lankford_from_theta_stress (Hill79_PS does not -- see
    get_yield_stress_vs_angle/get_lankford_vs_angle). YLD91_PS reproduces
    r0, r45 and r90 exactly (its 3 free coefficients C2, C3, C6 are
    solved for exactly that, with C1 = 1 fixed as the scale
    normalisation).

    Parameters
    ----------
    criteria : list[YieldCriterionPlaneStress]
        Bare (calibration-only) criteria to plot (excluding Hill79_PS).
    name : str
        Used in the title / legend title.
    savefile : str or Path, optional
        If given, the figure is saved there (nothing is saved otherwise).
    experimental_sigma : sequence of (theta_deg, sigma_ratio), optional
        Extra experimental points to mark on the sigmaY(theta)/sigmaY(0)
        panel (ax1), e.g. sigma0 -> (0, 1), sigma45 -> (45, sigma45/sigma0),
        sigma90 -> (90, sigma90/sigma0) -- an honest position here, unlike
        in plot_yield_loci, since theta really is the physical loading
        angle on this plot. Drawn as a single scatter series with one
        legend entry, normalised by whichever sigma0 the caller wants
        (e.g. the measured one, not necessarily each curve's own theta=0
        value, so the markers double as a fit-quality check).
    experimental_lankford : sequence of (theta_deg, r_value), optional
        Extra experimental points to mark on the r(theta) panel (ax2),
        e.g. r0 -> (0, r0), r45 -> (45, r45), r90 -> (90, r90). Drawn as a
        single scatter series ('x' markers) with one legend entry, same
        pattern as experimental_sigma.
    """

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    aspect_ratio = 0.5

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(figwidth, figwidth * aspect_ratio))

    for i, criterion in enumerate(criteria):
        color = colors[i % len(colors)]
        theta_deg, values = get_yield_stress_vs_angle(criterion)
        ax1.plot(theta_deg, values, label=criterion.name, color=color)
        theta_deg, values = get_lankford_vs_angle(criterion)
        ax2.plot(theta_deg, values, label=criterion.name, color=color)

    if experimental_sigma:
        theta_pts, ratio_pts = zip(*experimental_sigma)
        ax1.scatter(theta_pts, ratio_pts, marker="o", s=40, facecolor="none",
                    edgecolor="black", zorder=5, label="Experimental data")

    if experimental_lankford:
        theta_pts, r_pts = zip(*experimental_lankford)
        ax2.scatter(theta_pts, r_pts, marker="o", s=40, facecolor="none", 
                    edgecolor="black", zorder=5, label="Experimental data")

    ax1.set_xlabel(r"$\theta$ (deg from RD)", fontsize=12)
    ax1.set_ylabel(r"$\sigma_Y(\theta) / \sigma_Y(0)$", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_title(r"Yield stress vs. angle to RD", fontsize=12)

    ax2.set_xlabel(r"$\theta$ (deg from RD)", fontsize=12)
    ax2.set_ylabel(r"$r(\theta)$", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_title(r"Lankford coefficient vs. angle to RD", fontsize=12)

    plt.suptitle(name, fontsize=13)
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile)
        print(f"Figure saved in {savefile}")


def plot_beta_and_psi_noncoax_vs_angle(criteria, name='demo', figwidth=width_column_and_a_half,
                                       savefile=None):
    """
    beta(theta) = deps2/deps1 and psi_noncoax(theta) on two side-by-side
    subplots, for a uniaxial tension test (alpha = 0) at angle theta from
    principal direction 1 (e.g. the rolling direction), one colour per
    criterion (must implement calc_lankford_from_theta_stress /
    calc_psi_noncoax_from_theta_stress -- Hill79_PS does not).

    Parameters
    ----------
    criteria : list[YieldCriterionPlaneStress]
        Bare (calibration-only) criteria to plot (excluding Hill79_PS). Only
        the beta/psi_noncoax ANGLE dependence matters here, not the
        loading magnitude, so sigma1/sigma2 can be any convenient value.
    name : str
        Used in the title / legend title.
    savefile : str or Path, optional
        If given, the figure is saved there (nothing is saved otherwise).
    """

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    aspect_ratio = 0.5

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(figwidth, figwidth * aspect_ratio))

    for i, criterion in enumerate(criteria):
        color = colors[i % len(colors)]
        theta_deg, values = get_beta_vs_angle(criterion)
        ax1.plot(theta_deg, values, label=criterion.name, color=color)
        theta_deg, values = get_psi_noncoax_vs_angle(criterion)
        ax2.plot(theta_deg, values, label=criterion.name, color=color)

    ax1.set_xlabel(r"$\theta$ ($^\circ$ from RD)", fontsize=12)
    ax1.set_ylabel(r"$\beta(\theta) = d\varepsilon_2 / d\varepsilon_1$", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_title(r"Flow ratio $\beta$ vs. angle to RD", fontsize=12)

    ax2.set_xlabel(r"$\theta$ ($^\circ$ from RD)", fontsize=12)
    ax2.set_ylabel(r"$\psi_{noncoax}(\theta)$ (deg)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_title(r"Strain-rotation angle $\psi_{noncoax}$ vs. angle to RD", fontsize=12)

    title = rf"$\beta$ and $\psi_{{noncoax}}$ in uniaxial tension ($\alpha = 0$) vs. angle to RD - {name}"
    #plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile)
        print(f"Figure saved in {savefile}")


def plot_mosaic(criteria, name='demo', experimental_sigma_locus=None,
                 experimental_sigma=None, experimental_lankford=None, figwidth=width_2columns,
                 savefile=None):
    """
    Mosaic combining every get_* function in this module into a single
    figure:

        - yield locus                        (get_yield_loci, normalised)
        - alpha vs beta                       (get_alpha_vs_beta)
        - dimensionless yield stress vs angle (get_yield_stress_vs_angle)
        - Lankford coefficient vs angle       (get_lankford_vs_angle)
        - flow ratio beta vs angle            (get_beta_vs_angle)
        - strain-rotation angle psi_noncoax vs angle  (get_psi_noncoax_vs_angle)

    One quick-overview figure per material, instead of the several
    separate ones the individual plot_* functions save.

    Parameters
    ----------
    criteria : list[YieldCriterionPlaneStress]
        Bare (calibration-only) criteria to plot on every panel -- must implement
        the off-axis calc_*_from_theta_stress methods used by the
        angle-based panels (Hill79_PS does not; see
        get_yield_stress_vs_angle/get_lankford_vs_angle).
    name : str
        Used in the title / legend title.
    savefile : str or Path, optional
        If given, the figure is saved there (nothing is saved otherwise).
    experimental_sigma_locus : sequence of (sigma1, sigma2), optional
        Extra experimental points to mark on the yield locus panel, e.g.
        sigma0 -> (sigma0, 0), sigma90 -> (0, sigma90) -- see
        plot_yield_loci for the same pattern (NOT the same shape as
        experimental_sigma below -- this one is a (sigma1, sigma2) pair,
        that one a (theta_deg, ratio) pair).
    experimental_sigma : sequence of (theta_deg, sigma_ratio), optional
        Extra experimental points to mark on the sigmaY(theta)/sigmaY(0)
        panel, e.g. sigma0 -> (0, 1), sigma45 -> (45, sigma45/sigma0),
        sigma90 -> (90, sigma90/sigma0) -- see
        plot_yield_stress_and_lankford_vs_angle for the same pattern.
    experimental_lankford : sequence of (theta_deg, r_value), optional
        Extra experimental points to mark on the r(theta) panel, e.g.
        r0 -> (0, r0), r45 -> (45, r45), r90 -> (90, r90).
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    aspect_ratio = 0.75
    figsize = (figwidth, figwidth * aspect_ratio)

    mosaic = (("locus", "sigmaY", "beta"),
              ("alphabeta", "lankford", "psi_noncoax"))
    fig, axd = plt.subplot_mosaic(mosaic, figsize=figsize)

    # "alphabeta" becomes a single broken-y-axis plot: two stacked, x-shared
    # sub-axes -- one per vertical zone -- with no gap and a break mark
    # between them, rather than two separate plots (without disturbing the
    # other panels' layout).
    alphabeta_gs = axd["alphabeta"].get_subplotspec()
    axd["alphabeta"].remove()
    alphabeta_sub_gs = alphabeta_gs.subgridspec(2, 1, hspace=0.05)
    ax_alphabeta_top = fig.add_subplot(alphabeta_sub_gs[0])
    ax_alphabeta_bottom = fig.add_subplot(alphabeta_sub_gs[1], sharex=ax_alphabeta_top)

    ax = axd["locus"]
    for criterion in criteria:
        s1, s2 = get_yield_loci(criterion, normalize=True)
        ax.plot(s2, s1, label=criterion.name)     # sigma1 vertical, as plot_yield_loci
    if experimental_sigma_locus:
        sigma1_pts, sigma2_pts = zip(*experimental_sigma_locus)
        ax.scatter(sigma2_pts, sigma1_pts, marker="o", s=40, facecolor="none",
                   edgecolor="black", zorder=5, label="Experimental data")
    ax.set_xlabel(r"$\sigma_2 / \sigma_Y$", fontsize=11)
    ax.set_ylabel(r"$\sigma_1 / \sigma_Y$", fontsize=11)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Yield locus", fontsize=12)

    alphabeta_data = [get_alpha_vs_beta(criterion) for criterion in criteria]
    for ax, ylim in ((ax_alphabeta_top, (-3, 1.3)), (ax_alphabeta_bottom, (-100, 2))):
        for criterion, (beta_vals, alpha_vals) in zip(criteria, alphabeta_data):
            ax.plot(beta_vals, alpha_vals, label=criterion.name)
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)

    # broken y axis: no shared spine/ticks between the two zones, plus
    # diagonal break marks where they meet.
    ax_alphabeta_top.spines["bottom"].set_visible(False)
    ax_alphabeta_bottom.spines["top"].set_visible(False)
    ax_alphabeta_top.tick_params(labelbottom=False, bottom=False)

    break_kwargs = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=12, linestyle="none",
                         color="k", mec="k", mew=1, clip_on=False)
    ax_alphabeta_top.plot([0, 1], [0, 0], transform=ax_alphabeta_top.transAxes, **break_kwargs)
    ax_alphabeta_bottom.plot([0, 1], [1, 1], transform=ax_alphabeta_bottom.transAxes, **break_kwargs)

    ax_alphabeta_bottom.set_xlabel(r"$\beta = d\varepsilon_2 / d\varepsilon_1$", fontsize=11)
    ax_alphabeta_top.set_ylabel(r"$\alpha = \sigma_2 / \sigma_1$", fontsize=11)
    ax_alphabeta_top.legend(fontsize=8)
    ax_alphabeta_top.set_title(r"$\alpha$ vs. $\beta$ (axis 1 in RD, $\theta=0$)", fontsize=12)

    ax = axd["sigmaY"]
    for i, criterion in enumerate(criteria):
        theta_deg, values = get_yield_stress_vs_angle(criterion)
        ax.plot(theta_deg, values, label=criterion.name, color=colors[i % len(colors)])
    if experimental_sigma:
        theta_pts, ratio_pts = zip(*experimental_sigma)
        ax.scatter(theta_pts, ratio_pts, marker="o", s=40, facecolor="none",
                   edgecolor="black", zorder=5, label="Experimental data")
    ax.set_xlabel(r"$\theta$ ($^\circ$ from RD)", fontsize=11)
    ax.set_ylabel(r"$\sigma_Y(\theta) / \sigma_Y(0)$", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(r"Yield stress vs. angle to RD", fontsize=12)

    ax = axd["lankford"]
    for i, criterion in enumerate(criteria):
        theta_deg, values = get_lankford_vs_angle(criterion)
        ax.plot(theta_deg, values, label=criterion.name, color=colors[i % len(colors)])
    if experimental_lankford:
        theta_pts, r_pts = zip(*experimental_lankford)
        ax.scatter(theta_pts, r_pts, marker="o", s=40, facecolor="none",
                   edgecolor="black", zorder=5, label="Experimental data")
    ax.set_xlabel(r"$\theta$ ($^\circ$ from RD)", fontsize=11)
    ax.set_ylabel(r"$r(\theta)$", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(r"Lankford coefficient vs. angle to RD", fontsize=12)

    ax = axd["beta"]
    for criterion in criteria:
        theta_deg, values = get_beta_vs_angle(criterion)
        ax.plot(theta_deg, values, label=criterion.name)
    ax.set_xlabel(r"$\theta$ ($^\circ$ from RD)", fontsize=11)
    ax.set_ylabel(r"$\beta(\theta) = d\varepsilon_2 / d\varepsilon_1$", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(r"Flow ratio $\beta$ vs. angle to RD (uniaxial tension)", fontsize=12)

    ax = axd["psi_noncoax"]
    for criterion in criteria:
        theta_deg, values = get_psi_noncoax_vs_angle(criterion)
        ax.plot(theta_deg, values, label=criterion.name)
    ax.set_xlabel(r"$\theta$ ($^\circ$ from RD)", fontsize=11)
    ax.set_ylabel(r"$\psi_{noncoax}(\theta)$ (deg)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(r"Strain-rotation angle $\psi_{noncoax}$ vs. angle to RD (uniaxial tension)", fontsize=12)

    fig.suptitle(name, fontsize=15)
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile)
        print(f"Figure saved in {savefile}")
