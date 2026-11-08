"""
yield_criteria_3d.py
====================

General 3D (full stress tensor) versions of the yield criteria in
yield_criteria_plane_stress.py, for working with Abaqus reports from sheet-forming
simulations modelled with solid (3D) elements: Von Mises, Hill 1948 and
Barlat 1991 (Yld91).

Hill 1979 is NOT included: it is formulated on principal stresses only
(no shear terms), so it is only valid when the principal stress axes
coincide with the orthotropy axes -- it has no rigorous general 3D form.

Conventions
-----------
- Stresses are given in the MATERIAL (orthotropy) axes 1 = RD, 2 = TD,
  3 = ND -- e.g. Abaqus output referred to the sheet's local
  *ORIENTATION -- as Voigt 6-vectors in Abaqus component order:

      (s11, s22, s33, s12, s13, s23)

  Any array of shape (6,) or (..., 6) is accepted; every method is
  vectorised over the leading dimensions (one row per report point).
  Use rotate_voigt_stress() to bring global-axes data into material axes.

- flow_direction() returns d(sigmaEQ)/d(sigma) in that same Voigt order,
  treating each shear component as ONE independent variable. This makes
  it work-conjugate to the ENGINEERING shear strains (gamma_12 =
  2*eps_12, ...), which is what Abaqus reports for PE12, PE13, PE23.
  Since sigmaEQ is homogeneous of degree 1 in sigma (Euler: sigma .
  flow_direction = sigmaEQ), the associated flow rule reads

      dPE (Voigt, engineering shears) = dPEEQ * flow_direction(sigma)

  with dPEEQ the work-conjugate equivalent plastic strain increment
  (see equivalent_plastic_strain_increment).

- Like yield_criteria_plane_stress.py, the classes here are CALIBRATION ONLY (no
  stress/strain state stored), and every criterion is normalised to
  sigmaEQ = sigma for uniaxial tension sigma along axis 1 (RD). The
  plane-stress PointState is not used: inputs/outputs are plain arrays.

- Each anisotropic class has a from_plane_stress() constructor that
  reuses an already-calibrated yield_criteria_plane_stress.py instance, adding only
  the out-of-plane parameters that in-plane sheet tests cannot measure
  (Hill48_PS: L, M; Yld91: C4, C5). With s33 = s13 = s23 = 0 every class
  reproduces its plane-stress counterpart exactly (sigmaEQ, r(theta),
  sigma(theta)) -- see tests/test_yield_criteria_3d.py.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .yield_criteria_plane_stress import VonMises_PS, Hill48_PS, YLD91_PS


# ----------------------------------------------------------------------
# Voigt/tensor helpers (Abaqus order: 11, 22, 33, 12, 13, 23)
# ----------------------------------------------------------------------
# (i, j) tensor indices of each Voigt component.
_VOIGT_INDEX = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _as_voigt(sigma) -> np.ndarray:
    """Float array of shape (..., 6); raises if the last axis is not 6."""
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape[-1:] != (6,):
        raise ValueError(f"expected Voigt stress(es) of shape (..., 6), got {sigma.shape}")
    return sigma


def _scalar_or_array(x: np.ndarray):
    """Plain float for 0-d results (single-point input), array otherwise."""
    return float(x) if np.ndim(x) == 0 else x


def voigt_to_tensor(v) -> np.ndarray:
    """(..., 6) Voigt stresses -> (..., 3, 3) symmetric tensors."""
    v = _as_voigt(v)
    t = np.empty(v.shape[:-1] + (3, 3))
    for k, (i, j) in enumerate(_VOIGT_INDEX):
        t[..., i, j] = v[..., k]
        t[..., j, i] = v[..., k]
    return t


def tensor_to_voigt(t) -> np.ndarray:
    """(..., 3, 3) symmetric tensors -> (..., 6) Voigt stresses."""
    t = np.asarray(t, dtype=float)
    return np.stack([t[..., i, j] for i, j in _VOIGT_INDEX], axis=-1)


def rotate_voigt_stress(sigma, R) -> np.ndarray:
    """
    Stress components in a rotated frame: sigma' = R . sigma . R^T, with
    R (3x3, orthogonal) whose ROWS are the new axes expressed in the old
    frame. E.g. for global -> material axes, the rows of R are RD, TD, ND
    as unit vectors in global coordinates. R may also be (..., 3, 3), one
    rotation per point.
    """
    R = np.asarray(R, dtype=float)
    t = voigt_to_tensor(sigma)
    return tensor_to_voigt(R @ t @ np.swapaxes(R, -1, -2))


def in_plane_rotation(theta: float) -> np.ndarray:
    """
    Rotation matrix (rows = new axes in material axes) for a frame turned
    by theta (radians) about axis 3 (ND), from axis 1 (RD) towards 2 (TD).
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def uniaxial_stress(theta, sigma: float = 1.0) -> np.ndarray:
    """
    Voigt stress (material axes) of uniaxial tension sigma at angle theta
    (radians, scalar or array) from RD, in the sheet plane.
    """
    theta = np.asarray(theta, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    z = np.zeros_like(theta)
    return sigma * np.stack([c * c, s * s, z, s * c, z, z], axis=-1)


# ----------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------
@dataclass
class YieldCriterion3D(ABC):
    """
    Generic base class for 3D yield criteria (full Cauchy stress tensor,
    in material axes, Voigt order 11, 22, 33, 12, 13, 23). CALIBRATION
    ONLY: subclasses hold just their calibration parameters.
    """

    @abstractmethod
    def equivalent_stress(self, sigma):
        """Equivalent stress sigmaEQ for Voigt stress(es) of shape (..., 6)."""

    @abstractmethod
    def flow_direction(self, sigma) -> np.ndarray:
        """
        Gradient d(sigmaEQ)/d(sigma), shape (..., 6), Voigt order,
        work-conjugate to engineering shear strains (see module
        docstring). NaN where sigmaEQ = 0 (the gradient is undefined).
        """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def equivalent_plastic_strain_increment(self, sigma, deps):
        """
        Work-conjugate equivalent plastic strain increment

            dPEEQ = sigma . dPE / sigmaEQ

        with dPE the plastic strain increment(s) in Voigt order and
        ENGINEERING shears (as Abaqus reports PE12, PE13, PE23).
        """
        sigma = _as_voigt(sigma)
        deps = _as_voigt(deps)
        seq = np.asarray(self.equivalent_stress(sigma), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(seq > 0, np.sum(sigma * deps, axis=-1) / seq, np.nan)
        return _scalar_or_array(out)

    def plastic_strain_increment(self, sigma, depsEQ):
        """
        Associated flow rule: dPE = depsEQ * flow_direction(sigma), Voigt
        order, engineering shears. depsEQ is a scalar or one value per point.
        """
        return np.asarray(depsEQ, dtype=float)[..., None] * self.flow_direction(sigma)

    def yield_stress_vs_angle(self, theta):
        """
        Uniaxial yield stress at angle theta (radians, scalar or array)
        from RD, in the sheet plane, normalised to sigma(0) = 1. Same
        quantity as yield_criteria_plane_stress's calc_sigma_from_theta_stress.
        """
        seq = np.asarray(self.equivalent_stress(uniaxial_stress(theta)), dtype=float)
        return _scalar_or_array(1.0 / seq)

    def lankford_vs_angle(self, theta):
        """
        Lankford coefficient r(theta) = deps_width / deps_thickness for
        uniaxial tension at angle theta (radians, scalar or array) from
        RD, from the 3D associated flow rule -- the thickness strain is
        the true out-of-plane component dPE33 (not inferred from
        incompressibility). Same quantity as yield_criteria_plane_stress's
        calc_lankford_from_theta_stress.
        """
        theta = np.asarray(theta, dtype=float)
        g = self.flow_direction(uniaxial_stress(theta))
        c, s = np.cos(theta), np.sin(theta)
        # Width direction (-s, c, 0): eps_ww = s^2 e11 + c^2 e22 - s c gamma12.
        deps_width = s * s * g[..., 0] + c * c * g[..., 1] - s * c * g[..., 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = deps_width / g[..., 2]
        return _scalar_or_array(r)


# ----------------------------------------------------------------------
# Quadratic criteria: sigmaEQ^2 = sigma^T P sigma
# ----------------------------------------------------------------------
@dataclass
class _QuadraticYieldCriterion3D(YieldCriterion3D):
    """Shared implementation for criteria with sigmaEQ^2 = sigma^T P sigma."""

    @abstractmethod
    def P_matrix(self) -> np.ndarray:
        """6x6 symmetric matrix P (Voigt order) of the quadratic form."""

    def equivalent_stress(self, sigma):
        sigma = _as_voigt(sigma)
        q = np.einsum("...i,ij,...j->...", sigma, self.P_matrix(), sigma)
        return _scalar_or_array(np.sqrt(np.maximum(q, 0.0)))

    def flow_direction(self, sigma) -> np.ndarray:
        sigma = _as_voigt(sigma)
        P = self.P_matrix()
        Psig = sigma @ P  # P is symmetric
        seq = np.sqrt(np.maximum(np.sum(sigma * Psig, axis=-1), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(seq[..., None] > 0, Psig / seq[..., None], np.nan)


@dataclass
class VonMises_3D(_QuadraticYieldCriterion3D):
    """
    Isotropic von Mises criterion:

        sigmaEQ^2 = 1/2 [(s11-s22)^2 + (s22-s33)^2 + (s33-s11)^2]
                    + 3 (s12^2 + s13^2 + s23^2)

    i.e. Hill48_3D with F = G = H = 1/2, L = M = N = 3/2.
    """

    def P_matrix(self) -> np.ndarray:
        return _hill_P_matrix(0.5, 0.5, 0.5, 1.5, 1.5, 1.5)

    @classmethod
    def from_plane_stress(cls, criterion: Optional[VonMises_PS] = None) -> "VonMises_3D":
        """For symmetry with the anisotropic classes (nothing to copy)."""
        return cls()


def _hill_P_matrix(F, G, H, L, M, N) -> np.ndarray:
    P = np.zeros((6, 6))
    P[:3, :3] = [[G + H, -H, -G],
                 [-H, F + H, -F],
                 [-G, -F, F + G]]
    P[3, 3], P[4, 4], P[5, 5] = 2.0 * N, 2.0 * M, 2.0 * L   # s12, s13, s23
    return P


@dataclass
class Hill48_3D(_QuadraticYieldCriterion3D):
    """
    Anisotropic quadratic Hill 1948 criterion, general 3D form in the
    orthotropy axes:

        sigmaEQ^2 = F (s22-s33)^2 + G (s33-s11)^2 + H (s11-s22)^2
                    + 2 L s23^2 + 2 M s13^2 + 2 N s12^2

    With the normalisation G + H = 1 used throughout yield_criteria_plane_stress.py
    (sigmaEQ = sigma in RD tension), F, G, H, N follow from r0, r45, r90
    exactly as in yield_criteria_plane_stress.Hill48_PS (Hosford-Caddell). L and M
    (out-of-plane shear, s23 and s13) cannot be obtained from in-plane
    sheet tests; L = M = 3/2 (the isotropic value) is the usual default.

    See to_abaqus_potential()/from_abaqus_potential() for the equivalent
    Abaqus *POTENTIAL ratios R11..R23.
    """

    r0: Optional[float] = None
    r45: Optional[float] = None
    r90: Optional[float] = None
    F: Optional[float] = None
    G: Optional[float] = None
    H: Optional[float] = None
    N: Optional[float] = None
    L: float = 1.5
    M: float = 1.5

    def ensure_hill_coefficients(self) -> None:
        """Fill F, G, H, N from r0, r45, r90 (via yield_criteria_plane_stress.Hill48_PS) if missing."""
        if None in (self.F, self.G, self.H, self.N):
            coeffs = Hill48_PS(r0=self.r0, r45=self.r45, r90=self.r90).calc_hill_coefficients_from_lankford()
            if coeffs is not None:
                F, G, H, N = coeffs
                self.F = self.F if self.F is not None else F
                self.G = self.G if self.G is not None else G
                self.H = self.H if self.H is not None else H
                self.N = self.N if self.N is not None else N

    def P_matrix(self) -> np.ndarray:
        self.ensure_hill_coefficients()
        if None in (self.F, self.G, self.H, self.N):
            raise ValueError("Hill48_3D needs F, G, H, N (or r0, r45, r90 to derive them)")
        return _hill_P_matrix(self.F, self.G, self.H, self.L, self.M, self.N)

    @classmethod
    def from_plane_stress(cls, criterion: Hill48_PS, L: float = 1.5, M: float = 1.5) -> "Hill48_3D":
        """3D criterion reusing an already-calibrated yield_criteria_plane_stress.Hill48_PS (F, G, H, N)."""
        criterion.ensure_hill_coefficients()
        return cls(r0=criterion.r0, r45=criterion.r45, r90=criterion.r90,
                   F=criterion.F, G=criterion.G, H=criterion.H, N=criterion.N, L=L, M=M)

    def to_abaqus_potential(self) -> dict:
        """
        Abaqus *POTENTIAL (Hill) yield stress ratios, taking the reference
        yield stress sigma0 as the RD uniaxial yield stress (so R11 = 1
        under G + H = 1):

            1/R11^2 = G + H,  1/R22^2 = F + H,  1/R33^2 = F + G,
            R12 = sqrt(3/(2N)),  R13 = sqrt(3/(2M)),  R23 = sqrt(3/(2L))
        """
        self.ensure_hill_coefficients()
        F, G, H, L, M, N = self.F, self.G, self.H, self.L, self.M, self.N
        return {
            "R11": 1.0 / np.sqrt(G + H),
            "R22": 1.0 / np.sqrt(F + H),
            "R33": 1.0 / np.sqrt(F + G),
            "R12": np.sqrt(1.5 / N),
            "R13": np.sqrt(1.5 / M),
            "R23": np.sqrt(1.5 / L),
        }

    @classmethod
    def from_abaqus_potential(cls, R11: float, R22: float, R33: float,
                              R12: float, R13: float, R23: float) -> "Hill48_3D":
        """
        Inverse of to_abaqus_potential:

            F = (1/R22^2 + 1/R33^2 - 1/R11^2)/2, G = (1/R33^2 + 1/R11^2 - 1/R22^2)/2,
            H = (1/R11^2 + 1/R22^2 - 1/R33^2)/2,
            L = 3/(2 R23^2), M = 3/(2 R13^2), N = 3/(2 R12^2)

        sigmaEQ is then referred to Abaqus's sigma0; it equals the RD
        yield stress only if R11 = 1.
        """
        a, b, c = 1.0 / R11 ** 2, 1.0 / R22 ** 2, 1.0 / R33 ** 2
        return cls(F=0.5 * (b + c - a), G=0.5 * (c + a - b), H=0.5 * (a + b - c),
                   L=1.5 / R23 ** 2, M=1.5 / R13 ** 2, N=1.5 / R12 ** 2)


# ----------------------------------------------------------------------
# Barlat 1991 (Yld91)
# ----------------------------------------------------------------------
@dataclass
class YLD91_3D(YieldCriterion3D):
    """
    Anisotropic Barlat 1991 (Yld91) criterion, general 3D form (Barlat,
    Lege & Brem, Int. J. Plasticity 7 (1991) 693-712):

        phi = |S1-S2|^a + |S2-S3|^a + |S3-S1|^a = 2 sigmaEQ^a

    with S1, S2, S3 the principal values of S = L sigma,

        S11 = [C3 (s11-s22) - C2 (s33-s11)] / 3
        S22 = [C1 (s22-s33) - C3 (s11-s22)] / 3
        S33 = [C2 (s33-s11) - C1 (s22-s33)] / 3
        S23 = C4 s23,  S13 = C5 s13,  S12 = C6 s12

    (the same L as yield_criteria_plane_stress.YLD91_PS, which only uses C1, C2, C3, C6).
    C4, C5 (out-of-plane shear) cannot be obtained from in-plane sheet
    tests. NOTE: yield_criteria_plane_stress.YLD91_PS's calibration rescales ALL of C1..C6
    by a common factor (to get sigmaEQ = sigma0 in RD tension), so its
    C4 = C5 come out equal to that factor rather than 1 -- from_plane_stress
    copies them as they are unless C4/C5 are given explicitly.
    """

    a: float = 8.0
    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: float = 1.0
    C5: float = 1.0
    C6: Optional[float] = None

    @classmethod
    def from_plane_stress(cls, criterion: YLD91_PS, C4: Optional[float] = None,
                          C5: Optional[float] = None) -> "YLD91_3D":
        """3D criterion reusing an already-calibrated yield_criteria_plane_stress.YLD91_PS (a, C1..C6)."""
        criterion.ensure_barlat_coefficients()
        if None in (criterion.C1, criterion.C2, criterion.C3, criterion.C6):
            raise ValueError("the plane-stress YLD91_PS could not be calibrated (C1, C2, C3, C6 missing)")
        return cls(a=criterion.a, C1=criterion.C1, C2=criterion.C2, C3=criterion.C3,
                   C4=C4 if C4 is not None else (criterion.C4 if criterion.C4 is not None else 1.0),
                   C5=C5 if C5 is not None else (criterion.C5 if criterion.C5 is not None else 1.0),
                   C6=criterion.C6)

    def L_matrix(self) -> np.ndarray:
        """6x6 linear transformation S = L sigma (Voigt order 11, 22, 33, 12, 13, 23)."""
        if None in (self.C1, self.C2, self.C3, self.C6):
            raise ValueError("YLD91_3D needs C1, C2, C3, C6 (see from_plane_stress)")
        C1, C2, C3 = self.C1, self.C2, self.C3
        L = np.zeros((6, 6))
        L[:3, :3] = np.array([[C2 + C3, -C3, -C2],
                              [-C3, C1 + C3, -C1],
                              [-C2, -C1, C1 + C2]]) / 3.0
        L[3, 3], L[4, 4], L[5, 5] = self.C6, self.C5, self.C4   # s12, s13, s23
        return L

    def _principal_transformed(self, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Eigen-decomposition of S = L sigma, with sigma pre-scaled by its
        largest component (sigmaEQ is homogeneous of degree 1, so this
        only avoids overflow of |S|^a). Returns (eigenvalues (..., 3),
        eigenvectors (..., 3, 3), scale (...)).
        """
        scale = np.max(np.abs(sigma), axis=-1)
        safe = np.where(scale > 0, scale, 1.0)
        S = voigt_to_tensor((sigma / safe[..., None]) @ self.L_matrix().T)
        w, V = np.linalg.eigh(S)
        return w, V, scale

    def _phi(self, w: np.ndarray) -> np.ndarray:
        a = self.a
        s1, s2, s3 = w[..., 0], w[..., 1], w[..., 2]
        return np.abs(s1 - s2) ** a + np.abs(s2 - s3) ** a + np.abs(s3 - s1) ** a

    def equivalent_stress(self, sigma):
        sigma = _as_voigt(sigma)
        w, _, scale = self._principal_transformed(sigma)
        return _scalar_or_array(scale * (0.5 * self._phi(w)) ** (1.0 / self.a))

    def flow_direction(self, sigma) -> np.ndarray:
        """
        d(sigmaEQ)/d(sigma) by the spectral chain rule: with S = sum_k
        S_k n_k n_k, d(phi)/dS = sum_k (d phi/d S_k) n_k n_k (well
        defined even for repeated S_k when a > 1), then through S = L sigma.
        """
        sigma = _as_voigt(sigma)
        w, V, scale = self._principal_transformed(sigma)
        a = self.a
        s1, s2, s3 = w[..., 0], w[..., 1], w[..., 2]
        sp = lambda x: np.sign(x) * np.abs(x) ** (a - 1.0)
        dphi = a * np.stack([sp(s1 - s2) - sp(s3 - s1),
                             sp(s2 - s3) - sp(s1 - s2),
                             sp(s3 - s1) - sp(s2 - s3)], axis=-1)
        dphi_dS = np.einsum("...ik,...k,...jk->...ij", V, dphi, V)
        # Voigt gradient w.r.t. S, shears counted once (hence the factor 2).
        g_S = tensor_to_voigt(dphi_dS) * np.array([1, 1, 1, 2, 2, 2])
        phi = self._phi(w)
        with np.errstate(divide="ignore", invalid="ignore"):
            # sigmaEQ = (phi/2)^(1/a)  ->  d sigmaEQ/d phi = sigmaEQ/(a phi);
            # evaluated on the scaled stress (gradient is scale-invariant).
            dseq_dphi = (0.5 * phi) ** (1.0 / a) / (a * phi)
            g = (g_S @ self.L_matrix()) * dseq_dphi[..., None]
            return np.where((scale > 0)[..., None] & (phi > 0)[..., None], g, np.nan)
