"""
Consistency checks of sheetforming.yield_criteria_3d against the plane-stress
criteria. Run with: uv run pytest
"""

import numpy as np

from sheetforming.yield_criteria_plane_stress import Hill48_PS, PointState, VonMises_PS, YLD91_PS
from sheetforming.yield_criteria_3d import (
    Hill48_3D, VonMises_3D, YLD91_3D,
    in_plane_rotation, rotate_voigt_stress,
)

rng = np.random.default_rng(0)
THETAS = np.deg2rad(np.linspace(0.0, 90.0, 19))


def _check(label, err, tol):
    print(f"  {label:58s} max err = {err:.2e}")
    assert err < tol, f"{label}: {err} >= {tol}"


def _criteria_pairs():
    """(plane-stress criterion, its 3D counterpart) for every class in yield_criteria_3d."""
    r0, r45, r90 = 0.80, 0.43, 0.61
    hill = Hill48_PS(r0=r0, r45=r45, r90=r90)
    yld_r = YLD91_PS(r0=r0, r45=r45, r90=r90, a=8)
    return [
        (VonMises_PS(), VonMises_3D()),
        (hill, Hill48_3D.from_plane_stress(hill)),
        (yld_r, YLD91_3D.from_plane_stress(yld_r)),
    ]


def test_plane_stress_consistency():
    """With s33 = s13 = s23 = 0, each 3D criterion must match its plane-stress version."""
    print("--- Plane-stress consistency (s33 = s13 = s23 = 0) ---")
    for crit2d, crit3d in _criteria_pairs():
        print(f" {crit2d.name} (+ {crit3d.name})")

        # sigmaEQ at in-plane principal states aligned with RD/TD.
        s = rng.uniform(-1.0, 1.0, size=(200, 2))
        seq2d = np.array([crit2d.calc_equivalent_stress(PointState(sigma1=a, sigma2=b)) for a, b in s])
        voigt = np.column_stack([s[:, 0], s[:, 1], np.zeros((200, 4))])
        _check("sigmaEQ (aligned principal states)", np.max(np.abs(crit3d.equivalent_stress(voigt) - seq2d)), 1e-10)

        # Off-axis uniaxial yield stress and Lankford ratio.
        sig2d = np.array([crit2d.calc_sigma_from_theta_stress(PointState(theta_stress=t)) for t in THETAS])
        r2d = np.array([crit2d.calc_lankford_from_theta_stress(PointState(theta_stress=t)) for t in THETAS])
        _check("sigma(theta)", np.max(np.abs(crit3d.yield_stress_vs_angle(THETAS) - sig2d)), 1e-10)
        _check("r(theta) (3D: true dPE33 as thickness strain)",
               np.max(np.abs(crit3d.lankford_vs_angle(THETAS) - r2d)), 1e-8)


def test_off_axis_plane_stress():
    """
    General in-plane states (sigma_xx, sigma_yy, tau_xy) in material axes:
    the plane-stress solve() must give the same sigmaEQ as the 3D class,
    and its beta and flow-rule deps1, deps2 must be the 3D flow direction
    projected on the stress principal axes. Also checks the off-axis
    inverses alpha(beta) and sigma1(sigmaEQ, alpha) at fixed theta_stress.
    """
    print("--- Off-axis plane-stress states vs 3D ---")
    rng = np.random.default_rng(0)   # own generator: results must not depend on other tests' draws
    for crit2d, crit3d in _criteria_pairs():
        print(f" {crit2d.name}")
        e_seq = e_beta = e_flow = e_alpha = e_s1 = 0.0
        for sxx, syy, txy in rng.uniform(-1.0, 1.0, size=(200, 3)):
            v = np.array([sxx, syy, 0.0, txy, 0.0, 0.0])
            st = crit2d.solve(sigma_xx=sxx, sigma_yy=syy, tau_xy=txy, depsEQ=0.01)
            e_seq = max(e_seq, abs(st.sigmaEQ - crit3d.equivalent_stress(v)))

            g = crit3d.flow_direction(v)
            c, s = np.cos(st.theta_stress), np.sin(st.theta_stress)
            g1 = g[0] * c * c + g[1] * s * s + g[3] * s * c
            g2 = g[0] * s * s + g[1] * c * c - g[3] * s * c
            e_beta = max(e_beta, abs(st.beta - g2 / g1))
            e_flow = max(e_flow, abs(st.deps1 - 0.01 * g1), abs(st.deps2 - 0.01 * g2))

            alpha = crit2d.calc_alpha_from_beta(PointState(beta=st.beta, theta_stress=st.theta_stress))
            e_alpha = max(e_alpha, abs(alpha - st.alpha))
            if st.sigma1 > 0:   # sigma1(sigmaEQ, alpha) assumes sigma1 > 0
                s1 = crit2d.calc_sigma1_from_equivalent_stress_and_alpha(
                    PointState(sigmaEQ=st.sigmaEQ, alpha=st.alpha, theta_stress=st.theta_stress))
                e_s1 = max(e_s1, abs(s1 - st.sigma1))
        _check("sigmaEQ from (sigma_xx, sigma_yy, tau_xy)", e_seq, 1e-10)
        _check("beta = projected 3D flow direction", e_beta, 1e-8)
        _check("flow rule deps1, deps2 (depsEQ = 0.01)", e_flow, 1e-12)
        _check("alpha(beta) at fixed theta_stress", e_alpha, 1e-5)
        _check("sigma1(sigmaEQ, alpha) at fixed theta_stress", e_s1, 1e-10)


def test_flow_direction():
    """Euler identity, finite-difference gradient, and pressure insensitivity on random 3D states."""
    print("--- Flow direction on random full 3D stress states ---")
    sigma = rng.uniform(-1.0, 1.0, size=(100, 6))
    for _, crit3d in _criteria_pairs():
        print(f" {crit3d.name}")
        seq = crit3d.equivalent_stress(sigma)
        g = crit3d.flow_direction(sigma)
        _check("Euler: sigma . grad = sigmaEQ", np.max(np.abs(np.sum(sigma * g, axis=1) - seq)), 1e-10)

        h = 1e-6
        g_fd = np.empty_like(sigma)
        for k in range(6):
            dp, dm = sigma.copy(), sigma.copy()
            dp[:, k] += h
            dm[:, k] -= h
            g_fd[:, k] = (crit3d.equivalent_stress(dp) - crit3d.equivalent_stress(dm)) / (2 * h)
        _check("grad vs central finite differences", np.max(np.abs(g - g_fd)), 1e-6)

        p = rng.uniform(-1.0, 1.0, size=(100, 1))
        shifted = sigma + p * np.array([1, 1, 1, 0, 0, 0])
        _check("pressure insensitivity", np.max(np.abs(crit3d.equivalent_stress(shifted) - seq)), 1e-10)
        _check("plastic incompressibility: tr(grad) = 0", np.max(np.abs(g[:, :3].sum(axis=1))), 1e-10)


def test_isotropic_limits():
    """Yld91 with a = 2, C1..C6 = 1 is von Mises (shears included); von Mises is frame-indifferent."""
    print("--- Isotropic limits ---")
    sigma = rng.uniform(-1.0, 1.0, size=(100, 6))
    vm = VonMises_3D()
    yld_iso = YLD91_3D(a=2.0, C1=1.0, C2=1.0, C3=1.0, C4=1.0, C5=1.0, C6=1.0)
    _check("YLD91_3D(a=2, C=1) vs VonMises_3D",
           np.max(np.abs(yld_iso.equivalent_stress(sigma) - vm.equivalent_stress(sigma))), 1e-10)

    R = in_plane_rotation(0.7) @ np.array([[1, 0, 0], [0, np.cos(0.3), np.sin(0.3)], [0, -np.sin(0.3), np.cos(0.3)]])
    _check("VonMises_3D under a general rotation",
           np.max(np.abs(vm.equivalent_stress(rotate_voigt_stress(sigma, R)) - vm.equivalent_stress(sigma))), 1e-10)


def test_plastic_strain_round_trip():
    """dPE = dPEEQ * grad, then dPEEQ = sigma . dPE / sigmaEQ must give dPEEQ back."""
    print("--- Plastic strain increment round trip ---")
    sigma = rng.uniform(-1.0, 1.0, size=(50, 6))
    depsEQ = rng.uniform(0.0, 0.01, size=50)
    for _, crit3d in _criteria_pairs():
        deps = crit3d.plastic_strain_increment(sigma, depsEQ)
        _check(f"{crit3d.name}: dPEEQ round trip",
               np.max(np.abs(crit3d.equivalent_plastic_strain_increment(sigma, deps) - depsEQ)), 1e-12)


def test_hill48_abaqus_potential():
    """Hill48_3D <-> Abaqus *POTENTIAL ratios (R11 = 1 under G + H = 1)."""
    print("--- Hill48 <-> Abaqus *POTENTIAL ---")
    hill = Hill48_3D(r0=0.80, r45=0.43, r90=0.61)
    R = hill.to_abaqus_potential()
    print("  " + ", ".join(f"{k} = {v:.4f}" for k, v in R.items()))
    back = Hill48_3D.from_abaqus_potential(**R)
    err = max(abs(getattr(back, k) - getattr(hill, k)) for k in "FGHLMN")
    _check("round trip F, G, H, L, M, N", err, 1e-12)
