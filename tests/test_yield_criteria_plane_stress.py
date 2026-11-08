"""
Checks of sheetforming.yield_criteria_plane_stress: isotropic limits, flow rule
(gradients, Euler identity, alpha <-> beta), calibrations and the Yld2000-2d
against Barlat et al. (2003) and Bressan et al. (2025). Run with: uv run pytest
"""

import numpy as np
import pytest

from sheetforming.yield_criteria_plane_stress import (
    Hill48_PS, Hill79_PS, PointState, VonMises_PS, YLD91_PS, YLD2000_PS,
)

rng = np.random.default_rng(0)
STATES = rng.normal(size=(100, 3))          # random (sigma_xx, sigma_yy, tau_xy)
ISOTROPIC = {f'alpha{i}': 1.0 for i in range(1, 9)}


def r_theta(c, deg):
    return c.calc_lankford_from_theta_stress(PointState(theta_stress=np.deg2rad(deg)))


def s_theta(c, deg):
    return c.calc_sigma_from_theta_stress(PointState(theta_stress=np.deg2rad(deg)))


def hosford(sxx, syy, txy, a):
    s1, s2, _ = VonMises_PS().calc_principal_stresses_from_xy(PointState(sigma_xx=sxx, sigma_yy=syy, tau_xy=txy))
    return (0.5 * (abs(s1 - s2)**a + abs(s1)**a + abs(s2)**a))**(1 / a)


# ---- criteria with known (or self-made) calibrations, for the generic checks ----
CRITERIA = [
    VonMises_PS(),
    Hill48_PS(r0=0.8, r45=0.43, r90=0.61),
    YLD91_PS(r0=0.8, r45=0.43, r90=0.61, a=8),
    YLD2000_PS(a=8, sigma0=1.0, sigma45=0.8114, sigma90=0.9102, sigmab=1.0350,
               r0=0.2115, r45=1.5769, r90=0.6923, rb=0.6700),
]
IDS = ['VonMises', 'Hill48', 'YLD91', 'YLD2000']


# ---- isotropic limits ----
def test_yld2000_isotropic_is_von_mises():
    c = YLD2000_PS(a=2, **ISOTROPIC)
    vm = VonMises_PS()
    err = max(abs(c._equivalent_stress_xy(*x) - vm._equivalent_stress_xy(*x)) for x in STATES)
    assert err < 1e-12


def test_yld2000_isotropic_is_hosford():
    c = YLD2000_PS(a=8, **ISOTROPIC)
    err = max(abs(c._equivalent_stress_xy(*x) - hosford(*x, 8)) for x in STATES)
    assert err < 1e-12


def test_yld91_isotropic_is_von_mises():
    c = YLD91_PS(a=2, C1=1.0, C2=1.0, C3=1.0, C4=1.0, C5=1.0, C6=1.0)
    vm = VonMises_PS()
    err = max(abs(c._equivalent_stress_xy(*x) - vm._equivalent_stress_xy(*x)) for x in STATES)
    assert err < 1e-12


def test_von_mises_is_frame_indifferent():
    """sigmaEQ of a uniaxial stress does not depend on its angle to the axes."""
    vm = VonMises_PS()
    for deg in range(0, 91, 15):
        t = np.deg2rad(deg)
        st = vm.solve(sigma_xx=150 * np.cos(t)**2, sigma_yy=150 * np.sin(t)**2, tau_xy=150 * np.sin(t) * np.cos(t))
        assert st.sigmaEQ == pytest.approx(150.0)


# ---- flow rule ----
@pytest.mark.parametrize('c', CRITERIA[2:], ids=IDS[2:])
def test_gradient_vs_finite_differences(c):
    """Exact gradient of sigmaEQ (YLD91: up to a positive factor, so compare directions)."""
    h = 1e-6
    for x in STATES[:30]:
        g = np.array(c._flow_gradient_xy(*x))
        fd = np.array([(c._equivalent_stress_xy(*(x + h * e)) - c._equivalent_stress_xy(*(x - h * e))) / (2 * h)
                       for e in np.eye(3)])
        assert np.allclose(g / np.linalg.norm(g), fd / np.linalg.norm(fd), atol=1e-6)


@pytest.mark.parametrize('c', CRITERIA, ids=IDS)
def test_unit_rd_tension_normalisation(c):
    assert c.solve(sigma1=1.0, sigma2=0.0).sigmaEQ == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize('c', CRITERIA, ids=IDS)
def test_alpha_beta_round_trip(c):
    for alpha in (-0.8, -0.3, 0.0, 0.4, 0.9):
        beta = c.calc_beta_from_alpha(PointState(alpha=alpha))
        assert c.calc_alpha_from_beta(PointState(beta=beta)) == pytest.approx(alpha, abs=1e-6)


@pytest.mark.parametrize('c', CRITERIA, ids=IDS)
def test_plastic_work_conjugacy(c):
    """sigma1*deps1 + sigma2*deps2 = sigmaEQ*depsEQ for the associated flow rule (off-axis too)."""
    for sxx, syy, txy in STATES[:20]:
        st = c.solve(sigma_xx=sxx, sigma_yy=syy, tau_xy=txy, depsEQ=0.01)
        assert st.sigma1 * st.deps1 + st.sigma2 * st.deps2 == pytest.approx(st.sigmaEQ * 0.01, rel=1e-8)


def test_uniaxial_beta_from_r0():
    """Uniaxial tension in RD: beta = -r0/(1 + r0) for criteria that reproduce r0."""
    for c in (CRITERIA[1], CRITERIA[3]):
        st = c.solve(sigma1=1.0, sigma2=0.0)
        assert st.beta == pytest.approx(-c.r0 / (1 + c.r0), abs=1e-8)


# ---- calibrations ----
def test_hill48_reproduces_r_values():
    c = CRITERIA[1]
    assert [r_theta(c, d) for d in (0, 45, 90)] == pytest.approx([0.8, 0.43, 0.61], abs=1e-10)


def test_yld91_r_calibration_reproduces_r_values():
    c = CRITERIA[2]
    assert [r_theta(c, d) for d in (0, 45, 90)] == pytest.approx([0.8, 0.43, 0.61], abs=1e-8)


def test_yld91_stress_calibration_reproduces_stresses():
    c = YLD91_PS(sigma0=97.7, sigma45=99.0, sigma90=97.5, a=8)
    c.solve(sigmaEQ=97.7)
    assert [s_theta(c, d) for d in (0, 45, 90)] == pytest.approx([1.0, 99.0 / 97.7, 97.5 / 97.7], abs=1e-8)


def test_yld2000_aa2090_barlat_2003():
    """Yld2000-2d of AA2090-T3 (Barlat et al. 2003): the 8 data exactly, alphas close to the published ones."""
    c = CRITERIA[3]
    assert [s_theta(c, d) for d in (0, 45, 90)] == pytest.approx([1.0, 0.8114, 0.9102], abs=1e-8)
    assert [r_theta(c, d) for d in (0, 45, 90)] == pytest.approx([0.2115, 1.5769, 0.6923], abs=1e-8)
    assert 1 / c._equivalent_stress_xy(1.0, 1.0, 0.0) == pytest.approx(1.0350, abs=1e-8)
    assert c.calc_rb() == pytest.approx(0.6700, abs=1e-8)
    published = (0.4865, 1.3783, 0.7536, 1.0246, 1.0363, 0.9036, 1.2321, 1.4858)
    assert [getattr(c, f'alpha{i}') for i in range(1, 9)] == pytest.approx(published, abs=2e-3)


def test_yld2000_bressan_coefficients():
    """Bressan et al. (2025) eq. (13), written literally, vs YLD2000_PS.from_bressan_coefficients."""
    b, m = (1.025793, 1.058188, 1.049630, 1.022336, 0.948546, 0.989996, 0.890357, 1.331321), 8

    def phi(sxx, syy, txy):
        dpp = np.sqrt((b[2] * sxx - b[3] * syy)**2 + (2 * b[7] * txy)**2)
        dp = np.sqrt((b[4] * sxx - b[5] * syy)**2 + (2 * b[6] * txy)**2)
        lin = b[0] * sxx + b[1] * syy
        return abs(lin + dpp)**m / 2**m + abs(lin - dpp)**m / 2**m + dp**m

    c = YLD2000_PS.from_bressan_coefficients(b, a=m)
    assert max(abs((0.5 * phi(*x))**(1 / m) - c._equivalent_stress_xy(*x)) for x in STATES) < 1e-12
    assert c.bressan_coefficients() == pytest.approx(b, abs=1e-12)


def test_hill79_runs():
    c = Hill79_PS(r0=0.8, r45=0.43, r90=0.61, m=8)
    st = c.solve(sigma1=1.0, alpha=0.0)
    assert st.beta == pytest.approx(-0.8 / 1.8, abs=1e-6)
