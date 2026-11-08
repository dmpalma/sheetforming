"""
Checks of sheetforming.constitutive_laws: fits recover known laws, inverse
relations. Run with: uv run pytest
"""

import numpy as np
import pytest

from sheetforming.constitutive_laws import (
    Hollomon, Swift, TensileData, Voce, VoceSwift, best_law, compare_fits, fit_laws,
)

EPS = np.linspace(0.001, 0.2, 200)


def curve(law, eps=EPS):
    """Stress at each strain (Hollomon, Swift and Voce take scalars only)."""
    return np.array([law.stress_from_strain(e) for e in eps])


def synthetic(law):
    return TensileData(stress=list(curve(law)), strain=list(EPS))


@pytest.mark.parametrize('law', [
    Hollomon(K=314.0, n=0.13),
    Swift(K=349.0, eps_0=0.002, n=0.23),
    Voce(sigma_0=94.0, sigma_s=220.0, C=10.0),
], ids=['Hollomon', 'Swift', 'Voce'])
def test_fit_recovers_synthetic_law(law):
    fit = law.__class__.fit_from_data(synthetic(law))
    assert np.max(np.abs(curve(fit) - curve(law))) < 0.5


def test_voce_fit_near_saturation():
    """Data within 1 % of saturation (the sigma_s search used to start at 1.01*max(stress))."""
    law = Voce(sigma_0=94.0, sigma_s=220.0, C=22.0)     # stress at eps_p = 0.2: 218.5 MPa
    fit = Voce.fit_from_data(synthetic(law))
    assert np.max(np.abs(curve(fit) - curve(law))) < 0.5


def test_voce_swift_fit_reproduces_synthetic_curve():
    """The joint fit recovers a Voce-Swift curve; the default fit (Swift and Voce fitted
    separately, then alpha) only approximates it (RMSE ~1.3 MPa on this curve)."""
    law = VoceSwift(K=349.1, eps_0=0.0018, n=0.23, sigma_0=94.2, sigma_s=220.1, C=22.8, alpha=0.68)
    joint = VoceSwift.fit_from_data(synthetic(law), joint=True)
    assert np.max(np.abs(curve(joint) - curve(law))) < 0.2
    default = VoceSwift.fit_from_data(synthetic(law))
    assert np.sqrt(np.mean((curve(default) - curve(law))**2)) < 1.5


@pytest.mark.parametrize('law', [
    Hollomon(K=314.0, n=0.13),
    Swift(K=349.0, eps_0=0.002, n=0.23),
    VoceSwift(K=349.1, eps_0=0.0018, n=0.23, sigma_0=94.2, sigma_s=220.1, C=22.8, alpha=0.68),
], ids=['Hollomon', 'Swift', 'VoceSwift'])
def test_strain_from_stress_inverts_stress_from_strain(law):
    for e in (0.01, 0.05, 0.1, 0.3):
        assert law.strain_from_stress(law.stress_from_strain(e)) == pytest.approx(e, rel=1e-6)


# ---------------------------------------------------------------------------- generate_curve and export
def test_generate_curve_by_step_or_by_number_of_points():
    law = Swift(K=349.1, eps_0=0.0018, n=0.23)
    stress, strain = law.generate_curve(max_strain=2.0, step=0.01)
    assert len(strain) == 201 and strain[0] == 0.0 and strain[-1] == 2.0
    np.testing.assert_allclose(np.diff(strain), 0.01, atol=1e-12)
    np.testing.assert_allclose(stress, [law.stress_from_strain(e) for e in strain])
    assert len(law.generate_curve(max_strain=0.5, step=0.05)[0]) == 11
    assert len(law.generate_curve(max_strain=2.0, n_points=50)[0]) == 50              # unchanged without a step
    with pytest.raises(ValueError, match='positive'):
        law.generate_curve(step=0.0)


def test_export_curve_has_the_stress_first(tmp_path):
    import pandas as pd
    law = Swift(K=349.1, eps_0=0.0018, n=0.23)
    path = law.export_curve_csv(tmp_path / 'curve.csv', max_strain=2.0, step=0.01)
    table = pd.read_csv(path)
    assert list(table.columns) == ['true_stress', 'plastic_strain'] and len(table) == 201
    assert table['plastic_strain'].iloc[-1] == 2.0 and table['true_stress'].iloc[0] == pytest.approx(law.stress_from_strain(0.0))


# ---------------------------------------------------------------------------- comparison of fitted laws
def _noisy(law, noise=0.0, seed=0):
    eps = np.linspace(0.0005, 0.22, 200)
    stress = np.array([law.stress_from_strain(e) for e in eps])
    if noise:
        stress = stress + np.random.default_rng(seed).normal(0.0, noise, stress.shape)
    return TensileData(stress=list(stress), strain=list(eps))


def test_the_generating_law_is_the_best_fit():
    swift_data = _noisy(Swift(K=349.1, eps_0=0.0018, n=0.23), noise=0.05)
    assert best_law(compare_fits(swift_data, fit_laws(swift_data))) == 'Swift'
    voce_data = _noisy(Voce(sigma_0=95.0, sigma_s=230.0, C=9.0), noise=0.05)
    assert best_law(compare_fits(voce_data, fit_laws(voce_data))) == 'Voce'


def test_comparison_table_columns_and_contents():
    data = _noisy(Swift(K=349.1, eps_0=0.0018, n=0.23))
    table = compare_fits(data, fit_laws(data))
    assert list(table['law']) == ['Swift', 'Voce', 'VoceSwift']
    for column in ('K', 'eps_0', 'n', 'sigma_0', 'sigma_s', 'C', 'alpha', 'n_parameters', 'n_points', 'rmse',
                   'max_error', 'aic', 'bic', 'stress_at_end', 'monotonic', 'best'):
        assert column in table.columns
    by_law = table.set_index('law')
    assert by_law.loc['Swift', 'K'] == pytest.approx(349.1, rel=0.02) and by_law.loc['Swift', 'n'] == pytest.approx(0.23, abs=0.01)
    assert np.isnan(by_law.loc['Swift', 'sigma_0']) and by_law.loc['Swift', 'n_parameters'] == 3
    assert by_law.loc['VoceSwift', 'n_parameters'] == 7 and table['best'].sum() == 1
    assert (table['n_points'] == 200).all() and table['rmse'].lt(5.0).all()          # all describe the measured range
    assert by_law.loc['Swift', 'rmse'] < by_law.loc['Voce', 'rmse']
    assert by_law.loc['Swift', 'stress_at_end'] > by_law.loc['Voce', 'stress_at_end']    # Swift keeps hardening


def test_a_decreasing_extrapolation_is_flagged_and_not_chosen():
    """Data that soften are fitted best by a decreasing Voce, which must not be flagged 'best' next to a rising law."""
    softening = Voce(sigma_0=200.0, sigma_s=120.0, C=8.0)
    data = _noisy(softening)
    fits = {'Voce': softening, 'Swift': Swift(K=150.0, eps_0=0.01, n=0.05)}
    table = compare_fits(data, fits).set_index('law')
    assert not table.loc['Voce', 'monotonic'] and table.loc['Swift', 'monotonic']
    assert table.loc['Voce', 'bic'] < table.loc['Swift', 'bic']                        # fits better...
    assert best_law(compare_fits(data, fits)) == 'Swift'                              # ...but it is not recommended
    with pytest.warns(RuntimeWarning, match='non-decreasing'):
        compare_fits(data, {'Voce': softening})


def test_unknown_law_is_rejected():
    with pytest.raises(ValueError, match='unknown laws'):
        fit_laws(_noisy(Swift(K=349.1, eps_0=0.0018, n=0.23)), names=('Swift', 'Hollomon'))
