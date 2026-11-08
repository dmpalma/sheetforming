"""
Checks of sheetforming.triaxiality: the average triaxiality is the PEEQ-weighted
(Bao-Wierzbicki) average. Run with: uv run pytest
"""

from pathlib import Path

import numpy as np
import pytest

from sheetforming.yield_criteria_plane_stress import YLD91_PS
from sheetforming.triaxiality import calc_average_eta_and_lode

REPORT = Path(__file__).resolve().parent / 'data' / 'synthetic_spif_node.rpt'


@pytest.fixture(scope='module')
def df():
    if not REPORT.exists():
        pytest.skip(f'{REPORT} not found')
    return calc_average_eta_and_lode(REPORT, YLD91_PS(r0=0.80, r45=0.43, r90=0.61, a=8))


def test_avg_eta_is_peeq_weighted(df):
    """avg_eta(end) = (eta0*peeq0 + trapezoid of eta over peeq) / peeq, recomputed from the returned columns."""
    p, eta = df['peeq'].to_numpy(), df['eta'].to_numpy()
    dp = np.concatenate(([p[0]], np.clip(np.diff(p), 0.0, None)))
    pacc = np.cumsum(dp)
    integral = eta[0] * p[0] + np.sum(np.diff(pacc) * 0.5 * (eta[1:] + eta[:-1]))
    assert df['avg_eta'].iloc[-1] == pytest.approx(integral / pacc[-1], rel=1e-10)


def test_time_average_is_separate_column(df):
    t, eta = df['time'].to_numpy(), df['eta'].to_numpy()
    integral = np.sum(np.diff(t) * 0.5 * (eta[1:] + eta[:-1]))
    assert df['avg_eta_time'].iloc[-1] == pytest.approx(integral / (t[-1] - t[0]), rel=1e-10)


def _elemental_nodal_copy(tmp_path):
    """The same report as an Elemental Nodal export: 3 rows per instant (value, value, value) sharing X."""
    lines = REPORT.read_text().splitlines()
    blanks = [i for i, l in enumerate(lines) if l.strip() == '']
    k = blanks[1] + 1 if blanks[0] == 0 else blanks[0] + 1     # first data row
    out = lines[:k] + [l for l in lines[k:] if l.strip() for _ in range(3)]
    f = tmp_path / 'elemental.rpt'
    f.write_text('\n'.join(out) + '\n')
    return f


def test_elemental_nodal_report_is_rejected_unless_averaged(tmp_path):
    f = _elemental_nodal_copy(tmp_path)
    with pytest.raises(ValueError, match='several values per instant'):
        calc_average_eta_and_lode(f)
    crit = YLD91_PS(r0=0.80, r45=0.43, r90=0.61, a=8)
    a = calc_average_eta_and_lode(f, crit, repeated_times='average')
    b = calc_average_eta_and_lode(REPORT, crit)
    assert len(a) == len(b)
    assert a['avg_eta'].iloc[-1] == pytest.approx(b['avg_eta'].iloc[-1], rel=1e-10)


def _two_elements(tmp_path):
    """
    The report seen as two elements of the same node with different stress states (the second one
    has its principal stresses scaled by 0.7 / 1.5 / 1.0). Returns the files of element A alone,
    element B alone and both listed in each instant (A row, then B row).
    """
    lines = REPORT.read_text().splitlines()
    blanks = [i for i, l in enumerate(lines) if l.strip() == '']
    k = blanks[1] + 1 if blanks[0] == 0 else blanks[0] + 1
    head, data = lines[:k], [l for l in lines[k:] if l.strip()]

    def scaled(line):
        tok = line.split()
        tok[5:8] = [f'{float(t) * f:.6g}' for t, f in zip(tok[5:8], (0.7, 1.5, 1.0))]
        return ' '.join(tok)

    b = [scaled(l) for l in data]
    files = {}
    for name, rows in (('a', data), ('b', b), ('both', [x for pair in zip(data, b) for x in pair])):
        files[name] = tmp_path / f'{name}.rpt'
        files[name].write_text('\n'.join(head + rows) + '\n')
    return files


def test_averaging_elements_means_the_etas_of_the_elements(tmp_path):
    f = _two_elements(tmp_path)
    a, b = calc_average_eta_and_lode(f['a']), calc_average_eta_and_lode(f['b'])
    both = calc_average_eta_and_lode(f['both'], repeated_times='average')            # default: elements
    assert len(both) == len(a) == len(b)
    np.testing.assert_allclose(both['eta'], 0.5 * (a['eta'] + b['eta']), rtol=1e-10)
    np.testing.assert_allclose(both['theta'], 0.5 * (a['theta'] + b['theta']), rtol=1e-10)


def test_averaging_nodal_uses_the_mean_stress_and_differs(tmp_path):
    f = _two_elements(tmp_path)
    elements = calc_average_eta_and_lode(f['both'], repeated_times='average', averaging='elements')
    nodal = calc_average_eta_and_lode(f['both'], repeated_times='average', averaging='nodal')
    assert np.max(np.abs(elements['eta'].to_numpy() - nodal['eta'].to_numpy())) > 1e-3
    with pytest.raises(ValueError, match='averaging'):
        calc_average_eta_and_lode(f['both'], repeated_times='average', averaging='sideways')


def test_shell_report_with_components_uses_them_for_sigma_eq():
    from sheetforming.abaqus_utilities import read_abaqus_report
    from sheetforming.triaxiality import sigma_eq_from_report
    from sheetforming.yield_criteria_plane_stress import VonMises_PS
    df = read_abaqus_report(Path(__file__).resolve().parent / 'data' / 'synthetic_shell_node.rpt', None)
    s = sigma_eq_from_report(df, VonMises_PS())
    exact = np.sqrt(df.S11**2 - df.S11 * df.S22 + df.S22**2 + 3 * df.S12**2)
    np.testing.assert_allclose(s, exact, rtol=1e-12)
    # and the whole chain runs on a shell report that only has components
    out = calc_average_eta_and_lode(Path(__file__).resolve().parent / 'data' / 'synthetic_shell_node.rpt')
    assert len(out) > 10 and np.isfinite(out['avg_eta']).all()


def test_by_element_follows_each_row_of_the_instants(tmp_path):
    from sheetforming.triaxiality import calc_average_eta_and_lode_by_element
    f = _two_elements(tmp_path)
    with pytest.warns(RuntimeWarning, match='stable element order'):   # the synthetic PEEQ is noisy
        per = calc_average_eta_and_lode_by_element(f['both'])
    assert sorted(per) == [1, 2]
    for k, name in ((1, 'a'), (2, 'b')):
        single = calc_average_eta_and_lode(f[name])
        np.testing.assert_allclose(per[k]['avg_eta'], single['avg_eta'], rtol=1e-12)
        np.testing.assert_allclose(per[k]['peeq'], single['peeq'], rtol=1e-12)


def _solid_frame(sxx, syy, szz, sxy, sxz, syz):
    import pandas as pd
    df = pd.DataFrame({'S11': sxx, 'S22': syy, 'S33': szz, 'S12': sxy, 'S13': sxz, 'S23': syz})
    df.attrs['element_type'] = 'solid'
    return df


def test_solid_full_tensor_is_evaluated_with_the_3d_criterion():
    from sheetforming.triaxiality import sigma_eq_from_report
    from sheetforming.yield_criteria_3d import YLD91_3D
    from sheetforming.yield_criteria_plane_stress import VonMises_PS, YLD91_PS
    rng = np.random.default_rng(3)
    s = rng.uniform(-100, 200, size=(30, 6))
    df = _solid_frame(*s.T)
    # von Mises: closed form with the three shears
    exact = np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
                    + 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    np.testing.assert_allclose(sigma_eq_from_report(df, VonMises_PS()), exact, rtol=1e-10)
    # anisotropic: the 3D Yld91 on the Voigt vector (s11, s22, s33, s12, s13, s23)
    crit = YLD91_PS(r0=0.80, r45=0.43, r90=0.61, a=8)
    got = sigma_eq_from_report(df, crit)
    np.testing.assert_allclose(got, YLD91_3D.from_plane_stress(crit).equivalent_stress(s), rtol=1e-10)
    # rows with a missing component give NaN, not an error
    df.loc[0, 'S13'] = np.nan
    assert np.isnan(sigma_eq_from_report(df, crit)[0]) and np.isfinite(sigma_eq_from_report(df, crit)[1:]).all()


def test_add_principal_values_can_replace_the_exported_ones():
    from sheetforming import abaqus_utilities as au
    df = _solid_frame([100.0], [-50.0], [30.0], [20.0], [10.0], [-5.0])
    for c, v in (('PE11', 0.1), ('PE22', -0.05), ('PE33', -0.05), ('PE12', 0.0), ('PE13', 0.0), ('PE23', 0.0)):
        df[c] = v
    df['S_Max'], df['S_Mid'], df['S_Min'] = 999.0, 998.0, 997.0          # as exported by Abaqus
    au.add_principal_values(df)
    assert df.S_Max[0] == 999.0                                           # kept by default
    au.add_principal_values(df, overwrite=True)
    t = np.array([[100, 20, 10], [20, -50, -5], [10, -5, 30]], float)
    np.testing.assert_allclose([df.S_Max[0], df.S_Mid[0], df.S_Min[0]], np.sort(np.linalg.eigvalsh(t))[::-1])
