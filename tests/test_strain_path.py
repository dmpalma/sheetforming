"""
path_from_abaqus_report builds a strain path (full in-plane plastic strain tensor in
fixed axes + PEEQ increments) from an Abaqus report. Run with: uv run pytest
"""

from pathlib import Path

import numpy as np
import pytest

from sheetforming.abaqus_utilities import read_abaqus_report
from sheetforming.yield_functions import (
    DEFAULT_PATH, StrainIncrement, beta_from_path, downsample_path, path_from_abaqus_report,
)

DATA = Path(__file__).resolve().parent / 'data'
SHELL = DATA / 'synthetic_shell_node.rpt'      # PE11, PE22, PE12 (engineering), PEEQ; Unique Nodal
SOLID = DATA / 'synthetic_spif_node.rpt'       # principal values only


@pytest.fixture(scope='module')
def path():
    return path_from_abaqus_report(SHELL)


def test_one_increment_per_interval(path):
    df = read_abaqus_report(SHELL, None)
    assert len(path) == len(df) - 1
    assert all(isinstance(p, StrainIncrement) for p in path)


def test_increments_add_up_to_the_final_state(path):
    """Sum of the increments = final value of each column; PE12 is engineering, so deps_xy = PE12/2."""
    df = read_abaqus_report(SHELL, None)
    last = df.iloc[-1]
    assert sum(p.deps_xx for p in path) == pytest.approx(last['PE11'])
    assert sum(p.deps_yy for p in path) == pytest.approx(last['PE22'])
    assert sum(p.deps_xy for p in path) == pytest.approx(last['PE12'] / 2)
    assert sum(p.depsEQ for p in path) == pytest.approx(last['PEEQ'])


def test_shear_and_elastic_segments_are_kept(path):
    assert path[0].deps_xy == 0.0 and path[0].deps_xx > 0              # tension along x
    assert all(p.depsEQ == 0.0 for p in path[30:40])                   # elastic dwell
    assert path[45].deps_xy > 0 and path[45].deps_xx == 0.0            # pure shear
    assert path[-1].deps_xx < 0                                        # reversal


def test_depsEQ_is_consistent_with_the_tensor_increments(path):
    """depsEQ = sqrt(2/3 de:de) with e33 = -(exx+eyy) (plastic incompressibility) and tensor shear."""
    for p in path:
        e33 = -(p.deps_xx + p.deps_yy)
        de = np.sqrt(2 / 3 * (p.deps_xx ** 2 + p.deps_yy ** 2 + e33 ** 2 + 2 * p.deps_xy ** 2))
        assert p.depsEQ == pytest.approx(de, abs=1e-6)   # report values carry 6 digits


def test_downsample_preserves_every_component(path):
    short = downsample_path(path, 10)
    assert len(short) < len(path)
    for k in range(4):
        assert sum(s[k] for s in short) == pytest.approx(sum(p[k] for p in path))


def test_beta_from_path(path):
    betas = beta_from_path(path)
    assert len(betas) == len(path) + 1
    assert betas[1] == pytest.approx(-0.5)                              # uniaxial tension
    assert beta_from_path(DEFAULT_PATH) == [-0.5] * (len(DEFAULT_PATH) + 1)


def test_report_without_tensor_components_is_rejected():
    with pytest.raises(ValueError, match='PE11'):
        path_from_abaqus_report(SOLID)


def test_shell_principal_values_are_derived_from_the_components():
    """A shell report exported with the tensor components only gets S_*/PE_* (in-plane max, min, out-of-plane)."""
    from sheetforming import abaqus_utilities as au
    full = read_abaqus_report(SHELL, None)
    bare = full.drop(columns=['S_Max', 'S_Mid', 'S_Min', 'PE_Max', 'PE_Mid', 'PE_Min'])
    au._add_shell_principal_columns(bare)
    for c in ('S_Max', 'S_Mid', 'S_Min', 'PE_Max', 'PE_Mid', 'PE_Min'):
        np.testing.assert_allclose(bare[c], full[c], atol=1e-4)


def test_column_names_do_not_depend_on_the_averaging_label():
    from sheetforming.abaqus_utilities import _report_column_name
    for label in ("(Avg: 75p)", "(Not averaged)"):
        assert _report_column_name(f"PE:PE11 {label} SP:1 PI: SHEET-1 N: 19853") == "PE11"
        assert _report_column_name(f"PEEQ {label} SP:1 PI: SHEET-1 N: 19853") == "PEEQ"
        assert _report_column_name(f"S:S12 {label} SP:1 PI: SHEET-1 N: 19853") == "S12"


def test_shell_out_of_plane_stress_novalue_is_zero(tmp_path):
    """Abaqus writes NoValue for S33 (zero in a shell) in most rows of a node: it must read as 0, not NaN."""
    lines = SHELL.read_text().splitlines()
    blanks = [i for i, l in enumerate(lines) if l.strip() == '']
    k = blanks[1] + 1 if blanks[0] == 0 else blanks[0] + 1
    titles = lines[:k]
    out = titles[:]
    for n, l in enumerate(lines[k:]):
        if not l.strip():
            continue
        tok = l.split()
        out.append(l)
        tok[-1] = 'NoValue'                                      # S33, last column
        out.append(' '.join(tok))
    f = tmp_path / 'novalue.rpt'
    f.write_text('\n'.join(out) + '\n')
    d = read_abaqus_report(f, None, repeated_times='keep')
    assert d['S33'].notna().all() and d['S_Min'].notna().all()
    assert len(d) == 2 * len(read_abaqus_report(SHELL, None))


def test_principal_values_of_averaged_rows_come_from_the_mean_tensor(tmp_path):
    """Two rows per instant (two elements): with repeated_times='average' the shell principal values are
    those of the mean tensor, not the mean of the principal values."""
    lines = SHELL.read_text().splitlines()
    blanks = [i for i, l in enumerate(lines) if l.strip() == '']
    k = blanks[1] + 1 if blanks[0] == 0 else blanks[0] + 1
    head, data = lines[:k], [l for l in lines[k:] if l.strip()]
    names = read_abaqus_report(SHELL, None).columns.tolist()
    i11, i22, i12 = names.index('S11'), names.index('S22'), names.index('S12')
    rows = []
    for l in data:
        rows.append(l)
        tok = l.split()
        tok[i11], tok[i22], tok[i12] = (f'{float(tok[i]) * f:.6g}' for i, f in ((i11, 0.5), (i22, 2.0), (i12, -1.0)))
        rows.append(' '.join(tok))                       # principal columns left untouched on purpose
    f = tmp_path / 'two_elements.rpt'
    f.write_text('\n'.join(head + rows) + '\n')
    d = read_abaqus_report(f, None, repeated_times='average')
    mean_pairs = np.hypot(0.5 * (d.S11 - d.S22), d.S12)
    np.testing.assert_allclose(d.S_Max, 0.5 * (d.S11 + d.S22) + mean_pairs, atol=1e-4)
    np.testing.assert_allclose(d.S_Mid, 0.5 * (d.S11 + d.S22) - mean_pairs, atol=1e-4)


def test_solid_principal_values_come_from_the_full_tensor(tmp_path):
    """A solid report with the full tensor gets S_Max/S_Mid/S_Min (sorted) and PE_* (PE shears are engineering)."""
    from sheetforming import abaqus_utilities as au
    df = au.pd.DataFrame({'X': [0.0, 1.0], 'S11': [100.0, 10.0], 'S22': [-50.0, 20.0], 'S33': [30.0, 30.0],
                          'S12': [20.0, 0.0], 'S13': [10.0, 0.0], 'S23': [-5.0, 0.0],
                          'PE11': [0.1, 0.0], 'PE22': [-0.05, 0.0], 'PE33': [-0.05, 0.0],
                          'PE12': [0.04, 0.0], 'PE13': [0.02, 0.0], 'PE23': [0.0, 0.0]})
    au._add_solid_principal_columns(df)
    t = np.array([[100, 20, 10], [20, -50, -5], [10, -5, 30]], float)
    np.testing.assert_allclose([df.S_Max[0], df.S_Mid[0], df.S_Min[0]], np.sort(np.linalg.eigvalsh(t))[::-1])
    pe = np.array([[0.1, 0.02, 0.01], [0.02, -0.05, 0.0], [0.01, 0.0, -0.05]])
    np.testing.assert_allclose([df.PE_Max[0], df.PE_Mid[0], df.PE_Min[0]], np.sort(np.linalg.eigvalsh(pe))[::-1])
    np.testing.assert_allclose([df.S_Max[1], df.S_Mid[1], df.S_Min[1]], [30.0, 20.0, 10.0])


def test_solid_report_without_full_tensor_gets_a_clear_error(tmp_path):
    """Solid report exported with stress components but without S13, S23: no principal values, clear message."""
    from sheetforming.triaxiality import calc_average_eta_and_lode
    text = SOLID.read_text()
    for principal, component in (('Max Principal', 'S11'), ('Mid Principal', 'S22'), ('Min Principal', 'S33')):
        text = text.replace(f'S:{principal}', f'S:{component}')
    f = tmp_path / 'components_only.rpt'
    f.write_text(text)
    with pytest.raises(ValueError, match='S13'):
        calc_average_eta_and_lode(f)


def _two_element_shell(tmp_path):
    """The shell report as two elements of one node: the original path and the same path at half the strain."""
    lines = SHELL.read_text().splitlines()
    blanks = [i for i, l in enumerate(lines) if l.strip() == '']
    k = blanks[1] + 1 if blanks[0] == 0 else blanks[0] + 1
    head, data = lines[:k], [l for l in lines[k:] if l.strip()]
    names = read_abaqus_report(SHELL, None).columns.tolist()
    idx = [names.index(c) for c in ('PE11', 'PE22', 'PE12', 'PEEQ')]
    rows = []
    for l in data:
        rows.append(l)
        tok = l.split()
        for i in idx:
            tok[i] = f'{float(tok[i]) * 0.5:.6g}'
        rows.append(' '.join(tok))
    f = tmp_path / 'two_elements_path.rpt'
    f.write_text('\n'.join(head + rows) + '\n')
    return f


def test_path_by_element(tmp_path):
    f = _two_element_shell(tmp_path)
    one = path_from_abaqus_report(SHELL)
    first = path_from_abaqus_report(f, element=1)
    second = path_from_abaqus_report(f, element=2)
    mean = path_from_abaqus_report(f, repeated_times='average')
    assert len(first) == len(second) == len(mean) == len(one)
    for a, b, c, m in zip(one, first, second, mean):
        np.testing.assert_allclose(b, a, atol=1e-9)
        np.testing.assert_allclose(c, 0.5 * np.array(a), atol=1e-6)
        np.testing.assert_allclose(m, 0.75 * np.array(a), atol=1e-6)        # the mean tensor
    from sheetforming.yield_functions import paths_by_element
    both = paths_by_element(f)
    assert sorted(both) == [1, 2] and both[2] == second
    with pytest.raises(ValueError, match='element 3'):
        path_from_abaqus_report(f, element=3)


def test_path_cut(tmp_path):
    full = path_from_abaqus_report(SHELL)
    df = read_abaqus_report(SHELL, None)
    t_cut = float(df.X.iloc[40])
    cut = path_from_abaqus_report(SHELL, until_time=t_cut)
    assert len(cut) == 40 and cut == full[:40]
    peeq_cut = float(df.PEEQ.iloc[50]) - 1e-9
    by_peeq = path_from_abaqus_report(SHELL, until_peeq=peeq_cut)
    n = int(np.argmax(df.PEEQ.to_numpy() >= peeq_cut))                      # first instant reaching it, included
    assert len(by_peeq) == n and by_peeq == full[:n]
    assert sum(p.depsEQ for p in by_peeq) >= peeq_cut - 1e-9
    shorter = path_from_abaqus_report(SHELL, until_time=t_cut, until_peeq=peeq_cut)
    assert len(shorter) == min(len(cut), len(by_peeq))


def _rewrite_rows(tmp_path, transform, name):
    """Copy of SHELL with every data row passed through transform(tokens, row_index) -> list of rows."""
    text = SHELL.read_text()
    lines = text.splitlines()
    blanks = [i for i, l in enumerate(lines) if l.strip() == '']
    k = blanks[1] + 1 if blanks[0] == 0 else blanks[0] + 1
    head = lines[:k]
    rows = []
    for n, l in enumerate([x for x in lines[k:] if x.strip()]):
        rows.extend(transform(l.split(), n))
    f = tmp_path / name
    f.write_text('\n'.join(head + [' '.join(r) if isinstance(r, list) else r for r in rows]) + '\n')
    return f


def _write_report(path, columns, rows, node=1):
    """Abaqus-style .rpt (wrapped fixed-width header; X 27 characters, other columns 19) from a list of rows."""
    titles = [c + f" (Avg: 75p) PI: SHEET-1 N: {node}" for c in columns]
    chunks = [[t[i:i + 17] for i in range(0, len(t), 17)] for t in titles]
    n_lines = max(len(c) for c in chunks)
    lines = [""]
    for k in range(n_lines):
        lines.append(("X" if k == n_lines - 1 else "").ljust(27) +
                     "".join("  " + (c[k] if k < len(c) else "").ljust(17) for c in chunks) + " ")
    lines.append("")
    for row in rows:
        lines.append(f"{row[0]:.6g}".rjust(27) + "".join(f"{v:.6g}".rjust(19) for v in row[1:]))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_path_from_a_solid_report(tmp_path):
    """A solid report with the full tensor: the same path as the shell report, and the principal values are derived."""
    df = read_abaqus_report(SHELL, None)
    zero = np.zeros(len(df))
    cols = ['PE:PE11', 'PE:PE22', 'PE:PE33', 'PE:PE12', 'PE:PE13', 'PE:PE23', 'PEEQ',
            'S:S11', 'S:S22', 'S:S33', 'S:S12', 'S:S13', 'S:S23']
    data = np.column_stack([df.X, df.PE11, df.PE22, df.PE33, df.PE12, zero, zero, df.PEEQ,
                            df.S11, df.S22, df.S33, df.S12, zero, zero])
    f = _write_report(tmp_path / 'solid.rpt', cols, data)
    solid = read_abaqus_report(f, None)
    assert solid.attrs['element_type'] == 'solid' and {'S_Max', 'S_Mid', 'S_Min', 'PE_Max'} <= set(solid.columns)
    for a, b in zip(path_from_abaqus_report(f), path_from_abaqus_report(SHELL)):
        np.testing.assert_allclose(a, b, atol=1e-5)


def test_path_of_an_element_that_has_not_yielded_yet(tmp_path):
    """Abaqus writes NoValue for the plastic strain of an element before it yields: it counts as zero."""
    names = read_abaqus_report(SHELL, None).columns.tolist()
    idx = [names.index(c) for c in ('PE11', 'PE22', 'PE12', 'PEEQ')]

    def transform(tok, n):
        second = list(tok)
        for i in idx:
            second[i] = 'NoValue' if n < 10 else f'{float(tok[i]) * 0.5:.6g}'
        return [tok, second]

    f = _rewrite_rows(tmp_path, transform, 'late_yield.rpt')
    second = path_from_abaqus_report(f, element=2)
    assert all(p == StrainIncrement(0.0, 0.0, 0.0, 0.0) for p in second[:9])      # zeros while NoValue
    assert sum(p.depsEQ for p in second) == pytest.approx(0.5 * sum(p.depsEQ for p in path_from_abaqus_report(SHELL)),
                                                          abs=1e-5)


def test_unstable_element_order_is_reported(tmp_path):
    """If the rows of an instant swap places the PEEQ of a row goes down and up: warning."""
    names = read_abaqus_report(SHELL, None).columns.tolist()
    ipeeq = names.index('PEEQ')

    def transform(tok, n):
        other = list(tok)
        other[ipeeq] = f'{float(tok[ipeeq]) * 0.5:.6g}'
        return [tok, other] if n % 2 == 0 else [other, tok]

    f = _rewrite_rows(tmp_path, transform, 'swapped.rpt')
    with pytest.warns(RuntimeWarning, match='stable element order'):
        path_from_abaqus_report(f, element=1)


def test_element_path_with_cut_and_downsampling(tmp_path):
    f = _two_element_shell(tmp_path)
    df = read_abaqus_report(SHELL, None)
    t_cut = float(df.X.iloc[40])
    cut = path_from_abaqus_report(f, element=2, until_time=t_cut)
    assert len(cut) == 40
    short = downsample_path(cut, 8)
    assert len(short) == 8
    for k in range(4):
        assert sum(s[k] for s in short) == pytest.approx(sum(c[k] for c in cut))
    assert path_from_abaqus_report(f, element=1, until_time=-1.0) == []


def test_missing_components_are_reported_per_element(tmp_path):
    f = _two_element_shell(tmp_path)
    broken = f.read_text().replace('PE:PE12', 'PE:PE13')
    g = tmp_path / 'no_pe12.rpt'
    g.write_text(broken)
    with pytest.raises(ValueError, match='PE12'):
        path_from_abaqus_report(g, element=1)
    with pytest.raises(ValueError, match='PE12'):
        path_from_abaqus_report(g, repeated_times='average')
