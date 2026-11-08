# sheetforming

Plasticity models for sheet metal forming (Python package): plane-stress and 3D
yield criteria (von Mises, Hill 1948, Hill 1979, Barlat Yld91 and Yld2000-2d),
hardening laws, yield functions with isotropic/kinematic hardening, stress
triaxiality along a strain path and Abaqus report readers.

- `src/sheetforming/` — the package
- `tests/` — automated checks (`uv run pytest`); `tests/data/` holds the few small input files they need

Related projects (sibling folders): `sheetforming-lab` (laboratory tools for the characterisation
of sheet metal, private for the moment: tensile and Nakajima tests, anisotropy, results files), `sheetforming-notes`
(course notes built on this package) and `sheetforming-research` (research notebooks, private). They use this
package; it does not depend on any of them.

## Development

```bash
uv sync          # creates .venv with the package (editable) and pytest
uv run pytest
```

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).
