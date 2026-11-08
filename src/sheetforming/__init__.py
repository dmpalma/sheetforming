"""
sheetforming
============

Plasticity models for sheet metal forming: plane-stress yield criteria
(yield_criteria_plane_stress), general 3D yield criteria (yield_criteria_3d),
hardening laws, their fit and the comparison of fits (constitutive_laws), yield
functions with isotropic/kinematic hardening (yield_functions), the Material
facade (material), stress triaxiality along a strain path (triaxiality) and
Abaqus .rpt report readers (abaqus_utilities).

The laboratory side (tensile and Nakajima tests, anisotropy, result files,
specimen drawings) is in the separate package sheetforming-lab, which uses this
one.

Importing any submodule that plots also applies the shared figure style
(see plot_style).
"""
