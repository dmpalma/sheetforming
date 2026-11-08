"""Constitutive law models for plasticity: hardening laws, their fit to tensile data and the comparison of fits."""

from __future__ import annotations

import csv
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import plot_style  # applies the shared figure style on import; also provides width_1column & co.


@dataclass
class TensileData:
    """Container for tensile test data loaded from a file."""

    stress: list[float] = field(default_factory=list)
    strain: list[float] = field(default_factory=list)

    @staticmethod
    def _detect_delimiter(sample: str) -> str:
        if "\t" in sample:
            return "\t"
        if ";" in sample:
            return ";"
        return ","

    @classmethod
    def _resolve_columns(cls, fieldnames: Sequence[str], stress_column: str, strain_column: str) -> tuple[str, str]:
        normalized = {name.strip().lower(): name for name in fieldnames}
        aliases_stress = {"stress", "sigma", "stress_cauchy", "sigma_true", "true_stress"}
        aliases_strain = {
            "strain",
            "epsilon",
            "epsilon_p",
            "epsilon_p_true",
            "plastic_strain",
            "plastic_strain_true",
            "strain_true",
        }

        stress_key = stress_column
        strain_key = strain_column

        if stress_key.lower() not in normalized:
            for alias in aliases_stress:
                if alias in normalized:
                    stress_key = normalized[alias]
                    break

        if strain_key.lower() not in normalized:
            for alias in aliases_strain:
                if alias in normalized:
                    strain_key = normalized[alias]
                    break

        if stress_key.lower() not in normalized or strain_key.lower() not in normalized:
            available = ", ".join(fieldnames)
            raise ValueError(
                f"Could not resolve stress/strain columns. Expected '{stress_column}' and '{strain_column}'. "
                f"Available columns: {available}"
            )

        return stress_key, strain_key

    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
        stress_column: str = "stress",
        strain_column: str = "strain",
    ) -> "TensileData":
        """Load tensile data from a CSV or TXT file with two columns.

        Supported formats include comma-separated CSV files and tab-separated TXT files
        such as 'sigma\tepsilon_P'.
        """
        file_path = Path(file_path)

        with file_path.open("r", newline="") as file_obj:
            sample = file_obj.read(4096)
            file_obj.seek(0)
            delimiter = cls._detect_delimiter(sample)

            reader = csv.DictReader(file_obj, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError(f"File '{file_path}' has no header row.")

            resolved_stress, resolved_strain = cls._resolve_columns(
                reader.fieldnames,
                stress_column,
                strain_column,
            )

            stress_values: list[float] = []
            strain_values: list[float] = []

            for row in reader:
                stress_values.append(float(row[resolved_stress]))
                strain_values.append(float(row[resolved_strain]))

        return cls(stress=stress_values, strain=strain_values)



@dataclass
class ConstitutiveLaw(ABC):
    """Generic constitutive model interface: stress-strain relation, fit to tensile data, curve generation and plotting."""

    @abstractmethod
    def stress_from_strain(self, plastic_strain: float) -> float:
        ...

    @abstractmethod
    def strain_from_stress(self, stress: float) -> float:
        ...

    @staticmethod
    def _filter_valid_tensile_data(data: TensileData) -> tuple[np.ndarray, np.ndarray]:
        stress = np.asarray(data.stress, dtype=float)
        eps = np.asarray(data.strain, dtype=float)

        mask = np.isfinite(stress) & np.isfinite(eps)
        stress = stress[mask]
        eps = eps[mask]

        if stress.size == 0:
            raise ValueError("No valid data available for fitting.")

        return stress, eps

    def generate_curve(
        self,
        max_strain: float = 2.0,
        n_points: int = 1000,
        step: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate the calibrated stress-strain curve up to a given plastic strain: n_points equally
        spaced values from 0 to max_strain or, if step is given, one value every step (0, step,
        2 step, ... max_strain included; n_points is then ignored). Returns (stress, strain).
        """
        end_strain = max(0.0, float(max_strain))
        if step is not None:
            if step <= 0:
                raise ValueError("step must be positive")
            strain = np.round(np.arange(int(round(end_strain / step)) + 1) * step, 12)
        else:
            strain = np.linspace(0.0, end_strain, int(max(2, n_points)))
        stress = np.asarray([self.stress_from_strain(value) for value in strain], dtype=float)
        return stress, strain

    def export_curve_csv(
        self,
        file_path: str | Path,
        max_strain: float = 2.0,
        n_points: int = 1000,
        include_header: bool = True,
        step: Optional[float] = None,
    ) -> Path:
        """
        Export the calibrated curve as a CSV file with the columns true_stress and plastic_strain
        (stress first, the order of the *PLASTIC data of Abaqus). See generate_curve for n_points and step.
        """
        stress, strain = self.generate_curve(max_strain=max_strain, n_points=n_points, step=step)
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if include_header:
                writer.writerow(["true_stress", "plastic_strain"])
            for sigma, eps in zip(stress, strain):
                writer.writerow([float(sigma), float(eps)])

        return path

    def plot_fit(
        self,
        data: TensileData,
        ax=None,
        n_points: int = 500,
        title: Optional[str] = None,
        figwidth: float = plot_style.width_1column,
    ):
        """Plot experimental data and the fitted constitutive model."""
        stress_exp, strain_exp = self._filter_valid_tensile_data(data)
        if ax is None:
            _, ax = plt.subplots(figsize=(figwidth, figwidth * 0.75))

        max_strain_exp = float(np.max(strain_exp)) if strain_exp.size else 0.0
        max_stress_exp = float(np.max(stress_exp)) if stress_exp.size else 0.0

        x_limit = max(0.0, round(1.5 * max_strain_exp, 1))
        y_limit = max(10.0, round(1.5 * max_stress_exp))

        strain_fit = np.linspace(0.0, x_limit, n_points)
        stress_fit = np.asarray([self.stress_from_strain(value) for value in strain_fit], dtype=float)

        ax.plot(strain_exp, stress_exp, "o", label="Experimental")
        ax.plot(strain_fit, stress_fit, "-", linewidth=2, label=f"{self.__class__.__name__} fit")
        ax.set_xlim(0.0, x_limit)
        ax.set_ylim(0.0, y_limit)
        ax.set_xlabel("Plastic strain")
        ax.set_ylabel("Stress")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(title or f"{self.__class__.__name__} fit")

        if isinstance(self, Hollomon):
            formula = (
                rf"$\sigma = {self.K:.4g}\,\varepsilon_{{p}}^{{{self.n:.4g}}}$"
            )
        elif isinstance(self, Swift):
            formula = (
                rf"$\sigma = {self.K:.4g}\,({self.eps_0:.4g} + \varepsilon_{{p}})^{{{self.n:.4g}}}$"
            )
        elif isinstance(self, Voce):
            Q = self.sigma_s - self.sigma_0
            A, B = self.equivalent_form()
            formula = (
                rf"$\sigma = {self.sigma_s:.4g} - ({self.sigma_s:.4g} - {self.sigma_0:.4g})"
                rf"\,e^{{-{self.C:.4g}\,\varepsilon_{{p}}}}$"
                "\n"
                rf"$\sigma = {self.sigma_s:.4g} - {Q:.4g}\,e^{{-{self.C:.4g}\,\varepsilon_{{p}}}}$"
                "\n"
                rf"$\sigma = {A:.4g} (1 - {B:.4g}\,e^{{-{self.C:.4g}\,\varepsilon_{{p}}}})$"
            )
        elif isinstance(self, VoceSwift):
            formula = (
                rf"$\sigma = {self.alpha:.4g}\,\sigma_V + {1-self.alpha:.4g}\,\sigma_S$"
            )
        else:
            formula = r"$\sigma = f(\varepsilon_p)$"

        ax.text(
            0.03,
            0.97,
            formula,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35"},
        )

        return ax

    @classmethod
    @abstractmethod
    def fit_from_data(cls, data: TensileData):
        ...


@dataclass
class Hollomon(ConstitutiveLaw):
    """Power-law hardening model: sigma = K * eps_p^n."""

    K: float = 100.0
    n: float = 0.2

    def stress_from_strain(self, plastic_strain: float) -> float:
        eps_p = max(plastic_strain, 0.0)
        return self.K * (eps_p ** self.n)

    def strain_from_stress(self, stress: float) -> float:
        # Inverse of sigma = K * eps_p^n: eps_p = (sigma/K)^(1/n).
        sigma = max(stress, 0.0)
        return (sigma / self.K) ** (1.0 / self.n)

    @classmethod
    def fit_from_data(cls, data: TensileData) -> "Hollomon":
        stress, eps = cls._filter_valid_tensile_data(data)
        mask = (stress > 0.0) & (eps > 0.0)
        if np.count_nonzero(mask) < 2:
            raise ValueError("At least two positive stress/plastic-strain points are required.")

        x = np.log(eps[mask])
        y = np.log(stress[mask])
        A = np.vstack([np.ones_like(x), x]).T
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
        log_k, n = coeffs
        return cls(K=float(np.exp(log_k)), n=float(n))


@dataclass
class Swift(ConstitutiveLaw):
    """Swift law: sigma = K * (eps_0 + eps_p)^n."""

    K: float = 100.0
    eps_0: float = 0.001
    n: float = 0.2

    def stress_from_strain(self, plastic_strain: float) -> float:
        eps_p = max(plastic_strain, 0.0)
        return self.K * (self.eps_0 + eps_p) ** self.n

    def strain_from_stress(self, stress: float) -> float:
        # Inverse of sigma = K * (eps_0 + eps_p)^n: eps_p = (sigma/K)^(1/n) - eps_0, clamped to >= 0.
        sigma = max(stress, 0.0)
        return max((sigma / self.K) ** (1.0 / self.n) - self.eps_0, 0.0)

    @classmethod
    def fit_from_data(cls, data: TensileData, eps0_grid: Optional[Sequence[float]] = None) -> "Swift":
        stress, eps = cls._filter_valid_tensile_data(data)
        mask = (stress > 0.0) & (eps >= 0.0)
        stress = stress[mask]
        eps = eps[mask]

        if stress.size < 3:
            raise ValueError("At least three valid points are required for Swift fitting.")

        if eps0_grid is None:
            eps0_grid = np.geomspace(np.finfo(float).eps, max(eps) * 0.5 if max(eps) > 0 else 1e-6, 200)

        best = None
        for eps_0 in np.asarray(eps0_grid, dtype=float):
            positive_mask = (eps + eps_0) > 0.0
            if np.count_nonzero(positive_mask) < 2:
                continue

            x = np.log(eps[positive_mask] + eps_0)
            y = np.log(stress[positive_mask])
            A = np.vstack([np.ones_like(x), x]).T
            coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
            log_k, n = coeffs
            K = float(np.exp(log_k))
            pred = K * (eps_0 + eps) ** n
            sse = float(np.sum((stress - pred) ** 2))
            if best is None or sse < best[0]:
                best = (sse, K, float(eps_0), float(n))

        if best is None:
            raise ValueError("Swift fitting failed.")

        _sse, K, eps_0, n = best
        return cls(K=K, eps_0=eps_0, n=n)


@dataclass
class Voce(ConstitutiveLaw):
    """
    Voce law: sigma = sigma_s - (sigma_s - sigma_0) * exp(-C * eps_p) --
    the parametrization most commonly seen in the literature, directly
    exposing the initial yield stress sigma_0 and the saturation stress
    sigma_s. Algebraically identical to sigma = A * (1 - B * exp(-C *
    eps_p)) (A = sigma_s, B = 1 - sigma_0/sigma_s) -- see
    equivalent_form()/print_formulas() to view both at once.
    """

    sigma_0: float = 100.0
    sigma_s: float = 200.0
    C: float = 10.0

    def stress_from_strain(self, plastic_strain: float) -> float:
        eps_p = max(plastic_strain, 0.0)
        return self.sigma_s - (self.sigma_s - self.sigma_0) * np.exp(-self.C * eps_p)

    def strain_from_stress(self, stress: float) -> float:
        # Inverse of sigma = sigma_s - (sigma_s - sigma_0)*exp(-C*eps_p); sigma is bounded above by sigma_s.
        sigma = min(max(stress, 0.0), self.sigma_s * (1.0 - 1e-12))
        ratio = (self.sigma_s - sigma) / (self.sigma_s - self.sigma_0)
        return max(-np.log(ratio) / self.C, 0.0)

    def equivalent_form(self) -> tuple[float, float]:
        """
        This law's parameters in the equivalent (A, B, C) form: sigma =
        A * (1 - B * exp(-C * eps_p)), with A = sigma_s, B = 1 -
        sigma_0/sigma_s (C is shared, unchanged) -- see the class
        docstring for the derivation. Returns (A, B).
        """
        A = self.sigma_s
        B = 1.0 - self.sigma_0 / self.sigma_s
        return A, B

    def print_formulas(self) -> None:
        """Prints this law's formula in both its equivalent parametrizations."""
        A, B = self.equivalent_form()
        Q = self.sigma_s - self.sigma_0
        print(f"Voce (sigma_0, sigma_s, C): sigma = {self.sigma_s:.4g} - ({self.sigma_s:.4g} - {self.sigma_0:.4g}) "
              f"* exp(-{self.C:.4g} * eps_p)")
        print(f"                                  = {self.sigma_s:.4g} - {Q:.4g} "
              f"* exp(-{self.C:.4g} * eps_p)")
        print(f"Voce (A, B, C):             sigma = {A:.4g} * (1 - {B:.4g} * exp(-{self.C:.4g} * eps_p))")

    @classmethod
    def fit_from_data(cls, data: TensileData, sigma_s_grid: Optional[Sequence[float]] = None) -> "Voce":
        """
        For each trial saturation stress sigma_s, log(sigma_s - sigma) is linear in eps_p
        (slope -C, intercept log(sigma_s - sigma_0)); the sigma_s with the lowest SSE in
        stress is kept. Default trial values: from just above the maximum stress of the
        data (data close to saturation need sigma_s within 1 % of it) to twice it,
        geometrically spaced near the maximum, then refined around the best one.
        """
        stress, eps = cls._filter_valid_tensile_data(data)
        if stress.size < 3:
            raise ValueError("At least three valid points are required for Voce fitting.")

        def search(grid):
            best = None
            for sigma_s in np.asarray(grid, dtype=float):
                y = sigma_s - stress
                mask = y > 0.0
                if np.count_nonzero(mask) < 2:
                    continue
                x = eps[mask]
                A = np.vstack([np.ones_like(x), x]).T
                (intercept, slope), *_ = np.linalg.lstsq(A, np.log(y[mask]), rcond=None)
                C = float(-slope)
                sigma_0 = float(sigma_s - np.exp(intercept))
                pred = sigma_s - (sigma_s - sigma_0) * np.exp(-C * eps)
                sse = float(np.sum((stress - pred) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, sigma_0, float(sigma_s), C)
            return best

        if sigma_s_grid is not None:
            best = search(sigma_s_grid)
        else:
            grid = stress.max() * (1.0 + np.geomspace(1e-5, 1.0, 200))
            best = search(grid)
            if best is not None:      # refine between the neighbours of the best trial value
                k = int(np.argmin(np.abs(grid - best[2])))
                best = min(best, search(np.linspace(grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)], 201)))

        if best is None:
            raise ValueError("Voce fitting failed.")

        _sse, sigma_0, sigma_s, C = best
        return cls(sigma_0=sigma_0, sigma_s=sigma_s, C=C)


@dataclass
class VoceSwift(ConstitutiveLaw):
    """Mixed Voce + Swift model with alpha controlling the Voce contribution."""

    K: float = 100.0
    eps_0: float = 0.001
    n: float = 0.2
    sigma_0: float = 100.0
    sigma_s: float = 200.0
    C: float = 10.0
    alpha: float = 0.5

    def __post_init__(self):
        self.alpha = float(np.clip(self.alpha, 0.0, 1.0))

    def stress_from_strain(self, plastic_strain: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        eps_p = np.asarray(plastic_strain, dtype=float)
        eps_p = np.maximum(eps_p, 0.0)
        swift_term = self.K * (self.eps_0 + eps_p) ** self.n
        voce_term = self.sigma_s - (self.sigma_s - self.sigma_0) * np.exp(-self.C * eps_p)
        result = (1.0 - self.alpha) * swift_term + self.alpha * voce_term
        if np.isscalar(plastic_strain):
            return float(result)
        return result

    def strain_from_stress(self, stress: float, tol: float = 1e-9, max_iter: int = 200) -> float:
        """Invert stress_from_strain numerically (no closed form for the mixed law)."""
        # The Swift+Voce mix has no algebraic inverse, so solve stress_from_strain(eps_p) = stress
        # by bisection, relying on stress_from_strain being monotonically increasing in eps_p.
        target = max(float(stress), 0.0)
        if target <= self.stress_from_strain(0.0):
            return 0.0

        # Expand the upper bound until it brackets the target stress.
        lo, hi = 0.0, 1.0
        while self.stress_from_strain(hi) < target:
            hi *= 2.0
            if hi > 1e6:
                raise ValueError("Could not bracket plastic strain for the requested stress.")

        # Standard bisection: halve the [lo, hi] interval until it is tol-small.
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if self.stress_from_strain(mid) < target:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break

        return 0.5 * (lo + hi)

    @classmethod
    def fit_from_data(cls, data: TensileData, alpha_grid: Optional[Sequence[float]] = None,
                      joint: bool = False) -> "VoceSwift":
        """
        Default (joint=False): Swift and Voce are fitted SEPARATELY to the
        whole curve, then only alpha is searched. Each component is then a
        compromise over the whole range, so the mix cannot do better than
        either when the curve needs both behaviours -- fast early hardening
        (Voce-like) AND sustained hardening at large strains (Swift-like),
        e.g. a tensile curve extrapolated with compression data. Use
        joint=True there (see _fit_joint).
        """
        if joint:
            return cls._fit_joint(data)

        stress, eps = cls._filter_valid_tensile_data(data)
        if stress.size < 3:
            raise ValueError("At least three valid points are required for mixed Voce-Swift fitting.")

        swift = Swift.fit_from_data(data)
        voce = Voce.fit_from_data(data)

        if alpha_grid is None:
            alpha_grid = np.linspace(0.0, 1.0, 101)

        best = None
        for alpha in np.asarray(alpha_grid, dtype=float):
            alpha = float(np.clip(alpha, 0.0, 1.0))
            swift_term = swift.K * (swift.eps_0 + eps) ** swift.n
            voce_term = voce.sigma_s - (voce.sigma_s - voce.sigma_0) * np.exp(-voce.C * eps)
            mixed = (1.0 - alpha) * swift_term + alpha * voce_term
            sse = float(np.sum((stress - mixed) ** 2))
            if best is None or sse < best[0]:
                best = (sse, float(alpha))
#            print(f"alpha={alpha:.4gf}, sse={sse:.4g}, best_sse={best[0]:.4g}, best_alpha={best[1]:.4gf}")

        if best is None:
            raise ValueError("Voce-Swift fitting failed.")

        alpha = best[1]
        return cls(
            K=swift.K,
            eps_0=swift.eps_0,
            n=swift.n,
            sigma_0=voce.sigma_0,
            sigma_s=voce.sigma_s,
            C=voce.C,
            alpha=alpha,
        )

    @classmethod
    def _fit_joint(cls, data: TensileData,
                   eps0_grid: Optional[Sequence[float]] = None,
                   n_grid: Optional[Sequence[float]] = None,
                   C_grid: Optional[Sequence[float]] = None) -> "VoceSwift":
        """
        Joint least-squares fit of all the parameters. The law is

            sigma = a*(eps_0 + eps_p)^n + b - c*exp(-C*eps_p),
            a = (1-alpha)*K,  b = alpha*sigma_s,  c = alpha*(sigma_s - sigma_0),

        LINEAR in (a, b, c) for fixed (eps_0, n, C): those are searched on a
        grid, solving (a, b, c) by least squares at each node, keeping only
        nodes where both terms are proper hardening curves (a > 0, and
        sigma_s > sigma_0 > 0, i.e. b > c > 0).

        alpha is not identifiable on its own (only a, b, c are): it is fixed
        by making both terms start at the same yield stress,
        K*eps_0^n = sigma_0 = sigma(eps_p = 0).
        """
        stress, eps = cls._filter_valid_tensile_data(data)
        mask = eps >= 0.0
        stress, eps = stress[mask], eps[mask]
        if stress.size < 6:
            raise ValueError("At least six valid points are required for the joint Voce-Swift fit.")

        eps0_grid = np.geomspace(1e-5, 0.05, 25) if eps0_grid is None else np.asarray(eps0_grid, float)
        n_grid = np.linspace(0.02, 0.8, 79) if n_grid is None else np.asarray(n_grid, float)
        C_grid = np.geomspace(0.5, 300.0, 80) if C_grid is None else np.asarray(C_grid, float)

        voce_basis = -np.exp(-np.outer(C_grid, eps))                     # (nC, N)
        best = None
        for eps_0 in eps0_grid:
            for n in n_grid:
                swift_basis = np.broadcast_to((eps_0 + eps) ** n, voce_basis.shape)
                A = np.stack([swift_basis, np.ones_like(voce_basis), voce_basis], axis=-1)   # (nC, N, 3)
                coef = np.linalg.solve(np.einsum("kni,knj->kij", A, A),
                                       np.einsum("kni,n->ki", A, stress)[..., None])[..., 0]
                sse = np.sum((np.einsum("kni,ki->kn", A, coef) - stress) ** 2, axis=1)
                a, b, c = coef.T
                sse = np.where((a > 0.0) & (c > 0.0) & (b > c), sse, np.inf)
                k = int(np.argmin(sse))
                if np.isfinite(sse[k]) and (best is None or sse[k] < best[0]):
                    best = (sse[k], eps_0, n, C_grid[k], *coef[k])

        if best is None:
            raise ValueError("Joint Voce-Swift fitting failed (no admissible solution on the grid).")

        _sse, eps_0, n, C, a, b, c = best
        alpha = (b - c) / ((b - c) + a * eps_0 ** n)
        return cls(K=float(a / (1.0 - alpha)), eps_0=float(eps_0), n=float(n),
                   sigma_0=float((b - c) / alpha), sigma_s=float(b / alpha), C=float(C),
                   alpha=float(alpha))


def plot_constitutive_laws(
    data: TensileData,
    models: Sequence[ConstitutiveLaw],
    fit_file=None,
    comparison_file=None,
    figwidth: float = plot_style.width_2columns,
):
    """
    Create the individual model-fit plot and the combined law comparison plot.
    Each figure is saved to fit_file / comparison_file only if that path is given.
    """
    stress_exp, strain_exp = ConstitutiveLaw._filter_valid_tensile_data(data)
    max_strain_exp = float(np.max(strain_exp)) if strain_exp.size else 0.0
    max_stress_exp = float(np.max(stress_exp)) if stress_exp.size else 0.0

    fig, axes = plt.subplots(2, 2, figsize=(figwidth, figwidth * (10 / 12)))
    model_labels = [model.__class__.__name__ for model in models]
    for ax, model, label in zip(axes.flat, models, model_labels):
        model.plot_fit(data, ax=ax, title=label)
    fig.tight_layout()

    all_fig, all_ax = plt.subplots(figsize=(plot_style.width_1column, plot_style.width_1column * (5 / 6)))
    all_ax.plot(strain_exp, stress_exp, "o", label="Experimental", color="black")
    x_limit = max(0.0, round(1.5 * max_strain_exp, 1))
    y_limit = max(10.0, round(1.5 * max_stress_exp))
    for model, label in zip(models, model_labels):
        eps_fit = np.linspace(0.0, x_limit, 500)
        sigma_fit = np.asarray([model.stress_from_strain(value) for value in eps_fit], dtype=float)
        all_ax.plot(eps_fit, sigma_fit, label=label, linewidth=2)
    all_ax.set_xlim(0.0, x_limit)
    all_ax.set_ylim(0.0, y_limit)
    all_ax.set_xlabel("Plastic strain")
    all_ax.set_ylabel("Stress")
    all_ax.grid(True, alpha=0.3)
    all_ax.legend(title="Fitted constitutive laws")
#    all_ax.set_title("Comparison of fitted constitutive laws")
    all_fig.tight_layout()

    for figure, path in ((fig, fit_file), (all_fig, comparison_file)):
        if path is not None:
            figure.savefig(path)
            print(f"Plot saved to {path}")

    return fig, all_fig


# ----------------------------------------------------------------------------------------------
# Fit of several laws to the same data and comparison of the fits
# ----------------------------------------------------------------------------------------------
LAWS = {"Swift": Swift, "Voce": Voce, "VoceSwift": VoceSwift}
_PARAMETER_COLUMNS = ("K", "eps_0", "n", "sigma_0", "sigma_s", "C", "alpha")


def fit_laws(data: TensileData, names: Sequence[str] = ("Swift", "Voce", "VoceSwift")) -> Dict[str, ConstitutiveLaw]:
    """{name: law fitted to data} for the laws in names (keys of LAWS)."""
    unknown = [n for n in names if n not in LAWS]
    if unknown:
        raise ValueError(f"unknown laws {unknown}; choose among {list(LAWS)}")
    return {name: LAWS[name].fit_from_data(data) for name in names}


def compare_fits(data: TensileData, fits: Dict[str, ConstitutiveLaw], max_plastic_strain: float = 2.0,
                 step: float = 0.01) -> pd.DataFrame:
    """
    Comparison of fitted laws (fit_laws) against the data they were fitted to, to help to choose one.

    One row per law: its parameters (columns K, eps_0, n, sigma_0, sigma_s, C, alpha; empty if the law has no
    such parameter), n_parameters, n_points, rmse and max_error (MPa, on the data), aic and bic (from the sum of
    squares: n ln(RSS/n) + 2k and n ln(RSS/n) + k ln n; lower is better), stress_at_end (MPa, at
    max_plastic_strain), monotonic (the extrapolated curve, sampled every step, never decreases) and best.

    A law with more parameters always fits better, so the error alone would always choose VoceSwift: aic and bic
    charge the number of parameters. The law flagged 'best' has the lowest BIC among the laws whose
    extrapolation does not decrease. The extrapolation far beyond the measured range (a tensile test stops at the
    maximum force) is a choice of the law -- Swift keeps hardening, Voce saturates -- so compare stress_at_end
    before choosing; the choice is the user's.
    """
    stress = np.asarray(data.stress, dtype=float)
    strain = np.asarray(data.strain, dtype=float)
    keep = np.isfinite(stress) & np.isfinite(strain)
    stress, strain = stress[keep], strain[keep]
    n = len(stress)
    rows = []
    for name, law in fits.items():
        residual = np.array([law.stress_from_strain(e) for e in strain], dtype=float) - stress
        rss = max(float(np.dot(residual, residual)), 1e-12)
        k = len(fields(law))
        extrapolated = law.generate_curve(max_plastic_strain, step=step)[0]
        row = {"law": name}
        for column in _PARAMETER_COLUMNS:
            row[column] = float(getattr(law, column)) if hasattr(law, column) else np.nan
        row.update({
            "n_parameters": k, "n_points": n,
            "rmse": float(np.sqrt(rss / n)) if rss > 1e-12 else 0.0, "max_error": float(np.abs(residual).max()),
            "aic": n * np.log(rss / n) + 2 * k, "bic": n * np.log(rss / n) + k * np.log(n),
            "stress_at_end": float(extrapolated[-1]), "monotonic": bool(np.all(np.diff(extrapolated) >= -1e-9)),
        })
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table["monotonic"].any():
        warnings.warn("no law has a non-decreasing extrapolation: 'best' is the lowest BIC anyway", RuntimeWarning)
    candidates = table[table["monotonic"]] if table["monotonic"].any() else table
    table["best"] = False
    table.loc[candidates["bic"].idxmin(), "best"] = True
    return table


def best_law(table: pd.DataFrame) -> str:
    """Name of the law flagged 'best' in compare_fits."""
    return str(table.loc[table["best"], "law"].iloc[0])
