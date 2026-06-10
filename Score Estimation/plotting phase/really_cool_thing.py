"""Analytical toy model data generator.

Provides a small, self-contained analytical model to generate synthetic
datasets for experiments. The model combines a sinusoid, a Gaussian bump,
and a linear trend so datasets have multiple predictable components.

Usage (example):
	python analytical.py --n 500 --noise 0.05 --out sample.csv --plot

Functions:
 - `toy_function(x, params)` : compute analytic y for given x and params
 - `generate_data(...)` : create noisy samples and return a pandas.DataFrame
 - `plot_data(...)` : plot the generated dataset

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Params:
	A: float = 1.0
	omega: float = 2.0
	phi: float = 0.0
	B: float = 1.0
	mu: float = 5.0
	sigma: float = 0.8
	slope: float = 0.1
	intercept: float = 0.0


def toy_function(x: np.ndarray, params: Optional[Params] = None) -> np.ndarray:
	"""Analytical toy function.

	y(x) = A * sin(omega * x + phi)
		   + B * exp(-0.5 * ((x - mu)/sigma)**2)
		   + slope * x + intercept

	"""
	if params is None:
		params = Params()

	s = params
	y_sin = s.A * np.sin(s.omega * x + s.phi)
	y_gauss = s.B * np.exp(-0.5 * ((x - s.mu) / s.sigma) ** 2)
	y_lin = s.slope * x + s.intercept
	return y_sin + y_gauss + y_lin


def generate_data(
	n: int = 1000,
	x_range: Tuple[float, float] = (0.0, 10.0),
	params: Optional[Params] = None,
	noise_std: float = 0.1,
	random_state: Optional[int] = None,
) -> pd.DataFrame:
	"""Generate synthetic dataset from the analytical toy model.

	Returns a DataFrame with columns: `x`, `y`, `y_true`, `noise`.
	"""
	rng = np.random.default_rng(random_state)
	x = rng.uniform(x_range[0], x_range[1], size=n)
	x = np.sort(x)
	y_true = toy_function(x, params=params)
	noise = rng.normal(loc=0.0, scale=noise_std, size=n)
	y = y_true + noise
	df = pd.DataFrame({"x": x, "y": y, "y_true": y_true, "noise": noise})
	return df


def plot_data(df: pd.DataFrame, out_path: Optional[str] = None, show: bool = False) -> None:
	"""Plot observed data and underlying analytic function."""
	fig, ax = plt.subplots(figsize=(8, 4.5))
	ax.scatter(df["x"], df["y"], s=10, alpha=0.6, label="observed")
	ax.plot(df["x"], df["y_true"], color="C1", lw=2, label="analytic (true)")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.legend()
	ax.set_title("Analytical Toy Model — Generated Data")
	fig.tight_layout()
	if out_path is not None:
		fig.savefig(out_path, dpi=150)
	if show:
		plt.show()
	plt.close(fig)


def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Analytical toy model data generator")
	p.add_argument("--n", type=int, default=500, help="number of samples")
	p.add_argument("--x-min", type=float, default=0.0)
	p.add_argument("--x-max", type=float, default=10.0)
	p.add_argument("--noise", type=float, default=0.1, help="std dev of additive Gaussian noise")
	p.add_argument("--seed", type=int, default=None)
	p.add_argument("--out", type=str, default="generated.csv", help="CSV output path")
	p.add_argument("--plot", action="store_true", help="save a plot named generated.png")
	p.add_argument("--show", action="store_true", help="show the plot interactively")
	return p.parse_args()


def main() -> None:
	args = _parse_args()
	params = Params()
	df = generate_data(
		n=args.n,
		x_range=(args.x_min, args.x_max),
		params=params,
		noise_std=args.noise,
		random_state=args.seed,
	)
	df.to_csv(args.out, index=False)
	if args.plot or args.show:
		plot_path = None
		if args.plot:
			plot_path = args.out.rsplit(".", 1)[0] + ".png"
		plot_data(df, out_path=plot_path, show=args.show)


if __name__ == "__main__":
	main()

