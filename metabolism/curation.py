"""Reproduce public MathSBML curation trajectories, with pixel-scale uncertainty."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.integrate import solve_ivp

from .benchmark import ROOT, write_json
from .sbml import SBMLModel

# Axis calibration in the 887x670 unresized original BioModels curation JPEG.
# x0/x40, y0 and pixels per concentration unit; fixed before evaluating residuals.
PANELS = {
    "cglcex": (32, 210, 123, 116 / 3),
    "cfdp": (325, 502, 123, 116 / 3),
    "cg1p": (609, 787, 123, 116 / 3),
    "cg6p": (29, 210, 269, 118 / 6),
    "cpep": (325, 502, 269, 118 / 6),
    "cpyr": (610, 787, 267, 118 / 6),
    "cf6p": (29, 215, 422, 100),
    "cgap": (325, 502, 422, 100),
    "cpg": (610, 787, 423, 100),
}


def main():
    path = ROOT / "models/evidence/BIOMD0000000051-curation.jpeg"
    image = np.asarray(Image.open(path).convert("RGB"), dtype=int)
    assert image.shape == (670, 887, 3)
    redness = np.minimum(image[:, :, 0] - image[:, :, 1], image[:, :, 0] - image[:, :, 2])
    model = SBMLModel(ROOT / "models/BIOMD0000000051.xml")
    times = np.linspace(0, 40, 1601)
    solution = solve_ivp(model.rhs, (0, 40), model.y0, method="Radau", rtol=1e-11,
                         atol=1e-13 * np.maximum(model.y0, 1e-3), dense_output=True,
                         jac_sparsity=model.sparsity)
    if not solution.success:
        raise RuntimeError(solution.message)
    rows = []
    for species, (x0, x40, y0, pixels_per_unit) in PANELS.items():
        for target_time in (2, 5, 10, 15, 20, 25, 30, 35, 39):
            x = int(round(x0 + (x40 - x0) * target_time / 40))
            column = redness[max(0, y0 - 120):y0, x]
            if column.max() < 20:
                raise RuntimeError(f"Missing red curve: {species} at x={x}")
            y = float(np.argmax(column) + max(0, y0 - 120))
            t = (x - x0) * 40 / (x40 - x0)
            digitized = (y0 - y) / pixels_per_unit
            calculated = float(solution.sol(t)[model.species.index(species)])
            residual_pixels = abs(calculated - digitized) * pixels_per_unit
            rows.append({"species": species, "time": t, "x_pixel": x, "y_pixel": y,
                         "digitized_mM": digitized, "computed_mM": calculated,
                         "residual_pixels": residual_pixels})
    report = {"source_url": "https://www.ebi.ac.uk/biomodels/model/download/BIOMD0000000051?filename=curation_image.jpeg",
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_description": "BioModels original curation: figures 4,5,6 reproduced using MathSBML; not digitized experimental measurements",
              "calibration": PANELS, "samples": rows,
              "tolerance_pixels": 5.0, "uncertainty": "Approximate raster axis/curve calibration; one x pixel ≈0.22 seconds. No claim of high-precision validation from a JPEG.",
              "max_residual_pixels": max(row["residual_pixels"] for row in rows),
              "pass": all(row["residual_pixels"] <= 5 for row in rows)}
    output = ROOT / "results/m1-completion"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "chassagnole-curation.json", report)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), constrained_layout=True)
    for ax, species in zip(axes.flat, PANELS):
        ax.plot(times, solution.sol(times)[model.species.index(species)], label="Radau / original SBML")
        points = [r for r in rows if r['species'] == species]
        ax.scatter([r['time'] for r in points], [r['digitized_mM'] for r in points], marker="x", color="red", label="MathSBML curation pixels")
        ax.set(title=species, xlabel="time (s)", ylabel="mM")
    axes[0, 0].legend(fontsize=7)
    fig.savefig(output / "chassagnole-curation.png", dpi=160)
    print(json.dumps({k: v for k, v in report.items() if k not in {"samples", "calibration"}}, indent=2))
    if not report['pass']:
        raise RuntimeError("Curation comparison failed")


if __name__ == "__main__":
    main()
