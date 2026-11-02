"""M1 reaction structure, restart cost and open-system balance diagnostics."""

import time

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .benchmark import ROOT, write_json
from .sbml import SBMLModel


def analyze_groups(model, t, y, scale, macro_steps):
    """Complete net reactions are indivisible groups; complex-step flux slopes."""
    n = len(y)
    started = time.perf_counter()
    gradient = np.column_stack([model.flux(t, y.astype(complex) + 1j * 1e-24 * np.eye(n)[i]).imag / 1e-24
                                for i in range(n)])
    scaled_columns = model.S / model.volumes[:, None] / scale[:, None]
    kappas = np.max(abs(scaled_columns), axis=0) * np.sum(abs(gradient) * scale[None, :], axis=1)
    results = []
    for h in macro_steps:
        fast = np.flatnonzero(h * kappas > 1)
        slow = np.setdiff1d(np.arange(len(model.reactions)), fast)
        touched = np.flatnonzero(np.any(model.S[:, fast] != 0, axis=1))
        depends = np.flatnonzero(np.any(model.dependencies[fast], axis=0)) if len(fast) else np.array([], dtype=int)
        connectivity = (model.S[:, fast] != 0).astype(int)
        adjacency = connectivity @ connectivity.T
        blocks = []
        if len(touched):
            count, labels = connected_components(csr_matrix(adjacency[np.ix_(touched, touched)]))
            blocks = [list(map(int, touched[labels == i])) for i in range(count)]
        jf = (model.S[:, fast] / model.volumes[:, None]) @ gradient[fast]
        js = (model.S[:, slow] / model.volumes[:, None]) @ gradient[slow]
        evf, evs = np.linalg.eigvals(jf), np.linalg.eigvals(js)
        fastest_total = float(np.max(abs(np.linalg.eigvals(jf + js))))
        fastest_slow = float(np.max(abs(evs)))
        results.append({"h": h, "fast_reactions": [model.reactions[i] for i in fast],
                        "updated_state_count": len(touched), "rate_dependency_state_count": len(depends),
                        "connected_block_states": [[model.species[i] for i in block] for block in blocks],
                        "fast_spectral_radius": float(np.max(abs(evf))), "slow_spectral_radius": fastest_slow,
                        "full_spectral_radius": fastest_total,
                        "spectral_radius_reduction": fastest_total / max(fastest_slow, 1e-30),
                        "fast_max_real_eigenvalue": float(evf.real.max()),
                        "h_times_slow_spectral_radius": h * fastest_slow,
                        "note": "Proxy screening and linearized local modes, not nonlinear stability or global error guarantees"})
    return {"time": t, "kappa": dict(zip(model.reactions, map(float, kappas))),
            "classifications": results, "gradient_and_classification_seconds": time.perf_counter() - started}


def main():
    out = ROOT / "results/m1-completion"
    out.mkdir(parents=True, exist_ok=True)
    model = SBMLModel(ROOT / "models/BIOMD0000000064.xml")
    reference = np.load(ROOT / "results/m1/BIOMD0000000064-reference.npz")
    times, states, scale = reference['times'], reference['concentrations'], reference['scale']
    snapshots = []
    for i in (0, np.searchsorted(times, 1e-4), np.searchsorted(times, .01), len(times) - 1):
        snapshots.append(analyze_groups(model, float(times[i]), states[:, i], scale, (1e-5, 1e-4, 1e-3, .01)))
    y, t = states[:, -1], times[-1]
    rhs_samples, restart_samples = [], []
    for _ in range(7):
        start = time.perf_counter()
        for j in range(1000):
            model.rhs(t, y)
        rhs_samples.append((time.perf_counter() - start) / 1000)
        start = time.perf_counter()
        solve_ivp(model.rhs, (t, t + 1e-5), y, method="Radau", rtol=1e-8,
                  atol=1e-10 * scale, jac_sparsity=model.sparsity)
        restart_samples.append(time.perf_counter() - start)
    structural = {"model": "BIOMD0000000064", "snapshots": snapshots,
                  "rhs_seconds_samples": rhs_samples, "full_Radau_restart_seconds_samples": restart_samples,
                  "decision": "Allow bounded M2 correctness experiments; local fast blocks exist only at small h. Do not infer net speedup; global blocks appear at h>=1e-3 min."}
    write_json(out / "reaction-structure.json", structural)

    # Formal carbon-weighted concentration ledger, not a total physical mass
    # claim: the encoded PTS extracellular stoichiometry includes biomass scaling.
    model = SBMLModel(ROOT / "models/BIOMD0000000051.xml")
    carbon = np.array([3, 6, 6, 3, 6, 6, 6, 6, 7, 3, 4, 5, 5, 3, 3, 3, 3, 5], dtype=float)
    column_balance = carbon @ (model.S / model.volumes[:, None])
    open_columns = np.flatnonzero(abs(column_balance) > 1e-10)
    closed_columns = np.flatnonzero(abs(column_balance) <= 1e-10)
    solution = solve_ivp(model.rhs, (0, 40), model.y0, method="Radau", rtol=1e-11,
                         atol=1e-13 * np.maximum(model.y0, 1e-3), dense_output=True,
                         jac_sparsity=model.sparsity)
    if not solution.success:
        raise RuntimeError(solution.message)
    rows = []
    for endpoint in (1, 10, 20, 40):
        integral, quadrature_error = quad(lambda t: float(column_balance @ model.flux(t, solution.sol(t))),
                                          0, endpoint, epsabs=1e-10, epsrel=1e-10, limit=500)
        delta = float(carbon @ (solution.sol(endpoint) - model.y0))
        residual = abs(delta - integral) / max(1, abs(float(carbon @ model.y0)))
        rows.append({"time": endpoint, "ledger_change": delta, "integrated_open_columns": integral,
                     "relative_residual": residual, "quad_error_estimate": quadrature_error})
    balance = {"weights": dict(zip(model.species, carbon.tolist())),
               "open_columns": [model.reactions[i] for i in open_columns],
               "closed_columns": [model.reactions[i] for i in closed_columns],
               "meaning": "Carbon-weighted concentration ledger for the encoded model; PTS biomass scaling is treated as an open contribution. Not a whole-culture physical mass balance.",
               "samples": rows, "pass": max(r['relative_residual'] for r in rows) < 1e-9}
    write_json(out / "open-balance.json", balance)
    if not balance['pass']:
        raise RuntimeError("Open-system ledger failed")
    print("Structure and open balance checks saved", flush=True)


if __name__ == "__main__":
    main()
