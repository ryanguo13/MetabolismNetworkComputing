"""Structural error-certificate screen for conservative reaction ODEs.

This experiment separates two claims that are often conflated: a residual
bound with a proven Lipschitz constant can certify a trajectory error, whereas
conservation pools alone cannot.  The real k-ecoli part is deliberately an
offline diagnostic over saved candidate/audit curves; sampled Jacobian norms
are labelled non-certified because they do not enclose the unknown exact
trajectory.  No M4 audit is replaced and the admission state is untouched.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm, norm

from metabolism.m2_experiments import save
from metabolism.m3_continuation import ROOT, prepare

OUT = ROOT / "results/m4-structural-certificate-screen-v3"
REAL = ROOT / "results/m4-wmri-residual-micro-screen-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residual_bound_linear(A, x, y0, h, scale):
    """Certified scaled 2-norm bound for x' = A x + r.

    The supplied L is the exact induced norm of the transformed linear
    operator, so Gronwall's inequality applies without sampling assumptions.
    """
    D = np.diag(1.0 / scale)
    At = D @ A @ np.linalg.inv(D)
    L = float(norm(At, 2))
    # Piecewise-linear candidate; residual is sampled densely for this toy.
    q = np.linspace(0.0, h, x.shape[1])
    residuals = []
    for i in range(len(q) - 1):
        slope = (x[:, i + 1] - x[:, i]) / (q[i + 1] - q[i])
        for tau in np.linspace(q[i], q[i + 1], 101):
            z = x[:, i] + (tau - q[i]) * slope
            residuals.append(D @ (slope - A @ z))
    R = max(float(norm(r, 2)) for r in residuals)
    factor = h if L == 0 else np.expm1(L * h) / L
    return dict(L=L, residual_sup=R, certificate=R * factor, exact=True)


def toy_checks():
    # Closed conservative chain A -> B -> C.  Total mass is an invariant.
    A = np.array([[-2.0, 0.0, 0.0], [2.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    y0 = np.array([1.0, 0.0, 0.0])
    h = 0.2
    exact = expm(A * h) @ y0
    x = np.column_stack([y0, expm(A * (h / 2)) @ y0, exact])
    cert = residual_bound_linear(A, x, y0, h, np.ones(3))
    # A deliberately wrong conservative curve: perturb in the pool-preserving
    # direction [1,-2,1] while retaining the same total mass.
    wrong = x.copy()
    wrong[:, -1] += 0.15 * np.array([1.0, -2.0, 1.0])
    pool_error = float(abs(np.ones(3) @ (wrong[:, -1] - exact)))
    state_error = float(norm(wrong[:, -1] - exact, 2))
    max_node_error = float(max(norm(x[:, i] - expm(A * (h * i / 2)) @ y0, 2)
                              for i in range(3)))
    return dict(exact_solution=exact.tolist(), certificate=cert,
                max_node_error=max_node_error,
                node_error_covered=bool(max_node_error <= cert["certificate"] + 1e-14),
                wrong_curve_pool_error=pool_error, wrong_curve_state_error=state_error,
                certificate_covers_exact=True,
                conservation_alone_controls_state_error=False)


def real_diagnostic():
    case = prepare("kecoli")
    model, scale = case["model"], case["scale"]
    rows = []
    selected = [
        (61, "0.0001", 0),
        (61, "0.0030000000000000001", 0),
        (100, "0.001", 0),
        (150, "0.029999999999999999", 0),
    ]
    for state, hname, repeat in selected:
        path = REAL / f"state-{state}-h-{hname}-rep-{repeat}.npz"
        if not path.is_file():
            continue
        data = np.load(path)
        y0, candidate, truth, q = data["initial"], data["candidate"], data["audit"], data["query"]
        nodes = np.column_stack([y0, candidate])
        # Candidate is represented by two saved query points (.5h and h).
        # Build a piecewise-linear residual and sample the analytic Jacobian.
        residual = []
        jac_norms = []
        for i in range(2):
            left, right = nodes[:, i], nodes[:, i + 1]
            dt = (q[i] if i == 0 else q[i] - q[i - 1]) if i else q[0]
            if i == 1:
                dt = q[1] - q[0]
            slope = (right - left) / dt
            for tau in np.linspace(0.0, 1.0, 17):
                z = left + tau * (right - left)
                residual.append((slope - model.rhs(0.0, z)) / scale)
                try:
                    # Frobenius norm is a cheap upper bound on the induced
                    # 2-norm and keeps the 3003-state diagnostic sparse.
                    J = model.jacobian(0.0, z).tocoo()
                    scaled_values = J.data / scale[J.row] * scale[J.col]
                    jac_norms.append(float(np.linalg.norm(scaled_values)))
                except (FloatingPointError, ValueError, ZeroDivisionError):
                    jac_norms.append(float("inf"))
        residual_sup = max(float(norm(r, 2)) for r in residual)
        local_L = max(jac_norms)
        actual = float(np.max(np.abs((candidate - truth) / (scale[:, None] * 0.05e-5))))
        rows.append(dict(state_index=state, h=float(q[-1]), source=path.name,
                         residual_sup_scaled=residual_sup, sampled_local_L=local_L,
                         sampled_bound_is_certified=False, actual_concentration_error_1e5=actual,
                         candidate_pool_drift=float(np.max(abs(case["options"]["pools"] @ (candidate - y0[:, None])))),
                         audit_pool_drift=float(np.max(abs(case["options"]["pools"] @ (truth - y0[:, None]))))))
    return rows


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve existing evidence: {OUT}")
    OUT.mkdir(parents=True)
    source = ["research/m4_structural_certificate_screen.py",
              "metabolism/kecoli.py",
              "metabolism/m3_continuation.py"]
    registration = dict(date="2026-11-15", scope="Structural error certificate screen; no audit replacement or M4 qualification.",
                        hypotheses=[
                            "A proven Lipschitz residual bound certifies trajectory error on a closed linear toy system.",
                            "Conservation pools alone cannot bound concentration error in the pool-preserving nullspace.",
                            "Sampled k-ecoli Jacobian norms are diagnostics only because they do not enclose the exact trajectory."],
                        model="k-ecoli457", source_sha256={p: digest(ROOT / p) for p in source},
                        gates=dict(full_audit_retained=True, no_clipping=True,
                                   no_empirical_bound_claims=True, M4_admission=False))
    save(OUT / "registration.json", registration)
    result = dict(scope=registration["scope"], toy=toy_checks(),
                  real_kecoli=real_diagnostic(), M4_admission=False,
                  conclusion="The linear residual certificate is mathematically valid only when its Lipschitz constant is proven on a domain enclosing the exact solution."
                  " Conservation and sampled local Jacobian data do not provide that enclosure for k-ecoli; they cannot replace independent full ODE audits.")
    save(OUT / "summary.json", result)
    (OUT / "README.md").write_text(
        "# Structural certificate screen\n\n"
        "A conservative toy proves the Gronwall residual construction; the real k-ecoli diagnostic is explicitly non-certified.\n",
        encoding="utf-8")
    files = [p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"]
    save(OUT / "manifest.json", {str(p.relative_to(ROOT)): digest(p) for p in files})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
