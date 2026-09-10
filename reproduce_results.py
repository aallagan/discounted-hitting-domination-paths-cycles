"""
Reproducibility code for
"Exact and Fault-Tolerant Discounted Hitting Domination on Paths and Cycles".

The script checks the principal finite-order formulas and enumeration results
of the paper on small instances against exhaustive search or direct counting:

  1. delta_{lambda,tau}(C_n)=ceil(n/L) and
     delta_{lambda,tau}(P_n)=1+ceil((n-1-2B)_+/L);
  2. minimum-placement counts nu(C_n), nu(P_n) against the
     generating-function formulas;
  3. fault-tolerant values delta^{(f)}(C_n), delta^{(f)}(P_n) against
     exhaustive search over source sets and failure sets;
  4. robust cycle counts nu^{(f)}(C_n)=(n/q)A, including an independent
     check of A=[z^n] tr(T(z)^q) by transfer-state dynamic programming;
  5. the exact q-source capacity C_q^{(f)} on paths, on selected small
     feasible instances; and
  6. the complete-graph value delta^{(f)}(K_n)=q_0+f, including tau=1.

All parameters used in the verification grid are fractions.Fraction objects.
The path, cycle, and complete-graph equilibrium systems are solved in exact
rational arithmetic. The attenuation scales B and L are also computed exactly
from the rational transfer recurrences; no floating-point evaluation of the
hyperbolic closed forms is used.

Requirement:
    networkx
"""

from __future__ import annotations

import itertools
import sys
from fractions import Fraction

import networkx as nx


# ---------------------------------------------------------------------------
# Exact rational linear algebra and equilibrium
# ---------------------------------------------------------------------------

def as_fraction(x):
    """Convert an integer/Fraction/string to Fraction without float round-trip."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x, 1)
    if isinstance(x, str):
        return Fraction(x)
    raise TypeError(
        "Verification parameters must be int, str, or Fraction; "
        "do not pass binary floats to exact-arithmetic routines."
    )


def solve_linear_system_exact(A, b):
    """Solve A x = b over the rationals by Gauss-Jordan elimination."""
    n = len(A)
    if n == 0:
        return []

    M = [
        [as_fraction(A[i][j]) for j in range(n)] + [as_fraction(b[i])]
        for i in range(n)
    ]

    for col in range(n):
        pivot = next((r for r in range(col, n) if M[r][col] != 0), None)
        if pivot is None:
            raise ArithmeticError("Singular rational linear system.")

        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]

        piv = M[col][col]
        M[col] = [x / piv for x in M[col]]

        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor != 0:
                M[r] = [
                    M[r][j] - factor * M[col][j]
                    for j in range(n + 1)
                ]

    return [M[i][n] for i in range(n)]


def equilibrium_exact(G, nodes, lam, sources):
    """Return the pinned equilibrium h^S exactly as Fractions."""
    lam = as_fraction(lam)
    if not (0 < lam < 1):
        raise ValueError("lambda must satisfy 0 < lambda < 1.")

    n = len(nodes)
    source_set = set(sources)

    if any(i < 0 or i >= n for i in source_set):
        raise ValueError("Source indices must refer to positions in nodes.")

    if not source_set:
        return [Fraction(0) for _ in range(n)]

    if len(source_set) == n:
        return [Fraction(1) for _ in range(n)]

    index = {u: i for i, u in enumerate(nodes)}
    U = [i for i in range(n) if i not in source_set]
    pos = {u_idx: r for r, u_idx in enumerate(U)}

    A = [[Fraction(0) for _ in U] for _ in U]
    b = [Fraction(0) for _ in U]

    for row, i in enumerate(U):
        u = nodes[i]
        deg = G.degree(u)
        if deg <= 0:
            raise ValueError("The graph must have no isolated vertices.")

        A[row][row] = Fraction(1)
        step = lam / deg

        for v in G.neighbors(u):
            j = index[v]
            if j in source_set:
                b[row] += step
            else:
                A[row][pos[j]] -= step

    x = solve_linear_system_exact(A, b)
    h = [Fraction(0) for _ in range(n)]

    for i in source_set:
        h[i] = Fraction(1)
    for row, i in enumerate(U):
        h[i] = x[row]

    return h


def worst_support_exact(G, nodes, lam, sources):
    """Return min_i h_i^S exactly."""
    if not sources:
        return Fraction(0)
    return min(equilibrium_exact(G, nodes, lam, sources))


# ---------------------------------------------------------------------------
# Exact attenuation scales from transfer recurrences
# ---------------------------------------------------------------------------

def transfer_sequences_exact(kmax, lam):
    """Return p_k and q_k for 0 <= k <= kmax exactly."""
    lam = as_fraction(lam)
    if not (0 < lam < 1):
        raise ValueError("lambda must satisfy 0 < lambda < 1.")

    if kmax < 1:
        return [Fraction(0)], [Fraction(1)]

    factor = Fraction(2, 1) / lam
    p = [Fraction(0), Fraction(1)]
    q = [Fraction(1), Fraction(1, 1) / lam]

    while len(p) <= kmax:
        p.append(factor * p[-1] - p[-2])
        q.append(factor * q[-1] - q[-2])

    return p, q


def scales_LB_exact(lam, tau, cap=1000):
    """Compute the exact integer scales L and B from rational recurrences.

    The monotonicity proved in the paper implies that the first inadmissible
    tail and gap determine B and L. If cap is reached before that happens,
    an error is raised rather than silently returning a truncated scale.
    """
    lam = as_fraction(lam)
    tau = as_fraction(tau)

    if not (0 < lam < 1):
        raise ValueError("lambda must satisfy 0 < lambda < 1.")
    if not (0 < tau <= 1):
        raise ValueError("tau must satisfy 0 < tau <= 1.")

    p, q = transfer_sequences_exact(cap + 1, lam)

    B = 0
    while B + 1 <= cap and Fraction(1, 1) / q[B + 1] >= tau:
        B += 1
    if B == cap and Fraction(1, 1) / q[cap] >= tau:
        raise RuntimeError(f"Tail scale B reached cap={cap}; increase cap.")

    def psi(m):
        if m == 1:
            return Fraction(1)
        return min((p[m - j] + p[j]) / p[m] for j in range(1, m))

    L = 1
    while L + 1 <= cap and psi(L + 1) >= tau:
        L += 1
    if L == cap and psi(cap) >= tau:
        raise RuntimeError(f"Gap scale L reached cap={cap}; increase cap.")

    return L, B


# ---------------------------------------------------------------------------
# Exhaustive search over source sets and failure sets
# ---------------------------------------------------------------------------

def is_dominating(G, nodes, lam, tau, S):
    """Check the discounted hitting floor exactly."""
    tau = as_fraction(tau)
    return bool(S) and worst_support_exact(G, nodes, lam, S) >= tau


def is_fault_tolerant(G, nodes, lam, tau, S, f):
    """Check all failures F subseteq S with |F| <= f exactly."""
    S = tuple(S)

    if f < 0:
        raise ValueError("f must be nonnegative.")
    if len(S) < f + 1:
        return False

    for r in range(f + 1):
        for F in itertools.combinations(S, r):
            failure_set = set(F)
            surviving = [v for v in S if v not in failure_set]
            if not is_dominating(G, nodes, lam, tau, surviving):
                return False

    return True


def delta_brute(n, lam, tau, path=True):
    """Exact brute-force delta for P_n or C_n."""
    G = nx.path_graph(n) if path else nx.cycle_graph(n)
    nodes = list(G.nodes())

    for k in range(1, n + 1):
        for S in itertools.combinations(range(n), k):
            if is_dominating(G, nodes, lam, tau, S):
                return k

    return None


def nu_brute(n, lam, tau, path=True):
    """Return (minimum size, number of minimum placements) exactly."""
    G = nx.path_graph(n) if path else nx.cycle_graph(n)
    nodes = list(G.nodes())

    for k in range(1, n + 1):
        count = 0
        for S in itertools.combinations(range(n), k):
            if is_dominating(G, nodes, lam, tau, S):
                count += 1
        if count:
            return k, count

    return None, 0


def delta_fault_brute(n, lam, tau, f, path=True):
    """Exact brute-force fault-tolerant delta for P_n or C_n."""
    G = nx.path_graph(n) if path else nx.cycle_graph(n)
    nodes = list(G.nodes())

    for k in range(1, n + 1):
        for S in itertools.combinations(range(n), k):
            if is_fault_tolerant(G, nodes, lam, tau, S, f):
                return k

    return None


def nu_fault_brute(n, lam, tau, f, path=False):
    """Return (minimum robust size, number of minimum robust placements)."""
    G = nx.path_graph(n) if path else nx.cycle_graph(n)
    nodes = list(G.nodes())

    for k in range(1, n + 1):
        count = 0
        for S in itertools.combinations(range(n), k):
            if is_fault_tolerant(G, nodes, lam, tau, S, f):
                count += 1
        if count:
            return k, count

    return None, 0


# ---------------------------------------------------------------------------
# Closed-form formulas
# ---------------------------------------------------------------------------

def delta_cycle(n, lam, tau):
    L, _ = scales_LB_exact(lam, tau)
    return -(-n // L)


def delta_path(n, lam, tau):
    L, B = scales_LB_exact(lam, tau)
    return 1 + -(-max(0, n - 1 - 2 * B) // L)


def poly_mul(a, b):
    c = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            c[i + j] += ai * bj
    return c


def poly_pow(a, k):
    r = [1]
    for _ in range(k):
        r = poly_mul(r, a)
    return r


def coeff(poly, n):
    return poly[n] if 0 <= n < len(poly) else 0


def nu_cycle(n, lam, tau):
    L, _ = scales_LB_exact(lam, tau)
    q = -(-n // L)
    c = coeff(poly_pow([0] + [1] * L, q), n)
    if (n * c) % q != 0:
        raise ArithmeticError("(n/q) times the cycle coefficient is not integral.")
    return (n * c) // q


def nu_path(n, lam, tau):
    L, B = scales_LB_exact(lam, tau)
    q = delta_path(n, lam, tau)
    end = [1] * (B + 1)
    gap = [0] + [1] * L
    poly = poly_mul(poly_mul(end, end), poly_pow(gap, q - 1))
    return coeff(poly, n - 1)


def delta_fault_cycle(n, lam, tau, f):
    L, _ = scales_LB_exact(lam, tau)

    if f < 0 or f >= n:
        return None
    if L < f + 1:
        return None

    return max(f + 1, -(-(f + 1) * n // L))


def delta_fault_path(n, lam, tau, f):
    L, B = scales_LB_exact(lam, tau)

    if f < 0 or f >= n or B < f:
        return None

    if n - 1 <= 2 * B:
        return max(f + 1, n + 2 * f - 2 * B)

    w = f + 1
    H = L - w
    R = n - 1 - 2 * B
    alpha = (R - 1) // L
    rho = R - alpha * L

    return 2 * f + 1 + alpha * w + max(1, rho - H)


# ---------------------------------------------------------------------------
# Cycle enumeration: direct gap count and transfer-state count
# ---------------------------------------------------------------------------

def A_count_direct(n, q, L, f):
    """Count ordered cyclic gap q-tuples directly from the window condition."""
    if q <= 0 or f < 0 or L < f + 1:
        return 0

    count = 0

    def rec(i, seq, total):
        nonlocal count

        if i == q:
            if total != n:
                return
            if all(
                sum(seq[(j + t) % q] for t in range(f + 1)) <= L
                for j in range(q)
            ):
                count += 1
            return

        remaining = q - i - 1
        max_gap = min(L, n - total - remaining)

        for m in range(1, max_gap + 1):
            rec(i + 1, seq + [m], total + m)

    rec(0, [], 0)
    return count


def A_count_transfer(n, q, L, f):
    """Compute [z^n] tr(T(z)^q) by transfer-state dynamic programming."""
    if q <= 0 or f < 0 or L < f + 1:
        return 0

    if f == 0:
        return coeff(poly_pow([0] + [1] * L, q), n)

    states = list(itertools.product(range(1, L + 1), repeat=f))
    total_count = 0

    for start in states:
        dp = {(start, 0): 1}

        for _ in range(q):
            new_dp = {}

            for (state, total), multiplicity in dp.items():
                for g in range(1, L + 1):
                    if sum(state) + g > L:
                        continue

                    new_state = state[1:] + (g,)
                    new_total = total + g

                    if new_total > n:
                        continue

                    key = (new_state, new_total)
                    new_dp[key] = new_dp.get(key, 0) + multiplicity

            dp = new_dp

        total_count += dp.get((start, n), 0)

    return total_count


def nu_fault_cycle(n, lam, tau, f, use_transfer=True):
    """Robust cycle placement count from the theorem."""
    q = delta_fault_cycle(n, lam, tau, f)
    if q is None:
        return 0

    L, _ = scales_LB_exact(lam, tau)
    A = (
        A_count_transfer(n, q, L, f)
        if use_transfer
        else A_count_direct(n, q, L, f)
    )

    if (n * A) % q != 0:
        raise ArithmeticError("(n/q)A is not integral.")

    return (n * A) // q


# ---------------------------------------------------------------------------
# Exact q-source capacity on paths
# ---------------------------------------------------------------------------

def path_capacity_formula(q, f, L, B):
    """Return C_q^{(f)} from the paper, or None when infeasible."""
    if f < 0 or q < f + 1:
        return None
    if B < f:
        return None

    if q == f + 1:
        return 2 * B - f

    w = f + 1
    H = L - w
    p = q - 1
    m = p - 2 * f

    if m <= 0:
        return 2 * B + m

    return 2 * B + m + H * -(-m // w)


def path_has_ft_placement_exactly_q(n, q, lam, tau, f):
    """Whether P_n has an f-fault-tolerant placement of exactly q sources."""
    if q > n:
        return False

    G = nx.path_graph(n)
    nodes = list(G.nodes())

    return any(
        is_fault_tolerant(G, nodes, lam, tau, S, f)
        for S in itertools.combinations(range(n), q)
    )


def path_capacity_brute(q, f, lam, tau, search_limit=16):
    """Brute-force C_q^{(f)} on a bounded small-order range.

    Returns the largest n-1 found. Raises if feasibility persists at the
    search limit, since then the bounded scan has not certified the maximum.
    """
    best = None

    for n in range(q, search_limit + 1):
        feasible = path_has_ft_placement_exactly_q(n, q, lam, tau, f)
        if feasible:
            best = n - 1

    if best is not None and best == search_limit - 1:
        raise RuntimeError(
            "Capacity brute-force scan reached search_limit; increase it."
        )

    return best


# ---------------------------------------------------------------------------
# Complete graph: exact rational verification
# ---------------------------------------------------------------------------

def q0_complete(n, lam, tau):
    lam = as_fraction(lam)
    tau = as_fraction(tau)

    if tau == 1:
        return n

    val = tau * (n - 1) * (1 - lam) / (lam * (1 - tau))
    return min(n, -(-val.numerator // val.denominator))


def delta_fault_complete_formula(n, lam, tau, f):
    if f < 0 or f >= n:
        return None

    q0 = q0_complete(n, lam, tau)
    return q0 + f if q0 + f <= n else None


def complete_graph_worst_exact(n, lam, s):
    """Exact common non-source support for a size-s source set in K_n."""
    lam = as_fraction(lam)

    if s >= n:
        return Fraction(1)

    return lam * s / ((n - 1) - lam * (n - 1 - s))


def delta_fault_complete_brute(n, lam, tau, f):
    """Exact scan over source cardinalities using K_n symmetry."""
    tau = as_fraction(tau)

    if f < 0 or f >= n:
        return None

    for q in range(1, n + 1):
        if q < f + 1:
            continue
        if complete_graph_worst_exact(n, lam, q - f) >= tau:
            return q

    return None


# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------

def require(condition, message):
    """Raise AssertionError with a clear message if a check fails."""
    if not condition:
        raise AssertionError(message)


def print_pass(label):
    print(f"   PASS: {label}")


# ---------------------------------------------------------------------------
# Main verification suite
# ---------------------------------------------------------------------------

def main():
    print("Exact and fault-tolerant discounted hitting domination: checks")
    print("Python:", sys.version.split()[0])
    print("NetworkX:", nx.__version__)

    params = [
        (Fraction(4, 5), Fraction(1, 2)),
        (Fraction(17, 20), Fraction(11, 20)),
        (Fraction(9, 10), Fraction(3, 5)),
        (Fraction(3, 4), Fraction(2, 5)),
        (Fraction(1, 2), Fraction(1, 4)),
    ]

    print("\n[1] delta(C_n), delta(P_n) versus exact brute force")
    for lam, tau in params:
        for n in range(3, 10):
            require(
                delta_brute(n, lam, tau, path=False) == delta_cycle(n, lam, tau),
                f"Cycle delta mismatch: n={n}, lambda={lam}, tau={tau}",
            )
        for n in range(2, 10):
            require(
                delta_brute(n, lam, tau, path=True) == delta_path(n, lam, tau),
                f"Path delta mismatch: n={n}, lambda={lam}, tau={tau}",
            )
    print_pass("ordinary path/cycle values")

    print("\n[2] nu(C_n), nu(P_n) versus direct counting")
    for lam, tau in params:
        for n in range(3, 9):
            _, direct = nu_brute(n, lam, tau, path=False)
            require(
                direct == nu_cycle(n, lam, tau),
                f"Cycle enumeration mismatch: n={n}, lambda={lam}, tau={tau}",
            )
        for n in range(2, 10):
            _, direct = nu_brute(n, lam, tau, path=True)
            require(
                direct == nu_path(n, lam, tau),
                f"Path enumeration mismatch: n={n}, lambda={lam}, tau={tau}",
            )
    print_pass("ordinary path/cycle enumeration")

    print("\n[3] fault-tolerant values versus exact brute force")
    for lam, tau in params:
        for n in range(3, 9):
            for f in range(n):
                require(
                    delta_fault_brute(n, lam, tau, f, path=False)
                    == delta_fault_cycle(n, lam, tau, f),
                    f"Robust cycle mismatch: n={n}, f={f}, lambda={lam}, tau={tau}",
                )
        for n in range(2, 9):
            for f in range(n):
                require(
                    delta_fault_brute(n, lam, tau, f, path=True)
                    == delta_fault_path(n, lam, tau, f),
                    f"Robust path mismatch: n={n}, f={f}, lambda={lam}, tau={tau}",
                )
    print_pass("fault-tolerant path/cycle values, including infeasible cases")

    print("\n[4] robust cycle enumeration: sets, gaps, and transfer trace")
    robust_params = [
        (Fraction(4, 5), Fraction(1, 2)),
        (Fraction(9, 10), Fraction(3, 5)),
    ]

    for lam, tau in robust_params:
        for f in (0, 1):
            for n in range(4, 10):
                q = delta_fault_cycle(n, lam, tau, f)
                _, direct_sets = nu_fault_brute(n, lam, tau, f, path=False)

                if q is None:
                    require(
                        direct_sets == 0,
                        f"Infeasible robust cycle unexpectedly exists: n={n}, f={f}, lambda={lam}, tau={tau}",
                    )
                    continue

                L, _ = scales_LB_exact(lam, tau)
                A_direct = A_count_direct(n, q, L, f)
                A_transfer = A_count_transfer(n, q, L, f)

                require(
                    A_direct == A_transfer,
                    f"Transfer-matrix count mismatch: n={n}, q={q}, f={f}, L={L}",
                )

                formula_sets = (n * A_transfer) // q
                require(
                    direct_sets == formula_sets,
                    f"Robust cycle enumeration mismatch: n={n}, f={f}, lambda={lam}, tau={tau}",
                )

    print_pass("robust cycle enumeration and transfer-matrix coefficient")

    print("\n[5] exact q-source path capacity versus bounded brute force")
    capacity_cases = [
        (Fraction(4, 5), Fraction(1, 2), 0, 1),
        (Fraction(4, 5), Fraction(1, 2), 0, 2),
        (Fraction(4, 5), Fraction(1, 2), 1, 2),
        (Fraction(4, 5), Fraction(1, 2), 1, 3),
        (Fraction(9, 10), Fraction(3, 5), 0, 2),
    ]

    for lam, tau, f, q in capacity_cases:
        L, B = scales_LB_exact(lam, tau)
        formula = path_capacity_formula(q, f, L, B)
        brute = path_capacity_brute(q, f, lam, tau, search_limit=16)
        require(
            formula == brute,
            f"Path capacity mismatch: q={q}, f={f}, lambda={lam}, tau={tau}, formula={formula}, brute={brute}",
        )

    print_pass("selected exact q-source path capacities")

    print("\n[6] complete graph robust formula in exact rational arithmetic")
    complete_params = [
        (Fraction(3, 5), Fraction(1, 2)),
        (Fraction(4, 5), Fraction(2, 5)),
        (Fraction(1, 2), Fraction(3, 10)),
        (Fraction(7, 10), Fraction(1, 1)),
    ]

    for lam, tau in complete_params:
        for n in range(2, 9):
            for f in range(n):
                require(
                    delta_fault_complete_brute(n, lam, tau, f)
                    == delta_fault_complete_formula(n, lam, tau, f),
                    f"Complete-graph mismatch: n={n}, f={f}, lambda={lam}, tau={tau}",
                )

    print_pass("complete-graph robust formula")
    print("\nAll reproducibility checks passed.")


if __name__ == "__main__":
    main()
