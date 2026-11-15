"""Discretised observation spaces with empirical-Bayes cell values (H-DS1..H-DS3, program 2026-11-12, agent C).

Reusable, deterministic pieces. Every partition is FITTED on training rows only (quantile edges, k-means centroids,
tree splits) and then ASSIGNS any row with those frozen parameters, so a test row never moves an edge.

Partitions (all expose fit(...) -> self, assign(X) -> int cell ids in [0, n_cells), n_cells, describe(cell)):
  Lattice        product of QuantileDim (training quantile edges) and CategoricalDim (fixed value groups)
  KMeansCodebook k-means++ / Lloyd on training-standardised columns, best inertia over n_init seeds
  HonestTree     CART regression tree grown on the SPLIT half (histogram splits on split-half quantile bins, depth <=
                 max_depth, min_leaf rows per child); cell = leaf. Leaf VALUES are estimated by the caller on the
                 ESTIMATION half (honest_halves), never on the rows that chose the splits (Athey and Imbens 2016).

Cell values (cell_effects / eb_shrink): for values v_i (for example log(RV_next / log-HAR forecast)) and cells c_i,
  mu = pooled mean (or a given centre), e_i = v_i - mu, m_c = mean_c(e), n_c = rows in c,
  V_c = sum_d (sum_{i in c, date d} (e_i - m_c))^2 / n_c^2   (date-clustered sampling variance of m_c),
  sigma2 = sum_c n_c^2 V_c / sum_c n_c                       (per-row variance incl. the cross-sectional design effect),
  tau2 = max(0, sum_c n_c (m_c^2 - V_c) / sum_c n_c)          (moment estimator of the between-cell variance),
  kappa = sigma2 / tau2 (one pooled kappa; +inf when tau2 = 0), w_c = n_c / (n_c + kappa), shrunk_c = w_c m_c.
So a cell with few rows is pulled toward 0 (no adjustment) and kappa is estimated from the same (training) sample;
with tau2 = 0 every cell value is exactly 0.
Everything is computed from per-date x per-cell sums (S, N), which also makes date-block bootstraps cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KAPPA_INF = 1e12
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# ------------------------------------------------------------------ empirical-Bayes cell effects


@dataclass
class CellEffects:
    mu: float
    n: np.ndarray        # rows per cell
    n_dates: np.ndarray  # distinct dates per cell
    mean: np.ndarray     # raw centred cell mean (nan for empty cells)
    var: np.ndarray      # clustered sampling variance of the mean
    kappa: float
    tau2: float
    sigma2: float
    w: np.ndarray
    shrunk: np.ndarray   # w * mean (0 for empty cells)

    def value(self, cells: np.ndarray) -> np.ndarray:
        """Shrunk effect for each cell id (0 for ids outside the table)."""
        cells = np.asarray(cells, dtype=int)
        out = np.zeros(len(cells))
        ok = (cells >= 0) & (cells < len(self.shrunk))
        out[ok] = self.shrunk[cells[ok]]
        return out

    def to_dict(self) -> dict:
        return {"mu": self.mu, "kappa": self.kappa, "tau2": self.tau2, "sigma2": self.sigma2, "n": self.n.tolist(),
                "n_dates": self.n_dates.tolist(), "mean": np.where(np.isfinite(self.mean), self.mean, 0.0).tolist(),
                "var": self.var.tolist(), "w": self.w.tolist(), "shrunk": self.shrunk.tolist()}


def date_cell_sums(values: np.ndarray, cells: np.ndarray, date_codes: np.ndarray, n_cells: int,
                   n_dates: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(S, N): per date x cell sums of values and row counts."""
    v = np.asarray(values, dtype=float)
    c = np.asarray(cells, dtype=int)
    d = np.asarray(date_codes, dtype=int)
    nd = int(d.max()) + 1 if n_dates is None else n_dates
    S = np.zeros((nd, n_cells))
    N = np.zeros((nd, n_cells))
    np.add.at(S, (d, c), v)
    np.add.at(N, (d, c), 1.0)
    return S, N


def eb_shrink(S: np.ndarray, N: np.ndarray, center: float | None = None, kappa: float | None = None) -> CellEffects:
    """Empirical-Bayes shrunk cell means from per-date x cell sums (module docstring). `center` None = pooled mean."""
    S = np.asarray(S, dtype=float)
    N = np.asarray(N, dtype=float)
    ntot = N.sum()
    mu = float(S.sum() / ntot) if center is None else float(center)
    Se = S - N * mu  # sums of centred values
    n = N.sum(0)
    nd = (N > 0).sum(0)
    with np.errstate(all="ignore"):
        m = np.where(n > 0, Se.sum(0) / n, np.nan)
        resid = Se - N * np.where(np.isfinite(m), m, 0.0)[None, :]
        V = np.where(n > 0, (resid ** 2).sum(0) / n ** 2, np.nan)
    ok = n > 0
    sigma2 = float((n[ok] ** 2 * V[ok]).sum() / n[ok].sum()) if ok.any() else float("nan")
    tau2 = float(max(0.0, (n[ok] * (m[ok] ** 2 - V[ok])).sum() / n[ok].sum())) if ok.any() else 0.0
    if kappa is None:
        kappa = sigma2 / tau2 if tau2 > 0 and np.isfinite(sigma2) else KAPPA_INF
    kappa = float(min(kappa, KAPPA_INF))
    # kappa = +inf (no between-cell variance): every cell is exactly 0, so no spurious ordering survives in float residue
    w = np.where(ok, n / (n + kappa), 0.0) if kappa < KAPPA_INF else np.zeros(len(n))
    shrunk = np.where(ok, w * np.where(np.isfinite(m), m, 0.0), 0.0)
    return CellEffects(mu, n, nd, m, np.where(ok, V, np.nan), kappa, tau2, sigma2, w, shrunk)


def cell_effects(values: np.ndarray, cells: np.ndarray, dates: np.ndarray, n_cells: int, center: float | None = None,
                 kappa: float | None = None) -> CellEffects:
    """Row-level wrapper: `dates` are any hashable day labels (datetime64[D] or ints)."""
    _, codes = np.unique(np.asarray(dates), return_inverse=True)
    S, N = date_cell_sums(values, cells, codes, n_cells)
    return eb_shrink(S, N, center=center, kappa=kappa)


def honest_halves(dates: np.ndarray) -> np.ndarray:
    """True = split half, False = estimation half: alternating calendar weeks (rows of one date stay together and both
    halves see every weekday and the whole period)."""
    d = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    week = (d + 3) // 7  # 1970-01-01 is a Thursday: weeks start on Monday
    return (week % 2) == 0


# ------------------------------------------------------------------ lattices


@dataclass
class QuantileDim:
    """n_bins training-quantile bins of column `col`."""
    name: str
    col: int
    n_bins: int
    words: tuple[str, ...] = ()
    edges: np.ndarray | None = None

    @property
    def size(self) -> int:
        return self.n_bins

    def fit(self, X: np.ndarray) -> QuantileDim:
        x = np.asarray(X, dtype=float)[:, self.col]
        x = x[np.isfinite(x)]
        self.edges = np.quantile(x, np.arange(1, self.n_bins) / self.n_bins)
        return self

    def assign(self, X: np.ndarray) -> np.ndarray:
        assert self.edges is not None, "fit first"
        return np.searchsorted(self.edges, np.asarray(X, dtype=float)[:, self.col], side="right")

    def label(self, b: int) -> str:
        w = self.words[b] if self.words else f"bin {b + 1}/{self.n_bins}"
        e = self.edges
        if e is None or not len(e):
            return f"{self.name} {w}"
        rng = (f"< {e[0]:.3g}" if b == 0 else f">= {e[-1]:.3g}" if b == self.n_bins - 1 else f"{e[b - 1]:.3g} to {e[b]:.3g}")
        return f"{self.name} {w} ({rng})"


@dataclass
class CategoricalDim:
    """Fixed groups of the values of column `col` (no fitting); values outside every group go to the last group."""
    name: str
    col: int
    groups: tuple[tuple[float, ...], ...]
    words: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.groups)

    def fit(self, X: np.ndarray) -> CategoricalDim:
        return self

    def assign(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float)[:, self.col]
        out = np.full(len(x), len(self.groups) - 1)
        for g, vals in enumerate(self.groups):
            out[np.isin(x, vals)] = g
        return out

    def label(self, b: int) -> str:
        return self.words[b]


@dataclass
class Lattice:
    dims: list
    kind: str = "lattice"

    @property
    def n_cells(self) -> int:
        return int(np.prod([d.size for d in self.dims]))

    def fit(self, X: np.ndarray, y=None, dates=None) -> Lattice:
        for d in self.dims:
            d.fit(X)
        return self

    def assign(self, X: np.ndarray) -> np.ndarray:
        cell = np.zeros(len(X), dtype=int)
        for d in self.dims:
            cell = cell * d.size + d.assign(X)
        return cell

    def coords(self, cell: int) -> list[int]:
        out = []
        for d in reversed(self.dims):
            out.append(cell % d.size)
            cell //= d.size
        return out[::-1]

    def describe(self, cell: int) -> str:
        return ", ".join(d.label(b) for d, b in zip(self.dims, self.coords(cell), strict=True))

    def spec(self) -> dict:
        return {"kind": self.kind, "dims": [{"name": d.name, "col": d.col, "size": d.size,
                                             "edges": (d.edges.tolist() if isinstance(d, QuantileDim) and d.edges is not None else None),
                                             "words": list(d.words)} for d in self.dims]}


# ------------------------------------------------------------------ k-means codebook


def _sqdist(Z: np.ndarray, C: np.ndarray) -> np.ndarray:
    return (Z ** 2).sum(1)[:, None] - 2 * Z @ C.T + (C ** 2).sum(1)[None, :]


@dataclass
class KMeansCodebook:
    k: int
    cols: tuple[int, ...]
    n_init: int = 3
    max_iter: int = 100
    seed: int = 0
    max_fit_rows: int = 20000
    kind: str = "kmeans"
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    centroids: np.ndarray | None = None
    inertia: float = float("nan")

    @property
    def n_cells(self) -> int:
        return self.k

    def _z(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float)[:, list(self.cols)] - self.mu) / self.sd

    def fit(self, X: np.ndarray, y=None, dates=None) -> KMeansCodebook:
        Xc = np.asarray(X, dtype=float)[:, list(self.cols)]
        self.mu = Xc.mean(0)
        sd = Xc.std(0)
        self.sd = np.where(sd > 0, sd, 1.0)
        Z = (Xc - self.mu) / self.sd
        rng = np.random.default_rng(self.seed)
        if len(Z) > self.max_fit_rows:
            Z = Z[rng.choice(len(Z), self.max_fit_rows, replace=False)]
        best = None
        for _ in range(self.n_init):
            C = self._pp_init(Z, rng)
            for _ in range(self.max_iter):
                lab = np.argmin(_sqdist(Z, C), 1)
                Cn = C.copy()
                for j in range(self.k):
                    sel = lab == j
                    if sel.any():
                        Cn[j] = Z[sel].mean(0)
                    else:  # empty cluster: move to the point farthest from its centroid
                        dmin = _sqdist(Z, C).min(1)
                        Cn[j] = Z[int(np.argmax(dmin))]
                if np.allclose(Cn, C, atol=1e-7):
                    C = Cn
                    break
                C = Cn
            inert = float(_sqdist(Z, C).min(1).sum())
            if best is None or inert < best[0]:
                best = (inert, C)
        self.inertia, self.centroids = best
        return self

    def _pp_init(self, Z: np.ndarray, rng) -> np.ndarray:
        C = [Z[rng.integers(len(Z))]]
        d2 = ((Z - C[0]) ** 2).sum(1)
        for _ in range(1, self.k):
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(len(Z), 1 / len(Z))
            C.append(Z[rng.choice(len(Z), p=p)])
            d2 = np.minimum(d2, ((Z - C[-1]) ** 2).sum(1))
        return np.array(C)

    def assign(self, X: np.ndarray) -> np.ndarray:
        assert self.centroids is not None, "fit first"
        Z = self._z(X)
        out = np.empty(len(Z), dtype=int)
        for i in range(0, len(Z), 50000):
            out[i:i + 50000] = np.argmin(_sqdist(Z[i:i + 50000], self.centroids), 1)
        return out

    def describe(self, cell: int, names: list[str] | None = None, top: int = 3) -> str:
        c = self.centroids[cell]
        order = np.argsort(-np.abs(c))[:top]
        nm = names or [str(j) for j in self.cols]
        return "centroid " + ", ".join(f"{nm[j]} {c[j]:+.1f} sd" for j in order)

    def spec(self) -> dict:
        return {"kind": self.kind, "k": self.k, "cols": list(self.cols), "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "centroids": self.centroids.tolist(), "inertia": self.inertia}


# ------------------------------------------------------------------ honest regression tree


@dataclass
class _Node:
    depth: int
    feature: int = -1
    threshold: float = float("nan")
    left: int = -1
    right: int = -1
    leaf: int = -1
    n_split: int = 0
    path: list = field(default_factory=list)  # [(feature, '<' | '>=', threshold), ...]


@dataclass
class HonestTree:
    """Regression tree grown on the split half only. min_leaf: rows per child in the split half (int) or a fraction."""
    max_depth: int = 3
    min_leaf: float = 0.01
    min_leaf_abs: int = 150
    n_bins: int = 32
    kind: str = "tree"
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        return sum(1 for nd in self.nodes if nd.leaf >= 0)

    def fit(self, X: np.ndarray, y: np.ndarray, dates=None) -> HonestTree:
        assert self.max_depth <= 6, "at most 64 leaves"
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, F = X.shape
        ml = max(self.min_leaf_abs, int(self.min_leaf * n) if self.min_leaf < 1 else int(self.min_leaf))
        self.edges = [np.unique(np.quantile(X[:, j], np.arange(1, self.n_bins) / self.n_bins)) for j in range(F)]
        nb = self.n_bins
        B = np.column_stack([np.searchsorted(self.edges[j], X[:, j], side="right") for j in range(F)])
        offs = (np.arange(F) * nb)[None, :]
        self.nodes = []
        leaves = [0]

        def grow(idx: np.ndarray, depth: int, path: list) -> int:
            me = len(self.nodes)
            node = _Node(depth=depth, n_split=len(idx), path=path)
            self.nodes.append(node)
            best = None
            if depth < self.max_depth and len(idx) >= 2 * ml:
                yy = y[idx]
                flat = (B[idx] + offs).ravel()
                cnt = np.bincount(flat, minlength=F * nb).reshape(F, nb)
                sm = np.bincount(flat, weights=np.repeat(yy, F), minlength=F * nb).reshape(F, nb)
                cl, sl = np.cumsum(cnt, 1)[:, :-1], np.cumsum(sm, 1)[:, :-1]
                N, S = len(idx), yy.sum()
                nr, sr = N - cl, S - sl
                with np.errstate(all="ignore"):
                    gain = sl ** 2 / cl + sr ** 2 / nr - S ** 2 / N
                valid = (cl >= ml) & (nr >= ml)
                for j in range(F):  # bins beyond the feature's real edges are not thresholds
                    valid[j, len(self.edges[j]):] = False
                gain = np.where(valid, gain, -np.inf)
                j, b = np.unravel_index(int(np.argmax(gain)), gain.shape)
                if np.isfinite(gain[j, b]) and gain[j, b] > 1e-12:
                    best = (int(j), int(b))
            if best is None:
                node.leaf = leaves[0]
                leaves[0] += 1
                return me
            j, b = best
            thr = float(self.edges[j][b])
            node.feature, node.threshold = j, thr
            go_left = X[idx, j] < thr
            node.left = grow(idx[go_left], depth + 1, [*path, (j, "<", thr)])
            node.right = grow(idx[~go_left], depth + 1, [*path, (j, ">=", thr)])
            return me

        grow(np.arange(n), 0, [])
        return self

    def assign(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.full(len(X), -1, dtype=int)
        stack = [(0, np.arange(len(X)))]
        while stack:
            k, idx = stack.pop()
            nd = self.nodes[k]
            if nd.leaf >= 0:
                out[idx] = nd.leaf
                continue
            lft = X[idx, nd.feature] < nd.threshold
            stack.append((nd.left, idx[lft]))
            stack.append((nd.right, idx[~lft]))
        return out

    def leaf_paths(self) -> dict[int, list]:
        return {nd.leaf: nd.path for nd in self.nodes if nd.leaf >= 0}

    def spec(self) -> dict:
        return {"kind": self.kind, "max_depth": self.max_depth, "n_leaves": self.n_cells,
                "nodes": [{"depth": nd.depth, "feature": nd.feature, "threshold": nd.threshold, "left": nd.left, "right": nd.right,
                           "leaf": nd.leaf, "n_split": nd.n_split} for nd in self.nodes]}


__all__ = ["CategoricalDim", "CellEffects", "HonestTree", "KMeansCodebook", "Lattice", "QuantileDim", "WEEKDAYS", "cell_effects",
           "date_cell_sums", "eb_shrink", "honest_halves"]
