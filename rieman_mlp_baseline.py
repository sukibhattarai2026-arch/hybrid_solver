"""
Baseline: can an MLP replace the Riemann solver?   F_hat = MLP([W_L, W_R])

Pipeline
  1. load HDF5 (27 components), split by phi-slices (so the test set is a set of
     full (r, theta) images -> imshow works)
  2. train MLP 16 -> 128 -> 128 -> 128 -> 8 (SiLU), loss = MSE + relative L2
  3. test metrics + parity plots (GT flux vs MLP flux, per variable)
  4. equal-state consistency test  F_hat(W, W) vs F_physical(W)
  5. imshow of GT / MLP / |error| per flux variable on a test (r, theta) slice

Usage:
  # use the dir_0 (radial-direction) file of each CR
  python riemann_mlp_baseline.py --train cr1.h5 cr2.h5 cr3.h5 --val cr4.h5 --test cr5.h5 --epochs 100
  python riemann_mlp_baseline.py --h5 one_file.h5      # quick debug (phi-slice split)
  python riemann_mlp_baseline.py --h5 one_file.h5 --inspect
"""
import argparse, itertools, os
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARS = ["density", "Vr", "Vt", "Vp", "p", "Br", "Bt", "Bp"]
FLUX_NAMES = ["F_density", "F_Vr", "F_Vt", "F_Vp", "F_p", "F_Br", "F_Bt", "F_Bp"]
EPS = 1e-12


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def inspect_h5(path):
    """Print the HDF5 tree (use this and paste it back if loading fails)."""
    with h5py.File(path, "r") as f:
        print("ROOT ATTRS:", {k: v for k, v in f.attrs.items() if not k.startswith("component_")})
        def show(n, o):
            if isinstance(o, h5py.Dataset):
                print(f"{n}  shape={o.shape} dtype={o.dtype}")
            else:
                print(f"{n}/  attrs={dict(o.attrs)}")
        f.visititems(show)


EXPECTED_NAMES = (["r", "theta", "phi"]
                  + [f"WL_{v}" for v in VARS]      # components 3..10   -> X[:, 0:8]
                  + [f"WR_{v}" for v in VARS]      # components 11..18  -> X[:, 8:16]
                  + FLUX_NAMES)                    # components 19..26  -> Y


def check_component_names(f):
    """Verify the file's component_i attributes match the layout the code assumes."""
    names = []
    for i in range(27):
        v = f.attrs[f"component_{i}"]
        names.append(v.decode() if isinstance(v, bytes) else str(v))
    bad = [(i, n, e) for i, (n, e) in enumerate(zip(names, EXPECTED_NAMES)) if n != e]
    if bad:
        raise ValueError(f"component names differ from expected layout: {bad}")
    print("[data] component names verified: X = WL_*(8) + WR_*(8) = components 3..18, "
          "Y = F_*(8) = components 19..26")


def _read_chombo(f):
    """Chombo / 'VanillaAMRFileType' reader (level_0 only).
    Per box the flat data is [comp0 cells, comp1 cells, ...], each in Fortran order
    (i fastest).  Ghost cells / face-centering are inferred from the box length.
    Returns data (N, ncomp) and global integer cell indices ijk (N, 3)."""
    ncomp = int(f.attrs["num_components"])
    check_component_names(f)
    if int(f.attrs.get("num_levels", 1)) > 1:
        print("[data] WARNING: >1 AMR level, using level_0 only")
    lev = f["level_0"]
    boxes = lev["boxes"][...]
    flat = lev["data:datatype=0"][...]
    nb = len(boxes)
    if "data:offsets=0" in lev:
        offs = lev["data:offsets=0"][...]
    else:
        sizes = [int(np.prod([boxes[c.replace("lo", "hi")][b] - boxes[c][b] + 1
                              for c in ("lo_i", "lo_j", "lo_k")])) * ncomp for b in range(nb)]
        offs = np.r_[0, np.cumsum(sizes)]
    rows, ijks = [], []
    for b in range(nb):
        lo = np.array([boxes[c][b] for c in ("lo_i", "lo_j", "lo_k")])
        hi = np.array([boxes[c][b] for c in ("hi_i", "hi_j", "hi_k")])
        dims = hi - lo + 1
        chunk = flat[offs[b]:offs[b + 1]]
        ncells = chunk.size // ncomp
        found = None
        for g in range(5):                                   # ghost width
            for ext in itertools.product((0, 1), repeat=3):  # face-centered extension
                if np.prod(dims + 2 * g + np.array(ext)) == ncells:
                    found = (g, np.array(ext)); break
            if found: break
        if found is None:
            raise ValueError(f"box {b}: cannot match {ncells} cells to box dims {dims}")
        g, ext = found
        d = dims + 2 * g + ext
        arr = chunk.reshape(ncomp, d[2], d[1], d[0])         # [comp, k, j, i]
        arr = arr[:, g:d[2] - g, g:d[1] - g, g:d[0] - g]
        nk, nj, ni = arr.shape[1:]
        if b == 0:
            print(f"[data] Chombo: {nb} boxes, ncomp={ncomp}, box0 dims={dims.tolist()}, "
                  f"ghost={g}, face-ext={ext.tolist()}")
        kk, jj, ii = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
        ijks.append(np.stack([ii.ravel() + lo[0], jj.ravel() + lo[1], kk.ravel() + lo[2]], 1))
        rows.append(arr.reshape(ncomp, -1).T)
    data, ijk = np.concatenate(rows).astype(np.float64), np.concatenate(ijks)
    # face-centered boxes share faces -> drop duplicated cells
    _, first = np.unique(ijk, axis=0, return_index=True)
    if len(first) != len(ijk):
        first = np.sort(first)
        print(f"[data] removed {len(ijk) - len(first)} duplicated shared-face rows")
        data, ijk = data[first], ijk[first]
    return data, ijk


def load_h5(path, key=None):
    """Returns data (N, 27) and integer grid indices ijk (N, 3)."""
    with h5py.File(path, "r") as f:
        if key is None and "level_0" in f:
            data, ijk = _read_chombo(f)
        else:
            if key is not None:
                d = f[key][...]
            elif all(f"component_{i}" in f for i in range(27)):
                d = np.stack([f[f"component_{i}"][...].ravel() for i in range(27)], axis=1)
            else:
                cands = []
                f.visititems(lambda n, o: cands.append(n) if isinstance(o, h5py.Dataset) else None)
                hit = [n for n in cands if 27 in f[n].shape and f[n].ndim == 2]
                if not hit:
                    raise ValueError(f"No 27-component dataset found. Datasets: {cands}")
                d = f[hit[0]][...]
            if d.shape[0] == 27 and d.shape[1] != 27:
                d = d.T
            data = d.astype(np.float64)
            ijk = np.stack([np.unique(data[:, c], return_inverse=True)[1] for c in range(3)], 1)
    assert data.shape[1] == 27, data.shape
    print(f"[data] unique coords r/theta/phi = "
          f"{[len(np.unique(data[:, c])) for c in range(3)]}, unique i/j/k = "
          f"{[len(np.unique(ijk[:, c])) for c in range(3)]}  (assumed i<->r, j<->theta, k<->phi)")
    return data, ijk


def split_indices(ijk, frac_val=0.15, frac_test=0.15, seed=0):
    """Split by whole phi-slices (random choice of slices). Falls back to a random
    row split if there are fewer than 5 distinct phi (k) indices (e.g. 2D data)."""
    rng = np.random.default_rng(seed)
    phi_u, ip = np.unique(ijk[:, 2], return_inverse=True)
    n = len(ijk)
    if len(phi_u) >= 5:
        perm = rng.permutation(len(phi_u))
        nt = max(1, round(frac_test * len(phi_u)))
        nv = max(1, round(frac_val * len(phi_u)))
        test_phi, val_phi = perm[:nt], perm[nt:nt + nv]
        te = np.where(np.isin(ip, test_phi))[0]
        va = np.where(np.isin(ip, val_phi))[0]
        tr = np.where(~np.isin(ip, np.r_[test_phi, val_phi]))[0]
        print(f"[split] by phi slices: {len(phi_u)} phi values -> test {nt}, val {nv}")
    else:
        perm = rng.permutation(n)
        nt, nv = int(frac_test * n), int(frac_val * n)
        te, va, tr = perm[:nt], perm[nt:nt + nv], perm[nt + nv:]
        print("[split] random row split (few phi values)")
    return tr, va, te


# --------------------------------------------------------------------------- #
# Physical MHD flux (radial direction), for the equal-state consistency test
# --------------------------------------------------------------------------- #
def physical_flux_r(W, gamma=5 / 3, mu=1.0):
    """W: (N,8) = [rho,vr,vt,vp,p,Br,Bt,Bp].  Flux in the +r direction.
    mu: magnetic permeability in your code units (1 for Heaviside-Lorentz /
    'B^2/2' units, 4*pi for Gaussian 'B^2/8pi', mu0 for SI).
    F_p is assumed to be the TOTAL ENERGY flux (conservative form)."""
    rho, vr, vt, vp, p, Br, Bt, Bp = [W[:, i] for i in range(8)]
    B2 = Br**2 + Bt**2 + Bp**2
    v2 = vr**2 + vt**2 + vp**2
    vB = vr * Br + vt * Bt + vp * Bp
    pt = p + B2 / (2 * mu)
    E = p / (gamma - 1) + 0.5 * rho * v2 + B2 / (2 * mu)
    return np.stack([
        rho * vr,
        rho * vr * vr + pt - Br * Br / mu,
        rho * vr * vt - Br * Bt / mu,
        rho * vr * vp - Br * Bp / mu,
        (E + pt) * vr - Br * vB / mu,
        np.zeros_like(rho),                 # d(Br)/dt flux in r-direction is 0
        vr * Bt - vt * Br,
        vr * Bp - vp * Br,
    ], axis=1)


# --------------------------------------------------------------------------- #
# Model / loss
# --------------------------------------------------------------------------- #
def make_mlp():
    return nn.Sequential(
        nn.Linear(16, 128), nn.SiLU(),
        nn.Linear(128, 128), nn.SiLU(),
        nn.Linear(128, 128), nn.SiLU(),
        nn.Linear(128, 8),
    )


def rel_l2(pred, y):
    """||pred - y||_2 / (||y||_2 + eps), aggregated over the batch (physical units)."""
    return torch.linalg.norm(pred - y) / (torch.linalg.norm(y) + EPS)


class Normalizer:
    def __init__(self, X, Y):
        W = np.concatenate([X[:, :8], X[:, 8:]], axis=0)       # shared L/R stats
        self.xm, self.xs = W.mean(0), W.std(0)
        self.ym, self.ys = Y.mean(0), Y.std(0)
        # guard constant channels (e.g. F_Br == 0)
        self.xs = np.where(self.xs < 1e-12 * max(self.xs.max(), 1e-30), 1.0, self.xs)
        self.ys = np.where(self.ys < 1e-12 * max(self.ys.max(), 1e-30), 1.0, self.ys)

    def x(self, X):  return (X - np.tile(self.xm, 2)) / np.tile(self.xs, 2)
    def y(self, Y):  return (Y - self.ym) / self.ys
    def y_inv(self, Yn): return Yn * self.ys + self.ym


def predict(model, norm, X, device, bs=65536):
    model.eval()
    Xn = torch.tensor(norm.x(X), dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xn), bs):
            out.append(model(Xn[i:i + bs].to(device)).cpu().numpy())
    return norm.y_inv(np.concatenate(out).astype(np.float64))


# --------------------------------------------------------------------------- #
# Metrics / plots
# --------------------------------------------------------------------------- #
def per_var_metrics(pred, gt):
    rows = []
    for k in range(8):
        err = pred[:, k] - gt[:, k]
        mse = np.mean(err**2)
        rel = np.linalg.norm(err) / (np.linalg.norm(gt[:, k]) + EPS)
        var = np.var(gt[:, k])
        r2 = 1 - mse / var if var > 0 else np.nan
        rows.append((FLUX_NAMES[k], mse, rel, r2))
    return rows


def print_metrics(title, rows):
    print(f"\n{title}\n{'variable':<12}{'MSE':>14}{'relL2':>12}{'R2':>10}")
    for n, mse, rel, r2 in rows:
        print(f"{n:<12}{mse:>14.4e}{rel:>12.4e}{r2:>10.4f}")


def parity_plot(gt, pred, title, path, max_pts=30000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(gt), min(max_pts, len(gt)), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for k, ax in enumerate(axes.ravel()):
        g, p = gt[idx, k], pred[idx, k]
        ax.scatter(g, p, s=2, alpha=0.3, rasterized=True)
        lo, hi = min(g.min(), p.min()), max(g.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        var = np.var(gt[:, k])
        r2 = 1 - np.mean((pred[:, k] - gt[:, k])**2) / var if var > 0 else np.nan
        ax.set_title(f"{FLUX_NAMES[k]}   R²={r2:.4f}")
        ax.set_xlabel("ground truth"); ax.set_ylabel("MLP")
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def error_vs_radius(r, gt, pred, path, title):
    """Relative L2 error per radius (aggregated over all theta, phi at that r).
    Left: all 8 fluxes together. Right: per flux variable."""
    r_u, inv = np.unique(r, return_inverse=True)
    n = len(r_u)
    err2 = np.stack([np.bincount(inv, weights=(pred[:, k] - gt[:, k]) ** 2, minlength=n) for k in range(8)], 1)
    ref2 = np.stack([np.bincount(inv, weights=gt[:, k] ** 2, minlength=n) for k in range(8)], 1)
    tot = np.sqrt(err2.sum(1)) / (np.sqrt(ref2.sum(1)) + EPS)
    per = np.sqrt(err2) / (np.sqrt(ref2) + EPS)
    rms = np.sqrt(np.mean(gt ** 2, 0))
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ax[0].semilogy(r_u, tot, "k-o", ms=3)
    ax[0].set_title("relative L2, all 8 fluxes")
    for k in range(8):
        if rms[k] > 1e-12 * rms.max():            # skip identically-zero fluxes (e.g. F_Br)
            ax[1].semilogy(r_u, per[:, k], label=FLUX_NAMES[k])
    ax[1].set_title("relative L2 per flux variable"); ax[1].legend(fontsize=8, ncol=2)
    for a_ in ax:
        a_.set_xlabel("r"); a_.set_ylabel("relative L2 error"); a_.grid(alpha=0.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return r_u, tot


def imshow_theta_phi(ijk_te, coords_te, gt_te, pred_te, out_prefix, radii=("inner", "middle", "outer")):
    """GT / MLP / |error| maps in (phi, theta) at fixed radii, one figure per radius."""
    i_u, ii = np.unique(ijk_te[:, 0], return_inverse=True)
    j_u, jj = np.unique(ijk_te[:, 1], return_inverse=True)
    k_u, kk = np.unique(ijk_te[:, 2], return_inverse=True)
    pick = {"inner": 0, "middle": len(i_u) // 2, "outer": len(i_u) - 1}
    th, ph = coords_te[:, 1], coords_te[:, 2]
    extent = [ph.min(), ph.max(), th.max(), th.min()]
    for name in radii:
        ri = pick[name]
        m = ii == ri
        r_val = coords_te[m, 0][0]
        print(f"[imshow] (theta, phi) map at {name} radius: r = {r_val:.4g}  "
              f"({m.sum()} test points, grid {len(j_u)} theta x {len(k_u)} phi)")

        def to_img(v):
            img = np.full((len(j_u), len(k_u)), np.nan)
            img[jj[m], kk[m]] = v[m]
            return img

        fig, axes = plt.subplots(8, 3, figsize=(16, 2.7 * 8))
        for k in range(8):
            g, p_ = to_img(gt_te[:, k]), to_img(pred_te[:, k])
            vmin, vmax = np.nanmin(g), np.nanmax(g)
            if vmin == vmax:
                vmax = vmin + 1e-30
            for j, (arr, ttl, kw) in enumerate([
                (g, f"GT {FLUX_NAMES[k]}", dict(vmin=vmin, vmax=vmax)),
                (p_, f"MLP {FLUX_NAMES[k]}", dict(vmin=vmin, vmax=vmax)),
                (np.abs(g - p_), "|GT - MLP|", {}),
            ]):
                ax = axes[k, j]
                im = ax.imshow(arr, cmap="gnuplot", origin="upper", aspect="auto",
                               extent=extent, **kw)
                ax.set_title(ttl, fontsize=9)
                ax.set_xlabel("phi"); ax.set_ylabel("theta")
                fig.colorbar(im, ax=ax)
        fig.suptitle(f"{name} radius, r = {r_val:.4g}")
        fig.tight_layout(); fig.savefig(f"{out_prefix}_{name}.png", dpi=110); plt.close(fig)


def imshow_flux(ijk_te, coords_te, gt_te, pred_te, path, k_index=None):
    """GT / MLP / |error| per flux variable on one test (r, theta) slice."""
    i_u, ii = np.unique(ijk_te[:, 0], return_inverse=True)
    j_u, jj = np.unique(ijk_te[:, 1], return_inverse=True)
    k_u, kk = np.unique(ijk_te[:, 2], return_inverse=True)
    if k_index is None:                         # test slice with most points
        k_index = np.bincount(kk).argmax()
    m = kk == k_index
    print(f"[imshow] phi-index k={k_u[k_index]}  ({m.sum()} test points, "
          f"grid {len(j_u)} theta x {len(i_u)} r)")
    r_u = np.unique(coords_te[:, 0])
    t_u = np.unique(coords_te[:, 1])
    it, ir = jj, ii

    def to_img(v):
        img = np.full((len(t_u), len(r_u)), np.nan)
        img[it[m], ir[m]] = v[m]
        return img

    extent = [r_u.min(), r_u.max(), t_u.max(), t_u.min()]
    fig, axes = plt.subplots(8, 3, figsize=(16, 2.7 * 8))
    for k in range(8):
        g, p = to_img(gt_te[:, k]), to_img(pred_te[:, k])
        e = np.abs(g - p)
        vmin, vmax = np.nanmin(g), np.nanmax(g)
        if vmin == vmax:
            vmax = vmin + 1e-30
        for j, (arr, ttl, kw) in enumerate([
            (g, f"GT {FLUX_NAMES[k]}", dict(vmin=vmin, vmax=vmax)),
            (p, f"MLP {FLUX_NAMES[k]}", dict(vmin=vmin, vmax=vmax)),
            (e, f"|GT - MLP|", {}),
        ]):
            ax = axes[k, j]
            im = ax.imshow(arr, cmap="gnuplot", origin="upper", aspect="auto",
                           extent=extent, **kw)
            ax.set_title(ttl, fontsize=9)
            ax.set_xlabel("r"); ax.set_ylabel("theta")
            fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", help="HDF5 files (one per CR) used for training")
    ap.add_argument("--val", nargs="+", help="HDF5 file(s) used for validation")
    ap.add_argument("--test", nargs="+", help="HDF5 file(s) used for testing")
    ap.add_argument("--h5", default=None, help="single-file debug mode (splits by phi-slices)")
    ap.add_argument("--inspect", action="store_true", help="print HDF5 tree and exit")
    ap.add_argument("--key", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rel_weight", type=float, default=1.0, help="weight of relL2 term")
    ap.add_argument("--gamma", type=float, default=5 / 3)
    ap.add_argument("--mu", type=float, default=1.0, help="mu in B^2/(2 mu); match your code units")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if a.inspect:
        inspect_h5(a.h5 or (a.train or a.val or a.test)[0]); return

    def load_clean(path):
        d, ij = load_h5(path, a.key)
        ok = (np.isfinite(d).all(1) & (d[:, 3] > 0) & (d[:, 11] > 0)
              & (d[:, 7] > 0) & (d[:, 15] > 0))          # rho, p > 0 on both sides
        if (~ok).any():
            print(f"[data] {path}: dropping {(~ok).sum()} invalid rows (non-finite or rho/p <= 0)")
        return d[ok], ij[ok]

    if a.train and a.val and a.test:
        # ---- split by whole Carrington rotation (one file per CR) ----
        parts, fid, sizes, fid_names = [], [], {}, []
        for name in ("train", "val", "test"):
            n_rows = 0
            for path in getattr(a, name):
                print(f"[load] {name}: {path}")
                d, ij = load_clean(path)
                parts.append((d, ij)); fid.append(np.full(len(d), len(fid_names)))
                fid_names.append(path); n_rows += len(d)
            sizes[name] = n_rows
        data = np.concatenate([p_[0] for p_ in parts]); ijk = np.concatenate([p_[1] for p_ in parts])
        file_id = np.concatenate(fid)
        n1, n2 = sizes["train"], sizes["train"] + sizes["val"]
        tr, va, te = np.arange(0, n1), np.arange(n1, n2), np.arange(n2, len(data))
        print(f"[split] by CR file: train {len(tr)} rows, val {len(va)} rows, test {len(te)} rows")
    elif a.h5:
        data, ijk = load_clean(a.h5)
        file_id = np.zeros(len(data), dtype=int)
        tr, va, te = split_indices(ijk)
    else:
        raise SystemExit("give --train/--val/--test (one file per CR) or --h5 for single-file debug")
    coords, X, Y = data[:, :3], data[:, 3:19], data[:, 19:27]
    print(f"[data] N={len(data)}  X{X.shape}  Y{Y.shape}")

    # sanity check of the analytic flux against the reference solver:
    same = np.all(np.isclose(X[:, :8], X[:, 8:], rtol=1e-9, atol=0), axis=1)
    if same.sum() > 0:
        Fp = physical_flux_r(X[same, :8], a.gamma, a.mu)
        d = np.linalg.norm(Fp - Y[same]) / (np.linalg.norm(Y[same]) + EPS)
        print(f"[check] {same.sum()} rows with W_L==W_R: |F_phys - F_ref|/|F_ref| = {d:.3e}")
    else:
        print("[check] no rows with W_L == W_R in the data")

    # normalization statistics come from the TRAIN rows only
    norm = Normalizer(X[tr], Y[tr])
    to_t = lambda A: torch.tensor(A, dtype=torch.float32, device=device)
    Xtr, Ytr = to_t(norm.x(X[tr])), to_t(norm.y(Y[tr]))
    Xva, Yva = to_t(norm.x(X[va])), to_t(norm.y(Y[va]))
    ym, ys = to_t(norm.ym), to_t(norm.ys)

    def loss_fn(pred_n, y_n):
        mse = torch.mean((pred_n - y_n) ** 2)                       # normalized space
        rel = rel_l2(pred_n * ys + ym, y_n * ys + ym)               # physical space
        return mse + a.rel_weight * rel, mse, rel

    model = make_mlp().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    hist = {"train": [], "val": [], "val_mse": [], "val_rel": []}
    best, best_state = np.inf, None
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        tot = 0.0
        for i in range(0, len(perm), a.bs):
            b = perm[i:i + a.bs]
            loss, _, _ = loss_fn(model(Xtr[b]), Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl, vm, vr = loss_fn(model(Xva), Yva)
        hist["train"].append(tot / len(perm)); hist["val"].append(vl.item())
        hist["val_mse"].append(vm.item()); hist["val_rel"].append(vr.item())
        if vl.item() < best:
            best = vl.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % max(1, a.epochs // 20) == 0 or ep == a.epochs - 1:
            print(f"ep {ep:4d}  train {hist['train'][-1]:.4e}  val {vl.item():.4e}  "
                  f"(mse {vm.item():.3e}, relL2 {vr.item():.3e})")
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "norm": norm.__dict__}, f"{a.out}/mlp_baseline.pt")

    # loss curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(hist["train"], label="train"); ax[0].semilogy(hist["val"], label="val")
    ax[0].set_title("total loss"); ax[0].legend(); ax[0].set_xlabel("epoch")
    ax[1].semilogy(hist["val_mse"], label="val MSE (norm.)")
    ax[1].semilogy(hist["val_rel"], label="val relL2")
    ax[1].legend(); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(f"{a.out}/loss_curves.png", dpi=130); plt.close(fig)

    # ---------------- test set ----------------
    pred_te, gt_te = predict(model, norm, X[te], device), Y[te]
    print_metrics("TEST metrics per flux variable", per_var_metrics(pred_te, gt_te))
    glob = np.linalg.norm(pred_te - gt_te) / (np.linalg.norm(gt_te) + EPS)
    print(f"\nTEST global relative L2 = {glob:.4e}")
    parity_plot(gt_te, pred_te, "Test set: GT flux (x) vs MLP flux (y)",
                f"{a.out}/parity_test.png")
    error_vs_radius(coords[te, 0], gt_te, pred_te, f"{a.out}/error_vs_radius_test.png",
                    "Test set: relative L2 error vs radius")
    m0 = file_id[te] == file_id[te][0]        # imshow: first test file only
    imshow_flux(ijk[te][m0], coords[te][m0], gt_te[m0], pred_te[m0], f"{a.out}/imshow_test.png")
    imshow_theta_phi(ijk[te][m0], coords[te][m0], gt_te[m0], pred_te[m0],
                     f"{a.out}/imshow_theta_phi_test")

    # ---------------- equal-state consistency ----------------
    # Use every left AND right state of the test set as W; feed X=[W,W].
    W = np.concatenate([X[te, :8], X[te, 8:]], axis=0)
    F_phys = physical_flux_r(W, a.gamma, a.mu)
    F_hat = predict(model, norm, np.concatenate([W, W], axis=1), device)

    E = np.linalg.norm(F_hat - F_phys, axis=1) / (np.linalg.norm(F_phys, axis=1) + EPS)
    print("\nEqual-state consistency  E = ||F_hat(W,W) - F_phys(W)|| / (||F_phys|| + eps)")
    print(f"  mean {E.mean():.4e}  median {np.median(E):.4e}  "
          f"p95 {np.percentile(E, 95):.4e}  max {E.max():.4e}")
    print(f"  global relL2 = {np.linalg.norm(F_hat - F_phys) / np.linalg.norm(F_phys):.4e}")
    print_metrics("Consistency per flux variable (F_hat(W,W) vs F_phys)",
                  per_var_metrics(F_hat, F_phys))
    # reference: same states, but how far is the *reference solver's* own flux from F_phys?
    # (only meaningful if W_L ~ W_R; see [check] above)

    parity_plot(F_phys, F_hat, "Equal-state consistency: F_phys(W) (x) vs MLP F_hat(W,W) (y)",
                f"{a.out}/parity_consistency.png")
    error_vs_radius(np.r_[coords[te, 0], coords[te, 0]], F_phys, F_hat,
                    f"{a.out}/error_vs_radius_consistency.png",
                    "Equal-state consistency: relative L2 error vs radius")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log10(E + 1e-16), bins=80)
    ax.set_xlabel("log10 E_consistent"); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(f"{a.out}/consistency_hist.png", dpi=130); plt.close(fig)
    print(f"\nSaved everything to ./{a.out}/")


if __name__ == "__main__":
    main()