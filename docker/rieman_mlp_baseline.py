"""
Baseline: can an MLP replace the Riemann solver (radial / dir_0 flux)?   F_hat = MLP([W_L, W_R])

MEMORY-LIGHT VERSION (minibatch / streaming)
  1. each CR's dir_0 Chombo HDF5 file is converted ONCE into a compact on-disk cache
     (float32 X, Y + coords + grid indices), read one k-plane at a time.
  2. training streams random contiguous segments from the cache (memmap), shuffles them in a
     buffer (~1M rows) and runs minibatches on the GPU.  Nothing close to the full data is in RAM.
  3. validation uses a fixed random subset; testing is done chunk by chunk on the full test CR.

  X = [WL_density..WL_Bp, WR_density..WR_Bp]  (components 3..18)  ->  16
  Y = [F_density..F_Bp]                       (components 19..26) ->  8

PREPROCESSING (fitted on the TRAIN CRs only, saved to results/scalers.json and mlp_baseline.pt)
  inputs : log10 on density and pressure (L and R), then StandardScaler; one mean/std per physical
           variable, shared by W_L and W_R (Vr, Vt, Vp, Br, Bt, Bp: scaler only)
  outputs: StandardScaler on each of the 8 fluxes individually (including F_Br)
  predictions are inverse-transformed before all physical errors, plots and the consistency test

Usage:
  python riemann_mlp_baseline.py --train cr1_dir0.h5 cr2_dir0.h5 cr3_dir0.h5 \\
         --val cr4_dir0.h5 --test cr5_dir0.h5 --epochs 50 --samples_per_epoch 5000000
  # 8 independent MLPs (one per flux, all 16 inputs); add "--inputs own" for the V_r-only-style ablation
  python riemann_mlp_baseline.py ... --mode separate --out results_separate
  python riemann_mlp_baseline.py --train cr1_dir0.h5 --inspect      # print HDF5 tree
"""
import argparse, collections, hashlib, json, os, queue, threading, time
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
EXPECTED_NAMES = (["r", "theta", "phi"]
                  + [f"WL_{v}" for v in VARS]      # components 3..10   -> X[:, 0:8]
                  + [f"WR_{v}" for v in VARS]      # components 11..18  -> X[:, 8:16]
                  + FLUX_NAMES)                    # components 19..26  -> Y


# --------------------------------------------------------------------------- #
# HDF5 -> on-disk cache (streamed, one k-plane at a time)
# --------------------------------------------------------------------------- #
def inspect_h5(path):
    with h5py.File(path, "r") as f:
        print("ROOT ATTRS:", {k: v for k, v in f.attrs.items() if not k.startswith("component_")})
        def show(n, o):
            if isinstance(o, h5py.Dataset):
                print(f"{n}  shape={o.shape} dtype={o.dtype}")
            else:
                print(f"{n}/  attrs={dict(o.attrs)}")
        f.visititems(show)


def check_component_names(f):
    names = []
    for i in range(27):
        v = f.attrs[f"component_{i}"]
        names.append(v.decode() if isinstance(v, bytes) else str(v))
    bad = [(i, n, e) for i, (n, e) in enumerate(zip(names, EXPECTED_NAMES)) if n != e]
    if bad:
        raise ValueError(f"component names differ from expected layout: {bad}")


def build_cache(h5_path, cache_root, rebuild=False, skip_r=0):
    """Convert one Chombo file (level_0) to memmap-able .npy files. Returns (cache_dir, meta).
    skip_r: drop the first skip_r radial layers (i < i_min + skip_r), e.g. 5 starts at the 6th radius."""
    st = os.stat(h5_path)
    tag = hashlib.md5(os.path.abspath(h5_path).encode()).hexdigest()[:8]
    stem = os.path.splitext(os.path.basename(h5_path))[0]
    cdir = os.path.join(cache_root, f"{stem}_{tag}" + (f"_skip{skip_r}" if skip_r else ""))
    meta_path = os.path.join(cdir, "meta.json")
    if os.path.exists(meta_path) and not rebuild:
        meta = json.load(open(meta_path))
        if meta.get("src_size") == st.st_size and meta.get("src_mtime") == st.st_mtime:
            print(f"[cache] reusing {cdir}  ({meta['n']} rows)")
            return cdir, meta
    os.makedirs(cdir, exist_ok=True)
    print(f"[cache] building {cdir} from {h5_path}")
    t0 = time.time()
    with h5py.File(h5_path, "r") as f:
        check_component_names(f)
        ncomp = int(f.attrs["num_components"])
        assert ncomp == 27, ncomp
        if int(f.attrs.get("num_levels", 1)) > 1:
            print("[cache] WARNING: >1 AMR level, using level_0 only")
        lev = f["level_0"]
        boxes = lev["boxes"][...]
        ds = lev["data:datatype=0"]
        lo = np.stack([boxes[c] for c in ("lo_i", "lo_j", "lo_k")], 1).astype(np.int64)
        hi = np.stack([boxes[c] for c in ("hi_i", "hi_j", "hi_k")], 1).astype(np.int64)
        nb = len(boxes)
        if "data:offsets=0" in lev:
            offs = lev["data:offsets=0"][...].astype(np.int64)
        else:
            offs = np.r_[0, np.cumsum((hi - lo + 1).prod(1) * ncomp)]
        ghi = hi.max(0)
        i_start = int(lo[:, 0].min()) + skip_r                  # first radial index that is kept

        # per box: ghost width g, face extension, array dims in file, dims we keep
        plans = []
        for b in range(nb):
            dims = hi[b] - lo[b] + 1
            ncells = int(offs[b + 1] - offs[b]) // ncomp
            found = None
            for g in range(5):
                # dir_0 = radial faces: the only possible extra face is along r (i). Allowing
                # other axes makes the size match ambiguous when box dims in i and j are equal.
                for ext in ((1, 0, 0), (0, 0, 0)):
                    ext = np.array(ext)
                    if int(np.prod(dims + 2 * g + ext)) == ncells:
                        found = (g, ext); break
                if found: break
            if found is None:
                raise ValueError(f"box {b}: cannot match {ncells} cells to box dims {dims}")
            g, ext = found
            # shared faces: only the box at the domain's upper edge keeps its extra face
            keep = dims + ext * (hi[b] == ghi)
            plans.append((g, dims + 2 * g + ext, keep))
        if nb:
            g0, d0_, k0_ = plans[0]
            print(f"[cache] {nb} boxes; box0: ghost={g0}, file dims={d0_.tolist()}, kept dims={k0_.tolist()}")
        nmax = int(sum(int(np.prod(p[2])) for p in plans))

        mk = lambda name, shape, dt: np.lib.format.open_memmap(
            os.path.join(cdir, name + ".npy"), mode="w+", dtype=dt, shape=shape)
        Xm, Ym = mk("X", (nmax, 16), np.float32), mk("Y", (nmax, 8), np.float32)
        Cm, Im = mk("C", (nmax, 3), np.float64), mk("I", (nmax, 3), np.int32)

        p = 0
        for b in range(nb):
            g, d, keep = plans[b]
            ncells = int(np.prod(d))
            plane = int(d[1] * d[0])
            jj, ii = np.meshgrid(np.arange(keep[1]), np.arange(keep[0]), indexing="ij")
            for kl in range(keep[2]):
                kf = g + kl
                arr = np.empty((ncomp, keep[1], keep[0]))
                for c in range(ncomp):
                    s0 = int(offs[b]) + c * ncells + kf * plane
                    pl = ds[s0:s0 + plane].reshape(d[1], d[0])
                    arr[c] = pl[g:g + keep[1], g:g + keep[0]]
                rows = arr.reshape(ncomp, -1).T
                ok = (np.isfinite(rows).all(1) & (rows[:, 3] > 0) & (rows[:, 11] > 0)
                      & (rows[:, 7] > 0) & (rows[:, 15] > 0))        # rho, p > 0 both sides
                ok &= (lo[b][0] + ii.ravel()) >= i_start                 # skip the first skip_r radii
                m = int(ok.sum())
                if m:
                    sel = rows[ok]
                    Xm[p:p + m] = sel[:, 3:19]; Ym[p:p + m] = sel[:, 19:27]
                    Cm[p:p + m] = sel[:, :3]
                    Im[p:p + m, 0] = (lo[b][0] + ii.ravel())[ok]
                    Im[p:p + m, 1] = (lo[b][1] + jj.ravel())[ok]
                    Im[p:p + m, 2] = lo[b][2] + kl
                    p += m
            print(f"[cache]   box {b + 1}/{nb} done ({p} rows so far, {time.time() - t0:.0f}s)")
        for mm in (Xm, Ym, Cm, Im):
            mm.flush()
        del Xm, Ym, Cm, Im
        meta = dict(n=int(p), nmax=nmax, dropped=int(nmax - p),
                    imin=i_start, imax=int((lo[:, 0] + np.array([q[2][0] for q in plans]) - 1).max()),
                    jmin=int(lo[:, 1].min()), jmax=int((lo[:, 1] + np.array([q[2][1] for q in plans]) - 1).max()),
                    kmin=int(lo[:, 2].min()), kmax=int((lo[:, 2] + np.array([q[2][2] for q in plans]) - 1).max()),
                    skip_r=skip_r, src_size=st.st_size, src_mtime=st.st_mtime)
    json.dump(meta, open(meta_path, "w"))
    print(f"[cache] done: {meta['n']} valid rows ({meta['dropped']} dropped: non-finite or rho/p<=0), "
          f"{time.time() - t0:.0f}s")
    return cdir, meta


_OPEN_MM = collections.OrderedDict()      # path -> open memmap (LRU)
_OPEN_LOCK = threading.Lock()
MAX_OPEN_FILES = 64                       # at most this many cache files are memory-mapped at once


class LazyNpy:
    """Row-sliceable view of a .npy cache file. The file is memory-mapped only when it is read, and at most
    MAX_OPEN_FILES files are mapped at a time (LRU), so hundreds of CR caches do not exhaust the OS
    open-file limit ("Too many open files")."""
    def __init__(self, path, n):
        self.path, self.n = path, n
        arr = np.load(path, mmap_mode="r")
        self.shape, self.dtype = (n,) + arr.shape[1:], arr.dtype
        del arr

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        with _OPEN_LOCK:
            mm = _OPEN_MM.get(self.path)
            if mm is None:
                mm = np.load(self.path, mmap_mode="r")
                _OPEN_MM[self.path] = mm
                while len(_OPEN_MM) > MAX_OPEN_FILES:
                    _OPEN_MM.popitem(last=False)
            else:
                _OPEN_MM.move_to_end(self.path)
        return mm[:self.n][idx]


def open_cache(cdir, meta):
    n = meta["n"]
    return tuple(LazyNpy(os.path.join(cdir, f"{nm}.npy"), n) for nm in ("X", "Y", "C", "I"))


# Preprocessing
#   inputs : log10 on density and pressure (left AND right), then StandardScaler with ONE
#            mean/std per physical variable, shared by W_L and W_R.  Vr, Vt, Vp, Br, Bt, Bp: scaler only.
#   outputs: StandardScaler on each of the 8 fluxes individually (including F_Br).
#   All parameters are fitted on the TRAIN rows only.
LOG_VARS = (0, 4)                                              # density, pressure within the 8 variables
LOG_COLS = tuple(LOG_VARS) + tuple(8 + c for c in LOG_VARS)    # the same for the 16 columns [W_L, W_R]


def log_transform_inputs(X):
    """float64 copy of X (N,16) with log10 applied to density and pressure (L and R).
    Raises if any of them is not strictly positive and finite."""
    Z = np.array(X, dtype=np.float64, copy=True)
    for c in LOG_COLS:
        v = Z[:, c]
        if not (np.all(np.isfinite(v)) and np.all(v > 0)):
            raise ValueError(f"input column {c} ({'WL_' if c < 8 else 'WR_'}{VARS[c % 8]}) has "
                             f"non-positive or non-finite values; cannot take log10")
        Z[:, c] = np.log10(v)
    return Z


def compute_stats(Xs, Ys, chunk=2_000_000):
    """Two-pass mean/std over the TRAIN caches (streamed).
    Inputs are log10-transformed (rho, p) first; L and R states share the statistics."""
    sx = np.zeros(8); sy = np.zeros(8); nx = ny = 0
    for X, Y in zip(Xs, Ys):
        for s in range(0, len(X), chunk):
            x = log_transform_inputs(X[s:s + chunk]); y = np.asarray(Y[s:s + chunk], dtype=np.float64)
            sx += x[:, :8].sum(0) + x[:, 8:].sum(0); nx += 2 * len(x)
            sy += y.sum(0); ny += len(y)
    xm, ym = sx / nx, sy / ny
    vx = np.zeros(8); vy = np.zeros(8)
    for X, Y in zip(Xs, Ys):
        for s in range(0, len(X), chunk):
            x = log_transform_inputs(X[s:s + chunk]); y = np.asarray(Y[s:s + chunk], dtype=np.float64)
            vx += ((x[:, :8] - xm) ** 2).sum(0) + ((x[:, 8:] - xm) ** 2).sum(0)
            vy += ((y - ym) ** 2).sum(0)
    return xm, np.sqrt(vx / nx), ym, np.sqrt(vy / ny)


class Normalizer:
    """x(): raw [W_L, W_R] (N,16) -> log10(rho, p) -> standardize -> float32.
       y(): raw flux (N,8) -> standardize.   y_inv(): standardized prediction -> physical flux.
       state()/from_state(): save and restore the fitted scalers for inference."""
    def __init__(self, xm, xs, ym, ys):
        def safe(sd, what):                              # only guard exactly-degenerate std
            bad = ~(np.isfinite(sd) & (sd > 0))
            if bad.any():
                print(f"[scalers] WARNING: zero/invalid std for {what} {np.where(bad)[0].tolist()}; using 1.0")
            return np.where(bad, 1.0, sd)
        self.xm, self.xs = np.asarray(xm, dtype=np.float64), safe(np.asarray(xs, dtype=np.float64), "inputs")
        self.ym, self.ys = np.asarray(ym, dtype=np.float64), safe(np.asarray(ys, dtype=np.float64), "fluxes")
        self.xm_t, self.xs_t = np.tile(self.xm, 2), np.tile(self.xs, 2)
        self.ym32, self.ys32 = self.ym.astype(np.float32), self.ys.astype(np.float32)

    def x(self, X):  return ((log_transform_inputs(X) - self.xm_t) / self.xs_t).astype(np.float32)
    def y(self, Y):  return ((np.asarray(Y, dtype=np.float64) - self.ym) / self.ys).astype(np.float32)
    def y_inv(self, Yn): return np.asarray(Yn, dtype=np.float64) * self.ys + self.ym

    def state(self):
        return dict(xm=self.xm, xs=self.xs, ym=self.ym, ys=self.ys, log10_vars=list(LOG_VARS),
                    input_names=VARS, flux_names=FLUX_NAMES,
                    note="x = (log10 on density,pressure; then (x-xm)/xs), same xm/xs for W_L and W_R; "
                         "y = (F-ym)/ys; invert prediction with F = y*ys + ym")

    @classmethod
    def from_state(cls, st):
        return cls(st["xm"], st["xs"], st["ym"], st["ys"])


# --------------------------------------------------------------------------- #
# Physical MHD flux (radial direction)
# --------------------------------------------------------------------------- #
def physical_flux_r(W, gamma=5 / 3, mu=1.0):
    """W: (N,8) = [rho,vr,vt,vp,p,Br,Bt,Bp]. Flux in +r.  F_p assumed TOTAL ENERGY flux.
    mu: B^2/(2 mu) convention (1 Heaviside-Lorentz, 4*pi Gaussian, mu0 SI)."""
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
        np.zeros_like(rho),
        vr * Bt - vt * Br,
        vr * Bp - vp * Br,
    ], axis=1)


# --------------------------------------------------------------------------- #
# Model / loss
# --------------------------------------------------------------------------- #
def _mlp(n_in, n_out):
    return nn.Sequential(
        nn.Linear(n_in, 128), nn.SiLU(),
        nn.Linear(128, 128), nn.SiLU(),
        nn.Linear(128, 128), nn.SiLU(),
        nn.Linear(128, n_out),
    )


class SeparateMLPs(nn.Module):
    """8 independent MLPs, one per flux (no shared weights); forward returns (N, 8).
    inputs='all': every net sees all 16 inputs  [WL_*, WR_*].
    inputs='own': net j sees only its own variable pair [WL_j, WR_j]
                  (F_Vr <- WL_Vr, WR_Vr, ...): an ablation, NOT expected to work well."""
    def __init__(self, inputs="all"):
        super().__init__()
        self.cols = [list(range(16)) if inputs == "all" else [j, 8 + j] for j in range(8)]
        self.nets = nn.ModuleList([_mlp(len(c), 1) for c in self.cols])

    def forward(self, x):
        return torch.cat([net(x[:, c]) for net, c in zip(self.nets, self.cols)], dim=1)


def make_model(mode, inputs):
    if mode == "multi":
        return _mlp(16, 8)                    # one MLP, 16 -> 8
    return SeparateMLPs(inputs)               # 8 MLPs, each -> 1


def rel_l2(pred, y):
    return torch.linalg.norm(pred - y) / (torch.linalg.norm(y) + EPS)


def predict_norm(model, xn, device, bs=262144):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xn), bs):
            out.append(model(torch.from_numpy(xn[i:i + bs]).to(device)).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# Streaming minibatch loader
# --------------------------------------------------------------------------- #
def make_segments(Xs, seg):
    """List of (file_idx, start) for contiguous row segments of length seg."""
    return np.array([(f, s) for f, X in enumerate(Xs) for s in range(0, len(X) - seg + 1, seg)], dtype=np.int64)


def read_segments(Xs, Ys, segs, seg, norm):
    idx = segs[np.lexsort((segs[:, 1], segs[:, 0]))]               # sorted -> sequential disk reads
    xb = np.concatenate([Xs[f][s:s + seg] for f, s in idx])
    yb = np.concatenate([Ys[f][s:s + seg] for f, s in idx])
    return norm.x(xb), norm.y(yb)


def epoch_stream(Xs, Ys, segs, seg, norm, n_use, buf_segs, rng):
    """Yields normalized (X, Y) buffers of buf_segs*seg rows, drawn from random segments."""
    perm = rng.permutation(len(segs))[:n_use]
    for g0 in range(0, len(perm), buf_segs):
        yield read_segments(Xs, Ys, segs[perm[g0:g0 + buf_segs]], seg, norm)


def prefetch(gen, maxsize=2):
    q = queue.Queue(maxsize)
    done = object()

    def work():
        try:
            for item in gen:
                q.put(item)
            q.put(done)
        except BaseException as e:
            q.put(e)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


# --------------------------------------------------------------------------- #
# Metrics / plots
# --------------------------------------------------------------------------- #
def rows_from_sums(err2, ref2, sg, n):
    rows = []
    for k in range(8):
        mse = err2[k] / n
        rel = np.sqrt(err2[k]) / (np.sqrt(ref2[k]) + EPS) if ref2[k] > 1e-12 * ref2.max() else np.nan
        var = ref2[k] / n - (sg[k] / n) ** 2
        rows.append((FLUX_NAMES[k], mse, rel, 1 - mse / var if var > 0 else np.nan))
    return rows


def print_metrics(title, rows):
    print(f"\n{title}\n{'variable':<12}{'MSE':>14}{'relL2':>12}{'R2':>10}")
    for n, mse, rel, r2 in rows:
        print(f"{n:<12}{mse:>14.4e}{rel:>12.4e}{r2:>10.4f}")


def parity_plot(gt, pred, title, path):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for k, ax in enumerate(axes.ravel()):
        g, p = gt[:, k], pred[:, k]
        ax.scatter(g, p, s=2, alpha=0.3, rasterized=True)
        lo, hi = min(g.min(), p.min()), max(g.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        var = np.var(g)
        r2 = 1 - np.mean((p - g) ** 2) / var if var > 0 else np.nan
        ax.set_title(f"{FLUX_NAMES[k]}   R²={r2:.4f}")
        ax.set_xlabel("ground truth"); ax.set_ylabel("MLP")
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_error_vs_radius(r_i, err2_i, ref2_i, rms, path, title):
    """Relative L2 error per radius (all theta, phi at that r). Left: all 8 fluxes; right: per variable."""
    ok = np.isfinite(r_i) & (ref2_i.sum(1) > 0)
    r, e2, f2 = r_i[ok], err2_i[ok], ref2_i[ok]
    tot = np.sqrt(e2.sum(1)) / (np.sqrt(f2.sum(1)) + EPS)
    per = np.sqrt(e2) / (np.sqrt(f2) + EPS)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ax[0].semilogy(r, tot, "k-o", ms=3); ax[0].set_title("relative L2, all 8 fluxes")
    for k in range(8):
        if rms[k] > 1e-12 * rms.max():            # skip identically-zero fluxes (F_Br)
            ax[1].semilogy(r, per[:, k], label=FLUX_NAMES[k])
    ax[1].set_title("relative L2 per flux variable"); ax[1].legend(fontsize=8, ncol=2)
    for a_ in ax:
        a_.set_xlabel("r"); a_.set_ylabel("relative L2 error"); a_.grid(alpha=0.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_error_vs_jump(jedges, jcnt, jerr2, jref2, jcen2, path):
    """Relative L2 error (all 8 fluxes) binned by the size of the L/R jump: MLP vs central-flux baseline."""
    c = 0.5 * (jedges[:-1] + jedges[1:])
    ok = jcnt > 0
    mlp = np.sqrt(jerr2) / (np.sqrt(jref2) + EPS)
    cen = np.sqrt(jcen2) / (np.sqrt(jref2) + EPS)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].semilogy(c[ok], mlp[ok], "o-", label="MLP")
    ax[0].semilogy(c[ok], cen[ok], "s--", label="central flux (no dissipation)")
    ax[0].set_xlabel("log10 jump  ||W_R - W_L|| (standardized inputs)"); ax[0].set_ylabel("relative L2 error")
    ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_title("error vs L/R jump size")
    ax[1].bar(c, jcnt, width=jedges[1] - jedges[0]); ax[1].set_yscale("log")
    ax[1].set_xlabel("log10 jump"); ax[1].set_ylabel("number of faces"); ax[1].set_title("how many faces per jump size")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _imshow_triplet(fig, axes_row, g, p_, name, extent, xl, yl):
    vmin, vmax = np.nanmin(g), np.nanmax(g)
    if vmin == vmax:
        vmax = vmin + 1e-30
    for j, (arr, ttl, kw) in enumerate([
        (g, f"GT {name}", dict(vmin=vmin, vmax=vmax)),
        (p_, f"MLP {name}", dict(vmin=vmin, vmax=vmax)),
        (np.abs(g - p_), "|GT - MLP|", {}),
    ]):
        ax = axes_row[j]
        im = ax.imshow(arr, cmap="gnuplot", origin="upper", aspect="auto", extent=extent, **kw)
        ax.set_title(ttl, fontsize=9); ax.set_xlabel(xl); ax.set_ylabel(yl)
        fig.colorbar(im, ax=ax)


def imshow_flux(ijk, coords, gt, pred, path):
    """(r, theta) image at one phi slice: GT / MLP / |error| per flux variable."""
    i_u, ii = np.unique(ijk[:, 0], return_inverse=True)
    j_u, jj = np.unique(ijk[:, 1], return_inverse=True)
    k_u, kk = np.unique(ijk[:, 2], return_inverse=True)
    k_index = np.bincount(kk).argmax()
    m = kk == k_index
    print(f"[imshow] (r, theta) slice at phi-index k={k_u[k_index]}  ({m.sum()} test points, "
          f"grid {len(j_u)} theta x {len(i_u)} r)")
    extent = [coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].max(), coords[:, 1].min()]
    fig, axes = plt.subplots(8, 3, figsize=(16, 2.7 * 8))
    for k in range(8):
        imgs = []
        for v in (gt[:, k], pred[:, k]):
            img = np.full((len(j_u), len(i_u)), np.nan); img[jj[m], ii[m]] = v[m]; imgs.append(img)
        _imshow_triplet(fig, axes[k], imgs[0], imgs[1], FLUX_NAMES[k], extent, "r", "theta")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def imshow_theta_phi(ijk, coords, gt, pred, out_prefix):
    """(phi, theta) maps at the inner / middle / outer radius, one figure each."""
    i_u, ii = np.unique(ijk[:, 0], return_inverse=True)
    j_u, jj = np.unique(ijk[:, 1], return_inverse=True)
    k_u, kk = np.unique(ijk[:, 2], return_inverse=True)
    pick = {"inner": 0, "middle": len(i_u) // 2, "outer": len(i_u) - 1}
    extent = [coords[:, 2].min(), coords[:, 2].max(), coords[:, 1].max(), coords[:, 1].min()]
    for name, ri in pick.items():
        m = ii == ri
        r_val = coords[m, 0][0]
        print(f"[imshow] (theta, phi) map at {name} radius: r = {r_val:.4g}  "
              f"({m.sum()} test points, grid {len(j_u)} theta x {len(k_u)} phi)")
        fig, axes = plt.subplots(8, 3, figsize=(16, 2.7 * 8))
        for k in range(8):
            imgs = []
            for v in (gt[:, k], pred[:, k]):
                img = np.full((len(j_u), len(k_u)), np.nan); img[jj[m], kk[m]] = v[m]; imgs.append(img)
            _imshow_triplet(fig, axes[k], imgs[0], imgs[1], FLUX_NAMES[k], extent, "phi", "theta")
        fig.suptitle(f"{name} radius, r = {r_val:.4g}", y=0.995, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(f"{out_prefix}_{name}.png", dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- #
# Streaming evaluation on the test CR: metrics, parity, error vs r, maps, consistency
# --------------------------------------------------------------------------- #
def evaluate(predict_fn, norm, cache, meta, a, outdir):
    X, Y, C, I = cache
    n = len(X)
    os.makedirs(outdir, exist_ok=True)
    imin, imax, kmin, kmax = meta["imin"], meta["imax"], meta["kmin"], meta["kmax"]
    n_i = imax - imin + 1
    i_sel = sorted({imin, (imin + imax) // 2, imax})
    k_sel = (kmin + kmax) // 2
    rng = np.random.default_rng(a.seed)
    frac, fracc = min(1.0, a.max_pts / n), min(1.0, a.max_pts / (2 * n))
    edges = np.linspace(-14, 6, 2001)

    z8 = lambda: np.zeros(8)
    T = dict(err2=z8(), ref2=z8(), sg=z8())
    Tc = dict(err2=z8(), ref2=z8(), sg=z8())
    err2_i, ref2_i = np.zeros((n_i, 8)), np.zeros((n_i, 8))
    cerr2_i, cref2_i = np.zeros((n_i, 8)), np.zeros((n_i, 8))
    r_of_i = np.full(n_i, np.nan)
    par, parc = [], []
    subA, subB = [], []
    Tcen = dict(err2=np.zeros(8))
    jedges = np.linspace(-6, 2, 33); nj = len(jedges) - 1
    jcnt, jerr2, jref2, jcen2 = np.zeros(nj), np.zeros(nj), np.zeros(nj), np.zeros(nj)
    Ehist = np.zeros(len(edges) - 1); Esum = 0.0; Emax = 0.0; nE = 0
    same_n, same_err2, same_ref2 = 0, 0.0, 0.0

    for s in range(0, n, a.eval_chunk):
        e = min(n, s + a.eval_chunk)
        x = np.asarray(X[s:e]); y = np.asarray(Y[s:e], dtype=np.float64)
        c = np.asarray(C[s:e]); ij = np.asarray(I[s:e])
        ii = ij[:, 0] - imin
        r_of_i[ii] = c[:, 0]

        # ---- test-set prediction ----
        xn = norm.x(x)
        pred = norm.y_inv(predict_fn(xn))
        err = pred - y
        T["err2"] += (err ** 2).sum(0); T["ref2"] += (y ** 2).sum(0); T["sg"] += y.sum(0)
        for k in range(8):
            err2_i[:, k] += np.bincount(ii, weights=err[:, k] ** 2, minlength=n_i)
            ref2_i[:, k] += np.bincount(ii, weights=y[:, k] ** 2, minlength=n_i)
        # ---- diagnostics: central-flux baseline (no dissipation) and error vs L/R jump size ----
        Fc = 0.5 * (physical_flux_r(x[:, :8].astype(np.float64), a.gamma, a.mu)
                    + physical_flux_r(x[:, 8:].astype(np.float64), a.gamma, a.mu))
        cen_err = Fc - y
        Tcen["err2"] += (cen_err ** 2).sum(0)
        jump = np.linalg.norm(xn[:, 8:] - xn[:, :8], axis=1)
        jb = np.clip(np.digitize(np.log10(jump + 1e-12), jedges) - 1, 0, nj - 1)
        jcnt += np.bincount(jb, minlength=nj)
        jerr2 += np.bincount(jb, weights=(err ** 2).sum(1), minlength=nj)
        jref2 += np.bincount(jb, weights=(y ** 2).sum(1), minlength=nj)
        jcen2 += np.bincount(jb, weights=(cen_err ** 2).sum(1), minlength=nj)
        m = rng.random(len(x)) < frac
        par.append((y[m], pred[m]))
        mA = ij[:, 2] == k_sel
        mB = np.isin(ij[:, 0], i_sel)
        for msk, lst in ((mA, subA), (mB, subB)):
            if msk.any():
                lst.append((ij[msk], c[msk], y[msk], pred[msk]))

        # ---- sanity: rows with W_L == W_R vs analytic flux ----
        xl, xr = x[:, :8], x[:, 8:]
        same = np.all(np.isclose(xl, xr, rtol=1e-6, atol=0), axis=1)
        if same.any():
            Fp = physical_flux_r(xl[same].astype(np.float64), a.gamma, a.mu)
            same_err2 += ((Fp - y[same]) ** 2).sum(); same_ref2 += (y[same] ** 2).sum(); same_n += int(same.sum())

        # ---- equal-state consistency: every W_L and W_R as [W, W] ----
        W = np.concatenate([xl, xr], 0).astype(np.float64)
        Fphys = physical_flux_r(W, a.gamma, a.mu)
        Fhat = norm.y_inv(predict_fn(norm.x(np.concatenate([W, W], 1))))
        cerr = Fhat - Fphys
        Tc["err2"] += (cerr ** 2).sum(0); Tc["ref2"] += (Fphys ** 2).sum(0); Tc["sg"] += Fphys.sum(0)
        ii2 = np.concatenate([ii, ii])
        for k in range(8):
            cerr2_i[:, k] += np.bincount(ii2, weights=cerr[:, k] ** 2, minlength=n_i)
            cref2_i[:, k] += np.bincount(ii2, weights=Fphys[:, k] ** 2, minlength=n_i)
        E = np.linalg.norm(cerr, axis=1) / (np.linalg.norm(Fphys, axis=1) + EPS)
        Ehist += np.histogram(np.log10(E + 1e-16), bins=edges)[0]
        Esum += E.sum(); Emax = max(Emax, float(E.max())); nE += len(E)
        mc = rng.random(len(W)) < fracc
        parc.append((Fphys[mc], Fhat[mc]))
        print(f"[eval] {e}/{n} rows")

    cat = lambda lst, j: np.concatenate([t[j] for t in lst])
    rms = np.sqrt(T["ref2"] / n)

    # ---------------- test metrics ----------------
    print_metrics("TEST metrics per flux variable", rows_from_sums(T["err2"], T["ref2"], T["sg"], n))
    print(f"\nTEST global relative L2 = {np.sqrt(T['err2'].sum()) / (np.sqrt(T['ref2'].sum()) + EPS):.4e}")
    tot_mlp = np.sqrt(T["err2"].sum()) / (np.sqrt(T["ref2"].sum()) + EPS)
    tot_cen = np.sqrt(Tcen["err2"].sum()) / (np.sqrt(T["ref2"].sum()) + EPS)
    print_metrics("CENTRAL-FLUX baseline  0.5*(F_phys(W_L)+F_phys(W_R)), i.e. NO dissipation",
                  rows_from_sums(Tcen["err2"], T["ref2"], T["sg"], n))
    print(f"\nglobal relL2:  MLP {tot_mlp:.4e}   central-flux baseline {tot_cen:.4e}")
    print("  (the baseline error = share of the reference flux that is solver dissipation, i.e. what the MLP must learn;\n"
          "   if it is huge the analytic flux convention (gamma, mu, F_p = energy flux, geometry) probably differs)")
    plot_error_vs_jump(jedges, jcnt, jerr2, jref2, jcen2, f"{outdir}/error_vs_jump.png")
    parity_plot(cat(par, 0), cat(par, 1), "Test set: GT flux (x) vs MLP flux (y)", f"{outdir}/parity_test.png")
    plot_error_vs_radius(r_of_i, err2_i, ref2_i, rms, f"{outdir}/error_vs_radius_test.png",
                         "Test set: relative L2 error vs radius")
    if subA:
        imshow_flux(cat(subA, 0), cat(subA, 1), cat(subA, 2), cat(subA, 3), f"{outdir}/imshow_test.png")
    if subB:
        imshow_theta_phi(cat(subB, 0), cat(subB, 1), cat(subB, 2), cat(subB, 3), f"{outdir}/imshow_theta_phi_test")

    # ---------------- consistency ----------------
    if same_n:
        print(f"\n[check] {same_n} rows with W_L==W_R: |F_phys - F_ref|/|F_ref| = "
              f"{np.sqrt(same_err2) / (np.sqrt(same_ref2) + EPS):.3e}")
    else:
        print("\n[check] no rows with W_L == W_R in the test data")
    cdf = np.cumsum(Ehist) / Ehist.sum()
    centers = 10 ** (0.5 * (edges[:-1] + edges[1:]))
    q = lambda p: centers[min(np.searchsorted(cdf, p), len(centers) - 1)]
    print("\nEqual-state consistency  E = ||F_hat(W,W) - F_phys(W)|| / (||F_phys|| + eps)")
    print(f"  mean {Esum / nE:.4e}  median~ {q(0.5):.4e}  p95~ {q(0.95):.4e}  max {Emax:.4e}"
          "   (median/p95 from a 2000-bin histogram)")
    print(f"  global relL2 = {np.sqrt(Tc['err2'].sum()) / np.sqrt(Tc['ref2'].sum()):.4e}")
    print_metrics("Consistency per flux variable (F_hat(W,W) vs F_phys)",
                  rows_from_sums(Tc["err2"], Tc["ref2"], Tc["sg"], nE))
    parity_plot(cat(parc, 0), cat(parc, 1), "Equal-state consistency: F_phys(W) (x) vs MLP F_hat(W,W) (y)",
                f"{outdir}/parity_consistency.png")
    plot_error_vs_radius(r_of_i, cerr2_i, cref2_i, np.sqrt(Tc["ref2"] / nE),
                         f"{outdir}/error_vs_radius_consistency.png",
                         "Equal-state consistency: relative L2 error vs radius")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(0.5 * (edges[:-1] + edges[1:]), Ehist, width=edges[1] - edges[0])
    ax.set_xlabel("log10 E_consistent"); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(f"{outdir}/consistency_hist.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, help="dir_0 HDF5 files (one per CR) for training")
    ap.add_argument("--val", nargs="+", help="dir_0 HDF5 file(s) for validation")
    ap.add_argument("--test", nargs="+", help="dir_0 HDF5 file(s) for testing")
    ap.add_argument("--inspect", action="store_true", help="print the HDF5 tree of the first --train file and exit")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the on-disk caches")
    ap.add_argument("--skip_r", type=int, default=0,
                    help="skip the first N radial layers (i = i_min .. i_min+N-1); prediction starts at i_min+N")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rel_weight", type=float, default=1.0, help="weight of relL2 term")
    ap.add_argument("--mode", choices=["multi", "separate"], default="multi",
                    help="multi: one MLP 16->8.  separate: 8 independent MLPs, one per flux")
    ap.add_argument("--inputs", choices=["all", "own"], default="all",
                    help="separate mode only. all: each net sees all 16 inputs. "
                         "own: net j sees only [WL_j, WR_j] (e.g. F_Vr <- WL_Vr, WR_Vr)")
    ap.add_argument("--samples_per_epoch", type=int, default=0,
                    help="random rows seen per epoch (0 = all training rows)")
    ap.add_argument("--seg", type=int, default=4096, help="rows per contiguous read segment")
    ap.add_argument("--buf_segs", type=int, default=256, help="segments per shuffle buffer (buf = seg*buf_segs rows)")
    ap.add_argument("--val_samples", type=int, default=1_000_000, help="rows in the fixed validation subset")
    ap.add_argument("--eval_chunk", type=int, default=500_000)
    ap.add_argument("--max_pts", type=int, default=30000, help="points in parity plots")
    ap.add_argument("--gamma", type=float, default=5 / 3)
    ap.add_argument("--mu", type=float, default=1.0, help="mu in B^2/(2 mu); match your code units")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.inspect:
        inspect_h5(a.train[0]); return
    if not (a.val and a.test):
        raise SystemExit("need --train, --val and --test (one dir_0 file per CR)")
    if a.inputs == "own" and a.mode != "separate":
        raise SystemExit("--inputs own only makes sense with --mode separate")

    try:                                           # many CR files -> raise the open-file limit if allowed
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = 65536 if hard == resource.RLIM_INFINITY else hard
        if soft != resource.RLIM_INFINITY and soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            print(f"[sys] open-file limit raised {soft} -> {target}")
    except (ImportError, ValueError, OSError) as e:
        print(f"[sys] could not raise open-file limit: {e}")

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    caches = {}
    for name in ("train", "val", "test"):
        caches[name] = []
        for path in getattr(a, name):
            cdir, meta = build_cache(path, a.cache_dir, a.rebuild, a.skip_r)
            caches[name].append((open_cache(cdir, meta), meta))
    Xtr = [c[0][0] for c in caches["train"]]; Ytr = [c[0][1] for c in caches["train"]]
    Xva = [c[0][0] for c in caches["val"]];   Yva = [c[0][1] for c in caches["val"]]
    print(f"[split] train {sum(map(len, Xtr))} rows, val {sum(map(len, Xva))} rows, "
          f"test {sum(len(c[0][0]) for c in caches['test'])} rows")

    print("[stats] computing normalization from TRAIN only ...")
    norm = Normalizer(*compute_stats(Xtr, Ytr))
    print("[scalers] inputs (shared by W_L and W_R; log10 applied first where marked)")
    for i, v in enumerate(VARS):
        print(f"   {('log10 ' if i in LOG_VARS else '') + v:<14} mean {norm.xm[i]: .4e}   std {norm.xs[i]:.4e}")
    print("[scalers] flux targets (each standardized individually)")
    for i, v in enumerate(FLUX_NAMES):
        print(f"   {v:<14} mean {norm.ym[i]: .4e}   std {norm.ys[i]:.4e}")
    json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in norm.state().items()},
              open(f"{a.out}/scalers.json", "w"), indent=2)

    seg_tr = min(a.seg, min(map(len, Xtr))); segs_tr = make_segments(Xtr, seg_tr)
    seg_va = min(a.seg, min(map(len, Xva))); segs_va = make_segments(Xva, seg_va)
    n_use = len(segs_tr) if a.samples_per_epoch <= 0 else min(len(segs_tr), -(-a.samples_per_epoch // seg_tr))
    buf_segs = max(1, a.buf_segs)
    print(f"[train] {len(segs_tr)} segments of {seg_tr} rows; {n_use * seg_tr} rows/epoch; "
          f"shuffle buffer {buf_segs * seg_tr} rows")

    n_val_segs = min(len(segs_va), -(-a.val_samples // seg_va))
    vsel = segs_va[rng.permutation(len(segs_va))[:n_val_segs]]
    xv, yv = read_segments(Xva, Yva, vsel, seg_va, norm)
    xv, yv = torch.from_numpy(xv).to(device), torch.from_numpy(yv).to(device)
    print(f"[val] fixed subset of {len(xv)} rows")

    ym = torch.tensor(norm.ym32, device=device); ys = torch.tensor(norm.ys32, device=device)

    def loss_fn(pred_n, y_n):
        if a.mode == "separate":
            # each flux has its own loss (the nets share nothing): MSE (normalized) + relL2 (physical), summed
            mse = ((pred_n - y_n) ** 2).mean(0)                                   # (8,)
            pr, yt = pred_n * ys + ym, y_n * ys + ym
            eps = 1e-3 * ys * (len(y_n) ** 0.5)                                   # scale-aware floor
            rel = torch.linalg.norm(pr - yt, dim=0) / (torch.linalg.norm(yt, dim=0) + eps)
            return (mse + a.rel_weight * rel).sum(), mse.mean(), rel.mean()
        mse = torch.mean((pred_n - y_n) ** 2)                       # normalized space
        rel = rel_l2(pred_n * ys + ym, y_n * ys + ym)               # physical space
        return mse + a.rel_weight * rel, mse, rel

    model = make_model(a.mode, a.inputs).to(device)
    print(f"[model] mode={a.mode}, inputs={a.inputs if a.mode == 'separate' else 'all (16)'}, "
          f"{sum(p_.numel() for p_ in model.parameters())} parameters")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    hist = {"train": [], "val": [], "val_mse": [], "val_rel": []}
    best, best_state = np.inf, None
    for ep in range(a.epochs):
        t0 = time.time()
        model.train()
        tot = torch.zeros((), device=device); seen = 0
        stream = prefetch(epoch_stream(Xtr, Ytr, segs_tr, seg_tr, norm, n_use, buf_segs, rng))
        for xb_np, yb_np in stream:
            xb = torch.from_numpy(xb_np).to(device); yb = torch.from_numpy(yb_np).to(device)
            perm = torch.randperm(len(xb), device=device)
            for i in range(0, len(perm), a.bs):
                b = perm[i:i + a.bs]
                loss, _, _ = loss_fn(model(xb[b]), yb[b])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.detach() * len(b)
            seen += len(xb)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(xv)
            vl, vm, vr = loss_fn(pv, yv)
            per_t = ((pv - yv) ** 2).mean(0).cpu().numpy()          # val MSE per flux (normalized)
        hist["train"].append(tot.item() / seen); hist["val"].append(vl.item())
        hist["val_mse"].append(vm.item()); hist["val_rel"].append(vr.item())
        if vl.item() < best:
            best = vl.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"ep {ep:4d}  train {hist['train'][-1]:.4e}  val {vl.item():.4e}  "
              f"(mse {vm.item():.3e}, relL2 {vr.item():.3e})  {time.time() - t0:.0f}s")
        print("      val nMSE per flux " + "  ".join(f"{n.replace('F_', '')}={v:.2e}"
                                                    for n, v in zip(FLUX_NAMES, per_t)))
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "norm": norm.state(), "mode": a.mode, "inputs": a.inputs},
               f"{a.out}/mlp_baseline.pt")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(hist["train"], label="train"); ax[0].semilogy(hist["val"], label="val")
    ax[0].set_title("total loss"); ax[0].legend(); ax[0].set_xlabel("epoch")
    ax[1].semilogy(hist["val_mse"], label="val MSE (norm.)"); ax[1].semilogy(hist["val_rel"], label="val relL2")
    ax[1].legend(); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(f"{a.out}/loss_curves.png", dpi=130); plt.close(fig)

    predict_fn = lambda xn: predict_norm(model, xn, device)
    for t, (cache, meta) in enumerate(caches["test"]):
        outdir = a.out if len(caches["test"]) == 1 else f"{a.out}/test{t}"
        print(f"\n=== evaluating test file {a.test[t]} ===")
        evaluate(predict_fn, norm, cache, meta, a, outdir)
    print(f"\nSaved everything to ./{a.out}/")


if __name__ == "__main__":
    main()