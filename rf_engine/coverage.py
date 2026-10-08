"""Grid generation, clutter classification, RSRP tensors and plan evaluation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import shapely
from scipy.special import ndtr
from scipy.spatial import cKDTree

from config import settings as S
from rf_engine.pathloss import path_loss_db
from rf_engine.antenna import sector_pattern_db
from rf_engine.diffraction import terrain_loss_db


@dataclass
class Grid:
    x: np.ndarray
    y: np.ndarray
    ix: np.ndarray
    iy: np.ndarray
    nx: int
    ny: int
    x0: float
    y0: float
    cell_m: float
    clutter: np.ndarray = None      # int index into S.CLUTTERS
    weight: np.ndarray = None       # relative demand weight (sums to 1)

    @property
    def n(self):
        return len(self.x)


def build_grid(poly_xy, cell_m, max_points):
    minx, miny, maxx, maxy = poly_xy.bounds
    area = poly_xy.area
    cell_m = max(cell_m, float(np.sqrt(area / max_points)))
    xs = np.arange(minx + cell_m / 2, maxx, cell_m)
    ys = np.arange(miny + cell_m / 2, maxy, cell_m)
    gx, gy = np.meshgrid(xs, ys)
    inside = shapely.contains_xy(poly_xy, gx.ravel(), gy.ravel())
    ii, jj = np.meshgrid(np.arange(len(xs)), np.arange(len(ys)))
    return Grid(gx.ravel()[inside], gy.ravel()[inside], ii.ravel()[inside], jj.ravel()[inside],
                len(xs), len(ys), xs[0], ys[0], cell_m)


def classify_clutter(grid, bld_xy, thresholds, default_idx, radius_m=300.0):
    """Building-density based clutter class (buildings per ha within radius_m)."""
    if bld_xy is None or len(bld_xy) == 0:
        return np.full(grid.n, default_idx, int), None
    tree = cKDTree(bld_xy)
    cnt = tree.query_ball_point(np.column_stack([grid.x, grid.y]), r=radius_m, return_length=True)
    dens = np.asarray(cnt, float) / (np.pi * radius_m ** 2 / 1e4)
    cl = np.full(grid.n, 3, int)
    cl[dens >= thresholds["Suburban"]] = 2
    cl[dens >= thresholds["Urban"]] = 1
    cl[dens >= thresholds["Dense Urban"]] = 0
    return cl, dens


@dataclass
class RadioParams:
    model: str
    f_mhz: float
    hm: float
    p_ref_dbm: float
    gtx_dbi: float
    cable_db: float
    grx_dbi: float
    body_db: float
    pen_db: np.ndarray        # (K,)
    sigma_db: np.ndarray      # (K,)
    margin_db: np.ndarray     # (K,)
    thr_dbm: np.ndarray       # (K,) effective DL threshold (no margin)
    sectors: int
    az0: float
    tilt_deg: float
    cm_db: float = None
    hbw: float = 65.0
    vbw: float = 10.0


def rsrp_tensor(rp: RadioParams, grid: Grid, sx, sy, hb_eff, az_off=None):
    """Mean RSRP (dBm) at each grid point from each sector of each site -> (S, N, M) float32."""
    sx = np.asarray(sx, float); sy = np.asarray(sy, float)
    M = len(sx); N = grid.n
    hb_eff = np.broadcast_to(np.asarray(hb_eff, float), (M,))
    az_off = np.zeros(M) if az_off is None else np.asarray(az_off, float)
    dx = grid.x[:, None] - sx[None, :]
    dy = grid.y[:, None] - sy[None, :]
    d_m = np.maximum(np.hypot(dx, dy), 20.0)
    d_km = d_m / 1000.0
    pl = np.empty((N, M))
    for k, env in enumerate(S.CLUTTERS):
        idx = np.where(grid.clutter == k)[0]
        if len(idx):
            pl[idx] = path_loss_db(rp.model, rp.f_mhz, hb_eff[None, :], rp.hm, d_km[idx], env, rp.cm_db)
    base = (rp.p_ref_dbm + rp.gtx_dbi - rp.cable_db + rp.grx_dbi - rp.body_db
            - rp.pen_db[grid.clutter][:, None] - pl)
    S_ = max(rp.sectors, 1)
    out = np.empty((S_, N, M), np.float32)
    if rp.sectors <= 1:
        out[0] = base
        return out
    bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
    theta = np.degrees(np.arctan((hb_eff[None, :] - rp.hm) / d_m))
    for s in range(S_):
        bore = rp.az0 + az_off[None, :] + 360.0 / S_ * s
        out[s] = base + sector_pattern_db(bearing, bore, theta, rp.tilt_deg, rp.hbw, rp.vbw)
    return out


def soft_prob(best_dbm, grid, rp):
    """P(RSRP >= threshold) under log-normal shadowing, per grid point."""
    thr = rp.thr_dbm[grid.clutter]; sg = rp.sigma_db[grid.clutter]
    z = (np.asarray(best_dbm) - (thr[:, None] if np.ndim(best_dbm) == 2 else thr)) / \
        (sg[:, None] if np.ndim(best_dbm) == 2 else sg)
    return ndtr(z)


def evaluate(rp: RadioParams, grid: Grid, sites_xy, hb_eff, az_off, hb_agl, rf_kind,
             noise_ref_dbm, load, dem=None, projector=None, demand=None, cell_cap=None):
    """Final evaluation of a chosen set of sites (best server, SINR, loads)."""
    sx, sy = sites_xy[:, 0], sites_xy[:, 1]
    T = rsrp_tensor(rp, grid, sx, sy, hb_eff, az_off)            # (S,N,M)
    S_, N, M = T.shape
    R = T.transpose(1, 0, 2).reshape(N, S_ * M)                  # cell idx = s*M + m
    serv = R.argmax(1)
    best = R[np.arange(N), serv].astype(float)
    serv_site = serv % M
    diff = np.zeros(N)
    if dem is not None and projector is not None:
        diff = terrain_loss_db(dem, projector, sites_xy[serv_site], hb_agl,
                               np.column_stack([grid.x, grid.y]), rp.hm, rp.f_mhz)
    best_adj = best - diff
    thr = rp.thr_dbm[grid.clutter]; sg = rp.sigma_db[grid.clutter]; mg = rp.margin_db[grid.clutter]
    p = ndtr((best_adj - thr) / sg)
    res = dict(serv_cell=serv, serv_site=serv_site, rsrp=best_adj, rsrp_nodiff=best, diff=diff, p=p,
               design_ok=best_adj >= thr + mg, sinr=np.full(N, np.nan))
    if rf_kind in ("lte", "nr"):
        lin = 10 ** (R.astype(float) / 10.0)
        s_lin = 10 ** (best_adj / 10.0)
        interf = load * (lin.sum(1) - lin[np.arange(N), serv])
        res["sinr"] = 10 * np.log10(s_lin / (interf + 10 ** (noise_ref_dbm / 10.0)))
    if demand is not None and cell_cap:
        load_cell = np.bincount(serv, weights=demand * p, minlength=S_ * M)
        res["cell_load"] = load_cell
        res["cell_cap"] = cell_cap
    return res
