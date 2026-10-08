"""Propagation models (vectorised) with validity checks and inverse solving.

References: Hata (1980); COST-231 Final Report (1999); 3GPP TR 38.901 Table 7.4.1-1
(UMa); ITU-R P.525 (free space).
"""
from __future__ import annotations
import numpy as np

ENVS = ["Dense Urban", "Urban", "Suburban", "Rural/Open"]
MODELS = ["Auto", "Free Space", "Okumura-Hata", "COST-231 Hata", "3GPP TR 38.901 UMa"]
_log = np.log10


def resolve_model(model: str, f_mhz: float) -> str:
    if model != "Auto":
        return model
    if f_mhz <= 1500:
        return "Okumura-Hata"
    if f_mhz <= 2000:
        return "COST-231 Hata"
    return "3GPP TR 38.901 UMa"


def fspl_db(f_mhz, d_km):
    d = np.maximum(np.asarray(d_km, float), 1e-4)
    return 32.44 + 20 * _log(d) + 20 * _log(f_mhz)


def _a_hm_medium(f, hm):
    return (1.1 * _log(f) - 0.7) * hm - (1.56 * _log(f) - 0.8)


def _a_hm_large(f, hm):
    if f <= 300:
        return 8.29 * _log(1.54 * hm) ** 2 - 1.1
    return 3.2 * _log(11.75 * hm) ** 2 - 4.97


def _env_correction(f, env):
    if env in ("Dense Urban", "Urban"):
        return 0.0
    if env == "Suburban":
        return -2 * _log(f / 28.0) ** 2 - 5.4
    return -4.78 * _log(f) ** 2 + 18.33 * _log(f) - 40.94


def okumura_hata_db(f_mhz, hb, hm, d_km, env="Urban"):
    d = np.maximum(np.asarray(d_km, float), 1e-3)
    hb = np.asarray(hb, float)
    a = _a_hm_large(f_mhz, hm) if env == "Dense Urban" else _a_hm_medium(f_mhz, hm)
    L = (69.55 + 26.16 * _log(f_mhz) - 13.82 * _log(hb) - a
         + (44.9 - 6.55 * _log(hb)) * _log(d))
    return L + _env_correction(f_mhz, env)


def cost231_hata_db(f_mhz, hb, hm, d_km, env="Urban", cm_db=None):
    d = np.maximum(np.asarray(d_km, float), 1e-3)
    hb = np.asarray(hb, float)
    cm = (3.0 if env == "Dense Urban" else 0.0) if cm_db is None else cm_db
    L = (46.3 + 33.9 * _log(f_mhz) - 13.82 * _log(hb) - _a_hm_medium(f_mhz, hm)
         + (44.9 - 6.55 * _log(hb)) * _log(d) + cm)
    return L + _env_correction(f_mhz, env)   # suburban/open: Hata corrections (common extension)


def uma_nlos_db(f_mhz, hb, hm, d_km):
    """3GPP TR 38.901 UMa: PL = max(PL_LOS, PL'_NLOS), hE = 1 m (hUT < 13 m)."""
    fc = f_mhz / 1000.0
    d2d = np.maximum(np.asarray(d_km, float) * 1000.0, 10.0)
    hb = np.asarray(hb, float)
    d3d = np.sqrt(d2d ** 2 + (hb - hm) ** 2)
    dbp = 4.0 * (hb - 1.0) * (hm - 1.0) * fc * 1e9 / 3e8
    pl1 = 28.0 + 22 * _log(d3d) + 20 * _log(fc)
    pl2 = (28.0 + 40 * _log(d3d) + 20 * _log(fc)
           - 9 * _log(np.maximum(dbp ** 2 + (hb - hm) ** 2, 1e-6)))
    los = np.where(d2d <= dbp, pl1, pl2)
    nlos = 13.54 + 39.08 * _log(d3d) + 20 * _log(fc) - 0.6 * (hm - 1.5)
    return np.maximum(los, nlos)


def path_loss_db(model, f_mhz, hb, hm, d_km, env="Urban", cm_db=None):
    model = resolve_model(model, f_mhz)
    if model == "Free Space":
        return fspl_db(f_mhz, d_km)
    if model == "Okumura-Hata":
        return okumura_hata_db(f_mhz, hb, hm, d_km, env)
    if model == "COST-231 Hata":
        return cost231_hata_db(f_mhz, hb, hm, d_km, env, cm_db)
    if model == "3GPP TR 38.901 UMa":
        return uma_nlos_db(f_mhz, hb, hm, d_km)
    raise ValueError(f"Unknown model: {model}")


def validity_warnings(model, f_mhz, hb, hm, d_km=None):
    model = resolve_model(model, f_mhz)
    w = []

    def rng(name, v, lo, hi, unit):
        if v is not None and not (lo <= v <= hi):
            w.append(f"{model}: {name} = {v:g} {unit} is outside the validity range "
                     f"[{lo:g}, {hi:g}] {unit} - results are an extrapolation.")
    if model == "Okumura-Hata":
        rng("frequency", f_mhz, 150, 1500, "MHz"); rng("BS height", hb, 30, 200, "m")
        rng("UE height", hm, 1, 10, "m"); rng("distance", d_km, 1, 20, "km")
    elif model == "COST-231 Hata":
        rng("frequency", f_mhz, 1500, 2000, "MHz"); rng("BS height", hb, 30, 200, "m")
        rng("UE height", hm, 1, 10, "m"); rng("distance", d_km, 1, 20, "km")
    elif model == "3GPP TR 38.901 UMa":
        rng("frequency", f_mhz, 500, 100000, "MHz"); rng("UE height", hm, 1.5, 22.5, "m")
        if hb != 25:
            w.append("3GPP TR 38.901 UMa: model is defined for hBS = 25 m; other heights are an extrapolation.")
        rng("distance", d_km, 0.01, 5.0, "km")
    elif model == "Free Space":
        w.append("Free Space is a LOS lower bound (optimistic); do not use it for urban coverage planning.")
    return w


def solve_distance_km(model, f_mhz, hb, hm, pl_max_db, env="Urban", cm_db=None,
                      d_lo=1e-3, d_hi=100.0):
    """Largest distance with PL(d) <= pl_max (all supported models are monotonic in d)."""
    def pl(d):
        return float(path_loss_db(model, f_mhz, hb, hm, d, env, cm_db))
    if pl_max_db <= pl(d_lo):
        return 0.0
    if pl_max_db >= pl(d_hi):
        return d_hi
    lo, hi = d_lo, d_hi
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if pl(mid) <= pl_max_db:
            lo = mid
        else:
            hi = mid
    return float(lo)


def loss_exponent(model, f_mhz, hb, hm, d_km=1.0, env="Urban", cm_db=None):
    """Local path-loss exponent n (PL = const + 10 n log10 d)."""
    a = float(path_loss_db(model, f_mhz, hb, hm, d_km, env, cm_db))
    b = float(path_loss_db(model, f_mhz, hb, hm, d_km * 1.5, env, cm_db))
    return (b - a) / (10 * np.log10(1.5))
