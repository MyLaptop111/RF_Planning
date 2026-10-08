"""Link budget (DL + UL), noise, and shadowing margin from area-coverage probability."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from scipy.special import erfc, erfcx, ndtr

from config import settings as S
from rf_engine.models import RFConfig
from rf_engine.pathloss import resolve_model, loss_exponent, solve_distance_km


def noise_floor_dbm(bw_hz: float, nf_db: float = 0.0) -> float:
    return float(S.THERMAL_NOISE_DBM_HZ + 10 * np.log10(bw_hz) + nf_db)


def mapl_db(tx_power, tx_gain, rx_gain, cable_loss, body_loss, penetration,
            shadow_margin, interference_margin, rx_sensitivity):
    """Generic maximum allowed path loss (kept compatible with the V2 signature)."""
    return (tx_power + tx_gain + rx_gain - cable_loss - body_loss - penetration
            - shadow_margin - interference_margin - rx_sensitivity)


def rsrp_dbm(tx_power, tx_gain, rx_gain, path_loss, other_losses=0.0):
    return tx_power + tx_gain + rx_gain - path_loss - other_losses


def q_func(x):
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2))


def area_coverage_probability(margin_db: float, sigma_db: float, n_exp: float) -> float:
    """Fraction of the cell area above threshold given the mean-power margin at the edge
    (Reudink / Jakes). a = -M/(sigma*sqrt2), b = 10 n log10(e)/(sigma*sqrt2).
    F = 1/2 [erfc(a) + exp(-a^2) erfcx((1-ab)/b)]   (numerically stable form)."""
    s2 = sigma_db * np.sqrt(2)
    a = -margin_db / s2
    b = 10 * n_exp * np.log10(np.e) / s2
    return float(0.5 * (erfc(a) + np.exp(-a * a) * erfcx((1 - a * b) / b)))


def edge_margin_for_area_prob(target: float, sigma_db: float, n_exp: float) -> float:
    """Shadowing margin (dB) at the cell edge so that the AREA coverage probability = target."""
    target = min(max(target, 0.51), 0.999)
    lo, hi = -5 * sigma_db, 8 * sigma_db
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if area_coverage_probability(mid, sigma_db, n_exp) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class LinkBudget:
    clutter: str
    rows: List[dict]
    mapl_dl: float
    mapl_ul: float
    mapl: float
    limiting: str
    margin_db: float
    sigma_db: float
    n_exp: float
    edge_prob: float
    pen_db: float
    thr_eff_dbm: float       # DL reference-signal threshold equivalent to the binding MAPL (no margin)


def rsrp_threshold_dbm(cfg: RFConfig) -> float:
    """Per-RE (or reference-signal) DL threshold implied by the coverage criterion."""
    if cfg.coverage_mode == "RSRP threshold":
        return cfg.rsrp_min_dbm
    sens = noise_floor_dbm(cfg.bw_mhz * 1e6, cfg.ue_nf_db) + cfg.sinr_min_db
    return float(sens + cfg.dl_interference_margin_db - 10 * np.log10(cfg.n_sc))


def build_link_budgets(cfg: RFConfig) -> Dict[str, LinkBudget]:
    model = resolve_model(cfg.model, cfg.freq_mhz)
    out = {}
    p_ref = cfg.p_ref_dbm
    thr_dl = rsrp_threshold_dbm(cfg)
    for env in S.CLUTTERS:
        sigma = cfg.clutter[env]["sigma_db"]
        pen = cfg.clutter[env]["pen_db"]
        n_exp = loss_exponent(model, cfg.freq_mhz, cfg.hb_m, cfg.hm_m, 1.0, env, cfg.cm_db)
        if cfg.shadow_mode == "Auto":
            margin = edge_margin_for_area_prob(cfg.coverage_target_pct / 100.0, sigma, n_exp)
        else:
            margin = cfg.shadow_manual_db
        edge_p = float(ndtr(margin / sigma))
        common = cfg.gtx_dbi - cfg.cable_loss_db + cfg.grx_dbi - cfg.body_loss_db - pen - margin
        mapl_dl = p_ref + common - thr_dl
        shadow_note = (f"sigma={sigma:g} dB, n={n_exp:.2f}, edge prob={edge_p*100:.1f}% -> area prob "
                       f"{cfg.coverage_target_pct:g}%") if cfg.shadow_mode == "Auto" else "manual"
        rows = [
            dict(item="Tx power (total)", value=cfg.ptx_dbm, unit="dBm", note=""),
            dict(item="Reference-signal power", value=round(p_ref, 2), unit="dBm",
                 note=(f"Ptx - 10log10({cfg.n_sc}) (equal power per RE)" if cfg.n_sc > 1 else "reference channel")),
            dict(item="Tx antenna gain (boresight)", value=cfg.gtx_dbi, unit="dBi", note=""),
            dict(item="Feeder/cable loss", value=-cfg.cable_loss_db, unit="dB", note=""),
            dict(item="UE antenna gain", value=cfg.grx_dbi, unit="dBi", note=""),
            dict(item="Body loss", value=-cfg.body_loss_db, unit="dB", note=""),
            dict(item="Penetration loss", value=-pen, unit="dB", note=env),
            dict(item="Shadowing margin", value=-round(margin, 2), unit="dB", note=shadow_note),
            dict(item="Required RSRP / rx level (DL)", value=round(thr_dl, 2), unit="dBm", note=cfg.coverage_mode),
            dict(item="MAPL downlink", value=round(mapl_dl, 2), unit="dB", note=""),
        ]
        mapl_ul = np.inf
        if cfg.include_uplink:
            sens_bs = noise_floor_dbm(cfg.ul_bw_khz * 1e3, cfg.bs_nf_db) + cfg.ul_sinr_min_db
            mapl_ul = (cfg.ue_ptx_dbm + cfg.ue_gain_dbi - cfg.body_loss_db + cfg.gtx_dbi - cfg.cable_loss_db
                       - pen - margin - cfg.ul_interference_margin_db - sens_bs)
            rows += [
                dict(item="UE Tx power (UL)", value=cfg.ue_ptx_dbm, unit="dBm", note=""),
                dict(item="BS Rx sensitivity (UL)", value=round(sens_bs, 2), unit="dBm",
                     note=f"-174+10log10({cfg.ul_bw_khz:g}kHz)+NF {cfg.bs_nf_db:g}+SINR {cfg.ul_sinr_min_db:g}"),
                dict(item="UL interference margin", value=-cfg.ul_interference_margin_db, unit="dB", note=""),
                dict(item="MAPL uplink", value=round(float(mapl_ul), 2), unit="dB", note=""),
            ]
        mapl = float(min(mapl_dl, mapl_ul))
        limiting = "Downlink" if mapl_dl <= mapl_ul else "Uplink"
        rows.append(dict(item="MAPL (binding)", value=round(mapl, 2), unit="dB", note=limiting + "-limited"))
        thr_eff = thr_dl + (mapl_dl - mapl)       # DL level that yields the binding MAPL
        out[env] = LinkBudget(env, rows, float(mapl_dl), float(mapl_ul), mapl, limiting,
                              float(margin), float(sigma), float(n_exp), edge_p, float(pen), float(thr_eff))
    return out


def radius_km(cfg: RFConfig, lb: LinkBudget, env: str, hb: float = None) -> float:
    model = resolve_model(cfg.model, cfg.freq_mhz)
    return solve_distance_km(model, cfg.freq_mhz, hb or cfg.hb_m, cfg.hm_m, lb.mapl, env, cfg.cm_db)
