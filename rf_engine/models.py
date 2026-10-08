"""Dataclasses shared across the project."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
from config import settings as S


@dataclass
class RFConfig:
    technology: str = "4G LTE"
    freq_mhz: float = 1800.0
    bw_mhz: float = 20.0
    ptx_dbm: float = 43.0
    gtx_dbi: float = 17.0
    grx_dbi: float = 0.0
    hb_m: float = 30.0
    hm_m: float = 1.5
    cable_loss_db: float = 2.0
    body_loss_db: float = 0.0
    ue_nf_db: float = 7.0
    coverage_mode: str = "RSRP threshold"     # or "SINR sensitivity"
    rsrp_min_dbm: float = -110.0
    sinr_min_db: float = -6.0
    dl_interference_margin_db: float = 3.0
    include_uplink: bool = True
    ue_ptx_dbm: float = 23.0
    ue_gain_dbi: float = 0.0
    bs_nf_db: float = 3.0
    ul_bw_khz: float = 1800.0
    ul_sinr_min_db: float = -3.0
    ul_interference_margin_db: float = 2.0
    model: str = "Auto"
    cm_db: Optional[float] = None
    coverage_target_pct: float = 95.0
    shadow_mode: str = "Auto"                 # Auto (area probability) / Manual
    shadow_manual_db: float = 8.0
    sectors: int = 3
    default_clutter: str = "Urban"
    clutter: Dict[str, dict] = field(default_factory=lambda: S.default_clutter_params("Indoor"))

    @property
    def kind(self) -> str:
        return S.TECHNOLOGIES[self.technology]["kind"]

    @property
    def n_sc(self) -> int:
        return S.n_subcarriers(self.technology, self.bw_mhz)

    @property
    def p_ref_dbm(self) -> float:
        """Reference-signal power per RE (LTE/NR) or total reference power (2G/3G)."""
        return float(self.ptx_dbm - 10 * np.log10(self.n_sc))

    @property
    def noise_ref_dbm(self) -> float:
        """Noise power in the reference-signal bandwidth (per RE = SCS)."""
        kind = self.kind
        bw_hz = 15e3 if kind == "lte" else 30e3 if kind == "nr" else self.bw_mhz * 1e6
        return float(S.THERMAL_NOISE_DBM_HZ + 10 * np.log10(bw_hz) + self.ue_nf_db)


@dataclass
class CapacityConfig:
    enabled: bool = True
    use_density: bool = True
    pop_density_km2: float = 5000.0           # DEMO value - replace with CAPMAS/WorldPop
    population_total: float = 0.0
    subscriber_share_pct: float = 30.0
    monthly_gb_per_user: float = 10.0
    busy_hour_share_pct: float = 8.0
    user_rate_override_kbps: float = 0.0
    se_bps_hz: float = 1.7
    duty_dl: float = 1.0
    max_load_pct: float = 70.0
    erl_per_sub_me: float = 25.0              # GSM
    carriers_per_sector: int = 3
    signalling_ts: int = 2
    gos_pct: float = 2.0


@dataclass
class OptSettings:
    cell_m: float = 100.0
    max_grid_points: int = 6000
    max_candidates: int = 350
    cand_spacing_factor: float = 0.5
    az0_deg: float = 0.0
    optimize_azimuth: bool = True
    tilt_deg: Optional[float] = None           # None = auto from cell radius
    force_user_sites: bool = False
    prune: bool = True
    max_sites: int = 400
    use_buildings: bool = True
    use_dem: bool = True
    thresholds: Dict[str, float] = field(default_factory=lambda: dict(S.BLD_PER_HA_THRESHOLDS))
    tall_building_m: float = 18.0


@dataclass
class PlanResult:
    meta: dict
    area_km2: float
    link_budgets: dict
    radius_table: object
    dimensioning: dict
    sites: object
    grid: object
    stats: dict
    warnings: List[str]
    raster: dict
    boundary_geojson: dict
