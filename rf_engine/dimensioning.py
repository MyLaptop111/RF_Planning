"""Coverage- and capacity-based site counts."""
from __future__ import annotations
import math
import numpy as np
from config import settings as S

HEX_FACTOR = 3 * math.sqrt(3) / 2          # 2.598 R^2  (omni hexagon)
TRI_FACTOR = 9 * math.sqrt(3) / 8          # 1.949 R^2  (3-sector site)


def hex_cell_area_km2(radius_km, sectors=1):
    return (TRI_FACTOR if sectors == 3 else HEX_FACTOR) * radius_km ** 2


def isd_km(radius_km, sectors=1):
    return 1.5 * radius_km if sectors == 3 else math.sqrt(3) * radius_km


def sites_from_area(area_km2, radius_km, sectors=1):
    if radius_km <= 0:
        return 0
    return int(math.ceil(area_km2 / hex_cell_area_km2(radius_km, sectors)))


def sites_from_capacity(demand, capacity_per_site):
    if capacity_per_site <= 0:
        return 0
    return int(math.ceil(demand / capacity_per_site))


def required_sites(n_cov, n_cap):
    return max(n_cov, n_cap)


# ---------- traffic ----------
def busy_hour_rate_mbps(monthly_gb, bh_share_pct):
    """Average busy-hour DL rate per active subscriber (decimal GB -> Mbit)."""
    return monthly_gb * 8000.0 / 30.0 * (bh_share_pct / 100.0) / 3600.0


def erlang_b(traffic_erl, channels):
    b = 1.0
    for n in range(1, int(channels) + 1):
        b = traffic_erl * b / (n + traffic_erl * b)
    return b


def erlang_capacity(channels, gos):
    """Max offered traffic (Erlang) for `channels` at blocking `gos` (bisection)."""
    if channels <= 0:
        return 0.0
    lo, hi = 0.0, float(channels) * 3
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if erlang_b(mid, channels) <= gos:
            lo = mid
        else:
            hi = mid
    return lo


def cell_capacity(rf, cap):
    """Usable capacity of ONE cell (sector). Returns (value, unit)."""
    if rf.kind == "gsm":
        ch = rf.sectors and cap.carriers_per_sector * 8 - cap.signalling_ts
        return erlang_capacity(ch, cap.gos_pct / 100.0), "Erlang"
    occ = S.occupied_bw_mhz(rf.technology, rf.bw_mhz)
    thr = cap.se_bps_hz * occ * cap.duty_dl                      # Mbps (avg cell throughput)
    return thr * cap.max_load_pct / 100.0, "Mbps"


def demand_per_user(rf, cap):
    if rf.kind == "gsm":
        return cap.erl_per_sub_me / 1000.0, "Erlang"
    if cap.user_rate_override_kbps > 0:
        return cap.user_rate_override_kbps / 1000.0, "Mbps"
    return busy_hour_rate_mbps(cap.monthly_gb_per_user, cap.busy_hour_share_pct), "Mbps"


def total_demand(area_km2, rf, cap):
    pop = cap.pop_density_km2 * area_km2 if cap.use_density else cap.population_total
    users = pop * cap.subscriber_share_pct / 100.0
    per, unit = demand_per_user(rf, cap)
    return dict(population=pop, users=users, per_user=per, unit=unit, demand=users * per)
