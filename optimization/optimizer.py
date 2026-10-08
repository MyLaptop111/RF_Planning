"""End-to-end planning pipeline: grid -> RSRP -> greedy coverage -> capacity relief -> evaluation."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.special import ndtr
from shapely.geometry import mapping

from config import settings as S
from rf_engine.models import RFConfig, CapacityConfig, OptSettings, PlanResult
from rf_engine.geometry import projector_for
from rf_engine.link_budget import build_link_budgets, radius_km, rsrp_threshold_dbm
from rf_engine.pathloss import resolve_model, validity_warnings
from rf_engine import dimensioning as D
from rf_engine.coverage import (build_grid, classify_clutter, RadioParams, rsrp_tensor, evaluate)
from map_engine.site_generator import build_candidates


def _noop(msg, frac=None):
    pass


def greedy_coverage(P, target, forced=(), max_sites=400):
    """Greedy max-coverage on a soft-probability matrix P (N x M). Returns (selected, gains)."""
    N, M = P.shape
    sel = list(forced)
    cur = P[:, sel].max(1) if sel else np.zeros(N, np.float32)
    gains = [None] * len(sel)
    while cur.mean() < target and len(sel) < max_sites:
        gain = np.maximum(P - cur[:, None], 0).sum(0)
        gain[sel] = -1
        j = int(np.argmax(gain))
        if gain[j] <= 1e-6:
            break
        sel.append(j); gains.append(float(gain[j]))
        cur = np.maximum(cur, P[:, j])
    return sel, gains


def prune(P, sel, target, protected=()):
    sel = list(sel)
    changed = True
    while changed and len(sel) > 1:
        changed = False
        sub = P[:, sel]
        cur_mean = sub.max(1).mean()
        contrib = []
        for k, j in enumerate(sel):
            if j in protected:
                continue
            m = np.delete(sub, k, axis=1).max(1).mean()
            contrib.append((cur_mean - m, j))
        for loss, j in sorted(contrib):
            trial = [s for s in sel if s != j]
            if P[:, trial].max(1).mean() >= target:
                sel = trial; changed = True
                break
    return sel


def capacity_expand(T, sel, rp, grid, demand, cell_cap, max_sites):
    """Add candidate sites that relieve overloaded cells. T: (S,N,M) candidate RSRP."""
    S_, N, M = T.shape
    best_cand = T.max(0)                                      # (N,M)
    thr = rp.thr_dbm[grid.clutter]; sg = rp.sigma_db[grid.clutter]
    added = []
    sel = list(sel)
    while len(sel) < max_sites:
        Rs = T[:, :, sel].transpose(1, 0, 2).reshape(N, -1)
        serv = Rs.argmax(1); best = Rs[np.arange(N), serv]
        p = ndtr((best - thr) / sg)
        w = demand * p
        load = np.bincount(serv, weights=w, minlength=Rs.shape[1])
        over = load > cell_cap
        if not over.any():
            break
        over_pts = over[serv]
        sw = (best_cand > best[:, None]) & over_pts[:, None]
        gain = (sw.astype(np.float32).T @ w.astype(np.float32))
        gain[sel] = -1
        j = int(np.argmax(gain))
        if gain[j] <= 1e-9:
            break
        sel.append(j); added.append((j, float(gain[j])))
    return sel, added


def run_plan(geom_ll, rf: RFConfig, cap: CapacityConfig, opt: OptSettings, buildings=None, dem=None,
             user_sites_ll=None, area_name="Area", progress=_noop) -> PlanResult:
    warnings = []
    model = resolve_model(rf.model, rf.freq_mhz)
    proj = projector_for(geom_ll)
    poly_xy = proj.geom_to_xy(geom_ll)
    area_km2 = poly_xy.area / 1e6
    progress("Link budget", 0.05)

    # ---------- link budget & radius ----------
    lbs = build_link_budgets(rf)
    warnings += validity_warnings(model, rf.freq_mhz, rf.hb_m, rf.hm_m)
    radii = {env: radius_km(rf, lbs[env], env) for env in S.CLUTTERS}
    for env, r in radii.items():
        if r == 0:
            warnings.append(f"{env}: link budget cannot close (MAPL below path loss at 1 m).")
        elif r < 1.0 and model in ("Okumura-Hata", "COST-231 Hata"):
            warnings.append(f"{env}: radius {r*1000:.0f} m is below the 1 km lower validity bound of {model}; "
                            "extrapolated result - verify with drive tests / ray tracing.")

    # ---------- grid / clutter / demand ----------
    progress("Grid & clutter", 0.10)
    bxy, bh = None, None
    if buildings is not None and len(buildings):
        bx, by = proj.to_xy(buildings["lon"].values, buildings["lat"].values)
        bxy = np.column_stack([bx, by]); bh = buildings["height_m"].values
    grid = build_grid(poly_xy, opt.cell_m, opt.max_grid_points)
    default_idx = S.CLUTTERS.index(rf.default_clutter)
    grid.clutter, dens = classify_clutter(grid, bxy, opt.thresholds, default_idx)
    if bxy is not None:
        grid.weight = (np.maximum(dens, 0.05 * dens.mean() if dens.mean() > 0 else 1.0))
    else:
        grid.weight = np.ones(grid.n)
    grid.weight = grid.weight / grid.weight.sum()
    cell_area_km2 = grid.cell_m ** 2 / 1e6
    clutter_area = {env: float((grid.clutter == k).sum() * cell_area_km2) for k, env in enumerate(S.CLUTTERS)}
    scale = area_km2 / max(sum(clutter_area.values()), 1e-9)       # grid area -> exact polygon area
    clutter_area = {k: v * scale for k, v in clutter_area.items()}

    # ---------- formula-based dimensioning ----------
    n_cov = sum(D.sites_from_area(a, radii[e], rf.sectors) for e, a in clutter_area.items() if a > 0 and radii[e] > 0)
    dem_info = D.total_demand(area_km2, rf, cap) if cap.enabled else None
    cell_cap, cap_unit = D.cell_capacity(rf, cap)
    site_cap = cell_cap * max(rf.sectors, 1)
    n_cap = D.sites_from_capacity(dem_info["demand"], site_cap) if dem_info else 0
    n_req = D.required_sites(n_cov, n_cap)
    a_site_avg = area_km2 / max(n_cov, 1)
    r_eff = math.sqrt(a_site_avg / (D.TRI_FACTOR if rf.sectors == 3 else D.HEX_FACTOR))
    isd_eff = D.isd_km(r_eff, rf.sectors)
    if grid.cell_m > r_eff * 1000 / 2:
        warnings.append(f"Grid cell {grid.cell_m:.0f} m is coarse relative to the cell radius {r_eff*1000:.0f} m; "
                        "reduce the cell size or the area for better accuracy.")

    # ---------- radio parameters ----------
    tilt = opt.tilt_deg if opt.tilt_deg is not None else float(
        np.clip(math.degrees(math.atan((rf.hb_m - rf.hm_m) / max(r_eff * 1000, 50))), 1.0, 12.0))
    rp = RadioParams(model, rf.freq_mhz, rf.hm_m, rf.p_ref_dbm, rf.gtx_dbi, rf.cable_loss_db, rf.grx_dbi,
                     rf.body_loss_db, np.array([lbs[e].pen_db for e in S.CLUTTERS]),
                     np.array([lbs[e].sigma_db for e in S.CLUTTERS]),
                     np.array([lbs[e].margin_db for e in S.CLUTTERS]),
                     np.array([lbs[e].thr_eff_dbm for e in S.CLUTTERS]),
                     rf.sectors, opt.az0_deg, tilt, rf.cm_db)

    # ---------- candidates ----------
    progress("Candidate sites", 0.18)
    user_xy = None
    if user_sites_ll:
        ux, uy = proj.to_xy([p[1] for p in user_sites_ll], [p[0] for p in user_sites_ll])
        user_xy = np.column_stack([ux, uy])
    S_ = max(rf.sectors, 1)
    mem_cap = int(40e6 / (S_ * grid.n))
    cands = build_candidates(poly_xy, isd_eff * 1000 * opt.cand_spacing_factor,
                             min(opt.max_candidates, max(mem_cap, 50)), bxy, bh, opt.tall_building_m, user_xy)
    cxy = cands["xy"]; M = len(cxy)
    if cands["spacing_m"] > isd_eff * 1000 * opt.cand_spacing_factor * 1.05:
        warnings.append(f"Candidate lattice was coarsened to {cands['spacing_m']:.0f} m to respect the "
                        "candidate/memory limit; raise limits or shrink the area for a finer search.")
    # effective BS height from DEM (relative to area mean terrain)
    ground = np.zeros(M); hb_eff = np.full(M, rf.hb_m)
    if dem is not None:
        lon_c, lat_c = proj.to_lonlat(cxy[:, 0], cxy[:, 1])
        ground = dem.elevation(lat_c, lon_c)
        hb_eff = np.clip(rf.hb_m + ground - dem.mean, 5.0, None)

    progress("RSRP tensor", 0.25)
    T = rsrp_tensor(rp, grid, cxy[:, 0], cxy[:, 1], hb_eff)
    best = T.max(0)
    thr_pt = rp.thr_dbm[grid.clutter]; sg_pt = rp.sigma_db[grid.clutter]
    P = ndtr((best - thr_pt[:, None]) / sg_pt[:, None]).astype(np.float32)

    # ---------- greedy coverage ----------
    progress("Greedy coverage", 0.45)
    forced = [i for i, s in enumerate(cands["source"]) if s == "User"] if opt.force_user_sites else []
    sel, gains = greedy_coverage(P, rf.coverage_target_pct / 100.0, forced, opt.max_sites)
    if opt.prune and len(sel) > 2:
        kept = prune(P, sel, rf.coverage_target_pct / 100.0, protected=set(forced))
        gains = [g for s, g in zip(sel, gains) if s in kept]; sel = [s for s in sel if s in kept]
    reasons = {}
    for j, g in zip(sel, gains):
        reasons[j] = ("Coverage", f"Coverage: +{g * cell_area_km2 * scale:.2f} km² newly covered when selected"
                      if g is not None else "Coverage: user site (forced)")
    cov_only = len(sel)
    cov_mean = float(P[:, sel].max(1).mean())
    if cov_mean < rf.coverage_target_pct / 100.0:
        warnings.append(f"Coverage target not reachable with the candidate set: {cov_mean*100:.1f}% achieved "
                        f"(target {rf.coverage_target_pct:g}%).")

    # ---------- capacity relief ----------
    demand_pt = None
    if dem_info and dem_info["demand"] > 0:
        demand_pt = dem_info["demand"] * grid.weight
        progress("Capacity relief", 0.65)
        sel, added = capacity_expand(T, sel, rp, grid, demand_pt, cell_cap, opt.max_sites)
        for j, g in added:
            reasons[j] = ("Capacity", f"Capacity: relieves {g:.2f} {cap_unit} from overloaded cells")
    elif cap.enabled:
        warnings.append("Demand is zero - capacity stage skipped.")

    # ---------- azimuth refinement ----------
    az_off = np.zeros(M)
    if opt.optimize_azimuth and rf.sectors > 1:
        progress("Azimuth refinement", 0.75)
        Psel = {j: ndtr((T[:, :, j].max(0) - thr_pt) / sg_pt) for j in sel}
        for j in sel:
            others = [Psel[k] for k in sel if k != j]
            base = np.max(others, axis=0) if others else np.zeros(grid.n)
            bestv, bestoff = -1, 0.0
            for off in np.arange(0, 120, 10.0):
                t1 = rsrp_tensor(rp, grid, cxy[j:j+1, 0], cxy[j:j+1, 1], hb_eff[j:j+1], [off])
                pj = ndtr((t1.max(0)[:, 0] - thr_pt) / sg_pt)
                v = np.maximum(base, pj).mean() - 1e-6 * off
                if v > bestv:
                    bestv, bestoff = v, off
            az_off[j] = bestoff; Psel[j] = ndtr((rsrp_tensor(rp, grid, cxy[j:j+1, 0], cxy[j:j+1, 1],
                                                              hb_eff[j:j+1], [bestoff]).max(0)[:, 0] - thr_pt) / sg_pt)

    # ---------- final evaluation ----------
    progress("Final evaluation", 0.85)
    sel = list(sel)
    load = cap.max_load_pct / 100.0
    ev = evaluate(rp, grid, cxy[sel], hb_eff[sel], az_off[sel], rf.hb_m, rf.kind, rf.noise_ref_dbm, load,
                  dem, proj, demand_pt, cell_cap if demand_pt is not None else None)
    Ms = len(sel)
    p = ev["p"]
    stats = dict(
        expected_coverage_pct=float(p.mean() * 100),
        design_coverage_pct=float(ev["design_ok"].mean() * 100),
        mean_rsrp_dbm=float(ev["rsrp"].mean()),
        p5_rsrp_dbm=float(np.percentile(ev["rsrp"], 5)),
        terrain_loss_mean_db=float(ev["diff"].mean()),
        n_sites=Ms, n_sites_coverage_stage=cov_only, n_sites_capacity_stage=Ms - cov_only,
        sites_per_km2=Ms / area_km2,
    )
    if rf.kind in ("lte", "nr"):
        sinr = ev["sinr"]
        stats.update(sinr_p5_db=float(np.percentile(sinr, 5)), sinr_median_db=float(np.median(sinr)),
                     sinr_ge_min_pct=float((sinr >= rf.sinr_min_db).mean() * 100))
    if "cell_load" in ev:
        util = ev["cell_load"] / cell_cap
        stats.update(overloaded_cells=int((util > 1.0).sum()), max_cell_util_pct=float(util.max() * 100),
                     mean_cell_util_pct=float(util.mean() * 100))

    if stats.get("overloaded_cells", 0) > 0:
        warnings.append(f"{stats['overloaded_cells']} cell(s) remain above the maximum load "
                        f"(peak {stats['max_cell_util_pct']:.0f}%): no candidate site could relieve them. "
                        "Allow more candidates, add carriers/bandwidth, or accept the overload.")

    # ---------- per-site table ----------
    lon_s, lat_s = proj.to_lonlat(cxy[sel, 0], cxy[sel, 1])
    rows = []
    sector_az = lambda off: [round((opt.az0_deg + off + 360.0 / S_ * s) % 360, 1) for s in range(S_)] if rf.sectors > 1 else []
    for k, j in enumerate(sel):
        served = ev["serv_site"] == k
        util_s = None
        if "cell_load" in ev:
            cl = [ev["cell_load"][s * Ms + k] / cell_cap * 100 for s in range(S_)]
            util_s = max(cl)
        typ, why = reasons.get(j, ("Coverage", "Coverage"))
        cl_k = int(np.bincount(grid.clutter[served], minlength=4).argmax()) if served.any() else default_idx
        rows.append({
            "Site": f"S{k+1:02d}", "Latitude": round(float(lat_s[k]), 6), "Longitude": round(float(lon_s[k]), 6),
            "Type": typ, "Reason": why, "Source": cands["source"][j],
            "Roof height (m)": None if np.isnan(cands["bld_height"][j]) else round(float(cands["bld_height"][j]), 1),
            "Antenna height (m)": rf.hb_m, "Azimuths (deg)": ", ".join(map(str, sector_az(az_off[j]))) or "omni",
            "Downtilt (deg)": round(tilt, 1), "Dominant clutter": S.CLUTTERS[cl_k],
            "Served area (km²)": round(float((p * served).sum() * cell_area_km2 * scale), 3),
            "Served demand": round(float((demand_pt * p * served).sum()), 3) if demand_pt is not None else None,
            "Peak sector load (%)": None if util_s is None else round(util_s, 1),
        })
    sites_df = pd.DataFrame(rows)

    # ---------- result tables ----------
    rad_rows = []
    for env in S.CLUTTERS:
        lb = lbs[env]; r = radii[env]
        pl_edge = lb.mapl
        rad_rows.append({"Clutter": env, "Area (km²)": round(clutter_area[env], 3), "MAPL (dB)": round(lb.mapl, 2),
                         "Limiting link": lb.limiting, "Shadow margin (dB)": round(lb.margin_db, 2),
                         "Penetration (dB)": lb.pen_db, "Path-loss exponent n": round(lb.n_exp, 2),
                         "Radius (km)": round(r, 3), "Site area (km²)": round(D.hex_cell_area_km2(r, rf.sectors), 3),
                         "ISD (km)": round(D.isd_km(r, rf.sectors), 3),
                         "Coverage sites": D.sites_from_area(clutter_area[env], r, rf.sectors) if clutter_area[env] > 0 else 0})
    dimensioning = dict(coverage_sites=n_cov, capacity_sites=n_cap, required_sites=n_req,
                        optimized_sites=Ms, cell_capacity=cell_cap, cell_capacity_unit=cap_unit,
                        site_capacity=site_cap, demand=dem_info, representative_radius_km=r_eff,
                        representative_isd_km=isd_eff, tilt_deg=tilt)
    lon_g, lat_g = proj.to_lonlat(grid.x, grid.y)
    grid_df = pd.DataFrame(dict(lat=lat_g, lon=lon_g, clutter=[S.CLUTTERS[i] for i in grid.clutter],
                                rsrp_dbm=ev["rsrp"], p_cov=p, sinr_db=ev["sinr"],
                                serving_site=[f"S{k+1:02d}" for k in ev["serv_site"]],
                                demand=demand_pt if demand_pt is not None else 0.0))
    # raster for the map overlay
    minx = grid.x0 - grid.cell_m / 2; miny = grid.y0 - grid.cell_m / 2
    maxx = minx + grid.nx * grid.cell_m; maxy = miny + grid.ny * grid.cell_m
    cl, ca = proj.to_lonlat([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
    raster = dict(ix=grid.ix, iy=grid.iy, nx=grid.nx, ny=grid.ny,
                  bounds=[[float(ca.min()), float(cl.min())], [float(ca.max()), float(cl.max())]])
    meta = dict(area_name=area_name, model=model, frequency_mhz=rf.freq_mhz, technology=rf.technology,
                bandwidth_mhz=rf.bw_mhz, grid_cell_m=grid.cell_m, grid_points=grid.n, candidates=M,
                dem_used=dem is not None, buildings_used=bxy is not None,
                n_buildings=0 if buildings is None else int(len(buildings)), tilt_deg=tilt)
    if dem is None:
        warnings.append("No DEM used: terrain effects (effective antenna height, diffraction) are ignored.")
    if bxy is None:
        warnings.append(f"No building data: clutter fixed to '{rf.default_clutter}' and demand spread uniformly.")
    if cap.enabled and cap.use_density:
        warnings.append("Population density is a USER INPUT (demo default 5000/km² if untouched) - replace it with "
                        "CAPMAS / WorldPop / GHSL data; capacity results scale linearly with it.")
    progress("Done", 1.0)
    return PlanResult(meta, area_km2, lbs, pd.DataFrame(rad_rows), dimensioning, sites_df, grid_df, stats,
                      warnings, raster, mapping(geom_ll))
