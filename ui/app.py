"""RF Planner V3 - Streamlit UI.   Run:  streamlit run ui/app.py   (or: python main.py)"""
import sys, json, io, time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
try:                                   # pandas>=3 string dtype can break the Arrow decoder in some Streamlit builds
    pd.options.future.infer_string = False
except Exception:
    pass
import streamlit as st
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape

from config import settings as S
from rf_engine.models import RFConfig, CapacityConfig, OptSettings
from rf_engine.pathloss import MODELS, resolve_model, validity_warnings, path_loss_db
from rf_engine.link_budget import (build_link_budgets, radius_km, noise_floor_dbm, rsrp_threshold_dbm)
from rf_engine import dimensioning as D
from rf_engine.geometry import (circle_polygon, geojson_to_geometry, polygon_area_km2)
from data_sources.geocode import search_place
from data_sources.buildings import fetch_buildings
from data_sources.dem import fetch_dem
from optimization.optimizer import run_plan
from map_engine.map_view import base_map, add_draw_tools, build_result_map
from reporting.report import build_zip, build_excel, _kml
from reporting.references import FACTORS, REFERENCES, ASSUMPTIONS

st.set_page_config(page_title="RF Planner V3", layout="wide")


def show_df(df):
    """Display a DataFrame safely: plain object/float columns only (no mixed None/number, no pandas string dtype)."""
    d = pd.DataFrame(df).copy()
    for c in d.columns:
        col = d[c]
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            d[c] = pd.to_numeric(col, errors="coerce").astype("float64")
        else:
            d[c] = col.astype(object).where(col.notna(), "").astype(str)
    d.columns = [str(c) for c in d.columns]
    st.dataframe(d, width="stretch", hide_index=True)
st.title("📡 RF Planning & Dimensioning Tool — V3")

ss = st.session_state
ss.setdefault("area", None)
ss.setdefault("sites", [])
ss.setdefault("result", None)
ss.setdefault("places", [])

# =====================================================================  SIDEBAR
sb = st.sidebar
sb.header("📍 Planning Area")
mode = sb.radio("Area source", ["Place name (OSM)", "Draw on map", "Upload GeoJSON", "Center + radius"])
if mode == "Place name (OSM)":
    q = sb.text_input("Place", "Zagazig, Egypt")
    cc = sb.text_input("Country code filter (optional)", "eg")
    if sb.button("🔎 Search boundary"):
        try:
            ss.places = search_place(q, 5, cc or None)
            if not ss.places:
                sb.warning("No results.")
        except Exception as e:
            sb.error(f"Nominatim request failed: {e}")
    if ss.places:
        labels = [f"{p['name'][:60]} [{p['cls']}]" + ("" if p["has_polygon"] else " (point only)") for p in ss.places]
        i = sb.selectbox("Result", range(len(ss.places)), format_func=lambda k: labels[k])
        p = ss.places[i]
        if p["has_polygon"]:
            ss.area = dict(name=q, geom=p["geometry"])
        else:
            rad = sb.number_input("Point only - radius (km)", 0.5, 50.0, 3.0, 0.5)
            ss.area = dict(name=q, geom=circle_polygon(p["lat"], p["lon"], rad))
elif mode == "Upload GeoJSON":
    up = sb.file_uploader("GeoJSON (Polygon/MultiPolygon)", type=["geojson", "json"])
    if up:
        try:
            g = geojson_to_geometry(json.load(up))
            ss.area = dict(name=up.name, geom=g) if g is not None else None
            if g is None:
                sb.error("No polygon found in the file.")
        except Exception as e:
            sb.error(f"Invalid GeoJSON: {e}")
elif mode == "Center + radius":
    lat0 = sb.number_input("Latitude", value=30.5877, format="%.6f")
    lon0 = sb.number_input("Longitude", value=31.5020, format="%.6f")
    rad0 = sb.number_input("Radius (km)", 0.5, 50.0, 3.0, 0.5)
    ss.area = dict(name=f"Circle {lat0:.4f},{lon0:.4f} r={rad0}km", geom=circle_polygon(lat0, lon0, rad0))
else:
    sb.caption("Draw a polygon/rectangle on the map in the 'Area & Map' tab.")

sb.header("RF Parameters")
technology = sb.selectbox("Technology", list(S.TECHNOLOGIES))
T = S.TECHNOLOGIES[technology]
model = sb.selectbox("Propagation Model", MODELS, help="Auto: Hata <=1500, COST-231 1500-2000, 3GPP UMa above 2000 MHz.")
freq = sb.number_input("Frequency (MHz)", 100.0, 6000.0, T["default_freq"], 10.0, key=f"f_{technology}")
if len(T["bw_options"]) > 1:
    bw = sb.selectbox("Bandwidth (MHz)", T["bw_options"], index=T["bw_options"].index(T["default_bw"]), key=f"bw_{technology}")
else:
    bw = sb.number_input("Bandwidth (MHz)", 0.1, 100.0, T["default_bw"], key=f"bw_{technology}")
tx_power = sb.number_input("Tx Power (dBm, total)", value=T["ptx"], step=1.0, key=f"p_{technology}")
tx_gain = sb.number_input("Tx Antenna Gain (dBi)", value=T["gtx"], step=1.0, key=f"g_{technology}")
rx_gain = sb.number_input("Rx Antenna Gain (dBi)", value=0.0, step=1.0)
hb = sb.number_input("BS Antenna Height (m)", 1.0, 200.0, 30.0, 1.0)
hm = sb.number_input("UE Height (m)", 0.5, 20.0, 1.5, 0.5)
sectors = sb.radio("Sectors per site", [3, 1], format_func=lambda s: "3 sectors (120°)" if s == 3 else "Omni")
target = sb.slider("Coverage target (% of area)", 80.0, 99.0, 95.0, 0.5)

sb.header("Link Budget Losses")
cable_loss = sb.number_input("Cable/Feeder Loss (dB)", value=2.0, step=0.5)
body_loss = sb.number_input("Body Loss (dB)", value=0.0, step=0.5)
service = sb.selectbox("Service level (penetration)", S.SERVICE_LEVELS, index=0,
                       help="Outdoor = 0 dB (same as V2 default). Indoor/In-vehicle use the per-clutter values below.")
shadow_mode = sb.radio("Shadowing margin", ["Auto", "Manual"], horizontal=True,
                       help="Auto solves the edge margin so the AREA coverage probability equals the target (Reudink/Jakes).")
shadow_manual = sb.number_input("Manual shadowing margin (dB)", value=8.0, step=1.0, disabled=shadow_mode == "Auto")
cov_mode = sb.selectbox("Coverage criterion", ["RSRP threshold", "SINR sensitivity"])
rsrp_min = sb.number_input("Min RSRP / rx level (dBm)", value=T["rsrp_min"], step=1.0, key=f"r_{technology}",
                           disabled=cov_mode != "RSRP threshold")
sinr_min = sb.number_input("Min SINR (dB)", value=T["sinr_min"], step=0.5, key=f"s_{technology}",
                           disabled=cov_mode == "RSRP threshold")
ue_nf = sb.number_input("UE noise figure (dB)", value=T["ue_nf"], step=0.5, key=f"nf_{technology}")
dl_im = sb.number_input("DL interference margin (dB) [SINR mode]", value=3.0, step=0.5)

with sb.expander("Uplink budget"):
    inc_ul = st.checkbox("Include uplink (binding MAPL = min(DL, UL))", True)
    ue_ptx = st.number_input("UE Tx power (dBm)", value=T["ue_ptx"], key=f"uep_{technology}")
    ue_g = st.number_input("UE antenna gain (dBi)", value=0.0)
    bs_nf = st.number_input("BS noise figure (dB)", value=T["bs_nf"], key=f"bnf_{technology}")
    ul_bw = st.number_input("UL allocated bandwidth (kHz)", value=T["ul_bw_khz"], key=f"ulb_{technology}")
    ul_sinr = st.number_input("UL required SINR (dB)", value=T["ul_sinr"], key=f"uls_{technology}")
    ul_im = st.number_input("UL interference margin (dB)", value=2.0)

clutter = S.default_clutter_params(service)
with sb.expander("Clutter parameters (assumptions - calibrate)"):
    default_clutter = st.selectbox("Default clutter (when no building data)", S.CLUTTERS, index=1)
    for env in S.CLUTTERS:
        c1_, c2_ = st.columns(2)
        clutter[env]["sigma_db"] = c1_.number_input(f"{env} σ (dB)", value=clutter[env]["sigma_db"], key=f"sg_{env}_{service}")
        clutter[env]["pen_db"] = c2_.number_input(f"{env} pen. (dB)", value=clutter[env]["pen_db"], key=f"pn_{env}_{service}")
    cm_on = st.checkbox("Override COST-231 Cm")
    cm_val = st.number_input("COST-231 Cm (dB)", value=3.0, disabled=not cm_on)

with sb.expander("Capacity & traffic"):
    cap_on = st.checkbox("Enable capacity planning", True)
    use_density = st.radio("Population input", ["Density (per km²)", "Total population"]) == "Density (per km²)"
    pop_density = st.number_input("Population density (/km²)", value=5000.0, step=500.0, disabled=not use_density,
                                  help="DEMO value - replace with CAPMAS / WorldPop / GHSL.")
    pop_total = st.number_input("Total population", value=0.0, step=1000.0, disabled=use_density)
    sub_share = st.number_input("Subscriber share of population (%)", value=30.0, step=1.0)
    if T["kind"] == "gsm":
        erl_sub = st.number_input("Traffic per subscriber (mErl)", value=25.0)
        carriers = st.number_input("Carriers per sector", value=3, min_value=1)
        sig_ts = st.number_input("Signalling timeslots per sector", value=2, min_value=0)
        gos = st.number_input("Blocking (GoS, %)", value=2.0)
        monthly = bh = rate_ovr = se_val = duty = 0.0
        max_load = 100.0
    else:
        se_name = st.selectbox("MIMO / spectral-efficiency preset", list(T["se"]), key=f"se_{technology}")
        se_val = st.number_input("Avg cell spectral efficiency (bps/Hz)", value=float(T["se"][se_name]),
                                 key=f"seval_{technology}_{se_name}")
        duty = st.number_input("DL duty cycle (TDD ratio)", 0.1, 1.0, T["duty"], key=f"duty_{technology}")
        monthly = st.number_input("Monthly data per subscriber (GB)", value=10.0, step=1.0)
        bh = st.number_input("Busy-hour share of daily traffic (%)", value=8.0, step=0.5)
        rate_ovr = st.number_input("Override busy-hour rate per user (kbps, 0 = use above)", value=0.0)
        max_load = st.number_input("Max planned cell load (%)", 10.0, 100.0, 70.0, 5.0)
        erl_sub = carriers = sig_ts = gos = 0

with sb.expander("Optimizer & data sources"):
    cell_m = st.number_input("Grid cell (m)", 25.0, 1000.0, 100.0, 25.0)
    max_pts = st.number_input("Max grid points", 500, 20000, 6000, 500)
    max_cand = st.number_input("Max candidate sites", 50, 2000, 800, 50)
    az0 = st.number_input("Sector azimuth offset (deg)", 0.0, 119.0, 0.0, 10.0)
    opt_az = st.checkbox("Refine azimuths per site", True)
    tilt_auto = st.checkbox("Auto downtilt from cell radius", True)
    tilt_val = st.number_input("Downtilt (deg)", 0.0, 15.0, 4.0, disabled=tilt_auto)
    prune_on = st.checkbox("Prune redundant sites", True)
    force_user = st.checkbox("Force-include my candidate sites", False)
    use_bld = st.checkbox("Use OSM buildings (clutter, density, rooftops)", True)
    use_dem = st.checkbox("Use SRTM DEM (terrain)", True)
    tall_m = st.number_input("Tall building threshold (m)", value=18.0)
    st.caption("Building density thresholds (buildings / ha) - assumptions")
    th = {k: st.number_input(f"≥ {k}", value=v, key=f"th_{k}") for k, v in S.BLD_PER_HA_THRESHOLDS.items()}

# ---- assemble config objects
rf = RFConfig(technology=technology, freq_mhz=freq, bw_mhz=float(bw), ptx_dbm=tx_power, gtx_dbi=tx_gain, grx_dbi=rx_gain,
              hb_m=hb, hm_m=hm, cable_loss_db=cable_loss, body_loss_db=body_loss, ue_nf_db=ue_nf,
              coverage_mode=cov_mode, rsrp_min_dbm=rsrp_min, sinr_min_db=sinr_min, dl_interference_margin_db=dl_im,
              include_uplink=inc_ul, ue_ptx_dbm=ue_ptx, ue_gain_dbi=ue_g, bs_nf_db=bs_nf, ul_bw_khz=ul_bw,
              ul_sinr_min_db=ul_sinr, ul_interference_margin_db=ul_im, model=model, cm_db=cm_val if cm_on else None,
              coverage_target_pct=target, shadow_mode=shadow_mode, shadow_manual_db=shadow_manual, sectors=sectors,
              default_clutter=default_clutter, clutter=clutter)
cap = CapacityConfig(enabled=cap_on, use_density=use_density, pop_density_km2=pop_density, population_total=pop_total,
                     subscriber_share_pct=sub_share, monthly_gb_per_user=monthly, busy_hour_share_pct=bh,
                     user_rate_override_kbps=rate_ovr, se_bps_hz=se_val, duty_dl=duty, max_load_pct=max_load,
                     erl_per_sub_me=erl_sub, carriers_per_sector=int(carriers), signalling_ts=int(sig_ts), gos_pct=gos)
opt = OptSettings(cell_m=cell_m, max_grid_points=int(max_pts), max_candidates=int(max_cand), az0_deg=az0,
                  optimize_azimuth=opt_az, tilt_deg=None if tilt_auto else tilt_val, force_user_sites=force_user,
                  prune=prune_on, use_buildings=use_bld, use_dem=use_dem, thresholds=th, tall_building_m=tall_m)

# =====================================================================  LIVE BUDGET
lbs = build_link_budgets(rf)
used_model = resolve_model(rf.model, rf.freq_mhz)
ref_lb = lbs[default_clutter]
noise = noise_floor_dbm(rf.bw_mhz * 1e6, rf.ue_nf_db)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("MAPL (binding)", f"{ref_lb.mapl:.1f} dB", ref_lb.limiting, delta_color="off")
c2.metric("Noise Floor", f"{noise:.1f} dBm")
c3.metric("Technology", technology)
c4.metric("Frequency", f"{freq:.0f} MHz")
c5.metric("Model", used_model)
for w in validity_warnings(used_model, freq, hb, hm):
    st.warning(w)

tab_map, tab_lb, tab_dim, tab_res, tab_pl, tab_ref = st.tabs(
    ["📏 Area & Map", "📶 Link Budget & Radius", "📐 Site Dimensioning", "🛰 Optimization & Results",
     "🧮 Path Loss / RSRP Calculator", "📚 Factors & References"])

# =====================================================================  TAB: MAP
with tab_map:
    area = ss.area
    center = (30.5877, 31.5020); zoom = 12
    if area is not None:
        c = area["geom"].centroid; center = (c.y, c.x); zoom = 12
    m = base_map(center, zoom)
    add_draw_tools(m)
    if area is not None:
        folium.GeoJson(json.loads(json.dumps(area["geom"].__geo_interface__)), name="Planning area",
                       style_function=lambda f: dict(color="#1f3b73", weight=2, fillOpacity=0.05)).add_to(m)
    for i, (lat, lon) in enumerate(ss.sites, 1):
        folium.Marker([lat, lon], tooltip=f"Candidate Site {i}", icon=folium.Icon(icon="signal", prefix="fa")).add_to(m)
    map_data = st_folium(m, height=560, width=None, returned_objects=["last_clicked", "all_drawings"], key="areamap")
    if mode == "Draw on map" and map_data and map_data.get("all_drawings"):
        polys = [d for d in map_data["all_drawings"] if d["geometry"]["type"] == "Polygon"]
        if polys:
            ss.area = dict(name="Drawn polygon", geom=geojson_to_geometry(polys[-1]))
            area = ss.area
    if area is not None:
        st.success(f"Area: **{area['name']}** — {polygon_area_km2(area['geom']):.2f} km² "
                   "(equal-area projection, exact polygon area)")
    else:
        st.info("Select or draw a planning area (sidebar).")
    colA, colB = st.columns(2)
    with colA:
        if map_data and map_data.get("last_clicked"):
            p = map_data["last_clicked"]
            if st.button("➕ Add Candidate Site at last click"):
                ss.sites.append((p["lat"], p["lng"])); st.rerun()
    with colB:
        if st.button("🗑 Clear Candidate Sites"):
            ss.sites = []; st.rerun()
    if ss.sites:
        show_df([{"Site": i, "Latitude": round(p[0], 6), "Longitude": round(p[1], 6)}
                      for i, p in enumerate(ss.sites, 1)])

# =====================================================================  TAB: LINK BUDGET
with tab_lb:
    rows = []
    for env in S.CLUTTERS:
        lb = lbs[env]
        r = radius_km(rf, lb, env)
        rows.append({"Clutter": env, "MAPL DL (dB)": round(lb.mapl_dl, 1),
                     "MAPL UL (dB)": None if not np.isfinite(lb.mapl_ul) else round(lb.mapl_ul, 1),
                     "Binding": lb.limiting, "Shadow margin (dB)": round(lb.margin_db, 2), "n": round(lb.n_exp, 2),
                     "Radius (km)": round(r, 3), "Site area (km²)": round(D.hex_cell_area_km2(r, sectors), 3),
                     "ISD (km)": round(D.isd_km(r, sectors), 3)})
    show_df(pd.DataFrame(rows))
    env_sel = st.selectbox("Show full link budget for", S.CLUTTERS, index=S.CLUTTERS.index(default_clutter))
    show_df(pd.DataFrame(lbs[env_sel].rows))
    st.caption("Shadowing margin (Auto) is solved so that the area coverage probability equals the target "
               "(Reudink/Jakes formula) using the local path-loss exponent n of the selected model.")

# =====================================================================  TAB: DIMENSIONING
with tab_dim:
    area_default = polygon_area_km2(ss.area["geom"]) if ss.area is not None else 25.0
    area_km2 = st.number_input("Planning Area (km²)", min_value=0.01, value=float(round(area_default, 2)), step=1.0,
                               key=f"dimarea_{round(area_default, 2)}")
    r_auto = radius_km(rf, ref_lb, default_clutter)
    use_manual_r = st.checkbox("Override cell radius manually", False)
    radius_use = st.number_input("Cell radius (km)", 0.01, 50.0, float(max(round(r_auto, 3), 0.01)), 0.1,
                                 disabled=not use_manual_r) if use_manual_r else r_auto
    n_cov = D.sites_from_area(area_km2, radius_use, sectors)
    dm = D.total_demand(area_km2, rf, cap) if cap_on else None
    cell_cap, cu = D.cell_capacity(rf, cap)
    site_cap = cell_cap * sectors
    n_cap = D.sites_from_capacity(dm["demand"], site_cap) if dm else 0
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Site area", f"{D.hex_cell_area_km2(radius_use, sectors):.3f} km²")
    d2.metric("Coverage Sites", n_cov)
    d3.metric("Capacity Sites", n_cap)
    d4.metric("Required Sites", D.required_sites(n_cov, n_cap))
    if dm:
        st.write(f"Population **{dm['population']:,.0f}** → subscribers **{dm['users']:,.0f}** → demand "
                 f"**{dm['demand']:,.1f} {dm['unit']}** (per user {dm['per_user']*1000 if dm['unit']=='Mbps' else dm['per_user']:.3f} "
                 f"{'kbps' if dm['unit']=='Mbps' else 'Erlang'}); usable capacity per sector "
                 f"**{cell_cap:.2f} {cu}**, per site **{site_cap:.2f} {cu}**.")
    st.info("Formula-based counts ignore boundary effects, terrain and demand clustering. "
            "Use '🛰 Optimization & Results' for the geometry-aware plan.")

# =====================================================================  TAB: RESULTS
def run_pipeline():
    area = ss.area
    geom = area["geom"]
    a_km2 = polygon_area_km2(geom)
    bbox = (geom.bounds[1], geom.bounds[0], geom.bounds[3], geom.bounds[2])
    bld, dem = None, None
    with st.status("Running RF plan…", expanded=True) as status:
        if a_km2 > 150:
            st.warning(f"Large area ({a_km2:.0f} km²): Overpass/DEM downloads may be slow or fail.")
        if opt.use_buildings:
            st.write("Downloading OSM buildings (Overpass)…")
            try:
                bld = fetch_buildings(bbox); st.write(f"✔ {len(bld):,} buildings")
                if len(bld) == 0:
                    bld = None
            except Exception as e:
                st.warning(f"Buildings unavailable ({e}). Continuing without clutter/density data.")
        if opt.use_dem:
            st.write("Downloading SRTM30 DEM (OpenTopoData)…")
            try:
                dem = fetch_dem(bbox); st.write("✔ DEM ready")
            except Exception as e:
                st.warning(f"DEM unavailable ({e}). Continuing without terrain.")
        bar = st.progress(0.0)
        res = run_plan(geom, rf, cap, opt, bld, dem, ss.sites, area["name"],
                       progress=lambda msg, f=None: (bar.progress(f or 0.0, text=msg)))
        html = build_result_map(res, ss.sites).get_root().render()
        ss.result = dict(res=res, html=html, rf=rf, cap=cap, opt=opt, stamp=time.strftime("%Y%m%d_%H%M%S"))
        pdir = S.PROJECTS_DIR / f"{ss.result['stamp']}_{''.join(ch if ch.isalnum() else '_' for ch in area['name'])[:30]}"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "RF_Plan_Output.zip").write_bytes(build_zip(res, rf, cap, opt, html))
        status.update(label=f"Done - saved in {pdir.relative_to(S.BASE_DIR)}", state="complete")


with tab_res:
    if ss.area is None:
        st.info("Choose a planning area first.")
    else:
        st.caption(f"Area: {ss.area['name']} — {polygon_area_km2(ss.area['geom']):.2f} km². "
                   f"Target {target:g}% | {technology} {freq:g} MHz | model {used_model}")
        if st.button("🚀 Run RF Plan", type="primary"):
            try:
                run_pipeline()
            except Exception as e:
                st.exception(e)
    R = ss.result
    if R:
        res = R["res"]; dmn = res.dimensioning; stt = res.stats
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Coverage sites (formula)", dmn["coverage_sites"])
        k2.metric("Capacity sites (formula)", dmn["capacity_sites"])
        k3.metric("Recommended sites", dmn["optimized_sites"],
                  f"{stt['n_sites_coverage_stage']} cov + {stt['n_sites_capacity_stage']} cap", delta_color="off")
        k4.metric("Expected coverage", f"{stt['expected_coverage_pct']:.1f}%")
        k5.metric("Design coverage (w/ margin)", f"{stt['design_coverage_pct']:.1f}%")
        if "sinr_p5_db" in stt:
            s1, s2, s3 = st.columns(3)
            s1.metric("SINR median", f"{stt['sinr_median_db']:.1f} dB")
            s2.metric("SINR 5th pct", f"{stt['sinr_p5_db']:.1f} dB")
            s3.metric(f"Area SINR ≥ {rf.sinr_min_db:g} dB", f"{stt['sinr_ge_min_pct']:.1f}%")
        for w in res.warnings:
            st.warning(w)
        st.subheader("Radius & path loss per clutter")
        show_df(res.radius_table)
        st.subheader("Recommended sites")
        show_df(res.sites)
        st.subheader("Coverage map")
        if hasattr(st, "iframe"):
            st.iframe(R["html"], height=620)
        else:                                   # older Streamlit versions
            components.html(R["html"], height=620, scrolling=False)
        st.subheader("Download")
        d_a, d_b, d_c, d_d = st.columns(4)
        d_a.download_button("⬇ Full package (ZIP)", build_zip(res, R["rf"], R["cap"], R["opt"], R["html"]),
                            f"RF_Plan_{R['stamp']}.zip", "application/zip")
        d_b.download_button("⬇ Excel report", build_excel(res, R["rf"], R["cap"], R["opt"]),
                            f"RF_Plan_{R['stamp']}.xlsx")
        d_c.download_button("⬇ Sites CSV", res.sites.to_csv(index=False).encode("utf-8-sig"), "recommended_sites.csv")
        d_d.download_button("⬇ Sites KML", _kml(res.sites), "recommended_sites.kml")

# =====================================================================  TAB: PATH LOSS CALC
with tab_pl:
    distance_km = st.number_input("Distance (km)", min_value=0.001, value=1.0, step=0.1)
    env_c = st.selectbox("Environment", S.CLUTTERS, index=S.CLUTTERS.index(default_clutter), key="plenv")
    for w in validity_warnings(used_model, freq, hb, hm, distance_km):
        st.warning(w)
    pl = float(path_loss_db(used_model, freq, hb, hm, distance_km, env_c, rf.cm_db))
    pen = clutter[env_c]["pen_db"]
    rx = rf.p_ref_dbm + tx_gain - cable_loss + rx_gain - body_loss - pen - pl
    st.write(f"**Path Loss:** {pl:.2f} dB  |  **Received level / RSRP proxy (incl. penetration {pen:g} dB):** {rx:.2f} dBm  "
             f"|  threshold {rsrp_threshold_dbm(rf):.1f} dBm")

# =====================================================================  TAB: REFERENCES
with tab_ref:
    st.subheader("Factors affecting RF planning")
    show_df(pd.DataFrame(FACTORS, columns=["Factor", "Effect", "Handled in this tool"]))
    st.subheader("Assumptions & limitations")
    for a in ASSUMPTIONS:
        st.markdown(f"- {a}")
    st.subheader("References")
    st.caption("Cited from knowledge - verify editions/pages before formal citation.")
    for r_ in REFERENCES:
        st.markdown(f"- {r_}")

st.caption("V3: OSM buildings → DEM → COST-231/Hata/UMa → RSRP → coverage probability → greedy site optimisation "
           "→ capacity relief → recommended sites. Planning-level estimates only.")
