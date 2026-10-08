# RF Planner V3 — RF Planning & Dimensioning Tool

Streamlit app that turns an area name (e.g. *Zagazig, Egypt*) plus RF inputs into:
link budget → maximum path loss → cell radius per clutter → coverage/capacity site counts →
RSRP/SINR coverage map → optimised, recommended site coordinates (+ Excel/CSV/KML/GeoJSON/HTML report).

## Run
```
pip install -r requirements.txt
python main.py          # or: streamlit run ui/app.py     (Windows: run.bat, Linux/macOS: ./run.sh)
python -m pytest -q     # unit tests (formulas, inverse solve, Erlang-B, diffraction, geometry)
```
Internet is needed only for: Nominatim (boundary), Overpass (OSM buildings), OpenTopoData (SRTM DEM).
If a download fails the tool continues without it and says so in the warnings. Results are cached in `data/cache`.
Edit `USER_AGENT` in `config/settings.py` (put your e-mail) — required by the Nominatim/Overpass usage policies.

## Pipeline
```
Area (OSM boundary / drawn polygon / GeoJSON / centre+radius)  -> exact area (Lambert equal-area projection)
OSM buildings -> density -> clutter classes (Dense Urban / Urban / Suburban / Rural) + demand weights + tall-building rooftops
SRTM DEM      -> effective antenna height + single knife-edge terrain loss (ITU-R P.526) on serving links
Link budget   -> DL (RSRP or SINR-sensitivity) and UL -> binding MAPL, shadow margin from area-coverage probability
Model         -> Okumura-Hata | COST-231 Hata | 3GPP TR 38.901 UMa | Free space (Auto selects by frequency; validity warnings)
Radius        -> inverse path-loss solve per clutter -> hex site area (omni 2.598R², 3-sector 1.949R²)
RSRP grid     -> 3GPP TR 36.814 sector pattern, auto downtilt, best-server, log-normal coverage probability, SINR
Optimizer     -> greedy max-coverage (soft probability) -> prune -> capacity relief of overloaded cells -> azimuth refinement
Report        -> Excel (inputs, warnings, link budgets, radius, dimensioning, sites, KPIs, factors, references), CSV, KML, GeoJSON, map
```

## Structure
```
main.py  requirements.txt  README.md  run.bat  run.sh
config/settings.py            constants, technology presets, clutter defaults (ALL assumptions flagged)
ui/app.py                     Streamlit app (V2 layout kept: sidebar params, map+draw, dimensioning, path-loss calculator)
map_engine/map_view.py        folium maps, RSRP/SINR/probability overlays
map_engine/site_generator.py  hex-lattice + OSM tall-building + user candidate sites
rf_engine/pathloss.py         propagation models, validity checks, inverse solver
rf_engine/link_budget.py      DL/UL budgets, noise, Reudink area-coverage margin
rf_engine/dimensioning.py     hex geometry, coverage/capacity site counts, Erlang-B, traffic model
rf_engine/coverage.py         grid, clutter, RSRP tensor, evaluation (best server, SINR, load)
rf_engine/antenna.py          3GPP sector pattern
rf_engine/diffraction.py      knife-edge terrain loss
rf_engine/geometry.py         projection, area, haversine, GeoJSON
rf_engine/models.py           dataclasses (RFConfig, CapacityConfig, OptSettings, PlanResult)
data_sources/                 geocode.py (Nominatim), buildings.py (Overpass), dem.py (OpenTopoData)
optimization/optimizer.py     end-to-end planning pipeline
reporting/                    report.py (Excel/CSV/KML/GeoJSON/ZIP), references.py (factors, references, assumptions)
tests/test_rf.py              unit tests
data/projects, data/cache     outputs of each run / downloaded data cache
```

## Accuracy notes — read before trusting numbers
* Formulas are implemented from the published models and unit-tested against hand-computed values.
* **Assumptions (editable, NOT measurements):** shadowing σ, penetration losses, building-density → clutter thresholds,
  spectral-efficiency presets, UL SINR, busy-hour share, 3 m per building level. Calibrate with drive tests.
* **Population is your input.** The 5000/km² default is a demo value; use CAPMAS / WorldPop / GHSL. Capacity results scale with it.
* Hata/COST-231 are valid for d ≥ 1 km, hb ≥ 30 m; LTE urban cells are often < 1 km, so radii are extrapolations (the tool warns).
* Not modelled: frequency planning/ICIC, neighbour lists/handover overlap, per-building diffraction, MIMO channel, backhaul, NTRA licensing.
* Planning-level estimate only; validate with a calibrated commercial tool / drive test before deployment.
