"""Report generation: Excel, CSV, KML, GeoJSON, HTML map, JSON -> ZIP."""
from __future__ import annotations
import io, json, zipfile, datetime
from dataclasses import asdict
import pandas as pd
from reporting.references import FACTORS, REFERENCES, ASSUMPTIONS


def _inputs_df(rf, cap, opt):
    rows = []
    for title, obj in (("RF", rf), ("Capacity", cap), ("Optimizer", opt)):
        for k, v in asdict(obj).items():
            rows.append((title, k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v))
    return pd.DataFrame(rows, columns=["Group", "Parameter", "Value"])


def _kml(sites):
    pm = []
    for _, r in sites.iterrows():
        pm.append(f"<Placemark><name>{r['Site']}</name><description>{r['Type']} - {r['Reason']} | "
                  f"Azimuths: {r['Azimuths (deg)']}</description><Point><coordinates>{r['Longitude']},{r['Latitude']},0"
                  f"</coordinates></Point></Placemark>")
    return ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            + "".join(pm) + "</Document></kml>")


def _geojson(sites):
    feats = [dict(type="Feature", geometry=dict(type="Point", coordinates=[r["Longitude"], r["Latitude"]]),
                  properties={k: (None if pd.isna(v) else v) for k, v in r.items() if k not in ("Latitude", "Longitude")})
             for _, r in sites.iterrows()]
    return json.dumps(dict(type="FeatureCollection", features=feats), ensure_ascii=False, indent=1)


def build_excel(result, rf, cap, opt) -> bytes:
    buf = io.BytesIO()
    summary = {
        "Area": result.meta["area_name"], "Area size (km²)": round(result.area_km2, 3),
        "Technology / frequency": f'{result.meta["technology"]} / {result.meta["frequency_mhz"]:g} MHz',
        "Propagation model": result.meta["model"], "Coverage target (%)": rf.coverage_target_pct,
        "Coverage sites (formula)": result.dimensioning["coverage_sites"],
        "Capacity sites (formula)": result.dimensioning["capacity_sites"],
        "Required sites (max of both, formula)": result.dimensioning["required_sites"],
        "Recommended sites (optimized)": result.dimensioning["optimized_sites"],
        "Expected area coverage (%)": round(result.stats["expected_coverage_pct"], 2),
        "Design coverage with margin (%)": round(result.stats["design_coverage_pct"], 2),
        "Grid cell (m) / points": f'{result.meta["grid_cell_m"]:.0f} / {result.meta["grid_points"]}',
        "DEM used": result.meta["dem_used"], "Buildings used": result.meta["buildings_used"],
        "Generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    stats = pd.DataFrame([(k, v) for k, v in result.stats.items()], columns=["Metric", "Value"])
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame(summary.items(), columns=["Item", "Value"]).to_excel(xw, sheet_name="Summary", index=False)
        _inputs_df(rf, cap, opt).to_excel(xw, sheet_name="Inputs", index=False)
        pd.DataFrame({"Warnings / validity": result.warnings or ["none"]}).to_excel(xw, sheet_name="Warnings", index=False)
        for env, lb in result.link_budgets.items():
            pd.DataFrame(lb.rows).to_excel(xw, sheet_name=("LinkBudget " + env.replace("/", "-"))[:31], index=False)
        result.radius_table.to_excel(xw, sheet_name="Radius & PathLoss", index=False)
        pd.DataFrame([dict(Metric=k, Value=str(v)) for k, v in result.dimensioning.items()]).to_excel(xw, sheet_name="Dimensioning", index=False)
        result.sites.to_excel(xw, sheet_name="Recommended Sites", index=False)
        stats.to_excel(xw, sheet_name="Coverage & KPI", index=False)
        pd.DataFrame(FACTORS, columns=["Factor", "Effect", "Handled in this tool"]).to_excel(xw, sheet_name="RF Factors", index=False)
        pd.DataFrame({"Assumptions": ASSUMPTIONS}).to_excel(xw, sheet_name="Assumptions", index=False)
        pd.DataFrame({"References": REFERENCES}).to_excel(xw, sheet_name="References", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = min(max(len(str(c.value)) if c.value is not None else 0 for c in col) + 2, 70)
                ws.column_dimensions[col[0].column_letter].width = width
    return buf.getvalue()


def build_zip(result, rf, cap, opt, map_html: str) -> bytes:
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("RF_Plan_Report.xlsx", build_excel(result, rf, cap, opt))
        zf.writestr("recommended_sites.csv", result.sites.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("recommended_sites.kml", _kml(result.sites))
        zf.writestr("recommended_sites.geojson", _geojson(result.sites))
        zf.writestr("coverage_grid.csv", result.grid.round(4).to_csv(index=False))
        zf.writestr("coverage_map.html", map_html)
        zf.writestr("radius_table.csv", result.radius_table.to_csv(index=False))
        zf.writestr("inputs.json", json.dumps(dict(rf=asdict(rf), capacity=asdict(cap), optimizer=asdict(opt)),
                                              ensure_ascii=False, indent=1))
    return z.getvalue()
