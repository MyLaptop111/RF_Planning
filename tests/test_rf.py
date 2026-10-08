"""Run:  python -m pytest -q   (from the RF_Planner folder)  or  python tests/test_rf.py"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from rf_engine import pathloss as P, link_budget as LB, dimensioning as D
from rf_engine.geometry import circle_polygon, polygon_area_km2, haversine_km
from rf_engine.diffraction import knife_edge_loss_db, terrain_loss_db
from rf_engine.geometry import Projector
from rf_engine.models import RFConfig
from data_sources.dem import DEM


def test_hata_reference_values():
    assert abs(float(P.okumura_hata_db(900, 50, 1.5, 5)) - 146.94) < 0.05      # hand-computed
    assert abs(float(P.cost231_hata_db(1800, 30, 1.5, 1)) - 136.20) < 0.05
    assert abs(float(P.fspl_db(1800, 1)) - 97.55) < 0.05                       # 32.44+20log1800


def test_inverse_roundtrip():
    for m, f in [("Okumura-Hata", 900), ("COST-231 Hata", 1800), ("3GPP TR 38.901 UMa", 3500), ("Free Space", 1800)]:
        d = 0.8
        pl = float(P.path_loss_db(m, f, 30, 1.5, d, "Urban"))
        assert abs(P.solve_distance_km(m, f, 30, 1.5, pl, "Urban") - d) < 1e-3


def test_validity_warnings():
    assert P.validity_warnings("Okumura-Hata", 1800, 30, 1.5)          # f out of range
    assert not P.validity_warnings("COST-231 Hata", 1800, 30, 1.5, 2.0)
    assert P.validity_warnings("COST-231 Hata", 2600, 30, 1.5)


def test_area_probability():
    f0 = LB.area_coverage_probability(0.0, 8, 3.5)
    assert 0.7 < f0 < 0.8                                                  # > 50% edge value
    m = LB.edge_margin_for_area_prob(0.95, 8, 3.5)
    assert abs(LB.area_coverage_probability(m, 8, 3.5) - 0.95) < 1e-6
    assert LB.area_coverage_probability(5, 8, 3.5) > LB.area_coverage_probability(0, 8, 3.5)


def test_budget_binding_and_radius_monotonic():
    cfg = RFConfig()
    lb = LB.build_link_budgets(cfg)
    assert lb["Urban"].mapl <= min(lb["Urban"].mapl_dl, lb["Urban"].mapl_ul) + 1e-9
    r = [LB.radius_km(cfg, lb[e], e) for e in ("Dense Urban", "Urban", "Suburban", "Rural/Open")]
    assert r == sorted(r)


def test_geometry_area():
    a = polygon_area_km2(circle_polygon(30.6, 31.5, 5.0))
    assert abs(a - math.pi * 25) / (math.pi * 25) < 1e-3
    assert abs(haversine_km(30, 31, 31, 31) - 111.2) < 0.2


def test_hex_factors():
    assert abs(D.hex_cell_area_km2(1.0, 1) - 2.598) < 1e-3
    assert abs(D.hex_cell_area_km2(1.0, 3) - 1.9486) < 1e-3
    assert abs(D.isd_km(1.0, 3) - 1.5) < 1e-9


def test_erlang_b():
    assert abs(D.erlang_b(5.0, 10) - 0.01838) < 1e-4                       # standard table value
    assert abs(D.erlang_capacity(10, 0.01838) - 5.0) < 0.01


def test_knife_edge():
    assert knife_edge_loss_db(np.array([0.0]))[0] > 5.9 and knife_edge_loss_db(np.array([0.0]))[0] < 6.1  # ~6 dB at grazing
    assert knife_edge_loss_db(np.array([-1.0]))[0] == 0.0


def test_terrain_blocking():
    proj = Projector(30.6, 31.5)
    lats = np.linspace(30.55, 30.65, 40); lons = np.linspace(31.45, 31.55, 40)
    z = np.zeros((40, 40)); z[:, 20] = 80.0                                # a 80 m ridge across the path
    dem = DEM(lats, lons, z)
    tx = np.array([[-1500.0, 0.0]]); rx = np.array([[1500.0, 0.0]])
    loss = terrain_loss_db(dem, proj, tx, 30.0, rx, 1.5, 1800)
    assert loss[0] > 10
    flat = DEM(lats, lons, np.zeros((40, 40)))
    assert terrain_loss_db(flat, proj, tx, 30.0, rx, 1.5, 1800)[0] == 0.0


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
