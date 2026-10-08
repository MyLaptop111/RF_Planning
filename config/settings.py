"""Global settings, constants and technology presets.

NOTE: every number marked "assumption" is a typical planning default, NOT a
measured value. Replace with operator / vendor / drive-test figures.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
CACHE_DIR = DATA_DIR / "cache"
for _d in (PROJECTS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Nominatim / Overpass policies require an identifying User-Agent.
USER_AGENT = "RF-Planner/3.0 (RF planning tool; put-your-email-here)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OPENTOPODATA_URL = "https://api.opentopodata.org/v1/srtm30m"

THERMAL_NOISE_DBM_HZ = -174.0          # kT at 290 K (3GPP convention)
CLUTTERS = ["Dense Urban", "Urban", "Suburban", "Rural/Open"]

# Per-clutter defaults (assumptions). sigma = log-normal shadowing std-dev.
CLUTTER_DEFAULTS = {
    "Dense Urban": dict(sigma_db=8.0, pen_indoor=20.0),
    "Urban":       dict(sigma_db=8.0, pen_indoor=16.0),
    "Suburban":    dict(sigma_db=8.0, pen_indoor=12.0),
    "Rural/Open":  dict(sigma_db=8.0, pen_indoor=8.0),
}
SERVICE_LEVELS = ["Outdoor", "In-vehicle", "Indoor"]
PEN_IN_VEHICLE_DB = 8.0                # assumption

# Building density (OSM buildings per hectare, 300 m radius) -> clutter class. ASSUMPTION: calibrate.
BLD_PER_HA_THRESHOLDS = {"Dense Urban": 35.0, "Urban": 15.0, "Suburban": 4.0}

LTE_RB = {1.4: 6, 3.0: 15, 5.0: 25, 10.0: 50, 15.0: 75, 20.0: 100}                # 3GPP TS 36.101
NR_RB_30KHZ = {5.0: 11, 10.0: 24, 15.0: 38, 20.0: 51, 25.0: 65, 30.0: 78, 40.0: 106,
               50.0: 133, 60.0: 162, 70.0: 189, 80.0: 217, 90.0: 245, 100.0: 273}  # TS 38.101-1

TECHNOLOGIES = {
    "4G LTE": dict(kind="lte", default_freq=1800.0, default_bw=20.0, bw_options=list(LTE_RB),
                   ptx=43.0, gtx=17.0, ue_nf=7.0, rsrp_min=-110.0, sinr_min=-6.0,
                   ul_bw_khz=1800.0, ul_sinr=-3.0, ue_ptx=23.0, bs_nf=3.0, duty=1.0,
                   se={"2x2 MIMO": 1.7, "4x2 MIMO": 1.9, "4x4 MIMO": 2.7}),
    "5G NR": dict(kind="nr", default_freq=3500.0, default_bw=100.0, bw_options=list(NR_RB_30KHZ),
                  ptx=46.0, gtx=17.0, ue_nf=8.0, rsrp_min=-105.0, sinr_min=-6.0,
                  ul_bw_khz=3600.0, ul_sinr=-3.0, ue_ptx=23.0, bs_nf=3.5, duty=0.75,
                  se={"4T4R": 3.0, "16T16R": 4.5, "Massive MIMO 32T32R": 5.5}),
    "3G UMTS": dict(kind="umts", default_freq=2100.0, default_bw=5.0, bw_options=[5.0],
                    ptx=43.0, gtx=17.0, ue_nf=8.0, rsrp_min=-105.0, sinr_min=-20.0,
                    ul_bw_khz=3840.0, ul_sinr=-20.0, ue_ptx=21.0, bs_nf=3.0, duty=1.0,
                    se={"HSPA+ (assumption)": 0.8}),
    "2G GSM": dict(kind="gsm", default_freq=900.0, default_bw=0.2, bw_options=[0.2],
                   ptx=43.0, gtx=15.0, ue_nf=8.0, rsrp_min=-95.0, sinr_min=9.0,
                   ul_bw_khz=200.0, ul_sinr=9.0, ue_ptx=33.0, bs_nf=4.0, duty=1.0,
                   se={"n/a (Erlang-B)": 0.0}),
}


def n_resource_blocks(tech: str, bw_mhz: float) -> int:
    kind = TECHNOLOGIES[tech]["kind"]
    table = LTE_RB if kind == "lte" else NR_RB_30KHZ if kind == "nr" else None
    if table is None:
        return 0
    key = min(table, key=lambda k: abs(k - bw_mhz))
    return table[key]


def n_subcarriers(tech: str, bw_mhz: float) -> int:
    rb = n_resource_blocks(tech, bw_mhz)
    return 12 * rb if rb else 1


def occupied_bw_mhz(tech: str, bw_mhz: float) -> float:
    kind = TECHNOLOGIES[tech]["kind"]
    rb = n_resource_blocks(tech, bw_mhz)
    if kind == "lte":
        return rb * 0.180
    if kind == "nr":
        return rb * 0.360
    return bw_mhz


def default_clutter_params(level: str) -> dict:
    out = {}
    for name, d in CLUTTER_DEFAULTS.items():
        pen = 0.0 if level == "Outdoor" else PEN_IN_VEHICLE_DB if level == "In-vehicle" else d["pen_indoor"]
        out[name] = dict(sigma_db=d["sigma_db"], pen_db=pen)
    return out
