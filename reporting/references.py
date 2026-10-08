"""Factors affecting RF planning and literature references.
References are cited from the author's knowledge: verify edition/page before quoting."""

FACTORS = [
    ("Frequency / band", "Path loss grows ~20 log f (free space) and ~33.9 log f (COST-231); higher bands = smaller cells.", "Yes - model + MAPL"),
    ("BS antenna height & terrain-relative height", "Hata terms -13.82 log hb and (44.9-6.55 log hb) log d.", "Yes - hb, DEM-based effective height"),
    ("UE height", "Mobile antenna correction a(hm).", "Yes"),
    ("Clutter / buildings", "Environment corrections, building density, penetration.", "Partially - OSM density -> 4 clutter classes (thresholds are assumptions)"),
    ("Terrain obstruction", "Diffraction over hills (ITU-R P.526).", "Partially - single knife-edge on SRTM30 for the serving link"),
    ("Penetration loss (outdoor / in-car / indoor)", "Adds 8-20 dB typical.", "Yes - service level + per-clutter values"),
    ("Shadow fading and area coverage probability", "Log-normal sigma 6-8 dB; edge vs area probability (Reudink).", "Yes - margin solved from the area-coverage target"),
    ("Interference (SINR), load, frequency reuse", "Limits capacity and edge rate.", "Partially - SINR map with load factor, no frequency planning/ICIC"),
    ("Sectorization, azimuth, tilt, antenna patterns", "3GPP TR 36.814 3D pattern.", "Yes - pattern, auto tilt, azimuth refinement"),
    ("Traffic / user density and busy-hour demand", "Defines capacity sites.", "Partially - needs your population & usage inputs; OSM density only as spatial weight"),
    ("Bandwidth, MIMO, modulation, spectral efficiency", "Cell throughput = SE x BW.", "Yes - editable SE presets (assumptions)"),
    ("Uplink limitation", "UE power 23 dBm usually makes UL the cell-edge limit.", "Yes - UL budget, binding MAPL"),
    ("Handover overlap / neighbour planning", "Needs 10-20% overlap.", "No"),
    ("Backhaul, site acquisition, power, regulation (NTRA licensing)", "Practical feasibility.", "No - candidate sites are geometric/OSM proposals"),
    ("Fast fading, mobility, MIMO channel", "Link-level performance.", "No"),
]

REFERENCES = [
    "M. Hata, 'Empirical formula for propagation loss in land mobile radio services,' IEEE Trans. Vehicular Technology, vol. VT-29, no. 3, pp. 317-325, 1980.",
    "Y. Okumura et al., 'Field strength and its variability in VHF and UHF land-mobile radio service,' Rev. Elec. Commun. Lab., vol. 16, 1968.",
    "COST Action 231, 'Digital Mobile Radio Towards Future Generation Systems - Final Report,' European Commission, EUR 18957, 1999.",
    "3GPP TR 38.901, 'Study on channel model for frequencies from 0.5 to 100 GHz' (Table 7.4.1-1, UMa).",
    "3GPP TR 36.814, 'Further advancements for E-UTRA physical layer aspects' (antenna pattern, Table A.2.1.1-2).",
    "3GPP TS 36.101 / TS 38.101-1 (channel bandwidth and resource-block tables).",
    "ITU-R P.525 (free-space attenuation); ITU-R P.526 (diffraction); ITU-R P.1546; ITU-R P.1411; ITU-R P.452.",
    "W. C. Jakes (Ed.), Microwave Mobile Communications, Wiley, 1974 (area coverage probability); D. O. Reudink, 'Properties of mobile radio propagation above 400 MHz,' IEEE Trans. Veh. Technol., 1974.",
    "H. Holma and A. Toskala, LTE for UMTS: Evolution to LTE-Advanced, Wiley.",
    "T. S. Rappaport, Wireless Communications: Principles and Practice, Prentice Hall.",
    "S. R. Saunders and A. Aragon-Zavala, Antennas and Propagation for Wireless Communication Systems, Wiley.",
    "A. F. Molisch, Wireless Communications, Wiley-IEEE Press.",
    "J. Laiho, A. Wacker, T. Novosad (Eds.), Radio Network Planning and Optimisation for UMTS, Wiley.",
    "A. R. Mishra, Fundamentals of Cellular Network Planning and Optimisation: 2G/2.5G/3G... Evolution to 4G, Wiley.",
    "ITU-R M.2134 / 3GPP TR 25.912 (spectral-efficiency targets) - for the SE presets, replace with vendor/operator figures.",
]

ASSUMPTIONS = [
    "Per-clutter shadowing sigma (8 dB), penetration losses (8/12/16/20 dB), building-density thresholds, spectral-efficiency presets, UL SINR and busy-hour share are TYPICAL planning assumptions, not measurements.",
    "Hata / COST-231 are statistical macro models valid for d >= 1 km and hb >= 30 m; sub-kilometre cells are extrapolations.",
    "Population density default is a demo value. Replace with official census / WorldPop data.",
    "Terrain handling is a single knife-edge on a 30 m SRTM DEM; buildings are used only for density classification (no per-building diffraction).",
    "Results are planning-level estimates and must be validated by drive tests / a calibrated commercial planning tool before deployment.",
]
