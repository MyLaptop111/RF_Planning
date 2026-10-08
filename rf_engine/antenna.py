"""3GPP TR 36.814 (Table A.2.1.1-2) 3D sector antenna pattern."""
import numpy as np


def sector_pattern_db(bearing_deg, boresight_deg, theta_deg, tilt_deg,
                      hbw=65.0, vbw=10.0, a_m=30.0, sla_v=20.0):
    """Relative gain (dB, <= 0). theta = angle below horizon of the UE seen from the BS."""
    dphi = (bearing_deg - boresight_deg + 180.0) % 360.0 - 180.0
    ah = -np.minimum(12.0 * (dphi / hbw) ** 2, a_m)
    av = -np.minimum(12.0 * ((theta_deg - tilt_deg) / vbw) ** 2, sla_v)
    return np.maximum(ah + av, -a_m)
