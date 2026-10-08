"""Candidate-site generation: hex lattice + tall OSM buildings + user sites."""
from __future__ import annotations
import numpy as np
import shapely
from scipy.spatial import cKDTree


def hex_lattice(poly_xy, spacing_m, buffer_m=0.0):
    area = poly_xy.buffer(buffer_m) if buffer_m else poly_xy
    minx, miny, maxx, maxy = area.bounds
    dy = spacing_m * np.sqrt(3) / 2
    pts = []
    for j, y in enumerate(np.arange(miny - dy, maxy + dy, dy)):
        off = spacing_m / 2 if j % 2 else 0.0
        xs = np.arange(minx - spacing_m + off, maxx + spacing_m, spacing_m)
        pts.append(np.column_stack([xs, np.full_like(xs, y)]))
    p = np.vstack(pts)
    return p[shapely.contains_xy(area, p[:, 0], p[:, 1])]


def build_candidates(poly_xy, spacing_m, max_candidates, bld_xy=None, bld_h=None,
                     tall_m=18.0, user_xy=None, buffer_m=None):
    """Returns dict(xy (M,2), source list[str], bld_height (M,))."""
    buffer_m = spacing_m if buffer_m is None else buffer_m
    for _ in range(30):
        lat = hex_lattice(poly_xy, spacing_m, buffer_m)
        if len(lat) <= max_candidates:
            break
        spacing_m *= 1.1
    xy = [lat]; src = ["Lattice"] * len(lat); hh = [np.full(len(lat), np.nan)]
    if bld_xy is not None and len(bld_xy) and bld_h is not None:
        tall = np.where(np.asarray(bld_h) >= tall_m)[0]
        if len(tall):
            order = tall[np.argsort(-np.asarray(bld_h)[tall])]
            kept = []; tree_pts = []
            min_sep = spacing_m * 0.5
            for i in order:
                p = bld_xy[i]
                if not shapely.contains_xy(poly_xy.buffer(buffer_m), p[0], p[1]):
                    continue
                if tree_pts and np.min(np.hypot(*(np.array(tree_pts) - p).T)) < min_sep:
                    continue
                kept.append(i); tree_pts.append(p)
                if len(kept) >= max(20, max_candidates // 3):
                    break
            if kept:
                xy.append(bld_xy[kept]); src += ["OSM tall building"] * len(kept); hh.append(np.asarray(bld_h)[kept])
    if user_xy is not None and len(user_xy):
        xy.append(np.asarray(user_xy)); src += ["User"] * len(user_xy); hh.append(np.full(len(user_xy), np.nan))
    return dict(xy=np.vstack(xy), source=src, bld_height=np.concatenate(hh), spacing_m=spacing_m)
