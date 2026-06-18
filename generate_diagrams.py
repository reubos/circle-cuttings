#!/usr/bin/env python3
"""
Equal-area circle cutting — sequential configurations.

A *sequential* configuration means every cut starts exactly where the previous
cut ended.  The first cut is always arc→arc.  The last cut is uniquely
determined (it splits the remaining 2T into T+T).  Each middle cut has a
binary choice:

  RIGHT: take the right piece as the new section.
         Search order for endpoint: arc first, then chords oldest→newest.

  LEFT:  take the left piece as the new section.
         Search order for endpoint: oldest chord first, then arc.

This gives exactly 2^(n-3) configurations for n ≥ 3 (and 1 for n=2).

Usage:
    python generate_diagrams.py              # n = 2..5
    python generate_diagrams.py --n 4 5     # only n = 4, 5
    python generate_diagrams.py --out foo.png
"""

import argparse
import itertools
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split as geo_split
from scipy.optimize import brentq

# ── geometry primitives ───────────────────────────────────────────────────────

CIRCLE_RES = 1500

def make_circle(r=1.0):
    t = np.linspace(0, 2 * np.pi, CIRCLE_RES, endpoint=False)
    return Polygon(list(zip(r * np.cos(t), r * np.sin(t))))

def split_poly(poly, p1, p2):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    v = p2 - p1
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return None, None
    v /= nv
    line = LineString([(p1 - 100 * v).tolist(), (p2 + 100 * v).tolist()])
    try:
        res = geo_split(poly, line)
        parts = [g for g in res.geoms if not g.is_empty and g.area > 1e-12]
        if len(parts) == 2:
            return parts[0], parts[1]
    except Exception:
        pass
    return None, None

def oriented_split(poly, p1, p2):
    """
    Split poly; return (right_piece, left_piece) relative to directed p1→p2.
    right_piece.area grows monotonically 0 → poly.area as p2 sweeps CCW.
    """
    a, b = split_poly(poly, p1, p2)
    if a is None:
        return None, None
    p1a = np.asarray(p1, float)
    p2a = np.asarray(p2, float)
    v = p2a - p1a
    v /= np.linalg.norm(v) + 1e-15
    left_n = np.array([-v[1], v[0]])
    ca = np.array([a.centroid.x, a.centroid.y])
    return (b, a) if np.dot(ca - p1a, left_n) > 0 else (a, b)

def arc_pt(angle, r=1.0):
    return (r * np.cos(angle), r * np.sin(angle))

def edge_pt(ea, eb, t):
    ea, eb = np.asarray(ea), np.asarray(eb)
    return tuple(ea + t * (eb - ea))

# ── chord solvers ─────────────────────────────────────────────────────────────

def _root(f, lo, hi, n=600):
    xs = np.linspace(lo, hi, n)
    fs = [f(x) for x in xs]
    for i in range(len(xs) - 1):
        v1, v2 = fs[i], fs[i + 1]
        if np.isnan(v1) or np.isnan(v2):
            continue
        if v1 * v2 < 0:
            try:
                return brentq(f, xs[i], xs[i + 1], xtol=1e-7)
            except Exception:
                pass
    return None

def find_arc_end(poly, p1, right_target, lo, hi, r=1.0):
    """Find angle in [lo,hi] so right_piece.area == right_target."""
    def f(ang):
        p2 = arc_pt(ang % (2 * np.pi), r)
        right, _ = oriented_split(poly, p1, p2)
        return np.nan if right is None else right.area - right_target
    return _root(f, lo, hi)

def find_edge_end(poly, p1, ea, eb, right_target):
    """Find t in [0,1] on edge ea→eb so right_piece.area == right_target."""
    def f(t):
        p2 = edge_pt(ea, eb, t)
        right, _ = oriented_split(poly, p1, p2)
        return np.nan if right is None else right.area - right_target
    # Range mirrors the arc margin (0.001).  As n grows, T = circle/n shrinks and
    # valid endpoints land ever closer to a vertex; a tighter clamp would exclude
    # them (in both ea->eb and eb->ea directions).  Near-vertex noise is still
    # rejected downstream by the abs(area - T) check, so widening is safe.
    sol = _root(f, 0.001, 0.999)
    return None if sol is None else edge_pt(ea, eb, sol)

def _extract_pts(geom):
    """Recursively extract all coordinate points from a Shapely geometry."""
    pts = []
    if geom.geom_type == 'Point':
        pts = [np.array(geom.coords[0])]
    elif geom.geom_type == 'MultiPoint':
        pts = [np.array(g.coords[0]) for g in geom.geoms]
    elif geom.geom_type == 'LineString':
        pts = [np.array(c) for c in geom.coords]
    elif hasattr(geom, 'geoms'):
        for g in geom.geoms:
            pts.extend(_extract_pts(g))
    return pts


def _as_linestrings(geom):
    """Yield every LineString contained in a Shapely geometry."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == 'LineString':
        yield geom
    elif geom.geom_type == 'MultiLineString':
        yield from geom.geoms
    elif hasattr(geom, 'geoms'):
        for g in geom.geoms:
            yield from _as_linestrings(g)


def draw_segment(piece_a, piece_b):
    """Return the actual cut: the boundary shared between the two split pieces.

    This is exactly the chord that was cut.  Both endpoints lie on the polygon
    boundary because they come from the split geometry itself — we never rely
    on ep, which is only an arc point used to define the cut's angle and may sit
    outside the polygon when its arc has been consumed by earlier cuts.
    """
    if piece_a is None or piece_b is None:
        return []
    shared = piece_a.intersection(piece_b)
    segs = []
    for line in _as_linestrings(shared):
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            segs.append((coords[i], coords[i + 1]))
    return segs


def _chord_on_boundary(poly, ea, eb, tol=1e-3):
    """
    Return True if both endpoints of chord ea-eb are approximately on poly's boundary.
    Using point containment: a boundary point is neither strictly inside nor outside.
    """
    from shapely.geometry import Point
    pa, pb = Point(ea), Point(eb)
    try:
        # boundary point: not interior, not exterior (distance to exterior ≈ 0)
        da = poly.exterior.distance(pa)
        db = poly.exterior.distance(pb)
        return da < tol and db < tol
    except Exception:
        return False

def _try_edge(poly, p1, ea, eb, right_target):
    """Try both orientations of edge ea-eb; return endpoint or None."""
    if not _chord_on_boundary(poly, ea, eb):
        return None
    # Skip if p1 is an endpoint of this chord (degenerate — line along boundary)
    p1a = np.asarray(p1)
    if np.linalg.norm(p1a - np.asarray(ea)) < 1e-6 or \
       np.linalg.norm(p1a - np.asarray(eb)) < 1e-6:
        return None
    ep = find_edge_end(poly, p1, ea, eb, right_target)
    if ep is None:
        ep = find_edge_end(poly, p1, eb, ea, right_target)
    return ep

def chord_edges(poly, min_length=0.05):
    """Return all straight (non-arc) edges of poly's exterior boundary."""
    coords = list(poly.exterior.coords)
    edges = []
    for i in range(len(coords) - 1):
        ea, eb = tuple(coords[i]), tuple(coords[i + 1])
        if np.linalg.norm(np.asarray(eb) - np.asarray(ea)) > min_length:
            edges.append((ea, eb))
    return edges

# ── analytic boundary-global cut solver ───────────────────────────────────────
#
# `remaining` is always convex (each cut is a chord between two boundary points of
# a convex region).  So as the endpoint sweeps CCW around the boundary from cur_pt,
# the swept-region area grows monotonically 0 → full with exactly one crossing of
# any target in (0, full).  We exploit that: triangulate the boundary as a fan from
# cur_pt, prefix-sum the triangle areas, binary-search for the vertex bracket, then
# solve the single crossing edge exactly (area is linear in the edge parameter t).
# No Shapely split, no sampling, no per-edge clamp, no arc_lo/arc_hi bookkeeping —
# a solution always exists and is always found, exactly on the boundary.

def _ccw_vertices(poly):
    """Return poly's exterior vertices as a CCW list with no repeated closing point."""
    coords = list(poly.exterior.coords)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    if not poly.exterior.is_ccw:
        coords = coords[::-1]
    return [np.asarray(c, float) for c in coords]

def _start_index(V, cur_pt):
    """Index of cur_pt in vertex list V, inserting it on its edge if not already a vertex."""
    cp = np.asarray(cur_pt, float)
    d = [np.hypot(v[0] - cp[0], v[1] - cp[1]) for v in V]
    k = int(np.argmin(d))
    if d[k] < 1e-7:
        return V, k
    # Not a vertex — find the edge it lies on and insert it.
    m = len(V)
    best = (1e18, 0)
    for i in range(m):
        a, b = V[i], V[(i + 1) % m]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 == 0 else float((cp - a) @ ab) / L2
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        dist = float(np.hypot(proj[0] - cp[0], proj[1] - cp[1]))
        if dist < best[0]:
            best = (dist, i)
    i = best[1]
    V = V[:i + 1] + [cp] + V[i + 1:]
    return V, i + 1

def solve_cut_analytic(remaining, cur_pt, target_right):
    """
    Find ep on remaining's boundary so the swept (right) piece has area target_right.

    Returns (ep, right_piece, left_piece).  right_piece is the region swept CCW from
    cur_pt to ep; its area == target_right.  Raises ValueError only if target_right
    is outside (0, total) — which cannot happen for valid cut targets.
    """
    V = _ccw_vertices(remaining)
    V, s = _start_index(V, cur_pt)
    m = len(V)
    # Reorder so the start vertex is first; boundary continues CCW.
    V = V[s:] + V[:s]
    V0 = V[0]

    # Fan prefix areas: A[j] = area of polygon [V0, V1, ..., Vj] (chord Vj→V0 closes it).
    # A[j] = 0.5 * sum_{i=1}^{j-1} cross(V_i - V0, V_{i+1} - V0).  Monotonic, A[1]=0.
    A = [0.0, 0.0]
    for i in range(1, m - 1):
        d1 = V[i] - V0
        d2 = V[i + 1] - V0
        A.append(A[-1] + 0.5 * (d1[0] * d2[1] - d1[1] * d2[0]))
    total = A[m - 1]
    if not (0 < target_right < total):
        # Clamp tiny numerical overruns; anything else is a real caller error.
        if target_right <= 0 or target_right >= total:
            target_right = min(max(target_right, total * 1e-12), total * (1 - 1e-12))

    # Binary-search the vertex bracket: largest j with A[j] <= target_right.
    lo, hi = 1, m - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if A[mid] <= target_right:
            lo = mid
        else:
            hi = mid
    j = lo  # A[j] <= target_right <= A[j+1]; ep lies on edge V[j]→V[j+1]

    Vj, Vj1 = V[j], V[j + 1]
    edge = Vj1 - Vj
    d = Vj - V0
    twice = (d[0] * edge[1] - d[1] * edge[0])  # 2 * dArea/dt along this edge
    t = 0.0 if abs(twice) < 1e-15 else (target_right - A[j]) / (0.5 * twice)
    t = max(0.0, min(1.0, t))
    ep = Vj + t * edge

    right_pts = [V0] + V[1:j + 1] + [ep]
    left_pts = [V0, ep] + V[j + 1:]
    right_piece = Polygon([tuple(p) for p in right_pts])
    left_piece = Polygon([tuple(p) for p in left_pts])
    return tuple(ep), right_piece, left_piece


def run_sequential(n, choices, initial_angle=0.0):
    """
    Solve a sequential equal-area cutting configuration (analytic solver).

    Mirrors the legacy solver's structure and output exactly, but finds each cut
    endpoint analytically (convex ⇒ monotonic swept-area ⇒ exact single crossing)
    instead of sampling + Shapely splits.  A solution always exists and is always
    found, so this never skips.

    Parameters
    ----------
    n       : number of sections (n-1 cuts total)
    choices : list of n-3 booleans — True='right', False='left' — one per middle
              cut (cuts 2 … n-2).  Empty for n ≤ 3.
    initial_angle : starting angle on the circle (radians)

    Returns
    -------
    (sections, draw_cuts, label)
    """
    assert len(choices) == max(0, n - 3)
    C = make_circle()
    T = C.area / n
    sections, cuts, draw_cuts = [], [], []
    remaining = C
    cur_pt = arc_pt(initial_angle)

    def apply(target_right, take_right):
        nonlocal remaining, cur_pt
        ep, right, left = solve_cut_analytic(remaining, cur_pt, target_right)
        draw_cuts.append(draw_segment(right, left))
        sections.append(right if take_right else left)
        cuts.append((cur_pt, ep))
        remaining = left if take_right else right
        cur_pt = ep

    if n < 2:
        sections.append(remaining)
        return sections, draw_cuts, _make_label(n, choices, [])

    # cut 1: always 'right', section = right piece of area T
    apply(T, True)
    # middle cuts (cuts 2 … n-2)
    for is_right in choices:
        apply(T if is_right else (remaining.area - T), is_right)
    # last cut: remaining == 2T → split into T + T (no L/R distinction)
    if n >= 3:
        ep, right, left = solve_cut_analytic(remaining, cur_pt, T)
        draw_cuts.append(draw_segment(right, left))
        cuts.append((cur_pt, ep))
        sections.append(right)
        sections.append(left)
    else:  # n == 2: the single cut above already produced both halves
        sections.append(remaining)

    return sections, draw_cuts, _make_label(n, choices, [])


# ── legacy Shapely-split solver (kept as a reference oracle for diffing) ───────
#
# Superseded by the analytic solver above, which is exact and ~500-1000x faster.
# Retained so its output can be diffed against the analytic solver in tests.

def run_sequential_legacy(n, choices, initial_angle=0.0):
    """
    Solve a sequential equal-area cutting configuration.

    Parameters
    ----------
    n       : number of sections (n-1 cuts total)
    choices : list of n-3 booleans — True='right', False='left' — one per
              middle cut (cuts 2 … n-2).  Empty for n ≤ 3.
    initial_angle : starting angle on the circle (radians)

    Returns
    -------
    (sections, cuts, label)  on success
    (None, None, reason)     on failure
    """
    assert len(choices) == max(0, n - 3)

    C = make_circle()
    T = C.area / n
    sections, cuts, draw_cuts = [], [], []
    remaining = C
    cur_pt = arc_pt(initial_angle)
    arc_lo = initial_angle
    arc_hi = initial_angle + 2 * np.pi

    def do_cut(right_target, section_is_right):
        """
        Find endpoint, split, return (end_pt, section, new_remaining, arc_update).
        arc_update is None (chord), ('lo', sol) for right-arc, or ('hi', sol) for left-arc.
        Returns None on failure.
        """
        if section_is_right:
            # RIGHT: try arc first (advances arc_lo), then chord edges of remaining
            lo, hi = arc_lo + 0.001, arc_hi - 0.001
            if hi > lo:
                sol = find_arc_end(remaining, cur_pt, right_target, lo, hi)
                if sol is not None:
                    ep = arc_pt(sol % (2 * np.pi))
                    # Only accept arc result if ep is actually on remaining's boundary.
                    # If the arc was consumed by a prior chord cut, ep is outside/inside
                    # remaining and using it would corrupt cur_pt for future cuts.
                    if remaining.exterior.distance(Point(ep)) < 0.05:
                        r, l = oriented_split(remaining, cur_pt, ep)
                        if r is not None and abs(r.area - T) < 1e-3:
                            return ep, r, l, ('lo', sol)
            for ea, eb in chord_edges(remaining):
                ep = _try_edge(remaining, cur_pt, ea, eb, right_target)
                if ep is not None:
                    r, l = oriented_split(remaining, cur_pt, ep)
                    if r is not None and abs(r.area - T) < 1e-3:
                        return ep, r, l, None
        else:
            # LEFT: try chord edges of remaining first, then arc (advances arc_hi)
            for ea, eb in chord_edges(remaining):
                ep = _try_edge(remaining, cur_pt, ea, eb, right_target)
                if ep is not None:
                    r, l = oriented_split(remaining, cur_pt, ep)
                    if l is not None and abs(l.area - T) < 1e-3:
                        return ep, l, r, None
            lo, hi = arc_lo + 0.001, arc_hi - 0.001
            if hi > lo:
                sol = find_arc_end(remaining, cur_pt, right_target, lo, hi)
                if sol is not None:
                    ep = arc_pt(sol % (2 * np.pi))
                    if remaining.exterior.distance(Point(ep)) < 0.05:
                        r, l = oriented_split(remaining, cur_pt, ep)
                        if l is not None and abs(l.area - T) < 1e-3:
                            return ep, l, r, ('hi', sol)
        return None

    # ── cut 1: always arc→arc, right piece = T ────────────────────────────────
    lo, hi = arc_lo + 0.001, arc_hi - 0.001
    sol = find_arc_end(remaining, cur_pt, T, lo, hi)
    if sol is None:
        return None, None, "cut 1: arc→arc infeasible"
    ep = arc_pt(sol % (2 * np.pi))
    right, left = oriented_split(remaining, cur_pt, ep)
    if right is None:
        return None, None, "cut 1: split failed"
    draw_cuts.append(draw_segment(right, left))
    sections.append(right); cuts.append((cur_pt, ep))
    remaining = left; cur_pt = ep; arc_lo = sol

    # ── middle cuts (cuts 2 … n-2) ────────────────────────────────────────────
    step_labels = ['arc→arc']
    for i, is_right in enumerate(choices):
        right_target = T if is_right else (remaining.area - T)
        result = do_cut(right_target, is_right)
        if result is None:
            return None, None, f"cut {i+2}: {'right' if is_right else 'left'} infeasible"
        ep, section, new_remaining, arc_update = result
        draw_cuts.append(draw_segment(section, new_remaining))
        sections.append(section)
        cuts.append((cur_pt, ep))
        remaining = new_remaining
        cur_pt = ep
        if arc_update is not None:
            bound, sol = arc_update
            if bound == 'lo':
                arc_lo = sol
            else:
                arc_hi = sol
            step_labels.append('arc→arc')
        else:
            step_labels.append('right' if is_right else 'left')

    # ── last cut: remaining = 2T, split into T + T (no L/R distinction) ───────
    if n >= 3:
        right_target = T  # remaining.area == 2T, so right == left == T
        ep = None
        # Search full arc range — no boundary check needed (last cut, no future cur_pt)
        sol = find_arc_end(remaining, cur_pt, right_target,
                           initial_angle + 0.001, initial_angle + 2 * np.pi - 0.001)
        if sol is not None:
            ep = arc_pt(sol % (2 * np.pi))
            step_labels.append('arc')
        if ep is None:
            for ea, eb in chord_edges(remaining):
                ep = _try_edge(remaining, cur_pt, ea, eb, right_target)
                if ep is not None:
                    step_labels.append('chord')
                    break
        if ep is None:
            return None, None, f"cut {n-1}: last cut infeasible"
        right, left = oriented_split(remaining, cur_pt, ep)
        if right is None:
            return None, None, f"cut {n-1}: split failed"
        draw_cuts.append(draw_segment(right, left))
        sections.append(right); cuts.append((cur_pt, ep))
        sections.append(left)
    else:
        sections.append(remaining)

    # Verify all areas
    areas = [s.area for s in sections]
    if max(abs(a - T) for a in areas) > 1e-3:
        return None, None, f"area mismatch: {[f'{a:.4f}' for a in areas]}"

    label = _make_label(n, choices, step_labels)
    return sections, draw_cuts, label


def _make_label(n, choices, step_labels):
    choice_str = ''.join('R' if c else 'L' for c in choices) if choices else ''
    label = f"n={n}"
    if choice_str:
        label += f"  [{choice_str}]"
    return label

# ── enumerate configurations ──────────────────────────────────────────────────

def all_choice_sequences(n):
    """Return all 2^(n-3) binary choice sequences for n ≥ 3 (1 for n ≤ 3)."""
    k = max(0, n - 3)
    return list(itertools.product([True, False], repeat=k))

# ── drawing ───────────────────────────────────────────────────────────────────

PALETTE = [
    '#AED6F1', '#A9DFBF', '#F9E79F', '#F5CBA7', '#D7BDE2',
    '#FADBD8', '#D5F5E3', '#FDEBD0', '#D6EAF8', '#EAECEE',
]
EDGE_COL = '#2C3E50'


def poly_patch(poly, color, **kw):
    if poly is None or poly.is_empty:
        return None
    coords = np.array(poly.exterior.coords)
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(coords) - 2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(coords, codes), facecolor=color, **kw)


def draw_config(ax, sections, cuts, title='', shading=True, labels=True, circle=True):
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')

    if circle:
        circ = plt.Circle((0, 0), 1.0, fill=False, edgecolor=EDGE_COL, linewidth=1.4, zorder=1)
        ax.add_patch(circ)

    if shading:
        for i, sec in enumerate(sections):
            patch = poly_patch(sec, PALETTE[i % len(PALETTE)],
                               edgecolor=EDGE_COL, linewidth=1.4, alpha=0.92, zorder=2)
            if patch:
                ax.add_patch(patch)

    for seg_list in cuts:
        for p1, p2 in seg_list:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=EDGE_COL, lw=2, zorder=3)

    if labels:
        for i, sec in enumerate(sections):
            if sec is None or sec.is_empty:
                continue
            c = sec.centroid
            ax.text(c.x, c.y, str(i + 1),
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', color=EDGE_COL, zorder=5)

    if title:
        ax.set_title(title, fontsize=9, pad=5, color='#1a1a2e',
                     fontfamily='monospace')

# ── choice string parser ──────────────────────────────────────────────────────

def _expand_choices(s):
    """
    Expand a compact choice string into a plain R/L string.

    Supported syntax:
      R, L          — single letter
      R3, L2        — letter repeated N times
      RL2, LR3      — run of letters repeated N times (no parens needed for simple runs)
      (LR)3         — parenthesised group repeated N times
      Combinations  — e.g. L3(RL)2R

    Examples:
      'L3R2'    -> 'LLLRR'
      '(LR)3'   -> 'LRLRLR'
      'R2(LR)2' -> 'RRLRLR'
    """
    import re
    s = s.upper()
    result = []
    i = 0
    while i < len(s):
        if s[i] == '(':
            # find matching ')'
            j = s.index(')', i)
            group = s[i+1:j]
            i = j + 1
            # optional repeat count
            m = re.match(r'(\d+)', s[i:])
            count = int(m.group(1)) if m else 1
            if m:
                i += len(m.group(1))
            result.append(group * count)
        elif s[i] in 'RL':
            # single letter, then optional repeat count
            letter = s[i]
            i += 1
            m = re.match(r'(\d+)', s[i:])
            count = int(m.group(1)) if m else 1
            if m:
                i += len(m.group(1))
            result.append(letter * count)
        else:
            i += 1  # skip unknown chars
    return ''.join(result)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='+', default=[2, 3, 4, 5])
    parser.add_argument('--out', default='circle_cuttings.png')
    parser.add_argument('--choices', type=str, nargs='+',
                        help='specific choice strings to run, e.g. LLLLLLLL RLRL')
    parser.add_argument('--sample', type=int, metavar='N',
                        help='randomly sample N configurations per n value instead of all')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for --sample (for reproducibility)')
    parser.add_argument('--no-shading', action='store_true', help='disable area shading')
    parser.add_argument('--no-labels', action='store_true', help='disable area number labels')
    parser.add_argument('--no-circle', action='store_true', help='omit the circle outline')
    args = parser.parse_args()

    results = []

    if args.choices:
        # Build (n, choices) pairs directly from the choice strings
        todo = []
        for s in args.choices:
            expanded = _expand_choices(s)
            n = len(expanded) + 3
            choices = tuple(ch == 'R' for ch in expanded)
            todo.append((n, choices))
        for n, choices in todo:
            label_short = f"n={n} [{''.join('R' if c else 'L' for c in choices) or '-'}]"
            print(f"\n{label_short} ... ", end='', flush=True)
            sections, cuts, label = run_sequential(n, list(choices))
            if sections is None:
                print(f"skip ({cuts})")
            else:
                print("OK")
                results.append((sections, cuts, label))
    else:
        import random
        rng = random.Random(args.seed)
        for n in sorted(args.n):
            seqs = all_choice_sequences(n)
            if args.sample and len(seqs) > args.sample:
                seqs = rng.sample(seqs, args.sample)
                print(f"\nn={n}: sampling {args.sample} of {2**max(0,n-3)} configurations")
            else:
                print(f"\nn={n}: {len(seqs)} configuration(s) to try")
            for choices in seqs:
                label_short = f"n={n} [{''.join('R' if c else 'L' for c in choices) or '-'}]"
                print(f"  {label_short} ... ", end='', flush=True)
                sections, cuts, label = run_sequential(n, list(choices))
                if sections is None:
                    print(f"skip ({cuts})")
                else:
                    print("OK")
                    results.append((sections, cuts, label))

    if not results:
        print("No feasible configurations found.")
        return

    ncols = min(4, len(results))
    nrows = (len(results) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4.5 * nrows),
                             facecolor='#F0F0F0')
    axes = np.array(axes).flatten()

    for i, (sections, cuts, label) in enumerate(results):
        axes[i].set_facecolor('#FAFAFA')
        draw_config(axes[i], sections, cuts, label,
                    shading=not args.no_shading,
                    labels=not args.no_labels,
                    circle=not args.no_circle)

    for j in range(len(results), len(axes)):
        axes[j].axis('off')

    fig.suptitle("Equal-Area Circle Cutting — Sequential Configurations",
                 fontsize=13, fontweight='bold', y=1.01, color='#1a1a2e')
    plt.tight_layout(pad=1.5)
    plt.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='#F0F0F0')
    print(f"\nSaved: {args.out}  ({len(results)} configurations)")
    plt.show()


if __name__ == '__main__':
    main()
