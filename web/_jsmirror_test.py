"""
Mechanical transcription of the JavaScript solver in web/index.html into plain
Python (no numpy/shapely), used ONLY to validate the JS port against the trusted
analytic solver in generate_diagrams.run_sequential.  Not shipped.
"""
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_diagrams as gd

CIRCLE_RES = 1500

def sub(a, b): return [a[0]-b[0], a[1]-b[1]]
def cross(a, b): return a[0]*b[1] - a[1]*b[0]

def makeCircle(r=1.0):
    pts = []
    for i in range(CIRCLE_RES):
        t = 2*math.pi*i/CIRCLE_RES
        pts.append([r*math.cos(t), r*math.sin(t)])
    return pts

def polyArea(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        q = pts[(i+1) % n]
        a += pts[i][0]*q[1] - q[0]*pts[i][1]
    return abs(a)/2

def centroid(pts):
    a = cx = cy = 0.0
    n = len(pts)
    for i in range(n):
        p = pts[i]; q = pts[(i+1) % n]
        f = p[0]*q[1] - q[0]*p[1]
        a += f; cx += (p[0]+q[0])*f; cy += (p[1]+q[1])*f
    if abs(a) < 1e-15:
        mx = sum(p[0] for p in pts)/n; my = sum(p[1] for p in pts)/n
        return [mx, my]
    a *= 0.5
    return [cx/(6*a), cy/(6*a)]

def ccwOpen(pts):
    p = pts[:]
    if len(p) > 1 and p[-1][0] == p[0][0] and p[-1][1] == p[0][1]:
        p = p[:-1]
    a = 0.0
    n = len(p)
    for i in range(n):
        q = p[(i+1) % n]
        a += p[i][0]*q[1] - q[0]*p[i][1]
    if a < 0:
        p = p[::-1]
    return p

def startIndex(V, cp):
    k = 0; best = float('inf')
    for i in range(len(V)):
        d = math.hypot(V[i][0]-cp[0], V[i][1]-cp[1])
        if d < best: best = d; k = i
    if best < 1e-7:
        return V, k
    m = len(V); bd = float('inf'); bi = 0
    for i in range(m):
        a = V[i]; b = V[(i+1) % m]; ab = sub(b, a)
        L2 = ab[0]*ab[0] + ab[1]*ab[1]
        t = 0.0 if L2 == 0 else ((cp[0]-a[0])*ab[0] + (cp[1]-a[1])*ab[1])/L2
        t = max(0.0, min(1.0, t))
        proj = [a[0]+t*ab[0], a[1]+t*ab[1]]
        dist = math.hypot(proj[0]-cp[0], proj[1]-cp[1])
        if dist < bd: bd = dist; bi = i
    NV = V[:bi+1] + [cp] + V[bi+1:]
    return NV, bi+1

def solveCut(remaining, curPt, targetRight):
    V = ccwOpen(remaining)
    V, s = startIndex(V, curPt)
    m = len(V)
    V = V[s:] + V[:s]
    V0 = V[0]
    A = [0.0, 0.0]
    for i in range(1, m-1):
        d1 = sub(V[i], V0); d2 = sub(V[i+1], V0)
        A.append(A[-1] + 0.5*cross(d1, d2))
    total = A[m-1]
    tr = targetRight
    if not (0 < tr < total):
        tr = min(max(tr, total*1e-12), total*(1-1e-12))
    lo, hi = 1, m-1
    while hi - lo > 1:
        mid = (lo+hi) >> 1
        if A[mid] <= tr: lo = mid
        else: hi = mid
    j = lo
    Vj = V[j]; Vj1 = V[j+1]; edge = sub(Vj1, Vj); d = sub(Vj, V0)
    twice = cross(d, edge)
    t = 0.0 if abs(twice) < 1e-15 else (tr - A[j])/(0.5*twice)
    t = max(0.0, min(1.0, t))
    ep = [Vj[0]+t*edge[0], Vj[1]+t*edge[1]]
    right = [V0] + V[1:j+1] + [ep]
    left = [V0, ep] + V[j+1:]
    return ep, right, left

def runSequential(n, choices):
    C = makeCircle()
    T = polyArea(C)/n
    sections = []; cuts = []
    remaining = C; curPt = [1, 0]
    def apply(target, takeRight):
        nonlocal remaining, curPt
        ep, right, left = solveCut(remaining, curPt, target)
        sections.append(right if takeRight else left)
        cuts.append([curPt, ep])
        remaining = left if takeRight else right
        curPt = ep
    if n < 2:
        sections.append(remaining)
        return sections, cuts
    apply(T, True)
    for r in choices:
        apply(T if r else (polyArea(remaining)-T), r)
    if n >= 3:
        ep, right, left = solveCut(remaining, curPt, T)
        cuts.append([curPt, ep]); sections.append(right); sections.append(left)
    else:
        sections.append(remaining)
    return sections, cuts

# ── compare JS-mirror vs trusted Python analytic solver ──────────────────────
def compare(n, choices):
    js_sec, js_cuts = runSequential(n, choices)
    py = gd.run_sequential(n, list(choices))
    py_sec = py[0]
    T = polyArea(makeCircle()) / n  # match solver's polygon-based T (1500-gon, not π)
    # area exactness (JS)
    js_area_err = max(abs(polyArea(s) - T) for s in js_sec)
    # match sections by sorted centroid
    jc = sorted((round(centroid(s)[0], 6), round(centroid(s)[1], 6)) for s in js_sec)
    pc = sorted((round(s.centroid.x, 6), round(s.centroid.y, 6)) for s in py_sec)
    worst = max(max(abs(a[0]-b[0]), abs(a[1]-b[1])) for a, b in zip(jc, pc)) if len(jc) == len(pc) else 9.99
    # overshoot: every cut endpoint lies on some section boundary? approximate by
    # checking cut endpoints are within the polygon set (cuts come from solver, exact)
    return js_area_err, worst, len(js_sec) == len(py_sec)

# ── transcription of sectionColors() for adjacency-colouring validation ──────
REDUCED = ['#AED6F1','#A9DFBF','#F9E79F','#F5CBA7','#D7BDE2','#FADBD8']
def onCircle(p): return abs(math.hypot(p[0], p[1]) - 1) < 1e-6
def cutEdges(pts):
    m = len(pts); out = []
    for k in range(m):
        a = pts[k]; b = pts[(k+1) % m]
        if onCircle(a) and onCircle(b) and math.hypot(b[0]-a[0], b[1]-a[1]) < 0.01:
            continue
        out.append((a, b))
    return out
def _cross(u, v): return u[0]*v[1] - u[1]*v[0]
def segAdjacent(e1, e2):
    (a1, a2), (b1, b2) = e1, e2
    d = (a2[0]-a1[0], a2[1]-a1[1]); dd = d[0]*d[0] + d[1]*d[1]
    if dd < 1e-18: return False
    L = math.sqrt(dd)
    for b in (b1, b2):
        if abs(_cross(d, (b[0]-a1[0], b[1]-a1[1]))) / L > 1e-7: return False
    t = lambda p: ((p[0]-a1[0])*d[0] + (p[1]-a1[1])*d[1]) / dd
    tb = sorted([t(b1), t(b2)]); lo = max(0.0, tb[0]); hi = min(1.0, tb[1])
    return (hi - lo) * L > 1e-6
def section_adjacency(sections):
    edges = [cutEdges(p) for p in sections]
    adj = [set() for _ in sections]
    for i in range(len(sections)):
        for j in range(i+1, len(sections)):
            if any(segAdjacent(e1, e2) for e1 in edges[i] for e2 in edges[j]):
                adj[i].add(j); adj[j].add(i)
    return adj
REDUCED = ['#AED6F1','#A9DFBF','#F9E79F','#F5CBA7','#D7BDE2','#FADBD8']
def _corner_adj(sections):
    def vkey(p): return (round(p[0], 4), round(p[1], 4))
    cv = [set(vkey(a) for e in cutEdges(s) for a in e) for s in sections]
    adj = [set() for _ in sections]
    for i in range(len(sections)):
        for j in range(i+1, len(sections)):
            if cv[i] & cv[j]: adj[i].add(j); adj[j].add(i)
    return adj
def section_colors(sections):
    hard = section_adjacency(sections)              # edge-sharing (hard)
    soft = _corner_adj(sections)                    # corner-touch (soft preference)
    S = len(sections); P = len(REDUCED)
    col = [-1]*S
    for i in range(S):                              # pass 1: lowest-index on hard
        forb = {col[j] for j in hard[i] if col[j] >= 0}
        c = 0
        while c in forb: c += 1
        col[i] = c
    for _ in range(6):                              # pass 2: hard forbid + soft prefer + least-used
        use = [0]*P
        for c in col:
            if c < P: use[c] += 1
        for i in range(S):
            hforb = {col[j] for j in hard[i]}
            sforb = {col[j] for j in soft[i]}
            best, bestkey = -1, (9, float('inf'))
            for c in range(P):
                if c in hforb: continue
                k = (1 if c in sforb else 0, use[c])
                if k < bestkey: best, bestkey = c, k
            use[col[i]] -= 1; col[i] = best; use[best] += 1
    return col, hard

def _true_adj(sec):
    """Ground-truth adjacency via Shapely shared-boundary length (not the JS method)."""
    from shapely.geometry import Polygon
    polys = [Polygon(s) for s in sec]
    adj = [set() for _ in sec]
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if polys[i].intersection(polys[j]).length > 1e-7:
                adj[i].add(j); adj[j].add(i)
    return adj

def check_colouring(n, choices):
    sec, _ = runSequential(n, choices)
    col, _ = section_colors(sec)
    true_adj = _true_adj(sec)                       # validate colours against TRUTH
    bad = sum(1 for i in range(len(sec)) for j in true_adj[i] if j > i and col[i] == col[j])
    maxc = max(col) + 1 if col else 0
    return bad, maxc

print('--- adjacency colouring (no neighbour shares a colour) ---')
worst_colors = 0
for n, cs in [(6,'RRR'),(6,'LLL'),(9,'RRRRLR'),(16,'R'*13),(20,'LLLLRLRRRLLLRLRRL'),(43,'R'*40)]:
    ch = [c == 'R' for c in cs]
    bad, maxc = check_colouring(n, ch)
    worst_colors = max(worst_colors, maxc)
    print(f'n={n:>3} {cs[:14]:<14} colours-used={maxc}  neighbour-clashes={bad}  {"OK" if bad==0 else "CLASH"}')

import random
rng = random.Random(0)
clash_total = 0
for _ in range(40):
    n = rng.choice([8,12,20,30,43])
    ch = [rng.random() < 0.5 for _ in range(n-3)]
    bad, maxc = check_colouring(n, ch)
    worst_colors = max(worst_colors, maxc)
    clash_total += bad
print(f'40 random configs: {clash_total} neighbour-clashes; max colours ever used = {worst_colors} (palette has {len(REDUCED)})')

rng = random.Random(0)
cases = [(6,'RRR'),(6,'LLL'),(6,'RLR'),(9,'RRRRLR'),(9,'RRRLLR'),(16,'R'*13),(20,'LLLLRLRRRLLLRLRRL'),(43,'R'*40)]
worst_all = 0.0
for n, cs in cases:
    ch = [c == 'R' for c in cs]
    ae, w, samelen = compare(n, ch)
    worst_all = max(worst_all, w)
    flag = 'OK' if (ae < 1e-9 and w < 1e-4 and samelen) else 'DIFF'
    print(f'n={n:>3} {cs[:14]:<14} JS area-err={ae:.1e}  vs-Py centroid={w:.1e}  {flag}')
# random sample
bad = 0
for _ in range(20):
    n = rng.choice([8,12,20,30])
    ch = [rng.random() < 0.5 for _ in range(n-3)]
    ae, w, samelen = compare(n, ch)
    if not (ae < 1e-9 and w < 1e-3 and samelen):
        print(f'  SAMPLE DIFF n={n}: area-err={ae:.1e} centroid={w:.1e}'); bad += 1
print(f'random sample: {20-bad}/20 consistent;  worst fixed-case centroid diff = {worst_all:.1e}')
