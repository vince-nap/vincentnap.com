#!/usr/bin/env python3
"""
Builds the generic face for the landing page: a neutral head that is nobody in
particular, as a wireframe with sparse dots, bright clusters along the features
and eight tracking points. No photos are used. (--procedural-hair adds generated
long wavy hair; it is not used on the site.)

The face is MediaPipe's canonical face model (an average face, symmetric, in
centimetres). Around it the rest of the head is made of simple solids:

  cranium  a superellipsoid, 15.8 x 21 x 19.5 cm, centred behind the eyes
  jaw      an ellipsoid that fills the space between the jawline and the neck
  ears     two small ellipsoids at the tragus
  neck     an elliptical cylinder, fading out below the chin

A lat-long grid is cast from inside the head against all of them and the face
mesh; whichever surface is furthest out along a ray is the head there. Where the
face mesh is that surface, the face mesh's own triangles take over.

Output, in the same layout as build_face.py: site/assets/face/face.bin and
face.json. Run with the venv from tools/face/README.md:
    python tools/face/build_generic.py
"""
import argparse
import json
import os

import numpy as np
import trimesh
from mediapipe.tasks.python.vision import FaceLandmarksConnections as FLC

from build_face import (HERE, ROOT, K_SKIN, K_NECK, K_IRIS, K_CONTOUR, K_NODE, K_KEY, K_TIP, K_HAIR,
                        L_HAIR, load_obj, poisson_disk, sample_mesh)

# the solids, in the canonical model's centimetres (x right, y up, z towards the viewer)
CRANIUM = dict(c=(0.0, 3.0, -4.25), r=(7.9, 10.6, 9.75), n=2.2)
JAW = dict(c=(0.0, -4.6, -3.0), r=(7.4, 5.2, 6.4), n=2.6)           # boxy: a square jaw
EARS = [dict(c=(s * 7.75, 0.3, -3.7), r=(0.95, 3.0, 1.9), n=2.0) for s in (-1, 1)]
NECK = dict(cx=0.0, cz=-3.6, rx=5.3, rz=5.7, y0=-19.0, y1=0.0)


def exit_superellipsoid(o, d, s):
    """Distance along each ray (origin o, direction d) to where it leaves the solid, or 0 if it misses."""
    c, r, n = np.array(s['c']), np.array(s['r']), s['n']
    lo = np.zeros(len(o))
    hi = np.full(len(o), 60.0)
    f = lambda t: (np.abs((o + d * t[:, None] - c) / r) ** n).sum(1)
    inside0 = f(lo) < 1
    # rays starting outside: march to the first point inside, if any
    t_in = np.zeros(len(o))
    if (~inside0).any():
        ts = np.linspace(0, 60, 600)
        vals = np.stack([f(np.full(len(o), t)) for t in ts], 1)
        hit = vals < 1
        has = hit.any(1)
        t_in = np.where(has, ts[np.argmax(hit, 1)], -1.0)
        lo = np.where(inside0, 0.0, np.maximum(t_in, 0))
        valid = inside0 | has
    else:
        valid = inside0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        ins = f(mid) < 1
        lo = np.where(ins, mid, lo)
        hi = np.where(ins, hi, mid)
    return np.where(valid, lo, 0.0)


def exit_neck(o, d):
    a = (d[:, 0] / NECK['rx']) ** 2 + (d[:, 2] / NECK['rz']) ** 2
    b = 2 * ((o[:, 0] - NECK['cx']) * d[:, 0] / NECK['rx'] ** 2 + (o[:, 2] - NECK['cz']) * d[:, 2] / NECK['rz'] ** 2)
    c = ((o[:, 0] - NECK['cx']) / NECK['rx']) ** 2 + ((o[:, 2] - NECK['cz']) / NECK['rz']) ** 2 - 1
    disc = b * b - 4 * a * c
    t = np.where((a > 1e-9) & (disc > 0), (-b + np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-9)), 0.0)
    y = o[:, 1] + d[:, 1] * t
    return np.where((t > 0) & (y > NECK['y0']) & (y < NECK['y1']), t, 0.0)


def masculine(v, fn):
    """Reshapes MediaPipe's average face into a more masculine one: a square jaw with a
    marked angle and a broader, flatter chin, cheekbones that stand out with hollows
    under them, a slight bump on the bridge of the nose and thinner lips.

    Landmarks are given a displacement in cm (the left side; the right is mirrored)
    and landmarks that must stay put are anchored; a Gaussian radial basis function
    through all of them moves the rest of the mesh smoothly. The lips are then
    pressed together towards the line of the mouth."""
    mirror = lambda i: int(np.argmin(((v - v[i] * [-1, 1, 1]) ** 2).sum(1)))
    ctrl = {}

    def move(i, d):
        ctrl[i] = np.array(d, float)
        if abs(v[i, 0]) > 0.05:
            ctrl[mirror(i)] = np.array(d, float) * [-1, 1, 1]

    # the jaw: the angle out and down, the bottom of the face flattened into a broad chin
    for i, d in [(132, (0.15, 0, 0)), (58, (0.5, -0.2, 0)), (172, (1.0, -0.5, 0)), (136, (0.95, -0.7, 0.05)),
                 (150, (0.9, -0.8, 0.1)), (149, (0.7, -0.65, 0.15)), (176, (0.45, -0.42, 0.2)),
                 (148, (0.25, -0.2, 0.22)), (152, (0, -0.14, 0.22))]:
        move(i, d)
    # cheekbones out and forward, the cheeks under them pulled in
    for i, d in [(116, (0.3, 0.05, 0.3)), (123, (0.35, 0.0, 0.3)), (117, (0.15, 0.05, 0.25))]:
        move(i, d)
    for i, depth in [(187, 0.45), (207, 0.3), (192, 0.35), (213, 0.3), (147, 0.2), (214, 0.15)]:
        move(i, -depth * fn[i])
    # a bump on the bridge of the nose, a third of the way down, and a small dip after it
    for i, dz in [(6, 0.04), (197, 0.2), (195, 0.25), (5, -0.03)]:
        move(i, (0, 0, dz))
    # anchors: the eyes, brows, forehead, the tip and base of the nose, the mouth, the face's edge at the ears
    for i in [33, 133, 159, 145, 70, 105, 107, 10, 151, 109, 168, 4, 1, 2, 98, 61, 0, 17, 13, 14,
              37, 84, 234, 127, 93, 50, 101, 206]:
        if i not in ctrl:
            move(i, (0, 0, 0))

    ids = np.array(list(ctrl))
    disp = np.array([ctrl[i] for i in ids])
    sigma = 1.3
    k = lambda a, b: np.exp(-((a[:, None] - b[None]) ** 2).sum(-1) / (2 * sigma ** 2))
    w = np.linalg.solve(k(v[ids], v[ids]) + 1e-4 * np.eye(len(ids)), disp)
    out = v + k(v, v[ids]) @ w

    # thinner lips: everything near the mouth pressed towards the line between the lips
    y_m = -4.27
    smooth = lambda e0, e1, x: (lambda t: t * t * (3 - 2 * t))(np.clip((x - e0) / (e1 - e0), 0, 1))
    wl = (1 - smooth(2.5, 3.1, np.abs(v[:, 0]))) * (1 - smooth(1.0, 2.0, np.abs(v[:, 1] - y_m)))
    out[:, 1] = y_m + (out[:, 1] - y_m) * (1 - 0.35 * wl)
    out[:, 2] -= 0.18 * wl
    return out


def glasses(eye, top, bottom, face_z, rng, sp=0.055):
    """The glasses, measured on the photo against the eyes, brows and nose: rounded
    square rims from just under the brows to the top of the nose wings, 1.5 cm apart
    over the nose, flat on top and leaning in at the outer bottom; short wings to the
    hinges with two rivets each; a bridge high on the nose; the arms back over the
    ears. Drawn as dense dots like the features, mottled for the tortoiseshell.

    eye: (x, y) of the eye on the +x side; face_z(x, y): depth of the face seen from
    the front (-inf where there is none), so the frame never sinks into the cheeks.
    Returns points, normals and tones."""
    inner, outer = eye[0] - 2.4, eye[0] + 3.5
    cx, cy = (inner + outer) / 2, (top + bottom) / 2
    a, b = (outer - inner) / 2 - 0.17, (top - bottom) / 2 - 0.17

    def band(line, width, closed):
        """Dots filling a band of the given width (per point) around a polyline."""
        nxt = np.roll(line, -1, 0) if closed else np.vstack([line[1:], 2 * line[-1:] - line[-2:-1]])
        seg = nxt - line
        ln = np.linalg.norm(seg, axis=1)
        nrm2 = np.stack([seg[:, 1], -seg[:, 0]], 1) / ln[:, None]
        m = int((ln * width).sum() / sp ** 2 * 2.5)
        i = rng.choice(len(line), m, p=ln * width / (ln * width).sum())
        t, v = rng.random(m), (rng.random(m) - 0.5) * width[i]
        return line[i] + seg[i] * t[:, None] + nrm2[i] * v[:, None]

    def inside(poly):
        """Dots filling a polygon."""
        from matplotlib.path import Path
        lo, hi = poly.min(0), poly.max(0)
        q = lo + rng.random((int(np.prod(hi - lo) / sp ** 2 * 2.5), 2)) * (hi - lo)
        return q[Path(poly).contains_points(q)]

    # one rim, the +x side: a superellipse, squarer on top, the outer side leaning in at the bottom
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    c, s_ = np.cos(th), np.sin(th)
    n = np.where(s_ > 0, 3.6, 2.4)
    x, y = a * np.sign(c) * np.abs(c) ** (2 / n), b * np.sign(s_) * np.abs(s_) ** (2 / n)
    x -= 0.22 * np.clip(-y / b, 0, 1) * np.clip(x / a, 0, 1)
    rim = np.stack([x + cx, y + cy], 1)
    w = 0.26 + 0.16 * np.clip((rim[:, 1] - bottom) / (top - bottom), 0, 1)   # thicker on top
    # the nose comes through between the rims: move the inner side of each rim out
    # until its inner edge clears the side of the nose
    zf = lambda x, y: 5.45 - 0.043 * np.maximum(np.abs(x) - 0.7, 0) ** 2 + 0.14 * (y - cy)
    shift = np.zeros(len(rim))
    near = np.where(rim[:, 0] < cx)[0]
    for dx in np.arange(0, 1.5, 0.04):
        xe = rim[near, 0] - w[near] / 2 + dx
        clear = face_z(xe, rim[near, 1]) + 0.25 <= zf(xe, rim[near, 1])
        todo = (shift[near] == 0) & ~clear
        if not todo.any():
            break
        shift[near[todo]] = dx + 0.04
    shift = np.convolve(np.concatenate([shift[-15:], shift, shift[:15]]), np.ones(31) / 31, 'same')[15:-15]
    rim[:, 0] += np.maximum(shift, 0) * np.clip((cx - rim[:, 0]) / a * 1.5, 0, 1)
    wing = np.array([(outer - 0.5, top), (outer + 0.55, top + 0.08), (outer + 0.68, top - 0.05),
                     (outer + 0.62, top - 0.55), (outer - 0.15, top - 0.95)])
    half = np.vstack([band(rim, w, True), inside(wing)])
    rivets = np.array([(outer + 0.12, top - 0.27), (outer + 0.34, top - 0.25)])
    half = half[np.min(np.linalg.norm(half[:, None] - rivets[None], axis=2), 1) > 0.08]
    bx = np.linspace(-inner - 0.15, inner + 0.15, 40)
    bridge = band(np.stack([bx, np.full(40, top - 0.68)], 1), np.full(40, 0.5), False)
    bridge = bridge[bridge[:, 1] > top - 0.93 + 0.12 * (1 - (bridge[:, 0] / inner) ** 2)]   # arched below
    front = np.vstack([half, half * [-1, 1], bridge])

    # the front wraps back towards the hinges and tilts in at the bottom (zf above)
    fz = np.maximum(zf(front[:, 0], front[:, 1]), face_z(front[:, 0], front[:, 1]) + 0.3)
    f3 = np.column_stack([front, fz])
    gx = -0.086 * np.maximum(np.abs(front[:, 0]) - 0.7, 0) * np.sign(front[:, 0])
    f_n = np.column_stack([-gx, np.full(len(front), -0.14), np.ones(len(front))])
    f_n /= np.linalg.norm(f_n, axis=1, keepdims=True)

    # the arms: from the hinges back over the ears, clear of the cranium
    def head_x(y, z):
        out = np.zeros(len(y))
        for sol in [CRANIUM] + EARS[1:]:
            (_, sy, sz), (rx, ry, rz), sn = sol['c'], sol['r'], sol['n']
            rem = 1 - np.abs((y - sy) / ry) ** sn - np.abs((z - sz) / rz) ** sn
            out = np.maximum(out, np.where(rem > 0, sol['c'][0] + rx * np.clip(rem, 0, 1) ** (1 / sn), 0))
        return out
    hx, hy = outer + 0.62, top - 0.3
    hz = zf(hx, hy)
    zz = np.linspace(hz, -4.3, 160)
    ay = hy - 0.35 * (hz - zz) / (hz + 4.3)
    ax = np.maximum(hx + 0.1 * (hz - zz), head_x(ay, zz) + 0.3)
    arm_line = np.stack([zz, ay], 1)                                  # a band in the (z, y) plane
    arm = band(arm_line, np.linspace(0.42, 0.3, 160), False)
    arm_x = np.interp(-arm[:, 0], -zz, ax)
    a3 = np.column_stack([arm_x, arm[:, 1], arm[:, 0]])
    arms = np.vstack([a3, a3 * [-1, 1, 1]])
    arm_n = np.column_stack([np.sign(arms[:, 0]), np.zeros(len(arms)), np.zeros(len(arms))])

    pts = np.vstack([f3, arms])
    nrm = np.vstack([f_n, arm_n])
    k = poisson_disk(pts, sp, 31)
    pts, nrm = pts[k], nrm[k]
    # tortoiseshell: a few crossing waves, thresholded into darker and lighter patches
    noise = sum(np.sin(pts[:, :2] @ (rng.normal(0, 1, 2) * 2 * np.pi / rng.uniform(0.5, 1.1)) + rng.uniform(0, 6.3))
                for _ in range(4)) / 2
    tone = np.clip(0.66 + 0.2 * np.tanh(2 * noise), 0.4, 0.95)
    return pts, nrm, tone


def sculpted_hair(path, rng):
    """A sculpted hairstyle, fitted onto the cranium and drawn as flow lines and dots.

    The hair's inner cavity (where the scalp would be) is found by taking, for every
    direction from inside the head, the nearest hair vertex; an ellipsoid fitted to
    those points is matched to the cranium with one uniform scale, so the hairstyle
    keeps its own proportions. The lines are evenly spaced streamlines running down
    the outer surface (steepest descent, so they follow the sculpted locks from the
    parting to the tips). Nothing is drawn as a line: the dots are scattered loosely
    along each streamline, so the locks show through the same stipple as the skin,
    with a lighter scatter in between. A coarse copy of the hair hides what lies
    behind it.
    """
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    import fast_simplification

    z = np.load(path)
    v, f = z['V'].astype(float), z['F'].astype(np.int64)
    # the inner cavity, from roughly the middle of the hair's head
    c = np.array([0.0, v[:, 1].min() + 0.68 * np.ptp(v[:, 1]), 0.0])
    c[2] = v[:, 2].min() + 0.53 * np.ptp(v[:, 2])
    d = v - c
    r = np.linalg.norm(d, axis=1)
    u = d / r[:, None]
    pol = np.degrees(np.arccos(np.clip(u[:, 1], -1, 1)))
    az = np.degrees(np.arctan2(u[:, 0], u[:, 2]))
    key = (pol // 6).astype(int) * 100 + ((az + 180) // 8).astype(int)
    order = np.lexsort((r, key))
    first = order[np.r_[True, key[order][1:] != key[order][:-1]]]
    inner = v[first[pol[first] < 95]]
    rin = np.linalg.norm(inner - c, axis=1)
    inner = inner[rin < np.median(rin) * 1.25]

    def res(p):
        q = (inner - [0, p[0], p[1]]) / p[2:]
        return (np.sqrt((q ** 2).sum(1)) - 1) * p[2:].mean()
    cy, cz, a, b, cc = least_squares(res, [c[1], c[2], np.median(rin), np.median(rin), np.median(rin)], loss='soft_l1').x
    s = 0.5 * (CRANIUM['r'][0] / a + CRANIUM['r'][2] / cc) * 1.04       # a few mm of clearance
    top = CRANIUM['c'][1] + CRANIUM['r'][1] + 0.3
    h = np.stack([v[:, 0] * s, (v[:, 1] - (cy + b)) * s + top, (v[:, 2] - cz) * s + CRANIUM['c'][2]], 1)
    print('hair: cavity radii %.1f %.1f %.1f, scale %.4f, bounds %s %s' % (a, b, cc, s, h.min(0).round(1), h.max(0).round(1)))

    hm = trimesh.Trimesh(h, f, process=False)
    hn = hm.vertex_normals.copy()
    axis_pt = lambda p: np.stack([np.zeros(len(p)), np.clip(p[:, 1], -20, CRANIUM['c'][1]), np.full(len(p), CRANIUM['c'][2])], 1)
    if ((h - axis_pt(h)) * hn).sum(1).mean() < 0:
        hn = -hn
    # a dense cloud on the surface, for projecting onto it; only the outer side is drawn
    sp, sn = sample_mesh(h, f, hn, 500000, rng)
    outer = ((sp - axis_pt(sp)) * sn).sum(1) > 0.05 * np.linalg.norm(sp - axis_pt(sp), axis=1)
    sp, sn = sp[outer], sn[outer]
    tree = cKDTree(sp)

    def on_surface(p):
        dist, i = tree.query(p)
        n = sn[i]
        return p - ((p - sp[i]) * n).sum(-1, keepdims=True) * n, n, dist

    # evenly spaced streamlines, from the top down
    dsep, step = 0.5, 0.28
    cell = dsep / 2
    occupied = {}

    def near_other(p, line_id):
        k = np.floor(p / cell).astype(int)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q, lid in occupied.get((k[0] + dx, k[1] + dy, k[2] + dz), ()):
                        if lid != line_id and ((q - p) ** 2).sum() < (0.5 * dsep) ** 2:
                            return True
        return False

    seeds = sp[poisson_disk(sp, 0.7, 21)]
    seeds = seeds[np.argsort(-seeds[:, 1])]
    lines, normals = [], []
    for sid, seed in enumerate(seeds):
        if near_other(seed, -1):
            continue
        p, n, _ = on_surface(seed[None])
        p, n = p[0], n[0]
        pts, nrm, last = [p], [n], None
        for _ in range(170):
            g = np.array([0.0, -1.0, 0.0])
            dvec = g - (g @ n) * n
            if np.linalg.norm(dvec) < 0.3:                               # flat on top: away from the parting
                dvec = dvec + np.array([0.35 * np.sign(p[0] + 1e-6), 0, 0])
                dvec -= (dvec @ n) * n
            dvec /= np.linalg.norm(dvec) + 1e-9
            if last is not None:
                dvec = 0.7 * dvec + 0.3 * last
                dvec /= np.linalg.norm(dvec) + 1e-9
            q, qn, dist = on_surface((p + dvec * step)[None])
            q, qn = q[0], qn[0]
            if dist[0] > 0.35 or (q[1] > p[1] + 0.02) or near_other(q, sid):
                break
            pts.append(q); nrm.append(qn); last, p, n = dvec, q, qn
        if len(pts) >= 6:
            for q in pts:
                k = tuple(np.floor(q / cell).astype(int))
                occupied.setdefault(k, []).append((q, sid))
            lines.append(np.array(pts)); normals.append(np.array(nrm))
    hair_t = []
    for pts in lines:
        sl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        hair_t.append(sl / max(sl[-1], 1e-6))
    print('hair: %d flow lines, %d points' % (len(lines), sum(len(l) for l in lines)))

    # dots: scattered loosely along every flow line (irregular steps, pushed a little
    # to either side, a tone per lock), and a light scatter over the surface in between
    s_dots, s_n, s_tone = [], [], []
    for pts, nrm in zip(lines, normals):
        sl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        at = np.cumsum(rng.uniform(0.1, 0.32, int(sl[-1] / 0.1) + 2))
        at = at[at <= sl[-1]]
        p = np.stack([np.interp(at, sl, pts[:, k]) for k in range(3)], 1)
        n = np.stack([np.interp(at, sl, nrm[:, k]) for k in range(3)], 1)
        tan = np.stack([np.interp(at, sl, np.gradient(pts[:, k])) for k in range(3)], 1)
        side = np.cross(n, tan)
        side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-9
        s_dots.append(p + side * rng.normal(0, 0.07, (len(at), 1)))
        s_n.append(n)
        s_tone.append(np.clip(rng.uniform(0.32, 0.6) + rng.normal(0, 0.06, len(at)), 0, 1))
    k = poisson_disk(sp, 0.24, 22)
    dots = np.concatenate(s_dots + [sp[k]])
    dots_n = np.concatenate(s_n + [sn[k]])
    dots_n /= np.linalg.norm(dots_n, axis=1, keepdims=True) + 1e-9
    tone = np.concatenate(s_tone + [rng.uniform(0.2, 0.36, len(k))])

    # a coarse copy of the hair as an occluder
    ov, of = fast_simplification.simplify(h.astype(np.float32), f.astype(np.int32), target_reduction=1 - 9000 / len(f))
    return lines, normals, hair_t, dots, dots_n, tone, ov.astype(float), of.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=os.path.join(HERE, 'models'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'site', 'assets', 'face'))
    ap.add_argument('--procedural-hair', action='store_true', help='grow the generated hair (not used on the site)')
    ap.add_argument('--hair-mesh', default=os.path.join(HERE, 'input', 'hair_sculpt.npz'),
                    help='sculpted hair from prepare_hair.py (used when the file exists)')
    ap.add_argument('--glasses', action='store_true', help='add the glasses (not used on the site)')
    ap.add_argument('--birthmark', action='store_true', help='add the birthmark by the right nostril (not used on the site)')
    args = ap.parse_args()

    face_v, face_f = load_obj(os.path.join(args.models, 'canonical_face_model.obj'))
    fn = trimesh.Trimesh(face_v, face_f, process=False).vertex_normals.copy()
    if fn[:, 2].mean() < 0:
        fn = -fn
    face_v = masculine(face_v, fn)
    fm = trimesh.Trimesh(face_v, face_f, process=False)
    fn = fm.vertex_normals.copy()
    if fn[:, 2].mean() < 0:
        fn = -fn
    tri_a, tri_b, tri_c = face_v[face_f[:, 0]], face_v[face_f[:, 1]], face_v[face_f[:, 2]]

    def hit_face(o, d):
        """Distance along each ray to the nearest face triangle (Moeller-Trumbore), 0 if none."""
        e1, e2 = tri_b - tri_a, tri_c - tri_a
        out = np.zeros(len(o))
        for s0 in range(0, len(o), 256):
            oo, dd = o[s0:s0 + 256, None], d[s0:s0 + 256, None]
            pv = np.cross(dd, e2[None])
            det = (e1[None] * pv).sum(-1)
            inv = 1.0 / np.where(np.abs(det) < 1e-12, np.nan, det)
            tv = oo - tri_a[None]
            u = (tv * pv).sum(-1) * inv
            qv = np.cross(tv, e1[None])
            v = (dd * qv).sum(-1) * inv
            t = (e2[None] * qv).sum(-1) * inv
            ok = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
            t = np.where(ok, t, np.inf)
            best = t.min(1)
            out[s0:s0 + 256] = np.where(np.isfinite(best), best, 0.0)
        return out

    def surface(o, d):
        """The head surface along each ray: the face where the ray meets it, else the furthest exit over the solids."""
        t = np.max(np.stack([exit_superellipsoid(o, d, CRANIUM), exit_superellipsoid(o, d, JAW),
                             exit_superellipsoid(o, d, EARS[0]), exit_superellipsoid(o, d, EARS[1]),
                             exit_neck(o, d)]), 0)
        tf = hit_face(o, d)
        # where a ray meets the face mesh, the face is the surface (the cranium would
        # otherwise bulge out in front of the eye sockets)
        on_face = tf > 0
        return o + d * np.where(on_face, tf, t)[:, None], on_face

    # 1. the lat-long grid: a dome of rings from the centre of the cranium, then
    # horizontal rings around the neck axis down to the base of the neck
    n_az = 44
    th = np.arange(n_az) * 2 * np.pi / n_az
    c0 = np.array(CRANIUM['c'])
    rings, cover = [], []
    for phi in np.radians(np.linspace(9, 90, 10)):
        d = np.stack([np.sin(phi) * np.sin(th), np.full(n_az, np.cos(phi)), np.sin(phi) * np.cos(th)], 1)
        p, c = surface(np.repeat(c0[None], n_az, 0), d)
        rings.append(p); cover.append(c)
    for y in np.arange(c0[1] - 1.1, NECK['y0'] + 0.5, -1.1):
        z_axis = np.interp(y, [NECK['y0'], -4.0, c0[1]], [NECK['cz'], -3.4, c0[2]])
        o = np.repeat(np.array([[0.0, y, z_axis]]), n_az, 0)
        d = np.stack([np.sin(th), np.zeros(n_az), np.cos(th)], 1)
        p, c = surface(o, d)
        rings.append(p); cover.append(c)
    grid = np.stack(rings)
    covered = np.concatenate(cover)
    crown, _ = surface(c0[None], np.array([[0.0, 1.0, 0.0]]))
    n_r = len(grid)
    t_az = np.roll(grid, -1, 1) - np.roll(grid, 1, 1)
    g_up = np.concatenate([np.repeat(crown[None], n_az, 1), grid[:-1]])
    g_dn = np.concatenate([grid[1:], grid[-1:]])
    g_n = np.cross(g_dn - g_up, t_az)
    g_n /= np.linalg.norm(g_n, axis=2, keepdims=True) + 1e-9
    axis_pt = np.stack([np.zeros_like(grid[..., 0]), grid[..., 1], np.full(grid.shape[:2], -3.8)], -1)
    g_n[((grid - axis_pt) * g_n).sum(-1) < 0] *= -1
    gp, gn = grid.reshape(-1, 3), g_n.reshape(-1, 3)
    gid = np.arange(n_r * n_az).reshape(n_r, n_az)
    quads = np.stack([gid[:-1], np.roll(gid, -1, 1)[:-1], np.roll(gid, -1, 1)[1:], gid[1:]], -1).reshape(-1, 4)
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    crown_id = len(gp)
    gp = np.vstack([gp, crown]); gn = np.vstack([gn, [0, 1, 0]]); covered = np.append(covered, False)
    tris = np.concatenate([tris, np.stack([np.full(n_az, crown_id), gid[0], np.roll(gid[0], -1)], 1)])
    head_tris = tris[~covered[tris].all(1)]

    # the eye outlines (for the irises and two of the tracking points)
    eye_l = {c.start for c in FLC.FACE_LANDMARKS_LEFT_EYE} | {c.end for c in FLC.FACE_LANDMARKS_LEFT_EYE}
    eye_r = {c.start for c in FLC.FACE_LANDMARKS_RIGHT_EYE} | {c.end for c in FLC.FACE_LANDMARKS_RIGHT_EYE}

    # 2. sparse dots: on the (smoothed) face, and on the head grid outside the face
    sm = fm
    for _ in range(2):
        v2, f2 = trimesh.remesh.subdivide_loop(sm.vertices, sm.faces)
        sm = trimesh.Trimesh(v2, f2, process=False)
    sn = sm.vertex_normals.copy()
    if sn[:, 2].mean() < 0:
        sn = -sn

    def dots(verts, faces, normals, radius, seed):
        tri = verts[faces]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
        p, n = sample_mesh(verts, faces, normals, int(area / radius ** 2 * 2.2), np.random.default_rng(seed))
        k = poisson_disk(p, radius, seed)
        return p[k], n[k]

    rng = np.random.default_rng(3)
    p_face, n_face = dots(sm.vertices, sm.faces, sn, 0.15, 1)
    p_head, n_head = dots(gp, head_tris, gn, 0.2, 2)
    keep = n_head[:, 2] > -0.62                                      # never turns towards the viewer
    p_head, n_head = p_head[keep], n_head[keep]
    from scipy.spatial import cKDTree
    far = cKDTree(p_face).query(p_head)[0] > 0.15
    p_head, n_head = p_head[far], n_head[far]
    face_tone = 0.18 + rng.uniform(0, 0.1, len(p_face))
    # shading drawn in, as in a stipple drawing: a shadow running down under each
    # cheekbone towards the corner of the mouth, light on the cheekbones, and a
    # darker edge along the jaw
    g = lambda p, c, sg: np.exp(-(((p - np.array(c)) / sg) ** 2).sum(1) / 2)

    def band(p, a, b, width):
        a, b = np.array(a), np.array(b)
        t = np.clip(((p - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
        d = np.linalg.norm(p - (a + t[:, None] * (b - a)), axis=1)
        return np.exp(-(d / width) ** 2 / 2) * np.sin(np.pi * t) ** 0.6

    for sx in (1, -1):
        m = np.array([sx, 1, 1])
        face_tone += 0.3 * band(p_face, (6.6 * sx, -1.2, 1.4), (3.7 * sx, -3.9, 4.1), 0.6)
        face_tone -= 0.08 * g(p_face, face_v[123] * m, 0.9)
        face_tone += 0.1 * band(p_face, face_v[58] * m, face_v[149] * m, 0.45)
    head_kind = np.where(p_head[:, 1] < face_v[152, 1] - 0.6, K_NECK, K_SKIN)
    head_tone = 0.16 + rng.uniform(0, 0.08, len(p_head))

    # 2b. procedural hair: off by default (python build_generic.py --procedural-hair).
    # A modelled hairstyle is to replace it.
    if args.procedural_hair:
        # 2b. the hair: long and wavy, parted in the middle. Strands grow from the
        # parting over the head, combed by gravity: sideways over the top and across
        # the temples (the curtains), down over the sides, back behind the ears. Above
        # the ears they lie on the head; lower down they hang free, with more volume,
        # to the jaw at the sides and the collar at the back, and flick out at the ends.
        # Waves are shared per lock of strands and grow towards the tips.
        brow_top = face_v[sorted({c.start for c in FLC.FACE_LANDMARKS_LEFT_EYEBROW}), 1].max()
        top_line = face_v[10, 1] - 0.4

        def hairline(x):
            return brow_top + 1.3 + (top_line - brow_top - 1.3) * (1 - np.clip(np.abs(x) / 5.2, 0, 1)) ** 1.4

        def hair_end(p):
            return np.interp(p[:, 2], [-10.0, -2.0], [-11.5, -8.8])

        def in_hair(p, n):
            face_zone = (n[:, 2] > 0.3) & (np.abs(p[:, 0]) < 6.9) & (p[:, 1] < hairline(p[:, 0]))
            cheek = (p[:, 2] > -1.2) & (p[:, 1] < 2.2) & (n[:, 2] > -0.2)    # in front of the ears
            return ~(face_zone | cheek | (p[:, 1] < hair_end(p)))

        axis_z = lambda y: np.interp(y, [NECK['y0'], -4.0, c0[1]], [NECK['cz'], -3.4, c0[2]])

        def place(p, off):
            """On the head (plus the hair's thickness) above the ears; below them hanging free, never inside."""
            y = p[:, 1]
            o = np.where((y > c0[1])[:, None], c0[None], np.stack([np.zeros(len(p)), y, axis_z(y)], 1))
            d = p - o
            d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
            surf, _ = surface(o, d)
            # thicker on top, fuller again where it hangs
            rs = np.linalg.norm(surf - o, axis=1) + off + 0.5 * np.clip((y - 4.0) / 4.0, 0, 1) + 0.9 * np.clip((4.0 - y) / 9.0, 0, 1)
            rp = np.linalg.norm(p - o, axis=1)
            return o + d * np.where(y > 0.5, rs, np.maximum(rs, rp))[:, None], d

        hr = np.random.default_rng(11)

        def grow(roots, sides, steps, step=0.4, lift=None):
            n = len(roots)
            base = hr.uniform(0.75, 1.5, n)
            lift = np.zeros(n) if lift is None else lift
            end = hair_end(roots) - hr.uniform(0, 0.9, n)
            travelled = np.zeros(n)
            off = base + lift
            pos, nrm = place(roots, off)
            path, alive, miss, count, last = [pos.copy()], np.ones(n, bool), np.zeros(n, int), np.ones(n, int), np.zeros((n, 3))
            for k in range(steps):
                sideness = np.clip(np.abs(nrm[:, 0]) + 0.6 * np.maximum(-nrm[:, 2], 0), 0, 1)
                w = np.clip(0.25 + 0.85 * np.clip((sideness - 0.35) / 0.5, 0, 1), 0, 1) * (1 - 0.6 * np.maximum(nrm[:, 1], 0))
                d = np.stack([(1 - w) * sides, -w, np.zeros(n)], 1)
                ear = (np.abs(pos[:, 0]) > 5.5) * np.exp(-((pos[:, 1] - 0.3) / 3.2) ** 2) * np.clip((pos[:, 2] + 6.0) / 3.0, 0, 1)
                d[:, 2] -= 0.95 * ear
                d -= (d * nrm).sum(1, keepdims=True) * nrm * (pos[:, 1] > 0.5)[:, None]
                d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
                if k:
                    d = 0.6 * d + 0.4 * last
                    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
                travelled += step
                off = base + lift * np.exp(-travelled / 2.2)                 # the curtains rise from the parting
                new, nn = place(pos + d * step, off)
                miss = np.where(in_hair(new, nn), 0, miss + 1)
                alive &= (miss < 2) & (new[:, 1] > end)
                pos = np.where(alive[:, None], new, pos)
                nrm = np.where(alive[:, None], nn, nrm)
                last = np.where(alive[:, None], d, last)
                count += alive
                path.append(pos.copy())
                if not alive.any():
                    break
            path = np.stack(path, 1)
            out = []
            for i in range(n):
                pts = path[i, :count[i]]
                if len(pts) < 5:
                    continue
                if last[i, 1] < -0.4 and pts[-1, 1] < -3.0:                # falling ends flick out
                    _, on = place(pts[-1:], off[i:i + 1])
                    for kk in range(1, 4):
                        dd = last[i] + on[0] * 0.5 * kk + np.array([0, 0.22 * kk, 0])
                        pts = np.vstack([pts, pts[-1] + dd / np.linalg.norm(dd) * step * 0.9])
                out.append((pts, off[i]))
            return out

        strands, locks = [], []
        roots, sides, lock_ids, lifts = [], [], [], []
        for idx, phi in enumerate(np.radians(np.arange(-61, 52, 1.5))):
            rp, _ = surface(c0[None], np.array([[0.0, np.cos(phi), -np.sin(phi)]]))
            for side in (-1, 1):
                for j in range(2):
                    roots.append(rp[0] + np.array([side * (0.08 + 0.18 * j + hr.normal(0, 0.03)), hr.normal(0, 0.06), 0]))
                    sides.append(side); lock_ids.append(('p', side, idx // 2))
                    lifts.append(1.5 * np.clip(rp[0, 2] / 4.0, 0, 1) * hr.uniform(0.6, 1.0))
        for (pts, _), lk in zip(grow(np.array(roots), np.array(sides, float), 75, lift=np.array(lifts)), lock_ids):
            strands.append(pts); locks.append(lk)
        # shorter strands rooted all over the hair for coverage, and a band behind the
        # ears for the hair that hangs to the jaw and the collar
        cand = gp[(gp[:, 1] > -6) & in_hair(gp, gn) & ~((np.abs(gp[:, 0]) < 1.5) & (gn[:, 1] > 0.7))]
        cand = cand + hr.normal(0, 0.3, cand.shape)
        band = gp[(gp[:, 1] > -2.5) & (gp[:, 1] < 4.0) & (gp[:, 2] < -3.0) & in_hair(gp, gn)]
        band = np.concatenate([band + hr.normal(0, 0.35, band.shape) for _ in range(3)])
        fill = np.concatenate([cand[poisson_disk(cand, 1.1, 12)], band[poisson_disk(band, 0.8, 13)]])
        f_sides = np.where(np.abs(fill[:, 0]) < 0.15, hr.choice([-1.0, 1.0], len(fill)), np.sign(fill[:, 0]))
        for (pts, _), r0 in zip(grow(fill, f_sides, 40), fill):
            strands.append(pts); locks.append(('f',) + tuple(np.round(r0 / 1.4).astype(int)))

        lock_look = {}
        hair_v, hair_n, hair_t, hair_dot, hair_dot_n, hair_dot_tone = [], [], [], [], [], []
        for pts, lk in zip(strands, locks):
            if lk not in lock_look:
                lock_look[lk] = (hr.uniform(0, 2 * np.pi), hr.uniform(2.6, 3.4), hr.uniform(0.45, 0.9))
            ph, lam, tone = lock_look[lk]
            ph += hr.normal(0, 0.35)
            sl = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            tan = np.gradient(pts, axis=0)
            tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
            y = pts[:, 1]
            o = np.where((y > c0[1])[:, None], c0[None], np.stack([np.zeros(len(pts)), y, axis_z(y)], 1))
            nrm = pts - o
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
            b = np.cross(nrm, tan)
            b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
            amp = 0.05 + 0.42 * np.clip(sl / 9.0, 0, 1) ** 1.3
            w = pts + b * (amp * np.sin(2 * np.pi * sl / lam + ph))[:, None]
            hair_v.append(w); hair_n.append(nrm); hair_t.append(sl / max(sl[-1], 1e-6))
            hair_dot.append(w[::5]); hair_dot_n.append(nrm[::5]); hair_dot_tone.append(np.full(len(w[::5]), tone))
        hair_dot, hair_dot_n, hair_dot_tone = np.concatenate(hair_dot), np.concatenate(hair_dot_n), np.concatenate(hair_dot_tone)
        print('hair strands', len(hair_v), 'segments', sum(len(v) - 1 for v in hair_v))

        # the hair hides the scattered dots under it
        under = in_hair(p_head, n_head)
        p_head, n_head, head_kind, head_tone = p_head[~under], n_head[~under], head_kind[~under], head_tone[~under]
    else:
        hair_v, hair_n, hair_t = [], [], []
        hair_dot, hair_dot_n, hair_dot_tone = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    def clear_of_head(p, clearance=0.35):
        axis = np.stack([np.zeros(len(p)), p[:, 1], np.full(len(p), -3.6)], 1)
        d = p - axis
        d[:, 1] = 0
        dist = np.linalg.norm(d, axis=1) + 1e-9
        d /= dist[:, None]
        surf, _ = surface(axis, d)
        need = np.maximum(np.linalg.norm(surf - axis, axis=1) + clearance - dist, 0)
        tree = cKDTree(p)
        spread = np.array([need[nb].max() if len(nb) else 0 for nb in tree.query_ball_point(p, 1.0)])
        push = np.maximum(need, 0.5 * spread)
        print('hair pushed clear of the head: %d of %d points, at most %.2f cm' % ((push > 0.01).sum(), len(p), push.max()))
        return p + d * push[:, None]

    hair_occ_v, hair_occ_f = np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    if not args.procedural_hair and args.hair_mesh and os.path.exists(args.hair_mesh):
        (strand_v, _, _, hair_dot, hair_dot_n, hair_dot_tone,
         hair_occ_v, hair_occ_f) = sculpted_hair(args.hair_mesh, np.random.default_rng(5))
        hair_v = []                                                  # dots only, no lines
        hair_extent = list(strand_v)
        # the hair was fitted to the cranium; where the wider jaw now reaches into it,
        # push it out (a lock moves as a whole, so the push is spread over neighbours)
        hair_dot = clear_of_head(hair_dot)
        hair_occ_v = clear_of_head(hair_occ_v)
    else:
        hair_extent = list(hair_v)

    # 3. bright clusters along the features, like a scan picking them out
    def contour(conns, spacing, tone, jitter, seed, keep=None):
        r = np.random.default_rng(seed)
        pts, nrm, tn = [], [], []
        for c in conns:
            a, b = face_v[c.start], face_v[c.end]
            if keep is not None and not keep(a, b):
                continue
            m = max(1, int(np.linalg.norm(b - a) / spacing))
            t = (np.arange(m) + r.random(m)) / m
            n = fn[c.start][None] * (1 - t[:, None]) + fn[c.end][None] * t[:, None]
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            p = a[None] + (b - a)[None] * t[:, None] + n * 0.05 + r.normal(0, jitter, (m, 3)) * np.array([1, 1, 0.3])
            pts.append(p); nrm.append(n); tn.append(np.full(m, tone))
        return pts, nrm, tn

    tip_y = face_v[4, 1] - 0.2
    groups = [
        (FLC.FACE_LANDMARKS_LEFT_EYE, 0.07, 0.85, 0.01, None), (FLC.FACE_LANDMARKS_RIGHT_EYE, 0.07, 0.85, 0.01, None),
        (FLC.FACE_LANDMARKS_LEFT_EYEBROW, 0.09, 0.6, 0.04, None), (FLC.FACE_LANDMARKS_RIGHT_EYEBROW, 0.09, 0.6, 0.04, None),
        (FLC.FACE_LANDMARKS_LIPS, 0.08, 0.7, 0.01, None),
        (FLC.FACE_LANDMARKS_NOSE, 0.09, 0.6, 0.015, lambda a, b: (a[1] < tip_y or b[1] < tip_y) and not (abs(a[0]) < 0.01 and abs(b[0]) < 0.01)),
        (FLC.FACE_LANDMARKS_NOSE, 0.12, 0.4, 0.015, lambda a, b: abs(a[0]) < 0.01 and abs(b[0]) < 0.01),
        (FLC.FACE_LANDMARKS_FACE_OVAL, 0.1, 0.72, 0.02, lambda a, b: a[1] < face_v[132, 1] + 0.5 and b[1] < face_v[132, 1] + 0.5),
    ]
    c_pts, c_nrm, c_tone = [], [], []
    for k, (conns, sp, tn, jit, kf) in enumerate(groups):
        p, n, t = contour(conns, sp, tn, jit, 20 + k, kf)
        c_pts += p; c_nrm += n; c_tone += t
    c_pts, c_nrm, c_tone = np.concatenate(c_pts), np.concatenate(c_nrm), np.concatenate(c_tone)
    k = poisson_disk(c_pts, 0.055, 5)
    c_pts, c_nrm, c_tone = c_pts[k], c_nrm[k], c_tone[k]

    def face_z(x, y):
        """Depth of the face seen straight from the front, -inf where there is none."""
        x, y = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        t = hit_face(np.stack([x, y, np.full(len(x), 30.0)], 1), np.tile([0.0, 0.0, -1.0], (len(x), 1)))
        return np.where(t > 0, 30.0 - t, -np.inf)

    # the glasses (--glasses; off on the site)
    if args.glasses:
        eye_c = face_v[sorted(eye_l)].mean(0)
        eye_c = (abs(eye_c[0]), eye_c[1])
        g_pts, g_nrm, g_tone = glasses(eye_c, eye_c[1] + 1.36, face_v[49, 1] - 0.1, face_z, np.random.default_rng(9))
        c_pts, c_nrm, c_tone = np.vstack([c_pts, g_pts]), np.vstack([c_nrm, g_nrm]), np.concatenate([c_tone, g_tone])
        print('glasses', len(g_pts), 'dots')

    # the birthmark (--birthmark; off on the site): at the edge of the right nostril
    # wing, as in the photo (the subject's right is -x; on screen it is on the left)
    if args.birthmark:
        bx, by = face_v[129, 0] - 0.16, face_v[129, 1] - 0.2
        bc = np.array([bx, by, face_z(bx, by)[0]])
        bn = fn[np.argmin(((face_v - bc) ** 2).sum(1))]
        e1 = np.cross(bn, [0.0, 1.0, 0.0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(bn, e1)
        r_ = 0.17 * np.sqrt(np.random.default_rng(12).random(300)); a_ = np.random.default_rng(13).random(300) * 2 * np.pi
        m_pts = bc + bn * 0.04 + np.outer(r_ * np.cos(a_), e1) + np.outer(r_ * np.sin(a_), e2)
        m_pts = m_pts[poisson_disk(m_pts, 0.045, 14)]
        c_pts, c_nrm = np.vstack([c_pts, m_pts]), np.vstack([c_nrm, np.tile(bn, (len(m_pts), 1))])
        c_tone = np.concatenate([c_tone, np.full(len(m_pts), 0.95)])

    # irises: a ring and a pupil in the middle of each eye opening; they follow the gaze
    i_pts = []
    rad = 0.55
    for ids in (eye_l, eye_r):
        ctr = face_v[list(ids)].mean(0)
        ang = np.linspace(0, 2 * np.pi, 22, endpoint=False)
        z = ctr[2] + 0.12
        i_pts += [np.stack([ctr[0] + rad * np.cos(ang), ctr[1] + rad * np.sin(ang), np.full(22, z)], 1),
                  np.stack([ctr[0] + 0.17 * np.cos(ang[::4]), ctr[1] + 0.17 * np.sin(ang[::4]), np.full(6, z + 0.03)], 1),
                  np.array([[ctr[0], ctr[1], z + 0.03]])]
    i_pts = np.concatenate(i_pts)
    i_nrm = np.tile([0.0, 0.0, 1.0], (len(i_pts), 1))
    eye_y = face_v[list(eye_l | eye_r), 1].mean()


    # 5. export -------------------------------------------------------------------
    # model units: the face (forehead to chin) about 1.85 tall, the eyes at y = 0.107
    scale = 1.85 / (face_v[10, 1] - face_v[152, 1])
    origin = np.array([0.0, eye_y - 0.107 / scale, 0.0])
    P = lambda p: (np.array(p, dtype=float, ndmin=2) - origin) * scale
    q = 1.0 / 8192.0                                                 # int16 covers +-4 model units
    dt = [('p', '<i2', 3), ('n', 'i1', 3), ('k', 'u1'), ('t', 'u1'), ('pad', 'u1')]

    def rec(raw, nrm, kind, tone):
        pos = P(raw)
        r = np.zeros(len(pos), dtype=dt)
        r['p'] = np.round(pos / q).clip(-32767, 32767)
        r['n'] = np.round(np.array(nrm, ndmin=2) * 127).clip(-127, 127)
        r['k'] = kind
        r['t'] = np.round(np.clip(tone, 0, 1) * 255)
        return r

    pts = np.concatenate([
        rec(p_face, n_face, K_SKIN, face_tone),
        rec(p_head, n_head, head_kind, head_tone),
        rec(c_pts, c_nrm, K_CONTOUR, c_tone),
        rec(i_pts, i_nrm, K_IRIS, 0.9),
        rec(hair_dot, hair_dot_n, K_HAIR, hair_dot_tone),
    ])

    lines = np.zeros(0, dtype=dt)                                    # no line segments any more
    # hair: line strips, each strand's points once; index 0xFFFF starts the next
    # strand (primitive restart). t runs from 0 at the root to 255 at the tip.
    if hair_v:
        hair = rec(np.concatenate(hair_v), np.concatenate(hair_n), L_HAIR, np.concatenate(hair_t))
        starts = np.cumsum([0] + [len(v) for v in hair_v[:-1]])
        hair_i = np.concatenate([np.append(np.arange(st, st + len(v)), 0xFFFF) for st, v in zip(starts, hair_v)]).astype(np.uint16)
        assert len(hair) < 0xFFFF
    else:
        hair, hair_i = rec(np.zeros((0, 3)), np.zeros((0, 3)), L_HAIR, np.zeros(0)), np.zeros(0, np.uint16)

    tri_v = np.concatenate([P(face_v), P(gp), P(hair_occ_v)])
    assert len(tri_v) < 0xFFFF
    tri_i = np.concatenate([face_f, head_tris + len(face_v), hair_occ_f + len(face_v) + len(gp)]).astype(np.uint16).ravel()
    tv = np.zeros(len(tri_v), dtype=[('p', '<i2', 3), ('pad', '<i2')])
    tv['p'] = np.round(tri_v / q).clip(-32767, 32767)

    os.makedirs(args.out, exist_ok=True)
    o_hair = pts.nbytes + lines.nbytes + tv.nbytes + tri_i.nbytes
    o_hair += (-o_hair) % 4                                           # keep the vertex data 4-byte aligned
    blob = pts.tobytes() + lines.tobytes() + tv.tobytes() + tri_i.tobytes()
    blob += b'\0' * (o_hair - len(blob)) + hair.tobytes() + hair_i.tobytes()
    with open(os.path.join(args.out, 'face.bin'), 'wb') as fh:
        fh.write(blob)
    allp = P(np.concatenate([p_face, p_head] + hair_extent))
    chin = float(P(face_v[152])[0, 1])
    meta = dict(
        posScale=q,
        points=dict(offset=0, count=int(len(pts)), stride=12),
        lines=dict(offset=int(pts.nbytes), count=int(len(lines)), stride=12),
        tris=dict(offset=int(pts.nbytes + lines.nbytes), vertices=int(len(tv)), stride=8,
                  indexOffset=int(pts.nbytes + lines.nbytes + tv.nbytes), indices=int(len(tri_i))),
        iris=dict(radius=float(rad * scale)),
        kinds=dict(skin=K_SKIN, hair=K_HAIR, neck=K_NECK, iris=K_IRIS, contour=K_CONTOUR, node=K_NODE, key=K_KEY, tip=K_TIP),
        frame=dict(x=float(np.abs(allp[allp[:, 1] > chin, 0]).max().round(3)), top=float(allp[:, 1].max().round(3)),
                   chin=round(chin, 3)),
    )
    if hair_v:
        meta['hair'] = dict(offset=int(o_hair), vertices=int(len(hair)), stride=12,
                            indexOffset=int(o_hair + hair.nbytes), indices=int(len(hair_i)))
    with open(os.path.join(args.out, 'face.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    print('points', len(pts), 'line vertices', len(lines), 'tri vertices', len(tv), 'indices', len(tri_i),
          'bytes', len(blob), 'frame', meta['frame'])


if __name__ == '__main__':
    main()
