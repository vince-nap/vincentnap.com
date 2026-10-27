#!/usr/bin/env python3
"""
Builds the generic face for the landing page: a neutral head that is nobody in
particular, with long wavy hair parted in the middle, as a wireframe with sparse
dots, bright clusters along the features and eight tracking points. No photos
are used.

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
                        L_FACE, L_HEAD, L_LEADER, L_HAIR, load_obj, poisson_disk, sample_mesh)

# the solids, in the canonical model's centimetres (x right, y up, z towards the viewer)
CRANIUM = dict(c=(0.0, 3.0, -4.25), r=(7.9, 10.6, 9.75), n=2.2)
JAW = dict(c=(0.0, -4.6, -3.0), r=(6.8, 5.4, 6.4), n=2.0)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=os.path.join(HERE, 'models'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'site', 'assets', 'face'))
    args = ap.parse_args()

    face_v, face_f = load_obj(os.path.join(args.models, 'canonical_face_model.obj'))
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
    edges = np.concatenate([np.stack([gid, np.roll(gid, -1, 1)], -1).reshape(-1, 2),
                            np.stack([gid[:-1], gid[1:]], -1).reshape(-1, 2),
                            np.stack([gid[:-1], np.roll(gid, -1, 1)[1:]], -1).reshape(-1, 2)])
    quads = np.stack([gid[:-1], np.roll(gid, -1, 1)[:-1], np.roll(gid, -1, 1)[1:], gid[1:]], -1).reshape(-1, 4)
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    crown_id = len(gp)
    gp = np.vstack([gp, crown]); gn = np.vstack([gn, [0, 1, 0]]); covered = np.append(covered, False)
    edges = np.concatenate([edges, np.stack([np.full(n_az, crown_id), gid[0]], 1)])
    tris = np.concatenate([tris, np.stack([np.full(n_az, crown_id), gid[0], np.roll(gid[0], -1)], 1)])
    head_edges = edges[~covered[edges].any(1)]
    head_tris = tris[~covered[tris].all(1)]

    # the face mesh's own edges, without the ones that close the eyes and the mouth
    eye_l = {c.start for c in FLC.FACE_LANDMARKS_LEFT_EYE} | {c.end for c in FLC.FACE_LANDMARKS_LEFT_EYE}
    eye_r = {c.start for c in FLC.FACE_LANDMARKS_RIGHT_EYE} | {c.end for c in FLC.FACE_LANDMARKS_RIGHT_EYE}
    lips_inner = {78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415}
    open_tri = np.array([set(t) <= eye_l or set(t) <= eye_r or set(t) <= lips_inner for t in face_f])
    ft = face_f[~open_tri]
    face_edges = np.unique(np.sort(np.concatenate([ft[:, [0, 1]], ft[:, [1, 2]], ft[:, [2, 0]]]), 1), axis=0)

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
    p_face, n_face = dots(sm.vertices, sm.faces, sn, 0.2, 1)
    p_head, n_head = dots(gp, head_tris, gn, 0.27, 2)
    keep = n_head[:, 2] > -0.62                                      # never turns towards the viewer
    p_head, n_head = p_head[keep], n_head[keep]
    from scipy.spatial import cKDTree
    far = cKDTree(p_face).query(p_head)[0] > 0.2
    p_head, n_head = p_head[far], n_head[far]
    face_tone = 0.14 + rng.uniform(0, 0.08, len(p_face))
    head_kind = np.where(p_head[:, 1] < face_v[152, 1] - 0.6, K_NECK, K_SKIN)
    head_tone = 0.12 + rng.uniform(0, 0.08, len(p_head))

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

    # the hair hides the head grid and the scattered dots under it
    grid_hair = in_hair(gp, gn) & ~covered
    head_edges = head_edges[~grid_hair[head_edges].all(1)]
    under = in_hair(p_head, n_head)
    p_head, n_head, head_kind, head_tone = p_head[~under], n_head[~under], head_kind[~under], head_tone[~under]

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
        (FLC.FACE_LANDMARKS_LEFT_EYE, 0.05, 1.0, 0.02, None), (FLC.FACE_LANDMARKS_RIGHT_EYE, 0.05, 1.0, 0.02, None),
        (FLC.FACE_LANDMARKS_LEFT_EYEBROW, 0.08, 0.8, 0.06, None), (FLC.FACE_LANDMARKS_RIGHT_EYEBROW, 0.08, 0.8, 0.06, None),
        (FLC.FACE_LANDMARKS_LIPS, 0.055, 0.95, 0.025, None),
        (FLC.FACE_LANDMARKS_NOSE, 0.065, 0.85, 0.025, lambda a, b: (a[1] < tip_y or b[1] < tip_y) and not (abs(a[0]) < 0.01 and abs(b[0]) < 0.01)),
        (FLC.FACE_LANDMARKS_NOSE, 0.1, 0.45, 0.02, lambda a, b: abs(a[0]) < 0.01 and abs(b[0]) < 0.01),
        (FLC.FACE_LANDMARKS_FACE_OVAL, 0.065, 0.85, 0.025, lambda a, b: a[1] < face_v[132, 1] + 0.5 and b[1] < face_v[132, 1] + 0.5),
    ]
    c_pts, c_nrm, c_tone = [], [], []
    for k, (conns, sp, tn, jit, kf) in enumerate(groups):
        p, n, t = contour(conns, sp, tn, jit, 20 + k, kf)
        c_pts += p; c_nrm += n; c_tone += t
    c_pts, c_nrm, c_tone = np.concatenate(c_pts), np.concatenate(c_nrm), np.concatenate(c_tone)
    k = poisson_disk(c_pts, 0.04, 5)
    c_pts, c_nrm, c_tone = c_pts[k], c_nrm[k], c_tone[k]

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

    # 4. nodes and tracking points with leader lines
    node_face = np.setdiff1d(np.arange(len(face_v)), list(eye_l | eye_r))
    hn = np.where(~covered & ~grid_hair)[0]
    key_ids = [33, 263, 123, 352, 172, 397, 234, 454]
    leaders = []
    for kid in key_ids:
        b = face_v[kid]
        out = np.array([np.sign(b[0]), 0.0, 0.0]) + 0.9 * fn[kid] + np.array([0, 0.15 if b[1] > eye_y - 3 else -0.25, 0])
        leaders.append((b, out / np.linalg.norm(out), 0.2 + 0.07 * (kid % 5) / 4))

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
        rec(face_v[node_face], fn[node_face], K_NODE, 0.55),
        rec(gp[hn], gn[hn], K_NODE, 0.35),
        rec(face_v[key_ids], fn[key_ids], K_KEY, 1.0),
        rec([b for b, _, _ in leaders], [o for _, o, _ in leaders], K_TIP, [l for _, _, l in leaders]),
        rec(i_pts, i_nrm, K_IRIS, 0.9),
        rec(hair_dot, hair_dot_n, K_HAIR, hair_dot_tone),
    ])

    def line_rec(a, b, na, nb, kind, pa, pb):
        r = np.zeros(2 * len(a), dtype=dt)
        r['p'][0::2] = np.round(P(a) / q).clip(-32767, 32767); r['p'][1::2] = np.round(P(b) / q).clip(-32767, 32767)
        r['n'][0::2] = np.round(na * 127); r['n'][1::2] = np.round(nb * 127)
        r['k'] = kind
        r['t'][0::2] = pa; r['t'][1::2] = pb
        return r

    lb = np.array([b for b, _, _ in leaders]); lo_ = np.array([o for _, o, _ in leaders])
    lines = np.concatenate([
        line_rec(face_v[face_edges[:, 0]], face_v[face_edges[:, 1]], fn[face_edges[:, 0]], fn[face_edges[:, 1]], L_FACE, 0, 0),
        line_rec(gp[head_edges[:, 0]], gp[head_edges[:, 1]], gn[head_edges[:, 0]], gn[head_edges[:, 1]], L_HEAD, 0, 0),
        line_rec(lb, lb, lo_, lo_, L_LEADER, 0, np.round(np.array([l for _, _, l in leaders]) * 255)),
    ])
    # hair: line strips, each strand's points once; index 0xFFFF starts the next
    # strand (primitive restart). t runs from 0 at the root to 255 at the tip.
    hair = rec(np.concatenate(hair_v), np.concatenate(hair_n), L_HAIR, np.concatenate(hair_t))
    starts = np.cumsum([0] + [len(v) for v in hair_v[:-1]])
    hair_i = np.concatenate([np.append(np.arange(st, st + len(v)), 0xFFFF) for st, v in zip(starts, hair_v)]).astype(np.uint16)
    assert len(hair) < 0xFFFF

    tri_v = np.concatenate([P(face_v), P(gp)])
    tri_i = np.concatenate([face_f, head_tris + len(face_v)]).astype(np.uint16).ravel()
    tv = np.zeros(len(tri_v), dtype=[('p', '<i2', 3), ('pad', '<i2')])
    tv['p'] = np.round(tri_v / q).clip(-32767, 32767)

    os.makedirs(args.out, exist_ok=True)
    o_hair = pts.nbytes + lines.nbytes + tv.nbytes + tri_i.nbytes
    o_hair += (-o_hair) % 4                                           # keep the vertex data 4-byte aligned
    blob = pts.tobytes() + lines.tobytes() + tv.tobytes() + tri_i.tobytes()
    blob += b'\0' * (o_hair - len(blob)) + hair.tobytes() + hair_i.tobytes()
    with open(os.path.join(args.out, 'face.bin'), 'wb') as fh:
        fh.write(blob)
    allp = P(np.concatenate([p_face, p_head] + hair_v))
    chin = float(P(face_v[152])[0, 1])
    meta = dict(
        posScale=q,
        points=dict(offset=0, count=int(len(pts)), stride=12),
        lines=dict(offset=int(pts.nbytes), count=int(len(lines)), stride=12),
        tris=dict(offset=int(pts.nbytes + lines.nbytes), vertices=int(len(tv)), stride=8,
                  indexOffset=int(pts.nbytes + lines.nbytes + tv.nbytes), indices=int(len(tri_i))),
        hair=dict(offset=int(o_hair), vertices=int(len(hair)), stride=12,
                  indexOffset=int(o_hair + hair.nbytes), indices=int(len(hair_i))),
        iris=dict(radius=float(rad * scale)),
        kinds=dict(skin=K_SKIN, hair=K_HAIR, neck=K_NECK, iris=K_IRIS, contour=K_CONTOUR, node=K_NODE, key=K_KEY, tip=K_TIP),
        frame=dict(x=float(np.abs(allp[allp[:, 1] > chin, 0]).max().round(3)), top=float(allp[:, 1].max().round(3)),
                   chin=round(chin, 3)),
    )
    with open(os.path.join(args.out, 'face.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    print('points', len(pts), 'line vertices', len(lines), 'tri vertices', len(tv), 'indices', len(tri_i),
          'bytes', len(blob), 'frame', meta['frame'])


if __name__ == '__main__':
    main()
