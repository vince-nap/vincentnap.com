#!/usr/bin/env python3
"""
Builds the face that assembles on the landing page: a wireframe head with
sparse dots, bright clusters along the features and eight tracking points.

Input: three photos in tools/face/input (front.jpg, left.jpg, right.jpg; the
profiles are named after the way the head is turned: in left.jpg the nose
points to the left of the frame). Output: site/assets/face/face.bin and
face.json (the byte layout of the sections is in face.json).

How the head is made, in order:

  1. MediaPipe finds the 478 face landmarks and segments each photo into
     hair / face skin / body skin / clothes / background.
  2. The landmarks of the front photo are turned upright (head pose removed)
     and made symmetric: every point is averaged with its mirror twin. The jaw
     is squared a little (MediaPipe draws it as a V), and the forehead is
     brought forward to the side outline (MediaPipe lets it recede).
  3. The two profiles are aligned to that face through its midline points
     (forehead, nose, lips, chin), which gives the side outline of the head:
     how far the hair reaches to the front and back at every height.
  4. The rest of the head is inflated from the silhouettes: a front and a
     back height field, each the solution of a Poisson equation pinned to
     the face mesh and zero on the silhouette, so both meet at a vertical
     tangent like a real skull; each row is calibrated to the side outline.
  5. The wireframe: the face mesh's own triangles, and a lat-long grid cast
     against the height fields for the rest of the head. Both are also
     written as occluder triangles, for hidden-line removal in the browser.
  6. Sparse dots on the surface, toned from the photos: on the face only the
     features keep their detail, the skin is smoothed and every dot is
     averaged with its mirror twin (no blemishes, no asymmetry); the hair
     keeps its streaks; a shadow under the jaw.
  7. Bright clusters along the lids, brows, nose, lips and jawline; irises;
     nodes on the wireframe; tracking points with leader lines.

Run with the venv described in tools/face/README.md:
    python tools/face/build_face.py
"""
import argparse
import json
import os

import cv2
import mediapipe as mp
import numpy as np
import trimesh
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import FaceLandmarksConnections as FLC
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# Segmenter classes (selfie_multiclass_256x256)
BG, HAIR, BODY, FACE, CLOTHES, OTHER = range(6)

# Dot kinds, mirrored in site/assets/js/face.js
K_SKIN, K_HAIR, K_NECK, K_IRIS, K_CONTOUR, K_NODE, K_KEY, K_TIP = range(8)
# Line kinds
L_FACE, L_HEAD, L_LEADER = 0, 1, 2

H = 6.0  # grid step of the height fields, in front-photo pixels


# --------------------------------------------------------------------------- detection

def detect(path, models):
    img = mp.Image.create_from_file(path)
    fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(models, 'face_landmarker.task')),
        num_faces=1))
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(models, 'selfie_multiclass_256x256.tflite')),
        output_confidence_masks=True))
    bgr = cv2.imread(path)
    h, w = bgr.shape[:2]
    lm = fl.detect(img).face_landmarks[0]
    pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm])
    conf = np.stack([c.numpy_view().squeeze() for c in seg.segment(img).confidence_masks]).astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return pts, conf, gray


def to3(p):
    """Image landmarks (x right, y down, z away) -> x right, y up, z towards the viewer."""
    return np.stack([p[:, 0], -p[:, 1], -p[:, 2]], 1)


def umeyama(a, b):
    ma, mb = a.mean(0), b.mean(0)
    a0, b0 = a - ma, b - mb
    u, sg, vt = np.linalg.svd(b0.T @ a0 / len(a))
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[2, 2] = -1
    r = u @ d @ vt
    s = (sg * np.diag(d)).sum() / a0.var(0).sum()
    return s, r, mb - s * r @ ma


def load_obj(path):
    v, f = [], []
    for line in open(path):
        if line.startswith('v '):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            f.append([int(t.split('/')[0]) - 1 for t in line.split()[1:4]])
    return np.array(v), np.array(f)


def bilinear(img, u, v):
    hh, ww = img.shape[-2:]
    u = np.clip(u, 0, ww - 1.001)
    v = np.clip(v, 0, hh - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - u0, v - v0
    return (img[..., v0, u0] * (1 - fu) * (1 - fv) + img[..., v0, u0 + 1] * fu * (1 - fv)
            + img[..., v0 + 1, u0] * (1 - fu) * fv + img[..., v0 + 1, u0 + 1] * fu * fv)


def largest_component(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def poisson_disk(points, radius, seed):
    """Keeps a subset of points with no two closer than radius (dart throwing on a hash grid)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points))
    cell = radius / np.sqrt(3)
    keys = np.floor(points / cell).astype(np.int64)
    grid = {}
    keep = []
    r2 = radius * radius
    reach = 2
    for i in order:
        k = keys[i]
        p = points[i]
        ok = True
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                for dz in range(-reach, reach + 1):
                    j = grid.get((k[0] + dx, k[1] + dy, k[2] + dz))
                    if j is not None and ((points[j] - p) ** 2).sum() < r2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid[(k[0], k[1], k[2])] = i
            keep.append(i)
    return np.array(sorted(keep))


def sample_mesh(verts, faces, normals, n, rng):
    """Area-weighted random points on a mesh with interpolated vertex normals."""
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    fi = rng.choice(len(faces), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    bary = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    p = (tri[fi] * bary[:, :, None]).sum(1)
    nn = (normals[faces[fi]] * bary[:, :, None]).sum(1)
    nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-9
    return p, nn


# --------------------------------------------------------------------------- build

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=os.path.join(HERE, 'input'))
    ap.add_argument('--models', default=os.path.join(HERE, 'models'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'site', 'assets', 'face'))
    ap.add_argument('--debug', default=None, help='folder for a dump of the model (model.npz)')
    args = ap.parse_args()

    # 1. detection
    lm_f, conf_f, gray_f = detect(os.path.join(args.input, 'front.jpg'), args.models)
    side = {}
    for name in ('left', 'right'):
        side[name] = detect(os.path.join(args.input, name + '.jpg'), args.models)

    # 2. canonical, symmetric face (pixel units of the front photo)
    cv_, cf = load_obj(os.path.join(args.models, 'canonical_face_model.obj'))
    mirror = cv_ * np.array([-1, 1, 1])
    twin = ((mirror[:, None] - cv_[None]) ** 2).sum(-1).argmin(1)   # mirror twin of every vertex
    assert (twin[twin] == np.arange(len(twin))).all()
    f3 = to3(lm_f)
    s_f, r_f, t_f = umeyama(cv_, f3[:468])
    a_all = (f3 - t_f) @ r_f                                         # upright, centred
    a_face = a_all[:468]
    a_face = 0.5 * (a_face + (a_face * np.array([-1, 1, 1]))[twin])
    a_face[twin == np.arange(468), 0] = 0.0

    def front_uv(p):
        q = p @ r_f.T + t_f
        return q[:, 0], -q[:, 1]

    # 3. profiles: 2D similarity from the face midline (z, y) to the photo
    midline = np.where(twin == np.arange(468))[0]
    prof = {}
    for name, (lm, conf, gray) in side.items():
        facing = np.sign(lm[1, 0] - lm[[234, 454], 0].mean())        # +1: nose points right
        a = a_face[midline][:, 2] * facing + 1j * (-a_face[midline][:, 1])
        b = lm[midline, 0] + 1j * lm[midline, 1]
        ma, mb = a.mean(), b.mean()
        k = ((b - mb) * np.conj(a - ma)).sum() / (abs(a - ma) ** 2).sum()
        # nose to the left = head turned to its right = the left cheek (x > 0) faces the camera
        prof[name] = dict(k=k, t=mb - k * ma, facing=facing, conf=conf, gray=gray,
                          xsign=1 if facing < 0 else -1)

    def side_uv(pr, p):
        w = pr['k'] * (p[:, 2] * pr['facing'] + 1j * (-p[:, 1])) + pr['t']
        return w.real, w.imag

    ys = np.arange(-1000, 1100 + H, H)
    zs = np.arange(-1400, 700 + H, H)
    xs = np.arange(-720, 720 + H, H)
    j0 = int(np.argmin(abs(xs)))
    zf_list, zb_list = [], []
    for pr in prof.values():
        zz, yy = np.meshgrid(zs, ys)
        u, v = side_uv(pr, np.stack([np.zeros(zz.size), yy.ravel(), zz.ravel()], 1))
        head = bilinear(pr['conf'][HAIR] + pr['conf'][BODY] + pr['conf'][FACE], u, v).reshape(zz.shape) > 0.5
        head = ndimage.binary_fill_holes(largest_component(head))
        has = head.any(1)
        zf = np.where(has, zs[np.where(head, np.arange(len(zs)), -1).max(1)], np.nan)
        zb = np.where(has, zs[np.where(head, np.arange(len(zs)), len(zs) - 1).min(1)], np.nan)
        zf_list.append(zf)
        zb_list.append(zb)
    zf_sil = np.nanmean(zf_list, 0)
    zb_sil = np.nanmean(zb_list, 0)

    def smooth_nan(v, sigma):
        ok = ~np.isnan(v)
        num = ndimage.gaussian_filter1d(np.where(ok, v, 0), sigma)
        den = ndimage.gaussian_filter1d(ok.astype(float), sigma)
        return np.where(den > 1e-3, num / np.maximum(den, 1e-3), np.nan)

    # Depth of the silhouette (where front and back meet). The middle of the side
    # outline, but held behind the face: the nose must not pull it forward.
    head_front = np.minimum(zf_sil, 260.0)
    z_c = smooth_nan(0.5 * (head_front + zb_sil), 90 / H)
    z_c = np.where(np.isnan(z_c), np.nanmean(z_c), z_c)
    zf_s = smooth_nan(zf_sil, 4 / H * 3)
    zb_s = smooth_nan(zb_sil, 30 / H)

    # front silhouette on the (x, y) grid, symmetric
    xx, yy = np.meshgrid(xs, ys)
    zc_grid = np.repeat(z_c[:, None], len(xs), 1)
    u, v = front_uv(np.stack([xx.ravel(), yy.ravel(), zc_grid.ravel()], 1))
    head_f = bilinear(conf_f[HAIR] + conf_f[BODY] + conf_f[FACE], u, v).reshape(xx.shape)
    head_f = 0.5 * (head_f + head_f[:, ::-1])
    dom = ndimage.binary_fill_holes(largest_component(head_f > 0.5))
    y_cut = -900.0                                                   # below the throat: the collar
    dom &= (yy >= y_cut)

    # A squarer jaw. MediaPipe's jaw contour runs in an almost straight V from the
    # ears to a narrow chin; on dots that reads as an oval with no jawline. The
    # jaw angle is widened by ~8%, the lower jaw and chin by ~10-12%, and the
    # vertices around them follow with a smooth falloff into the cheeks.
    jaw_widen = {132: 1.015, 58: 1.04, 172: 1.08, 136: 1.10, 150: 1.10, 149: 1.10, 176: 1.11, 148: 1.12}
    ctrl, delta = [], []
    for i, f in jaw_widen.items():
        for j in (i, twin[i]):
            ctrl.append(a_face[j].copy())
            delta.append([a_face[j, 0] * (f - 1), 0.0, 0.0])
    ctrl, delta = np.array(ctrl), np.array(delta)
    g = np.exp(-((a_face[:, None, :] - ctrl[None]) ** 2).sum(-1) / (2 * 55.0 ** 2))
    a_face = a_face + (g[:, :, None] * delta[None]).sum(1) / np.maximum(1.0, g.sum(1))[:, None]
    left = [132, 58, 172, 136, 150, 149, 176, 148]
    jaw_ids = left + [152] + [twin[i] for i in left[::-1]]

    # MediaPipe lets the forehead recede too far; the side outline knows better.
    # Rows above the brows are moved forward to it, fading in from the brow line.
    mesh_mid = np.interp(ys, *zip(*sorted(zip(a_face[midline, 1], a_face[midline, 2]))))
    gap = np.nan_to_num(zf_s - mesh_mid, nan=0.0)
    ramp = np.clip((ys - 100.0) / 220.0, 0, 1)
    shift = ndimage.gaussian_filter1d(gap * ramp, 30 / H)
    shift = np.where(ys > a_face[:, 1].max() + 20, shift[ys <= a_face[:, 1].max()].max(), shift)
    a_face = a_face.copy()
    a_face[:, 2] += np.interp(a_face[:, 1], ys, shift)

    # face mesh, subdivided for smooth dots, rasterised to a depth map
    fm = trimesh.Trimesh(a_face, cf, process=False)
    for _ in range(2):
        v2, f2 = trimesh.remesh.subdivide_loop(fm.vertices, fm.faces)
        fm = trimesh.Trimesh(v2, f2, process=False)
    fv, ff = fm.vertices.copy(), fm.faces.copy()
    fn = fm.vertex_normals.copy()
    if fn[:, 2].mean() < 0:
        fn = -fn
    zface = np.full(xx.shape, -np.inf)
    for tri in fv[ff]:
        x0, y0 = tri[:, 0].min(), tri[:, 1].min()
        x1, y1 = tri[:, 0].max(), tri[:, 1].max()
        ja, jb = int(np.ceil((x0 - xs[0]) / H)), int(np.floor((x1 - xs[0]) / H))
        ia, ib = int(np.ceil((y0 - ys[0]) / H)), int(np.floor((y1 - ys[0]) / H))
        if jb < ja or ib < ia:
            continue
        px, py = np.meshgrid(xs[ja:jb + 1], ys[ia:ib + 1])
        (ax_, ay_), (bx_, by_), (cx_, cy_) = tri[:, :2]
        det = (by_ - cy_) * (ax_ - cx_) + (cx_ - bx_) * (ay_ - cy_)
        if abs(det) < 1e-9:
            continue
        l1 = ((by_ - cy_) * (px - cx_) + (cx_ - bx_) * (py - cy_)) / det
        l2 = ((cy_ - ay_) * (px - cx_) + (ax_ - cx_) * (py - cy_)) / det
        l3 = 1 - l1 - l2
        inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        z = l1 * tri[0, 2] + l2 * tri[1, 2] + l3 * tri[2, 2]
        sub = zface[ia:ib + 1, ja:jb + 1]
        sub[inside] = np.maximum(sub[inside], z[inside])
    face_dom = np.isfinite(zface)
    dom |= face_dom

    # 4. height fields
    half_w = np.array([(xs[row].max() if row.any() else 0) for row in dom]) + H
    def solve(fixed_mask, fixed_val, c_row):
        idx = -np.ones(dom.shape, int)
        free = dom & ~fixed_mask
        idx[free] = np.arange(free.sum())
        n = free.sum()
        rows, cols, vals = [], [], []
        rhs = np.repeat(c_row[:, None], len(xs), 1)[free] * H * H
        fi, fj = np.nonzero(free)
        rows += list(idx[fi, fj]); cols += list(idx[fi, fj]); vals += [4.0] * n
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = fi + di, fj + dj
            okb = (ni >= 0) & (ni < dom.shape[0]) & (nj >= 0) & (nj < dom.shape[1])
            ni, nj = np.clip(ni, 0, dom.shape[0] - 1), np.clip(nj, 0, dom.shape[1] - 1)
            nb_free = okb & free[ni, nj]
            rows += list(idx[fi, fj][nb_free]); cols += list(idx[ni, nj][nb_free]); vals += [-1.0] * int(nb_free.sum())
            nb_fixed = okb & fixed_mask[ni, nj] & dom[ni, nj]
            rhs[nb_fixed] += fixed_val[ni[nb_fixed], nj[nb_fixed]]
        a = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        sol = spsolve(a, rhs)
        out = np.zeros(dom.shape)
        out[free] = sol
        out[fixed_mask & dom] = fixed_val[fixed_mask & dom]
        return np.clip(out, 0, None)

    # Each row is pinned at the silhouette (u = 0) and, for the front, to the face
    # mesh. The source term c = 2 b^2 / a^2 makes a row of half-width a reach depth
    # b on the midline: b comes from the side outline (hair front and back).
    def calibrated(fixed, val, b, rows):
        """Solves, then rescales c row by row until the midline reaches depth b."""
        c = 2.0 * b ** 2 / half_w ** 2
        for _ in range(6):
            u = solve(fixed, val, c)
            got = np.sqrt(u[:, j0])
            ratio = np.where(rows & (got > 1), (b / np.maximum(got, 1)) ** 2, 1.0)
            ratio = ndimage.gaussian_filter1d(np.clip(ratio, 0.5, 3.0), 2)
            c = c * ratio
        return u

    fixed = np.zeros(dom.shape, bool)
    val = np.zeros(dom.shape)
    fixed[face_dom] = True
    val[face_dom] = np.maximum(zface[face_dom] - zc_grid[face_dom], 4.0) ** 2
    b_f = np.maximum(np.nan_to_num(zf_s, nan=0) - z_c, 40)
    rows_f = dom[:, j0] & ~face_dom[:, j0] & ~np.isnan(zf_s) & (dom.sum(1) > 12)
    u_front = calibrated(fixed, val, b_f, rows_f)
    b_b = np.maximum(z_c - np.nan_to_num(zb_s, nan=0), 40)
    rows_b = dom[:, j0] & ~np.isnan(zb_s) & (dom.sum(1) > 12)
    u_back = calibrated(np.zeros(dom.shape, bool), np.zeros(dom.shape), b_b, rows_b)
    z_front = zc_grid + np.sqrt(u_front)
    z_back = zc_grid - np.sqrt(u_back)
    z_front[~dom] = zc_grid[~dom]
    z_back[~dom] = zc_grid[~dom]

    # surface meshes of the head outside the face (for the sparse dots)
    dom_ext = ndimage.binary_dilation(dom)
    vid = np.arange(dom.size).reshape(dom.shape)
    q = dom_ext[:-1, :-1] & dom_ext[1:, :-1] & dom_ext[:-1, 1:] & dom_ext[1:, 1:]
    q &= dom[:-1, :-1] | dom[1:, :-1] | dom[:-1, 1:] | dom[1:, 1:]
    face_cell = face_dom[:-1, :-1] & face_dom[1:, :-1] & face_dom[:-1, 1:] & face_dom[1:, 1:]

    def cells_to_faces(mask, flip):
        i, j = np.nonzero(mask)
        a, b, c, d = vid[i, j], vid[i, j + 1], vid[i + 1, j], vid[i + 1, j + 1]
        t = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
        return t[:, ::-1] if flip else t

    grid_xy = np.stack([xx.ravel(), yy.ravel()], 1)
    front_mesh = trimesh.Trimesh(np.column_stack([grid_xy, z_front.ravel()]), cells_to_faces(q & ~face_cell, False), process=False)
    back_mesh = trimesh.Trimesh(np.column_stack([grid_xy, z_back.ravel()]), cells_to_faces(q, True), process=False)

    def mesh_normals(m, want_z):
        n = m.vertex_normals.copy()
        if (n[:, 2].mean() < 0) != (want_z < 0):
            n = -n
        return n

    # 5. the wireframe of the head: a lat-long grid, cast against the height fields.
    # Above the eyes the rings come from one centre (a dome); below, horizontal
    # rings around the vertical axis (cheeks, jaw, neck).
    face_cover = ndimage.binary_erosion(face_dom, iterations=2)

    def lookup(p):
        j = np.round((p[:, 0] - xs[0]) / H).astype(int)
        i = np.round((p[:, 1] - ys[0]) / H).astype(int)
        ok = (i >= 0) & (i < len(ys)) & (j >= 0) & (j < len(xs))
        return ok, np.clip(i, 0, len(ys) - 1), np.clip(j, 0, len(xs) - 1)

    def inside(p):
        ok, i, j = lookup(p)
        return ok & dom[i, j] & (p[:, 2] <= z_front[i, j]) & (p[:, 2] >= z_back[i, j])

    def cast(origin, dirs):
        t = np.zeros(len(dirs))
        alive = np.ones(len(dirs), bool)
        for _ in range(800):
            ins = inside(origin + dirs * (t + 3.0)[:, None])
            t[alive & ins] += 3.0
            alive &= ins
            if not alive.any():
                break
        return origin + dirs * t[:, None]

    n_az = 44
    th = np.arange(n_az) * 2 * np.pi / n_az                          # 0 = straight ahead
    y0 = 150.0
    c0 = np.array([0.0, y0, float(np.interp(y0, ys, z_c))])
    rings = []
    for phi in np.radians(np.linspace(9, 90, 10)):
        d = np.stack([np.sin(phi) * np.sin(th), np.full(n_az, np.cos(phi)), np.sin(phi) * np.cos(th)], 1)
        rings.append(cast(np.repeat(c0[None], n_az, 0), d))
    for y in np.arange(y0 - 58, y_cut + 1, -58):
        o = np.array([0.0, y, float(np.interp(y, ys, z_c))])
        d = np.stack([np.sin(th), np.zeros(n_az), np.cos(th)], 1)
        rings.append(cast(np.repeat(o[None], n_az, 0), d))
    grid = np.stack(rings)                                           # (rings, n_az, 3)
    crown = cast(c0[None], np.array([[0.0, 1.0, 0.0]]))[0]
    n_r = len(grid)
    # normals from the neighbours, pointing outwards
    t_az = np.roll(grid, -1, 1) - np.roll(grid, 1, 1)
    g_up = np.concatenate([crown[None, None].repeat(n_az, 1), grid[:-1]])
    g_dn = np.concatenate([grid[1:], grid[-1:]])
    t_rn = g_dn - g_up
    g_n = np.cross(t_rn, t_az)
    g_n /= np.linalg.norm(g_n, axis=2, keepdims=True) + 1e-9
    axis_pt = np.stack([np.zeros_like(grid[..., 0]), grid[..., 1], np.interp(grid[..., 1], ys, z_c)], -1)
    flip = ((grid - axis_pt) * g_n).sum(-1) < 0
    g_n[flip] *= -1
    gp = grid.reshape(-1, 3)
    gn = g_n.reshape(-1, 3)
    ok_c, ci, cj = lookup(gp)
    covered = ok_c & face_cover[ci, cj] & (gp[:, 2] > np.interp(gp[:, 1], ys, z_c))
    gid = np.arange(n_r * n_az).reshape(n_r, n_az)
    e_ring = np.stack([gid, np.roll(gid, -1, 1)], -1).reshape(-1, 2)
    e_mer = np.stack([gid[:-1], gid[1:]], -1).reshape(-1, 2)
    e_dia = np.stack([gid[:-1], np.roll(gid, -1, 1)[1:]], -1).reshape(-1, 2)
    head_edges = np.concatenate([e_ring, e_mer, e_dia])
    head_edges = head_edges[~covered[head_edges].any(1)]
    quads = np.stack([gid[:-1], np.roll(gid, -1, 1)[:-1], np.roll(gid, -1, 1)[1:], gid[1:]], -1).reshape(-1, 4)
    head_tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    head_tris = head_tris[~covered[head_tris].all(1)]
    # the crown: a fan
    crown_id = len(gp)
    gp = np.vstack([gp, crown])
    gn = np.vstack([gn, [0, 1, 0]])
    covered = np.append(covered, False)
    head_edges = np.concatenate([head_edges, np.stack([np.full(n_az, crown_id), gid[0]], 1)])
    head_tris = np.concatenate([head_tris, np.stack([np.full(n_az, crown_id), gid[0], np.roll(gid[0], -1)], 1)])

    # the face mesh's own edges, without the ones that close the eyes and the mouth
    eye_l = {c.start for c in FLC.FACE_LANDMARKS_LEFT_EYE} | {c.end for c in FLC.FACE_LANDMARKS_LEFT_EYE}
    eye_r = {c.start for c in FLC.FACE_LANDMARKS_RIGHT_EYE} | {c.end for c in FLC.FACE_LANDMARKS_RIGHT_EYE}
    lips_inner = {78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415}
    open_tri = np.array([set(t) <= eye_l or set(t) <= eye_r or set(t) <= lips_inner for t in cf])
    face_tris = cf[~open_tri]
    fe = np.sort(np.concatenate([face_tris[:, [0, 1]], face_tris[:, [1, 2]], face_tris[:, [2, 0]]]), 1)
    face_edges = np.unique(fe, axis=0)
    fm0 = trimesh.Trimesh(a_face, cf, process=False)
    fn0 = fm0.vertex_normals.copy()
    if fn0[:, 2].mean() < 0:
        fn0 = -fn0
    print('open triangles removed', int(open_tri.sum()), 'face edges', len(face_edges), 'head edges', len(head_edges))

    # 6. sparse dots on the surface
    def dots(verts, faces, normals, radius, seed):
        tri = verts[faces]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
        n_cand = int(area / (radius * radius) * 2.2)
        p, nn = sample_mesh(verts, faces, normals, n_cand, np.random.default_rng(seed))
        keep = poisson_disk(p, radius, seed)
        return p[keep], nn[keep]

    p_face, n_face = dots(fv, ff, fn, 13.0, 1)
    p_fr, n_fr = dots(front_mesh.vertices, front_mesh.faces, mesh_normals(front_mesh, 1), 15.0, 2)
    p_bk, n_bk = dots(back_mesh.vertices, back_mesh.faces, mesh_normals(back_mesh, -1), 17.0, 3)
    p_head = np.concatenate([p_fr, p_bk])
    n_head = np.concatenate([n_fr, n_bk])
    keep = n_head[:, 2] > -0.62                                      # never turns towards the viewer
    p_head, n_head = p_head[keep], n_head[keep]
    from scipy.spatial import cKDTree
    near_face = cKDTree(p_face).query(p_head, k=1)[0] < 13.0
    p_head, n_head = p_head[~near_face], n_head[~near_face]

    # tones: skin smoothed with the features kept, symmetric; hair keeps its streaks
    skin = ((conf_f[FACE] + conf_f[BODY]) > 0.5).astype(np.float32)
    sig = 0.012 * gray_f.shape[1]
    low = cv2.GaussianBlur(gray_f * skin, (0, 0), sig) / np.maximum(cv2.GaussianBlur(skin, (0, 0), sig), 1e-3)
    feat = np.zeros_like(gray_f)
    def hull_of(conns, dil):
        ids = sorted({c.start for c in conns} | {c.end for c in conns})
        m = np.zeros_like(gray_f, np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(lm_f[ids, :2].astype(np.int32)), 1)
        return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dil)).astype(np.float32)
    for conns, dil in ((FLC.FACE_LANDMARKS_LEFT_EYE, (41, 71)), (FLC.FACE_LANDMARKS_RIGHT_EYE, (41, 71)),
                       (FLC.FACE_LANDMARKS_LEFT_EYEBROW, (31, 31)), (FLC.FACE_LANDMARKS_RIGHT_EYEBROW, (31, 31)),
                       (FLC.FACE_LANDMARKS_LIPS, (21, 21))):
        feat = np.maximum(feat, hull_of(conns, dil))
    feat = cv2.GaussianBlur(feat, (0, 0), 9)
    skin_ref = float(np.median(gray_f[(conf_f[FACE] > 0.8)]))
    low = skin_ref + 0.55 * (low - skin_ref)
    tex_face = low + feat * (cv2.GaussianBlur(gray_f, (0, 0), 1.2) - low)
    tex_hair_f = cv2.GaussianBlur(gray_f, (0, 0), 1.5)
    hair_ref = float(np.percentile(gray_f[conf_f[HAIR] > 0.8], 92))

    def front_sample(img, p):
        return bilinear(img, *front_uv(p))

    def tone_from(lum, ref, span):
        return np.clip((ref - lum) / span, 0, 1)

    mirror_x = np.array([-1, 1, 1])
    face_lum = 0.5 * (front_sample(tex_face, p_face) + front_sample(tex_face, p_face * mirror_x))
    face_hair = 0.5 * (front_sample(conf_f[HAIR], p_face) + front_sample(conf_f[HAIR], p_face * mirror_x))
    face_kind = np.where(face_hair > 0.5, K_HAIR, K_SKIN)
    face_tone = np.where(face_hair > 0.5, 0.45 + 0.55 * tone_from(front_sample(tex_hair_f, p_face), hair_ref, 0.55),
                         0.12 + 0.88 * tone_from(face_lum, skin_ref + 0.05, 0.42))

    w_front = np.clip((n_head[:, 2] + 0.3) / 0.7, 0, 1)
    lum = np.zeros(len(p_head))
    cls = np.zeros((len(p_head), 6))
    wsum = np.zeros(len(p_head)) + 1e-6
    u, v = front_uv(p_head)
    lum += w_front * bilinear(tex_hair_f, u, v)
    cls += w_front[:, None] * np.stack([bilinear(conf_f[k], u, v) for k in range(6)], 1)
    wsum += w_front
    for pr in prof.values():
        w_side = (1 - w_front) * (n_head[:, 0] * pr['xsign'] > -0.15)
        u, v = side_uv(pr, p_head)
        lum += w_side * bilinear(cv2.GaussianBlur(pr['gray'], (0, 0), 1.5), u, v)
        cls += w_side[:, None] * np.stack([bilinear(pr['conf'][k], u, v) for k in range(6)], 1)
        wsum += w_side
    lum /= wsum
    cls /= wsum[:, None]
    is_skin = (cls[:, FACE] + cls[:, BODY]) > cls[:, HAIR]
    head_kind = np.where(is_skin, K_SKIN, K_HAIR)
    head_kind = np.where(is_skin & (p_head[:, 1] < a_face[152, 1] - 40), K_NECK, head_kind)
    head_tone = np.where(head_kind == K_HAIR, 0.45 + 0.55 * tone_from(lum, hair_ref, 0.55), 0.18)

    # the shadow under the jaw, so the jaw reads against the neck
    jp = a_face[jaw_ids]
    order_x = np.argsort(jp[:, 0])
    jx, jy = jp[order_x, 0], jp[order_x, 1]
    below = np.interp(p_head[:, 0], jx, jy) - p_head[:, 1]
    shade = np.clip(1 - below / 95.0, 0, 1) ** 1.6 * (below > -4) * (np.abs(p_head[:, 0]) < jx.max() + 10) * (n_head[:, 2] > -0.3)
    head_tone = np.where(head_kind != K_HAIR, np.maximum(head_tone, 0.2 + 0.55 * shade), head_tone)

    # 7. bright clusters along the features: lids, brows, nose, lips, face outline
    def contour_dots(conns, spacing, tone, jitter, seed, keep=None):
        r = np.random.default_rng(seed)
        pts, nrm, tn = [], [], []
        for c in conns:
            a, b = a_face[c.start], a_face[c.end]
            if keep is not None and not keep(a, b):
                continue
            n_seg = max(1, int(np.linalg.norm(b - a) / spacing))
            t = (np.arange(n_seg) + r.random(n_seg)) / n_seg
            p = a[None] + (b - a)[None] * t[:, None]
            nn = fn0[c.start][None] * (1 - t[:, None]) + fn0[c.end][None] * t[:, None]
            nn /= np.linalg.norm(nn, axis=1, keepdims=True)
            p = p + nn * 3.0 + r.normal(0, jitter, p.shape) * np.array([1, 1, 0.3])
            pts.append(p); nrm.append(nn); tn.append(np.full(len(p), tone))
        return np.concatenate(pts), np.concatenate(nrm), np.concatenate(tn)

    jaw_keep = lambda a, b: a[1] < a_face[132, 1] + 30 and b[1] < a_face[132, 1] + 30
    # the nose: the bridge faintly, the wings and the base clearly, but not the
    # segments across the nose above its tip (they close it into a ring)
    tip_y = a_face[4, 1] - 12
    nose_low = lambda a, b: (a[1] < tip_y or b[1] < tip_y) and not (abs(a[0]) < 1 and abs(b[0]) < 1)
    nose_bridge = lambda a, b: abs(a[0]) < 1 and abs(b[0]) < 1
    groups = [
        (FLC.FACE_LANDMARKS_LEFT_EYE, 3.2, 1.0, 1.4, None), (FLC.FACE_LANDMARKS_RIGHT_EYE, 3.2, 1.0, 1.4, None),
        (FLC.FACE_LANDMARKS_LEFT_EYEBROW, 5.0, 0.8, 4.0, None), (FLC.FACE_LANDMARKS_RIGHT_EYEBROW, 5.0, 0.8, 4.0, None),
        (FLC.FACE_LANDMARKS_LIPS, 3.6, 0.95, 1.6, None),
        (FLC.FACE_LANDMARKS_NOSE, 4.0, 0.9, 1.6, nose_low), (FLC.FACE_LANDMARKS_NOSE, 7.0, 0.5, 1.2, nose_bridge),
    ]
    c_pts, c_nrm, c_tone = [], [], []
    for k, (conns, sp, tn, jit, keep_fn) in enumerate(groups):
        p, n, t = contour_dots(conns, sp, tn, jit, 20 + k, keep=keep_fn)
        c_pts.append(p); c_nrm.append(n); c_tone.append(t)
    p, n, t = contour_dots(FLC.FACE_LANDMARKS_FACE_OVAL, 4.0, 0.9, 1.5, 30, keep=jaw_keep)
    c_pts.append(p); c_nrm.append(n); c_tone.append(t)
    c_pts, c_nrm, c_tone = np.concatenate(c_pts), np.concatenate(c_nrm), np.concatenate(c_tone)
    k = poisson_disk(c_pts, 2.6, 5)
    c_pts, c_nrm, c_tone = c_pts[k], c_nrm[k], c_tone[k]

    # the irises: a ring and a pupil per eye, symmetric; they move with the gaze
    ir = a_all[468:478]
    c_r, c_l = ir[0], ir[5]
    rad = 0.5 * (np.linalg.norm(ir[1:5, :2] - c_r[:2], axis=1).mean() + np.linalg.norm(ir[6:10, :2] - c_l[:2], axis=1).mean())
    cx = 0.5 * (abs(c_r[0]) + abs(c_l[0]))
    cy = 0.5 * (c_r[1] + c_l[1])
    i_pts = []
    for sgn in (-1, 1):
        ok, i, j = lookup(np.array([[sgn * cx, cy, 0.0]]))
        zc_eye = float(zface[i[0], j[0]]) + 4.0
        ang = np.linspace(0, 2 * np.pi, 22, endpoint=False)
        ring = np.stack([sgn * cx + rad * 0.95 * np.cos(ang), cy + rad * 0.95 * np.sin(ang), np.full(22, zc_eye)], 1)
        pupil = np.stack([sgn * cx + rad * 0.3 * np.cos(ang[::4]), cy + rad * 0.3 * np.sin(ang[::4]), np.full(len(ang[::4]), zc_eye + 2)], 1)
        i_pts += [ring, pupil, np.array([[sgn * cx, cy, zc_eye + 2]])]
    i_pts = np.concatenate(i_pts)
    i_nrm = np.tile([0.0, 0.0, 1.0], (len(i_pts), 1))

    # 8. nodes: the vertices of the wireframe, and the tracking points with their leader lines
    node_face = np.setdiff1d(np.arange(468), list(eye_l | eye_r))
    hn = np.where(~covered)[0]
    key_ids = [33, 263, 123, 352, 172, 397, 234, 454]
    leaders = []
    for kid in key_ids:
        b = a_face[kid]
        out = np.array([np.sign(b[0]) * 1.0, 0.0, 0.0]) + 0.9 * fn0[kid] + np.array([0, 0.15 if b[1] > 0 else -0.25, 0])
        out /= np.linalg.norm(out)
        length = 0.2 + 0.07 * (kid % 5) / 4                          # model units
        leaders.append((b, out, length))

    # 9. export ---------------------------------------------------------------
    origin = np.array([0.0, cy - 60.0, 0.0])
    scale = 1.0 / 560.0
    P = lambda p: (np.asarray(p) - origin) * scale
    q = 1.0 / 16384.0

    def rec(pos, nrm, kind, tone):
        r = np.zeros(len(pos), dtype=[('p', '<i2', 3), ('n', 'i1', 3), ('k', 'u1'), ('t', 'u1'), ('pad', 'u1')])
        r['p'] = np.round(pos / q).clip(-32767, 32767)
        r['n'] = np.round(np.asarray(nrm) * 127).clip(-127, 127)
        r['k'] = kind
        r['t'] = np.round(np.clip(tone, 0, 1) * 255)
        return r

    pts = [
        rec(P(p_face), n_face, face_kind, face_tone),
        rec(P(p_head), n_head, head_kind, head_tone),
        rec(P(c_pts), c_nrm, K_CONTOUR, c_tone),
        rec(P(a_face[node_face]), fn0[node_face], K_NODE, 0.55),
        rec(P(gp[hn]), gn[hn], K_NODE, 0.35),
        rec(P(a_face[key_ids]), fn0[key_ids], K_KEY, 1.0),
        rec(P([b for b, _, _ in leaders]), [o for _, o, _ in leaders], K_TIP, [l for _, _, l in leaders]),
        rec(P(i_pts), i_nrm, K_IRIS, 0.9),
    ]
    pts = np.concatenate(pts)

    def line_rec(a, b, na, nb, kind, pa, pb):
        r = np.zeros(2 * len(a), dtype=[('p', '<i2', 3), ('n', 'i1', 3), ('k', 'u1'), ('t', 'u1'), ('pad', 'u1')])
        r['p'][0::2] = np.round(P(a) / q); r['p'][1::2] = np.round(P(b) / q)
        r['n'][0::2] = np.round(na * 127); r['n'][1::2] = np.round(nb * 127)
        r['k'] = kind
        r['t'][0::2] = pa; r['t'][1::2] = pb
        return r

    lines = np.concatenate([
        line_rec(a_face[face_edges[:, 0]], a_face[face_edges[:, 1]], fn0[face_edges[:, 0]], fn0[face_edges[:, 1]], L_FACE, 0, 0),
        line_rec(gp[head_edges[:, 0]], gp[head_edges[:, 1]], gn[head_edges[:, 0]], gn[head_edges[:, 1]], L_HEAD, 0, 0),
        # leader lines: both ends at the base; the tip is base + direction * length * growth
        line_rec(np.array([b for b, _, _ in leaders]), np.array([b for b, _, _ in leaders]),
                 np.array([o for _, o, _ in leaders]), np.array([o for _, o, _ in leaders]), L_LEADER,
                 0, np.round(np.array([l for _, _, l in leaders]) * 255)),
    ])

    # occluders for hidden lines: the face mesh and the head grid outside it
    tri_v = np.concatenate([P(a_face), P(gp)])
    tri_i = np.concatenate([cf, head_tris + len(a_face)]).astype(np.uint16).ravel()
    tv = np.zeros(len(tri_v), dtype=[('p', '<i2', 3), ('pad', '<i2')])
    tv['p'] = np.round(tri_v / q)

    os.makedirs(args.out, exist_ok=True)
    blob = pts.tobytes() + lines.tobytes() + tv.tobytes() + tri_i.tobytes()
    with open(os.path.join(args.out, 'face.bin'), 'wb') as fh:
        fh.write(blob)
    o_lines = pts.nbytes
    o_tv = o_lines + lines.nbytes
    o_ti = o_tv + tv.nbytes
    allp = P(np.concatenate([p_face, p_head]))
    meta = dict(
        posScale=q,
        points=dict(offset=0, count=int(len(pts)), stride=12),
        lines=dict(offset=int(o_lines), count=int(len(lines)), stride=12),
        tris=dict(offset=int(o_tv), vertices=int(len(tv)), stride=8, indexOffset=int(o_ti), indices=int(len(tri_i))),
        iris=dict(radius=float(rad * scale)),
        kinds=dict(skin=K_SKIN, hair=K_HAIR, neck=K_NECK, iris=K_IRIS, contour=K_CONTOUR, node=K_NODE, key=K_KEY, tip=K_TIP),
        frame=dict(x=float(np.abs(allp[allp[:, 1] > P(a_face[152])[1], 0]).max().round(3)),
                   top=float(allp[:, 1].max().round(3)),
                   chin=float(round(P(a_face[152])[1], 3))),
    )
    with open(os.path.join(args.out, 'face.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    print('points', len(pts), 'line vertices', len(lines), 'tri vertices', len(tv), 'indices', len(tri_i),
          'bytes', len(blob), 'kinds', np.bincount(pts['k']).tolist())

    if args.debug:
        os.makedirs(args.debug, exist_ok=True)
        np.savez_compressed(os.path.join(args.debug, 'model.npz'), pts=pts, lines=lines, q=q)


if __name__ == '__main__':
    main()
