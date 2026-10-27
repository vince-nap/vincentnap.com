#!/usr/bin/env python3
"""
Reduces a sculpted hair model (OBJ, as exported by ZBrush and the like) to a
mesh small enough to work with, and stores it for build_generic.py:

    python tools/face/prepare_hair.py path/to/hair.obj

writes tools/face/input/hair_sculpt.npz (about 80,000 triangles, in the model's
own units; build_generic.py fits it onto the head). The input folder is not in
git: the hair is a purchased model and must not be redistributed. Only the lines
and dots drawn from it end up in the site.
"""
import os
import sys

import numpy as np
import fast_simplification

HERE = os.path.dirname(os.path.abspath(__file__))


def read_obj(path):
    vs, fs = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith('v '):
                vs.append(line[2:])
            elif line.startswith('f '):
                idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):                    # fan-triangulate quads and n-gons
                    fs.append((idx[0], idx[k], idx[k + 1]))
    v = np.array(' '.join(vs).split(), dtype=float).reshape(-1, 3)
    return v, np.array(fs, dtype=np.int64)


def main():
    src = sys.argv[1]
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 80000
    v, f = read_obj(src)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    print('read', len(v), 'vertices,', len(f), 'triangles')
    v2, f2 = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=1 - target / len(f))
    os.makedirs(os.path.join(HERE, 'input'), exist_ok=True)
    out = os.path.join(HERE, 'input', 'hair_sculpt.npz')
    np.savez_compressed(out, V=v2.astype(np.float32), F=f2.astype(np.int32))
    print('wrote', out, len(v2), 'vertices,', len(f2), 'triangles')


if __name__ == '__main__':
    main()
