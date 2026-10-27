# The face model

`build_face.py` turns three phone photos into `site/assets/face/face.bin` (~300 KB): sparse dots,
bright clusters along the features, a wireframe (face mesh plus a lat-long grid over the head),
occluder triangles for hidden lines, and eight tracking points with leader lines.

How the head is made — the docstring at the top of `build_face.py` has the details:

1. MediaPipe finds the 478 face landmarks and segments each photo (hair, skin, clothes).
2. The front landmarks are turned upright and made symmetric (every point averaged with its
   mirror twin); the jaw is squared a little and the forehead brought forward to the profile.
3. The profiles are aligned to the face through its midline, giving the side outline of the head.
4. The rest of the head is inflated from the silhouettes as two Poisson height fields (front and
   back) that meet at a vertical tangent.
5. Dots, contours, wireframe and occluders are sampled and written out.

## Running it

The photos are not in git. Put them in `tools/face/input/` as `front.jpg`, `left.jpg` and
`right.jpg` (in `left.jpg` the nose points to the left of the frame).

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r tools/face/requirements.txt
sh tools/face/fetch_models.sh
python tools/face/build_face.py
```

On a headless Linux box MediaPipe also needs `libegl1` (`apt-get install libegl1`).
