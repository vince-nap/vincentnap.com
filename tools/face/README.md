# The face model

Two builders write the same file, `site/assets/face/face.bin` (and `face.json`): dots on the head,
fine clusters along the features and occluder triangles for hidden dots (the photo build adds
tracking points with leader lines). `site/assets/js/face.js` draws whichever was built last.

**`build_generic.py` — what the site uses.** A neutral head that is nobody in particular: MediaPipe's
canonical face model (an average face), reshaped in `masculine()` (a leaner jaw, a longer
chin, cheekbones with hollows under them, a slight bump on the nose, thinner lips), with a cranium, jaw,
ears and neck made of simple solids around it, and a sculpted hairstyle fitted onto the cranium. Face and hair are the same stipple: no lines.
The hair's dots are scattered along streamlines that run down the sculpt's surface, so the locks
still show. Builds in about half a minute.

The hair is a purchased model (CGTrader, "Hair 52") and may not be redistributed, so it is not in
git; only the dots drawn from it are. To rebuild with it, prepare it once:

```sh
python tools/face/prepare_hair.py path/to/hair.obj     # writes tools/face/input/hair_sculpt.npz
```

Without that file the builder makes a bald head (or `--procedural-hair` for the old generated hair).
The hair is cut at the height of the chin, each lock ending a little higher or lower than the
next. Where the jaw reaches into the hair, the hair is pushed outward clear of the head. The jaw solid
behind the face sits just inside the face's own jaw edge, so only one jawline shows.

Two extras are built in but off on the site: `--glasses` (the frames from the photo, as dots) and
`--birthmark` (the mark by the right nostril).

**`build_face.py` — kept, not used.** A head reconstructed from three photos (front and both
profiles) with hair strands; see the docstring at the top of the file. The photos are not in git;
they go in `tools/face/input/` as `front.jpg`, `left.jpg` and `right.jpg`.

## Running it

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r tools/face/requirements.txt
sh tools/face/fetch_models.sh
python tools/face/build_generic.py
```

On a headless Linux box MediaPipe also needs `libegl1` (`apt-get install libegl1`).
