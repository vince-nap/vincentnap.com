# The face model

Two builders write the same file, `site/assets/face/face.bin` (and `face.json`): a wireframe head
with sparse dots, bright clusters along the features, occluder triangles for hidden lines, and
eight tracking points with leader lines. `site/assets/js/face.js` draws whichever was built last.

**`build_generic.py` — what the site uses.** A neutral head that is nobody in particular: MediaPipe's
canonical face model (an average face) with a cranium, jaw, ears and neck made of simple solids around
it. No photos needed; it builds in a few seconds.

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
