#!/usr/bin/env python3
"""
Stamps the site's assets with a fingerprint of their contents, so a browser never
keeps an old copy after a deploy (the file names themselves never change):

  index.html   /assets/css/site.css?v=<hash>   /assets/js/app.js?v=<hash>
  app.js       import ... from './face.js?v=<hash>'
  face.json    "version": <hash of face.bin>   (face.js loads face.bin?v=<version>)

Run after changing CSS, JS or the face model:
    python3 tools/stamp.py
"""
import hashlib
import json
import os
import re

SITE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'site')


def digest(rel):
    with open(os.path.join(SITE, rel), 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:10]


def rewrite(rel, pattern, replacement):
    path = os.path.join(SITE, rel)
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    new, n = re.subn(pattern, replacement, text)
    if n == 0:
        raise SystemExit(f'{rel}: nothing matched {pattern}')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(new)


def main():
    # the model first, then the scripts that refer to it, then the page
    meta_path = os.path.join(SITE, 'assets/face/face.json')
    with open(meta_path) as fh:
        meta = json.load(fh)
    meta['version'] = digest('assets/face/face.bin')
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=1)
    rewrite('assets/js/app.js', r"'\./face\.js(\?v=[0-9a-f]+)?'", f"'./face.js?v={digest('assets/js/face.js')}'")
    rewrite('index.html', r'/assets/js/app\.js(\?v=[0-9a-f]+)?', f"/assets/js/app.js?v={digest('assets/js/app.js')}")
    rewrite('index.html', r'/assets/css/site\.css(\?v=[0-9a-f]+)?', f"/assets/css/site.css?v={digest('assets/css/site.css')}")
    print('stamped: face.bin', meta['version'], '· face.js', digest('assets/js/face.js'),
          '· app.js', digest('assets/js/app.js'), '· site.css', digest('assets/css/site.css'))


if __name__ == '__main__':
    main()
