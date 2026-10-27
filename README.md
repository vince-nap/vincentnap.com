# vincentnap.com

Personal site of Vincent Nap, in the same Detroit: Become Human house style as Argus and Verhaal:
the light menus of the game — white, navy ink, one android-blue accent, thin lines, corner
brackets, spaced capitals.

On load, dots fly in from in front of the screen and swirl into a generic 3D head with long,
wavy hair — all of it dots, no lines; a scan line runs down it. The head follows the cursor, the
date and time tick top right, and five tabs open on the left: Curriculum Vitae, Projects,
Brief History, Contact, Skills. Opening or closing a tab blows the head apart into dots that
gather again at its new place.

```
site/                    everything that is served (Render publishes this folder)
  index.html             all text: CV, projects, history, contact
  assets/css/site.css    the house style
  assets/js/app.js       the opening, the clock, the tabs
  assets/js/face.js      the face in WebGL2 (no libraries)
  assets/face/face.bin   the head model (built by tools/face/build_generic.py)
  assets/fonts/          Barlow (OFL) and TeX Gyre Termes (GUST), served from this site
tools/face/              builds face.bin (see tools/face/README.md)
render.yaml              Render Blueprint: static site, security headers, short links
```

No build step and no dependencies at runtime. To run it locally:

```sh
python3 -m http.server 8000 -d site     # then open http://localhost:8000
```

`http://localhost:8000/?freeze=2.5` holds the opening at 2.5 s, for looking at it frame by frame.

After changing CSS, JS or the face model, run `python3 tools/stamp.py` before committing. It stamps
each file reference with a fingerprint of the contents (`site.css?v=…`), so browsers fetch the new
version instead of a cached old one.

## Editing the content

All text lives in `site/index.html`, one `<section>` per tab. A CV entry is one `<li class="cv-item">`;
a project is one `<article class="project">`; a history step is one `<li class="knoop">`. Copy an
existing one and change the text. The CV tab's "Download CV (PDF)" button serves the original CV,
`site/assets/cv/CV_Vincent_Nap.pdf`: replace that file to update it.

Short links work once deployed: `vincentnap.com/cv`, `/projects`, `/history`, `/contact`, `/skills`
(also `vincentnap.com/#cv` and so on).

A skill in the Skills tab is one `<details class="project vaardigheid">` card: the tags list where
it comes from, and each source in its trace is one `<li class="knoop">`. Every source should be an
entry on the CV.

## The easter egg

Four quick taps or clicks on the face (each within half a second of the last) open a code box.
The code is `kabouter`; it hands over a beard trimmer. With a mouse it follows the pointer; on a
touch screen it appears while a finger is down. Whatever hair it passes over comes loose from
there to the tip and falls; underneath, a buzzcut shows. "Put the trimmer down" (or Escape)
ends it; a reload brings the hair back. The code is in `site/assets/js/app.js` (`CODE`).

## The question form

"Ask a question" in the Contact tab sends the question to the inbox through
[Web3Forms](https://web3forms.com) (free tier, no server needed). To switch it on:

1. Go to web3forms.com, enter the email address that should receive the questions, and open the
   access key they send there.
2. Put the key in `site/index.html`, on the form: `<form class="vraag" id="vraag" data-key="…">`.
   The key is meant to be public — it only names the inbox — so it can sit in the page.

Until a key is set, the form opens the visitor's own mail program with the question filled in.
The Content-Security-Policy in `render.yaml` allows `https://api.web3forms.com` for this.

## Security

The site is static: no login, no database, no server code, nothing a visitor can change. It can
only be changed by pushing to `main` on GitHub (Render deploys that), or by taking over the
domain at Versio. So securing it means securing those three accounts.

```
GitHub (repo, main) ──auto-deploy──► Render ──► vincentnap.com ◄── DNS at Versio
```

In the site itself:

- **Content-Security-Policy**: only the site's own scripts, styles, fonts and images; the form
  may only talk to `api.web3forms.com`. In the headers (render.yaml) and in `index.html` itself.
- **No framing** (clickjacking): `frame-ancestors 'none'` and `X-Frame-Options` in the headers;
  `app.js` also blanks the page if it finds itself in a frame, for when the headers are missing.
- **HSTS, nosniff, Referrer-Policy, Permissions-Policy, COOP, CORP** in the headers.
- No `innerHTML`, `eval` or inline scripts; no third-party scripts, fonts or trackers; every
  external link has `rel="noopener"`.
- `robots.txt` keeps the CV PDF (home address, phone number) out of search results.
- `/.well-known/security.txt` says where to report a problem (renew `Expires` yearly).

**The headers only apply if Render sends them.** render.yaml is read only by a service created
as a Blueprint; this service was created by hand. Either add each header from render.yaml under
**Render → the service → Settings → Headers** (path `/*`), or check what is sent at
securityheaders.com.

Accounts (once):

| Where | What |
|---|---|
| GitHub | Two-factor authentication on. Repo **private** (Settings → General → Danger Zone). Default branch `main`. Ruleset on `main`: block force pushes and deletion. Under Settings → Applications, Render and Claude limited to this repo. |
| Render | Two-factor authentication on. No other team members. Pull Request Previews off. |
| Versio | Two-factor authentication on. Transfer lock (registrar lock) on. DNSSEC on. |
| DNS | The domain sends no mail, so nobody may send it in its name: TXT `@` `v=spf1 -all`; TXT `_dmarc` `v=DMARC1; p=reject; adkim=s; aspf=s`. Optional CAA: `0 issue "letsencrypt.org"` and `0 issue "pki.goog"` (both, or Render cannot renew the certificate). |

## Going live: GitHub → Render → Versio

### 1. Make the repository private
GitHub → `vince-nap/vincentnap.com` → **Settings** → **General** → scroll to **Danger Zone** →
**Change repository visibility** → **Make private**.

### 2. Give the site a `main` branch
The work is on `claude/personal-portfolio-site-xkn81f`. On GitHub, open the branch dropdown, type
`main`, choose **Create branch main from claude/personal-portfolio-site-xkn81f**, then make `main` the
default branch: **Settings** → **General** → **Default branch**. Later changes arrive on the
`claude/…` branch and are merged into `main` with a pull request; Render deploys `main`.

### 3. Create the site on Render
1. Sign in at render.com with GitHub. If it asks, allow the Render GitHub App access to
   `vince-nap/vincentnap.com` (needed because the repo is private).
2. **New** → **Blueprint** → pick the repository, branch `main`. Render reads `render.yaml` and
   creates a static site called `vincentnap` (no build command, publish folder `site`).
   Alternative without the Blueprint: **New** → **Static Site**, publish directory `site`, build
   command empty — but then the security headers and short links in `render.yaml` must be added
   by hand under **Headers** and **Redirects/Rewrites**.
3. Wait for the first deploy, then open the `https://vincentnap.onrender.com` address it shows
   (the exact subdomain is on the service page) and check the site.

Static sites are free on Render and are served from its CDN — no sleeping, no cold starts.

### 4. Add the domain on Render
On the service: **Settings** → **Custom Domains** → **Add Custom Domain** → `vincentnap.com`.
Render normally adds `www.vincentnap.com` with it and redirects one to the other; if it doesn't,
add `www.vincentnap.com` as well. Render then shows the DNS records it expects. Use those
values if they differ from the ones below.

### 5. Point the DNS at Versio to Render
In the Versio control panel, open the domain `vincentnap.com` and its DNS records, then:

| Type  | Name / host | Value                          | TTL   |
|-------|-------------|--------------------------------|-------|
| A     | `@` (empty, the bare domain) | `216.24.57.1`   | 300   |
| CNAME | `www`       | `vincentnap.onrender.com.` (your own onrender.com address) | 300 |

Remove any other **A** and **AAAA** records on `@` and `www` (Versio's parking page usually sets
some). Leave MX/TXT records for e-mail alone. If Versio offers an **ALIAS/ANAME** type for the bare
domain, you may point that at the onrender.com address instead of the A record.

### 6. Verify
Back on Render, press **Verify** next to each domain. DNS can take minutes to a few hours.
Render issues the HTTPS certificate (Let's Encrypt) by itself once the records resolve.
Check with `dig vincentnap.com +short` (should say `216.24.57.1`).

## Rebuilding the face

See `tools/face/README.md`. The site uses the generic head with a purchased, sculpted hairstyle
(the hair model itself is not in git); the photo-based model is kept in `tools/face/build_face.py`
but not used.
