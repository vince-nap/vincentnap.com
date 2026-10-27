/* ============================================================================
   vincentnap.com — the opening, the clock, the tabs.

   The opening, in order (seconds after the dots start):
     0.0   the room is there: glow, grid, rain
     0.0   the dots fly in from in front of the screen and swirl to the face
     0.6   the clock comes up, top right
     3.3   the brackets lock onto the face; a scan line runs down it
     4.0   name and tabs slide in from the left
     5.2   the readout types in next to the face
   A click or key press during the opening skips to the end. With reduced
   motion everything is simply there.

   Opening a tab blows the head apart into dots that gather again beside the
   panel; closing it brings them back to the middle (face.layout, burst).

   An easter egg: four quick taps or clicks on the face open a code box; the
   code hands over a beard trimmer (wireTrimmer).

   Tabs are hash links (#cv, #projects, #history, #contact), so a link on a
   CV or LinkedIn can open one directly.
   ========================================================================== */
import { createFace, ASSEMBLED_AT, FINISHED_AT } from './face.js?v=afc95fd133';

const root = document.documentElement;
root.classList.remove('no-js');
const body = document.body;
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const narrow = window.matchMedia('(max-width: 820px)');
const $ = (sel) => document.querySelector(sel);

/* ---------- the clock ---------- */
function startClock() {
  const datum = $('#klok-datum'), uur = $('#klok-uur'), sec = $('#klok-sec'), zone = $('#klok-zone');
  const long = new Intl.DateTimeFormat('en-GB', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
  const short = new Intl.DateTimeFormat('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' });
  const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || '';
  const city = (tz.split('/').pop() || '').replace(/_/g, ' ');
  function tick() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    datum.textContent = (narrow.matches ? short : long).format(d).replace(',', '');
    uur.textContent = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
    sec.textContent = `:${pad(d.getSeconds())}`;
    const off = -d.getTimezoneOffset();
    const sign = off >= 0 ? '+' : '−';
    const h = Math.floor(Math.abs(off) / 60), m = Math.abs(off) % 60;
    zone.textContent = `${city ? city + ' · ' : ''}UTC${sign}${h}${m ? ':' + pad(m) : ''}`;
    setTimeout(tick, 1000 - (Date.now() % 1000) + 5);
  }
  tick();
  $('#jaar').textContent = String(new Date().getFullYear());
}

/* ---------- the opening ---------- */
const phases = [
  ['fase-klok', 0.6],
  ['fase-lock', ASSEMBLED_AT - 0.45],
  ['fase-menu', ASSEMBLED_AT + 0.2],
  ['fase-readout', ASSEMBLED_AT + 1.4],
];
let timers = [];
function runPhases(scale = 1) {
  timers.forEach(clearTimeout);
  timers = phases.map(([cls, at]) => setTimeout(() => body.classList.add(cls), at * 1000 * scale));
  timers.push(setTimeout(endIntro, FINISHED_AT * 1000 * scale));
}
function allPhases() {
  timers.forEach(clearTimeout);
  phases.forEach(([cls]) => body.classList.add(cls));
  endIntro();
}
function endIntro() {
  body.classList.remove('intro');
}

/* ---------- tabs ---------- */
const panel = $('#paneel');
const watermark = $('#watermerk');
const tabs = [...document.querySelectorAll('#menu [data-tab]')];
let face = null;

function setWatermark(text) {
  if (watermark.textContent === text) return;
  watermark.classList.add('wissel');
  setTimeout(() => { watermark.textContent = text; watermark.classList.remove('wissel'); }, 260);
}

// burst: the head blows apart into dots and gathers at its new place (a tab opening
// or closing); otherwise (a resize) it glides there
let settling = 0;
function layoutFace(burst = false) {
  if (!face) return;
  const open = body.classList.contains('open');
  const opts = { burst: burst === true };
  if (opts.burst) {
    // the readout stays away until the dots have gathered again
    body.classList.add('verplaatst');
    clearTimeout(settling);
    settling = setTimeout(() => body.classList.remove('verplaatst'), 1500);
  }
  if (!open) { face.layout(0, narrow.matches ? 0.06 : 0, 1, opts); return; }
  if (narrow.matches) { face.layout(0, 0.35, 0.55, opts); return; }
  // the panel takes the stage; the face steps right and waits behind its glass,
  // or, on screens wide enough to leave room next to the panel, beside it
  // (offsetLeft/Width: the panel's place without the slide it opens with)
  const right = panel.offsetLeft + panel.offsetWidth;
  const free = window.innerWidth - right;
  if (free > 320) {
    const nx = ((right + free / 2) / window.innerWidth) * 2 - 1;
    face.layout(nx, -0.04, Math.max(0.42, Math.min(0.62, free / window.innerWidth * 2.1)), opts);
  } else {
    face.layout(0.55, -0.04, 0.62, opts);
  }
}

function show(id) {
  const wasOpen = body.classList.contains('open');
  const pane = id && document.getElementById(id);
  const valid = pane && pane.classList.contains('pane');
  document.querySelectorAll('.pane').forEach((p) => { p.hidden = p !== pane; });
  tabs.forEach((a) => a.toggleAttribute('aria-current', false));
  if (!valid) {
    body.classList.remove('open');
    setWatermark('Vincent Nap');
    document.title = 'Vincent Nap';
  } else {
    const tab = tabs.find((a) => a.dataset.tab === id);
    if (tab) tab.setAttribute('aria-current', 'page');
    body.classList.add('open');
    panel.scrollTop = 0;
    setWatermark(pane.dataset.title);
    document.title = `${pane.dataset.title} — Vincent Nap`;
  }
  const changed = wasOpen !== body.classList.contains('open');
  requestAnimationFrame(() => layoutFace(changed));
}

function fromHash() { show(decodeURIComponent(location.hash.slice(1))); }

function close() {
  if (location.hash) history.pushState(null, '', location.pathname + location.search);
  show(null);
}

window.addEventListener('hashchange', fromHash);
$('.sluit').addEventListener('click', close);
$('.merk').addEventListener('click', (e) => { e.preventDefault(); close(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && body.classList.contains('open')) close(); });
tabs.forEach((a) => a.addEventListener('click', (e) => {
  // clicking the open tab again closes it
  if (a.getAttribute('aria-current') === 'page') { e.preventDefault(); close(); }
}));
// a click outside the panel and the menu closes it (only a click that also started
// outside, so selecting text in the panel and letting go outside does not)
let downOutside = false;
const outside = (el) => !el.closest('#paneel, #menu');
document.addEventListener('pointerdown', (e) => { downOutside = outside(e.target); });
document.addEventListener('click', (e) => {
  if (body.classList.contains('open') && downOutside && outside(e.target)) close();
  downOutside = false;
});
narrow.addEventListener('change', () => layoutFace());
window.addEventListener('resize', () => layoutFace());

/* ---------- skills: a card left alone on its row is centred (see site.css) ---------- */
function wireSkills() {
  const items = [...document.querySelectorAll('.vaardigheden > li')];
  const arrange = () => {
    let run = [];
    const flush = () => { if (run.length % 2) run[run.length - 1].classList.add('alleen'); run = []; };
    for (const li of items) {
      li.classList.remove('alleen');
      if (li.firstElementChild.open) flush(); else run.push(li);
    }
    flush();
  };
  items.forEach((li) => li.firstElementChild.addEventListener('toggle', arrange));
  arrange();
}

/* ---------- contact ---------- */
function wireContact() {
  const mail = $('#mail');
  const addr = `${mail.dataset.u}@${mail.dataset.d}`;
  mail.href = `mailto:${addr}`;
  mail.textContent = addr;
  const btn = $('#kopieer');
  btn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(addr);
      btn.textContent = 'Copied';
      btn.classList.add('gelukt');
    } catch {
      btn.textContent = 'Select & copy';
    }
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('gelukt'); }, 1800);
  });
  wireQuestion(addr);
}

/* ---------- ask a question: straight to the inbox, through Web3Forms ----------
   The access key in data-key is public by design (it only names the inbox).
   Without a key, or when sending fails, the visitor's own mail program takes over. */
function wireQuestion(addr) {
  const form = $('#vraag');
  const status = form.querySelector('.vraag-status');
  const send = form.querySelector('button[type="submit"]');
  const say = (text, cls = '') => { status.textContent = text; status.className = `vraag-status ${cls}`; };
  const mailto = (d) => `mailto:${addr}?subject=${encodeURIComponent('Question via vincentnap.com')}` +
    `&body=${encodeURIComponent(d.name ? `${d.message}\n\n— ${d.name}` : d.message)}`;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!form.reportValidity()) return;
    const d = Object.fromEntries(new FormData(form));
    if (d.botcheck) return;                                   // the honeypot: only bots tick it
    if (!form.dataset.key) { location.href = mailto(d); return; }
    send.disabled = true;
    say('Sending…');
    try {
      const res = await fetch('https://api.web3forms.com/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({
          access_key: form.dataset.key,
          subject: `Question via vincentnap.com${d.name ? ` — ${d.name}` : ''}`,
          from_name: 'vincentnap.com',
          name: d.name, email: d.email, message: d.message,
        }),
      });
      const out = await res.json().catch(() => ({}));
      if (!res.ok || !out.success) throw new Error(out.message || `HTTP ${res.status}`);
      form.reset();
      say('Sent — the answer comes by email', 'goed');
    } catch (err) {
      console.warn('question not sent:', err);
      say('Not sent. ', 'fout');
      const a = document.createElement('a');
      a.href = mailto(d);
      a.textContent = 'Send it by email instead';
      status.append(a);
    } finally {
      send.disabled = false;
    }
  });
}

/* ---------- the frame and readout follow the face ---------- */
function followFace() {
  const lock = $('#lock'), readout = $('#readout');
  face.onFrame(() => {
    const r = face.rect();
    const padX = r.width * 0.06, padY = r.height * 0.04;
    lock.style.left = `${r.left - padX}px`;
    lock.style.top = `${r.top - padY}px`;
    lock.style.width = `${r.width + 2 * padX}px`;
    lock.style.height = `${r.height + 2 * padY}px`;
    if (narrow.matches) {
      readout.style.left = `${Math.max(16, r.left + r.width / 2 - 110)}px`;
      readout.style.top = `${Math.min(window.innerHeight - 250, r.top + r.height + 2 * padY + 14)}px`;
    } else {
      const left = r.left + r.width + padX + 36;
      readout.style.left = `${Math.min(left, window.innerWidth - 340)}px`;
      readout.style.top = `${r.top + r.height * 0.42}px`;
    }
  });
}

/* ---------- the easter egg: four quick taps on the face, a code, a beard trimmer ----------
   With a mouse the trimmer follows the pointer and shaves whatever it passes over;
   on a touch screen it appears while a finger is down, a little above it so the
   finger does not hide what it cuts. face.shave does the cutting; while shaving the
   head turns away from the trimmer, so the side being shaved comes into view. */
const CODE = 'kabouter';
let shaving = false;

function wireTrimmer() {
  const box = $('#code'), form = $('#code-form'), input = $('#code-invoer');
  const status = form.querySelector('.code-status');
  const trimmer = $('#tondeuse'), done = $('#scheer-uit');
  const ui = (el) => el.closest('#menu, #paneel, #klok, #code, .scheer-uit, a, button, input');

  // four quick taps, each within half a second of the last, on the face (not on the menu or a panel)
  let taps = [];
  document.addEventListener('pointerdown', (e) => {
    if (shaving || !box.hidden || body.classList.contains('intro') || body.classList.contains('open') || ui(e.target)) {
      taps = [];
      return;
    }
    if (taps.length && e.timeStamp - taps[taps.length - 1] > 500) taps = [];
    taps.push(e.timeStamp);
    if (taps.length < 4) return;
    taps = [];
    status.textContent = '';
    status.className = 'code-status';
    input.value = '';
    box.hidden = false;
    // focus inside the tap itself, so a phone opens its keyboard; again as the tap
    // lifts, as a mouse press would otherwise move the focus to what it pressed on
    input.focus();
    document.addEventListener('pointerup', () => input.focus(), { once: true });
  });
  const closeBox = () => { box.hidden = true; input.blur(); };
  box.addEventListener('pointerdown', (e) => { if (e.target === box) closeBox(); });
  box.addEventListener('keydown', (e) => { if (e.key === 'Escape') { e.stopPropagation(); closeBox(); } });
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (input.value.trim().toLowerCase() === CODE) {
      status.textContent = 'Access granted';
      status.className = 'code-status goed';
      setTimeout(() => { closeBox(); start(); }, 650);
    } else {
      status.textContent = 'Access denied';
      status.className = 'code-status fout';
      input.value = '';
      form.classList.remove('schud');
      void form.offsetWidth;                                  // restart the shake
      form.classList.add('schud');
    }
  });

  // the trimmer
  let pressed = false, last = null, buzzing = 0, buzzedAt = 0;
  const show = (on) => { trimmer.classList.toggle('zichtbaar', on); if (!on) last = null; };
  const start = () => { shaving = true; root.classList.add('scheren'); done.hidden = false; };
  const stop = () => { shaving = false; root.classList.remove('scheren'); done.hidden = true; pressed = false; show(false); };
  done.addEventListener('click', stop);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && shaving && box.hidden && !body.classList.contains('open')) stop(); });

  function move(e) {
    if (!shaving) return;
    const touch = e.pointerType !== 'mouse';
    if (body.classList.contains('open') || ui(e.target) || (touch && !pressed)) { show(false); return; }
    const x = e.clientX, y = e.clientY - (touch ? 64 : 0);
    trimmer.style.transform = `translate(${x - 22}px, ${y - 2}px) rotate(-25deg)`;
    show(true);
    // shave along the way since the last event; a jump (a finger set down elsewhere,
    // the pointer back in the window) only moves the trimmer
    const from = last && Math.hypot(x - last[0], y - last[1]) < 160 ? last : [x, y];
    const cut = face.shave(from[0], from[1], x, y, touch ? 26 : 20);
    last = [x, y];
    if (!cut) return;
    trimmer.classList.add('snijdt');
    clearTimeout(buzzing);
    buzzing = setTimeout(() => trimmer.classList.remove('snijdt'), 140);
    if (touch && navigator.vibrate && e.timeStamp - buzzedAt > 120) { navigator.vibrate(12); buzzedAt = e.timeStamp; }
  }
  document.addEventListener('pointermove', move, { passive: true });
  document.addEventListener('pointerdown', (e) => {
    if (!shaving || e.pointerType === 'mouse') return;
    pressed = true;
    last = null;
    move(e);
  });
  const lift = (e) => { if (e.pointerType !== 'mouse') { pressed = false; show(false); } };
  document.addEventListener('pointerup', lift);
  document.addEventListener('pointercancel', lift);
  document.addEventListener('mouseout', (e) => { if (!e.relatedTarget) show(false); });
}

/* ---------- start ---------- */
async function main() {
  startClock();
  wireContact();
  wireSkills();
  const initial = location.hash.slice(1);

  try {
    face = await createFace($('#face'), { reduced });
  } catch (err) {
    console.warn('face unavailable:', err);
    body.classList.add('geen-gl');
    allPhases();
    fromHash();
    return;
  }

  followFace();
  layoutFace();
  face.start();
  wireTrimmer();
  // debugging: ?freeze=2.5 holds the opening at 2.5 s, &look=0.8,0.2 turns the head
  const params = new URLSearchParams(location.search);
  if (params.has('freeze')) {
    face.freeze(parseFloat(params.get('freeze')) || 0);
    allPhases();
    const look = (params.get('look') || '').split(',').map(Number);
    if (look.length === 2 && look.every(Number.isFinite)) face.look(look[0], look[1]);
    return;
  }

  if (reduced) {
    face.skip();
    allPhases();
  } else {
    // arriving on a tab link: a shorter opening, then the tab
    runPhases(initial ? 0.55 : 1);
    if (initial) setTimeout(() => face.skip(), 900);
    const skip = () => {
      if (!body.classList.contains('intro')) return;
      face.skip();
      allPhases();
    };
    window.addEventListener('pointerdown', skip, { once: true });
    window.addEventListener('keydown', skip, { once: true });
  }
  if (initial) setTimeout(fromHash, reduced ? 0 : ASSEMBLED_AT * 550);

  // the head follows the pointer, measured from the face's own centre
  window.addEventListener('pointermove', (e) => {
    const [cx, cy] = face.center();
    const nx = (e.clientX - cx) / (window.innerWidth * 0.5);
    const ny = -(e.clientY - cy) / (window.innerHeight * 0.5);
    // while shaving the head turns away from the trimmer instead, offering that side
    // of the head like someone in the barber's chair (above it, it bows)
    const k = shaving ? -0.75 : 1;
    face.look(nx * k, ny * k);
  }, { passive: true });

  document.addEventListener('visibilitychange', () => {
    const hidden = document.visibilityState === 'hidden';
    root.classList.toggle('rustig', hidden);
    if (hidden) face.pause(); else face.resume();
  });
}

main();
