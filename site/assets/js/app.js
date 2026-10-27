/* ============================================================================
   vincentnap.com — the opening, the clock, the tabs.

   The opening, in order (seconds after the dots start):
     0.0   the room is there: glow, grid, rain
     0.0   the dots fly in from in front of the screen and swirl to the face
     0.6   the clock comes up, top right
     3.3   the brackets lock onto the face; the scan line leaves the wireframe
     4.0   name and tabs slide in from the left
     4.6   tracking points light up, leader lines grow
     5.2   the readout types in next to the face
   A click or key press during the opening skips to the end. With reduced
   motion everything is simply there.

   Tabs are hash links (#cv, #projects, #history, #contact), so a link on a
   CV or LinkedIn can open one directly.
   ========================================================================== */
import { createFace, ASSEMBLED_AT, FINISHED_AT } from './face.js?v=27756c5e1c';

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

function layoutFace() {
  if (!face) return;
  const open = body.classList.contains('open');
  if (!open) { face.layout(0, narrow.matches ? 0.06 : 0, 1); return; }
  if (narrow.matches) { face.layout(0, 0.35, 0.55); return; }
  // the panel takes the stage; the face steps right and waits behind its glass,
  // or, on screens wide enough to leave room next to the panel, beside it
  const rect = panel.getBoundingClientRect();
  const free = window.innerWidth - rect.right;
  if (free > 320) {
    const nx = ((rect.right + free / 2) / window.innerWidth) * 2 - 1;
    face.layout(nx, -0.04, Math.max(0.42, Math.min(0.62, free / window.innerWidth * 2.1)));
  } else {
    face.layout(0.55, -0.04, 0.62);
  }
}

function show(id) {
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
  requestAnimationFrame(layoutFace);
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
narrow.addEventListener('change', layoutFace);
window.addEventListener('resize', layoutFace);

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
      readout.style.top = `${Math.min(window.innerHeight - 190, r.top + r.height + 2 * padY + 14)}px`;
    } else {
      const left = r.left + r.width + padX + 36;
      readout.style.left = `${Math.min(left, window.innerWidth - 300)}px`;
      readout.style.top = `${r.top + r.height * 0.42}px`;
    }
  });
}

/* ---------- start ---------- */
async function main() {
  startClock();
  wireContact();
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
    face.look(nx, ny);
  }, { passive: true });

  document.addEventListener('visibilitychange', () => {
    const hidden = document.visibilityState === 'hidden';
    root.classList.toggle('rustig', hidden);
    if (hidden) face.pause(); else face.resume();
  });
}

main();
