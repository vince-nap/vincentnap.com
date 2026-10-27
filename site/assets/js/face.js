/* ============================================================================
   The face: a generic head with sculpted hair, all of it dots
   (tools/face/build_generic.py), drawn with WebGL2.

   The opening, all in the shaders:
     0.0 - 3.75 s  the dots start close to the camera, all over the screen, and
                   swirl in to their place on the face (ease-out: a fast start
                   and a long run-out, like the rest of the site)
     3.3 - 4.8 s   a scan line runs down the head, lighting up the dots
                   it passes
   After that the head turns towards the cursor with a soft spring, the eyes
   lead the head and the scan line returns now and then.

   Hidden dots: the face mesh, the head grid and a coarse copy of the hair are
   drawn into the depth buffer first (no colour, pushed a little away from the
   camera), and the dots are tested against them. Without that the back of the
   head would show through the face.

   Shaving (the easter egg in app.js): the hair and the buzzcut under it are
   drawn in a second call with a little state per dot (shaver below). A lock
   the trimmer touches comes loose from there to its tip and falls as one
   strand: gravity against the drag of the air, a slow tumble, then it fades.
   Stray dots left with no lock nearby fall too; where no hair is left, the
   stubble shows and the hair's part of the depth pass goes.

   The line programs and the tracking-point kinds below draw nothing for the
   generic build (no wireframe, no hair strands, no tracking points or leader
   lines); the photo build and --procedural-hair still write them.
   ========================================================================== */

const COMMON = `
uniform float uPosScale;
uniform mat4 uProj;
uniform mat3 uRot;
uniform vec3 uPivot;
uniform float uDist;
uniform vec2 uShift;
uniform float uScale;
vec3 place(vec3 m) { return uRot * (m - uPivot) + uPivot; }
vec4 projectAt(vec3 q, vec2 shift, float scale) {
  vec4 clip = uProj * vec4(q.x, q.y, q.z - uDist, 1.0);
  clip.xy = clip.xy * scale + shift * clip.w;
  return clip;
}
vec4 project(vec3 q) { return projectAt(q, uShift, uScale); }
float facingCam(vec3 p, vec3 n) { return dot(n, normalize(vec3(0.0, 0.0, uDist) - p)); }
`;

const POINT_VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec2 aKT;
layout(location = 3) in vec4 aFall;    // hair: place in its strand when it came loose (xyz) and when (w; < 0 attached). Stubble: shown since (w)
layout(location = 4) in vec4 aDrift;   // the strand's velocity when it came loose (xyz) and its tumble (w)
layout(location = 5) in vec3 aCentre;  // the strand's centre when it came loose
${COMMON}
uniform int uIdBase;       // the first dot of this draw call, so every dot keeps its own random numbers
uniform float uTanHalf;
uniform float uAspect;
uniform float uTime;
uniform float uViewH;
uniform vec2 uGaze;
uniform float uScan;
uniform float uFlight;
uniform float uKeys;
uniform float uLeader;
uniform float uBurst;      // seconds since a tab opened or closed, < 0 when settled
uniform vec2 uShift0;      // where the head was before that
uniform float uScale0;
out vec4 vColor;
flat out int vShape;

uint hash(uint x) {
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  x ^= x >> 16; return x;
}
float rnd(uint i, uint k) { return float(hash(i * 8u + k) & 0xffffffu) / 16777216.0; }

void main() {
  uint id = uint(gl_VertexID + uIdBase);
  float r1 = rnd(id, 0u), r2 = rnd(id, 1u), r3 = rnd(id, 2u), r4 = rnd(id, 3u), r5 = rnd(id, 4u);
  int kind = int(aKT.x + 0.5);
  float tone = aKT.y / 255.0;

  vec3 target = aPos * uPosScale;
  if (kind == 3) target.xy += uGaze;
  if (kind == 7) target += aNrm * tone * uLeader;
  vec3 p = place(target);
  vec3 n = uRot * aNrm;
  if (kind == 7) n = uRot * vec3(0.0, 0.0, 1.0);

  // ---- the flight in (dots and nodes; the tracking points come later)
  bool flies = kind <= 5 || kind == 8;
  float delay = 0.1 + r3 * 1.15 + 0.45 * smoothstep(0.4, 1.7, length(target.xy));
  float dur = 2.0 + r4 * 0.7;
  float t = flies ? clamp((uTime - delay) / dur, 0.0, 1.0) : 1.0;
  t = mix(1.0, t, uFlight);
  float e = 1.0 - pow(1.0 - t, 4.0);
  float zs = uDist - 0.7 - r5 * 2.6;
  float spread = (uDist - zs) * uTanHalf * 1.35;
  vec3 start = vec3((r1 * 2.0 - 1.0) * spread * uAspect, (r2 * 2.0 - 1.0) * spread, zs);
  vec3 q = mix(start, p, e);
  float ang = (1.0 - e) * (1.0 - e) * (2.0 + 1.8 * r2) * 3.14159;
  vec2 d = q.xy - vec2(0.0, 0.05);
  q.xy = vec2(0.0, 0.05) + vec2(cos(ang) * d.x - sin(ang) * d.y, sin(ang) * d.x + cos(ang) * d.y);
  float landed = smoothstep(0.82, 1.0, e);
  q += n * 0.004 * sin(uTime * 1.3 + r1 * 6.2832) * landed * float(kind <= 2);

  // ---- the burst: opening or closing a tab blows the head apart into dots, which
  // swirl over to the head's new place and settle there again
  float lift = 0.0, go = 1.0;
  if (uBurst >= 0.0 && flies) {
    float bt = clamp((uBurst - r3 * 0.3) / (1.0 + r4 * 0.45), 0.0, 1.0);
    lift = smoothstep(0.0, 0.28, bt) * (1.0 - smoothstep(0.28, 1.0, bt));
    go = smoothstep(0.12, 0.8, bt);
    vec3 away = normalize(target - vec3(0.0, 0.1, -0.4) + 1e-4);
    vec3 dir = normalize(vec3(r1, r2, r5) * 2.0 - 1.0 + away * 0.9);
    q += dir * (0.45 + 1.1 * r4) * lift;
    float spin = lift * (0.9 + 1.2 * r2);
    vec2 c = q.xy - vec2(0.0, 0.05);
    q.xy = vec2(0.0, 0.05) + vec2(cos(spin) * c.x - sin(spin) * c.y, sin(spin) * c.x + cos(spin) * c.y);
  }
  // ---- shaved off: the strand slides off the head and falls with gravity against
  // the drag of the air (towards a slow terminal speed, like hair; sideways the air
  // slows it more gently), tumbling slowly, fluttering a little
  bool falling = kind == 1 && aFall.w >= 0.0;
  float ft = falling ? max(uTime - aFall.w, 0.0) : 0.0;
  if (falling) {
    float tau = 0.3, tauH = 0.9, vt = 3.0;
    float ex = 1.0 - exp(-ft / tau), eh = 1.0 - exp(-ft / tauH);
    float a = aDrift.w * tau * ex;
    vec3 o = aFall.xyz;
    o.xy = vec2(cos(a) * o.x - sin(a) * o.y, sin(a) * o.x + cos(a) * o.y);
    q = aCentre + o + vec3(aDrift.x * tauH * eh, (aDrift.y + vt) * tau * ex - vt * ft, aDrift.z * tauH * eh);
    q.x += sin(ft * 5.0 + r1 * 6.2832) * 0.012 * min(ft, 1.0);
    lift = 0.0; go = 1.0;
  }
  float settled = landed * (1.0 - lift);
  float scale = mix(uScale0, uScale, go);

  vec4 clip = go < 1.0 ? mix(projectAt(q, uShift0, uScale0), project(q), go) : project(q);
  gl_Position = clip;

  // ---- look
  vec3 L = normalize(vec3(-0.45, 0.55, 0.75));
  float dif = max(dot(n, L), 0.0);
  float ink = clamp(tone * 0.95 + (1.0 - dif) * 0.22 - 0.05, 0.0, 1.0);
  float vis = mix(1.0, smoothstep(-0.12, 0.25, facingCam(p, n)), settled);
  vis *= kind == 1 ? smoothstep(-2.25, -1.75, target.y) : smoothstep(-1.8, -1.2, target.y);
  if (falling) vis = (1.0 - smoothstep(1.4, 2.5, ft)) * smoothstep(-2.7, -2.0, q.y);
  float band = uScan > -9.0 ? exp(-pow((target.y - uScan) / 0.04, 2.0)) * landed : 0.0;

  vec3 light = vec3(0.56, 0.76, 0.95);
  vec3 deep = vec3(0.03, 0.2, 0.56);
  vec3 accent = vec3(0.114, 0.435, 0.878);

  float size;
  float alpha;
  vec3 col;
  vShape = 0;
  if (kind <= 2) {                       // sparse surface dots
    size = 0.0066 * (0.7 + 0.9 * ink);   // skin and hair alike: one stipple, the tone does the rest
    alpha = 0.18 + 0.62 * ink;
    col = mix(light, deep, ink);
  } else if (kind == 3) {                // irises
    size = 0.0085; alpha = 0.9; col = deep;
  } else if (kind == 4) {                // feature clusters
    size = 0.0052 * (0.85 + 0.4 * tone); alpha = 0.45 + 0.4 * tone; col = mix(accent, deep, 0.55);
  } else if (kind == 5) {                // wireframe nodes, twinkling
    size = 0.005 + 0.009 * tone; alpha = (0.35 + 0.45 * tone) * (0.7 + 0.3 * sin(uTime * 1.7 + r1 * 6.2832));
    col = accent;
  } else if (kind == 6) {                // tracking points
    float pop = clamp(uKeys, 0.0, 1.0);
    size = 0.034 * (pop + 0.25 * sin(pop * 3.14159)); alpha = pop; col = deep; vShape = 1;
  } else if (kind == 8) {                // the buzzcut, once the hair over it is gone
    size = 0.0046 * (0.8 + 0.5 * tone);
    alpha = (0.3 + 0.55 * tone) * (aFall.w >= 0.0 ? smoothstep(0.0, 0.4, uTime - aFall.w) : 0.0);
    col = mix(light, deep, 0.55 + 0.45 * tone);
  } else {                               // the rings at the ends of the leader lines
    size = 0.03 * smoothstep(0.6, 1.0, uLeader); alpha = smoothstep(0.6, 1.0, uLeader); col = accent; vShape = 2;
  }
  col = mix(accent, col, settled);
  col = mix(col, accent, band * 0.8);
  alpha = alpha * vis;
  alpha = mix(0.3 + 0.2 * r4, alpha, settled) * smoothstep(0.0, 0.06, t);
  alpha = min(1.0, alpha + band * 0.45 * vis);

  float px = size * uViewH / (2.0 * uTanHalf * clip.w) * scale * (1.0 + band * 0.5);
  px = mix(max(px, 1.5), clamp(px * 1.6, 1.5, 22.0), 1.0 - settled);
  gl_PointSize = falling && ft > 2.6 ? 0.0 : px;
  vColor = vec4(col, alpha);
}`;

const POINT_FRAG = `#version 300 es
precision mediump float;
in vec4 vColor;
flat in int vShape;
out vec4 outColor;
void main() {
  float r = length(gl_PointCoord - 0.5);
  float a;
  if (vShape == 1) {          // a solid core in a thin ring
    a = smoothstep(0.2, 0.14, r) + 0.85 * (smoothstep(0.34, 0.39, r) - smoothstep(0.43, 0.48, r));
  } else if (vShape == 2) {   // a ring
    a = smoothstep(0.3, 0.36, r) - smoothstep(0.42, 0.48, r);
  } else {
    a = smoothstep(0.5, 0.3, r);
  }
  a *= vColor.a;
  if (a < 0.004) discard;
  outColor = vec4(vColor.rgb * a, a);
}`;

const LINE_VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec2 aKT;
${COMMON}
uniform float uRevealY;
uniform float uScan;
uniform float uLeader;
uniform float uHair;
uniform float uTime;
out vec4 vColor;
void main() {
  int kind = int(aKT.x + 0.5);
  float par = aKT.y / 255.0;
  vec3 target = aPos * uPosScale;
  vec3 n = aNrm;
  if (kind == 2) { target += aNrm * par * uLeader; n = vec3(0.0, 0.0, 1.0); }
  // hair: the tips sway a little
  if (kind == 3) target.x += sin(uTime * 0.8 + target.y * 2.5) * 0.006 * par * par;
  vec3 p = place(target);
  n = uRot * n;
  gl_Position = project(p);

  float vis = smoothstep(-0.1, 0.3, facingCam(p, n));
  vis *= kind == 3 ? smoothstep(-2.25, -1.75, target.y) : smoothstep(-1.8, -1.2, target.y);
  float base = kind == 0 ? 0.3 : (kind == 1 ? 0.17 : (kind == 2 ? 0.7 : 0.48));
  float shown;
  if (kind == 2) shown = step(0.001, uLeader);
  else if (kind == 3) shown = smoothstep(par - 0.05, par, uHair);   // strands grow from the root
  else shown = smoothstep(uRevealY, uRevealY + 0.07, target.y);
  float band = uScan > -9.0 ? exp(-pow((target.y - uScan) / 0.05, 2.0)) : 0.0;
  float alpha = min(1.0, base + band * 0.5) * vis * shown;
  float deepMix = kind == 2 ? 0.4 : (kind == 3 ? 0.6 : 0.0);
  vColor = vec4(mix(vec3(0.114, 0.435, 0.878), vec3(0.03, 0.2, 0.56), deepMix), alpha);
}`;

const LINE_FRAG = `#version 300 es
precision mediump float;
in vec4 vColor;
out vec4 outColor;
void main() {
  if (vColor.a < 0.004) discard;
  outColor = vec4(vColor.rgb * vColor.a, vColor.a);
}`;

const DEPTH_VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
${COMMON}
void main() {
  vec3 p = place(aPos * uPosScale);
  p.z -= 0.022;   // a little behind the surface, so whatever lies on it stays visible
  gl_Position = project(p);
}`;

const DEPTH_FRAG = `#version 300 es
precision mediump float;
out vec4 outColor;
void main() { outColor = vec4(0.0); }`;

const FOV = 28 * Math.PI / 180;
export const ASSEMBLED_AT = 3.75;              // the last dot has landed
export const FINISHED_AT = ASSEMBLED_AT + 1.8; // the scan is done
const BURST_FOR = 1.8;                         // the burst, from blowing apart to settled

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function program(gl, vs, fs, names) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  for (const n of names) u[n] = gl.getUniformLocation(p, n);
  return { p, u };
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}

function rotation(yaw, pitch) {
  // R = Rx(pitch) * Ry(yaw), column-major
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  return new Float32Array([cy, sp * sy, -cp * sy, 0, cp, sp, sy, -sp * cy, cp * cy]);
}

const ease = (x) => 1 - Math.pow(1 - Math.min(1, Math.max(0, x)), 3);

/* ---------- shaving: the CPU side ----------
   Which dots the trimmer touches, which locks come loose (from there to the tip:
   the dots of a lock follow each other in face.bin from the root down), which
   stray dots and which stubble follow, and which triangles of the hair's depth
   pass go. Stubble and triangles go with the hair dot straight above them
   (face.bin); stray dots fall when no lock is left near them, found through a
   coarse grid over the hair, each cell counting the attached lock dots it holds. */
const PIVOT = [0, -0.15, -0.4];          // as uPivot
const CELL = 0.12;                        // grid cell, model units (about 1.2 cm)
const STATE = 11;                         // floats of state per dot: aFall 4, aDrift 4, aCentre 3

function createShaver(gl, bin, meta, triVao) {
  const sh = meta.shave;
  const first = sh.hairFirst, nh = sh.hairCount, n = nh + sh.stubbleCount;
  const raw = new Int16Array(bin, meta.points.offset + first * 12, n * 6);
  const rawN = new Int8Array(bin, meta.points.offset + first * 12, n * 12);
  const lock = new Int16Array(bin, sh.locksOffset, nh);
  const pos = new Float32Array(n * 3), nrm = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      pos[i * 3 + k] = raw[i * 6 + k] * meta.posScale;
      nrm[i * 3 + k] = rawN[i * 12 + 6 + k] / 127;
    }
  }

  // the state the shader reads, and the vertex array of this second draw call
  const dyn = new Float32Array(n * STATE);
  for (let i = 0; i < n; i++) dyn[i * STATE + 3] = -1;
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(bin, meta.points.offset + first * 12, n * 12), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.SHORT, false, 12, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.BYTE, true, 12, 6);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 2, gl.UNSIGNED_BYTE, false, 12, 9);
  const dynBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, dynBuf);
  gl.bufferData(gl.ARRAY_BUFFER, dyn, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(3);
  gl.vertexAttribPointer(3, 4, gl.FLOAT, false, STATE * 4, 0);
  gl.enableVertexAttribArray(4);
  gl.vertexAttribPointer(4, 4, gl.FLOAT, false, STATE * 4, 16);
  gl.enableVertexAttribArray(5);
  gl.vertexAttribPointer(5, 3, gl.FLOAT, false, STATE * 4, 32);
  gl.bindVertexArray(null);

  // the hair's triangles in the depth pass
  const idx = new Uint16Array(bin, meta.tris.indexOffset, meta.tris.indices);
  const base = meta.tris.hairIndexStart;
  const nt = (idx.length - base) / 3;
  const triAlive = new Uint8Array(nt).fill(1);
  const idxOut = new Uint16Array(idx);
  let triCount = idx.length, trisDirty = false;

  // the grid
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const grow = (a, i) => { for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], a[i * 3 + k]); hi[k] = Math.max(hi[k], a[i * 3 + k]); } };
  for (let i = 0; i < nh; i++) grow(pos, i);
  const dim = [0, 1, 2].map((k) => Math.ceil((hi[k] - lo[k]) / CELL) + 3);
  const cellAt = (a, i) => {
    const c = [0, 1, 2].map((k) => Math.floor((a[i * 3 + k] - lo[k]) / CELL) + 1);
    return (c[0] * dim[1] + c[1]) * dim[2] + c[2];
  };
  const cells = dim[0] * dim[1] * dim[2];
  const cell = new Int32Array(nh);
  for (let i = 0; i < nh; i++) cell[i] = cellAt(pos, i);
  const lockCount = new Int32Array(cells);
  const attached = new Uint8Array(nh).fill(1);
  for (let i = 0; i < nh; i++) if (lock[i] >= 0) lockCount[cell[i]]++;
  const near = (count, c) => {
    let sum = 0;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      sum += count[c + (dx * dim[1] + dy) * dim[2] + dz];
    }
    return sum;
  };
  // lists by key (compact): per cell its stray dots; per hair dot the stubble and the triangles under it
  function byKey(list, keyOf, keys) {
    const start = new Int32Array(keys + 1);
    for (const i of list) start[keyOf(i) + 1]++;
    for (let c = 0; c < keys; c++) start[c + 1] += start[c];
    const out = new Int32Array(list.length), fill = start.slice(0, keys);
    for (const i of list) out[fill[keyOf(i)]++] = i;
    return { start, out };
  }
  const strays = [];
  for (let i = 0; i < nh; i++) if (lock[i] < 0) strays.push(i);
  const strayAt = byKey(strays, (i) => cell[i], cells);
  const above = new Uint16Array(bin, sh.aboveOffset, n - nh);
  const stubbleOf = byKey([...Array(n - nh).keys()].map((j) => nh + j), (i) => above[i - nh], nh);
  const triAbove = new Uint16Array(bin, sh.triAboveOffset, nt);
  const trisOf = byKey([...Array(nt).keys()], (k) => triAbove[k], nh);

  // the locks: each from its first dot; the attached part ends at lockEnd
  let nl = 0;
  for (let i = 0; i < nh; i++) nl = Math.max(nl, lock[i] + 1);
  const lockEnd = new Int32Array(nl);
  for (let i = 0; i < nh; i++) if (lock[i] >= 0) lockEnd[lock[i]] = i + 1;

  let dirtyLo = n, dirtyHi = 0;
  const world = new Float32Array(nh * 3);
  const queueLock = [];
  const rand = (a, b) => a + Math.random() * (b - a);

  function loose(i, cx, cy, cz, vx, vy, vz, tumble, t) {
    const o = i * STATE;
    dyn[o] = world[i * 3] - cx; dyn[o + 1] = world[i * 3 + 1] - cy; dyn[o + 2] = world[i * 3 + 2] - cz; dyn[o + 3] = t;
    dyn[o + 4] = vx + rand(-0.04, 0.04); dyn[o + 5] = vy + rand(-0.04, 0.04); dyn[o + 6] = vz; dyn[o + 7] = tumble;
    dyn[o + 8] = cx; dyn[o + 9] = cy; dyn[o + 10] = cz;
    dirtyLo = Math.min(dirtyLo, i); dirtyHi = Math.max(dirtyHi, i + 1);
    attached[i] = 0;
    for (let j = stubbleOf.start[i]; j < stubbleOf.start[i + 1]; j++) {
      const k = stubbleOf.out[j];
      if (dyn[k * STATE + 3] < 0) { dyn[k * STATE + 3] = t + 0.15; dirtyHi = Math.max(dirtyHi, k + 1); }
    }
    for (let j = trisOf.start[i]; j < trisOf.start[i + 1]; j++) {
      const k = trisOf.out[j];
      if (triAlive[k]) { triAlive[k] = 0; trisDirty = true; }
    }
    if (lock[i] >= 0 && --lockCount[cell[i]] === 0) queueLock.push(cell[i]);
  }

  /** Cuts along the trimmer's path (x0,y0)-(x1,y1), radius r, all in CSS px. Returns the dots cut. */
  function cut(x0, y0, x1, y1, r, v) {
    const R = v.rot, f = 1 / v.tan, t = v.now;
    const hit = new Int32Array(nl).fill(-1);
    const strayHit = [];
    const sx = x1 - x0, sy = y1 - y0, sl = sx * sx + sy * sy || 1;
    let count = 0;
    for (let i = 0; i < nh; i++) {
      const mx = pos[i * 3] - PIVOT[0], my = pos[i * 3 + 1] - PIVOT[1], mz = pos[i * 3 + 2] - PIVOT[2];
      const wx = R[0] * mx + R[3] * my + R[6] * mz + PIVOT[0];
      const wy = R[1] * mx + R[4] * my + R[7] * mz + PIVOT[1];
      const wz = R[2] * mx + R[5] * my + R[8] * mz + PIVOT[2];
      world[i * 3] = wx; world[i * 3 + 1] = wy; world[i * 3 + 2] = wz;
      if (!attached[i]) continue;
      // only what faces the viewer: the trimmer is on this side of the head
      const nx = nrm[i * 3], ny = nrm[i * 3 + 1], nz = nrm[i * 3 + 2];
      const ex = -wx, ey = -wy, ez = v.dist - wz, el = Math.hypot(ex, ey, ez);
      if ((R[0] * nx + R[3] * ny + R[6] * nz) * ex + (R[1] * nx + R[4] * ny + R[7] * nz) * ey
          + (R[2] * nx + R[5] * ny + R[8] * nz) * ez < -0.05 * el) continue;
      const zc = v.dist - wz;
      const px = (((f / v.aspect) * wx / zc) * v.scale + v.shiftX + 1) / 2 * v.w;
      const py = (1 - ((f * wy / zc) * v.scale + v.shiftY)) / 2 * v.h;
      const u = Math.max(0, Math.min(1, ((px - x0) * sx + (py - y0) * sy) / sl));
      const dx = px - (x0 + u * sx), dy = py - (y0 + u * sy);
      if (dx * dx + dy * dy > r * r) continue;
      count++;
      if (lock[i] >= 0) { if (hit[lock[i]] < 0 || i < hit[lock[i]]) hit[lock[i]] = i; } else strayHit.push(i);
    }
    if (!count) return 0;

    // a lock comes loose from where it was cut down to its tip, and falls as one strand
    for (let l = 0; l < nl; l++) {
      const k = hit[l];
      if (k < 0 || k >= lockEnd[l]) continue;
      let cx = 0, cy = 0, cz = 0;
      for (let i = k; i < lockEnd[l]; i++) { cx += world[i * 3]; cy += world[i * 3 + 1]; cz += world[i * 3 + 2]; }
      const m = lockEnd[l] - k;
      cx /= m; cy /= m; cz /= m;
      // it slides off the head: outwards from its middle, a little towards the viewer
      const hx = cx, hz = cz - PIVOT[2], hl = Math.hypot(hx, hz) || 1, out = rand(0.45, 0.85);
      const vx = hx / hl * out + rand(-0.1, 0.1), vy = rand(0.1, 0.45), vz = hz / hl * out + 0.25;
      const tumble = rand(-1.6, 1.6);
      for (let i = k; i < lockEnd[l]; i++) loose(i, cx, cy, cz, vx, vy, vz, tumble, t);
      lockEnd[l] = k;
    }
    for (const i of strayHit) {
      if (!attached[i]) continue;
      const hx = world[i * 3], hz = world[i * 3 + 2] - PIVOT[2], hl = Math.hypot(hx, hz) || 1, out = rand(0.3, 0.7);
      loose(i, world[i * 3], world[i * 3 + 1], world[i * 3 + 2], hx / hl * out, rand(0.1, 0.5), hz / hl * out + 0.25, 0, t);
    }

    // what follows: stray dots with no lock left near them fall too
    while (queueLock.length) {
      const c0 = queueLock.pop();
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
        const c = c0 + (dx * dim[1] + dy) * dim[2] + dz;
        for (let j = strayAt.start[c]; j < strayAt.start[c + 1]; j++) {
          const i = strayAt.out[j];
          if (attached[i] && near(lockCount, cell[i]) === 0) {
            loose(i, world[i * 3], world[i * 3 + 1], world[i * 3 + 2], rand(-0.2, 0.2), rand(0, 0.25), rand(0, 0.2), 0, t + rand(0, 0.15));
          }
        }
      }
    }
    return count;
  }

  /** Sends what changed to the GPU; once a frame, before drawing. */
  function upload() {
    if (dirtyHi > dirtyLo) {
      gl.bindBuffer(gl.ARRAY_BUFFER, dynBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, dirtyLo * STATE * 4, dyn, dirtyLo * STATE, (dirtyHi - dirtyLo) * STATE);
      dirtyLo = n; dirtyHi = 0;
    }
    if (trisDirty) {
      let m = base;
      for (let k = 0; k < nt; k++) {
        if (!triAlive[k]) continue;
        idxOut[m++] = idx[base + k * 3]; idxOut[m++] = idx[base + k * 3 + 1]; idxOut[m++] = idx[base + k * 3 + 2];
      }
      triCount = m;
      gl.bindVertexArray(triVao);
      gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, idxOut, 0, triCount);
      gl.bindVertexArray(null);
      trisDirty = false;
    }
  }

  return { vao, first, count: n, cut, upload, triCount: () => triCount };
}

export async function createFace(canvas, { reduced = false } = {}) {
  const gl = canvas.getContext('webgl2', { antialias: true, alpha: true, premultipliedAlpha: true, depth: true });
  if (!gl) throw new Error('webgl2 unavailable');

  // face.json is always checked with the server (it is tiny); it names the version of
  // face.bin, so a new model is never shadowed by a cached old one (tools/stamp.py).
  // If either comes back wrong (an error page, or a copy that does not match the
  // description: a cache caught half-way through a deploy), both are fetched once
  // more straight from the server.
  async function load(fresh) {
    const res = await fetch('/assets/face/face.json', { cache: fresh ? 'reload' : 'no-cache' });
    if (!res.ok) throw new Error(`face.json: HTTP ${res.status}`);
    const m = await res.json();
    const r2 = await fetch(`/assets/face/face.bin?v=${m.version || ''}`, fresh ? { cache: 'reload' } : {});
    if (!r2.ok) throw new Error(`face.bin: HTTP ${r2.status}`);
    const b = await r2.arrayBuffer();
    if (m.bytes && b.byteLength !== m.bytes) throw new Error(`face.bin: ${b.byteLength} bytes, expected ${m.bytes}`);
    return [m, b];
  }
  let meta, bin;
  try {
    [meta, bin] = await load(false);
  } catch (err) {
    console.warn('face: fetching again from the server:', err.message);
    [meta, bin] = await load(true);
  }

  const shared = ['uPosScale', 'uProj', 'uRot', 'uPivot', 'uDist', 'uShift', 'uScale'];
  const P = program(gl, POINT_VERT, POINT_FRAG, [...shared, 'uTanHalf', 'uAspect', 'uTime', 'uViewH', 'uGaze', 'uScan', 'uFlight', 'uKeys', 'uLeader', 'uBurst', 'uShift0', 'uScale0', 'uIdBase']);
  const Ln = program(gl, LINE_VERT, LINE_FRAG, [...shared, 'uRevealY', 'uScan', 'uLeader', 'uHair', 'uTime']);
  const D = program(gl, DEPTH_VERT, DEPTH_FRAG, shared);
  const all = [P, Ln, D];

  function vaoFor(section) {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(bin, section.offset, section.count * section.stride), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.SHORT, false, section.stride, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.BYTE, true, section.stride, 6);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 2, gl.UNSIGNED_BYTE, false, section.stride, 9);
    return vao;
  }
  const pointVao = vaoFor(meta.points);
  const lineVao = vaoFor(meta.lines);
  // hair: line strips over an index buffer; 0xFFFF restarts the strip (always on in WebGL2)
  let hairVao = null;
  if (meta.hair) {
    hairVao = vaoFor({ offset: meta.hair.offset, count: meta.hair.vertices, stride: meta.hair.stride });
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(bin, meta.hair.indexOffset, meta.hair.indices), gl.STATIC_DRAW);
  }
  const triVao = gl.createVertexArray();
  gl.bindVertexArray(triVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(bin, meta.tris.offset, meta.tris.vertices * meta.tris.stride), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.SHORT, false, meta.tris.stride, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(bin, meta.tris.indexOffset, meta.tris.indices), gl.DYNAMIC_DRAW);
  gl.bindVertexArray(null);
  // the hair and the buzzcut in a draw call of their own, with a state per dot (shaving);
  // in the first call the state is a constant: attached, never shown
  const shaver = meta.shave ? createShaver(gl, bin, meta, triVao) : null;
  gl.vertexAttrib4f(3, 0, 0, 0, -1);
  gl.vertexAttrib4f(4, 0, 0, 0, 0);
  gl.vertexAttrib3f(5, 0, 0, 0);

  for (const { p, u } of all) {
    gl.useProgram(p);
    gl.uniform1f(u.uPosScale, meta.posScale);
    gl.uniform3f(u.uPivot, 0, -0.15, -0.4);
  }
  gl.useProgram(P.p);
  gl.uniform1f(P.u.uTanHalf, Math.tan(FOV / 2));
  gl.uniform1f(P.u.uFlight, reduced ? 0 : 1);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0, 0, 0, 0);

  const frame = meta.frame; // { x, top, chin } of the head, model units
  const state = {
    t0: null, skipBy: 0, frozen: undefined,
    yaw: 0, pitch: 0, tYaw: 0, tPitch: 0,
    shiftX: 0, shiftY: 0, scale: 1, tShiftX: 0, tShiftY: 0, tScale: 1,
    lastMove: -1e9, dist: 9, w: 1, h: 1,
    burst: null,                                // { t0, shiftX, shiftY, scale }: where the head came from
  };

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (w === 0 || h === 0) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
    state.w = w; state.h = h;
    const aspect = w / h;
    const tan = Math.tan(FOV / 2);
    const height = (frame.top - frame.chin) * 1.32;
    const fitH = height / 2 / (0.74 * tan);
    const fitW = (frame.x * 1.5) / (0.86 * tan * aspect);   // room around the head
    state.dist = Math.max(fitH, fitW) + 0.4;
    const proj = perspective(FOV, aspect, 0.1, 60);
    for (const { p, u } of all) {
      gl.useProgram(p);
      gl.uniformMatrix4fv(u.uProj, false, proj);
      gl.uniform1f(u.uDist, state.dist);
    }
    gl.useProgram(P.p);
    gl.uniform1f(P.u.uAspect, aspect);
    gl.uniform1f(P.u.uViewH, canvas.height);
  }
  window.addEventListener('resize', resize);
  resize();

  const now = () => performance.now() / 1000;
  function elapsed() {
    if (state.frozen !== undefined) return state.frozen;
    return state.t0 === null ? 0 : now() - state.t0 + state.skipBy;
  }

  /** Where the head is on screen: during a burst, on its way from the old place to the new. */
  function placement() {
    if (!state.burst) return state;
    const b = state.burst;
    const k = ease((now() - b.t0 - 0.15) / 1.0);
    return { shiftX: b.shiftX + (state.shiftX - b.shiftX) * k, shiftY: b.shiftY + (state.shiftY - b.shiftY) * k,
             scale: b.scale + (state.scale - b.scale) * k };
  }

  /** Screen position (CSS px) of a model point with the head facing forward. */
  function project(x, y, z) {
    const zc = z - state.dist;
    const f = 1 / Math.tan(FOV / 2);
    const aspect = state.w / state.h;
    const at = placement();
    const nx = (f / aspect) * x / -zc * at.scale + at.shiftX;
    const ny = f * y / -zc * at.scale + at.shiftY;
    return [(nx + 1) / 2 * state.w, (1 - ny) / 2 * state.h];
  }

  let last = now();
  function draw() {
    const tn = now();
    const dt = Math.min(0.05, tn - last);
    last = tn;
    const te = elapsed();

    // nobody moving a pointer: the head looks around on its own
    if (tn - state.lastMove > 4 && te > ASSEMBLED_AT && !reduced) {
      const s = tn * 0.35;
      state.tYaw = 0.22 * Math.sin(s);
      state.tPitch = 0.07 * Math.sin(s * 0.73 + 1.1);
    }
    const k = 1 - Math.exp(-dt * 4.2);
    state.yaw += (state.tYaw - state.yaw) * k;
    state.pitch += (state.tPitch - state.pitch) * k;
    const kl = 1 - Math.exp(-dt * 5.5);
    state.shiftX += (state.tShiftX - state.shiftX) * kl;
    state.shiftY += (state.tShiftY - state.shiftY) * kl;
    state.scale += (state.tScale - state.scale) * kl;

    // the eyes lead: they look where the head is going
    const r = meta.iris.radius;
    const gx = Math.max(-1, Math.min(1, (state.tYaw - state.yaw) * 2.2 + state.tYaw * 0.6)) * r * 0.32;
    const gy = Math.max(-1, Math.min(1, -(state.tPitch - state.pitch) * 2.2 - state.tPitch * 0.6)) * r * 0.22;

    // the scan line: once as the face assembles, then once every 9 s
    const sweep = (ph) => frame.top + 0.06 - ph * (frame.top + 1.85);
    const s0 = ASSEMBLED_AT - 0.45;
    let scan = -10;
    let revealY = 10;
    if (reduced) revealY = -10;
    else if (te > s0) {
      const ph = (te - s0) % 9;
      if (ph < 1.5) scan = sweep(ph / 1.5);
      revealY = te - s0 < 1.5 ? sweep((te - s0) / 1.5) : -10;
    }
    const hair = reduced ? 1 : ease((te - (ASSEMBLED_AT - 0.3)) / 1.6);
    const keys = reduced ? 1 : ease((te - (ASSEMBLED_AT + 0.85)) / 0.45);
    const leader = reduced ? 1 : ease((te - (ASSEMBLED_AT + 1.05)) / 0.7);
    // a burst is over once the slowest dot has settled; the hidden-dot pass waits for that
    let burst = -1;
    if (state.burst) {
      burst = tn - state.burst.t0;
      if (burst > BURST_FOR) { state.burst = null; burst = -1; }
    }
    const depthOn = te > ASSEMBLED_AT - 0.35 && burst < 0;

    const rot = rotation(state.yaw, state.pitch);
    for (const { p, u } of all) {
      gl.useProgram(p);
      gl.uniformMatrix3fv(u.uRot, false, rot);
      gl.uniform2f(u.uShift, state.shiftX, state.shiftY);
      gl.uniform1f(u.uScale, state.scale);
    }

    if (shaver) shaver.upload();
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (depthOn) {
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LESS);
      gl.depthMask(true);
      gl.colorMask(false, false, false, false);
      gl.useProgram(D.p);
      gl.bindVertexArray(triVao);
      gl.drawElements(gl.TRIANGLES, shaver ? shaver.triCount() : meta.tris.indices, gl.UNSIGNED_SHORT, 0);
      gl.colorMask(true, true, true, true);
      gl.depthMask(false);
      gl.depthFunc(gl.LEQUAL);
    } else {
      gl.disable(gl.DEPTH_TEST);
    }

    gl.useProgram(Ln.p);
    gl.uniform1f(Ln.u.uRevealY, revealY);
    gl.uniform1f(Ln.u.uScan, scan);
    gl.uniform1f(Ln.u.uLeader, leader);
    gl.uniform1f(Ln.u.uHair, hair);
    gl.uniform1f(Ln.u.uTime, te);
    gl.bindVertexArray(lineVao);
    gl.drawArrays(gl.LINES, 0, meta.lines.count);
    if (hairVao) {
      gl.bindVertexArray(hairVao);
      gl.drawElements(gl.LINE_STRIP, meta.hair.indices, gl.UNSIGNED_SHORT, 0);
    }

    gl.useProgram(P.p);
    gl.uniform1f(P.u.uTime, te);
    gl.uniform2f(P.u.uGaze, gx, gy);
    gl.uniform1f(P.u.uScan, scan);
    gl.uniform1f(P.u.uKeys, keys);
    gl.uniform1f(P.u.uLeader, leader);
    gl.uniform1f(P.u.uBurst, burst);
    const from = state.burst || state;
    gl.uniform2f(P.u.uShift0, from.shiftX, from.shiftY);
    gl.uniform1f(P.u.uScale0, from.scale);
    gl.uniform1i(P.u.uIdBase, 0);
    gl.bindVertexArray(pointVao);
    gl.drawArrays(gl.POINTS, 0, shaver ? shaver.first : meta.points.count);
    if (shaver) {
      gl.uniform1i(P.u.uIdBase, shaver.first);
      gl.bindVertexArray(shaver.vao);
      gl.drawArrays(gl.POINTS, 0, shaver.count);
    }
  }

  let raf = 0;
  const listeners = [];
  function loop() {
    draw();
    for (const fn of listeners) fn();
    raf = requestAnimationFrame(loop);
  }

  return {
    start() { state.t0 = now(); if (!raf) loop(); },
    skip() { const te = elapsed(); if (te < FINISHED_AT) state.skipBy += FINISHED_AT - te; },
    /** Holds the opening at one moment (?freeze=1.5 — for looking at it frame by frame). */
    freeze(t) { state.frozen = t; },
    elapsed,
    pause() { cancelAnimationFrame(raf); raf = 0; },
    resume() { if (!raf && state.t0 !== null) { last = now(); loop(); } },
    onFrame(fn) { listeners.push(fn); },
    /** Pointer in -1..1 (x right, y up) relative to the face centre. */
    look(nx, ny) {
      state.lastMove = now();
      state.tYaw = Math.max(-1, Math.min(1, nx)) * 0.55;
      state.tPitch = -Math.max(-1, Math.min(1, ny)) * 0.3;
    },
    /** Moves the head on screen: gliding there, or (burst) blown apart into dots that gather at the new place. */
    layout(shiftX, shiftY, scale, { burst = false } = {}) {
      state.tShiftX = shiftX; state.tShiftY = shiftY; state.tScale = scale;
      if (!burst || reduced || elapsed() < FINISHED_AT || state.frozen !== undefined) return;
      const at = placement();
      state.burst = { t0: now(), shiftX: at.shiftX, shiftY: at.shiftY, scale: at.scale };
      state.shiftX = shiftX; state.shiftY = shiftY; state.scale = scale;
    },
    /** The head's frame on screen, CSS px. */
    rect() {
      const [x0, y0] = project(-frame.x, frame.top, 0.2);
      const [x1, y1] = project(frame.x, frame.chin, 0.2);
      return { left: x0, top: y0, width: x1 - x0, height: y1 - y0 };
    },
    center() { return project(0, (frame.top + frame.chin) / 2, 0.2); },
    /** Shaving (app.js): the trimmer moved from (x0, y0) to (x1, y1), CSS px, cutting
        r px around it. Returns how many hair dots it cut; 0 while the head is not
        settled (the opening, a burst). */
    shave(x0, y0, x1, y1, r) {
      if (!shaver || elapsed() < FINISHED_AT || state.burst) return 0;
      const at = placement();
      return shaver.cut(x0, y0, x1, y1, r, {
        rot: rotation(state.yaw, state.pitch), dist: state.dist, tan: Math.tan(FOV / 2), aspect: state.w / state.h,
        w: state.w, h: state.h, shiftX: at.shiftX, shiftY: at.shiftY, scale: at.scale, now: elapsed(),
      });
    },
  };
}
