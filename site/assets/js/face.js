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
vec4 project(vec3 q) {
  vec4 clip = uProj * vec4(q.x, q.y, q.z - uDist, 1.0);
  clip.xy = clip.xy * uScale + uShift * clip.w;
  return clip;
}
float facingCam(vec3 p, vec3 n) { return dot(n, normalize(vec3(0.0, 0.0, uDist) - p)); }
`;

const POINT_VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec2 aKT;
${COMMON}
uniform float uTanHalf;
uniform float uAspect;
uniform float uTime;
uniform float uViewH;
uniform vec2 uGaze;
uniform float uScan;
uniform float uFlight;
uniform float uKeys;
uniform float uLeader;
out vec4 vColor;
flat out int vShape;

uint hash(uint x) {
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  x ^= x >> 16; return x;
}
float rnd(uint i, uint k) { return float(hash(i * 8u + k) & 0xffffffu) / 16777216.0; }

void main() {
  uint id = uint(gl_VertexID);
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
  bool flies = kind <= 5;
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

  vec4 clip = project(q);
  gl_Position = clip;

  // ---- look
  vec3 L = normalize(vec3(-0.45, 0.55, 0.75));
  float dif = max(dot(n, L), 0.0);
  float ink = clamp(tone * 0.95 + (1.0 - dif) * 0.22 - 0.05, 0.0, 1.0);
  float vis = mix(1.0, smoothstep(-0.12, 0.25, facingCam(p, n)), landed);
  vis *= kind == 1 ? smoothstep(-2.25, -1.75, target.y) : smoothstep(-1.8, -1.2, target.y);
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
  } else {                               // the rings at the ends of the leader lines
    size = 0.03 * smoothstep(0.6, 1.0, uLeader); alpha = smoothstep(0.6, 1.0, uLeader); col = accent; vShape = 2;
  }
  col = mix(accent, col, landed);
  col = mix(col, accent, band * 0.8);
  alpha = alpha * vis;
  alpha = mix(0.3 + 0.2 * r4, alpha, landed) * smoothstep(0.0, 0.06, t);
  alpha = min(1.0, alpha + band * 0.45 * vis);

  float px = size * uViewH / (2.0 * uTanHalf * clip.w) * uScale * (1.0 + band * 0.5);
  px = mix(max(px, 1.5), clamp(px * 1.6, 1.5, 22.0), 1.0 - landed);
  gl_PointSize = px;
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

export async function createFace(canvas, { reduced = false } = {}) {
  const gl = canvas.getContext('webgl2', { antialias: true, alpha: true, premultipliedAlpha: true, depth: true });
  if (!gl) throw new Error('webgl2 unavailable');

  // face.json is always checked with the server (it is tiny); it names the version of
  // face.bin, so a new model is never shadowed by a cached old one (tools/stamp.py)
  const meta = await fetch('/assets/face/face.json', { cache: 'no-cache' }).then((r) => r.json());
  const bin = await fetch(`/assets/face/face.bin?v=${meta.version || ''}`).then((r) => r.arrayBuffer());

  const shared = ['uPosScale', 'uProj', 'uRot', 'uPivot', 'uDist', 'uShift', 'uScale'];
  const P = program(gl, POINT_VERT, POINT_FRAG, [...shared, 'uTanHalf', 'uAspect', 'uTime', 'uViewH', 'uGaze', 'uScan', 'uFlight', 'uKeys', 'uLeader']);
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
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(bin, meta.tris.indexOffset, meta.tris.indices), gl.STATIC_DRAW);
  gl.bindVertexArray(null);

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

  /** Screen position (CSS px) of a model point with the head facing forward. */
  function project(x, y, z) {
    const zc = z - state.dist;
    const f = 1 / Math.tan(FOV / 2);
    const aspect = state.w / state.h;
    const nx = (f / aspect) * x / -zc * state.scale + state.shiftX;
    const ny = f * y / -zc * state.scale + state.shiftY;
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
    const depthOn = te > ASSEMBLED_AT - 0.35;

    const rot = rotation(state.yaw, state.pitch);
    for (const { p, u } of all) {
      gl.useProgram(p);
      gl.uniformMatrix3fv(u.uRot, false, rot);
      gl.uniform2f(u.uShift, state.shiftX, state.shiftY);
      gl.uniform1f(u.uScale, state.scale);
    }

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (depthOn) {
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LESS);
      gl.depthMask(true);
      gl.colorMask(false, false, false, false);
      gl.useProgram(D.p);
      gl.bindVertexArray(triVao);
      gl.drawElements(gl.TRIANGLES, meta.tris.indices, gl.UNSIGNED_SHORT, 0);
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
    gl.bindVertexArray(pointVao);
    gl.drawArrays(gl.POINTS, 0, meta.points.count);
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
    layout(shiftX, shiftY, scale) { state.tShiftX = shiftX; state.tShiftY = shiftY; state.tScale = scale; },
    /** The head's frame on screen, CSS px. */
    rect() {
      const [x0, y0] = project(-frame.x, frame.top, 0.2);
      const [x1, y1] = project(frame.x, frame.chin, 0.2);
      return { left: x0, top: y0, width: x1 - x0, height: y1 - y0 };
    },
    center() { return project(0, (frame.top + frame.chin) / 2, 0.2); },
  };
}
