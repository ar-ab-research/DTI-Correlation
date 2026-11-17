// one webgl canvas behind all three panes. each pane is drawn into its own rectangle
// of the canvas with the same quad and shader, looking along a different axis.
import * as THREE from "three";
import { VIEWS, fitRect, sliceCoord, windowRange } from "./geometry.mjs";
import { emptyTexture } from "./volume.js";

const VERTEX = /* glsl */ `
out vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

const FRAGMENT = /* glsl */ `
precision highp float;
precision highp sampler3D;

uniform sampler3D uMap;     // the FA or the MD map, whichever is on
uniform sampler3D uTracts;  // 0 outside every tract, 1 or 2 inside one
uniform int uAxis;          // the volume axis this view looks along
uniform bool uFlipH;
uniform float uSlice;       // texture coordinate along that axis
uniform vec3 uTexel;        // one voxel, in texture coordinates
uniform float uLow;
uniform float uSpan;
uniform float uOpacity;
uniform bool uShow[2];
uniform bool uOutline;
uniform vec3 uTractColors[2];

in vec2 vUv;
out vec4 fragColor;

vec3 position(vec2 uv) {
  float h = uFlipH ? 1.0 - uv.x : uv.x;
  if (uAxis == 2) return vec3(h, uv.y, uSlice);
  if (uAxis == 1) return vec3(h, uSlice, uv.y);
  return vec3(uSlice, uv.x, uv.y);
}

// the two in-plane steps of one voxel
void steps(out vec3 a, out vec3 b) {
  if (uAxis == 2) { a = vec3(uTexel.x, 0.0, 0.0); b = vec3(0.0, uTexel.y, 0.0); }
  else if (uAxis == 1) { a = vec3(uTexel.x, 0.0, 0.0); b = vec3(0.0, 0.0, uTexel.z); }
  else { a = vec3(0.0, uTexel.y, 0.0); b = vec3(0.0, 0.0, uTexel.z); }
}

int tractAt(vec3 p) { return int(texture(uTracts, p).r * 255.0 + 0.5); }

// inside a tract, with at least one in-plane neighbour outside it
bool onEdge(int code, vec3 p) {
  vec3 a; vec3 b;
  steps(a, b);
  return tractAt(p + a) != code || tractAt(p - a) != code
      || tractAt(p + b) != code || tractAt(p - b) != code;
}

void main() {
  vec3 p = position(vUv);
  float grey = clamp((texture(uMap, p).r - uLow) / uSpan, 0.0, 1.0);
  vec3 colour = vec3(grey);

  int code = tractAt(p);
  if (code >= 1 && code <= 2 && uShow[code - 1]) {
    vec3 tint = uTractColors[code - 1];
    if (!uOutline) colour = mix(colour, tint, uOpacity);
    else if (onEdge(code, p)) colour = mix(colour, tint, max(uOpacity, 0.85));
  }
  fragColor = vec4(colour, 1.0);
}`;

export class SliceRenderer {
  constructor(canvas, colours) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false });
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.autoClear = false;
    this.empty = emptyTexture();
    this.maps = {};
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.scene = new THREE.Scene();
    this.material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      depthTest: false,
      uniforms: {
        uMap: { value: this.empty },
        uTracts: { value: this.empty },
        uAxis: { value: 2 },
        uFlipH: { value: true },
        uSlice: { value: 0.5 },
        uTexel: { value: new THREE.Vector3(1, 1, 1) },
        uLow: { value: 0 },
        uSpan: { value: 1 },
        uOpacity: { value: 0.5 },
        uShow: { value: [true, true] },
        uOutline: { value: true },
        uTractColors: { value: colours.tracts.map((c) => new THREE.Color(c)) },
      },
    });
    this.scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material));
    this.dims = [1, 1, 1];
  }

  setSubject(subject) {
    const u = this.material.uniforms;
    const t = subject.textures;
    this.dims = subject.meta.dims;
    this.maps = { fa: t.fa ?? this.empty, md: t.md ?? this.empty };
    u.uTracts.value = t.tracts ?? this.empty;
    u.uTexel.value.set(1 / this.dims[0], 1 / this.dims[1], 1 / this.dims[2]);
  }

  // where each pane's slice is drawn, in css pixels relative to the pane. the dom
  // overlays (crosshair, pointer maths) use the same rectangles
  sliceRect(viewName, pane) {
    const view = VIEWS[viewName];
    return fitRect(pane.clientWidth, pane.clientHeight, this.dims[view.h], this.dims[view.v]);
  }

  draw(panes, state) {
    const { renderer, canvas } = this;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(canvas.clientWidth * ratio);
    const height = Math.round(canvas.clientHeight * ratio);
    if (canvas.width !== width || canvas.height !== height) renderer.setSize(width, height, false);
    renderer.setScissorTest(false);
    renderer.clear();
    renderer.setScissorTest(true);

    const u = this.material.uniforms;
    const range = windowRange(state.level, state.width);
    u.uMap.value = this.maps[state.map] ?? this.empty;
    u.uLow.value = range.low;
    u.uSpan.value = range.span;
    u.uOpacity.value = state.opacity;
    u.uShow.value = state.show;
    u.uOutline.value = state.outline;

    const frame = canvas.getBoundingClientRect();
    for (const [viewName, pane] of Object.entries(panes)) {
      const view = VIEWS[viewName];
      const box = pane.getBoundingClientRect();
      const rect = this.sliceRect(viewName, pane);
      if (rect.w < 1 || rect.h < 1) continue;
      const x = (box.left - frame.left + rect.x) * ratio;
      // webgl counts rows from the bottom of the canvas
      const y = (frame.bottom - box.top - rect.y - rect.h) * ratio;
      renderer.setViewport(x, y, rect.w * ratio, rect.h * ratio);
      renderer.setScissor(x, y, rect.w * ratio, rect.h * ratio);
      u.uAxis.value = view.axis;
      u.uFlipH.value = view.flipH;
      u.uSlice.value = sliceCoord(state.voxel[view.axis], this.dims[view.axis]);
      renderer.render(this.scene, this.camera);
    }
  }
}
