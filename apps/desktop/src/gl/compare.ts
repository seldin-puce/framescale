/**
 * WebGL2 before/after renderer.
 *
 * One full-screen triangle and two textures. Split position, zoom and pan are
 * uniforms, so moving the slider costs one draw call (no DOM layout), and the
 * two sides are pixel-aligned at every zoom level by construction. Above 2×
 * zoom, sampling switches to NEAREST for honest pixel-peeping.
 */

export type Mode = 0 | 1 | 2 | 3; // split | side-by-side | before | after

export interface View {
  mode: Mode;
  /** Split position across the canvas, 0–1. */
  split: number;
  /** 1 = fit to view. */
  zoom: number;
  /** Image-pixel offset of the view centre from the image centre. */
  panX: number;
  panY: number;
}

const VS = `#version 300 es
out vec2 vPos;
void main() {
  // Full-screen triangle: p ∈ {0,2}², and the visible screen is p ∈ [0,1]².
  // vPos is 0..1 across the screen with y pointing down.
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vPos = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
uniform sampler2D uBefore;
uniform sampler2D uAfter;
uniform vec2 uView;
uniform vec2 uImage;
uniform float uZoom;
uniform vec2 uPan;
uniform float uSplit;
uniform int uMode;
uniform vec3 uBg;
in vec2 vPos;
out vec4 outColor;

void main() {
  vec2 p = vPos * uView;
  vec2 view = uView;
  bool after;
  if (uMode == 1) {
    view.x = uView.x * 0.5;
    after = p.x >= view.x;
    if (after) p.x -= view.x;
  } else if (uMode == 2) {
    after = false;
  } else if (uMode == 3) {
    after = true;
  } else {
    after = p.x >= uSplit * uView.x;
  }
  float s = min(view.x / uImage.x, view.y / uImage.y) * uZoom;
  vec2 img = (p - view * 0.5) / s + uImage * 0.5 + uPan;
  vec2 uv = img / uImage;
  if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
    outColor = vec4(uBg, 1.0);
    return;
  }
  outColor = after ? texture(uAfter, uv) : texture(uBefore, uv);
}`;

export class CompareRenderer {
  private gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private tex: [WebGLTexture, WebGLTexture];
  private u: Record<string, WebGLUniformLocation | null> = {};
  private imageSize: [number, number] = [1, 1];
  private nearest = false;
  private bg: [number, number, number] = [0.043, 0.059, 0.098]; // #0B0F19

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false, alpha: false });
    if (!gl) throw new Error("WebGL2 is not available");
    this.gl = gl;
    this.prog = this.link(VS, FS);
    for (const name of ["uBefore", "uAfter", "uView", "uImage", "uZoom", "uPan", "uSplit", "uMode", "uBg"]) {
      this.u[name] = gl.getUniformLocation(this.prog, name);
    }
    const mk = () => {
      const t = gl.createTexture();
      if (!t) throw new Error("texture allocation failed");
      return t;
    };
    this.tex = [mk(), mk()];
  }

  private link(vs: string, fs: string): WebGLProgram {
    const gl = this.gl;
    const compile = (type: number, src: string) => {
      const sh = gl.createShader(type);
      if (!sh) throw new Error("shader allocation failed");
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) ?? "shader error");
      return sh;
    };
    const p = gl.createProgram();
    if (!p) throw new Error("program allocation failed");
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link error");
    return p;
  }

  /** Upload the pair. `before` may be lower resolution; it's displayed at `after`'s size. */
  setImages(before: TexImageSource, after: TexImageSource, width: number, height: number): void {
    const gl = this.gl;
    this.imageSize = [width, height];
    for (const [i, src] of [before, after].entries()) {
      gl.bindTexture(gl.TEXTURE_2D, this.tex[i as 0 | 1]);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, src);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
    this.nearest = !this.nearest; // force filter refresh on next draw
    this.applyFilter(false);
  }

  setBackground(rgb: [number, number, number]): void {
    this.bg = rgb;
  }

  private applyFilter(nearest: boolean): void {
    if (nearest === this.nearest) return;
    const gl = this.gl;
    const f = nearest ? gl.NEAREST : gl.LINEAR;
    for (const t of this.tex) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
    }
    this.nearest = nearest;
  }

  /** Pixel scale (canvas px per image px) for the given view; used for zoom maths. */
  scaleFor(view: View): number {
    const [w, h] = [this.canvas.width / (view.mode === 1 ? 2 : 1), this.canvas.height];
    return Math.min(w / this.imageSize[0], h / this.imageSize[1]) * view.zoom;
  }

  get image(): [number, number] {
    return this.imageSize;
  }

  draw(view: View): void {
    const gl = this.gl;
    const { width, height } = this.canvas;
    gl.viewport(0, 0, width, height);
    // Nearest sampling once one image pixel covers ≥ 2 device pixels.
    this.applyFilter(this.scaleFor(view) >= 2);
    gl.useProgram(this.prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex[0]);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.tex[1]);
    gl.uniform1i(this.u.uBefore!, 0);
    gl.uniform1i(this.u.uAfter!, 1);
    gl.uniform2f(this.u.uView!, width, height);
    gl.uniform2f(this.u.uImage!, this.imageSize[0], this.imageSize[1]);
    gl.uniform1f(this.u.uZoom!, view.zoom);
    gl.uniform2f(this.u.uPan!, view.panX, view.panY);
    gl.uniform1f(this.u.uSplit!, view.split);
    gl.uniform1i(this.u.uMode!, view.mode);
    gl.uniform3f(this.u.uBg!, ...this.bg);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  dispose(): void {
    const gl = this.gl;
    this.tex.forEach((t) => gl.deleteTexture(t));
    gl.deleteProgram(this.prog);
  }
}
