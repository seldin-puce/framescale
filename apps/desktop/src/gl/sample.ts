/**
 * Procedural sample pair shown until a real 2 s preview has been rendered.
 * "After" is drawn crisply at full resolution; "Before" is the same scene
 * drawn at ¼ resolution with sensor noise and compression-style softening.
 * It's a stand-in for UI review, never presented as model output.
 */

const W = 1920;
const H = 1080;

function scene(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  const s = w / W;
  const sky = ctx.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, "#1b2a4a");
  sky.addColorStop(0.55, "#d9825b");
  sky.addColorStop(1, "#2b1d2e");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h);

  ctx.fillStyle = "#ffd9a0";
  ctx.beginPath();
  ctx.arc(w * 0.72, h * 0.5, 90 * s, 0, Math.PI * 2);
  ctx.fill();

  const ridge = (y: number, amp: number, color: string, seed: number) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let x = 0; x <= w; x += 8 * s) {
      const t = x / w;
      ctx.lineTo(x, h * y - amp * s * (Math.sin(t * 9 + seed) * 0.6 + Math.sin(t * 23 + seed * 2) * 0.25 + Math.sin(t * 61) * 0.08));
    }
    ctx.lineTo(w, h);
    ctx.fill();
  };
  ridge(0.62, 120, "#4a3150", 1);
  ridge(0.72, 90, "#2c2038", 2.3);
  ridge(0.84, 60, "#161321", 4.1);

  // Fine detail that upscalers live or die by: hairlines, text, a fence.
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth = Math.max(1, 1.2 * s);
  for (let i = 0; i < 40; i++) {
    const x = w * 0.08 + i * 14 * s;
    ctx.beginPath();
    ctx.moveTo(x, h * 0.86);
    ctx.lineTo(x, h * 0.95);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(w * 0.08, h * 0.88);
  ctx.lineTo(w * 0.08 + 40 * 14 * s, h * 0.88);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,0.92)";
  ctx.font = `600 ${64 * s}px Inter, system-ui, sans-serif`;
  ctx.fillText("Harbour Lights", w * 0.08, h * 0.2);
  ctx.font = `400 ${22 * s}px Inter, system-ui, sans-serif`;
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  ctx.fillText("Archive reel 07 · restored", w * 0.08, h * 0.2 + 40 * s);

  // Siemens-star resolution target.
  const cx = w * 0.86;
  const cy = h * 0.22;
  const r = 110 * s;
  for (let i = 0; i < 48; i++) {
    ctx.fillStyle = i % 2 ? "#f4f4f4" : "#111";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, (i / 48) * Math.PI * 2, ((i + 1) / 48) * Math.PI * 2);
    ctx.fill();
  }
}

function canvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("2D canvas unavailable");
  return [c, ctx];
}

export interface SamplePair {
  before: HTMLCanvasElement;
  after: HTMLCanvasElement;
  width: number;
  height: number;
}

export function makeSamplePair(): SamplePair {
  const [after, actx] = canvas(W, H);
  scene(actx, W, H);

  const [small, sctx] = canvas(W / 4, H / 4);
  scene(sctx, W / 4, H / 4);
  const img = sctx.getImageData(0, 0, W / 4, H / 4);
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) - 0.5;
  for (let i = 0; i < img.data.length; i += 4) {
    const n = rand() * 22;
    img.data[i] = Math.max(0, Math.min(255, img.data[i]! + n));
    img.data[i + 1] = Math.max(0, Math.min(255, img.data[i + 1]! + n));
    img.data[i + 2] = Math.max(0, Math.min(255, img.data[i + 2]! + n));
  }
  sctx.putImageData(img, 0, 0);

  const [before, bctx] = canvas(W, H);
  bctx.imageSmoothingEnabled = true;
  bctx.imageSmoothingQuality = "low";
  bctx.filter = "blur(1.5px)";
  bctx.drawImage(small, 0, 0, W, H);
  return { before, after, width: W, height: H };
}
