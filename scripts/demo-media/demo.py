# Usage: see the header of each script. Inputs come from crates/fs-engine/examples/enhance_image.
"""FrameScale AI demo video (1920×1080, 24 fps) for Product Hunt / YouTube."""
import math, subprocess, sys
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ICON, SRC, AFTER, FFMPEG, OUT = sys.argv[1:6]
W, H, FPS = 1920, 1080, 24
NOTO = "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf"
BG, FG, MUTED, SUBTLE, ACCENT = (11, 15, 25), (230, 233, 240), (154, 163, 181), (107, 116, 137), (76, 141, 255)
N = 144
SW, SH, AW, AH = 640, 268, 1280, 536

def font(size, wght):
    f = ImageFont.truetype(NOTO, size)
    f.set_variation_by_axes([wght])
    return f

F = {k: font(*v) for k, v in {
    "brand": (64, 720), "h1": (92, 780), "h2": (60, 760), "sub": (34, 450), "lbl": (26, 760),
    "card_t": (40, 720), "card_m": (24, 620), "card_d": (26, 450), "small": (22, 500), "url": (40, 680)}.items()}

src = open(SRC, "rb").read()
aft = open(AFTER, "rb").read()
def frame_pair(i):
    i %= N
    b = Image.frombytes("RGB", (SW, SH), src[i * SW * SH * 3:(i + 1) * SW * SH * 3]).resize((AW, AH), Image.BICUBIC)
    a = Image.frombytes("RGB", (AW, AH), aft[i * AW * AH * 3:(i + 1) * AW * AH * 3])
    return b, a

icon = Image.open(ICON).convert("RGBA")
def bg():
    img = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), BG)
    ImageDraw.Draw(glow).ellipse((1000, -400, 2300, 800), fill=(29, 63, 143))
    return Image.blend(img, glow.filter(ImageFilter.GaussianBlur(220)), 0.75)
BG_IMG = bg()

def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)

def text(img, xy, s, f, color, alpha=1.0, anchor="la"):
    if alpha <= 0:
        return
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(xy, s, font=f, fill=color + (int(255 * alpha),), anchor=anchor)
    img.alpha_composite(layer)

def rounded_paste(img, im, xy, r=22, outline=(42, 58, 92)):
    m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, im.size[0] - 1, im.size[1] - 1), radius=r, fill=255)
    sh = Image.new("RGBA", (im.size[0] + 80, im.size[1] + 80), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((40, 54, im.size[0] + 40, im.size[1] + 54), radius=r, fill=(0, 0, 0, 170))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(24)), (xy[0] - 40, xy[1] - 40))
    img.paste(im, xy, m)
    ImageDraw.Draw(img).rounded_rectangle((xy[0], xy[1], xy[0] + im.size[0] - 1, xy[1] + im.size[1] - 1), radius=r, outline=outline, width=2)

def badge(d, x, y, s, f, color):
    w = d.textlength(s, font=f)
    d.rounded_rectangle((x, y, x + w + 28, y + 46), radius=10, fill=(0, 0, 0))
    d.text((x + 14, y + 7), s, font=f, fill=color)

def title_card(t):
    img = BG_IMG.convert("RGBA")
    a = ease(t / 0.8)
    ic = icon.resize((150, 150), Image.LANCZOS)
    ic.putalpha(ic.getchannel("A").point(lambda v: int(v * a)))
    img.alpha_composite(ic, ((W - 150) // 2, int(250 - 20 * (1 - a))))
    text(img, (W // 2, 450), "FrameScale AI", F["brand"], FG, a, "mm")
    b = ease((t - 0.5) / 0.8)
    text(img, (W // 2, 560), "Free AI video upscaler & enhancer", F["h1"], (255, 255, 255), b, "mm")
    c = ease((t - 1.1) / 0.8)
    text(img, (W // 2, 660), "Open source · Runs on your own GPU · No uploads", F["sub"], MUTED, c, "mm")
    return img

VX, VY, VW, VH = (W - 1640) // 2, 250, 1640, int(1640 * AH / AW)
def wipe(t, i):
    img = BG_IMG.convert("RGBA")
    a = ease(t / 0.6)
    text(img, (W // 2, 120), "Real footage. Real result.", F["h2"], (255, 255, 255), a, "mm")
    text(img, (W // 2, 190), "640-pixel source enlarged 2×: normal scaling vs FrameScale AI", F["sub"], MUTED, a, "mm")
    b, af = frame_pair(i)
    b, af = b.resize((VW, VH), Image.LANCZOS), af.resize((VW, VH), Image.LANCZOS)
    # Split sweeps in from the right, then eases back and forth.
    pos = 1.0 - 0.5 * ease((t - 0.6) / 1.6) if t < 2.2 else 0.5 + 0.22 * math.sin((t - 2.2) * 1.1)
    x = int(VW * pos)
    comp = af.copy()
    comp.paste(b.crop((0, 0, x, VH)), (0, 0))
    d = ImageDraw.Draw(comp)
    d.line((x, 0, x, VH), fill=(255, 255, 255), width=4)
    d.ellipse((x - 26, VH // 2 - 26, x + 26, VH // 2 + 26), fill=(255, 255, 255))
    for dx in (-1, 1):
        d.polygon([(x + dx * 7, VH // 2 - 9), (x + dx * 7, VH // 2 + 9), (x + dx * 17, VH // 2)], fill=(11, 15, 25))
    badge(d, 20, 20, "BEFORE · normal scaling", F["lbl"], (255, 255, 255))
    lw = d.textlength("AFTER · FrameScale AI", font=F["lbl"])
    badge(d, VW - lw - 48, 20, "AFTER · FrameScale AI", F["lbl"], (159, 196, 255))
    rounded_paste(img, comp, (VX, VY))
    text(img, (W // 2, VY + VH + 50), "Footage: Tears of Steel (CC BY 3.0, Blender Foundation)", F["small"], SUBTLE, a, "mm")
    return img

PW, PH = 760, 600
BOX = (300, 20, 680, 320)  # the two actors' upper bodies in the 1280×536 frames
def detail(t, i):
    img = BG_IMG.convert("RGBA")
    a = ease(t / 0.6)
    text(img, (W // 2, 120), "Zoom in: sharper edges, cleaner detail", F["h2"], (255, 255, 255), a, "mm")
    text(img, (W // 2, 190), "Real-ESRGAN General v3 inside FrameScale AI, on the GPU", F["sub"], MUTED, a, "mm")
    b, af = frame_pair(i)
    gx = (W - 2 * PW - 40) // 2
    for k, (im, lab, col) in enumerate([(b, "BEFORE", (255, 255, 255)), (af, "AFTER", (159, 196, 255))]):
        p = im.crop(BOX).resize((PW, PH), Image.LANCZOS)
        badge(ImageDraw.Draw(p), 20, 20, lab, F["lbl"], col)
        rounded_paste(img, p, (gx + k * (PW + 40), 270))
    return img

STAGES = [("Deinterlace", "BWDIF", "Old TV, VHS and DV\nto smooth 50/60p"),
          ("Denoise", "NAFNet · Real-CUGAN", "Camera grain and\ncompression noise"),
          ("Upscale", "Real-ESRGAN", "2× or 4×, from SD\nall the way to 4K"),
          ("Face Fix", "CodeFormer", "Finds and restores\nfaces automatically"),
          ("Interpolate", "RIFE 4.25", "2×–4× frame rate,\nscene-cut aware")]
def stages(t):
    img = BG_IMG.convert("RGBA")
    a = ease(t / 0.6)
    text(img, (W // 2, 170), "One pipeline. Five AI stages.", F["h1"], (255, 255, 255), a, "mm")
    text(img, (W // 2, 260), "Turn on what your footage needs, preview 2 seconds, then render.", F["sub"], MUTED, a, "mm")
    cw, gap = 330, 26
    x0 = (W - 5 * cw - 4 * gap) // 2
    for k, (name, model, desc) in enumerate(STAGES):
        c = ease((t - 0.5 - 0.35 * k) / 0.6)
        if c <= 0:
            continue
        card = Image.new("RGBA", (cw, 380), (0, 0, 0, 0))
        d = ImageDraw.Draw(card)
        d.rounded_rectangle((0, 0, cw - 1, 379), radius=22, fill=(17, 24, 39, 255), outline=(36, 50, 79, 255), width=2)
        d.rounded_rectangle((28, 28, 92, 92), radius=14, fill=(22, 38, 74), outline=(46, 76, 134), width=2)
        d.text((60, 60), f"0{k + 1}", font=F["card_m"], fill=ACCENT, anchor="mm")
        d.text((28, 122), name, font=F["card_t"], fill=FG)
        d.text((28, 180), model, font=F["card_m"], fill=ACCENT)
        for j, line in enumerate(desc.split("\n")):
            d.text((28, 248 + j * 38), line, font=F["card_d"], fill=MUTED)
        card.putalpha(card.getchannel("A").point(lambda v: int(v * c)))
        img.alpha_composite(card, (x0 + k * (cw + gap), int(380 + 40 * (1 - c))))
    f = ease((t - 2.6) / 0.8)
    text(img, (W // 2, 860), "HDR10 · audio & subtitles kept · HEVC 10-bit, AV1, ProRes · NVIDIA · AMD · Intel · Apple", F["small"], SUBTLE, f, "mm")
    return img

def end_card(t):
    img = BG_IMG.convert("RGBA")
    a = ease(t / 0.8)
    ic = icon.resize((120, 120), Image.LANCZOS)
    ic.putalpha(ic.getchannel("A").point(lambda v: int(v * a)))
    img.alpha_composite(ic, ((W - 120) // 2, 260))
    text(img, (W // 2, 460), "Free. Open source. Yours.", F["h1"], (255, 255, 255), a, "mm")
    text(img, (W // 2, 560), "Windows · macOS · Linux", F["sub"], MUTED, ease((t - 0.4) / 0.8), "mm")
    b = ease((t - 0.8) / 0.8)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w = d.textlength("framescale.org", font=F["url"]) + 80
    d.rounded_rectangle(((W - w) / 2, 640, (W + w) / 2, 724), radius=16, fill=ACCENT + (int(255 * b),))
    d.text((W / 2, 682), "framescale.org", font=F["url"], fill=(255, 255, 255, int(255 * b)), anchor="mm")
    img.alpha_composite(layer)
    return img

SCENES = [(3.5, lambda t, f: title_card(t)), (6.0, lambda t, f: wipe(t, f)), (6.0, lambda t, f: detail(t, f)),
          (6.5, lambda t, f: stages(t)), (4.5, lambda t, f: end_card(t))]
enc = subprocess.Popen([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
prev = None
total = 0
for si, (dur, draw) in enumerate(SCENES):
    n = int(dur * FPS)
    for k in range(n):
        img = draw(k / FPS, k).convert("RGB")
        # Crossfade the first 10 frames of each scene from the previous one.
        if prev is not None and k < 10:
            img = Image.blend(prev, img, (k + 1) / 10)
        enc.stdin.write(img.tobytes())
        last = img
        total += 1
    prev = last
enc.stdin.close()
enc.wait()
print("frames", total, "seconds", total / FPS, "exit", enc.returncode)
