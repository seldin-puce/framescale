"""Product Hunt gallery images (1270×760) for FrameScale AI."""
import sys
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ICON, BEFORE, AFTER, OUT = sys.argv[1:5]
W, H = 1270, 760
NOTO = "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf"
BG, FG, MUTED, SUBTLE, ACCENT = "#0b0f19", "#e6e9f0", "#9aa3b5", "#6b7489", "#4c8dff"

def font(size, wght):
    f = ImageFont.truetype(NOTO, size)
    f.set_variation_by_axes([wght])
    return f

def canvas(glow_box):
    img = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), BG)
    ImageDraw.Draw(glow).ellipse(glow_box, fill="#1d3f8f")
    return Image.blend(img, glow.filter(ImageFilter.GaussianBlur(170)), 0.8)

def rounded(im, r):
    m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, im.size[0] - 1, im.size[1] - 1), radius=r, fill=255)
    out = Image.new("RGBA", im.size)
    out.paste(im, (0, 0), m)
    return out

def pill(d, x, y, text, f, fill="#111827", outline="#2a3a5c", color="#c9d2e3"):
    w = d.textlength(text, font=f) + 32
    d.rounded_rectangle((x, y, x + w, y + 40), radius=20, fill=fill, outline=outline, width=2)
    d.text((x + 16, y + 6), text, font=f, fill=color)
    return w

icon = Image.open(ICON).convert("RGBA")

# --- 1: hero with the real before/after ---------------------------------------
img = canvas((700, -200, 1500, 700))
d = ImageDraw.Draw(img)
img.paste(icon.resize((64, 64), Image.LANCZOS), (60, 56), icon.resize((64, 64), Image.LANCZOS))
d.text((140, 62), "FrameScale AI", font=font(34, 700), fill=FG)
d.text((60, 140), "Free AI video", font=font(52, 760), fill="#ffffff")
d.text((60, 200), "upscaler &", font=font(52, 760), fill="#8fb6ff")
d.text((60, 260), "enhancer", font=font(52, 760), fill="#8fb6ff")
d.text((60, 344), "Upscale to 4K, denoise, restore faces", font=font(22, 450), fill=MUTED)
d.text((60, 374), "and convert to 60 fps, on your own GPU.", font=font(22, 450), fill=MUTED)
x = 60
for t in ["Open source", "No uploads", "No watermark"]:
    x += pill(d, x, 428, t, font(18, 550)) + 10
d.text((60, 494), "Windows · macOS · Linux", font=font(19, 600), fill=SUBTLE)
d.text((60, 522), "NVIDIA · AMD · Intel · Apple Silicon", font=font(19, 600), fill=SUBTLE)
d.text((60, 690), "framescale.org", font=font(24, 650), fill=ACCENT)

# The same detail before and after, enlarged.
box = (340, 30, 640, 430)  # the man, in the 1280×536 frames
pw, ph_ = 300, 400
def panel(path, label, col):
    im = Image.open(path).convert("RGB").crop(box).resize((pw, ph_), Image.LANCZOS)
    pd = ImageDraw.Draw(im)
    tw = pd.textlength(label, font=font(16, 750))
    pd.rounded_rectangle((12, 12, 12 + tw + 20, 44), radius=8, fill="#000000")
    pd.text((22, 16), label, font=font(16, 750), fill=col)
    return rounded(im, 16)
px0, py = W - 2 * pw - 16 - 56, 150
for i, (path, label, col) in enumerate([(BEFORE, "BEFORE · normal scaling", "#ffffff"), (AFTER, "AFTER · FrameScale", "#9fc4ff")]):
    x = px0 + i * (pw + 16)
    sh = Image.new("RGBA", (pw + 60, ph_ + 60), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((30, 40, pw + 30, ph_ + 40), radius=16, fill=(0, 0, 0, 160))
    sh = sh.filter(ImageFilter.GaussianBlur(16))
    img.paste(sh, (x - 30, py - 30), sh)
    pnl = panel(path, label, col)
    img.paste(pnl, (x, py), pnl)
    ImageDraw.Draw(img).rounded_rectangle((x, py, x + pw - 1, py + ph_ - 1), radius=16, outline="#2a3a5c", width=2)
d = ImageDraw.Draw(img)
d.text((px0, py + ph_ + 16), "Same 640 px frame, enlarged 2×. Real-ESRGAN General v3 in FrameScale AI.", font=font(15, 450), fill=SUBTLE)
d.text((px0, py + ph_ + 38), "Footage: Tears of Steel, CC BY 3.0 Blender Foundation.", font=font(15, 450), fill=SUBTLE)
img.save(f"{OUT}/1-before-after.png", optimize=True)

# --- 2: what it does ----------------------------------------------------------
img = canvas((200, -300, 1100, 500))
d = ImageDraw.Draw(img)
title = "One pipeline. Five AI stages."
tw = d.textlength(title, font=font(50, 760))
d.text(((W - tw) / 2, 62), title, font=font(50, 760), fill="#ffffff")
sub = "Turn on what your footage needs. Preview 2 seconds of the real result, then render."
sw = d.textlength(sub, font=font(22, 450))
d.text(((W - sw) / 2, 134), sub, font=font(22, 450), fill=MUTED)
stages = [
    ("01", "Deinterlace", "BWDIF", "Old TV, VHS and DV\nto smooth 50/60p"),
    ("02", "Denoise", "NAFNet · Real-CUGAN", "Camera grain and\ncompression noise"),
    ("03", "Upscale", "Real-ESRGAN", "2× or 4×, from SD\nall the way to 4K"),
    ("04", "Face Fix", "CodeFormer", "Detects and restores\nfaces automatically"),
    ("05", "Interpolate", "RIFE 4.25", "2×–4× frame rate,\nscene-cut aware"),
]
cw, gap, top = 220, 18, 206
x0 = (W - (5 * cw + 4 * gap)) / 2
for i, (n, name, model, desc) in enumerate(stages):
    x = x0 + i * (cw + gap)
    d.rounded_rectangle((x, top, x + cw, top + 280), radius=18, fill="#111827", outline="#24324f", width=2)
    d.rounded_rectangle((x + 20, top + 22, x + 64, top + 66), radius=11, fill="#16264a", outline="#2e4c86", width=2)
    d.text((x + 29, top + 31), n, font=font(19, 750), fill=ACCENT)
    d.text((x + 20, top + 86), name, font=font(27, 720), fill=FG)
    d.text((x + 20, top + 126), model, font=font(16, 600), fill=ACCENT)
    for k, line in enumerate(desc.split("\n")):
        d.text((x + 20, top + 172 + k * 28), line, font=font(18, 450), fill=MUTED)
    if i < 4:
        ax = x + cw + gap / 2
        d.text((ax - 7, top + 126), "›", font=font(28, 700), fill="#3a4f7a")
facts = [("Runs on your GPU", "NVIDIA · AMD · Intel · Apple"), ("Keeps everything", "HDR10 · audio · subtitles"),
         ("Modern output", "HEVC 10-bit · AV1 · ProRes"), ("Free & open source", "GPL-3.0 · no uploads")]
fw = (W - 120 - 3 * 18) / 4
for i, (a, b) in enumerate(facts):
    x = 60 + i * (fw + 18)
    d.rounded_rectangle((x, 530, x + fw, 630), radius=16, fill="#0f1626", outline="#1f2b45", width=2)
    d.text((x + 20, 548), a, font=font(21, 700), fill=FG)
    d.text((x + 20, 584), b, font=font(17, 500), fill=SUBTLE)
fx = (W - 64 - 14 - d.textlength("FrameScale AI  ·  framescale.org", font=font(24, 650))) / 2
img.paste(icon.resize((40, 40), Image.LANCZOS), (int(fx), 676), icon.resize((40, 40), Image.LANCZOS))
d.text((fx + 54, 680), "FrameScale AI  ·  framescale.org", font=font(24, 650), fill=FG)
img.save(f"{OUT}/2-features.png", optimize=True)
print("ok")
