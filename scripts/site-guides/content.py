# Guide content for framescale.org (exec'd by gen.py). Facts match FrameScale AI 0.1.

BA = """<figure class="ba-wrap" style="margin:28px 0 8px">
  <div class="ba" id="ba">
    <img src="/img/after.jpg" alt="The same frame upscaled by FrameScale AI: sharper edges, texture and faces" width="1280" height="536" loading="lazy">
    <img class="before" src="/img/before.jpg" alt="Low-resolution video frame enlarged with normal scaling" width="1280" height="536" loading="lazy">
    <span class="lbl l">Before</span><span class="lbl r">After · FrameScale</span>
    <div class="line"></div>
    <input type="range" min="0" max="100" value="50" aria-label="Drag to compare before and after">
    <div class="knob" aria-hidden="true">⇆</div>
  </div>
  <figcaption class="ba-cap">640-pixel-wide frame enlarged 2× normally (left) vs FrameScale AI with Real-ESRGAN General v3 (noise removal 20%) and Face Fix (right). Footage: <a href="https://mango.blender.org/">Tears of Steel</a> © Blender Foundation, CC BY 3.0.</figcaption>
</figure>
<script>(function(){var b=document.getElementById("ba");if(!b)return;var r=b.querySelector("input");var s=function(){b.style.setProperty("--pos",r.value+"%")};r.addEventListener("input",s);s();})();</script>
"""

INSTALL = ("Install FrameScale AI",
           'Download the installer for Windows, macOS or Linux from the <a href="/#download">download section</a> and install it. It is free, needs no account, and runs entirely on your computer.')
ADD = ("Add your video",
       "Open FrameScale AI and drag your clip onto <strong>Drop videos here</strong>. It reads anything FFmpeg can decode: MP4, MKV, MOV, AVI, MPEG-2, ProRes and more.")

GUIDES = [
dict(
 slug="upscale-video-to-4k",
 crumb="Upscale video to 4K",
 card="Turn 1080p, 720p or SD footage into sharp 4K with AI, for free, on your own GPU.",
 title="Upscale Video to 4K Free with AI (2026 Guide) | FrameScale",
 h1="How to upscale video to 4K for free with AI",
 desc="Step-by-step: upscale 1080p, 720p or SD video to 4K with free, open-source AI on your own GPU. Model choice, settings, speed and export tips.",
 lead="Normal scaling only makes pixels bigger. AI upscaling rebuilds detail the source never stored, so edges, text and textures look sharp at 4K. Here is how to do it for free with FrameScale AI, without uploading your video anywhere.",
 read="7",
 time="PT15M",
 body="""<h2>What AI upscaling actually does</h2>
<p>When a player or editor enlarges 1080p to 4K, it interpolates between existing pixels (bicubic or Lanczos). The result is bigger but soft. An AI super-resolution model such as <strong>Real-ESRGAN</strong> was trained on millions of image pairs, so it predicts plausible high-resolution detail: crisp edges, readable text, clean textures and less blocky compression.</p>
""" + BA + """
<h2>Which scale to use</h2>
<div class="table-wrap"><table class="table">
<thead><tr><th>Source</th><th>Scale</th><th>Result</th></tr></thead>
<tbody>
<tr><td>1920×1080 (Full HD)</td><td>2×</td><td>3840×2160 (4K UHD)</td></tr>
<tr><td>1280×720 (HD)</td><td>2×</td><td>2560×1440 (QHD)</td></tr>
<tr><td>960×540</td><td>4×</td><td>3840×2160 (4K UHD)</td></tr>
<tr><td>720×576 / 720×480 (DVD, SD)</td><td>4×</td><td>about 2880 pixels wide, beyond Full HD</td></tr>
</tbody></table></div>
<h2>Step by step</h2>
""",
 steps=[INSTALL, ADD,
  ("Turn on Upscale and pick a model",
   "In the pipeline on the right, switch on <strong>Upscale</strong>. For camera footage choose <strong>Real-ESRGAN x4+ (live action)</strong> for the best detail, or <strong>Real-ESRGAN General v3 (live action, fast)</strong> for speed, with a <strong>Noise removal</strong> slider. For anime and cartoons choose <strong>Real-ESRGAN x4+ Anime</strong> or <strong>Real-ESR AnimeVideo v3 (fast)</strong>."),
  ("Choose 2× or 4×",
   "Use <strong>2×</strong> for 1080p to 4K and <strong>4×</strong> for low-resolution sources (see the table above)."),
  ("Preview 2 seconds",
   "Drag the marker under the viewer to a busy scene and click <strong>Render 2 s</strong>. A progress bar shows the time left; then compare the result split or side by side, and zoom in to check fine detail."),
  ("Pick the output format",
   "Under output, choose <strong>HEVC 10-bit</strong> (best compatibility for 4K) or <strong>AV1 10-bit</strong> (smallest files) in MP4 or MKV, and a quality level. HDR, audio, subtitles and chapters are carried over automatically."),
  ("Render",
   "Click <strong>Render</strong>. The job is written in 4-second segments, so you can pause, quit or even survive a crash and continue where it stopped."),
 ],
 after="""<h2>Tips for the best 4K result</h2>
<ul>
<li><strong>Clean before you enlarge.</strong> Noise and compression blocks get upscaled too. Turn on <strong>Denoise</strong> (NAFNet for camera footage) before Upscale for grainy or heavily compressed sources.</li>
<li><strong>Faces look soft?</strong> Add <strong>Face Fix</strong>. It finds faces and restores them with CodeFormer; the Fidelity slider keeps them closer to the original.</li>
<li><strong>Speed.</strong> On a modern GPU, Real-ESRGAN General v3 turns 540p into 4K at around 13 frames per second; the heavier x4+ model is slower but more detailed. Eco mode caps GPU load for long, quiet renders.</li>
<li><strong>Keep the original.</strong> FrameScale never overwrites your source or an existing file.</li>
</ul>
<div class="tip"><p><strong>Old tapes?</strong> Interlaced VHS and DV need deinterlacing first. See <a href="/restore-vhs-tapes/">how to restore VHS tapes with AI</a>.</p></div>
""",
 faq=[
  ("Can I upscale video to 4K for free?", "Yes. FrameScale AI is free and open source, with no watermark, no resolution limit and no account. It uses your own GPU, so nothing is uploaded."),
  ("Does AI upscaling really add detail?", "It reconstructs plausible detail learned from training data, which looks far sharper than normal scaling. It cannot recover information that was never captured, so very blurry sources improve less."),
  ("How long does upscaling take?", "It depends on the model, resolution and GPU. On a current mid-to-high-end GPU, the fast General v3 model upscales 540p to 4K at roughly 13 frames per second; a 10-minute 1080p video to 4K takes from minutes to a few hours."),
  ("What GPU do I need?", "Any Vulkan-capable NVIDIA, AMD or Intel GPU, or a Mac with Apple Silicon. 4 GB of video memory is comfortable; it also runs on the CPU, much more slowly."),
 ],
),
dict(
 slug="restore-vhs-tapes",
 crumb="Restore VHS tapes with AI",
 card="Deinterlace, denoise, upscale and fix faces in old VHS and camcorder footage.",
 title="How to Restore Old VHS Tapes with AI (Free) | FrameScale",
 h1="How to restore old VHS tapes with AI",
 desc="Restore digitized VHS, Hi8 and camcorder tapes for free: deinterlace, remove noise, upscale to HD and fix faces with open-source AI on your own GPU.",
 lead="Old home videos are interlaced, noisy and only about 720×576 pixels. With the right order of AI steps you can turn them into smooth, clean HD video that looks good on a modern TV, while keeping the original look.",
 read="8",
 time="PT20M",
 body="""<h2>Before you start: digitize the tape</h2>
<p>FrameScale AI works on video files, so the tape has to be captured first, with a VCR (ideally with a time-base corrector) and a USB capture device. For the best restoration:</p>
<ul>
<li>Capture at the tape's native size (720×576 PAL or 720×480 NTSC) and <strong>keep it interlaced</strong>. Deinterlacing is done better later.</li>
<li>Use a lossless or high-bitrate format (for example FFV1, ProRes or high-bitrate MPEG-2), not a low-bitrate MP4.</li>
<li>Keep the capture file as your master and never overwrite it.</li>
</ul>
""" + BA + """
<h2>Step by step</h2>
""",
 steps=[INSTALL, ADD,
  ("Deinterlace at double rate",
   "Switch on <strong>Deinterlace</strong> (BWDIF, motion adaptive) and choose <strong>Double rate</strong>. Each field becomes a full frame, so PAL tapes play at 50 fps and NTSC at 60 fps, with the fluid motion of the original."),
  ("Remove tape noise",
   "Switch on <strong>Denoise</strong> with the <strong>NAFNet (live action, camera noise)</strong> model. Start at <strong>Medium</strong>; use <strong>Low</strong> or <strong>Gentle</strong> to keep more of the tape's grain."),
  ("Upscale to HD",
   "Switch on <strong>Upscale</strong> with <strong>Real-ESRGAN General v3</strong> and choose <strong>2×</strong> (about 1440×1152) or <strong>4×</strong> for 4K-class output. Keep its Noise removal near 50% if you already denoised."),
  ("Fix faces",
   "Switch on <strong>Face Fix</strong>. Faces in home videos are often only a few dozen pixels tall; CodeFormer restores eyes, mouths and skin. A Fidelity around 0.7 keeps people recognizable."),
  ("Preview, then render",
   "Move the marker to a scene with faces and motion, click <strong>Render 2 s</strong>, compare, adjust, then click <strong>Render</strong>. The 4:3 aspect ratio, audio and chapters are kept."),
 ],
 after="""<h2>Common VHS problems and fixes</h2>
<ul>
<li><strong>Comb lines on motion</strong> mean the video is still interlaced: turn on Deinterlace.</li>
<li><strong>Plastic-looking skin</strong> comes from too much denoising: lower Denoise to Low and raise Face Fix fidelity.</li>
<li><strong>Squeezed or stretched picture</strong>: FrameScale keeps the source's pixel aspect ratio, so 4:3 tapes stay 4:3. If your capture was already wrong, fix the aspect in your capture software.</li>
<li><strong>Jittery motion after deinterlacing</strong>: choose Double rate instead of Same rate.</li>
</ul>
<div class="tip"><p><strong>Want even smoother motion?</strong> After deinterlacing, <a href="/convert-video-to-60fps/">frame interpolation</a> can double the frame rate again.</p></div>
""",
 faq=[
  ("Can AI really restore VHS tapes?", "It can remove noise, sharpen, upscale and restore faces dramatically. It cannot repair tracking errors or dropouts that were captured into the file, so a clean capture still matters most."),
  ("Should I deinterlace before upscaling?", "Yes. Upscaling interlaced video enlarges the comb artifacts. FrameScale always runs Deinterlace first, then Denoise, Upscale, Face Fix and Interpolate."),
  ("What resolution should I upscale VHS to?", "2× (about 1440×1152 for PAL) is a good balance of detail and speed. 4× gives a 4K-class image, but VHS holds little real detail, so the gain over 2× is smaller."),
  ("Is it free?", "Yes. FrameScale AI is free and open-source software that runs locally, with no watermark and no upload."),
 ],
),
dict(
 slug="convert-video-to-60fps",
 crumb="Convert video to 60 fps",
 card="Double or triple the frame rate with RIFE frame interpolation for smoother motion.",
 title="Convert Video to 60 FPS with AI for Free | FrameScale",
 h1="How to convert video to 60 fps with AI frame interpolation",
 desc="Make 30 fps or 25 fps video smooth 60 or 50 fps for free with RIFE AI frame interpolation. Settings, scene cuts, best models and tips.",
 lead="Frame interpolation creates new in-between frames, so 30 fps video plays at a smooth 60 fps. Modern AI models such as RIFE estimate motion between frames and synthesize the missing ones with far fewer artifacts than a TV's motion smoothing.",
 read="6",
 time="PT10M",
 body="""<h2>Which multiplier gives which frame rate</h2>
<div class="table-wrap"><table class="table">
<thead><tr><th>Source</th><th>Multiplier</th><th>Result</th></tr></thead>
<tbody>
<tr><td>30 fps (phones, web)</td><td>2×</td><td>60 fps</td></tr>
<tr><td>25 fps (PAL)</td><td>2×</td><td>50 fps</td></tr>
<tr><td>24 fps (film)</td><td>2× / 3×</td><td>48 fps / 72 fps</td></tr>
<tr><td>30 fps</td><td>4×</td><td>120 fps (slow motion)</td></tr>
</tbody></table></div>
<p>FrameScale multiplies the frame rate by 2, 3 or 4. Interlaced sources (VHS, DV, broadcast) already reach 50 or 60 fps with <strong>Deinterlace → Double rate</strong>, without synthesizing frames.</p>
<h2>Step by step</h2>
""",
 steps=[INSTALL, ADD,
  ("Turn on Interpolate",
   "Switch on <strong>Interpolate</strong>. The default model, <strong>RIFE 4.25</strong>, is the current recommendation of RIFE's authors for most footage, live action and anime alike. RIFE 4.26 and 4.22 lite are also included."),
  ("Choose the multiplier",
   "Set <strong>Frame-rate multiplier</strong> to <strong>2×</strong> for 30→60 or 25→50 fps, or higher for slow motion."),
  ("Check scene-cut sensitivity",
   "At hard cuts FrameScale holds the frame instead of morphing two different shots into each other. Lower the <strong>Scene-cut sensitivity</strong> value to catch more cuts if you see blended frames at edits."),
  ("Preview and render",
   "Click <strong>Render 2 s</strong> on a scene with fast motion, play the preview, then click <strong>Render</strong>. Audio stays in sync."),
 ],
 after="""<h2>Tips</h2>
<ul>
<li><strong>Upscale and interpolate together.</strong> You can combine Upscale and Interpolate in one render, for example 1080p30 to 4K60.</li>
<li><strong>Very fast motion</strong>, such as sports or whip pans, is the hardest case for any interpolator. Preview those scenes first.</li>
<li><strong>Film look.</strong> Doubling 24 fps film to 48 fps changes its character (the "soap opera effect"). Many people prefer interpolation for sports, gaming and home video rather than movies.</li>
<li><strong>Speed.</strong> RIFE 4.25 interpolates 1080p at around 14 frames per second on a current GPU.</li>
</ul>
""",
 faq=[
  ("Is AI frame interpolation better than TV motion smoothing?", "Usually yes. RIFE estimates motion with a neural network trained for this task and runs offline at full quality, so it produces fewer artifacts than real-time TV processing."),
  ("Can I convert 24 fps to 60 fps?", "FrameScale multiplies by 2, 3 or 4, so 24 fps becomes 48 or 72 fps. For 60 fps output, 30 fps sources (×2) are the natural fit."),
  ("Does interpolation affect audio?", "No. The audio is copied unchanged and stays in sync, because the duration of the video doesn't change."),
  ("Which RIFE model should I use?", "RIFE 4.25 is the default and the best choice for most footage. RIFE 4.22 lite is lighter for older GPUs; RIFE 4.6 is the classic model."),
 ],
),
dict(
 slug="topaz-video-ai-alternative",
 crumb="Free Topaz Video AI alternative",
 card="A free, open-source option for AI upscaling, denoising, face recovery and 60 fps.",
 title="Free Topaz Video AI Alternative (Open Source) | FrameScale",
 h1="A free, open-source alternative to Topaz Video AI",
 desc="Looking for a free Topaz Video AI alternative? FrameScale AI upscales to 4K, denoises, restores faces and interpolates to 60 fps, open source and local.",
 lead="Topaz Video AI is a popular commercial video enhancer. If you want the same core tasks, AI upscaling, denoising, face restoration and frame interpolation, without a paid license, FrameScale AI is a free and open-source option that runs on your own GPU.",
 read="5",
 body="""<h2>What FrameScale AI covers</h2>
<div class="table-wrap"><table class="table">
<thead><tr><th>Task</th><th>FrameScale AI</th></tr></thead>
<tbody>
<tr><td>AI upscaling (2×, 4×, up to 4K and beyond)</td><td>Real-ESRGAN x4+, General v3, Anime and AnimeVideo v3</td></tr>
<tr><td>Denoising</td><td>NAFNet (camera noise), Real-CUGAN (anime)</td></tr>
<tr><td>Face restoration</td><td>CodeFormer with automatic face detection</td></tr>
<tr><td>Frame interpolation / 60 fps</td><td>RIFE 4.25, 4.26, 4.22 lite and 4.6, scene-cut aware</td></tr>
<tr><td>Deinterlacing</td><td>BWDIF, same or double rate</td></tr>
<tr><td>HDR</td><td>HDR10 and HLG carried through</td></tr>
<tr><td>Output</td><td>HEVC 10-bit, AV1 10-bit, H.264; ProRes on macOS</td></tr>
<tr><td>Platforms</td><td>Windows, macOS (Apple Silicon and Intel) and Linux</td></tr>
<tr><td>Price and license</td><td>Free, open source (GPL-3.0); no account, no watermark</td></tr>
<tr><td>Privacy</td><td>Runs locally; nothing is uploaded</td></tr>
</tbody></table></div>
""" + BA + """
<h2>Where each tool fits</h2>
<p><strong>Choose FrameScale AI</strong> if you want a free tool, open-source code you can inspect and modify, Linux support, or full control over which open models process your footage. Its models are widely used, well-documented open research models.</p>
<p><strong>A commercial suite may suit you better</strong> if you rely on its own proprietary models, plug-ins for editing software, or paid support. Both kinds of tools run locally on your GPU.</p>
<h2>Switching in three steps</h2>
""",
 steps=[INSTALL,
  ("Rebuild your usual pipeline",
   "Turn on the stages you used before: <strong>Upscale</strong> for enhancement, <strong>Denoise</strong> for noise, <strong>Face Fix</strong> for faces and <strong>Interpolate</strong> for frame-rate conversion."),
  ("Compare on your own footage",
   "Use <strong>Render 2 s</strong> to preview the exact result on a short section and compare it side by side with the original before rendering the whole video."),
 ],
 after="""<p>Step-by-step guides for the common jobs: <a href="/upscale-video-to-4k/">upscale video to 4K</a>, <a href="/restore-vhs-tapes/">restore VHS tapes</a> and <a href="/convert-video-to-60fps/">convert video to 60 fps</a>.</p>
<p class="meta">Topaz Video AI is a trademark of Topaz Labs. FrameScale AI is an independent open-source project, not affiliated with Topaz Labs.</p>
""",
 faq=[
  ("Is there a free alternative to Topaz Video AI?", "Yes. FrameScale AI is free and open source and covers AI upscaling, denoising, face restoration, deinterlacing and frame interpolation on Windows, macOS and Linux."),
  ("Is FrameScale AI as good as Topaz Video AI?", "They use different models, so results differ by footage. FrameScale uses established open models (Real-ESRGAN, NAFNet, CodeFormer, RIFE). The best test is a 2-second preview on your own clip, which is free."),
  ("Does FrameScale AI work on Linux?", "Yes. It runs on Windows, macOS and Linux, on NVIDIA, AMD, Intel and Apple GPUs through Vulkan."),
  ("Does it add a watermark or need an account?", "No. There is no watermark, no account and no limit on resolution or video length."),
 ],
),
]
