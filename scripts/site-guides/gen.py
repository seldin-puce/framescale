"""Generates the framescale.org guide pages (site/<slug>/index.html).

usage: python3 scripts/site-guides/gen.py site   (content lives in content.py)
"""
import json, os, sys, html

SITE = sys.argv[1]
R = "https://github.com/seldin-puce/framescale"
BASE = "https://framescale.org"
DATE = "2026-11-03"

GUIDES = []  # filled below: dict(slug, title, h1, desc, lead, steps(list of (name, html)), body(html before steps), after(html after steps), faq(list))

def page(g, all_guides):
    url = f"{BASE}/{g['slug']}/"
    ld = [
        {"@context": "https://schema.org", "@type": "Article", "headline": g["h1"], "description": g["desc"],
         "datePublished": DATE, "dateModified": DATE, "image": f"{BASE}/og.png", "mainEntityOfPage": url,
         "author": {"@type": "Organization", "name": "FrameScale AI", "url": BASE + "/"},
         "publisher": {"@type": "Organization", "name": "FrameScale AI", "logo": {"@type": "ImageObject", "url": f"{BASE}/icon-512.png"}}},
        {"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": [
            {"@type": "ListItem", "position": 1, "name": "FrameScale AI", "item": BASE + "/"},
            {"@type": "ListItem", "position": 2, "name": "Guides", "item": BASE + "/#guides"},
            {"@type": "ListItem", "position": 3, "name": g["crumb"], "item": url}]},
    ]
    if g.get("steps"):
        ld.append({"@context": "https://schema.org", "@type": "HowTo", "name": g["h1"], "description": g["desc"],
                   "tool": [{"@type": "HowToTool", "name": "FrameScale AI"}],
                   "totalTime": g.get("time", "PT10M"),
                   "step": [{"@type": "HowToStep", "position": i + 1, "name": n, "text": html.unescape(strip(t)), "url": f"{url}#step-{i+1}"} for i, (n, t) in enumerate(g["steps"])]})
    if g.get("faq"):
        ld.append({"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": [
            {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in g["faq"]]})
    steps = ""
    if g.get("steps"):
        steps = '<ol class="steps">\n' + "".join(
            f'  <li id="step-{i+1}"><strong>{n}.</strong> {t}</li>\n' for i, (n, t) in enumerate(g["steps"])) + "</ol>\n"
    faq = ""
    if g.get("faq"):
        faq = '<h2 id="faq">Frequently asked questions</h2>\n<div class="faq">\n' + "".join(
            f"  <details>\n    <summary>{q}</summary>\n    <p>{a}</p>\n  </details>\n" for q, a in g["faq"]) + "</div>\n"
    related = "".join(
        f'<a class="card" href="/{o["slug"]}/"><h3 style="margin-top:0">{o["crumb"]}</h3><p>{o["card"]}</p><span class="more">Read the guide →</span></a>'
        for o in all_guides if o["slug"] != g["slug"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{g['title']}</title>
<meta name="description" content="{g['desc']}">
<meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1">
<link rel="canonical" href="{url}">
<link rel="icon" href="/icon.svg" type="image/svg+xml">
<link rel="icon" href="/favicon-32.png" type="image/png" sizes="32x32">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<link rel="manifest" href="/site.webmanifest">
<meta name="theme-color" content="#0b0f19">
<meta name="color-scheme" content="dark">
<meta property="og:type" content="article">
<meta property="og:site_name" content="FrameScale AI">
<meta property="og:locale" content="en_US">
<meta property="og:url" content="{url}">
<meta property="og:title" content="{g['h1']}">
<meta property="og:description" content="{g['desc']}">
<meta property="og:image" content="{BASE}/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="article:published_time" content="{DATE}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{g['h1']}">
<meta name="twitter:description" content="{g['desc']}">
<meta name="twitter:image" content="{BASE}/og.png">
<link rel="stylesheet" href="/style.css">
<script type="application/ld+json">
{json.dumps(ld, indent=1, ensure_ascii=False)}
</script>
</head>
<body>
<header class="nav">
  <div class="wrap">
    <a class="brand" href="/" aria-label="FrameScale AI home"><img src="/icon.svg" alt="" width="28" height="28">FrameScale AI</a>
    <nav>
      <a class="opt" href="/#features">Features</a>
      <a class="opt" href="/#guides">Guides</a>
      <a href="/#download">Download</a>
      <a href="{R}">GitHub</a>
    </nav>
  </div>
</header>
<main>
<article class="guide">
<nav class="crumbs" aria-label="Breadcrumb"><a href="/">FrameScale AI</a> › <a href="/#guides">Guides</a> › {g['crumb']}</nav>
<h1>{g['h1']}</h1>
<p class="meta">Updated November 2026 · {g.get('read', '6')} min read</p>
<p class="lead">{g['lead']}</p>
{g.get('body', '')}
{steps}
{g.get('after', '')}
{faq}
<div class="cta-box">
  <h2>Try it on your own footage</h2>
  <p>FrameScale AI is free and open source for Windows, macOS and Linux. No account, no watermark, no upload.</p>
  <div class="cta" style="margin-top:16px"><a class="btn primary" href="/#download">Download FrameScale AI</a><a class="btn ghost" href="{R}">View on GitHub</a></div>
</div>
</article>
<section id="more-guides" style="border-top:0;padding-top:24px">
  <div class="wrap">
    <h2>More guides</h2>
    <div class="guides">{related}</div>
  </div>
</section>
</main>
<footer>
  <div class="wrap">
    <span>© 2026 FrameScale AI contributors · GPL-3.0-or-later</span>
    <nav>
      <a href="/">Home</a>
      <a href="{R}">GitHub</a>
      <a href="{R}/releases">Releases</a>
    </nav>
  </div>
</footer>
</body>
</html>
"""

import re
def strip(t):
    return re.sub(r"<[^>]+>", "", t)

exec(open(os.path.join(os.path.dirname(__file__), "content.py")).read())
for g in GUIDES:
    d = os.path.join(SITE, g["slug"])
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "index.html"), "w").write(page(g, GUIDES))
    print("wrote", g["slug"], len(page(g, GUIDES)))
