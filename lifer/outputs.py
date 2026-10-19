"""Run outputs: candidates CSV + a self-contained review HTML page.

The review page is a single file with no external dependencies (system fonts,
inline CSS/JS, images referenced by relative path) so it opens from file://
anywhere. It groups encounters into sections -- Rare here, Possible lifers,
Look-alikes, Everything else, No bird detected -- with a sticky header of
jump-links, a responsive card grid, and a lightbox that shows the full frame
(with the detection box) and toggles to the tight crop.
"""

from __future__ import annotations

import html
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

# Section keys in display order. "rare" wins over the others (see pipeline).
SECTIONS: list[tuple[str, str]] = [
    ("rare", "Rare here"),
    ("lifer", "Possible lifers"),
    ("lookalike", "Look-alikes"),
    ("other", "Everything else"),
]


@dataclass
class Candidate:
    enc_id: str
    time_start: str
    time_end: str
    n_frames: int
    best_frame: str
    det_conf: float
    raw_top3: list[tuple[str, str, float]]
    adj_top3: list[tuple[str, str, float]]
    possible_lifer: bool
    lifer_lookalike: str | None
    has_det: bool = False
    rare_here: bool = False
    rare_evidence: str = ""


def save_crop(crop: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.convert("RGB").save(out_path, "JPEG", quality=88)


def save_frame_preview(image: Image.Image, det, out_path: Path,
                       max_side: int = 2000, quality: int = 85) -> None:
    """Downscale the full frame to ~max_side px on the long edge, save as JPEG.

    If `det` is given its box (scaled to the preview) is drawn so the lightbox
    can show the bird in context; with det=None no box is drawn (used for the
    no-bird grid thumbnails).
    """
    W, H = image.size
    scale = max_side / max(W, H)
    if scale < 1.0:
        image = image.resize((int(round(W * scale)), int(round(H * scale))), Image.Resampling.LANCZOS)
    if det is not None:
        x1, y1 = det.x1 * scale, det.y1 * scale
        x2, y2 = det.x2 * scale, det.y2 * scale
        lw = max(3, int(round(max(W, H) * scale / 400)))
        ImageDraw.Draw(image).rectangle([x1, y1, x2, y2], outline=(255, 196, 0), width=lw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path, "JPEG", quality=quality)


def write_candidates_csv(cands: list[Candidate], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["enc_id", "time_start", "time_end", "n_frames", "best_frame",
                    "det_conf", "raw_top3", "adj_top3", "possible_lifer",
                    "lifer_lookalike", "rare_here", "rare_evidence"])
        for c in cands:
            w.writerow([
                c.enc_id,
                c.time_start,
                c.time_end,
                c.n_frames,
                c.best_frame,
                f"{c.det_conf:.3f}",
                "; ".join(f"{n} ({p:.3f})" for n, _s, p in c.raw_top3),
                "; ".join(f"{n} ({p:.3f})" for n, _s, p in c.adj_top3),
                c.possible_lifer,
                c.lifer_lookalike or "",
                c.rare_here,
                c.rare_evidence,
            ])


def _fmt_top3_str(items: list[tuple[str, str, float]]) -> str:
    return "; ".join(f"{n} ({p:.3f})" for n, _s, p in items)


def _month_name(ymd: str) -> str:
    import calendar
    return f"{calendar.month_name[int(ymd[5:7])]} {int(ymd[8:10])}, {ymd[:4]}"


def _enc_card(c: Candidate) -> dict:
    adj1 = c.adj_top3[0]
    badges = []
    if c.rare_here:
        badges.append("rare")
    if c.possible_lifer:
        badges.append("lifer")
    if c.lifer_lookalike:
        badges.append(f"lookalike:{c.lifer_lookalike}")
    return {
        "id": c.enc_id,
        "species": adj1[0],
        "sci": adj1[1],
        "time": c.time_start[11:16],
        "n_frames": c.n_frames,
        "best_frame": c.best_frame,
        "det_conf": c.det_conf,
        "frame": f"frames/{c.enc_id}.jpg",
        "crop": f"crops/{c.enc_id}.jpg",
        "thumb": f"crops/{c.enc_id}.jpg",
        "badges": badges,
        "rare_evidence": c.rare_evidence,
        "adj_top3": [list(x) for x in c.adj_top3],
        "raw_top3": [list(x) for x in c.raw_top3],
    }


def _nodet_card(name: str, thumb_rel: str) -> dict:
    # one no-bird frame (not an encounter): filename + thumbnail only
    return {
        "id": name.rsplit(".", 1)[0],
        "species": None,
        "time": "",
        "n_frames": 0,
        "best_frame": name,
        "det_conf": 0.0,
        "frame": thumb_rel,
        "crop": None,
        "thumb": thumb_rel,
    }


def write_review_html(cands: list[Candidate], no_det: list[tuple[str, str]],
                      out_path: Path, folder: str,
                      region: dict | None = None) -> None:
    """Write the self-contained review page.

    `region` (optional) is {"name", "short", "month", "window"} for the
    rare-section title and badges, e.g. name="Wakulla County, FL",
    short="Wakulla Co.", month="Dec", window="Nov\u2013Jan".
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    esc = html.escape

    # Detected encounters go to their sections (rare beats lookalike). No-
    # detection frames come from `no_det` (per-frame, built in the pipeline),
    # not from the encounter list.
    sections: dict[str, list[dict]] = {k: [] for k, _ in SECTIONS}
    for c in cands:
        if not c.has_det:
            continue
        if c.rare_here:
            sections["rare"].append(_enc_card(c))
        elif c.possible_lifer:
            sections["lifer"].append(_enc_card(c))
        elif c.lifer_lookalike:
            sections["lookalike"].append(_enc_card(c))
        else:
            sections["other"].append(_enc_card(c))
    nodet: list[dict] = [_nodet_card(name, thumb) for name, thumb in no_det]

    n_enc = len(cands)
    n_rare = len(sections["rare"])
    n_lifer = len(sections["lifer"])
    n_la = len(sections["lookalike"])
    n_nodet = len(nodet)

    if cands:
        t0 = min(c.time_start for c in cands)[11:16]
        t1 = max(c.time_end for c in cands)[11:16]
        date_range = f"{_month_name(cands[0].time_start[:10])} &middot; {t0}\u2013{t1}"
    else:
        date_range = ""

    # header jump-links (encounters always; the rest only when non-empty)
    counts = [f'<a href="#top"><b>{n_enc}</b> encounters</a>']
    if n_rare:
        counts.append(f'<a href="#rare"><b>{n_rare}</b> rare here</a>')
    if n_lifer:
        counts.append(f'<a href="#lifer"><b>{n_lifer}</b> possible lifers</a>')
    if n_la:
        counts.append(f'<a href="#lookalike"><b>{n_la}</b> look-alikes</a>')
    if n_nodet:
        counts.append(f'<a href="#nodet"><b>{n_nodet}</b> no bird</a>')
    counts_html = "".join(counts)

    sec_html: list[str] = []
    for key, title in SECTIONS:
        items = sections[key]
        if not items:
            continue
        if key == "rare" and region and region.get("name"):
            title = f"Rare for {region['name']} in {region['window']}"
        cards = "".join(_card_html(c, i, key, region) for i, c in enumerate(items))
        sec_html.append(
            f'<section id="{key}"><h2>{title} <span class="sec-n">{len(items)}</span></h2>'
            f'<div class="grid">{cards}</div></section>')
    if nodet:
        nodes = "".join(_nodet_html(c, i) for i, c in enumerate(nodet))
        sec_html.append(
            f'<section id="nodet"><h2>No bird detected <span class="sec-n">{len(nodet)}</span></h2>'
            f'<div class="nodet-grid">{nodes}</div></section>')

    data_json = json.dumps({"sec": {k: sections[k] for k, _ in SECTIONS}, "nodet": nodet,
                            "region": region or {}})
    data_json = data_json.replace("</", "<\\/")

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>lifer review &mdash; {esc(folder)}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <div class="folder">{esc(folder)}</div>
  <div class="daterange">{date_range}</div>
  <nav class="counts">{counts_html}</nav>
</header>
<main id="top">
{chr(10).join(sec_html)}
</main>
<div class="lb" id="lb" role="dialog" aria-modal="true" aria-hidden="true">
  <div class="lb-backdrop" id="lb-backdrop"></div>
  <button class="lb-close" id="lb-close" aria-label="Close">&times;</button>
  <button class="lb-nav lb-prev" id="lb-prev" aria-label="Previous">&#8249;</button>
  <button class="lb-nav lb-next" id="lb-next" aria-label="Next">&#8250;</button>
  <div class="lb-wrap"><div class="lb-inner">
    <img class="lb-img" id="lb-img" alt="">
    <button class="lb-toggle" id="lb-toggle">Show crop</button>
    <div class="lb-caption" id="lb-caption"></div>
  </div></div>
</div>
<script>
const DATA = {data_json};
{JS}
</script>
</body>
</html>
"""
    out_path.write_text(html_doc)


def _badge_html(b: str, region: dict | None = None, evidence: str = "") -> str:
    esc = html.escape
    if b == "rare":
        label = "Rare here"
        if region and region.get("short") and region.get("month"):
            label = f"Rare in {region['short']} \u00b7 {region['month']}"
        tip = f' title="{esc(evidence)}"' if evidence else ""
        return f'<span class="badge rare"{tip}>{esc(label)}</span>'
    if b == "lifer":
        return '<span class="badge lifer">Possible lifer</span>'
    if b.startswith("lookalike:"):
        return f'<span class="badge lookalike">Look-alike: {esc(b.split(":", 1)[1])}</span>'
    return ""


def _raw_for(c: dict, common: str) -> str:
    for rc, _rs, rp in c["raw_top3"]:
        if rc == common:
            return f"{rp:.3f}"
    return "n/a"


def _card_html(c: dict, idx: int, sec: str, region: dict | None = None) -> str:
    esc = html.escape
    badges = "".join(_badge_html(b, region, c.get("rare_evidence", ""))
                     for b in c["badges"])
    rows = []
    for common, sci, p in c["adj_top3"]:
        bar_w = max(0.0, min(1.0, p)) * 100
        rows.append(
            f'<li title="raw {_raw_for(c, common)}">'
            f'<span class="nm" title="{esc(sci)}">{esc(common)}</span>'
            f'<span class="track"><span class="bar" style="width:{bar_w:.1f}%"></span></span>'
            f'<span class="val">{p:.2f}</span></li>')
    top3 = "".join(rows)
    return (
        f'<article class="card" data-sec="{sec}" data-idx="{idx}">'
        f'<img class="crop" src="{c["thumb"]}" loading="lazy" alt="{esc(c["species"])}">'
        f'<h3 class="title">{esc(c["species"])}{badges}</h3>'
        f'<ul class="top3">{top3}</ul>'
        f'<div class="meta">{c["time"]} &middot; {c["n_frames"]} frames &middot; {esc(c["best_frame"])}</div>'
        f'</article>')


def _nodet_html(c: dict, idx: int) -> str:
    esc = html.escape
    return (
        f'<figure class="nodet" data-idx="{idx}">'
        f'<img src="{c["thumb"]}" loading="lazy" alt="{esc(c["best_frame"])}">'
        f'<figcaption>{esc(c["best_frame"])}</figcaption></figure>')


# ---------------------------------------------------------------------------
# Static assets (plain strings: braces here are literal, not f-string fields)
# ---------------------------------------------------------------------------

CSS = """
:root {
  --bg:#f6f7f9; --card:#ffffff; --fg:#1a1d21; --muted:#6b7280;
  --border:#e5e7eb; --header-bg:rgba(255,255,255,.86);
  --bar-track:#eceef1; --bar:#4f7cff; --thumb-bg:#14161a;
  --rare-bg:#fff3e2; --rare-fg:#b45309; --rare-bd:#f3c583;
  --lifer-bg:#e9f9ef; --lifer-fg:#15803d; --lifer-bd:#8fd3a4;
  --la-bg:#eaf1ff; --la-fg:#1d4ed8; --la-bd:#9db9f5;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#0f1115; --card:#171a20; --fg:#e6e8eb; --muted:#9aa3ad;
    --border:#262b33; --header-bg:rgba(15,17,21,.86);
    --bar-track:#232833; --bar:#6b93ff; --thumb-bg:#050608;
    --rare-bg:#3a2410; --rare-fg:#fbbf24; --rare-bd:#6b4a1e;
    --lifer-bg:#0e2f1c; --lifer-fg:#4ade80; --lifer-bd:#1f5a37;
    --la-bg:#14213e; --la-fg:#93c5fd; --la-bd:#27406e;
  }
}
* { box-sizing:border-box; }
html { -webkit-text-size-adjust:100%; }
body {
  margin:0; background:var(--bg); color:var(--fg); line-height:1.5;
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
header {
  position:sticky; top:0; z-index:50; padding:12px 20px;
  background:var(--header-bg); border-bottom:1px solid var(--border);
  backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
}
.folder { font-size:1.25rem; font-weight:700; letter-spacing:-.01em; }
.daterange { color:var(--muted); font-size:.85rem; margin-top:2px; }
.counts { display:flex; flex-wrap:wrap; gap:8px 10px; margin-top:10px; }
.counts a {
  text-decoration:none; color:var(--fg); font-size:.82rem; font-weight:500;
  padding:4px 11px; border-radius:999px; background:var(--card);
  border:1px solid var(--border);
}
.counts a:hover { border-color:var(--bar); }
.counts a b { font-weight:700; }
main { max-width:1500px; margin:0 auto; }
section { margin:26px 20px 8px; scroll-margin-top:96px; }
section h2 {
  font-size:1.12rem; margin:0 0 14px; padding-bottom:7px;
  border-bottom:1px solid var(--border);
}
.sec-n { color:var(--muted); font-weight:400; font-size:.9rem; }
.grid {
  display:grid; gap:16px;
  grid-template-columns:repeat(auto-fill, minmax(280px, 1fr));
}
.card {
  background:var(--card); border:1px solid var(--border); border-radius:12px;
  padding:10px; transition:transform .08s ease, box-shadow .08s ease;
}
.card:hover { transform:translateY(-2px); box-shadow:0 8px 22px rgba(0,0,0,.10); }
.card img.crop {
  width:100%; aspect-ratio:1/1; object-fit:cover; border-radius:8px;
  display:block; background:var(--thumb-bg); cursor:zoom-in;
}
.title { margin:10px 2px 0; font-size:.95rem; font-weight:600; line-height:1.35; }
.badge {
  display:inline-block; font-size:.66rem; font-weight:700; padding:2px 8px;
  border-radius:999px; margin-left:6px; vertical-align:middle;
  border:1px solid transparent; white-space:nowrap;
}
.badge.rare { background:var(--rare-bg); color:var(--rare-fg); border-color:var(--rare-bd); }
.badge.lifer { background:var(--lifer-bg); color:var(--lifer-fg); border-color:var(--lifer-bd); }
.badge.lookalike { background:var(--la-bg); color:var(--la-fg); border-color:var(--la-bd); }
.top3 { list-style:none; margin:10px 2px 0; padding:0; display:flex; flex-direction:column; gap:5px; }
.top3 li { display:flex; align-items:center; gap:8px; font-size:.78rem; }
.top3 .nm { flex:0 1 45%; min-width:0; line-height:1.25; overflow-wrap:break-word; }
.top3 .track { flex:1; height:6px; background:var(--bar-track); border-radius:999px; overflow:hidden; }
.top3 .bar { display:block; height:100%; background:var(--bar); border-radius:999px; }
.top3 .val { flex:0 0 auto; font-variant-numeric:tabular-nums; color:var(--muted); }
.meta {
  margin:9px 2px 0; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:.72rem; color:var(--muted); word-break:break-all;
}
.nodet-grid { display:grid; gap:10px; grid-template-columns:repeat(auto-fill, minmax(110px, 1fr)); }
.nodet { margin:0; cursor:pointer; }
.nodet img {
  width:100%; aspect-ratio:4/3; object-fit:cover; border-radius:8px;
  display:block; background:var(--thumb-bg);
}
.nodet figcaption {
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.66rem;
  color:var(--muted); margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
/* lightbox */
.lb { position:fixed; inset:0; z-index:100; display:none; }
.lb.open { display:block; }
.lb-backdrop { position:absolute; inset:0; background:rgba(0,0,0,.84); }
.lb-wrap { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; padding:18px; }
.lb-inner { display:flex; flex-direction:column; align-items:center; max-width:94vw; }
.lb-img {
  max-width:100%; max-height:70vh; object-fit:contain; border-radius:8px;
  box-shadow:0 12px 44px rgba(0,0,0,.55); background:#000;
}
.lb-toggle {
  margin-top:18px; padding:6px 14px; font-size:.8rem; font-weight:600; cursor:pointer;
  color:#fff; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.35);
  border-radius:999px;
}
.lb-toggle:hover { background:rgba(255,255,255,.26); }
.lb-caption { margin-top:14px; color:#fff; text-align:center; max-width:92vw; padding:12px 16px; background:rgba(10,12,16,0.95); border:1px solid rgba(255,255,255,0.12); border-radius:12px; }
.lb-title { font-size:1.12rem; font-weight:700; }
.lb-sci { font-weight:400; font-style:italic; color:#cdd3dc; font-size:.92rem; }
.lb-badges { margin-top:6px; }
.lb-ev { font-size:.78rem; margin-top:5px; color:#fcd34d; }
.lb-line { font-size:.85rem; margin-top:5px; color:#e7eaee; }
.lb-line.muted { color:#a7aeba; font-size:.8rem; }
.lb-meta {
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.74rem;
  color:#9aa2ac; margin-top:8px;
}
.lb-close {
  position:absolute; top:12px; right:16px; z-index:101; font-size:1.9rem; line-height:1;
  color:#fff; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.3);
  border-radius:999px; width:40px; height:40px; cursor:pointer;
}
.lb-nav {
  position:absolute; top:50%; transform:translateY(-50%); z-index:101;
  font-size:2.4rem; line-height:1; color:#fff; background:rgba(255,255,255,.1);
  border:1px solid rgba(255,255,255,.28); border-radius:999px; width:48px; height:56px;
  cursor:pointer;
}
.lb-nav:hover, .lb-close:hover { background:rgba(255,255,255,.24); }
.lb-prev { left:12px; } .lb-next { right:12px; }
"""

JS = """
const byId = {};
Object.keys(DATA.sec).forEach(function (sec) {
  DATA.sec[sec].forEach(function (c, i) { byId[c.id] = { sec: sec, idx: i }; });
});
DATA.nodet.forEach(function (c, i) { byId[c.id] = { sec: 'nodet', idx: i }; });

const lb = document.getElementById('lb');
const lbImg = document.getElementById('lb-img');
const lbCap = document.getElementById('lb-caption');
const lbToggle = document.getElementById('lb-toggle');
let cur = null, showCrop = false, savedScroll = 0;

function esc(s) {
  return String(s).replace(/[&<>"']/g, function (m) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m];
  });
}
function cardAt(sec, idx) { return sec === 'nodet' ? DATA.nodet[idx] : DATA.sec[sec][idx]; }
function listLen(sec) { return sec === 'nodet' ? DATA.nodet.length : DATA.sec[sec].length; }

function openLb(sec, idx) {
  cur = { sec: sec, idx: idx }; showCrop = false; savedScroll = window.scrollY;
  renderLb();
  lb.classList.add('open'); lb.setAttribute('aria-hidden', 'false');
  document.body.style.overflow = 'hidden';
}
function closeLb() {
  lb.classList.remove('open'); lb.setAttribute('aria-hidden', 'true');
  document.body.style.overflow = '';
  window.scrollTo(0, savedScroll); cur = null;
}
function nav(d) {
  if (!cur) return;
  let n = cur.idx + d; const L = listLen(cur.sec);
  if (n < 0) n = L - 1;
  if (n >= L) n = 0;
  cur.idx = n; showCrop = false; renderLb();
}
function badgeHtml(b, c) {
  if (b === 'rare') {
    const r = DATA.region || {};
    const label = (r.short && r.month)
      ? 'Rare in ' + r.short + ' \u00b7 ' + r.month : 'Rare here';
    const tip = (c && c.rare_evidence) ? ' title="' + esc(c.rare_evidence) + '"' : '';
    return '<span class="badge rare"' + tip + '>' + esc(label) + '</span>';
  }
  if (b === 'lifer') return '<span class="badge lifer">Possible lifer</span>';
  if (b.indexOf('lookalike:') === 0)
    return '<span class="badge lookalike">Look-alike: ' + esc(b.slice(10)) + '</span>';
  return '';
}
function fmtTop3(list) {
  return list.map(function (r) { return esc(r[0]) + ' ' + r[2].toFixed(2); }).join(' &middot; ');
}
function renderLb() {
  const c = cardAt(cur.sec, cur.idx);
  lbImg.src = (showCrop && c.crop) ? c.crop : c.frame;
  const hasToggle = !!(c.crop && c.frame);
  lbToggle.style.display = hasToggle ? '' : 'none';
  lbToggle.textContent = showCrop ? 'Show full frame' : 'Show crop';
  let cap = '';
  if (c.species) {
    cap += '<div class="lb-title">' + esc(c.species);
    if (c.sci) cap += ' <span class="lb-sci">' + esc(c.sci) + '</span>';
    cap += '</div>';
    if (c.badges && c.badges.length)
      cap += '<div class="lb-badges">' + c.badges.map(function (b) { return badgeHtml(b, c); }).join('') + '</div>';
    if (c.rare_evidence)
      cap += '<div class="lb-ev">' + esc(c.rare_evidence) + '</div>';
    cap += '<div class="lb-line">Adjusted: ' + fmtTop3(c.adj_top3) + '</div>';
    cap += '<div class="lb-line muted">Raw: ' + fmtTop3(c.raw_top3) + '</div>';
  } else {
    cap += '<div class="lb-title">No bird detected</div>';
  }
  cap += '<div class="lb-meta">' + c.time + ' &middot; ' + c.n_frames +
    ' frame' + (c.n_frames === 1 ? '' : 's') + ' &middot; ' + esc(c.best_frame) +
    ' &middot; det ' + c.det_conf.toFixed(3) + '</div>';
  lbCap.innerHTML = cap;
}

document.addEventListener('click', function (e) {
  const ci = e.target.closest('.card img.crop');
  if (ci) { const card = ci.closest('.card'); openLb(card.dataset.sec, +card.dataset.idx); return; }
  const nd = e.target.closest('.nodet');
  if (nd) { openLb('nodet', +nd.dataset.idx); return; }
  if (e.target.id === 'lb-backdrop') closeLb();
});
document.getElementById('lb-close').addEventListener('click', closeLb);
document.getElementById('lb-prev').addEventListener('click', function () { nav(-1); });
document.getElementById('lb-next').addEventListener('click', function () { nav(1); });
document.getElementById('lb-toggle').addEventListener('click', function () {
  if (cur) { showCrop = !showCrop; renderLb(); }
});
document.addEventListener('keydown', function (e) {
  if (!lb.classList.contains('open')) return;
  if (e.key === 'Escape') closeLb();
  else if (e.key === 'ArrowLeft') nav(-1);
  else if (e.key === 'ArrowRight') nav(1);
});
(function () {
  const m = location.hash.match(/^#open=(.+)$/);
  if (m) { const t = byId[decodeURIComponent(m[1])]; if (t) openLb(t.sec, t.idx); }
})();
"""
