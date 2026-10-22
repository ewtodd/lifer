"""Several species in one burst: which detections are extra birds.

A burst (today's encounter) has one *primary* bird, whose species comes from
the sharpest subject crops. Every other detection in the burst is classified
on its own; the ones that clearly are a different species become *extra
birds* (their own reviewable encounter, enc_0011-2, ...). Torch-free so the
grouping rules can be tested without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Crops smaller than this (long side of the box, px) are too few pixels for
# BioCLIP to say anything about.
MIN_LONG_SIDE = 64

# A different-species group is kept when its best detection's adjusted
# probability reaches this ...
EXTRA_MIN_PROB = 0.5
# ... or this when the species is one of the primary's own top 3 (a
# lookalike of the primary needs more evidence than an unrelated species).
LOOKALIKE_MIN_PROB = 0.8
# ... and only when the group holds up on average and YOLO is fairly sure it
# is a bird at all. Reviewed on two trips (95 extras): what these drop is
# almost all junk (blurred necks, a leg, bark, a motion streak, foliage),
# where one lucky crop cleared EXTRA_MIN_PROB.
EXTRA_MIN_MEAN_PROB = 0.5
EXTRA_MIN_CONF = 0.45
MAX_EXTRA = 3
# raw/adj probabilities of an extra bird average over its best detections
AVG_DETS = 3


@dataclass
class BirdDet:
    """One classified detection of a burst."""
    frame: str
    box: tuple[float, float, float, float]  # full-frame pixels
    conf: float  # YOLO confidence (1.0 for an owner-added box)
    sharp: float  # sharpness of its crop; breaks ties between equal detections
    raw: dict[int, float]  # T-calibrated BioCLIP probabilities by taxon id
    adj: dict[int, float]  # ... after the regional seasonal prior
    subject: bool = False  # the box the frame's primary bird is cut from
    added: bool = False  # drawn by the owner: always its own bird

    @property
    def top1(self) -> int:
        return max(self.adj, key=self.adj.__getitem__)


@dataclass
class ExtraBird:
    species: int  # adjusted top-1 taxon id of the group
    dets: list[BirdDet] = field(default_factory=list)  # most probable first
    added: bool = False

    @property
    def best(self) -> BirdDet:
        return self.dets[0]


def _rank(d: BirdDet, species: int) -> tuple[float, float]:
    return (d.adj.get(species, 0.0), d.sharp)


def top_taxa(vec: dict[int, float], n: int = 3) -> list[int]:
    return sorted(vec, key=vec.__getitem__, reverse=True)[:n]


def group_extra_birds(dets: list[BirdDet], primary_adj: dict[int, float],
                      frame_order: list[str]) -> list[ExtraBird]:
    """The extra birds of a burst from its classified detections.

    Detections are grouped by adjusted top-1 species S. A group of S != the
    primary's species P is an extra bird when its best detection reaches 0.5
    (0.8 when S is in P's top 3), its best detections average 0.5 and one of
    them has YOLO conf 0.45, unless it is made only of subject
    detections and S is in P's top 3: that is the primary bird misread, not
    a second one. At most MAX_EXTRA such groups, most probable first.
    Owner-added detections skip all of this: they always form a bird, one
    per species, outside the cap.
    """
    p_top = top_taxa(primary_adj)
    primary = p_top[0] if p_top else None
    auto: dict[int, list[BirdDet]] = {}
    manual: dict[int, list[BirdDet]] = {}
    for d in dets:
        s = d.top1
        if d.added:
            manual.setdefault(s, []).append(d)
        elif s != primary:
            auto.setdefault(s, []).append(d)

    kept: list[tuple[float, int, list[BirdDet]]] = []
    for s, ds in auto.items():
        lookalike = s in p_top
        best = max(d.adj[s] for d in ds)
        if best < (LOOKALIKE_MIN_PROB if lookalike else EXTRA_MIN_PROB):
            continue
        if lookalike and all(d.subject for d in ds):
            continue
        top = sorted(ds, key=lambda d, s=s: _rank(d, s), reverse=True)[:AVG_DETS]
        if (sum(d.adj[s] for d in top) / len(top) < EXTRA_MIN_MEAN_PROB
                or max(d.conf for d in ds) < EXTRA_MIN_CONF):
            continue
        kept.append((sum(d.adj[s] for d in ds), s, ds))
    kept.sort(key=lambda t: t[0], reverse=True)

    birds = [ExtraBird(s, sorted(ds, key=lambda d, s=s: _rank(d, s), reverse=True))
             for _w, s, ds in kept[:MAX_EXTRA]]
    birds += [ExtraBird(s, sorted(ds, key=lambda d, s=s: _rank(d, s), reverse=True),
                        added=True)
              for s, ds in manual.items()]
    # numbering follows the photographer's order, not the model's confidence
    order = {f: i for i, f in enumerate(frame_order)}
    birds.sort(key=lambda b: (min(order.get(d.frame, len(order)) for d in b.dets),
                              b.species))
    return birds


def bird_frames(bird: ExtraBird, frame_order: list[str]) -> list[str]:
    """The frames holding one of the bird's detections, in burst order."""
    have = {d.frame for d in bird.dets}
    return [f for f in frame_order if f in have]


def bird_boxes(bird: ExtraBird) -> dict[str, list[float]]:
    """frame -> the bird's box there (its most probable detection in it)."""
    out: dict[str, list[float]] = {}
    for d in bird.dets:  # most probable first
        out.setdefault(d.frame, [round(v, 1) for v in d.box])
    return out


def average_probs(bird: ExtraBird, attr: str) -> dict[int, float]:
    """Mean of `raw` or `adj` over the bird's best AVG_DETS detections."""
    ds = bird.dets[:AVG_DETS]
    keys = {t for d in ds for t in getattr(d, attr)}
    return {t: sum(getattr(d, attr).get(t, 0.0) for d in ds) / len(ds)
            for t in keys}


def moved_frames(birds: list[ExtraBird]) -> set[str]:
    """Frames whose *subject* detection belongs to an extra bird (the
    photographer re-aimed): they leave the primary bird's frame list."""
    return {d.frame for b in birds for d in b.dets if d.subject}


# ---------------------------------------------------------------------------
# flags shared by the pipeline, extra birds and the review app's re-identify
# ---------------------------------------------------------------------------

# A lifelist species in the top 3 whose adjusted probability is at least this
# fraction of the top-1's counts as "close behind": the encounter is then a
# lookalike of a known bird, not a possible lifer. 0.1 sits in the natural
# gap of the Dec 29 data between close second places (0.13-0.43, e.g. the
# plover and scaup pairs) and distant ones (<0.11); see REPORT.md Task 02.
LOOKALIKE_FRACTION = 0.1

# "Rare here": the model is confident in a species (raw, T-calibrated top-1
# above RARE_CONF) AND that species is uncommon at this place in the photo's
# month window (share of the seasonal counts below RARE_SHARE, or absent).
# RARE_CONF: at T=3.0 a top-1 of 0.60 means the model clearly leads one species
# (not a coin-flip) -- it catches the weakest AMGP plover (0.651) with margin.
# RARE_SHARE: 0.2% of the regional month-window observations. AMGP sits at
# ~0.17% (below), its common look-alike Black-bellied Plover at ~0.89% (above),
# so the prior's favorite stays unflagged while the true vagrant is lifted.
RARE_CONF = 0.60
RARE_SHARE = 0.002


def species_flags(p_clip: dict[int, float], p_adj: dict[int, float], classes: dict,
                  lifers: set, seasonal_counts: dict, alltime_counts: dict,
                  region_name: str, window_label: str) -> dict:
    """Top-3s and lifer / lookalike / rare flags of one bird, from its raw
    (T-calibrated) and prior-adjusted probabilities. One function for the
    primary bird, extra birds and re-identify, so they always agree."""
    def top3(dist: dict[int, float]) -> list[tuple[str, str, float]]:
        top = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:3]
        return [(classes[t].common, classes[t].scientific, p) for t, p in top]

    seasonal_total = sum(seasonal_counts.values()) or 1.0
    alltime_total = sum(alltime_counts.values()) or 1.0
    adj_sorted = sorted(p_adj.items(), key=lambda kv: kv[1], reverse=True)
    adj_top1, adj_top1_p = adj_sorted[0]

    # "Rare here": a confident raw top-1 on a species that is uncommon at
    # this place in the photo's month window. Computed on the RAW
    # (T-calibrated, prior-free) probabilities so the regional prior can't
    # mask a true vagrant; beats lifer_lookalike.
    raw_top1_tid, raw_top1_p = max(p_clip.items(), key=lambda kv: kv[1])
    s_share = seasonal_counts.get(raw_top1_tid, 0.0) / seasonal_total
    rare_here = raw_top1_p >= RARE_CONF and s_share < RARE_SHARE
    # evidence behind the rare flag: the species' share of the place's
    # month-window records and of its all-time records
    rare_evidence = ""
    if rare_here:
        a_share = alltime_counts.get(raw_top1_tid, 0.0) / alltime_total
        rare_evidence = (
            f"{s_share * 100:.2f}% of {window_label} iNaturalist records "
            f"in {region_name} (all-time {a_share * 100:.1f}%)")

    possible_lifer = False
    lookalike = None
    if adj_top1 not in lifers:
        # Closest lifelist species in the top 3; if it's within
        # LOOKALIKE_FRACTION of the top-1's probability, flag it as a
        # lookalike of a known bird instead of a possible lifer.
        close = max(((t, p) for t, p in adj_sorted[:3] if t in lifers),
                    key=lambda tp: tp[1], default=None)
        if close is not None and close[1] >= LOOKALIKE_FRACTION * adj_top1_p:
            lookalike = classes[close[0]].common
        else:
            possible_lifer = True
    return {
        "raw_top3": top3(p_clip),
        "adj_top3": top3(p_adj),
        "possible_lifer": possible_lifer,
        "lifer_lookalike": lookalike,
        "rare_here": rare_here,
        "rare_evidence": rare_evidence,
    }
