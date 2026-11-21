#!/usr/bin/env python3
"""Render VELUNARA poster -> entry.png (1024x1024).

Gouache travel poster: a stone wave spiralling open over a warm lagoon.
Pillow + numpy, deterministic (fixed seed, no time, no network).
Renders at 1536x1536, downsamples Lanczos to 1024.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SEED = 20261121
S = 1.5  # render scale: 1024 -> 1536
W = H = 1024
RW = RH = 1536

# ---- palette -------------------------------------------------------------
CREAM = (0xF4, 0xE6, 0xC8)
INK = (0x17, 0x3D, 0x49)
SKY = (0xA6, 0xCD, 0xD0)
TEAL = (0x45, 0xB7, 0xAD)
DTEAL = (0x17, 0x6E, 0x78)
APRICOT = (0xD9, 0x91, 0x69)
LIGHT = (0xFF, 0xD9, 0x9A)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entry.png")


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def sc(p):
    """1024-space point -> render-space point."""
    return (p[0] * S, p[1] * S)


# ---- curve helpers -------------------------------------------------------
def catmull_rom(pts, samples_per_seg=24):
    """Sample a Catmull-Rom spline through pts (list of (x, y))."""
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    out = []
    for i in range(n - 1):
        p0 = pts[max(i - 1, 0)]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[min(i + 2, n - 1)]
        t = np.linspace(0, 1, samples_per_seg, endpoint=False)[:, None]
        seg = 0.5 * ((2 * p1) + (-p0 + p2) * t
                     + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                     + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
        out.append(seg)
    out.append(pts[-1:].copy())
    return np.concatenate(out, axis=0)


def widths_along(center, ctrl_pts, ctrl_widths):
    """Interpolate ribbon half-width along the sampled centerline."""
    ctrl = np.asarray(ctrl_pts, dtype=float)
    # arc-length parameter of control points on the sampled curve
    seg_len = np.linalg.norm(np.diff(center, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    # find parameter value of each control point: nearest sample index
    params = []
    for cp in ctrl:
        d = np.linalg.norm(center - cp, axis=1)
        params.append(cum[int(np.argmin(d))])
    params = np.asarray(params)
    t = cum / total
    pt = params / total
    w = np.interp(t, pt, np.asarray(ctrl_widths, dtype=float))
    return w / 2.0  # half-width


def ribbon_polygon(center, half_w, cavity):
    """Return (outer_pts, inner_pts) for a ribbon around centerline."""
    tangent = np.gradient(center, axis=0)
    tl = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(tl, 1e-9)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)  # one side
    # inner side = side closer to cavity centre
    cand_a = center + normal * half_w[:, None]
    cand_b = center - normal * half_w[:, None]
    da = np.linalg.norm(cand_a - np.asarray(cavity), axis=1)
    db = np.linalg.norm(cand_b - np.asarray(cavity), axis=1)
    inner_side = np.where((da < db)[:, None], normal, -normal)
    inner = center + inner_side * half_w[:, None]
    outer = center - inner_side * half_w[:, None]
    return outer, inner


# ---- main render ---------------------------------------------------------
def render():
    rng = np.random.default_rng(SEED)
    img = Image.new("RGB", (RW, RH), CREAM)
    d = ImageDraw.Draw(img, "RGBA")
    inset = (int(28 * S), int(28 * S), int(996 * S), int(996 * S))

    # ---- sky: vertical gradient cream -> misty blue, down to horizon ----
    horizon_y = int(650 * S)
    top = np.array(CREAM, dtype=float)
    bot = np.array(SKY, dtype=float)
    ys = np.linspace(0, 1, horizon_y)[:, None, None]
    grad = (top[None, None, :] * (1 - ys) + bot[None, None, :] * ys)
    sky_arr = np.repeat(grad, RW, axis=1)
    sky = Image.fromarray(sky_arr.astype(np.uint8), "RGB")
    img.paste(sky, (0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    # ---- sun glow, top right -------------------------------------------
    glow = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy = int(830 * S), int(110 * S)
    gd.ellipse([cx - 260 * S, cy - 180 * S, cx + 260 * S, cy + 180 * S],
               fill=LIGHT + (70,))
    glow = glow.filter(ImageFilter.GaussianBlur(int(60 * S)))
    img.paste(Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB"), (0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    # ---- lagoon: gradient light teal -> dark teal -----------------------
    lag_top = lerp(TEAL, CREAM, 0.35)
    lag_bot = DTEAL
    lh = RH - horizon_y
    ys = np.linspace(0, 1, lh)[:, None, None]
    t = np.array(lag_top, dtype=float)
    b = np.array(lag_bot, dtype=float)
    grad = (t[None, None, :] * (1 - ys) + b[None, None, :] * ys)
    lag_arr = np.repeat(grad, RW, axis=1)
    lag = Image.fromarray(lag_arr.astype(np.uint8), "RGB")
    img.paste(lag, (0, horizon_y))

    # ---- soften the horizon: blur a thin strip so sky meets water -------
    # horizon stays at y=650; the blend band is only ~4 px wide, no hard ink line
    hy0 = int((650 - 8) * S)
    hy1 = int((650 + 8) * S)
    strip = img.crop((0, hy0, RW, hy1)).filter(ImageFilter.GaussianBlur(int(2 * S)))
    img.paste(strip, (0, hy0))

    # ---- light patch on the water under the cavity ----------------------
    patch = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    pd = ImageDraw.Draw(patch)
    pcx, pcy = int(560 * S), int(790 * S)
    pd.ellipse([pcx - 205 * S, pcy - 95 * S, pcx + 205 * S, pcy + 95 * S],
               fill=LIGHT + (115,))
    patch = patch.filter(ImageFilter.GaussianBlur(int(45 * S)))
    img.paste(Image.alpha_composite(img.convert("RGBA"), patch).convert("RGB"), (0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    # ---- spiral rock: centreline, widths, ribbon -----------------------
    cavity = (560.0, 480.0)
    ctrl = [(315, 800), (288, 660), (300, 480), (360, 370), (480, 310),
            (620, 295), (745, 330), (830, 410), (855, 500), (820, 585),
            (740, 640), (660, 630), (600, 590), (585, 530), (620, 490)]
    ctrl_w = [140, 115, 100, 95, 90, 88, 85, 85, 80, 75, 70, 65, 55, 45, 38]
    center = catmull_rom(ctrl, 24)
    half_w = widths_along(center, ctrl, ctrl_w)
    outer, inner = ribbon_polygon(center, half_w, cavity)

    rock = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rock)
    poly = [tuple(p) for p in outer] + [tuple(p) for p in inner[::-1]]
    poly = [(x * S, y * S) for (x, y) in poly]
    rd.polygon(poly, fill=APRICOT + (255,))
    # rounded cap at the base so the foot dips into the water, not sliced
    cap_c = (float(center[0][0] * S), float(center[0][1] * S))
    cap_r = float(half_w[0] * S)
    rd.ellipse([cap_c[0] - cap_r, cap_c[1] - cap_r,
                cap_c[0] + cap_r, cap_c[1] + cap_r], fill=APRICOT + (255,))

    # strata: curved lines following the band, clipped to the ribbon
    strata = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    sd = ImageDraw.Draw(strata)
    mask = Image.new("L", (RW, RH), 0)
    md = ImageDraw.Draw(mask)
    md.polygon(poly, fill=255)
    md.ellipse([cap_c[0] - cap_r, cap_c[1] - cap_r,
                cap_c[0] + cap_r, cap_c[1] + cap_r], fill=255)
    dark_apricot = lerp(APRICOT, INK, 0.35)
    for frac in (-0.45, -0.1, 0.25):
        line = center + (outer - center) * 0 + (inner - center) * (-frac)
        # offset from centre toward inner side by frac of half-width
        pts = [(float(x * S), float(y * S)) for (x, y) in line[:-6]]
        sd.line(pts, fill=dark_apricot + (70,), width=max(2, int(5 * S)), joint="curve")
    strata.putalpha(Image.composite(strata.split()[3], Image.new("L", (RW, RH), 0), mask))
    rock = Image.alpha_composite(rock, strata)

    # shaded side: band hugging the inner edge (cavity side)
    shade = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    shd = ImageDraw.Draw(shade)
    inner_rs = [(float(x * S), float(y * S)) for (x, y) in inner]
    shd.line(inner_rs[:-6], fill=DTEAL + (110,), width=int(34 * S), joint="curve")
    shade = shade.filter(ImageFilter.GaussianBlur(int(7 * S)))
    # clip shade to ribbon
    shade.putalpha(Image.composite(shade.split()[3], Image.new("L", (RW, RH), 0), mask))
    rock = Image.alpha_composite(rock, shade)

    # warm light wash over the top of the band
    wash = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    wd = ImageDraw.Draw(wash)
    outer_rs = [(float(x * S), float(y * S)) for (x, y) in outer]
    wd.line(outer_rs[20:200], fill=LIGHT + (60,), width=int(40 * S), joint="curve")
    wash = wash.filter(ImageFilter.GaussianBlur(int(10 * S)))
    wash.putalpha(Image.composite(wash.split()[3], Image.new("L", (RW, RH), 0), mask))
    rock = Image.alpha_composite(rock, wash)

    # sunlit lip: bright line along the outer top edge
    lip = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lip)
    ld.line(outer_rs[20:210], fill=LIGHT + (210,), width=int(11 * S), joint="curve")
    lip = lip.filter(ImageFilter.GaussianBlur(int(3 * S)))
    lip.putalpha(Image.composite(lip.split()[3], Image.new("L", (RW, RH), 0), mask))
    rock = Image.alpha_composite(rock, lip)

    # repair the faint dark seam crossing the spiral tip: apricot patch with
    # gouache dabs, clipped to the ribbon so the sky gap stays clear
    repair = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    rpd = ImageDraw.Draw(repair)
    rcx, rcy = 582 * S, 558 * S
    rpd.ellipse([rcx - 42 * S, rcy - 24 * S, rcx + 42 * S, rcy + 24 * S],
                fill=APRICOT + (255,))
    # gouache dabs on their own layer: translucent ink drawn over the opaque
    # patch on one layer would replace it (PIL ImageDraw), not blend, and
    # the dark seam would show through the dab spots
    rdabs = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    rdd = ImageDraw.Draw(rdabs)
    rdd.ellipse([555 * S, 546 * S, 595 * S, 562 * S],
                fill=lerp(APRICOT, LIGHT, 0.25) + (110,))
    rdd.ellipse([580 * S, 562 * S, 610 * S, 574 * S],
                fill=lerp(APRICOT, INK, 0.15) + (80,))
    rdd.ellipse([548 * S, 560 * S, 572 * S, 572 * S],
                fill=lerp(APRICOT, LIGHT, 0.15) + (70,))
    repair = Image.alpha_composite(repair, rdabs)
    repair = repair.filter(ImageFilter.GaussianBlur(int(2 * S)))
    repair.putalpha(Image.composite(repair.split()[3], Image.new("L", (RW, RH), 0), mask))
    rock = Image.alpha_composite(rock, repair)

    # ---- light beam through the cavity onto the water -------------------
    beam = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    bd = ImageDraw.Draw(beam)
    bd.polygon([sc((500, 470)), sc((640, 460)), sc((720, 880)), sc((430, 880))],
               fill=LIGHT + (34,))
    beam = beam.filter(ImageFilter.GaussianBlur(int(18 * S)))
    img.paste(Image.alpha_composite(img.convert("RGBA"), beam).convert("RGB"), (0, 0))

    # ---- beach: thin sand curve to the foot of the rock ------------------
    # drawn BEFORE the rock so the rock foot covers its end (sand meets stone)
    beach = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    bch = ImageDraw.Draw(beach)
    sand = lerp(CREAM, APRICOT, 0.35)
    # thin sand crescent: starts hidden behind the foreground rocks, widens
    # gradually toward the foot of the stone wave (drawn before the rock, so
    # its end tucks under the foot)
    crescent = [(180, 965, 8), (222, 948, 12), (258, 924, 18),
                (285, 895, 24), (302, 868, 28)]
    for (x1, y1, _), (x2, y2, w2) in zip(crescent, crescent[1:]):
        bch.line([sc((x1, y1)), sc((x2, y2))], fill=sand + (255,),
                 width=int(w2 * S), joint="curve")
    pts = [sc((x, y)) for (x, y, _) in crescent]
    # pale highlight on its own layer: translucent ink drawn over the opaque
    # sand on one layer would replace it (PIL ImageDraw), not blend, and
    # the water would show through as a dark stripe down the beach
    bhi = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    bhd = ImageDraw.Draw(bhi)
    bhd.line([(x + 6 * S, y - 2 * S) for (x, y) in pts], fill=CREAM + (45,),
             width=int(4 * S), joint="curve")
    beach = Image.alpha_composite(beach, bhi)
    beach = beach.filter(ImageFilter.GaussianBlur(int(2 * S)))
    img.paste(Image.alpha_composite(img.convert("RGBA"), beach).convert("RGB"), (0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    # ---- composite rock --------------------------------------------------
    img.paste(Image.alpha_composite(img.convert("RGBA"), rock).convert("RGB"), (0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    # ---- shadow under the lip, inside the cavity ------------------------
    cav = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    cd = ImageDraw.Draw(cav)
    # stop the shadow before the tight tip curl: there it crossed the stone
    # and read as a dark seam; the ribbon's own shade keeps the curl's depth
    cd.line(inner_rs[:-25], fill=DTEAL + (60,), width=int(36 * S), joint="curve")
    cav = cav.filter(ImageFilter.GaussianBlur(int(12 * S)))
    img.paste(Image.alpha_composite(img.convert("RGBA"), cav).convert("RGB"), (0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    # ---- mist at the foot of the rock ------------------------------------
    mist = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    md2 = ImageDraw.Draw(mist)
    md2.ellipse([int(300 * S), int(680 * S), int(760 * S), int(760 * S)],
                fill=SKY + (90,))
    mist = mist.filter(ImageFilter.GaussianBlur(int(28 * S)))
    img.paste(Image.alpha_composite(img.convert("RGBA"), mist).convert("RGB"), (0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    # ---- foreground rocks, bottom left -----------------------------------
    fg = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fg)
    dark_rock = lerp(INK, DTEAL, 0.25)
    fd.polygon([sc(p) for p in [(28, 996), (28, 905), (70, 865), (120, 850),
                                (170, 875), (210, 920), (235, 950), (235, 996)]],
               fill=dark_rock + (255,))
    fd.polygon([sc(p) for p in [(120, 996), (150, 945), (200, 960), (235, 996)]],
               fill=lerp(INK, DTEAL, 0.45) + (255,))
    fd.line([sc((70, 865)), sc((120, 850)), sc((170, 875))],
            fill=lerp(TEAL, CREAM, 0.3) + (160,), width=int(4 * S), joint="curve")
    img.paste(Image.alpha_composite(img.convert("RGBA"), fg).convert("RGB"), (0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    # ---- water reflections: broken horizontal fragments ------------------
    refl = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    rfd = ImageDraw.Draw(refl)
    y = 665
    i = 0
    while y < 985:
        gap = 14 + i * 6
        wdt = int(rng.uniform(90, 260) * S)
        cxr = int(rng.uniform(430, 700) * S)
        col = APRICOT if i % 3 else DTEAL
        alpha = int(rng.uniform(50, 110))
        rfd.rounded_rectangle([cxr - wdt // 2, int(y * S), cxr + wdt // 2,
                               int(y * S) + int(5 * S)],
                              radius=2, fill=col + (alpha,))
        y += gap
        i += 1
    # invitation strokes: broken horizontal light strokes, longest and
    # brightest at the patch centre (560, 790), shorter and dimmer away
    for dy in (-96, -64, -32, 0, 32, 64, 96):
        wdt = int((300 - abs(dy) * 1.7) * S)
        alp = int(135 - abs(dy) * 0.75)
        yy = int((790 + dy) * S)
        gap = int(18 * S)
        cxr = int(560 * S)
        rfd.rounded_rectangle([cxr - wdt // 2, yy, cxr - gap, yy + int(5 * S)],
                              radius=2, fill=LIGHT + (alp,))
        rfd.rounded_rectangle([cxr + gap, yy, cxr + wdt // 2, yy + int(5 * S)],
                              radius=2, fill=LIGHT + (alp,))
    # short wave strokes breaking the reflections
    for _ in range(46):
        wx = rng.uniform(300, 960)
        wy = rng.uniform(660, 990)
        wl = rng.uniform(18, 55)
        rfd.line([(wx * S, wy * S), ((wx + wl) * S, wy * S)],
                 fill=lerp(TEAL, CREAM, 0.55) + (int(rng.uniform(40, 90)),),
                 width=int(3 * S))
    refl = refl.filter(ImageFilter.GaussianBlur(int(1.5 * S)))
    img.paste(Image.alpha_composite(img.convert("RGBA"), refl).convert("RGB"), (0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    # ---- plant the foot: soft ink contact shadow + sand mound -----------
    # drawn last (after reflections) so no water shimmer stains the sand;
    # two overlapping ellipses make an irregular sandbar, not a flat plate
    ground = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    gd2 = ImageDraw.Draw(ground)
    gd2.ellipse([int((316 - 92) * S), int((894 - 12) * S),
                 int((316 + 92) * S), int((894 + 12) * S)], fill=INK + (90,))
    ground = ground.filter(ImageFilter.GaussianBlur(int(6 * S)))
    img.paste(Image.alpha_composite(img.convert("RGBA"), ground).convert("RGB"), (0, 0))
    mound = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    md3 = ImageDraw.Draw(mound)
    sand = lerp(CREAM, APRICOT, 0.35)
    md3.ellipse([int((315 - 72) * S), int((856 - 26) * S),
                 int((315 + 72) * S), int((856 + 26) * S)], fill=sand + (255,))
    md3.ellipse([int((298 - 48) * S), int((862 - 20) * S),
                 int((298 + 48) * S), int((862 + 20) * S)], fill=sand + (255,))
    # tonal wash on its own layer: translucent ink drawn over the opaque
    # sand on one layer would replace it (PIL ImageDraw), not blend, and
    # the foot would show through the wash as a pale ring
    mwash = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    mwd = ImageDraw.Draw(mwash)
    mwd.ellipse([int((315 - 55) * S), int((866 - 12) * S),
                 int((315 + 55) * S), int((866 + 12) * S)],
                fill=lerp(sand, APRICOT, 0.25) + (55,))
    mound = Image.alpha_composite(mound, mwash)
    mound = mound.filter(ImageFilter.GaussianBlur(int(5 * S)))
    img.paste(Image.alpha_composite(img.convert("RGBA"), mound).convert("RGB"), (0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    # ---- cream frame ------------------------------------------------------
    d.rectangle([0, 0, RW - 1, int(28 * S) - 1], fill=CREAM)
    d.rectangle([0, RH - int(28 * S), RW - 1, RH - 1], fill=CREAM)
    d.rectangle([0, 0, int(28 * S) - 1, RH - 1], fill=CREAM)
    d.rectangle([RW - int(28 * S), 0, RW - 1, RH - 1], fill=CREAM)

    # ---- name: VELUNARA, tracked, via its own mask -----------------------
    name = "VELUNARA"
    target_w = 850 * S
    tracking = 5 * S
    lo, hi = 60, 400
    for _ in range(24):
        mid = (lo + hi) / 2
        f = ImageFont.truetype(FONT_PATH, int(mid))
        wsum = sum(f.getlength(ch) for ch in name) + tracking * (len(name) - 1)
        if wsum < target_w:
            lo = mid
        else:
            hi = mid
    font = ImageFont.truetype(FONT_PATH, int(lo))
    widths = [font.getlength(ch) for ch in name]
    total = sum(widths) + tracking * (len(name) - 1)
    nmask = Image.new("L", (RW, RH), 0)
    nd = ImageDraw.Draw(nmask)
    x = (RW - total) / 2
    ymid = (78 + 182) / 2 * S
    for ch, wch in zip(name, widths):
        nd.text((x, ymid), ch, font=font, fill=255, anchor="lm")
        x += wch + tracking
    img.paste(INK, (0, 0), nmask)

    # ---- downsample, grain, vignette --------------------------------------
    img = img.resize((W, H), Image.LANCZOS)
    arr = np.asarray(img).astype(np.int16)
    # fine monochrome grain
    grain = rng.normal(0, 1.2, (H, W, 1))
    arr = arr + grain
    # broad tonal variation (low-res noise upscaled)
    low = rng.normal(0, 3, (16, 16, 1))
    low_img = Image.fromarray(((low[:, :, 0] - low.min()) / (low.ptp() + 1e-9) * 255)
                              .astype(np.uint8), "L").resize((W, H), Image.BICUBIC)
    low_arr = np.asarray(low_img).astype(np.int16)[..., None] - 128
    arr = arr + (low_arr * 3 // 128)
    # vignette, max ~5%
    yy, xx = np.mgrid[0:H, 0:W]
    dx = (xx - W / 2) / (W / 2)
    dy = (yy - H / 2) / (H / 2)
    r = np.sqrt(dx * dx + dy * dy)
    vig = 1.0 - 0.05 * np.clip(r - 0.55, 0, None) ** 2 / (0.45 ** 2)
    arr = arr * vig[..., None]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    # indexed 256-colour adaptive palette + Floyd-Steinberg dithering:
    # ~380 KB (vs ~1.03 MB RGB), no visible banding, big tie-break advantage.
    img = img.quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                       dither=Image.Dither.FLOYDSTEINBERG)
    img.save(OUT, optimize=True)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    render()
