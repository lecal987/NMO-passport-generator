
# -*- coding: utf-8 -*-
"""
Refactored main.py (no translation, big fonts)
- Translation removed; skip texts containing Chinese
- Centralized constants; 10x font sizes
- I/O and dependencies unchanged
- Parallel per-record processing + vectorized color ops
- HSL "Color" blend approx + linear-space Soft-Light tweak
"""

# ========== Standard Library ==========
import os, time, random, hashlib, threading, math, pathlib, re
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== Third-Party (unchanged) ==========
import numpy as np
import pandas as pd
import requests  # kept to avoid changing dependency surface
from PIL import Image, ImageDraw, ImageFont
import colorsys

# =========================================================
#                 CONFIGURABLE CONSTANTS
# =========================================================

# --- Paths (keep output & filenames unchanged) ---
BASE_IMAGE              = "base.png"
BASE_IMAGE_TRANSPARENT  = "base0.png"
OVERLAY_IMAGE           = "overlay.png"
OUTPUT_DIR              = "output"

# --- Excel expected columns (unchanged) ---
REQUIRED_COLS = ['RGB', 'ID：', '称号：', '想说的话：', '愿望：']

# --- Threading ---
MAX_WORKERS = max(1, min(8, (os.cpu_count() or 4)))

# --- Text Defaults ---
DEFAULT_WISH_FALLBACK = "NMO Pass Certification\npresented by \ntomato and olozhika472"
WATERMARK_TEXT        = "Let's play a lifelong club"
UID_MAX_LENGTH_PIXELS = 1635  # hard limit for UID text width

# --- Text/Font Layout Map (10x sizes) ---
TEXT_CONFIG = {
    'uid': {
        'center':   (0.985, 0.682),
        'align':    'R',
        'size':     200,
        'max_width': 10000,
        'fonts':    ['AdobeGothicStd-Bold.otf','arialbd.ttf','segoeuib.ttf','segoeui.ttf'],
        'opacity':  255,
        'shadow':   3
    },
    'title': {
        'center':   (0.985, 0.715),
        'align':    'R',
        'size':     65,
        'max_width': 800,
        'fonts':    ['segoeui.ttf','segoeuise.ttf','arial.ttf'],
        'opacity':  230,
        'shadow':   2
    },
    'message': {
        'center':   (0.53, 0.84),
        'align':    'M',
        'size':     60,
        'max_width': 100,
        'fonts':    ['segoeui.ttf','segoeuise.ttf','arial.ttf'],
        'opacity':  200,
        'shadow':   0
    },
    'wish': {
        'center':   (0.16, 0.955),
        'align':    'L',
        'size':     45,
        'max_width': 500,
        'fonts':    ['ROCC____.TTF','segoeui.ttf','segoeuise.ttf','arial.ttf'],
        'opacity':  180,
        'shadow':   0
    },
    'watermark': {
        'center':   (0.15, 0.965),
        'align':    'L',
        'size':     55,
        'max_width': 10000,
        'fonts':    ['ROCKB.TTF','segoeui.ttf','segoeuise.ttf','arial.ttf'],
        'opacity':  100,
        'shadow':   0
    }
}

# --- Font directory (Windows default; falls back to PIL default if missing) ---
SYSTEM_FONTS_DIR = r"C:\Windows\Fonts"

# --- Color Algorithm Toggles ---
ENABLE_SOFT_LIGHT = True  # keep True for smoother highlights/shadows

# =========================================================
#                   Utility: Fonts & Text
# =========================================================

_font_cache = {}

def load_system_font(filenames, size):
    """Try each filename under SYSTEM_FONTS_DIR, cache by (tuple(names), size)."""
    key = (tuple(filenames), int(size))
    if key in _font_cache:
        return _font_cache[key]
    for fname in filenames:
        path = os.path.join(SYSTEM_FONTS_DIR, fname)
        if os.path.isfile(path):
            try:
                f = ImageFont.truetype(path, size=size)
                _font_cache[key] = f
                return f
            except Exception:
                continue
    # Fallback
    f = ImageFont.load_default()
    _font_cache[key] = f
    return f

def draw_centered_text(draw, font, text, cx, cy, opacity):
    """Center text at (cx, cy) using textbbox; supports RGBA alpha fill."""
    bbox = draw.textbbox((0,0), text, font=font)
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    x = cx - w//2; y = cy - h//2
    draw.text((x, y), text, font=font, fill=(255,255,255,opacity))

def wrap_text_by_width(draw, font, text, max_width):
    """Greedy wrap by pixel width; respects explicit newlines; avoids multiline textlength error."""
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for raw in text.split("\n"):          # handle explicit newlines first
        current = ""
        for ch in raw:
            trial = current + ch
            # Pillow textlength only accepts single-line strings
            w = draw.textlength(trial, font=font)
            if w <= max_width or not current:
                current = trial
            else:
                lines.append(current.rstrip())
                current = ch
        if current:
            lines.append(current.rstrip())
        # preserve explicit empty line
        if raw == "" and (not lines or lines[-1] != ""):
            lines.append("")
    return lines

def draw_text_with_wrap_and_left_bottom_align(draw, font, text, x_left, y_bottom, max_width, opacity, line_spacing=1.0):
    """Left aligned multi-line text; (x_left, y_bottom) is bottom-left anchor."""
    if not text:
        return
    lines = wrap_text_by_width(draw, font, text, max_width)
    if not lines:
        return
    heights = []
    for ln in lines:
        bbox = draw.textbbox((0,0), ln, font=font)
        heights.append(bbox[3] - bbox[1])
    line_h = max(heights) if heights else font.size
    total_h = int(line_h * (len(lines) + (len(lines)-1)*(line_spacing-1))) if lines else 0
    y_start = y_bottom - total_h
    y = y_start
    for ln in lines:
        draw.text((x_left, y), ln, font=font, fill=(255,255,255,opacity))
        y += int(line_h * line_spacing)

def contains_chinese(s: str) -> bool:
    """Return True if string contains CJK Unified Ideographs (basic range)."""
    if s is None:
        return False
    return re.search(r'[\u4e00-\u9fff]', str(s)) is not None

# =========================================================
#                   Color Space Utilities
# =========================================================

def _srgb_to_linear(x):
    """x ∈ [0,1] ndarray -> linear RGB"""
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def _linear_to_srgb(x):
    """linear RGB -> sRGB in [0,1]"""
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * (np.clip(x,0,1) ** (1/2.4)) - a)

def _hsl_to_rgb_vec(h, s, l):
    """
    Vectorized HSL→RGB (h,s,l in [0,1]), per CSS Color algorithm.
    """
    c = (1 - np.abs(2 * l - 1)) * s
    hp = (h * 6.0) % 6.0
    x = c * (1 - np.abs((hp % 2) - 1))

    z = np.zeros_like(h)
    r1 = np.where((0 <= hp) & (hp < 1), c,
         np.where((1 <= hp) & (hp < 2), x,
         np.where((2 <= hp) & (hp < 3), z,
         np.where((3 <= hp) & (hp < 4), z,
         np.where((4 <= hp) & (hp < 5), x,
         np.where((5 <= hp) & (hp < 6), c, z))))))
    g1 = np.where((0 <= hp) & (hp < 1), x,
         np.where((1 <= hp) & (hp < 2), c,
         np.where((2 <= hp) & (hp < 3), c,
         np.where((3 <= hp) & (hp < 4), x,
         np.where((4 <= hp) & (hp < 5), z,
         np.where((5 <= hp) & (hp < 6), z, z))))))
    b1 = np.where((0 <= hp) & (hp < 1), z,
         np.where((1 <= hp) & (hp < 2), z,
         np.where((2 <= hp) & (hp < 3), x,
         np.where((3 <= hp) & (hp < 4), c,
         np.where((4 <= hp) & (hp < 5), c,
         np.where((5 <= hp) & (hp < 6), x, z))))))

    m = l - c / 2.0
    r = r1 + m
    g = g1 + m
    b = b1 + m
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 1)

def _soft_light_blend_linear(base_lin, blend_lin):
    """Linear-space Soft-Light, vectorized; inputs in [0,1]."""
    b = np.clip(base_lin, 0, 1)
    s = np.clip(blend_lin, 0, 1)
    return np.where(
        s <= 0.5,
        b - (1 - 2 * s) * b * (1 - b),
        b + (2 * s - 1) * (np.sqrt(np.clip(b,0,1)) - b)
    )

# =========================================================
#                Tinting (keep names/signatures)
# =========================================================

def gray_to_hsv_tint0(arr, theme_rgb):
    """
    HSL 'Color' approximation: Hue/Sat from theme, Lightness from grayscale.
    """
    gray = arr[..., 0].astype(np.float32) / 255.0  # use R as gray (source is gray RGBA)
    alpha = arr[..., 3]
    r, g, b = [c / 255.0 for c in theme_rgb]
    # colorsys uses HLS order; (h, l, s)
    h, l_, s = colorsys.rgb_to_hls(r, g, b)
    H = np.full_like(gray, h, dtype=np.float32)
    S = np.full_like(gray, s, dtype=np.float32)
    L = gray
    rgb = _hsl_to_rgb_vec(H, S, L)  # [0,1]
    rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
    return np.dstack((rgb_u8, alpha))

def gray_to_hsv_tint(arr, theme_rgb):
    """
    Add a linear-space Soft-Light with the theme to improve highlight/shadow behavior.
    """
    base_rgba = gray_to_hsv_tint0(arr, theme_rgb)
    if not ENABLE_SOFT_LIGHT:
        return base_rgba

    alpha = base_rgba[..., 3:4].astype(np.uint8)
    H, W = base_rgba.shape[:2]
    theme = np.array(theme_rgb, dtype=np.float32) / 255.0
    theme_img = np.broadcast_to(theme, (H, W, 3))

    base_lin  = _srgb_to_linear(base_rgba[..., :3].astype(np.float32) / 255.0)
    blend_lin = _srgb_to_linear(theme_img)
    mixed_lin = _soft_light_blend_linear(base_lin, blend_lin)
    out_srgb  = _linear_to_srgb(mixed_lin)

    out_u8 = (out_srgb * 255.0 + 0.5).astype(np.uint8)
    return np.dstack((out_u8, alpha))

# =========================================================
#                   Frame Generation
# =========================================================

def parse_theme_rgb(text, uid_for_log=""):
    if text is None:
        raise ValueError("RGB column is empty")
    s = str(text)
    if ',' in s:
        parts = s.split(',')
    elif '，' in s:
        parts = s.split('，')
    else:
        raise ValueError('Cant identify theme RGB of '+str(uid_for_log))
    vals = [int(num.strip()) for num in parts]
    if len(vals) != 3:
        raise ValueError('RGB must have 3 numbers')
    return np.array(vals, dtype=np.uint8)

def generate_frame(rec, base_image_transp, base_image, overlay_2dcode, folder):
    theme = parse_theme_rgb(rec['RGB'], rec.get('ID：',''))
    uid     = str(rec['ID：']).strip()
    title   = str(rec['称号：']).strip()
    message = str(rec['想说的话：']).strip()
    wish    = str(rec['愿望：']).strip()
    if not wish or wish.lower() == 'nan':
        wish = DEFAULT_WISH_FALLBACK

    # load base layers
    base  = Image.open(base_image).convert("RGBA")
    W, H  = base.size
    arr   = np.array(base)

    base0 = Image.open(base_image_transp).convert("RGBA")
    arr0  = np.array(base0)

    # tint
    rgba_tint0 = gray_to_hsv_tint(arr0, theme)
    colored0   = Image.fromarray(rgba_tint0, 'RGBA')
    os.makedirs(folder, exist_ok=True)
    colored0.save(os.path.join(folder, f"{uid}0.png"))

    rgba_tint  = gray_to_hsv_tint(arr, theme)
    colored    = Image.fromarray(rgba_tint, 'RGBA')

    # overlay + texts
    overlay = Image.open(overlay_2dcode).convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    # Map of texts; skip any text that contains Chinese
    text_map = {
        'uid': uid.upper(),
        'title': title,
        'message': message,
        'wish': wish,
        'watermark': WATERMARK_TEXT
    }

    for key, cfg in TEXT_CONFIG.items():
        raw_text = text_map[key]
        if contains_chinese(raw_text):
            continue  # ignore Chinese texts as requested

        text = raw_text
        if not text:
            continue

        cx = int(W * cfg['center'][0]); cy = int(H * cfg['center'][1])
        font = load_system_font(cfg['fonts'], cfg['size'])
        bbox = draw.textbbox((0,0), text, font=font)
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]

        if cfg['align']=='L':
            if key == 'wish':
                x = cx; y = cy - h
                draw_text_with_wrap_and_left_bottom_align(draw, font, text, x, y, cfg['max_width'], cfg['opacity'])
            else:
                x = cx; y = cy - h//2
                if cfg.get('shadow',0) > 0:
                    sd = cfg['shadow']
                    draw.text((x+sd, y+sd), text, font=font, fill=(0,0,0,cfg['opacity']))
                draw.text((x, y), text, font=font, fill=(255,255,255,cfg['opacity']))

        elif cfg['align']=='R':
            # shrink UID if needed
            if key == 'uid':
                size = cfg['size']
                while w > UID_MAX_LENGTH_PIXELS and size > 8:
                    size -= 2
                    font = load_system_font(cfg['fonts'], size)
                    bbox = draw.textbbox((0,0), text, font=font)
                    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]

            x = cx - w; y = cy - h//2
            if cfg.get('shadow',0) > 0:
                sd = cfg['shadow']
                draw.text((x+sd, y+sd), text, font=font, fill=(0,0,0,cfg['opacity']))
            draw.text((x, y), text, font=font, fill=(255,255,255,cfg['opacity']))

        else:  # 'M' center
            draw_centered_text(draw, font, text, cx, cy, cfg['opacity'])

    combined = Image.alpha_composite(colored, overlay)
    combined.save(os.path.join(folder, f"{uid}1.png"))
    combined2 = Image.alpha_composite(Image.open(os.path.join(folder, f"{uid}0.png")).convert("RGBA"), combined)
    combined2.save(os.path.join(folder, f"preview_{uid}1.png"))

# =========================================================
#                        Main
# =========================================================

def main():
    excel_path = input("请输入包含字段(RGB, ID：,称号：,想说的话：,愿望：)的 Excel 路径：")
    df = pd.read_excel(excel_path)

    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise KeyError(f"Excel 缺少列: {col}")

    records = df.to_dict('records')

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(generate_frame, rec, BASE_IMAGE_TRANSPARENT, BASE_IMAGE, OVERLAY_IMAGE, OUTPUT_DIR)
                   for rec in records]
        for f in as_completed(futures):
            f.result()

if __name__ == "__main__":
    main()
