# Draws runtime/mkw.ico: Mario Kart World PC app icon (red tile, chequered band, MKW).
# Run: py -3 tools/make_icon.py
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
S = 512


def font(size):
    for name in (
        'C:/Windows/Fonts/arialbd.ttf',
        'C:/Windows/Fonts/segoeuib.ttf',
        'C:/Windows/Fonts/tahomabd.ttf',
        'arialbd.ttf',
        'segoeuib.ttf',
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_icon(size: int) -> Image.Image:
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m = max(1, size // 32)          # outer margin
    r = max(8, size * 52 // 256)    # corner radius
    bw = max(2, size * 10 // 256)   # border width

    # Deep red tile + subtle top highlight (flat, not glossy purple AI sludge).
    d.rounded_rectangle((m, m, size - m - 1, size - m - 1), radius=r, fill=(196, 16, 28, 255))
    hi = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hi)
    hd.rounded_rectangle((m, m, size - m - 1, size // 2), radius=r, fill=(255, 255, 255, 36))
    img = Image.alpha_composite(img, hi)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((m, m, size - m - 1, size - m - 1), radius=r, outline=(255, 255, 255, 255), width=bw)

    # Chequered racing band
    inset = m + bw + max(2, size // 40)
    band_h = max(8, size * 40 // 256)
    band_top = size - inset - band_h - max(2, size // 64)
    cell = max(4, size * 20 // 256)
    usable = size - 2 * inset
    cols = max(1, (usable + cell - 1) // cell)
    rows = max(1, (band_h + cell - 1) // cell)
    for i in range(cols):
        for j in range(rows):
            x0 = inset + i * cell
            y0 = band_top + j * cell
            x1 = min(x0 + cell, size - inset) - 1
            y1 = min(y0 + cell, band_top + band_h) - 1
            if x1 <= x0 or y1 <= y0:
                continue
            col = (250, 250, 252, 255) if (i + j) % 2 == 0 else (18, 18, 22, 255)
            d.rectangle((x0, y0, x1, y1), fill=col)

    # MKW wordmark
    f = font(max(24, size * 74 // 256))
    text = 'MKW'
    box = d.textbbox((0, 0), text, font=f)
    tw, th = box[2] - box[0], box[3] - box[1]
    tx = (size - tw) / 2 - box[0]
    ty = (band_top - th) / 2 - box[1] + size * 4 // 256
    # soft shadow for small-size legibility
    shadow = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.text((tx + size // 64, ty + size // 64), text, font=f, fill=(0, 0, 0, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(1, size // 128)))
    img = Image.alpha_composite(img, shadow)
    d = ImageDraw.Draw(img)
    d.text((tx, ty), text, font=f, fill=(255, 255, 255, 255))
    return img


master = draw_icon(S)
sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
frames = []
for wh in sizes:
    fr = master.resize(wh, Image.Resampling.LANCZOS)
    frames.append(fr)

out = ROOT / 'runtime' / 'mkw.ico'
frames[0].save(out, format='ICO', sizes=[(f.width, f.height) for f in frames], append_images=frames[1:])
# Pillow ICO writer is picky; write via the largest and sizes= list (compatible path):
master.resize((256, 256), Image.Resampling.LANCZOS).save(
    out, format='ICO', sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
preview = ROOT / 'runtime' / 'mkw_icon_preview.png'
master.resize((256, 256), Image.Resampling.LANCZOS).save(preview)
print('wrote', out)
print('wrote', preview)
