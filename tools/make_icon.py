# Draws runtime/mkw.ico: an original icon for Mario Kart World PC (a red rounded tile, a chequered-flag band and the
# letters MKW). No game artwork is used. Run: python tools/make_icon.py
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
S = 256


def font(size):
    for name in ('arialbd.ttf', 'segoeuib.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((8, 8, S - 8, S - 8), radius=52, fill=(226, 30, 38, 255))          # red tile
d.rounded_rectangle((8, 8, S - 8, S - 8), radius=52, outline=(255, 255, 255, 255), width=10)
# chequered band across the lower third
band_top, band_h, cell = 168, 40, 20
for i in range((S - 40) // cell + 1):
    for j in range(band_h // cell):
        x0 = 20 + i * cell
        if x0 >= S - 20:
            continue
        col = (255, 255, 255, 255) if (i + j) % 2 == 0 else (24, 24, 28, 255)
        d.rectangle((x0, band_top + j * cell, min(x0 + cell, S - 20) - 1, band_top + (j + 1) * cell - 1), fill=col)
# letters
f = font(74)
text = 'MKW'
box = d.textbbox((0, 0), text, font=f)
tw, th = box[2] - box[0], box[3] - box[1]
d.text(((S - tw) / 2 - box[0], 70 - box[1]), text, font=f, fill=(255, 255, 255, 255))
out = ROOT / 'runtime' / 'mkw.ico'
img.save(out, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
img.save(ROOT / 'runtime' / 'mkw_icon_preview.png')
print('wrote', out)
