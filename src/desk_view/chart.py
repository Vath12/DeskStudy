"""Generate the printable calibration chart (PDF + PNG preview)."""

import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .board import BoardSpec

PAGES_MM = {"letter": (279.4, 215.9), "a4": (297.0, 210.0)}  # landscape
SQUARE_PX = 850  # each square is an exact number of pixels; the DPI is derived from it


def _font(size_px: int) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(path, size_px)
        except OSError:
            pass
    return ImageFont.load_default(size=size_px)


def render_page(spec: BoardSpec, page: str) -> tuple[Image.Image, float]:
    page_w_mm, page_h_mm = PAGES_MM[page]
    dpi = SQUARE_PX / (spec.square_mm / 25.4)
    px_per_mm = dpi / 25.4
    page_w, page_h = round(page_w_mm * px_per_mm), round(page_h_mm * px_per_mm)

    board_w, board_h = spec.squares_x * SQUARE_PX, spec.squares_y * SQUARE_PX
    board_img = spec.make().generateImage((board_w, board_h), marginSize=0, borderBits=1)

    canvas = Image.new("L", (page_w, page_h), 255)
    left, top = (page_w - board_w) // 2, (page_h - board_h) // 2
    canvas.paste(Image.fromarray(board_img), (left, top))

    draw = ImageDraw.Draw(canvas)
    font = _font(round(3.2 * px_per_mm))
    small = _font(round(2.6 * px_per_mm))
    margin_mid_top = top / 2
    draw.text(
        (page_w / 2, margin_mid_top),
        "DESK VIEW CALIBRATION CHART  -  print at 100% / \"Actual size\" (not \"Fit to page\")",
        font=font, fill=0, anchor="mm",
    )

    # A 100 mm ruler in the bottom margin so the print scale can be checked.
    ruler_y = top + board_h + (page_h - top - board_h) * 0.38
    x0 = left
    x1 = x0 + 100 * px_per_mm
    line_w = max(2, round(0.25 * px_per_mm))
    draw.line([(x0, ruler_y), (x1, ruler_y)], fill=0, width=line_w)
    for mm in range(0, 101, 10):
        x = x0 + mm * px_per_mm
        tick = (2.5 if mm % 50 == 0 else 1.5) * px_per_mm
        draw.line([(x, ruler_y - tick), (x, ruler_y)], fill=0, width=line_w)
    draw.text((x1 + 2 * px_per_mm, ruler_y), "100 mm", font=small, fill=0, anchor="lm")
    draw.text(
        (left + board_w, ruler_y),
        f"Square = {spec.square_mm:g} mm. Measure one and enter it in the app.",
        font=small, fill=0, anchor="rm",
    )

    # Pure black/white keeps the PDF lossless (Pillow would JPEG-compress greyscale).
    return canvas.point(lambda v: 255 if v >= 128 else 0).convert("1"), dpi


def generate(out_dir: Path, spec: BoardSpec = BoardSpec()) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for page in PAGES_MM:
        img, dpi = render_page(spec, page)
        pdf = out_dir / f"desk_view_chart_{page}.pdf"
        img.save(pdf, "PDF", resolution=dpi)
        written.append(pdf)
        if page == "letter":
            preview = out_dir / "preview.png"
            scale = 1200 / img.width
            img.convert("L").resize((1200, round(img.height * scale)), Image.LANCZOS).save(preview)
            written.append(preview)
    spec_file = out_dir / "board.json"
    spec_file.write_text(json.dumps(asdict(spec), indent=2) + "\n")
    written.append(spec_file)
    return written


if __name__ == "__main__":
    for path in generate(Path(__file__).resolve().parents[2] / "calibration_chart"):
        print(path)
