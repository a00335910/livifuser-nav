"""Generate exact-scale A4 calibration targets as vector PDF and 300-DPI JPEG."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "assets" / "calibration"
PDF_PATH = OUTPUT_DIR / "checkerboard_8x6_25mm_A4.pdf"
JPEG_PATH = OUTPUT_DIR / "checkerboard_8x6_25mm_A4_300dpi.jpg"

PAGE_WIDTH_MM = 297.0
PAGE_HEIGHT_MM = 210.0
SQUARE_MM = 25.0
COLUMNS = 9
ROWS = 7
BOARD_WIDTH_MM = COLUMNS * SQUARE_MM
BOARD_HEIGHT_MM = ROWS * SQUARE_MM
BOARD_X_MM = (PAGE_WIDTH_MM - BOARD_WIDTH_MM) / 2
BOARD_Y_MM = (PAGE_HEIGHT_MM - BOARD_HEIGHT_MM) / 2
DPI = 300


def mm_to_points(value: float) -> float:
    return value * 72.0 / 25.4


def mm_to_pixels(value: float) -> int:
    return round(value * DPI / 25.4)


def generate_pdf() -> None:
    page_size = landscape(A4)
    document = canvas.Canvas(str(PDF_PATH), pagesize=page_size)

    for row in range(ROWS):
        for column in range(COLUMNS):
            if (row + column) % 2 == 0:
                x_mm = BOARD_X_MM + column * SQUARE_MM
                # ReportLab coordinates begin at the bottom-left.
                y_mm = PAGE_HEIGHT_MM - BOARD_Y_MM - (row + 1) * SQUARE_MM
                document.rect(
                    mm_to_points(x_mm),
                    mm_to_points(y_mm),
                    mm_to_points(SQUARE_MM),
                    mm_to_points(SQUARE_MM),
                    stroke=0,
                    fill=1,
                )

    document.rect(
        mm_to_points(BOARD_X_MM),
        mm_to_points(BOARD_Y_MM),
        mm_to_points(BOARD_WIDTH_MM),
        mm_to_points(BOARD_HEIGHT_MM),
        stroke=1,
        fill=0,
    )
    document.setLineWidth(mm_to_points(0.5))
    document.line(mm_to_points(98.5), mm_to_points(8), mm_to_points(198.5), mm_to_points(8))
    document.line(mm_to_points(98.5), mm_to_points(5), mm_to_points(98.5), mm_to_points(11))
    document.line(mm_to_points(198.5), mm_to_points(5), mm_to_points(198.5), mm_to_points(11))
    document.setFont("Helvetica", 8)
    document.drawCentredString(
        mm_to_points(PAGE_WIDTH_MM / 2),
        mm_to_points(2.5),
        "This line must measure exactly 100 mm - print at 100% / Actual Size",
    )
    document.showPage()
    document.save()


def generate_jpeg() -> None:
    width = mm_to_pixels(PAGE_WIDTH_MM)
    height = mm_to_pixels(PAGE_HEIGHT_MM)
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)

    for row in range(ROWS):
        for column in range(COLUMNS):
            if (row + column) % 2 == 0:
                left = mm_to_pixels(BOARD_X_MM + column * SQUARE_MM)
                top = mm_to_pixels(BOARD_Y_MM + row * SQUARE_MM)
                right = mm_to_pixels(BOARD_X_MM + (column + 1) * SQUARE_MM)
                bottom = mm_to_pixels(BOARD_Y_MM + (row + 1) * SQUARE_MM)
                draw.rectangle((left, top, right, bottom), fill=0)

    draw.rectangle(
        (
            mm_to_pixels(BOARD_X_MM),
            mm_to_pixels(BOARD_Y_MM),
            mm_to_pixels(BOARD_X_MM + BOARD_WIDTH_MM),
            mm_to_pixels(BOARD_Y_MM + BOARD_HEIGHT_MM),
        ),
        outline=0,
        width=2,
    )
    line_y = mm_to_pixels(202)
    line_start = mm_to_pixels(98.5)
    line_end = mm_to_pixels(198.5)
    draw.line((line_start, line_y, line_end, line_y), fill=0, width=6)
    draw.line((line_start, line_y - 35, line_start, line_y + 35), fill=0, width=6)
    draw.line((line_end, line_y - 35, line_end, line_y + 35), fill=0, width=6)

    image.save(JPEG_PATH, quality=100, subsampling=0, dpi=(DPI, DPI))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_pdf()
    generate_jpeg()
    print(f"Created {PDF_PATH}")
    print(f"Created {JPEG_PATH}")


if __name__ == "__main__":
    main()
