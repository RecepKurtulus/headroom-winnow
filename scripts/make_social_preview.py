"""Render the GitHub social preview card (1280x640) from the logo and headline numbers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent.parent / "docs" / "assets"
FONTS = Path("C:/Windows/Fonts")
W, H = 1280, 640
BG_TOP, BG_BOTTOM = (30, 27, 75), (17, 16, 46)
GOLD, LAVENDER, WHITE, MUTED = (250, 204, 21), (165, 180, 252), (248, 250, 252), (165, 168, 200)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


def main() -> None:
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    for y in range(H):  # vertical gradient
        t = y / (H - 1)
        draw.line(
            [(0, y), (W, y)], fill=tuple(round(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM))
        )

    logo = Image.open(HERE / "logo.png").convert("RGBA").resize((300, 300), Image.LANCZOS)
    img.paste(logo, (90, 150), logo)

    x = 450
    draw.text((x, 128), "Winnow", font=font("segoeuib.ttf", 112), fill=WHITE)
    draw.text(
        (x, 268),
        "Separate the grain from the chaff\nin your coding agent's tool output.",
        font=font("segoeui.ttf", 38),
        fill=MUTED,
        spacing=10,
    )

    stats = [("−41%", "tokens"), ("0", "lost errors"), ("0.29 s", "median")]
    sx = x
    for value, label in stats:
        draw.text((sx, 400), value, font=font("segoeuib.ttf", 56), fill=GOLD)
        draw.text((sx, 468), label, font=font("segoeui.ttf", 26), fill=LAVENDER)
        sx += 260

    draw.text(
        (x, 556),
        "A task-aware pruner for Headroom · powered by Squeez models",
        font=font("segoeui.ttf", 24),
        fill=MUTED,
    )
    img.save(HERE / "social-preview.png", optimize=True)


if __name__ == "__main__":
    main()
