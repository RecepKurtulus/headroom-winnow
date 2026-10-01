"""Render docs/assets/demo.gif: a real before/after from scripts/demo/*.txt as a terminal clip.

The texts are produced by running Winnow (pooled model) on one example from the
Squeez test split; see scripts/demo/meta.json. Only long paths are shortened.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "scripts" / "demo"
OUT = ROOT / "docs" / "assets" / "demo.gif"
FONTS = Path("C:/Windows/Fonts")

W, H = 980, 600
PAD, TOP = 22, 74
LINE_H, VISIBLE, MAX_CHARS = 20, 23, 104
BG, CHROME, FG, DIM = (13, 17, 23), (22, 27, 34), (201, 209, 217), (125, 133, 144)
GOLD, LAVENDER, RED, GREEN = (250, 204, 21), (165, 180, 252), (248, 113, 113), (74, 222, 128)

MONO = ImageFont.truetype(str(FONTS / "consola.ttf"), 15)
MONO_B = ImageFont.truetype(str(FONTS / "consolab.ttf"), 15)
UI = ImageFont.truetype(str(FONTS / "segoeuib.ttf"), 17)


def shorten(line: str) -> str:
    line = line.replace("/home/dev/go/src/github.com/example/project/", ".../")
    line = line.replace("github.com/example/project/", ".../").replace("\t", "    ")
    return line if len(line) <= MAX_CHARS else line[: MAX_CHARS - 1] + "…"


def frame(
    lines: list[tuple[str, tuple[int, int, int]]],
    caption: str,
    status: str,
    status_color: tuple[int, int, int],
) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 36], fill=CHROME)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([16 + i * 22, 12, 28 + i * 22, 24], fill=c)
    d.text((W // 2, 18), "agent tool output", font=UI, fill=DIM, anchor="mm")
    d.text((PAD, 46), caption, font=UI, fill=LAVENDER)
    for i, (text, color) in enumerate(lines[-VISIBLE:]):
        font = MONO_B if color in (GOLD, LAVENDER) else MONO
        d.text((PAD, TOP + i * LINE_H), text, font=font, fill=color)
    d.rectangle([0, H - 34, W, H], fill=CHROME)
    d.text((PAD, H - 17), status, font=UI, fill=status_color, anchor="lm")
    return img


def main() -> None:
    meta = json.loads((DEMO / "meta.json").read_text(encoding="utf-8"))
    before = [shorten(x) for x in (DEMO / "before.txt").read_text(encoding="utf-8").splitlines()]
    after = [shorten(x) for x in (DEMO / "after.txt").read_text(encoding="utf-8").splitlines()]
    caption = "Task: find the missing GetObject method in UserHandler"
    frames: list[Image.Image] = []
    durations: list[int] = []

    def add(img: Image.Image, ms: int) -> None:
        frames.append(img)
        durations.append(ms)

    cmd = "$ go build ./..."
    for i in range(1, len(cmd) + 1):
        add(frame([(cmd[:i] + "▌", FG)], caption, "", DIM), 55)

    n_before = len(before)
    if before and before[0] == cmd:  # the recorded output starts with the command itself
        before = before[1:]
    shown: list[tuple[str, tuple[int, int, int]]] = [(cmd, FG)]
    raw_status = f"raw output · {n_before} lines · {meta['tokens_before']:,} tokens"
    for i in range(0, len(before), 4):
        shown += [(x, FG) for x in before[i : i + 4]]
        add(frame(shown, caption, raw_status, RED), 70)
    add(frame(shown, caption, raw_status, RED), 1600)

    add(frame([], caption, "Winnow is pruning…", LAVENDER), 700)

    saved = 1 - meta["tokens_after"] / meta["tokens_before"]
    win_status = (
        f"Winnow · {len(after)} lines · {meta['tokens_after']:,} tokens · −{saved:.0%}"
        "   ·   every <<ccr>> marker is retrievable"
    )
    styled = [(x, LAVENDER if x.startswith("<<ccr:") else FG) for x in after]
    for i in range(1, VISIBLE + 1, 2):
        add(frame(styled[:i], caption, win_status, GREEN), 90)
    add(frame(styled[:VISIBLE], caption, win_status, GREEN), 4500)

    palette = [f.convert("P", palette=Image.ADAPTIVE, colors=48) for f in frames]
    palette[0].save(
        OUT, save_all=True, append_images=palette[1:], duration=durations, loop=0, optimize=True
    )
    print(OUT, f"{OUT.stat().st_size / 1024:.0f} KB", len(frames), "frames")


if __name__ == "__main__":
    main()
