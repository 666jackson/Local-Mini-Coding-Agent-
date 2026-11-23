"""把 asciinema .cast 轉成 GIF：pyte 模擬終端，PIL 逐格畫出來（README 的示範動畫用這個產生）。

用法：python scripts/cast2gif.py in.cast out.gif [--max-idle 1.0]
字型路徑是 Ubuntu 的 Noto 字型（fonts-noto-core、fonts-noto-cjk），其他系統請改 MONO / CJK 常數。
"""

import argparse
import json
import re

import pyte
from fontTools.ttLib import TTCollection, TTFont
from PIL import Image, ImageDraw, ImageFont
from wcwidth import wcwidth

MONO = "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"
MONO_B = "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf"
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CJK_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
CJK_INDEX = 3  # Noto Sans CJK TC（.ttc 內的第幾個字型）
# 前兩種字型缺的符號（✓ ✗ ✋ ↺ 等）依序到這些字型找
FALLBACK = ["/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]

BG = (24, 26, 33)
FG = (220, 223, 228)
PALETTE = {
    "default": FG, "black": (40, 42, 54), "red": (255, 110, 110), "green": (120, 220, 130),
    "brown": (240, 200, 100), "yellow": (240, 200, 100), "blue": (110, 160, 255),
    "magenta": (220, 140, 240), "cyan": (110, 210, 230), "white": (230, 230, 230),
    "brightblack": (130, 135, 145), "brightred": (255, 140, 140), "brightgreen": (150, 240, 160),
    "brightyellow": (255, 220, 130), "brightblue": (140, 180, 255), "brightmagenta": (230, 170, 250),
    "brightcyan": (150, 230, 240), "brightwhite": (255, 255, 255),
}


def color(name: str, default: tuple) -> tuple:
    """pyte 的顏色名稱或 6 位 hex 轉成 RGB。"""
    if name == "default":
        return default
    if name in PALETTE:
        return PALETTE[name]
    if len(name) == 6:
        try:
            return tuple(int(name[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cast")
    ap.add_argument("gif")
    ap.add_argument("--max-idle", type=float, default=1.2, help="畫面靜止超過幾秒就壓縮成這麼長")
    ap.add_argument("--font-size", type=int, default=15)
    ap.add_argument("--hold-end", type=float, default=6.0)
    ap.add_argument("--chunk-lines", type=int, default=4, help="大量輸出每段幾行")
    ap.add_argument("--chunk-pause", type=float, default=0.35, help="大量輸出每段停幾秒")
    args = ap.parse_args()

    lines = open(args.cast, encoding="utf-8").read().splitlines()
    header = json.loads(lines[0])
    cols, rows = header["width"], header["height"]
    events = [json.loads(l) for l in lines[1:] if l.strip()]
    events = [e for e in events if e[1] == "o"]

    # pyte 不支援 SGR 2（暗色），改成灰色前景；後面若有其他前景色會覆蓋它
    def undim(m: "re.Match") -> str:
        parts = ["90" if p == "2" else p for p in m.group(1).split(";")]
        return "\x1b[" + ";".join(parts) + "m"

    for e in events:
        e[2] = re.sub(r"\x1b\[([0-9;]*)m", undim, e[2])

    # 一次印出很多行（例如結尾的 diff）時拆成幾段、每段停一下，GIF 裡才看得到捲動過程
    paced, shift = [], 0.0
    for t, kind, data in events:
        lines = data.split("\n")
        if len(lines) <= args.chunk_lines * 2:
            paced.append([t + shift, kind, data])
            continue
        for k in range(0, len(lines), args.chunk_lines):
            piece = "\n".join(lines[k:k + args.chunk_lines])
            if k + args.chunk_lines < len(lines):
                piece += "\n"
            paced.append([t + shift, kind, piece])
            shift += args.chunk_pause
    events = paced

    fs = args.font_size
    mono, mono_b = ImageFont.truetype(MONO, fs), ImageFont.truetype(MONO_B, fs)
    cjk = ImageFont.truetype(CJK, fs, index=CJK_INDEX)
    cjk_b = ImageFont.truetype(CJK_B, fs, index=CJK_INDEX)
    mono_cmap = TTFont(MONO, lazy=True).getBestCmap()
    cjk_cmap = TTCollection(CJK).fonts[CJK_INDEX].getBestCmap()
    fallbacks = [(ImageFont.truetype(f, fs), TTFont(f, lazy=True).getBestCmap()) for f in FALLBACK]

    def pick(data: str, wide: bool, bold: bool):
        """依字元選字型：等寬英數 → CJK → 符號後備字型。"""
        cp = ord(data[0])
        if not wide and cp <= 0x2FFF and cp in mono_cmap:
            return mono_b if bold else mono
        if cp in cjk_cmap:
            return cjk_b if bold else cjk
        for f, cmap in fallbacks:
            if cp in cmap:
                return f
        return mono

    cw = round(mono.getlength("M"))
    ch = round(fs * 1.45)
    pad = 14
    W, H = cols * cw + pad * 2, rows * ch + pad * 2

    screen = pyte.Screen(cols, rows)
    stream = pyte.Stream(screen)

    def render() -> Image.Image:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for y in range(rows):
            line = screen.buffer[y]
            x = 0
            while x < cols:
                c = line[x]
                data = c.data
                if data == "":
                    x += 1
                    continue
                wide = wcwidth(data) == 2
                fg = color(c.fg, FG)
                bg = color(c.bg, BG)
                if c.reverse:
                    fg, bg = bg, fg
                px, py = pad + x * cw, pad + y * ch
                span = 2 if wide else 1
                if bg != BG:
                    d.rectangle([px, py, px + cw * span, py + ch], fill=bg)
                if data.strip():
                    f = pick(data, wide, c.bold)
                    d.text((px, py + (ch - fs) // 2 - 1), data, font=f, fill=fg)
                x += span
        return img

    # 合併時間很接近的輸出，壓縮過長的等待
    frames, durations = [], []
    t_prev, virt = 0.0, 0.0
    i = 0
    while i < len(events):
        t = events[i][0]
        gap = min(t - t_prev, args.max_idle)
        if frames:
            durations[-1] = max(int(gap * 1000), 20)
        # 同一批：50ms 內的輸出畫成一格
        while i < len(events) and events[i][0] - t <= 0.05:
            stream.feed(events[i][2])
            i += 1
        frames.append(render())
        durations.append(100)
        t_prev = events[i - 1][0]
    durations[-1] = int(args.hold_end * 1000)

    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    pal[0].save(args.gif, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True,
                disposal=1)
    total = sum(durations) / 1000
    print(f"{len(frames)} frames, {W}x{H}, {total:.1f}s")


if __name__ == "__main__":
    main()
