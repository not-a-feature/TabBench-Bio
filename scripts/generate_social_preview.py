"""Generate the TabBench-Bio social preview from the current dashboard leaders."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = PROJECT_ROOT / "docs" / "data" / "dashboard.json"
OUTPUT = PROJECT_ROOT / "docs" / "assets" / "og.png"
SIZE = (1732, 908)

BG = "#f4f5f1"
SURFACE = "#ffffff"
SURFACE_STRONG = "#ecefe8"
INK = "#161a18"
MUTED = "#616862"
LINE = "#d9ded8"
GREEN = "#176b52"
GREEN_SOFT = "#dfeee7"
VIOLET = "#7357a8"
VIOLET_SOFT = "#e8e2f1"
AMBER = "#b46a14"
AMBER_SOFT = "#f3e7d7"

SERIF_FONTS = [
    Path("C:/Windows/Fonts/georgia.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    Path("/System/Library/Fonts/NewYork.ttf"),
]
SANS_FONTS = [
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/System/Library/Fonts/SFNS.ttf"),
]
SANS_BOLD_FONTS = [
    Path("C:/Windows/Fonts/segoeuib.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/System/Library/Fonts/SFNS.ttf"),
]


def font(candidates: list[Path], size: int) -> ImageFont.FreeTypeFont:
    available = [path for path in candidates if path.exists()]
    assert available, f"None of the required fonts exist: {candidates}"
    return ImageFont.truetype(str(available[0]), size)


def fitted_font(
    draw: ImageDraw.ImageDraw,
    candidates: list[Path],
    text: str,
    maximum_size: int,
    minimum_size: int,
    maximum_width: int,
) -> ImageFont.FreeTypeFont:
    for size in range(maximum_size, minimum_size - 1, -1):
        candidate = font(candidates, size)
        if draw.textbbox((0, 0), text, font=candidate)[2] <= maximum_width:
            return candidate
    raise ValueError(f"Text does not fit in {maximum_width}px: {text}")


def draw_podium(
    draw: ImageDraw.ImageDraw,
    leaders: list[dict[str, object]],
    left: int,
    top: int,
    width: int,
) -> None:
    rank_order = [1, 0, 2]
    step_heights = [112, 172, 78]
    rank_colors = [VIOLET, GREEN, AMBER]
    rank_fills = [VIOLET_SOFT, GREEN_SOFT, AMBER_SOFT]
    gap = 18
    column_width = (width - 2 * gap) // 3
    baseline = top + 360
    name_area_top = top + 26

    for column, leader_index in enumerate(rank_order):
        leader = leaders[leader_index]
        rank = leader_index + 1
        x = left + column * (column_width + gap)
        center = x + column_width // 2
        step_height = step_heights[column]
        color = rank_colors[column]
        fill = rank_fills[column]
        display = str(leader["display"])
        elo = f'{int(leader["Elo"]):,}'

        name_font = fitted_font(draw, SANS_BOLD_FONTS, display, 26, 18, column_width - 18)
        name_box = draw.textbbox((0, 0), display, font=name_font)
        draw.text((center - (name_box[2] - name_box[0]) / 2, name_area_top), display, font=name_font, fill=INK)

        elo_font = font(SERIF_FONTS, 44 if rank == 1 else 38)
        elo_box = draw.textbbox((0, 0), elo, font=elo_font)
        draw.text((center - (elo_box[2] - elo_box[0]) / 2, name_area_top + 50), elo, font=elo_font, fill=color)

        interval = f'{int(leader["Elo_lo"]):,}–{int(leader["Elo_hi"]):,}'
        interval_font = font(SANS_FONTS, 17)
        interval_box = draw.textbbox((0, 0), interval, font=interval_font)
        draw.text(
            (center - (interval_box[2] - interval_box[0]) / 2, name_area_top + 108),
            interval,
            font=interval_font,
            fill=MUTED,
        )

        draw.rounded_rectangle(
            (x, baseline - step_height, x + column_width, baseline),
            radius=14,
            fill=fill,
            outline=color,
            width=3,
        )
        rank_font = font(SERIF_FONTS, 70 if rank == 1 else 58)
        rank_text = str(rank)
        rank_box = draw.textbbox((0, 0), rank_text, font=rank_font)
        draw.text(
            (
                center - (rank_box[2] - rank_box[0]) / 2,
                baseline - step_height + (step_height - (rank_box[3] - rank_box[1])) / 2 - rank_box[1],
            ),
            rank_text,
            font=rank_font,
            fill=color,
        )


def main() -> None:
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    excluded = set(dashboard["meta"]["plot_excluded_models"])
    leaders = sorted(
        (row for row in dashboard["reference"] if row["model_id"] not in excluded),
        key=lambda row: row["Elo"],
        reverse=True,
    )[:3]
    assert len(leaders) == 3
    assert all(leader["cell_label"] == leaders[0]["cell_label"] for leader in leaders)

    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)

    for x in range(1030, 1690, 58):
        draw.line((x, 70, x, 838), fill="#e4e7e1", width=2)
    for y in range(84, 850, 58):
        draw.line((1016, y, 1690, y), fill="#e4e7e1", width=2)

    title_font = font(SERIF_FONTS, 112)
    draw.text((108, 262), "TabBench-Bio", font=title_font, fill=INK)
    subtitle_font = font(SANS_FONTS, 38)
    draw.text((116, 410), "A living benchmark for biomedical", font=subtitle_font, fill=INK)
    draw.text((116, 462), "tabular learning.", font=subtitle_font, fill=INK)
    draw.line((116, 572, 800, 572), fill=LINE, width=3)
    detail_font = font(SANS_FONTS, 24)
    detail = (
        f'{len(dashboard["datasets"])} datasets  ·  '
        f'{int(dashboard["meta"]["configured_model_count"])} model configurations  ·  5-fold CV'
    )
    draw.text((116, 616), detail, font=detail_font, fill=MUTED)
    url_font = font(SANS_BOLD_FONTS, 24)
    draw.text((116, 752), "tabbench-bio.eu", font=url_font, fill=GREEN)

    card = (932, 94, 1650, 814)
    draw.rounded_rectangle(card, radius=34, fill=SURFACE, outline=LINE, width=3)
    kicker_font = font(SANS_BOLD_FONTS, 20)
    draw.text((990, 146), "CURRENT REFERENCE PODIUM", font=kicker_font, fill=GREEN)
    setting_font = font(SANS_FONTS, 21)
    setting = f'Macro-F1 Elo  ·  {leaders[0]["cell_label"]}  ·  {int(leaders[0]["n_targets"])} targets'
    draw.text((990, 184), setting, font=setting_font, fill=MUTED)
    draw.line((990, 228, 1592, 228), fill=LINE, width=2)
    draw_podium(draw, leaders, 990, 258, 602)
    note_font = font(SANS_FONTS, 17)
    draw.text((990, 744), "Bars show rank; numbers show Elo and 95% interval.", font=note_font, fill=MUTED)

    assert image.size == SIZE
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    print(f"Wrote {OUTPUT.relative_to(PROJECT_ROOT)} ({SIZE[0]}×{SIZE[1]})")
    for rank, leader in enumerate(leaders, start=1):
        print(f'{rank}. {leader["display"]}: {int(leader["Elo"]):,}')


if __name__ == "__main__":
    main()
