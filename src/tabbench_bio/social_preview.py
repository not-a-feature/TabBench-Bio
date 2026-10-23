"""Generate the TabBench-Bio social preview from the current dashboard leaders."""

import hashlib
import json
import re
from pathlib import Path
from textwrap import wrap

from matplotlib import get_data_path
from PIL import Image, ImageDraw, ImageFont

BUNDLED_FONTS = Path(get_data_path()) / "fonts" / "ttf"
WEB_FONTS = Path(__file__).parent / "web" / "assets" / "fonts"
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
    BUNDLED_FONTS / "DejaVuSerif.ttf",
]
SANS_FONTS = [
    WEB_FONTS / "AtkinsonHyperlegibleNext-Regular.woff2",
]
SANS_BOLD_FONTS = [
    WEB_FONTS / "AtkinsonHyperlegibleNext-Bold.woff2",
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
    baseline = top + 410
    name_area_top = top + 26

    for column, leader_index in enumerate(rank_order):
        if leader_index >= len(leaders):
            continue
        leader = leaders[leader_index]
        rank = leader_index + 1
        x = left + column * (column_width + gap)
        center = x + column_width // 2
        step_height = step_heights[column]
        color = rank_colors[column]
        fill = rank_fills[column]
        display = str(leader["display"])
        elo = f"{int(leader['Elo']):,}"

        name = "\n".join(wrap(display, width=11, break_long_words=False, break_on_hyphens=False))
        name_font = fitted_font(draw, SANS_BOLD_FONTS, name, 34, 28, column_width - 12)
        name_box = draw.multiline_textbbox((0, 0), name, font=name_font, align="center")
        draw.multiline_text(
            (center - (name_box[2] - name_box[0]) / 2, name_area_top),
            name,
            font=name_font,
            fill=INK,
            align="center",
        )

        elo_font = font(SERIF_FONTS, 56 if rank == 1 else 50)
        elo_box = draw.textbbox((0, 0), elo, font=elo_font)
        draw.text(
            (center - (elo_box[2] - elo_box[0]) / 2, name_area_top + 92),
            elo,
            font=elo_font,
            fill=color,
        )

        interval = f"{int(leader['Elo_lo']):,}–{int(leader['Elo_hi']):,}"
        interval_font = font(SANS_FONTS, 26)
        interval_box = draw.textbbox((0, 0), interval, font=interval_font)
        draw.text(
            (center - (interval_box[2] - interval_box[0]) / 2, name_area_top + 160),
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
        rank_font = font(SERIF_FONTS, 84 if rank == 1 else 70)
        rank_text = str(rank)
        rank_box = draw.textbbox((0, 0), rank_text, font=rank_font)
        draw.text(
            (
                center - (rank_box[2] - rank_box[0]) / 2,
                baseline
                - step_height
                + (step_height - (rank_box[3] - rank_box[1])) / 2
                - rank_box[1],
            ),
            rank_text,
            font=rank_font,
            fill=color,
        )


def generate_social_preview(dashboard_path: Path, output: Path) -> None:
    """Render the reference-leader card from one generated site dashboard."""
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    if "analysis_views" in dashboard:
        dashboard.update(dashboard["analysis_views"]["strict"])
    excluded = set(dashboard["meta"]["plot_excluded_models"])
    leaders = sorted(
        (
            {**row, **dashboard["models"][row["model_id"]]}
            for row in dashboard["reference"]
            if row["model_id"] not in excluded
            and row["model_id"] != "AUTOGLUON"
            and not dashboard["models"][row["model_id"]]["training_data_overlap"]
        ),
        key=lambda row: row["Elo"],
        reverse=True,
    )[:3]
    assert all(leader["cell_label"] == leaders[0]["cell_label"] for leader in leaders)

    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)

    for x in range(1030, 1690, 58):
        draw.line((x, 70, x, 838), fill="#e4e7e1", width=2)
    for y in range(84, 850, 58):
        draw.line((1016, y, 1690, y), fill="#e4e7e1", width=2)

    title = "TabBench-Bio"
    title_font = fitted_font(draw, SERIF_FONTS, title, 104, 80, 748)
    draw.text((116, 262), title, font=title_font, fill=INK)
    subtitle_font = font(SANS_FONTS, 48)
    draw.text((116, 410), "A living benchmark for biomedical", font=subtitle_font, fill=INK)
    draw.text((116, 470), "tabular learning.", font=subtitle_font, fill=INK)
    draw.line((116, 572, 800, 572), fill=LINE, width=3)
    detail_font = font(SANS_FONTS, 30)
    detail = (
        f"{len(dashboard['datasets'])} datasets  ·  "
        f"{int(dashboard['meta']['configured_model_count'])} model configurations  ·  5-fold CV"
    )
    draw.text((116, 616), detail, font=detail_font, fill=MUTED)
    url_font = font(SANS_BOLD_FONTS, 32)
    draw.text((116, 752), "tabbench-bio.eu", font=url_font, fill=GREEN)

    card = (932, 94, 1650, 814)
    draw.rounded_rectangle(card, radius=34, fill=SURFACE, outline=LINE, width=3)
    kicker_font = font(SANS_BOLD_FONTS, 28)
    draw.text((990, 146), "CURRENT REFERENCE PODIUM", font=kicker_font, fill=GREEN)
    setting_font = font(SANS_FONTS, 28)
    setting = (
        f"Macro-F1 Elo  ·  {leaders[0]['cell_label']}\n{int(leaders[0]['n_targets'])} targets"
        if leaders
        else "Reference rankings pending"
    )
    draw.multiline_text((990, 194), setting, font=setting_font, fill=MUTED, spacing=8)
    draw.line((990, 282, 1592, 282), fill=LINE, width=2)
    draw_podium(draw, leaders, 990, 310, 602)

    assert image.size == SIZE
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    print(f"Wrote {output} ({SIZE[0]}×{SIZE[1]})")
    for rank, leader in enumerate(leaders, start=1):
        print(f"{rank}. {leader['display']}: {int(leader['Elo']):,}")


def write_social_preview(site_dir: Path) -> None:
    """Refresh the generated card and its Open Graph/Twitter cache key."""
    output = site_dir / "assets" / "og.png"
    generate_social_preview(site_dir / "data" / "dashboard.json", output)
    version = hashlib.sha256(output.read_bytes()).hexdigest()[:16]
    index = site_dir / "index.html"
    text, count = re.subn(
        r'assets/og\.png(?:\?v=[^"\s]*)?',
        f"assets/og.png?v={version}",
        index.read_text(encoding="utf-8"),
    )
    assert count == 2, "Expected Open Graph and Twitter image URLs"
    index.write_text(text, encoding="utf-8")
