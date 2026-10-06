"""Generate the repository method overview as SVG and high-resolution PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "assets"

INK = "#17212B"
MUTED = "#5B6773"
LINE = "#C7CED4"
PANEL = "#F7F9FA"
WHITE = "#FFFFFF"
NAVY = "#315A7D"
NAVY_LIGHT = "#E8F0F6"
TEAL = "#21867A"
TEAL_LIGHT = "#E3F2EF"
GOLD = "#C58A1A"
GOLD_LIGHT = "#FBF1D6"
CORAL = "#C75B4A"
CORAL_LIGHT = "#F9E8E4"


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    width: float = 1.2,
    dashed: bool = False,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 2,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=width,
        linestyle=(0, (3, 2)) if dashed else "solid",
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    *,
    face: str = WHITE,
    edge: str = LINE,
    title_color: str = INK,
    title_size: float = 7.2,
    subtitle_size: float = 5.7,
    radius: float = 0.008,
    linewidth: float = 0.9,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        transform=ax.transAxes,
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        zorder=3,
    )
    ax.add_patch(patch)
    title_y = y + height * (0.61 if subtitle else 0.50)
    ax.text(
        x + width / 2,
        title_y,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="semibold",
        color=title_color,
        zorder=4,
    )
    if subtitle:
        ax.text(
            x + width / 2,
            y + height * 0.29,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=MUTED,
            linespacing=1.15,
            zorder=4,
        )
    return patch


def panel_heading(ax: plt.Axes, label: str, title: str, x: float) -> None:
    ax.text(
        x,
        0.953,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        x + 0.025,
        0.953,
        title,
        transform=ax.transAxes,
        fontsize=8.3,
        fontweight="semibold",
        color=INK,
        va="top",
    )


def draw_split_icon(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            width * 0.77,
            height,
            transform=ax.transAxes,
            facecolor=TEAL,
            edgecolor="none",
            zorder=4,
        )
    )
    ax.add_patch(
        Rectangle(
            (x + width * 0.80, y),
            width * 0.20,
            height,
            transform=ax.transAxes,
            facecolor=CORAL,
            edgecolor="none",
            zorder=4,
        )
    )


def draw_local_space(ax: plt.Axes) -> None:
    x, y, width, height = 0.035, 0.355, 0.255, 0.305
    panel = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        transform=ax.transAxes,
        linewidth=0.8,
        edgecolor=LINE,
        facecolor=WHITE,
        zorder=1,
    )
    ax.add_patch(panel)
    ax.text(
        x + 0.012,
        y + height - 0.025,
        "Local feature space",
        transform=ax.transAxes,
        fontsize=6.8,
        fontweight="semibold",
        color=INK,
        va="top",
    )
    center = (x + width * 0.61, y + height * 0.49)
    for diameter, color, alpha in [
        ((0.135, 0.185), TEAL, 0.10),
        ((0.095, 0.130), TEAL, 0.14),
        ((0.055, 0.075), TEAL, 0.20),
    ]:
        ax.add_patch(
            Ellipse(
                center,
                diameter[0],
                diameter[1],
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=TEAL,
                linewidth=0.7,
                alpha=alpha,
                zorder=2,
            )
        )
    points = [
        (0.070, 0.414),
        (0.093, 0.568),
        (0.120, 0.460),
        (0.148, 0.596),
        (0.165, 0.455),
        (0.190, 0.530),
        (0.215, 0.435),
        (0.240, 0.575),
        (0.258, 0.490),
    ]
    for index, point in enumerate(points):
        highlighted = index in {2, 4, 5, 6, 8}
        ax.add_patch(
            Circle(
                point,
                0.006 if highlighted else 0.0045,
                transform=ax.transAxes,
                facecolor=TEAL if highlighted else "#AEB8C1",
                edgecolor=WHITE,
                linewidth=0.5,
                zorder=4,
            )
        )
    ax.plot(
        center[0],
        center[1],
        marker="D",
        markersize=5.8,
        markerfacecolor=CORAL,
        markeredgecolor=WHITE,
        markeredgewidth=0.7,
        transform=ax.transAxes,
        zorder=5,
    )
    ax.text(
        center[0] + 0.011,
        center[1] - 0.006,
        "query",
        transform=ax.transAxes,
        fontsize=5.4,
        color=CORAL,
        va="center",
        zorder=5,
    )
    ax.text(
        x + 0.012,
        y + 0.020,
        "nearest labeled training samples",
        transform=ax.transAxes,
        fontsize=5.3,
        color=MUTED,
        va="bottom",
    )


def build_connected_figure() -> plt.Figure:
    """Build the left-to-right evidence, optimization and inference figure."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
        }
    )
    fig = plt.figure(figsize=(7.2, 4.15), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for x0, x1 in [(0.012, 0.310), (0.325, 0.650), (0.665, 0.988)]:
        ax.add_patch(
            Rectangle(
                (x0, 0.035),
                x1 - x0,
                0.885,
                transform=ax.transAxes,
                facecolor=PANEL,
                edgecolor="none",
                zorder=0,
            )
        )

    panel_heading(ax, "a", "Training-only local evidence", 0.020)
    panel_heading(ax, "b", "DAC residual-guided rules", 0.335)
    panel_heading(ax, "c", "Locked-test inference", 0.675)

    box(
        ax,
        0.035,
        0.735,
        0.112,
        0.105,
        "Locked split",
        "train | test",
        face=WHITE,
        edge=TEAL,
    )
    draw_split_icon(ax, 0.049, 0.752, 0.084, 0.012)
    box(
        ax,
        0.177,
        0.735,
        0.112,
        0.105,
        "Standardize",
        "fit on train only",
        face=WHITE,
        edge=TEAL,
    )
    arrow(ax, (0.147, 0.787), (0.177, 0.787), color=TEAL)
    arrow(ax, (0.233, 0.735), (0.175, 0.670), color=TEAL)
    draw_local_space(ax)
    box(
        ax,
        0.035,
        0.175,
        0.255,
        0.115,
        "Multi-scale summary",
        "Biochar: k = 10 / 20 / 30     DAC: k = 7 / 11 / 15",
        face=TEAL_LIGHT,
        edge=TEAL,
        title_color=TEAL,
        title_size=7.0,
        subtitle_size=5.45,
    )
    arrow(ax, (0.162, 0.355), (0.162, 0.290), color=TEAL)
    ax.text(
        0.035,
        0.092,
        "Test targets remain sealed.",
        transform=ax.transAxes,
        fontsize=5.5,
        color=MUTED,
        va="center",
    )
    ax.plot(
        [0.035, 0.290],
        [0.125, 0.125],
        transform=ax.transAxes,
        color=LINE,
        linewidth=0.7,
    )

    box(
        ax,
        0.340,
        0.765,
        0.295,
        0.080,
        "Diagnostic subset",
        "outer-training rows only",
        face=WHITE,
        edge=INK,
        title_size=6.6,
        subtitle_size=5.0,
    )
    box(
        ax,
        0.340,
        0.600,
        0.130,
        0.090,
        "Run prompt",
        "diagnostic inference",
        face=CORAL_LIGHT,
        edge=CORAL,
        title_color=CORAL,
        title_size=6.0,
        subtitle_size=4.9,
    )
    box(
        ax,
        0.340,
        0.425,
        0.130,
        0.090,
        "Mine residuals",
        "signed errors + local context",
        face=TEAL_LIGHT,
        edge=TEAL,
        title_color=TEAL,
        title_size=6.0,
        subtitle_size=4.6,
    )
    box(
        ax,
        0.505,
        0.425,
        0.130,
        0.090,
        "Screen new rules",
        "structure + evidence IDs",
        face=GOLD_LIGHT,
        edge=GOLD,
        title_color=GOLD,
        title_size=5.8,
        subtitle_size=4.5,
    )
    box(
        ax,
        0.505,
        0.600,
        0.130,
        0.090,
        "Update prompt",
        "accepted corrections",
        face=CORAL_LIGHT,
        edge=CORAL,
        title_color=CORAL,
        title_size=6.0,
        subtitle_size=4.9,
    )
    arrow(ax, (0.488, 0.765), (0.405, 0.690), color=INK, width=1.0)
    arrow(ax, (0.405, 0.600), (0.405, 0.515), color=TEAL)
    arrow(ax, (0.470, 0.470), (0.505, 0.470), color=GOLD)
    arrow(ax, (0.570, 0.515), (0.570, 0.600), color=CORAL)
    box(
        ax,
        0.340,
        0.250,
        0.295,
        0.095,
        "Frozen prompt template",
        "physical priors + accepted correction rules",
        face=CORAL_LIGHT,
        edge=CORAL,
        title_color=CORAL,
        title_size=6.6,
        subtitle_size=4.9,
    )
    ax.plot(
        [0.635, 0.645, 0.645, 0.630],
        [0.645, 0.645, 0.365, 0.345],
        transform=ax.transAxes,
        color=CORAL,
        linewidth=1.0,
        zorder=2,
    )
    arrow(ax, (0.630, 0.345), (0.615, 0.345), color=CORAL)
    ax.text(
        0.340,
        0.092,
        "Propose from residuals; freeze before test.",
        transform=ax.transAxes,
        fontsize=5.4,
        color=MUTED,
        va="center",
    )
    ax.plot(
        [0.340, 0.635],
        [0.125, 0.125],
        transform=ax.transAxes,
        color=LINE,
        linewidth=0.7,
    )

    box(
        ax,
        0.680,
        0.700,
        0.120,
        0.105,
        "Local evidence",
        "neighbors + statistics\nfrom panel a",
        face=TEAL_LIGHT,
        edge=TEAL,
        title_color=TEAL,
        title_size=5.9,
        subtitle_size=4.5,
    )
    box(
        ax,
        0.680,
        0.525,
        0.120,
        0.105,
        "Numerical anchor",
        "surrogate prediction\nor local statistic",
        face=NAVY_LIGHT,
        edge=NAVY,
        title_color=NAVY,
        title_size=5.8,
        subtitle_size=4.5,
    )
    box(
        ax,
        0.680,
        0.245,
        0.120,
        0.105,
        "Frozen prompt\ntemplate",
        "from panel b",
        face=CORAL_LIGHT,
        edge=CORAL,
        title_color=CORAL,
        title_size=5.1,
        subtitle_size=4.5,
    )
    arrow(ax, (0.635, 0.298), (0.680, 0.298), color=CORAL, width=1.4)
    box(
        ax,
        0.825,
        0.455,
        0.090,
        0.295,
        "Inference\nprompt",
        "local context\nfrozen template\nnumerical anchor\nunlabeled query",
        face=WHITE,
        edge=CORAL,
        title_color=CORAL,
        title_size=6.3,
        subtitle_size=4.9,
    )
    arrow(ax, (0.800, 0.753), (0.825, 0.690), color=TEAL)
    arrow(ax, (0.800, 0.578), (0.825, 0.585), color=NAVY)
    ax.plot(
        [0.800, 0.815, 0.815],
        [0.298, 0.298, 0.490],
        transform=ax.transAxes,
        color=CORAL,
        linewidth=1.2,
        zorder=2,
    )
    arrow(ax, (0.815, 0.490), (0.825, 0.490), color=CORAL)
    box(
        ax,
        0.935,
        0.490,
        0.045,
        0.225,
        "LLM",
        "in-context\ninference",
        face=CORAL_LIGHT,
        edge=CORAL,
        title_color=CORAL,
        title_size=7.0,
        subtitle_size=4.3,
    )
    arrow(ax, (0.915, 0.603), (0.935, 0.603), color=CORAL)
    box(
        ax,
        0.700,
        0.105,
        0.260,
        0.095,
        "Target property prediction",
        "task-defined scalar material property",
        face=WHITE,
        edge=CORAL,
        title_color=CORAL,
        title_size=6.6,
        subtitle_size=4.8,
    )
    arrow(ax, (0.958, 0.490), (0.900, 0.200), color=CORAL, connectionstyle="arc3,rad=0.10")

    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure = build_connected_figure()
    metadata = {"Creator": "LSR", "Date": None}
    svg_path = OUTPUT_DIR / "method-overview.svg"
    figure.savefig(
        svg_path,
        format="svg",
        facecolor=WHITE,
        metadata=metadata,
    )
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    figure.savefig(
        OUTPUT_DIR / "method-overview.png",
        format="png",
        dpi=600,
        facecolor=WHITE,
        metadata={"Software": "LSR"},
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
