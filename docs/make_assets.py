"""Generate the README figures.

Run from the repository root:

    uv run python docs/make_assets.py

Produces, under docs/assets/:

- leak-vs-causal-light.png and leak-vs-causal-dark.png: the same
  mean-reversion strategy on the same synthetic prices, once with a centered
  rolling mean (which reads future rows) and once with a trailing mean.
- leak-sharpe-worlds-light.png and leak-sharpe-worlds-dark.png: the Sharpe
  ratio of both variants across many seeded random walks, one dot per world.
- terminal-session.svg: real output of the command-line interface, captured by
  running it on a small example project in a temporary directory and rendered
  as an animated terminal session (CSS animations only, no scripts).

Everything is synthetic and seeded, so the numbers in the figures are
reproducible and carry no claim about real markets.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from animated_terminal import animated_terminal_svg

ASSETS = Path(__file__).resolve().parent / "assets"

THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "border": (0.043, 0.043, 0.043, 0.12),
        "chip": "#f3f2ee",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e6e5df",
        "axis": "#c3c2b7",
        "causal": "#2a78d6",
        "leaky": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "border": (1.0, 1.0, 1.0, 0.12),
        "chip": "#242423",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "causal": "#3987e5",
        "leaky": "#d95926",
    },
}

FONT = ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"]
WINDOW = 20
N_DAYS = 1000
N_WORLDS = 50
SEED = 0
DPI = 200


def simulate(seed: int = SEED) -> pd.DataFrame:
    """Synthetic random-walk prices: no edge exists by construction."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=N_DAYS)
    ret = pd.Series(rng.normal(0.0, 0.01, N_DAYS), index=idx)
    price = 100.0 * (1.0 + ret).cumprod()
    return pd.DataFrame({"price": price, "ret": ret})


def mean_reversion_strategy(df: pd.DataFrame, *, center: bool) -> pd.Series:
    """Short when price is above its 20-day mean, long when below.

    Positions are shifted by one bar before being applied to returns, so
    execution is honest in both variants. The only difference is ``center``.
    """
    mean = df["price"].rolling(WINDOW, center=center).mean()
    position = -np.sign(df["price"] - mean)
    return (position.shift(1) * df["ret"]).fillna(0.0)


def sharpe(r: pd.Series) -> float:
    sd = r.std()
    return float(np.sqrt(252) * r.mean() / sd) if sd > 0 else 0.0


def measured_horizon(df: pd.DataFrame) -> int:
    """Rows after t that the centered mean reads: first row whose value changes when
    the history is truncated at t."""
    full = df["price"].rolling(WINDOW, center=True).mean()
    p = N_DAYS - 50
    part = df["price"].iloc[:p].rolling(WINDOW, center=True).mean()
    diff = ~np.isclose(full.iloc[:p].to_numpy(), part.to_numpy(), equal_nan=True)
    first = int(np.argmax(diff)) if diff.any() else p
    return p - first


# --- shared figure chrome -------------------------------------------------------------


def new_card(width_in: float, height_in: float, t: dict[str, Any]) -> Any:
    """A figure drawn as a rounded card on a transparent background."""
    plt.rcParams.update({"font.family": FONT, "font.size": 11})
    fig = plt.figure(figsize=(width_in, height_in), dpi=DPI)
    fig.patch.set_alpha(0.0)
    card = FancyBboxPatch(
        (0.04, 0.04),
        width_in - 0.08,
        height_in - 0.08,
        boxstyle="round,pad=0,rounding_size=0.16",
        transform=fig.dpi_scale_trans,
        facecolor=t["surface"],
        edgecolor=t["border"],
        linewidth=1.0,
        zorder=-10,
    )
    fig.patches.append(card)
    return fig


def style_axes(ax: Any, t: dict[str, Any]) -> None:
    ax.set_facecolor("none")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=t["muted"], labelsize=9.5, length=0, pad=6)
    ax.set_axisbelow(True)


def heading(fig: Any, t: dict[str, Any], title: str, subtitle: str) -> None:
    fig.text(0.062, 0.90, title, fontsize=14, weight="bold", color=t["ink"], va="baseline")
    fig.text(0.062, 0.85, subtitle, fontsize=9.5, color=t["ink2"], va="top", linespacing=1.45)
    fig.text(
        0.955,
        0.062,
        "peekproof",
        fontsize=8.5,
        weight="bold",
        color=t["muted"],
        ha="right",
        va="baseline",
    )


def chip(ax: Any, t: dict[str, Any], x: Any, y: float, text: str, color: str) -> None:
    """A label chip next to a line end: colored dot plus text in ink."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(18, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=9.5,
        color=t["ink"],
        linespacing=1.5,
        bbox={
            "boxstyle": "round,pad=0.45,rounding_size=0.9",
            "fc": t["chip"],
            "ec": t["border"],
            "lw": 0.8,
        },
        zorder=6,
    )
    ax.scatter([x], [y], s=46, color=color, edgecolor=t["surface"], linewidths=1.8, zorder=7)


def gradient_fill(ax: Any, x: pd.DatetimeIndex, y: np.ndarray, color: str) -> None:
    """Area under a line with a vertical fade, clipped to the line."""
    rgba = to_rgba(color)
    cmap = LinearSegmentedColormap.from_list("fade", [(*rgba[:3], 0.0), (*rgba[:3], 0.28)])
    xnum = mdates.date2num(x.to_pydatetime())
    y0, y1 = float(min(0.0, np.nanmin(y))), float(np.nanmax(y))
    grad = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    img = ax.imshow(
        grad,
        extent=[xnum[0], xnum[-1], y0, y1],
        aspect="auto",
        cmap=cmap,
        origin="lower",
        zorder=1,
        interpolation="bicubic",
    )
    poly = ax.fill_between(x, 0.0, y, color="none", lw=0)
    img.set_clip_path(poly.get_paths()[0], transform=ax.transData)


# --- figure 1: equity curves ----------------------------------------------------------


def draw_equity(df: pd.DataFrame, theme_name: str) -> None:
    t = THEMES[theme_name]
    leaky = mean_reversion_strategy(df, center=True)
    causal = mean_reversion_strategy(df, center=False)
    curve_leaky = ((1 + leaky).cumprod() - 1).to_numpy() * 100
    curve_causal = ((1 + causal).cumprod() - 1).to_numpy() * 100
    s_leaky, s_causal = sharpe(leaky), sharpe(causal)
    horizon = measured_horizon(df)
    x = df.index

    fig = new_card(9.4, 4.9, t)
    ax = fig.add_axes((0.075, 0.16, 0.60, 0.55))
    style_axes(ax, t)
    gradient_fill(ax, x, curve_leaky, t["leaky"])
    ax.plot(x, curve_leaky, color=t["leaky"], lw=2.2, solid_capstyle="round", zorder=3)
    ax.plot(x, curve_causal, color=t["causal"], lw=2.2, solid_capstyle="round", zorder=3)
    chip(
        ax,
        t,
        x[-1],
        float(curve_leaky[-1]),
        f"centered window\nSharpe {s_leaky:.1f}  ·  reads {horizon} rows ahead",
        t["leaky"],
    )
    chip(
        ax,
        t,
        x[-1],
        float(curve_causal[-1]),
        f"trailing window\nSharpe {s_causal:.1f}",
        t["causal"],
    )

    ax.yaxis.grid(True, color=t["grid"], lw=0.8)
    ax.xaxis.grid(False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}%"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.01)
    ax.set_ylim(bottom=min(-60.0, float(np.nanmin(curve_causal)) - 40))

    heading(
        fig,
        t,
        "Same strategy, same prices. One keyword argument.",
        f"Cumulative return of a 20-day mean-reversion rule on a synthetic random walk "
        f"({N_DAYS:,} days, seed {SEED}).\n"
        "Orange: rolling(20, center=True). Blue: rolling(20). Nothing else differs.",
    )
    fig.savefig(
        ASSETS / f"leak-vs-causal-{theme_name}.png", transparent=True, metadata={"Software": None}
    )
    plt.close(fig)
    print(
        f"{theme_name}: leaky Sharpe {s_leaky:.2f}, causal Sharpe {s_causal:.2f}, "
        f"horizon {horizon} rows, final return leaky {curve_leaky[-1]:.0f}% "
        f"causal {curve_causal[-1]:.0f}%"
    )


# --- figure 2: many worlds ------------------------------------------------------------


def draw_worlds(theme_name: str) -> None:
    """Sharpe ratio of the leaky and the causal variant over many synthetic worlds."""
    t = THEMES[theme_name]
    leaky_sharpes: list[float] = []
    causal_sharpes: list[float] = []
    for seed in range(N_WORLDS):
        df = simulate(seed)
        leaky_sharpes.append(sharpe(mean_reversion_strategy(df, center=True)))
        causal_sharpes.append(sharpe(mean_reversion_strategy(df, center=False)))

    fig = new_card(9.4, 3.4, t)
    ax = fig.add_axes((0.17, 0.22, 0.78, 0.50))
    style_axes(ax, t)
    rng = np.random.default_rng(0)
    rows = (
        ("with look-ahead", 1.0, leaky_sharpes, t["leaky"]),
        ("causal", 0.0, causal_sharpes, t["causal"]),
    )
    for _label, y, values, color in rows:
        jitter = rng.uniform(-0.17, 0.17, len(values))
        ax.scatter(
            values,
            y + jitter,
            s=52,
            color=color,
            alpha=0.92,
            edgecolor=t["surface"],
            linewidths=1.4,
            zorder=3,
        )
        med = float(np.median(values))
        ax.plot([med, med], [y - 0.34, y + 0.34], color=t["ink"], lw=1.6, zorder=4)
        ax.annotate(
            f"median {med:.1f}",
            xy=(med, y + 0.40),
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=t["ink"],
            bbox={
                "boxstyle": "round,pad=0.3,rounding_size=0.9",
                "fc": t["chip"],
                "ec": t["border"],
                "lw": 0.8,
            },
            zorder=5,
        )
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(["with look-ahead", "causal"], color=t["ink"], fontsize=10.5)
    ax.set_ylim(-0.6, 1.95)
    ax.set_xlim(-1.6, 6.4)
    ax.axvline(0, color=t["axis"], lw=1, zorder=1)
    ax.xaxis.grid(True, color=t["grid"], lw=0.8)
    ax.set_xlabel("annualized Sharpe ratio", color=t["muted"], fontsize=9.5, labelpad=8)
    heading(
        fig,
        t,
        f"The same rule in {N_WORLDS} random worlds with no edge at all.",
        f"One dot per seed. Same {N_DAYS:,}-day random walks, same 20-day mean-reversion rule; "
        "only the window is centered or trailing.",
    )
    fig.savefig(
        ASSETS / f"leak-sharpe-worlds-{theme_name}.png",
        transparent=True,
        metadata={"Software": None},
    )
    plt.close(fig)
    print(
        f"{theme_name}: worlds={N_WORLDS}, leaky Sharpe min/median/max "
        f"{min(leaky_sharpes):.2f}/{np.median(leaky_sharpes):.2f}/{max(leaky_sharpes):.2f}, "
        f"causal min/median/max {min(causal_sharpes):.2f}/{np.median(causal_sharpes):.2f}/"
        f"{max(causal_sharpes):.2f}"
    )


# --- figure 3: animated terminal session ----------------------------------------------

DEMO_FEATURES = """\
import pandas as pd


def smooth(prices: pd.DataFrame) -> pd.Series:
    return prices["close"].rolling(20, center=True).mean()


def trailing(prices: pd.DataFrame) -> pd.Series:
    return prices["close"].rolling(20).mean()
"""

DEMO_STRATEGY = """\
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["ret"] = df["close"].pct_change()
    df["mom_20"] = df["close"].pct_change(20)
    df["vol_20"] = df["ret"].rolling(20).std()
    df["smooth"] = df["close"].rolling(10, center=True).mean()
    df["gap"] = df["open"].shift(-1) / df["close"] - 1
    df["z_vol"] = (df["vol_20"] - df["vol_20"].mean()) / df["vol_20"].std()
    return df.bfill()


def forward_label(df: pd.DataFrame) -> pd.Series:
    return np.sign(df["ret"].shift(-1))


def fit(df: pd.DataFrame, model):
    X = df[["mom_20", "vol_20", "smooth", "gap", "z_vol"]]
    y = forward_label(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return model.fit(X_train, y_train)


def backtest(df: pd.DataFrame, signal: pd.Series) -> pd.Series:
    return (signal * df["ret"]).cumsum()
"""


def _run_cli(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "peekproof.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return (proc.stdout + proc.stderr).rstrip()


def _wrap_output(output: str, width: int) -> list[str]:
    """Wrap hint and over-long lines with a hanging indent so nothing is cut off."""
    lines: list[str] = []
    for line in output.splitlines():
        stripped = line.lstrip()
        indent = " " * (len(line) - len(stripped))
        if stripped.startswith(("hint:", "note:")):
            pieces = textwrap.wrap(
                stripped, width=width - len(indent) - 2, subsequent_indent=" " * 6
            )
            lines.extend(indent + piece for piece in pieces)
        elif len(line) > width - 2:
            pieces = textwrap.wrap(stripped, width=width - 2, subsequent_indent=" " * 4)
            lines.extend(indent + piece for piece in pieces)
        else:
            lines.append(line)
    return lines


def render_terminal_session(df: pd.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prices = pd.DataFrame(
            {
                "open": df["price"].shift(1).bfill().round(2),
                "close": df["price"].round(2),
            }
        )
        prices.rename_axis("date").to_csv(root / "prices.csv")
        (root / "features.py").write_text(DEMO_FEATURES, encoding="utf-8")
        (root / "strategy.py").write_text(DEMO_STRATEGY, encoding="utf-8")
        check_cmd = ["check", "features:smooth", "--data", "prices.csv", "--index-col", "date"]
        lint_cmd = ["lint", "strategy.py"]
        svg = animated_terminal_svg(
            [
                ("peekproof " + " ".join(check_cmd), _wrap_output(_run_cli(check_cmd, root), 104)),
                ("peekproof " + " ".join(lint_cmd), _wrap_output(_run_cli(lint_cmd, root), 104)),
            ]
        )
        (ASSETS / "terminal-session.svg").write_text(svg, encoding="utf-8")
        print("terminal-session.svg written")


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    df = simulate()
    for name in THEMES:
        draw_equity(df, name)
        draw_worlds(name)
    render_terminal_session(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
