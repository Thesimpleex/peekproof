"""Render a looping, animated terminal session as a self-contained SVG.

The animation uses CSS keyframes only (no scripts), so it plays wherever the
SVG is embedded as an image, including GitHub READMEs. Commands are typed
character by character, output lines appear one after another, and the whole
session loops.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"
CHAR_W = 7.85  # px per character at 13px monospace; used for cursor placement
LINE_H = 21
FONT_SIZE = 13
PAD_X = 22
TOP = 58
WIDTH = 940
MARGIN = 18  # room for the drop shadow

COLORS = {
    "bg": "#1a1a19",
    "fg": "#f3f2ec",
    "dim": "#898781",
    "prompt": "#898781",
    "rule": "#d95926",
    "high": "#e66767",
    "medium": "#c98500",
    "low": "#898781",
    "bad": "#e66767",
    "good": "#0ca30c",
    "title": "#c3c2b7",
}

TOKEN_RE = re.compile(r"(NOT CAUSAL|\bCAUSAL\b|\bPP\d{3}\b|\[high\]|\[medium\]|\[low\])")


@dataclass
class Segment:
    text: str
    fill: str = COLORS["fg"]
    bold: bool = False


@dataclass
class Line:
    segments: list[Segment]
    start: float  # seconds into the loop when the line becomes visible


@dataclass
class Session:
    duration: float
    lines: list[Line] = field(default_factory=list)
    typed: list[tuple[int, str, float, float]] = field(default_factory=list)
    # (row index, command text, typing start, typing end)
    cursors: list[tuple[int, int, float, float]] = field(default_factory=list)
    # (row index, column, visible from, visible to)


def _style_line(text: str) -> list[Segment]:
    stripped = text.lstrip()
    if stripped.startswith(("hint:", "note:")) or text.startswith(" " * 10):
        # hint lines and their hanging-indent continuations are dimmed
        return [Segment(text, COLORS["dim"])]
    bold_line = stripped.startswith("verdict:")
    segments: list[Segment] = []
    pos = 0
    for m in TOKEN_RE.finditer(text):
        if m.start() > pos:
            segments.append(Segment(text[pos : m.start()], bold=bold_line))
        tok = m.group(0)
        if tok == "NOT CAUSAL":
            segments.append(Segment(tok, COLORS["bad"], bold=True))
        elif tok == "CAUSAL":
            segments.append(Segment(tok, COLORS["good"], bold=True))
        elif tok.startswith("PP"):
            segments.append(Segment(tok, COLORS["rule"], bold=True))
        elif tok == "[high]":
            segments.append(Segment(tok, COLORS["high"]))
        elif tok == "[medium]":
            segments.append(Segment(tok, COLORS["medium"]))
        else:
            segments.append(Segment(tok, COLORS["low"]))
        pos = m.end()
    if pos < len(text):
        segments.append(Segment(text[pos:], bold=bold_line))
    return segments or [Segment("")]


def build_session(
    commands: list[tuple[str, list[str]]],
    *,
    type_speed: float = 0.045,
    think: float = 0.9,
    line_gap: float = 0.16,
    pause_between: float = 1.6,
    hold: float = 4.0,
) -> Session:
    """Lay out the timeline for a list of (command, output lines)."""
    session = Session(duration=0.0)
    t = 0.6
    row = 0
    for command, output in commands:
        prompt_row = row
        session.lines.append(
            Line([Segment("$ ", COLORS["prompt"]), Segment(command, bold=True)], start=t)
        )
        typing_end = t + type_speed * len(command)
        session.typed.append((prompt_row, command, t, typing_end))
        session.cursors.append((prompt_row, len(command) + 2, t, typing_end + think))
        row += 1
        t = typing_end + think
        for text in output:
            session.lines.append(Line(_style_line(text), start=t))
            t += line_gap
            row += 1
        t += pause_between
    session.duration = t + hold
    return session


def _pct(seconds: float, duration: float) -> float:
    return max(0.0, min(100.0, 100.0 * seconds / duration))


def render(session: Session, title: str = "peekproof") -> str:
    rows = len(session.lines)
    height = TOP + rows * LINE_H + 26
    d = session.duration
    css: list[str] = [
        f"text{{font-family:{FONT};font-size:{FONT_SIZE}px;white-space:pre}}",
        ".b{font-weight:700}",
    ]
    body: list[str] = []

    # window chrome with a soft drop shadow (drawn outside the translated content group)
    chrome: list[str] = [
        '<defs><filter id="shadow" x="-5%" y="-5%" width="110%" height="120%">'
        '<feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#000000" '
        'flood-opacity="0.28"/></filter></defs>',
        f'<rect x="{MARGIN + 0.5}" y="{MARGIN + 0.5}" width="{WIDTH - 1}" '
        f'height="{height - 1}" rx="12" fill="{COLORS["bg"]}" stroke="#ffffff" '
        'stroke-opacity="0.08" filter="url(#shadow)"/>',
    ]
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        chrome.append(
            f'<circle cx="{MARGIN + 24 + i * 20}" cy="{MARGIN + 21}" r="6.5" fill="{c}"/>'
        )
    chrome.append(
        f'<text x="{MARGIN + WIDTH / 2}" y="{MARGIN + 26}" text-anchor="middle" '
        f'fill="{COLORS["title"]}" style="font-family:system-ui,-apple-system,sans-serif;'
        f'font-size:13px;font-weight:600">{html.escape(title)}</text>'
    )

    # lines
    for i, line in enumerate(session.lines):
        y = TOP + i * LINE_H
        start = _pct(line.start, d)
        name = f"l{i}"
        css.append(
            f"@keyframes {name}{{0%,{start:.3f}%{{opacity:0}}"
            f"{start + 0.01:.3f}%,100%{{opacity:1}}}}"
        )
        css.append(f".{name}{{animation:{name} {d:.2f}s linear infinite}}")
        tspans = "".join(
            f'<tspan fill="{seg.fill}"{' class="b"' if seg.bold else ""}>'
            f"{html.escape(seg.text)}</tspan>"
            for seg in line.segments
        )
        body.append(f'<text class="{name}" x="{PAD_X}" y="{y}">{tspans}</text>')

    # typing: a clip rectangle that widens over the command's row
    for k, (row, command, t0, t1) in enumerate(session.typed):
        y = TOP + row * LINE_H - 15
        x0 = PAD_X + 2 * CHAR_W
        full = len(command) * CHAR_W + 4
        name = f"t{k}"
        a, b = _pct(t0, d), _pct(t1, d)
        css.append(
            f"@keyframes {name}{{0%,{a:.3f}%{{width:0px}}{b:.3f}%,100%{{width:{full:.1f}px}}}}"
        )
        css.append(
            f".{name}{{animation:{name} {d:.2f}s infinite;"
            f"animation-timing-function:steps({len(command)},end)}}"
        )
        body.append(
            f'<clipPath id="c{k}" clipPathUnits="userSpaceOnUse">'
            f'<rect class="{name}" x="{x0:.1f}" y="{y}" width="0" height="{LINE_H}"/></clipPath>'
        )
    # re-emit the command text clipped (the unclipped version is hidden under the clip)
    # We do this by wrapping the typed rows: simplest is to draw the command twice;
    # the base line text for typed rows is drawn without the command instead.
    for k, (row, command, _t0, _t1) in enumerate(session.typed):
        y = TOP + row * LINE_H
        body.append(
            f'<g clip-path="url(#c{k})"><text x="{PAD_X + 2 * CHAR_W:.1f}" y="{y}" '
            f'class="b" fill="{COLORS["fg"]}">{html.escape(command)}</text></g>'
        )

    # cursors: follow the typing, then blink while the command "runs"
    css.append("@keyframes blink{0%,49%{fill-opacity:1}50%,100%{fill-opacity:0}}")
    for k, ((row, col, t0, t1), (_r, command, _s, typing_end)) in enumerate(
        zip(session.cursors, session.typed, strict=True)
    ):
        y = TOP + row * LINE_H - 13
        x_start = PAD_X + 2 * CHAR_W
        shift = (col - 2) * CHAR_W
        name = f"k{k}"
        a, b, c = _pct(t0, d), _pct(typing_end, d), _pct(t1, d)
        css.append(
            f"@keyframes {name}{{0%,{a:.3f}%{{opacity:0;transform:translateX(0)}}"
            f"{a + 0.01:.3f}%{{opacity:1;transform:translateX(0);"
            f"animation-timing-function:steps({len(command)},end)}}"
            f"{b:.3f}%,{c:.3f}%{{opacity:1;transform:translateX({shift:.1f}px)}}"
            f"{c + 0.01:.3f}%,100%{{opacity:0;transform:translateX({shift:.1f}px)}}}}"
        )
        css.append(
            f".{name}{{animation:{name} {d:.2f}s linear infinite,blink 1s steps(1) infinite}}"
        )
        body.append(
            f'<rect class="{name}" x="{x_start:.1f}" y="{y}" width="{CHAR_W:.1f}" height="16" '
            f'fill="{COLORS["fg"]}" rx="1"/>'
        )

    style = "\n".join(css)
    total_w, total_h = WIDTH + 2 * MARGIN, height + 2 * MARGIN
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
        f'viewBox="0 0 {total_w} {total_h}" role="img" '
        'aria-label="Animated terminal session showing peekproof check and peekproof lint">'
    )
    inner = "\n".join(body)
    return (
        f"{head}\n<style>\n{style}\n</style>\n"
        + "\n".join(chrome)
        + f'\n<g transform="translate({MARGIN},{MARGIN})">\n{inner}\n</g>\n</svg>\n'
    )


def _typed_rows_without_command(session: Session) -> Session:
    """Drop the command text from typed rows; the clipped copy is the only visible one."""
    typed_rows = {row for row, *_ in session.typed}
    for i, line in enumerate(session.lines):
        if i in typed_rows:
            line.segments = [line.segments[0]]
    return session


def animated_terminal_svg(commands: list[tuple[str, list[str]]], title: str = "peekproof") -> str:
    session = build_session(commands)
    session = _typed_rows_without_command(session)
    return render(session, title=title)
