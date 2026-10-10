"""Self-contained HTML reports (charts embedded as base64 PNG)."""

from __future__ import annotations

import base64
import html
from collections.abc import Iterable, Sequence
from pathlib import Path

CSS = """
:root { --bg:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; --accent:#2a78d6;
        --bad:#b42323; --bad-bg:#fbe9e9; --warn:#8a5a00; --warn-bg:#fdf3dc; --ok:#1d6b3a; --ok-bg:#e5f4ea; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#1a1a19; --panel:#232322; --ink:#ffffff; --ink2:#c3c2b7; --line:#3a3a38; --accent:#3987e5;
  --bad:#ff8a8a; --bad-bg:#3a2020; --warn:#f2c46b; --warn-bg:#3a3020; --ok:#7fd69d; --ok-bg:#1f3427; } }
:root[data-theme="dark"] { --bg:#1a1a19; --panel:#232322; --ink:#ffffff; --ink2:#c3c2b7; --line:#3a3a38;
  --accent:#3987e5; --bad:#ff8a8a; --bad-bg:#3a2020; --warn:#f2c46b; --warn-bg:#3a3020; --ok:#7fd69d; --ok-bg:#1f3427; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:32px 0 8px; border-bottom:1px solid var(--line); padding-bottom:4px; }
.sub { color:var(--ink2); margin:0 0 16px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:16px 0; }
.tile { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
.tile .v { font-size:22px; font-weight:700; } .tile .k { color:var(--ink2); font-size:12px; }
.scroll { overflow-x:auto; } table { border-collapse:collapse; width:100%; font-size:12.5px; background:var(--panel); }
th, td { border-bottom:1px solid var(--line); padding:5px 8px; text-align:left; vertical-align:top; }
th { color:var(--ink2); font-weight:600; position:sticky; top:0; background:var(--panel); }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:6px; background:#fcfcfb; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px; }
.pill { display:inline-block; padding:1px 7px; border-radius:10px; font-size:11.5px; font-weight:600; }
.fail { color:var(--bad); background:var(--bad-bg); } .pass { color:var(--ok); background:var(--ok-bg); }
.warn { color:var(--warn); background:var(--warn-bg); } .muted { color:var(--ink2); }
.note { background:var(--panel); border-left:3px solid var(--accent); padding:8px 12px; margin:12px 0; }
ul.warnings li { color:var(--warn); }
"""


def esc(x: object) -> str:
    return html.escape("" if x is None else str(x))


def img(path: str | Path, alt: str) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{data}">'


def fmt(v: object, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:  # NaN
            return "n/a"
        return f"{v:.{digits}f}"
    return str(v)


def table(headers: Sequence[str], rows: Iterable[Sequence[object]], raw_cols: set[int] | None = None) -> str:
    raw_cols = raw_cols or set()
    out = [
        '<div class="scroll"><table><thead><tr>',
        *(f"<th>{esc(h)}</th>" for h in headers),
        "</tr></thead><tbody>",
    ]
    for r in rows:
        out.append("<tr>")
        for i, v in enumerate(r):
            if i in raw_cols:
                out.append(f"<td>{v}</td>")
            else:
                cls = ' class="num"' if isinstance(v, (int, float)) and not isinstance(v, bool) else ""
                out.append(f"<td{cls}>{esc(fmt(v))}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def tiles(items: Sequence[tuple[str, str]]) -> str:
    return (
        '<div class="tiles">'
        + "".join(
            f'<div class="tile"><div class="v">{esc(v)}</div><div class="k">{esc(k)}</div></div>'
            for k, v in items
        )
        + "</div>"
    )


def page(title: str, subtitle: str, body: str) -> str:
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)}</title>'
        f'<style>{CSS}</style></head><body><main><h1>{esc(title)}</h1><p class="sub">{esc(subtitle)}</p>'
        f"{body}</main></body></html>"
    )


def write(path: str | Path, content: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p
