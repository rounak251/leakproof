"""Render an audit report to standalone HTML (used by the CLI and for exports)."""
from __future__ import annotations

import os
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def jinja_env(extra_dirs=()) -> Environment:
    env = Environment(loader=FileSystemLoader([*extra_dirs, TEMPLATE_DIR]), autoescape=select_autoescape(["html"]))
    env.filters["pct"] = lambda v: "—" if v is None else f"{v:.1f}%"
    env.filters["num"] = lambda v: "—" if v is None else (f"{v:,}" if isinstance(v, int) else f"{v:.3f}" if isinstance(v, float) else str(v))
    return env


def render_html(report: Dict[str, Any], title: str = "Leakproof report", share_url: str = "") -> str:
    env = jinja_env()
    return env.get_template("report_standalone.html").render(report=report, title=title, share_url=share_url)
