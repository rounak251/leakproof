"""Leakproof command line.

    leakproof audit train.csv --target y [--test test.csv] [--json r.json] [--html r.html]
    leakproof gate  train.csv --target y --min-score 80 [--fail-on critical]   # CI gate, exit 1 on failure
    leakproof demo  [outdir]                                                    # write the demo CSVs
    leakproof lint  train.py [notebook.ipynb ...] [--fail-on error|warn]       # test-set leakage in training code
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import pandas as pd

from .engine import audit, AuditConfig, SEVERITY_ORDER
from . import __version__

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
COLORS = {"critical": "\033[91m", "high": "\033[93m", "medium": "\033[94m", "low": "\033[96m", "info": "\033[90m"}


def _read(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".feather", ".ftr"):
        return pd.read_feather(path)
    if ext in (".tsv",):
        return pd.read_csv(path, sep="\t", low_memory=False)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext in (".json", ".jsonl"):
        return pd.read_json(path, lines=ext == ".jsonl")
    return pd.read_csv(path, low_memory=False)


def _color(sev: str, text: str, tty: bool) -> str:
    return f"{COLORS.get(sev, '')}{text}{RESET}" if tty else text


def print_report(r: dict, tty: Optional[bool] = None):
    tty = sys.stdout.isatty() if tty is None else tty
    m = r["meta"]
    b = lambda s: f"{BOLD}{s}{RESET}" if tty else s
    sev = m["findings_by_severity"]
    print()
    print(b(f"Leakproof score {r['score']}/100 ({r['grade']})") + f"  ·  {sev['critical']} critical  ·  {sev['high']} high  ·  {sev['medium']} medium  ·  {sev['low']} low")
    print(f"{m['n_rows']:,} rows × {m['n_cols']} cols" + (f", test {m['n_test_rows']:,} rows" if m.get("n_test_rows") else "")
          + f"  ·  task {m['task']}  ·  metric {m['metric']}  ·  {m['runtime_s']}s")
    print(r["headline"])
    print()
    for f in r["findings"]:
        print(f"{_color(f['severity'], f['severity'].upper().ljust(9), tty)} {f['title']}")
        print(f"          {DIM if tty else ''}{f['detail'][:300]}{RESET if tty else ''}")
        if f.get("fix"):
            for line in f["fix"].splitlines()[:4]:
                print(f"          > {line}")
        print()
    cv = r["sections"]["cv"]
    base = r["sections"]["baseline"]
    print(b("Validation"))
    print(f"  recommended split : {cv['recommended_split']}")
    print(f"  reason            : {cv['reason']}")
    print(f"  naive CV {m['metric']:<9}: {base['random_cv_all_features']}")
    print(f"  honest estimate   : {cv['honest_cv_estimate']}")
    n = r.get("narrative")
    if n and not n.get("error"):
        print()
        print(b("Claude's read"))
        print("  " + n["headline"])
        print()
        for s in n["summary"].split("\n"):
            if s.strip():
                print("  " + s.strip())
        print()
        for i, st in enumerate(n["steps"], 1):
            print(f"  {i}. {st['title']}: {st['why']}")
    print()


def _run_audit(args) -> dict:
    train = _read(args.train)
    test = _read(args.test) if args.test else None
    cfg = AuditConfig(target=args.target, sample_rows=args.sample, exclude=args.exclude or [], task=args.task)
    t = time.time()

    def prog(msg):
        if not args.quiet:
            print(f"  .. {msg} ({time.time() - t:.0f}s)", file=sys.stderr)

    r = audit(train, test, cfg, progress=prog)
    if not args.no_narrative and os.environ.get("ANTHROPIC_API_KEY"):
        prog("asking Claude for the fix plan")
        from .narrative import write_narrative
        r["narrative"] = write_narrative(r)
    return r


def _write_outputs(r: dict, args):
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(r, fh, indent=2)
    if args.html:
        from .render import render_html
        with open(args.html, "w") as fh:
            fh.write(render_html(r, title=os.path.basename(args.train)))


def cmd_audit(args) -> int:
    r = _run_audit(args)
    _write_outputs(r, args)
    if not args.quiet:
        print_report(r)
    if args.json or args.html:
        print(f"wrote {args.json or ''} {args.html or ''}".strip(), file=sys.stderr)
    return 0


def cmd_gate(args) -> int:
    r = _run_audit(args)
    _write_outputs(r, args)
    print_report(r)
    worst = min((SEVERITY_ORDER[f["severity"]] for f in r["findings"]), default=99)
    fail = r["score"] < args.min_score or worst <= SEVERITY_ORDER[args.fail_on]
    if fail:
        print(f"GATE FAILED: score {r['score']} < {args.min_score} or a finding at severity >= {args.fail_on}", file=sys.stderr)
        return 1
    print(f"GATE PASSED: score {r['score']} >= {args.min_score}", file=sys.stderr)
    return 0


def cmd_demo(args) -> int:
    from .demo import make_churn_demo
    tr, te = make_churn_demo()
    os.makedirs(args.outdir, exist_ok=True)
    tr.to_csv(os.path.join(args.outdir, "train.csv"), index=False)
    te.to_csv(os.path.join(args.outdir, "test.csv"), index=False)
    print(f"wrote {args.outdir}/train.csv and test.csv  (target column: churned)\n"
          f"try:  leakproof audit {args.outdir}/train.csv --test {args.outdir}/test.csv --target churned")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "lint":                     # static leakage check of a training script / notebook
        from .lint import main as lint_main
        return lint_main(argv[1:])
    p = argparse.ArgumentParser(prog="leakproof", description="Catch data leakage, label noise and broken CV splits before you train. "
                                "`leakproof lint train.py` checks training code for test-set leakage.")
    p.add_argument("--version", action="version", version=f"leakproof {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("train", help="train file (csv, tsv, parquet, feather, xlsx, json)")
        sp.add_argument("--target", required=True, help="target column name")
        sp.add_argument("--test", help="optional test file (no target needed)")
        sp.add_argument("--exclude", nargs="*", help="columns to ignore")
        sp.add_argument("--task", choices=["binary", "multiclass", "regression"], help="override task inference")
        sp.add_argument("--sample", type=int, default=20000, help="row cap for model-based checks (default 20000)")
        sp.add_argument("--json", help="write full report JSON here")
        sp.add_argument("--html", help="write a standalone HTML report here")
        sp.add_argument("--no-narrative", action="store_true", help="skip the Claude narrative even if ANTHROPIC_API_KEY is set")
        sp.add_argument("--quiet", action="store_true")

    a = sub.add_parser("audit", help="run the audit and print findings")
    common(a)
    a.set_defaults(fn=cmd_audit)
    g = sub.add_parser("gate", help="CI gate: exit 1 if the dataset fails your bar")
    common(g)
    g.add_argument("--min-score", type=int, default=80)
    g.add_argument("--fail-on", choices=list(SEVERITY_ORDER), default="critical", help="also fail on any finding at this severity or worse")
    g.set_defaults(fn=cmd_gate)
    d = sub.add_parser("demo", help="write the planted-leak demo dataset")
    d.add_argument("outdir", nargs="?", default="leakproof_demo")
    d.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
