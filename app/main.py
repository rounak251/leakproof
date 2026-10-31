"""Leakproof web app: FastAPI + Jinja + SQLite/Postgres + Stripe.

Run:  uvicorn app.main:app --reload
Env:  SECRET_KEY, ANTHROPIC_API_KEY, BASE_URL, DATABASE_URL, and ONE payment provider:
      Lemon Squeezy (merchant of record, pays out to India): LEMON_CHECKOUT_PRO, LEMON_CHECKOUT_PRO_YEAR,
        LEMON_CHECKOUT_TEAM, LEMON_WEBHOOK_SECRET, LEMON_STORE_URL, optional LEMON_VARIANTS="id:pro,id:team"
      or Stripe: STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET, STRIPE_PRICE_PRO, STRIPE_PRICE_PRO_YEAR, STRIPE_PRICE_TEAM
"""
from __future__ import annotations

import datetime as dt
import json
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import bcrypt
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, URLSafeTimedSerializer
from sqlalchemy import func
from sqlalchemy.orm import Session

from leakproof import __version__
from leakproof.cli import _read as read_table
from leakproof.demo import make_churn_demo
from leakproof.engine import AuditConfig, audit
from leakproof.lint import lint_source, notebook_source
from leakproof.narrative import write_narrative
from leakproof.render import jinja_env, render_html

from .db import DATA_DIR, PLANS, Audit, Event, SessionLocal, User, get_db, init_db, now

BASE_DIR = os.path.dirname(__file__)
SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICES = {"pro": os.environ.get("STRIPE_PRICE_PRO", ""), "pro_year": os.environ.get("STRIPE_PRICE_PRO_YEAR", ""),
                 "team": os.environ.get("STRIPE_PRICE_TEAM", "")}
LEMON_CHECKOUT = {"pro": os.environ.get("LEMON_CHECKOUT_PRO", ""), "pro_year": os.environ.get("LEMON_CHECKOUT_PRO_YEAR", ""),
                  "team": os.environ.get("LEMON_CHECKOUT_TEAM", "")}
LEMON_WEBHOOK_SECRET = os.environ.get("LEMON_WEBHOOK_SECRET", "")
LEMON_STORE_URL = os.environ.get("LEMON_STORE_URL", "").rstrip("/")
LEMON_VARIANTS = dict(x.split(":", 1) for x in os.environ.get("LEMON_VARIANTS", "").split(",") if ":" in x)
PAYMENTS = "lemonsqueezy" if any(LEMON_CHECKOUT.values()) else ("stripe" if STRIPE_SECRET_KEY else "")
WORKERS = int(os.environ.get("LEAKPROOF_WORKERS", "2"))

app = FastAPI(title="Leakproof", version=__version__, docs_url="/api/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = jinja_env([os.path.join(BASE_DIR, "templates")])
templates.globals.update(PLANS=PLANS, BASE_URL=BASE_URL, stripe_enabled=bool(PAYMENTS), billing_enabled=bool(PAYMENTS),
                         payments=PAYMENTS, version=__version__)
signer = URLSafeTimedSerializer(SECRET_KEY, salt="session")
executor = ThreadPoolExecutor(max_workers=WORKERS)

init_db()


# ----------------------------------------------------------------------------- helpers

def render(request: Request, name: str, **ctx) -> HTMLResponse:
    ctx.setdefault("request", request)
    ctx.setdefault("user", getattr(request.state, "user", None))
    return HTMLResponse(templates.get_template(name).render(**ctx))


def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def check_pw(pw: str, h: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), h.encode())
    except ValueError:
        return False


def set_session(resp: Response, user_id: int):
    resp.set_cookie("lp_session", signer.dumps({"uid": user_id}), httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30,
                    secure=BASE_URL.startswith("https"))


def log_event(db: Session, user_id: Optional[int], kind: str, detail: str = ""):
    db.add(Event(user_id=user_id, kind=kind, detail=detail[:2000]))
    db.commit()


@app.middleware("http")
async def load_user(request: Request, call_next):
    request.state.user = None
    raw = request.cookies.get("lp_session")
    if raw:
        try:
            data = signer.loads(raw, max_age=60 * 60 * 24 * 30)
            db = SessionLocal()
            try:
                request.state.user = db.get(User, data["uid"])
            finally:
                db.close()
        except BadSignature:
            pass
    return await call_next(request)


def current_user(request: Request) -> User:
    u = getattr(request.state, "user", None)
    if not u:
        raise HTTPException(status_code=303, headers={"Location": "/login?next=" + request.url.path})
    return u


def api_user(request: Request, db: Session) -> User:
    auth = request.headers.get("authorization", "")
    key = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
    if not key:
        raise HTTPException(401, "Missing API key. Send 'Authorization: Bearer <key>'.")
    u = db.query(User).filter(User.api_key == key).first()
    if not u:
        raise HTTPException(401, "Invalid API key.")
    if not workspace_owner(db, u).limits["api"]:
        raise HTTPException(402, "API access needs the Team plan.")
    return u


def workspace_owner(db: Session, user: User) -> User:
    return db.get(User, user.workspace_id) if user.workspace_id else user


def workspace_ids(db: Session, owner: User):
    return [owner.id] + [m.id for m in db.query(User.id).filter(User.workspace_id == owner.id).all()]


def audits_this_month(db: Session, owner: User) -> int:
    """Usage is an immutable ledger (events), counted across the whole workspace: deleting an audit does not refund it."""
    start = now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return db.query(func.count(Event.id)).filter(Event.user_id.in_(workspace_ids(db, owner)), Event.kind == "audit.created",
                                                 Event.created_at >= start).scalar()


def check_quota(db: Session, owner: User):
    if audits_this_month(db, owner) >= owner.limits["audits_per_month"]:
        raise HTTPException(402, f"The {owner.limits['name']} plan allows {owner.limits['audits_per_month']} audits per month. Upgrade to continue.")


# ----------------------------------------------------------------------------- audit worker

def run_audit_job(audit_id: int):
    db = SessionLocal()
    try:
        a = db.get(Audit, audit_id)
        if not a:
            return
        a.status, a.progress = "running", "loading files"
        db.commit()
        owner = workspace_owner(db, a.user)

        def prog(msg):
            s = SessionLocal()
            try:
                row = s.get(Audit, audit_id)
                row.progress = msg
                s.commit()
            finally:
                s.close()

        t0 = time.time()
        if a.is_demo:
            train, test = make_churn_demo()
        else:
            train = read_table(a.train_path)
            test = read_table(a.test_path) if a.test_path else None
        report = audit(train, test, AuditConfig(target=a.target), progress=prog)
        if owner.limits["narrative"] and os.environ.get("ANTHROPIC_API_KEY"):
            prog("asking Claude for the fix plan")
            report["narrative"] = write_narrative(report)
        a.report_json = json.dumps(report)
        a.score, a.grade, a.headline = report["score"], report["grade"], report["headline"]
        a.n_rows, a.n_test_rows = report["meta"]["n_rows"], report["meta"]["n_test_rows"]
        a.runtime_s = round(time.time() - t0, 1)
        a.status, a.progress, a.finished_at = "done", "", now()
        db.commit()
        # uploaded files are deleted once the report exists: we keep statistics, not data
        for p in (a.train_path, a.test_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
    except Exception as e:  # noqa: BLE001
        db.rollback()
        a = db.get(Audit, audit_id)
        if a:
            a.status, a.error = "failed", f"{type(e).__name__}: {e}"[:2000]
            db.commit()
            _cleanup(a.train_path, a.test_path)
    finally:
        db.close()


def enqueue(audit_id: int):
    executor.submit(run_audit_job, audit_id)


def create_audit(db: Session, user: User, name: str, target: str, train_path: Optional[str], test_path: Optional[str], is_demo=False) -> Audit:
    owner = workspace_owner(db, user)
    if not is_demo:
        check_quota(db, owner)
    a = Audit(user_id=user.id, name=name[:255], target=target[:255], train_path=train_path, test_path=test_path,
              is_demo=is_demo, share_token=secrets.token_urlsafe(24))
    db.add(a)
    db.commit()
    log_event(db, user.id, "audit.demo" if is_demo else "audit.created", f"{a.id} {name} target={target}")
    enqueue(a.id)
    return a


def check_target(path: str, target: str):
    """Fail fast if the target column is not in the file header."""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv", ".tsv"):
            cols = list(pd.read_csv(path, nrows=5, sep="\t" if ext == ".tsv" else None, engine="python").columns)
        else:
            cols = list(read_table(path).columns)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not read the file: {type(e).__name__}: {e}")
    if target not in [str(c) for c in cols]:
        raise HTTPException(400, f"Target column '{target}' not found. Columns: {', '.join(str(c) for c in cols[:25])}{'…' if len(cols) > 25 else ''}")


async def save_upload(up: UploadFile, user: User, max_mb: int) -> str:
    ext = os.path.splitext(up.filename or "data.csv")[1].lower() or ".csv"
    if ext not in (".csv", ".tsv", ".parquet", ".pq", ".feather", ".xlsx", ".xls", ".json", ".jsonl"):
        raise HTTPException(400, f"Unsupported file type {ext}. Use csv, tsv, parquet, feather, xlsx or json.")
    path = os.path.join(DATA_DIR, "uploads", f"u{user.id}_{secrets.token_hex(8)}{ext}")
    size = 0
    with open(path, "wb") as fh:
        while chunk := await up.read(1024 * 1024):
            size += len(chunk)
            if size > max_mb * 1024 * 1024:
                fh.close()
                os.remove(path)
                raise HTTPException(413, f"File exceeds the {max_mb} MB limit of your plan.")
            fh.write(chunk)
    return path


# ----------------------------------------------------------------------------- public pages

@app.get("/", response_class=HTMLResponse)
def landing(request: Request):
    path = os.path.join(BASE_DIR, "templates", "landing.html")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return RedirectResponse("/signup")


@app.get("/pricing")
def pricing():
    return RedirectResponse("/#pricing")


PRIVACY = [
    ("What we store", "Your email, a password hash, the statistics in each audit report (scores, column names, counts, the short lists of suspect rows shown in the report) and an audit log of actions on your account."),
    ("What we delete", "Uploaded files are deleted as soon as the report is generated, usually within a minute. Failed uploads are deleted within 24 hours. Deleting an audit deletes its report immediately."),
    ("What we never do", "We never train models on your data, never sell it, and never share it with third parties other than the processors below."),
    ("Processors", "Anthropic (report narrative; receives the statistical report, never raw rows beyond the suspect lists), Stripe (payments; receives your email), and our hosting provider (encrypted volumes)."),
    ("On-premises", "Enterprise customers run the same software inside their own network; in that case nothing is sent to us."),
    ("Your rights", "Email hello@leakproof.dev to export or delete everything we hold about you. We answer within 7 days."),
]
TERMS = [
    ("Service", "Leakproof analyses datasets you upload and produces a report. Reports are advisory: you remain responsible for your models and for how you use the findings."),
    ("Accounts and plans", "One person per account on Free and Pro; Team seats are for named colleagues. Plans renew monthly or yearly through Stripe and can be cancelled any time from Settings; cancellation stops future charges and does not refund the current period."),
    ("Acceptable use", "Do not upload data you are not allowed to process, and do not attempt to access other users' reports or to overload the service."),
    ("Availability", "We aim for 99.5% monthly uptime on paid plans but offer no SLA except under a signed Enterprise agreement."),
    ("Liability", "To the extent permitted by law, our total liability is limited to the fees you paid in the previous 12 months."),
    ("Changes", "We may update these terms; material changes are announced by email 14 days in advance."),
]


@app.get("/privacy", response_class=HTMLResponse)
def privacy(request: Request):
    return render(request, "legal.html", heading="Privacy policy", sections=PRIVACY)


@app.get("/terms", response_class=HTMLResponse)
def terms(request: Request):
    return render(request, "legal.html", heading="Terms of service", sections=TERMS)


@app.get("/healthz")
def healthz():
    return {"ok": True, "version": __version__}


@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    if request.state.user:
        return RedirectResponse("/app")
    return render(request, "auth.html", mode="signup", error=None)


@app.post("/signup")
def signup(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    email = email.strip().lower()
    if "@" not in email or len(password) < 8:
        return render(request, "auth.html", mode="signup", error="Use a real email and a password of at least 8 characters.")
    if db.query(User).filter(User.email == email).first():
        return render(request, "auth.html", mode="signup", error="That email already has an account. Log in instead.")
    u = User(email=email, password_hash=hash_pw(password))
    db.add(u)
    db.commit()
    log_event(db, u.id, "user.signup", email)
    resp = RedirectResponse("/app?welcome=1", status_code=303)
    set_session(resp, u.id)
    return resp


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if request.state.user:
        return RedirectResponse("/app")
    return render(request, "auth.html", mode="login", error=None)


@app.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...), next: str = Form("/app"), db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == email.strip().lower()).first()
    if not u or not check_pw(password, u.password_hash):
        return render(request, "auth.html", mode="login", error="Wrong email or password.")
    resp = RedirectResponse(next if next.startswith("/") else "/app", status_code=303)
    set_session(resp, u.id)
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie("lp_session")
    return resp


# ----------------------------------------------------------------------------- app pages

@app.get("/app", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    user = current_user(request)
    owner = workspace_owner(db, user)
    if user.workspace_id and owner.plan not in ("team", "enterprise"):
        user.workspace_id, user.plan = None, "free"  # owner downgraded: members fall back to Free
        db.merge(user)
        db.commit()
    audits = db.query(Audit).filter(Audit.user_id == user.id).order_by(Audit.created_at.desc()).limit(100).all()
    used = audits_this_month(db, owner)
    return render(request, "dashboard.html", audits=audits, used=used, owner=owner, welcome=request.query_params.get("welcome"))


MAX_CODE_BYTES = 2_000_000


def _lint_text(code: str, filename: str):
    if filename.endswith(".ipynb"):
        src, where = notebook_source(code)
        return lint_source(src, filename, True, where)
    return lint_source(code, filename or "pasted code")


@app.get("/app/lint", response_class=HTMLResponse)
def lint_page(request: Request):
    current_user(request)
    return render(request, "lint.html", report=None, code="", error=None)


@app.post("/app/lint", response_class=HTMLResponse)
async def lint_run(request: Request, code: str = Form(""), file: Optional[UploadFile] = File(None),
                   db: Session = Depends(get_db)):
    """Static leakage check of a training script or notebook. Free on every plan; the code is not stored."""
    user = current_user(request)
    name = "pasted code"
    if file is not None and file.filename:
        raw = await file.read(MAX_CODE_BYTES + 1)
        if len(raw) > MAX_CODE_BYTES:
            return render(request, "lint.html", report=None, code="", error="File is over 2 MB.")
        code, name = raw.decode("utf-8", errors="replace"), file.filename
    if not code.strip():
        return render(request, "lint.html", report=None, code="", error="Paste a script or upload a .py / .ipynb file.")
    try:
        rep = _lint_text(code, name)
    except ValueError:
        return render(request, "lint.html", report=None, code=code, error="That notebook file could not be read as JSON.")
    log_event(db, user.id, "lint.run", f"{name} errors={len(rep.errors)} warnings={len(rep.warnings)}")
    shown = code if not name.endswith(".ipynb") else ""
    return render(request, "lint.html", report=rep, code=shown, error=None, filename=name)


@app.post("/api/v1/lint")
async def api_lint(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    user = api_user(request, db)
    raw = await file.read(MAX_CODE_BYTES + 1)
    if len(raw) > MAX_CODE_BYTES:
        raise HTTPException(413, "file over 2 MB")
    rep = _lint_text(raw.decode("utf-8", errors="replace"), file.filename or "code.py")
    log_event(db, user.id, "lint.api", f"{file.filename} errors={len(rep.errors)}")
    return {"path": rep.path, "errors": len(rep.errors), "warnings": len(rep.warnings),
            "findings": [f.to_dict() for f in rep.findings],
            "fits": [{"line": x.line, "where": x.where, "call": x.call, "sees": x.sees} for x in rep.fits]}


@app.get("/app/new", response_class=HTMLResponse)
def new_audit_page(request: Request, db: Session = Depends(get_db)):
    user = current_user(request)
    owner = workspace_owner(db, user)
    return render(request, "new.html", owner=owner, used=audits_this_month(db, owner), error=None)


@app.post("/app/new")
async def new_audit(request: Request, train: UploadFile = File(...), test: Optional[UploadFile] = File(None),
                    target: str = Form(...), name: str = Form(""), db: Session = Depends(get_db)):
    user = current_user(request)
    owner = workspace_owner(db, user)
    train_path = test_path = None
    try:
        check_quota(db, owner)
        train_path = await save_upload(train, user, owner.limits["max_mb"])
        test_path = await save_upload(test, user, owner.limits["max_mb"]) if test and test.filename else None
        check_target(train_path, target.strip())
        a = create_audit(db, user, name or (train.filename or "audit"), target.strip(), train_path, test_path)
    except HTTPException as e:
        _cleanup(train_path, test_path)
        return render(request, "new.html", owner=owner, used=audits_this_month(db, owner), error=e.detail)
    return RedirectResponse(f"/app/audits/{a.id}", status_code=303)


def _cleanup(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


@app.post("/app/demo")
def run_demo(request: Request, db: Session = Depends(get_db)):
    user = current_user(request)
    a = create_audit(db, user, "Demo: customer churn (planted leaks)", "churned", None, None, is_demo=True)
    return RedirectResponse(f"/app/audits/{a.id}", status_code=303)


@app.get("/demo")
def demo_entry(request: Request):
    if request.state.user:
        return RedirectResponse("/app?demo=1")
    return RedirectResponse("/signup?demo=1")


def _get_audit(db: Session, user: User, audit_id: int) -> Audit:
    a = db.get(Audit, audit_id)
    if not a or a.user_id != user.id:
        raise HTTPException(404, "Audit not found")
    return a


@app.get("/app/audits/{audit_id}", response_class=HTMLResponse)
def audit_page(request: Request, audit_id: int, db: Session = Depends(get_db)):
    user = current_user(request)
    a = _get_audit(db, user, audit_id)
    owner = workspace_owner(db, user)
    report = json.loads(a.report_json) if a.report_json else None
    share_url = f"{BASE_URL}/r/{a.share_token}" if (owner.limits["share"] or a.is_demo) else ""
    return render(request, "audit.html", audit=a, report=report, share_url=share_url, owner=owner)


@app.get("/app/audits/{audit_id}/status")
def audit_status(request: Request, audit_id: int, db: Session = Depends(get_db)):
    user = current_user(request)
    a = _get_audit(db, user, audit_id)
    return {"status": a.status, "progress": a.progress, "error": a.error, "score": a.score, "grade": a.grade}


@app.get("/app/audits/{audit_id}/report.json")
def audit_json(request: Request, audit_id: int, db: Session = Depends(get_db)):
    user = current_user(request)
    a = _get_audit(db, user, audit_id)
    if not a.report_json:
        raise HTTPException(404, "Report not ready")
    return Response(a.report_json, media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="leakproof-{a.id}.json"'})


@app.get("/app/audits/{audit_id}/report.html")
def audit_html(request: Request, audit_id: int, db: Session = Depends(get_db)):
    """Standalone HTML (print to PDF from the browser). Pro feature."""
    user = current_user(request)
    a = _get_audit(db, user, audit_id)
    owner = workspace_owner(db, user)
    if not owner.limits["share"]:
        raise HTTPException(402, "Exports need the Pro plan.")
    if not a.report_json:
        raise HTTPException(404, "Report not ready")
    html = render_html(json.loads(a.report_json), title=a.name, share_url=f"{BASE_URL}/r/{a.share_token}")
    return HTMLResponse(html, headers={"Content-Disposition": f'inline; filename="leakproof-{a.id}.html"'})


@app.get("/app/audits/{audit_id}/evidence.html")
def audit_evidence(request: Request, audit_id: int, db: Session = Depends(get_db)):
    """Model Validation Evidence: a formal, print-ready record for model-risk / compliance reviewers."""
    user = current_user(request)
    a = _get_audit(db, user, audit_id)
    owner = workspace_owner(db, user)
    if not owner.limits["share"]:
        raise HTTPException(402, "Evidence exports need the Pro plan.")
    if not a.report_json:
        raise HTTPException(404, "Report not ready")
    return render(request, "evidence.html", audit=a, report=json.loads(a.report_json), owner=owner, generated=now())


@app.post("/app/audits/{audit_id}/delete")
def audit_delete(request: Request, audit_id: int, db: Session = Depends(get_db)):
    user = current_user(request)
    a = _get_audit(db, user, audit_id)
    db.delete(a)
    db.commit()
    log_event(db, user.id, "audit.deleted", str(audit_id))
    return RedirectResponse("/app", status_code=303)


@app.get("/app/settings", response_class=HTMLResponse)
def settings(request: Request, db: Session = Depends(get_db)):
    user = current_user(request)
    owner = workspace_owner(db, user)
    members = db.query(User).filter(User.workspace_id == owner.id).all() if owner.id == user.id else []
    events = db.query(Event).filter(Event.user_id == user.id).order_by(Event.created_at.desc()).limit(50).all()
    return render(request, "settings.html", owner=owner, members=members, events=events, msg=request.query_params.get("msg"))


@app.post("/app/settings/rotate-key")
def rotate_key(request: Request, db: Session = Depends(get_db)):
    user = current_user(request)
    user.api_key = "lp_" + secrets.token_urlsafe(32)
    db.merge(user)
    db.commit()
    log_event(db, user.id, "apikey.rotated")
    return RedirectResponse("/app/settings?msg=API+key+rotated", status_code=303)


@app.post("/app/settings/invite")
def invite(request: Request, email: str = Form(...), db: Session = Depends(get_db)):
    user = current_user(request)
    if user.workspace_id or user.limits["seats"] <= 1:
        raise HTTPException(402, "Invites need the Team plan.")
    n = db.query(func.count(User.id)).filter(User.workspace_id == user.id).scalar()
    if n + 1 >= user.limits["seats"]:
        return RedirectResponse("/app/settings?msg=Seat+limit+reached", status_code=303)
    email = email.strip().lower()
    m = db.query(User).filter(User.email == email).first()
    if not m:
        temp = secrets.token_urlsafe(10)
        m = User(email=email, password_hash=hash_pw(temp), workspace_id=user.id, plan="team")
        db.add(m)
        db.commit()
        log_event(db, user.id, "member.invited", email)
        return RedirectResponse(f"/app/settings?msg=Invited+{email}.+Temporary+password:+{temp}", status_code=303)
    if m.id == user.id or m.workspace_id or m.plan != "free" or m.stripe_customer_id or m.lemon_customer_id:
        return RedirectResponse("/app/settings?msg=That+account+already+belongs+to+a+workspace+or+plan.+Ask+them+to+contact+support.", status_code=303)
    m.workspace_id, m.plan = user.id, "team"
    db.commit()
    log_event(db, user.id, "member.added", email)
    return RedirectResponse(f"/app/settings?msg=Added+{email}", status_code=303)


# ----------------------------------------------------------------------------- public share

@app.get("/r/{token}", response_class=HTMLResponse)
def shared_report(request: Request, token: str, db: Session = Depends(get_db)):
    a = db.query(Audit).filter(Audit.share_token == token).first()
    if not a or not a.report_json:
        raise HTTPException(404, "Report not found")
    owner = workspace_owner(db, a.user)
    if not (owner.limits["share"] or a.is_demo):
        raise HTTPException(404, "Report not found")
    return render(request, "share.html", audit=a, report=json.loads(a.report_json), share_url=f"{BASE_URL}/r/{token}")


@app.get("/r/{token}/badge.svg")
def badge(token: str, db: Session = Depends(get_db)):
    a = db.query(Audit).filter(Audit.share_token == token).first()
    if not a or a.score is None:
        raise HTTPException(404)
    color = {"A": "#34d399", "B": "#a3e635", "C": "#fbbf24", "D": "#fb923c", "F": "#f87171"}[a.grade]
    label, value = "audited by Leakproof", f"{a.score}/100 · {a.grade}"
    w1, w2 = 6.5 * len(label) + 12, 6.5 * len(value) + 12
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w1+w2:.0f}" height="20" role="img" aria-label="{label}: {value}">
<rect rx="3" width="{w1+w2:.0f}" height="20" fill="#27272a"/><rect rx="3" x="{w1:.0f}" width="{w2:.0f}" height="20" fill="{color}"/>
<g fill="#fff" text-anchor="middle" font-family="Verdana,DejaVu Sans,sans-serif" font-size="11">
<text x="{w1/2:.0f}" y="14">{label}</text><text x="{w1+w2/2:.0f}" y="14" fill="#111">{value}</text></g></svg>"""
    return Response(svg, media_type="image/svg+xml", headers={"Cache-Control": "max-age=3600"})


# ----------------------------------------------------------------------------- REST API (Team plan)

@app.post("/api/v1/audits")
async def api_create(request: Request, train: UploadFile = File(...), test: Optional[UploadFile] = File(None),
                     target: str = Form(...), name: str = Form(""), db: Session = Depends(get_db)):
    user = api_user(request, db)
    owner = workspace_owner(db, user)
    train_path = test_path = None
    try:
        check_quota(db, owner)
        train_path = await save_upload(train, user, owner.limits["max_mb"])
        test_path = await save_upload(test, user, owner.limits["max_mb"]) if test and test.filename else None
        check_target(train_path, target.strip())
        a = create_audit(db, user, name or (train.filename or "api audit"), target.strip(), train_path, test_path)
    except HTTPException:
        _cleanup(train_path, test_path)
        raise
    return {"id": a.id, "status": a.status, "url": f"{BASE_URL}/app/audits/{a.id}", "poll": f"{BASE_URL}/api/v1/audits/{a.id}"}


@app.get("/api/v1/audits/{audit_id}")
def api_get(request: Request, audit_id: int, db: Session = Depends(get_db)):
    user = api_user(request, db)
    a = _get_audit(db, user, audit_id)
    out = {"id": a.id, "status": a.status, "progress": a.progress, "error": a.error, "score": a.score, "grade": a.grade,
           "headline": a.headline, "share_url": f"{BASE_URL}/r/{a.share_token}"}
    if a.report_json:
        out["report"] = json.loads(a.report_json)
    return out


@app.get("/api/v1/audits")
def api_list(request: Request, db: Session = Depends(get_db)):
    user = api_user(request, db)
    rows = db.query(Audit).filter(Audit.user_id == user.id).order_by(Audit.created_at.desc()).limit(200).all()
    return [{"id": a.id, "name": a.name, "status": a.status, "score": a.score, "grade": a.grade, "created_at": a.created_at.isoformat()} for a in rows]


# ----------------------------------------------------------------------------- billing

def _stripe():
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    return stripe


@app.post("/billing/checkout")
def checkout(request: Request, plan: str = Form(...), db: Session = Depends(get_db)):
    user = current_user(request)
    if PAYMENTS == "lemonsqueezy":
        base = LEMON_CHECKOUT.get(plan, "")
        if not base:
            return RedirectResponse("/app/settings?msg=This+plan+is+not+on+sale+yet.", status_code=303)
        from urllib.parse import urlencode
        q = urlencode({"checkout[custom][user_id]": str(user.id), "checkout[custom][plan]": plan.split("_")[0],
                       "checkout[email]": user.email})
        log_event(db, user.id, "billing.checkout", f"lemonsqueezy {plan}")
        return RedirectResponse(base + ("&" if "?" in base else "?") + q, status_code=303)
    if not STRIPE_SECRET_KEY or not STRIPE_PRICES.get(plan):
        return RedirectResponse("/app/settings?msg=Billing+is+not+configured+yet.+Set+STRIPE_*+env+vars.", status_code=303)
    stripe = _stripe()
    if not user.stripe_customer_id:
        c = stripe.Customer.create(email=user.email, metadata={"user_id": user.id})
        user.stripe_customer_id = c.id
        db.merge(user)
        db.commit()
    session = stripe.checkout.Session.create(
        customer=user.stripe_customer_id, mode="subscription",
        line_items=[{"price": STRIPE_PRICES[plan], "quantity": 1}],
        success_url=f"{BASE_URL}/app/settings?msg=Welcome+to+{plan.split('_')[0].title()}",
        cancel_url=f"{BASE_URL}/app/settings",
        allow_promotion_codes=True,
        metadata={"user_id": user.id, "plan": plan.split("_")[0]},
        subscription_data={"metadata": {"user_id": user.id, "plan": plan.split("_")[0]}},
    )
    return RedirectResponse(session.url, status_code=303)


@app.post("/billing/portal")
def portal(request: Request):
    user = current_user(request)
    if PAYMENTS == "lemonsqueezy":
        target = user.billing_portal_url or (f"{LEMON_STORE_URL}/billing" if LEMON_STORE_URL else "")
        return RedirectResponse(target or "/app/settings", status_code=303)
    if not STRIPE_SECRET_KEY or not user.stripe_customer_id:
        return RedirectResponse("/app/settings", status_code=303)
    stripe = _stripe()
    s = stripe.billing_portal.Session.create(customer=user.stripe_customer_id, return_url=f"{BASE_URL}/app/settings")
    return RedirectResponse(s.url, status_code=303)


@app.post("/billing/webhook")
async def webhook(request: Request, db: Session = Depends(get_db)):
    stripe = _stripe()
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception:
        raise HTTPException(400, "bad signature")
    obj = event["data"]["object"]
    obj = obj.to_dict_recursive() if hasattr(obj, "to_dict_recursive") else dict(obj)
    t = event["type"]
    if t in ("customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"):
        cust = obj.get("customer")
        user = db.query(User).filter(User.stripe_customer_id == cust).first()
        if user:
            active = obj.get("status") in ("active", "trialing", "past_due") and t != "customer.subscription.deleted"
            plan = (obj.get("metadata") or {}).get("plan", "pro")
            user.plan = plan if active else "free"
            user.stripe_subscription_id = obj.get("id") if active else None
            db.merge(user)
            db.commit()
            log_event(db, user.id, "billing." + t, f"plan={user.plan}")
    return JSONResponse({"received": True})


LEMON_ACTIVE = {"on_trial", "active", "past_due", "cancelled"}   # cancelled = still paid up until it expires


@app.post("/billing/lemon-webhook")
async def lemon_webhook(request: Request, db: Session = Depends(get_db)):
    """Lemon Squeezy subscription events -> user.plan. Signature: HMAC-SHA256 hex of the raw body (X-Signature)."""
    import hashlib
    import hmac
    raw = await request.body()
    if not LEMON_WEBHOOK_SECRET:
        raise HTTPException(400, "webhook secret not configured")
    digest = hmac.new(LEMON_WEBHOOK_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(digest, request.headers.get("x-signature", "")):
        raise HTTPException(400, "bad signature")
    try:
        payload = json.loads(raw)
    except ValueError:
        raise HTTPException(400, "bad payload")
    meta = payload.get("meta") or {}
    event = meta.get("event_name", "")
    if not event.startswith("subscription_"):
        return JSONResponse({"received": True})
    data = payload.get("data") or {}
    attrs = data.get("attributes") or {}
    custom = meta.get("custom_data") or {}
    user = None
    uid = str(custom.get("user_id", ""))
    if uid.isdigit():
        user = db.get(User, int(uid))
    if user is None and attrs.get("customer_id") is not None:
        user = db.query(User).filter(User.lemon_customer_id == str(attrs["customer_id"])).first()
    if user is None and attrs.get("user_email"):
        user = db.query(User).filter(User.email == str(attrs["user_email"]).strip().lower()).first()
    if user is None:
        log_event(db, None, "billing.lemon.unmatched", f"{event} customer={attrs.get('customer_id')}")
        return JSONResponse({"received": True, "matched": False})
    status = attrs.get("status", "")
    plan = LEMON_VARIANTS.get(str(attrs.get("variant_id")), "") or str(custom.get("plan", "") or "pro")
    plan = plan.split("_")[0] if plan.split("_")[0] in PLANS else "pro"
    active = status in LEMON_ACTIVE and event != "subscription_expired"
    user.plan = plan if active else "free"
    user.billing_provider = "lemonsqueezy"
    if attrs.get("customer_id") is not None:
        user.lemon_customer_id = str(attrs["customer_id"])
    user.lemon_subscription_id = str(data.get("id")) if active else None
    portal_url = (attrs.get("urls") or {}).get("customer_portal")
    if portal_url:
        user.billing_portal_url = portal_url
    db.merge(user)
    db.commit()
    log_event(db, user.id, "billing.lemon." + event, f"status={status} plan={user.plan}")
    return JSONResponse({"received": True, "matched": True, "plan": user.plan})


@app.exception_handler(HTTPException)
async def http_exc(request: Request, exc: HTTPException):
    if exc.status_code == 303 and exc.headers and "Location" in exc.headers:
        return RedirectResponse(exc.headers["Location"], status_code=303)
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
    html = templates.get_template("error.html").render(request=request, user=getattr(request.state, "user", None),
                                                        status=exc.status_code, detail=exc.detail)
    return HTMLResponse(html, status_code=exc.status_code)
