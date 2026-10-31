"""SQLAlchemy models. SQLite by default (DATABASE_URL overrides, e.g. Postgres on Fly/Railway)."""
from __future__ import annotations

import datetime as dt
import os
import secrets

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

DATA_DIR = os.environ.get("LEAKPROOF_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
os.makedirs(os.path.join(DATA_DIR, "uploads"), exist_ok=True)
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{os.path.join(DATA_DIR, 'leakproof.db')}")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


PLANS = {
    "free": {"name": "Free", "audits_per_month": 3, "max_mb": 20, "narrative": False, "share": False, "api": False, "seats": 1, "price": 0},
    "pro": {"name": "Pro", "audits_per_month": 10**9, "max_mb": 500, "narrative": True, "share": True, "api": False, "seats": 1, "price": 29},
    "team": {"name": "Team", "audits_per_month": 10**9, "max_mb": 2000, "narrative": True, "share": True, "api": True, "seats": 5, "price": 99},
    "enterprise": {"name": "Enterprise", "audits_per_month": 10**9, "max_mb": 10000, "narrative": True, "share": True, "api": True, "seats": 10**6, "price": None},
}


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    plan = Column(String(32), default="free", nullable=False)
    api_key = Column(String(64), unique=True, index=True, default=lambda: "lp_" + secrets.token_urlsafe(32))
    stripe_customer_id = Column(String(64), index=True)
    stripe_subscription_id = Column(String(64))
    billing_provider = Column(String(16))                     # "stripe" | "lemonsqueezy"
    lemon_customer_id = Column(String(64), index=True)
    lemon_subscription_id = Column(String(64))
    billing_portal_url = Column(Text)
    workspace_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Team: members point at the owner
    created_at = Column(DateTime, default=now)
    audits = relationship("Audit", back_populates="user", cascade="all, delete-orphan")

    @property
    def limits(self):
        return PLANS.get(self.plan, PLANS["free"])


class Audit(Base):
    __tablename__ = "audits"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    name = Column(String(255), nullable=False)
    target = Column(String(255), nullable=False)
    status = Column(String(16), default="queued", nullable=False)  # queued | running | done | failed
    progress = Column(String(255), default="")
    error = Column(Text)
    train_path = Column(String(512))
    test_path = Column(String(512))
    n_rows = Column(Integer)
    n_test_rows = Column(Integer)
    score = Column(Integer)
    grade = Column(String(2))
    headline = Column(Text)
    report_json = Column(Text)
    share_token = Column(String(48), unique=True, index=True)
    is_demo = Column(Boolean, default=False)
    runtime_s = Column(Float)
    created_at = Column(DateTime, default=now, index=True)
    finished_at = Column(DateTime)
    user = relationship("User", back_populates="audits")


class Event(Base):
    """Minimal audit log (who ran what, when). Enterprise buyers ask for this."""
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)
    kind = Column(String(64), index=True)
    detail = Column(Text)
    created_at = Column(DateTime, default=now, index=True)


def init_db():
    Base.metadata.create_all(engine)
    _add_missing_columns()


def _add_missing_columns():
    """create_all() never alters existing tables: add columns introduced after a database was created."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in have:
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {col.type.compile(engine.dialect)}'))


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
