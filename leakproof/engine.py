"""
Leakproof audit engine.

Pure pandas / numpy / scikit-learn. No network. Given a training frame (and an
optional test frame) plus the target column name, it runs a battery of checks
and returns a JSON-serialisable report:

    {
      "score": 0-100, "grade": "A".."F",
      "meta": {...},
      "findings": [ {id, category, severity, title, detail, evidence, fix, penalty} ],
      "sections": { schema, duplicates, overlap, target_leakage, label_noise, shift, cv, baseline },
    }

Everything model-based runs on a row subsample (default 20k) so a 1M-row file
audits in about a minute on a laptop.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

# Column-name hints. These never decide a finding on their own; they raise or
# lower the bar for the statistical tests.
LEAK_NAME_HINTS = [
    "target", "label", "outcome", "result", "resolution", "churn", "fraud", "default",
    "paid", "payment_date", "closed", "resolved", "cancel", "refund", "returned",
    "_after", "post_", "final", "status", "approved", "delinq", "chargeoff", "write_off",
    "settled", "converted", "won", "lost", "verdict", "diagnosis", "death", "died",
    "survived", "readmit", "complication", "actual", "realized", "realised",
]
GROUP_NAME_HINTS = [
    "user", "customer", "client", "account", "patient", "subject", "person", "member",
    "session", "device", "machine", "vehicle", "car", "store", "shop", "site", "location",
    "household", "family", "company", "org", "merchant", "driver", "player", "team",
    "building", "sensor", "hospital", "doctor", "school", "student", "seller", "buyer",
    "group", "cluster", "batch", "lot", "well", "field", "plot", "farm", "molecule",
    "protein", "sequence", "video", "clip", "image_id", "photo", "speaker", "author",
    "article", "doc", "paper", "book", "query", "product", "sku", "item", "listing",
]
TIME_NAME_HINTS = ["date", "time", "timestamp", "datetime", "day", "month", "year", "week", "period", "ts", "created", "updated", "_at"]
ID_NAME_HINTS = ["id", "uuid", "guid", "key", "index", "idx", "no", "number", "ref", "code"]


@dataclass
class AuditConfig:
    target: str
    sample_rows: int = 20000          # cap for model-based checks
    max_features_modelled: int = 300  # single-feature leak scan cap
    cv_folds: int = 5
    random_state: int = 42
    leak_auc: float = 0.95            # single feature AUC above this -> leakage candidate
    leak_r2: float = 0.90
    shift_auc_warn: float = 0.70
    shift_auc_high: float = 0.85
    group_gap_warn: float = 0.02      # KFold minus GroupKFold score gap
    label_noise_max_listed: int = 50
    time_budget_s: float = 240.0      # soft budget; later stages shrink if exceeded
    exclude: List[str] = field(default_factory=list)
    task: Optional[str] = None        # "binary" | "multiclass" | "regression"; None = infer


@dataclass
class Finding:
    id: str
    category: str
    severity: str
    title: str
    detail: str
    penalty: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    fix: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["penalty"] = round(float(d["penalty"]), 1)
        return d


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _is_bool(s: pd.Series) -> bool:
    return pd.api.types.is_bool_dtype(s)


def _is_dt(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def _py(o):
    """Make numpy / pandas scalars JSON-safe."""
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    return o


def _name_has(name: str, hints: List[str]) -> bool:
    n = name.lower()
    return any(h in n for h in hints)


def _looks_like_id(name: str) -> bool:
    n = name.lower()
    toks = re.split(r"[^a-z0-9]+", n)
    return any(t in ID_NAME_HINTS for t in toks) or n.endswith("id") or n.endswith("_id")


def _naive(dt: Optional[pd.Series]) -> Optional[pd.Series]:
    if dt is not None and getattr(dt.dt, "tz", None) is not None:
        return dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt


def _parse_datetime(s: pd.Series) -> Optional[pd.Series]:
    """Return a tz-naive datetime series if the column is a date, else None."""
    return _naive(_parse_datetime_raw(s))


def _parse_datetime_raw(s: pd.Series) -> Optional[pd.Series]:
    if _is_dt(s):
        return s
    if not _is_num(s) and not _is_dt(s) and not _is_bool(s):
        sample = s.dropna().astype(str)
        if len(sample) == 0:
            return None
        sample = sample.sample(min(300, len(sample)), random_state=0)
        # reject pure numbers / short tokens
        if sample.str.match(r"^-?\d+(\.\d+)?$").mean() > 0.5:
            return None
        if sample.str.len().median() < 6:
            return None
        try:
            parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        except Exception:
            return None
        if parsed.notna().mean() >= 0.9:
            try:
                return pd.to_datetime(s.astype(str).where(s.notna(), None), errors="coerce", utc=True)
            except Exception:
                return None
    # integer epoch seconds / ms with a time-like name
    if _is_num(s) and _name_has(s.name or "", TIME_NAME_HINTS):
        v = s.dropna()
        if len(v) and v.between(9e8, 2e9).mean() > 0.95:
            return pd.to_datetime(s, unit="s", errors="coerce")
        if len(v) and v.between(9e11, 2e12).mean() > 0.95:
            return pd.to_datetime(s, unit="ms", errors="coerce")
    return None


def _row_hash(df: pd.DataFrame, cols: List[str], fuzzy: bool = False) -> pd.Series:
    """Stable per-row hash (vectorised). fuzzy=True rounds numerics to 3 significant
    figures and normalises strings so near-duplicates collide. NA is hashed as a
    distinct sentinel, not as the string 'nan'."""
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    frame = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c]
        if _is_num(s):
            v = s.astype(float)
            if fuzzy:
                with np.errstate(divide="ignore", invalid="ignore"):
                    mag = np.floor(np.log10(np.abs(v.values) + 1e-12))
                    scale = 10.0 ** (2 - mag)
                    v = pd.Series(np.round(v.values * scale) / scale, index=df.index)
            frame[c] = v  # NaN hashes to its own value in hash_pandas_object
        elif _is_dt(s):
            frame[c] = s
        else:
            v = s.astype(str)
            if fuzzy:
                v = v.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
            frame[c] = v.where(s.notna(), "\x00<NA>\x00")
    h = pd.util.hash_pandas_object(frame, index=False)
    return pd.Series(h.values, index=df.index)


def _encode_features(df: pd.DataFrame, cols: List[str], enc=None):
    """Numeric + ordinal-encoded categoricals, NaN preserved, for HistGradientBoosting.
    Categoricals with more than MAX_CAT levels keep their top levels and bucket the rest."""
    MAX_CAT = 200
    X = pd.DataFrame(index=df.index)
    cat_cols = []
    for c in cols:
        s = df[c]
        if _is_num(s):
            X[c] = s.astype(float)
        elif _is_dt(s):
            X[c] = (s - pd.Timestamp("1970-01-01")).dt.total_seconds().astype(float)
        elif _is_bool(s):
            X[c] = s.astype(float)
        else:
            cat_cols.append(c)
    if cat_cols:
        cats = df[cat_cols].astype(str).where(df[cat_cols].notna(), None)
        if enc is None:
            keep = {}
            for c in cat_cols:
                vc = cats[c].value_counts()
                keep[c] = set(vc.index[:MAX_CAT]) if len(vc) > MAX_CAT else None
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=np.nan)
            cats = _bucket_rare(cats, keep)
            oe.fit(cats)
            enc = (oe, keep)
        else:
            oe, keep = enc
            cats = _bucket_rare(cats, keep)
        arr = oe.transform(cats)
        for i, c in enumerate(cat_cols):
            X[c] = arr[:, i]
    X = X[cols]  # keep caller order so per-column flags line up
    is_cat = [c in cat_cols for c in X.columns]
    return X, enc, is_cat


def _bucket_rare(cats: pd.DataFrame, keep: Dict[str, Any]) -> pd.DataFrame:
    cats = cats.copy()
    for c, top in keep.items():
        if top is not None:
            m = cats[c].notna() & ~cats[c].isin(top)
            cats.loc[m, c] = "__other__"
    return cats


def _model(task: str, is_cat, fast: bool = False, seed: int = 42):
    kw = dict(max_iter=80 if fast else 200, learning_rate=0.1 if fast else 0.06, max_leaf_nodes=15 if fast else 31,
              min_samples_leaf=20, l2_regularization=1.0, random_state=seed,
              categorical_features=np.array(is_cat, dtype=bool) if any(is_cat) else None,
              early_stopping=False)
    if task == "regression":
        return HistGradientBoostingRegressor(**kw)
    return HistGradientBoostingClassifier(**kw)


def _score(task: str, y_true, pred) -> float:
    if task == "regression":
        return float(r2_score(y_true, pred))
    if task == "binary":
        return float(roc_auc_score(y_true, pred))
    # multiclass: macro OvR AUC
    try:
        return float(roc_auc_score(y_true, pred, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def _oof(task: str, X: pd.DataFrame, y: np.ndarray, is_cat, splitter, groups=None, fast=False, seed=42):
    """Out-of-fold predictions. Returns (oof, score)."""
    n = len(X)
    n_classes = int(len(np.unique(y))) if task != "regression" else 1
    oof = np.zeros((n, n_classes)) if task == "multiclass" else np.zeros(n)
    hit = np.zeros(n, dtype=bool)
    split_iter = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, va in split_iter:
        m = _model(task, is_cat, fast=fast, seed=seed)
        m.fit(X.iloc[tr], y[tr])
        if task == "regression":
            oof[va] = m.predict(X.iloc[va])
        elif task == "binary":
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        else:
            p = m.predict_proba(X.iloc[va])
            cls = m.classes_
            full = np.zeros((len(va), n_classes))
            full[:, cls.astype(int)] = p
            oof[va] = full
        hit[va] = True
    if not hit.all():  # TimeSeriesSplit never predicts the first chunk
        y_eval, p_eval = y[hit], oof[hit]
    else:
        y_eval, p_eval = y, oof
    return oof, _score(task, y_eval, p_eval), hit


def _metric_name(task: str) -> str:
    return {"binary": "AUC", "multiclass": "macro AUC", "regression": "R²"}[task]


def _chance(task: str) -> float:
    return 0.5 if task != "regression" else 0.0


# ----------------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------------

def audit(train: pd.DataFrame, test: Optional[pd.DataFrame], config: AuditConfig, progress=None) -> Dict[str, Any]:
    t0 = time.time()
    cfg = config
    rng = np.random.RandomState(cfg.random_state)
    findings: List[Finding] = []
    sections: Dict[str, Any] = {}

    def tick(msg: str):
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    cfg.target = str(cfg.target)
    cfg.exclude = [str(c) for c in cfg.exclude]
    train = train.copy()
    train.columns = [str(c) for c in train.columns]
    if cfg.target not in train.columns:
        raise ValueError(f"target column '{cfg.target}' not found in train (columns: {list(train.columns)[:20]}...)")
    dup_names = train.columns[train.columns.duplicated()].tolist()
    if dup_names:
        raise ValueError(f"duplicate column names after normalisation: {dup_names[:10]}")
    # cells holding lists / dicts are not hashable: stringify those columns
    for c in train.columns:
        if train[c].dtype == object:
            first = train[c].dropna().head(20)
            if len(first) and first.map(lambda v: isinstance(v, (list, dict, set, tuple))).any():
                train[c] = train[c].map(lambda v: json.dumps(v, default=str, sort_keys=True) if isinstance(v, (list, dict, set, tuple)) else v)
    if test is not None:
        test = test.copy()
        test.columns = [str(c) for c in test.columns]
        if cfg.target in test.columns and test[cfg.target].notna().mean() < 0.05:
            test = test.drop(columns=[cfg.target])

    y_raw = train[cfg.target]
    mask = y_raw.notna()
    dropped_target_na = int((~mask).sum())
    orig_pos = np.flatnonzero(mask.values)  # position in the caller's frame, for suspect rows
    train = train.loc[mask].reset_index(drop=True)
    y_raw = train[cfg.target]
    if len(train) == 0:
        raise ValueError("every row has a missing target")
    if len(train) < 50:
        raise ValueError(f"only {len(train)} rows with a target; Leakproof needs at least 50")
    if test is not None:
        test.columns = [str(c) for c in test.columns]
        for c in test.columns:
            if test[c].dtype == object:
                first = test[c].dropna().head(20)
                if len(first) and first.map(lambda v: isinstance(v, (list, dict, set, tuple))).any():
                    test[c] = test[c].map(lambda v: json.dumps(v, default=str, sort_keys=True) if isinstance(v, (list, dict, set, tuple)) else v)

    # ---------------- task type ----------------
    nun = y_raw.nunique()
    if cfg.task in ("binary", "multiclass", "regression"):
        task = cfg.task
        if task == "regression" and not _is_num(y_raw):
            raise ValueError("task=regression needs a numeric target")
        if task == "binary" and nun != 2:
            raise ValueError(f"task=binary but the target has {nun} distinct values")
    elif _is_num(y_raw) and nun > 20 and not (y_raw.dropna() % 1 == 0).all():
        task = "regression"
    elif _is_num(y_raw) and nun > 30:
        task = "regression"
    elif nun == 2:
        task = "binary"
    elif nun <= 30:
        task = "multiclass"
    else:
        task = "regression" if _is_num(y_raw) else "multiclass"

    if task == "regression":
        y = y_raw.astype(float).values
        classes = None
    else:
        classes = sorted(y_raw.unique().tolist(), key=lambda v: str(v))
        cls_map = {c: i for i, c in enumerate(classes)}
        y = y_raw.map(cls_map).astype(int).values

    # ---------------- schema profile ----------------
    tick("profiling columns")
    feature_cols = [c for c in train.columns if c != cfg.target and c not in cfg.exclude]
    profile = []
    id_cols, const_cols, dt_cols, text_cols, high_card_cols = [], [], [], [], []
    datetimes: Dict[str, pd.Series] = {}
    n = len(train)
    for c in feature_cols:
        s = train[c]
        nu = int(s.nunique(dropna=True))
        miss = float(s.isna().mean())
        kind = "numeric" if _is_num(s) else ("bool" if _is_bool(s) else "categorical")
        dt = _parse_datetime(s) if kind != "numeric" or _name_has(c, TIME_NAME_HINTS) else None
        if dt is not None and s.notna().any() and dt.notna().sum() / s.notna().sum() > 0.9:
            kind = "datetime"
            datetimes[c] = dt
            dt_cols.append(c)
        elif kind == "categorical":
            avg_len = float(s.dropna().astype(str).str.len().mean() or 0)
            if avg_len > 40 and nu > 0.5 * max(n, 1):
                kind = "text"
                text_cols.append(c)
        if nu <= 1:
            kind = "constant"
            const_cols.append(c)
        elif nu >= 0.98 * n and n > 50 and (kind in ("categorical", "numeric")) and (
            _looks_like_id(c) or kind == "categorical" or (kind == "numeric" and (s.dropna() % 1 == 0).all())
        ):
            # unique per row: an identifier, not a feature
            if kind == "categorical" or _looks_like_id(c) or (s.dropna().diff().dropna().abs().median() == 1 if kind == "numeric" else False):
                kind = "id"
                id_cols.append(c)
        if kind == "categorical" and nu > 0.5 * n and n > 200:
            high_card_cols.append(c)
        profile.append({"column": c, "kind": kind, "dtype": str(s.dtype), "missing_pct": round(100 * miss, 2),
                        "n_unique": nu, "example": _py(s.dropna().iloc[0]) if s.notna().any() else None,
                        "in_test": bool(test is not None and c in test.columns)})
    kinds = {p["column"]: p["kind"] for p in profile}
    # use parsed datetimes everywhere downstream (CSV input arrives as strings)
    for c, dt in datetimes.items():
        train[c] = dt
        if test is not None and c in test.columns:
            tdt = _parse_datetime(test[c])
            if tdt is not None:
                test[c] = tdt
    model_cols = [c for c in feature_cols if kinds[c] not in ("id", "constant", "text")]
    if test is not None:
        missing_in_test = [c for c in model_cols if c not in test.columns]
        if missing_in_test:
            findings.append(Finding(
                id="schema.missing_in_test", category="schema", severity="high",
                title=f"{len(missing_in_test)} train column(s) are absent from test",
                detail="Columns present in train but not in test cannot be used by a model at inference time. "
                       "If they are leaking the label in train, they also silently inflate your CV. "
                       f"Missing: {missing_in_test[:15]}",
                penalty=min(15, 3 * len(missing_in_test)),
                evidence={"columns": missing_in_test},
                fix=f"train = train.drop(columns={missing_in_test!r})"))
            model_cols = [c for c in model_cols if c in test.columns]
    if len(model_cols) > cfg.max_features_modelled:
        findings.append(Finding(id="schema.too_wide", category="schema", severity="info",
                                title=f"Only the first {cfg.max_features_modelled} of {len(model_cols)} feature columns were modelled",
                                detail="Counts and duplicate checks used every column. Raise max_features_modelled to model more.",
                                penalty=0, evidence={"n_model_columns": len(model_cols)}))
        model_cols = model_cols[: cfg.max_features_modelled]
    sections["schema"] = {"n_rows": n, "n_cols": len(train.columns), "n_test_rows": int(len(test)) if test is not None else None,
                          "task": task, "target": cfg.target, "classes": _py(classes) if classes is not None else None,
                          "target_summary": _py(y_raw.value_counts(normalize=True).head(10).round(4).to_dict()) if task != "regression"
                          else _py(y_raw.describe().round(4).to_dict()),
                          "dropped_rows_missing_target": dropped_target_na,
                          "columns": profile, "id_columns": id_cols, "constant_columns": const_cols,
                          "datetime_columns": dt_cols, "text_columns": text_cols, "model_columns": model_cols}
    if const_cols:
        findings.append(Finding(id="schema.constant", category="schema", severity="info",
                                title=f"{len(const_cols)} constant column(s)", detail=f"No information: {const_cols[:20]}",
                                penalty=0, evidence={"columns": const_cols}, fix=f"df = df.drop(columns={const_cols!r})"))
    if task != "regression":
        vc = y_raw.value_counts(normalize=True)
        if vc.iloc[0] > 0.95:
            findings.append(Finding(id="schema.imbalance", category="schema", severity="info",
                                    title=f"Severe class imbalance: majority class is {vc.iloc[0]*100:.1f}%",
                                    detail="Accuracy is meaningless here; use AUC / PR-AUC and stratified folds.", penalty=0,
                                    evidence={"majority_share": float(vc.iloc[0])},
                                    fix="from sklearn.model_selection import StratifiedKFold\ncv = StratifiedKFold(5, shuffle=True, random_state=42)"))

    # ---------------- duplicates ----------------
    tick("checking duplicates")
    dup_cols = [c for c in feature_cols if kinds[c] not in ("id", "constant")]
    h_exact = _row_hash(train, dup_cols) if dup_cols else pd.Series([], dtype=str)
    n_exact = int(h_exact.duplicated().sum()) if len(h_exact) else 0
    h_fuzzy = _row_hash(train, dup_cols, fuzzy=True) if dup_cols else pd.Series([], dtype=str)
    n_fuzzy = int(h_fuzzy.duplicated().sum()) if len(h_fuzzy) else 0
    # same features, different label
    conflict = 0
    if n_exact:
        g = pd.DataFrame({"h": h_exact, "y": y_raw.astype(str)}).groupby("h")["y"].nunique()
        conflict = int((g > 1).sum())
    sections["duplicates"] = {"exact_duplicate_rows": n_exact, "near_duplicate_rows": n_fuzzy,
                              "duplicate_groups_with_conflicting_labels": conflict,
                              "exact_pct": round(100 * n_exact / max(n, 1), 2), "near_pct": round(100 * n_fuzzy / max(n, 1), 2)}
    if n_exact / max(n, 1) > 0.005 or n_exact >= 20:
        pct = 100 * n_exact / n
        sev = "high" if pct > 5 else "medium"
        findings.append(Finding(
            id="dup.exact", category="duplicates", severity=sev,
            title=f"{n_exact:,} exact duplicate rows ({pct:.1f}% of train)",
            detail="Duplicates across folds let the model memorise a row in training and 'predict' its twin in "
                   "validation, so CV overstates generalisation. " +
                   (f"{conflict:,} duplicate groups carry conflicting labels, which is label noise." if conflict else ""),
            penalty=min(12, 2 + pct),
            evidence={"n": n_exact, "pct": round(pct, 2), "conflicting_groups": conflict},
            fix=f"feat = {dup_cols[:8]!r}  # + the rest of your feature columns\n"
                f"train = train.drop_duplicates(subset=feat, keep='first')\n"
                "# or keep them but use GroupKFold with groups = hash of the feature row"))
    elif n_fuzzy / max(n, 1) > 0.02 and n_fuzzy > n_exact + 20:
        pct = 100 * n_fuzzy / n
        findings.append(Finding(
            id="dup.near", category="duplicates", severity="medium",
            title=f"{n_fuzzy:,} near-duplicate rows ({pct:.1f}%) after rounding/normalising",
            detail="Rows that differ only in float precision, case or whitespace behave like duplicates for a model.",
            penalty=min(6, 1 + pct / 2), evidence={"n": n_fuzzy, "pct": round(pct, 2)},
            fix="train['_rowkey'] = train[feat].round(3).astype(str).agg('|'.join, axis=1)\n"
                "cv = GroupKFold(5); folds = cv.split(train, y, groups=train['_rowkey'])"))

    # ---------------- train / test overlap ----------------
    if test is not None:
        tick("checking train/test contamination")
        shared = [c for c in dup_cols if c in test.columns]
        ov = {}
        if shared and len(shared) < max(2, 0.5 * len(dup_cols)):
            ov = {"note": f"test shares only {len(shared)} of {len(dup_cols)} feature columns; row-copy check skipped", "shared_columns": shared}
            shared = []
        if shared:
            ht = _row_hash(test, shared)
            htr = set(_row_hash(train, shared))
            ex = int(ht.isin(htr).sum())
            htf = _row_hash(test, shared, fuzzy=True)
            htrf = set(_row_hash(train, shared, fuzzy=True))
            fz = int(htf.isin(htrf).sum())
            ov = {"exact_test_rows_in_train": ex, "near_test_rows_in_train": fz,
                  "exact_pct": round(100 * ex / max(len(test), 1), 2), "near_pct": round(100 * fz / max(len(test), 1), 2)}
            if ex / max(len(test), 1) > 0.002 or ex >= 10:
                pct = ov["exact_pct"]
                findings.append(Finding(
                    id="overlap.exact", category="contamination", severity="critical" if pct > 1 else "high",
                    title=f"{ex:,} test rows ({pct:.2f}%) are exact copies of train rows",
                    detail="Train/test contamination. Any score on these rows is memorisation, not generalisation. "
                           "On a competition this means a public/private shake-up; in production it means your "
                           "offline metric is fiction.",
                    penalty=min(30, 10 + pct * 2), evidence=ov,
                    fix="key = lambda d: d[shared].astype(str).agg('|'.join, axis=1)\n"
                        "leaked = test[key(test).isin(set(key(train)))]\n"
                        "# report your metric separately on `leaked` and on the clean remainder"))
            elif fz / max(len(test), 1) > 0.01:
                findings.append(Finding(
                    id="overlap.near", category="contamination", severity="medium",
                    title=f"{fz:,} test rows ({ov['near_pct']:.2f}%) are near-duplicates of train rows",
                    detail="Not byte-identical but equal after rounding/normalising. Treat as contamination.",
                    penalty=min(10, 3 + ov["near_pct"]), evidence=ov, fix="See the exact-overlap fix; use the fuzzy key."))
        # id overlap
        id_ov = {}
        for c in id_cols:
            if c in test.columns:
                inter = len(set(train[c].dropna()) & set(test[c].dropna()))
                id_ov[c] = {"shared_ids": inter, "pct_of_test": round(100 * inter / max(test[c].nunique(), 1), 2)}
                if inter > 0:
                    findings.append(Finding(
                        id=f"overlap.id.{c}", category="contamination", severity="high",
                        title=f"ID column '{c}' shares {inter:,} values between train and test",
                        detail="An identifier that appears in both splits usually means the same entity was split "
                               "across train and test. Check whether your rows are independent.",
                        penalty=8, evidence={c: id_ov[c]},
                        fix=f"both = set(train['{c}']) & set(test['{c}'])\nprint(len(both)); train[train['{c}'].isin(both)].head()"))
        ov["id_overlap"] = id_ov
        sections["overlap"] = ov
    else:
        sections["overlap"] = None

    # ---------------- subsample for model-based checks ----------------
    if n > cfg.sample_rows:
        idx = np.sort(rng.choice(n, cfg.sample_rows, replace=False))
    else:
        idx = np.arange(n)
    S = train.iloc[idx].reset_index(drop=True)
    yS = y[idx]
    # drop classes with < cv_folds examples in the subsample (StratifiedKFold needs them)
    if task != "regression":
        vc = pd.Series(yS).value_counts()
        ok = vc[vc >= cfg.cv_folds].index
        keep = np.isin(yS, ok)
        if keep.sum() < len(yS):
            S, yS, idx = S.loc[keep].reset_index(drop=True), yS[keep], idx[keep]
        kept = np.unique(yS)
        if len(kept) < 2:
            raise ValueError(f"fewer than 2 classes have at least {cfg.cv_folds} rows; cannot cross-validate")
        if len(kept) < len(classes):
            remap = {old: new for new, old in enumerate(kept)}
            yS = np.array([remap[v] for v in yS])
            classes = [classes[int(k)] for k in kept]
            if len(classes) == 2:
                task = "binary"
    nS = len(S)
    if task != "regression":
        cfg.cv_folds = int(max(2, min(cfg.cv_folds, pd.Series(yS).value_counts().min())))
    splitter = (StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.random_state) if task != "regression"
                else KFold(cfg.cv_folds, shuffle=True, random_state=cfg.random_state))
    XS, enc, is_catS = _encode_features(S, model_cols)

    # ---------------- full-model baseline (random CV) ----------------
    tick("fitting baseline model")
    if len(model_cols) == 0:
        raise ValueError("no usable feature columns")
    oof_full, score_full, _ = _oof(task, XS, yS, is_catS, splitter, seed=cfg.random_state)
    metric = _metric_name(task)

    # ---------------- single-feature target leakage scan ----------------
    tick("scanning features for target leakage")
    leak_rows = []
    scan_errors: List[Dict[str, str]] = []
    scan_cols = model_cols[: cfg.max_features_modelled]
    fast_split = (StratifiedKFold(3, shuffle=True, random_state=cfg.random_state) if task != "regression"
                  else KFold(3, shuffle=True, random_state=cfg.random_state))
    budget_t = time.time()
    for i, c in enumerate(scan_cols):
        if time.time() - budget_t > cfg.time_budget_s * 0.45 and i > 20:
            break
        Xc = XS[[c]]
        if Xc[c].notna().sum() < 30 or Xc[c].nunique() <= 1:
            continue
        try:
            _, sc, _ = _oof(task, Xc, yS, [is_catS[model_cols.index(c)]], fast_split, fast=True, seed=cfg.random_state)
        except Exception as e:  # noqa: BLE001
            scan_errors.append({"column": c, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        # deterministic mapping feature -> target?
        det = False
        nu_c = int(S[c].nunique())
        if task != "regression" and 1 < nu_c < 0.5 * nS:
            g = pd.DataFrame({"f": S[c].astype(str), "y": yS}).groupby("f")["y"].nunique()
            det = bool((g <= 1).mean() > 0.98)
        # missingness tied to target?
        miss_auc = None
        if task == "binary" and 0.01 < S[c].isna().mean() < 0.99:
            miss_auc = float(roc_auc_score(yS, S[c].isna().astype(int)))
            miss_auc = max(miss_auc, 1 - miss_auc)
        leak_rows.append({"column": c, "single_feature_score": round(sc, 4), "deterministic_map": det,
                          "missingness_auc": round(miss_auc, 4) if miss_auc else None,
                          "name_hint": _name_has(c, LEAK_NAME_HINTS), "n_unique": nu_c})
    leak_rows.sort(key=lambda r: -(r["single_feature_score"] if r["single_feature_score"] is not None else -1))
    thr = cfg.leak_auc if task != "regression" else cfg.leak_r2
    chance = _chance(task)
    leaky = []
    for r in leak_rows:
        sc = r["single_feature_score"]
        strong = sc is not None and sc >= thr
        near_full = sc is not None and score_full > chance and (sc - chance) >= 0.97 * (score_full - chance) and (score_full - chance) > 0.3
        r["carries_all_signal"] = bool(near_full)
        if strong or r["deterministic_map"] or (r["missingness_auc"] and r["missingness_auc"] > 0.9) or near_full:
            reasons = []
            if strong:
                reasons.append(f"alone it reaches {metric} {sc:.3f} (full model {score_full:.3f})")
            if r["deterministic_map"]:
                reasons.append("every value of it maps to exactly one label")
            if r["missingness_auc"] and r["missingness_auc"] > 0.9:
                reasons.append(f"whether it is missing predicts the label (AUC {r['missingness_auc']:.2f})")
            if near_full and not strong:
                reasons.append(f"on its own it matches the full {len(model_cols)}-feature model ({sc:.3f} vs {score_full:.3f})")
            if r["name_hint"]:
                reasons.append("its name suggests it is recorded after the outcome")
            r["reasons"] = reasons
            leaky.append(r)
    sections["target_leakage"] = {"metric": metric, "full_model_score": round(score_full, 4), "threshold": thr,
                                  "features": leak_rows[:60], "flagged": [r["column"] for r in leaky], "errors": scan_errors}
    for r in leaky[:10]:
        corroborated = r["name_hint"] or r["deterministic_map"] or (r["missingness_auc"] and r["missingness_auc"] > 0.9)
        sev = "critical" if corroborated and ((r["single_feature_score"] or 0) >= thr or r["deterministic_map"]
                                              or r.get("carries_all_signal")) else "high"
        findings.append(Finding(
            id=f"leak.{r['column']}", category="target_leakage", severity=sev,
            title=f"'{r['column']}' looks like target leakage",
            detail="; ".join(r["reasons"]).capitalize() + ". A feature this predictive is almost always derived from "
                   "the outcome (recorded after it, or computed from it). It will not exist, or not hold the same "
                   "value, at prediction time.",
            penalty=25 if sev == "critical" else 12,
            evidence={k: v for k, v in r.items() if k != "reasons"},
            fix=f"# verify: is '{r['column']}' known BEFORE the outcome? If not:\n"
                f"train = train.drop(columns=['{r['column']}'])\n"
                f"# then re-run CV; expect the score to drop toward the honest level"))
    name_only = [r["column"] for r in leak_rows if r["name_hint"] and r["column"] not in {l["column"] for l in leaky}]
    if name_only:
        findings.append(Finding(id="leak.name_hints", category="target_leakage", severity="low",
                                title=f"{len(name_only)} column name(s) hint at post-outcome data",
                                detail=f"Statistically they look fine, but double-check when they are populated: {name_only[:10]}",
                                penalty=1, evidence={"columns": name_only[:20]},
                                fix="Ask: would this value be known at the moment I need the prediction?"))

    # honest baseline without flagged leak columns
    clean_cols = [c for c in model_cols if c not in {l["column"] for l in leaky}]
    score_clean, oof_clean = score_full, oof_full
    if leaky and clean_cols:
        Xc, _, is_cat_c = _encode_features(S, clean_cols)
        oof_clean, score_clean, _ = _oof(task, Xc, yS, is_cat_c, splitter, seed=cfg.random_state)
        sections["target_leakage"]["score_without_flagged"] = round(score_clean, 4)
        sections["target_leakage"]["score_drop"] = round(score_full - score_clean, 4)
    elif leaky and not clean_cols:
        score_clean = _chance(task)
        sections["target_leakage"]["score_without_flagged"] = None
        sections["target_leakage"]["note"] = "every feature was flagged; no honest model can be fitted"

    # ---------------- label noise ----------------
    tick("estimating label noise")
    noise = {}
    if task != "regression":
        cols_for_noise = clean_cols if clean_cols else model_cols
        Xn, _, is_cat_n = _encode_features(S, cols_for_noise)
        oof_n, _, _ = _oof(task, Xn, yS, is_cat_n, splitter, seed=cfg.random_state + 1)
        if task == "binary":
            P = np.stack([1 - oof_n, oof_n], axis=1)
        else:
            P = oof_n
        K = P.shape[1]
        # confident-learning style: per-class threshold = mean self-confidence of rows labelled that class
        thr_k = np.array([P[yS == k, k].mean() if (yS == k).any() else 1.0 for k in range(K)])
        given_conf = P[np.arange(nS), yS]
        P_other = P.copy()
        P_other[np.arange(nS), yS] = -1
        best_other = P_other.argmax(1)
        best_other_p = P_other.max(1)
        strong_thr = np.maximum(thr_k, 0.85)
        suspect = (given_conf < thr_k[yS]) & (best_other_p >= strong_thr[best_other]) & (best_other_p > given_conf)
        n_sus = int(suspect.sum())
        rate = n_sus / nS
        order = np.argsort(-(best_other_p - given_conf))
        listed = []
        for i in order[: cfg.label_noise_max_listed]:
            if not suspect[i]:
                break
            row = {"row_index": int(orig_pos[idx[i]]),
                   "given_label": _py(classes[yS[i]]), "suggested_label": _py(classes[best_other[i]]),
                   "p_given": round(float(given_conf[i]), 3), "p_suggested": round(float(best_other_p[i]), 3)}
            for c in id_cols[:1]:
                row["id"] = _py(S[c].iloc[i])
            listed.append(row)
        noise = {"estimated_noise_rate": round(rate, 4), "n_suspect_in_sample": n_sus, "sample_size": nS,
                 "class_thresholds": _py({str(classes[k]): round(float(thr_k[k]), 3) for k in range(K)}),
                 "suspects": listed}
        if rate > 0.05 and n_sus >= 10:
            sev = "high" if rate > 0.12 else "medium"
            findings.append(Finding(
                id="noise.labels", category="label_noise", severity=sev,
                title=f"{rate*100:.1f}% of labels contradict a confident model ({n_sus:,} rows in a {nS:,}-row sample)",
                detail="Rows where an out-of-fold model is at least 85% sure the label is a different class. "
                       "Some of this is irreducible noise in the problem, but at this rate it usually includes "
                       "mislabelled rows. Noisy labels cap your achievable score and make CV jumpy between seeds. "
                       "The top suspects are listed in the report; eyeball 20 of them before deciding.",
                penalty=min(8, 25 * rate),
                evidence={"rate": round(rate, 4), "n_suspect": n_sus, "top_suspects": listed[:10]},
                fix="# inspect, then either relabel, drop, or down-weight:\nsuspect_idx = [...]  # from the report\n"
                    "w = np.ones(len(train)); w[suspect_idx] = 0.2\nmodel.fit(X, y, sample_weight=w)"))
    else:
        resid = yS - oof_full
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
        z = 0.6745 * (resid - np.median(resid)) / mad
        sus = np.where(np.abs(z) > 4)[0]
        order = sus[np.argsort(-np.abs(z[sus]))]
        listed = [{"row_index": int(orig_pos[idx[i]]), "target": round(float(yS[i]), 4), "oof_prediction": round(float(oof_full[i]), 4),
                   "robust_z": round(float(z[i]), 2)} for i in order[: cfg.label_noise_max_listed]]
        rate = len(sus) / nS
        noise = {"estimated_outlier_rate": round(rate, 4), "n_suspect_in_sample": int(len(sus)), "sample_size": nS, "suspects": listed}
        if rate > 0.01 and len(sus) >= 10:
            findings.append(Finding(
                id="noise.targets", category="label_noise", severity="medium" if rate < 0.05 else "high",
                title=f"{len(sus):,} targets ({rate*100:.1f}%) are extreme outliers against an honest model",
                detail="Robust z-score of the out-of-fold residual above 4. Typical causes: unit mix-ups, sentinel values "
                       "(-1, 9999), or mis-keyed entries. They dominate squared-error losses.",
                penalty=min(8, 60 * rate), evidence={"rate": round(rate, 4), "top": listed[:10]},
                fix="train = train[~train.index.isin(suspect_idx)]  # or clip / winsorise the target\n"
                    "# or train on log1p(target) / use a Huber loss"))
    sections["label_noise"] = noise

    # ---------------- distribution shift ----------------
    shift = None
    if test is not None:
        tick("measuring train/test shift (adversarial validation)")
        shared_model = [c for c in model_cols if c in test.columns]
        if shared_model:
            nt = min(len(test), cfg.sample_rows)
            T = test.sample(nt, random_state=cfg.random_state) if len(test) > nt else test
            both = pd.concat([S[shared_model], T[shared_model]], ignore_index=True)
            lab = np.r_[np.zeros(len(S), dtype=int), np.ones(len(T), dtype=int)]
            Xb, _, is_cat_b = _encode_features(both, shared_model)
            adv_split = StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
            oof_adv, adv_auc, _ = _oof("binary", Xb, lab, is_cat_b, adv_split, fast=True, seed=cfg.random_state)
            # per-feature drift
            per = []
            for c in shared_model:
                a, b = S[c], T[c]
                if kinds[c] == "datetime":
                    a = (_parse_datetime(a) if not _is_dt(a) else a)
                    b = (_parse_datetime(b) if not _is_dt(b) else b)
                    if a is None or b is None:
                        continue
                    a = (a - pd.Timestamp("1970-01-01")).dt.total_seconds()
                    b = (b - pd.Timestamp("1970-01-01")).dt.total_seconds()
                if _is_num(a) and _is_num(b):
                    aa, bb = a.dropna(), b.dropna()
                    if len(aa) > 10 and len(bb) > 10:
                        ks = float(stats.ks_2samp(aa, bb).statistic)
                        per.append({"column": c, "stat": "KS", "value": round(ks, 4),
                                    "train_mean": round(float(aa.mean()), 4), "test_mean": round(float(bb.mean()), 4)})
                else:
                    pa = a.astype(str).value_counts(normalize=True)
                    pb = b.astype(str).value_counts(normalize=True)
                    cats = pa.index.union(pb.index)
                    p, q = pa.reindex(cats, fill_value=1e-6).values, pb.reindex(cats, fill_value=1e-6).values
                    psi = float(np.sum((p - q) * np.log(p / q)))
                    unseen = float(pb[~pb.index.isin(pa.index)].sum())
                    per.append({"column": c, "stat": "PSI", "value": round(psi, 4), "unseen_in_train_pct": round(100 * unseen, 2)})
            per.sort(key=lambda r: -r["value"])
            miss_drift = []
            for c in shared_model:
                m1, m2 = float(S[c].isna().mean()), float(T[c].isna().mean())
                if abs(m1 - m2) > 0.1:
                    miss_drift.append({"column": c, "train_missing_pct": round(100 * m1, 1), "test_missing_pct": round(100 * m2, 1)})
            shift = {"adversarial_auc": round(adv_auc, 4), "features": per[:40], "missingness_drift": miss_drift}
            if adv_auc >= cfg.shift_auc_warn:
                sev = "high" if adv_auc >= cfg.shift_auc_high else "medium"
                top = [r["column"] for r in per[:5]]
                findings.append(Finding(
                    id="shift.adversarial", category="distribution_shift", severity=sev,
                    title=f"Train and test are distinguishable (adversarial AUC {adv_auc:.2f})",
                    detail="A classifier can tell train rows from test rows, so random-split CV is not estimating "
                           f"test performance. Most-shifted features: {top}. "
                           + ("Some categorical values in test never appear in train." if any(r.get("unseen_in_train_pct", 0) > 5 for r in per[:10]) else ""),
                    penalty=min(15, 30 * (adv_auc - 0.5)),
                    evidence={"adversarial_auc": round(adv_auc, 4), "top_features": per[:8]},
                    fix="# 1) build a validation set that looks like test: weight train rows by p(test|x)\n"
                        "# 2) or drop the most-shifted features and re-check\n"
                        f"train = train.drop(columns={top[:3]!r})"))
            for r in per[:40]:
                if r.get("unseen_in_train_pct", 0) > 20:
                    findings.append(Finding(
                        id=f"shift.unseen.{r['column']}", category="distribution_shift", severity="medium",
                        title=f"{r['unseen_in_train_pct']:.0f}% of test values of '{r['column']}' never occur in train",
                        detail="Target/one-hot encodings of this column will be undefined for most of test.",
                        penalty=3, evidence=r,
                        fix=f"# treat '{r['column']}' with frequency/hash encoding, or drop it"))
            if miss_drift:
                findings.append(Finding(
                    id="shift.missingness", category="distribution_shift", severity="medium",
                    title=f"Missingness differs sharply between train and test in {len(miss_drift)} column(s)",
                    detail="If a column is filled in train but empty in test (or vice-versa), the model learned a "
                           "pattern that will not exist at inference.",
                    penalty=4, evidence={"columns": miss_drift[:10]},
                    fix="Compare pipelines that produced train vs test; align the missingness or drop the column."))
    sections["shift"] = shift

    # ---------------- cross-validation structure ----------------
    tick("checking cross-validation structure")
    cv_sec: Dict[str, Any] = {"random_cv_score": round(score_clean, 4), "metric": metric, "candidates": []}
    cols_cv = clean_cols if clean_cols else model_cols
    Xcv, _, is_cat_cv = _encode_features(S, cols_cv)

    # group candidates (including id-like columns that are not unique per row)
    cands = []
    flagged_leaks = {l["column"] for l in leaky}
    for c in feature_cols:
        if kinds[c] in ("constant", "text", "datetime", "id") or c in flagged_leaks:
            continue
        s_c = S[c]
        if _is_num(s_c) and not (s_c.dropna() % 1 == 0).all():
            continue  # continuous numbers are not group keys
        nu = int(s_c.nunique())
        if nu < max(20, 4 * cfg.cv_folds) or nu > 0.5 * nS:
            continue
        mean_size = nS / nu
        if mean_size < 2:
            continue
        name_score = 2 if _name_has(c, GROUP_NAME_HINTS) else (1 if _looks_like_id(c) else 0)
        if name_score == 0 and mean_size < 3:
            continue
        disjoint = None
        if test is not None and c in test.columns:
            tv = set(test[c].dropna().unique())
            if tv:
                disjoint = 1 - len(tv & set(train[c].dropna().unique())) / len(tv)
        cands.append({"column": c, "n_groups": nu, "mean_group_size": round(mean_size, 1),
                      "name_hint": name_score, "test_groups_unseen_pct": round(100 * disjoint, 1) if disjoint is not None else None})
    cands.sort(key=lambda r: (-(r["test_groups_unseen_pct"] or 0), -r["name_hint"], -r["mean_group_size"]))
    tried = 0
    best_gap = None
    for cand in cands:
        if tried >= 3 or (time.time() - t0) > cfg.time_budget_s * 0.9:
            break
        c = cand["column"]
        groups = S[c].astype(str).values
        try:
            _, g_score, _ = _oof(task, Xcv, yS, is_cat_cv, GroupKFold(cfg.cv_folds), groups=groups, seed=cfg.random_state)
        except Exception:
            continue
        tried += 1
        gap = score_clean - g_score
        cand["group_cv_score"] = round(g_score, 4)
        cand["gap_vs_random"] = round(gap, 4)
        cv_sec["candidates"].append(cand)
        unseen = cand["test_groups_unseen_pct"]
        if (gap > cfg.group_gap_warn) or (unseen is not None and unseen > 50 and gap > 0.005):
            if best_gap is None or gap > best_gap:
                best_gap = gap
            sev = "critical" if gap > 0.08 or (unseen and unseen > 90 and gap > 0.02) else "high" if gap > 0.04 else "medium"
            why_test = (f" Test contains {unseen:.0f}% groups never seen in train, so the real task is to generalise "
                        f"to new '{c}' groups and the honest CV is GroupKFold." if unseen is not None and unseen > 50 else "")
            findings.append(Finding(
                id=f"cv.group.{c}", category="cross_validation", severity=sev,
                title=f"Random KFold is inflated by {gap:.3f} {metric} because rows share '{c}'",
                detail=f"Random CV: {score_clean:.3f}. GroupKFold on '{c}': {g_score:.3f}. Rows from the same "
                       f"'{c}' land in both train and validation folds, so the model recognises the entity instead of "
                       f"learning the pattern.{why_test}",
                penalty=min(20, 100 * gap),
                evidence=cand,
                fix=f"from sklearn.model_selection import GroupKFold\n"
                    f"cv = GroupKFold(n_splits=5)\n"
                    f"folds = list(cv.split(train, y, groups=train['{c}']))"))
    # time structure
    time_info = None
    if dt_cols:
        c = dt_cols[0]
        dts = datetimes[c].iloc[idx].reset_index(drop=True)
        order = np.argsort(dts.fillna(dts.min()).values)
        try:
            _, t_score, hit_t = _oof(task, Xcv.iloc[order].reset_index(drop=True), yS[order], is_cat_cv,
                                     TimeSeriesSplit(cfg.cv_folds), seed=cfg.random_state)
            # score the random-fold model on exactly the rows the time split evaluated
            rand_same = _score(task, yS[order][hit_t], oof_clean[order][hit_t])
        except Exception:
            t_score, rand_same = float("nan"), score_clean
        gap = rand_same - t_score
        time_info = {"column": c, "train_min": _py(datetimes[c].min()), "train_max": _py(datetimes[c].max()),
                     "time_cv_score": round(t_score, 4) if not math.isnan(t_score) else None, "gap_vs_random": round(gap, 4) if not math.isnan(gap) else None}
        if test is not None and c in test.columns:
            tdt = _parse_datetime(test[c])
            if tdt is not None and tdt.notna().any():
                time_info["test_min"], time_info["test_max"] = _py(tdt.min()), _py(tdt.max())
                time_info["test_is_after_train"] = bool(tdt.min() > datetimes[c].quantile(0.99))
        future = time_info.get("test_is_after_train")
        if (not math.isnan(gap) and gap > cfg.group_gap_warn) or future:
            sev = "critical" if (future and gap > 0.03) else "high" if gap > 0.05 else "medium"
            findings.append(Finding(
                id="cv.time", category="cross_validation", severity=sev,
                title=f"Random KFold is inflated by {gap:.3f} {metric} versus a time-ordered split on '{c}'"
                      if not math.isnan(gap) else f"Test is strictly later than train on '{c}'",
                detail=(f"Random CV on the same rows: {rand_same:.3f}; TimeSeriesSplit: {t_score:.3f}. " if not math.isnan(gap) else "")
                       + ("Test rows are all later than train rows, so the deployed model will be extrapolating in time. "
                          if future else "")
                       + "Shuffled folds let the model peek at the future of each validation row.",
                penalty=min(20, 100 * max(gap, 0) + (5 if future else 0)),
                evidence=time_info,
                fix=f"train = train.sort_values('{c}')\nfrom sklearn.model_selection import TimeSeriesSplit\n"
                    f"cv = TimeSeriesSplit(n_splits=5)\n# and never build features from rows later than the current row"))
    cv_sec["time"] = time_info

    # recommendation
    rec = "StratifiedKFold(5, shuffle=True)" if task != "regression" else "KFold(5, shuffle=True)"
    reason = "no group or time structure detected; rows look exchangeable"
    grp = [f for f in findings if f.id.startswith("cv.group.")]
    tim = [f for f in findings if f.id == "cv.time"]
    if grp:
        best = max(grp, key=lambda f: f.penalty)
        col = best.id.split("cv.group.", 1)[1]
        rec = f"GroupKFold(5) with groups=train['{col}']"
        reason = f"rows sharing '{col}' are not independent"
        if tim:
            rec += f"  (and keep time order within '{dt_cols[0]}' if you can)"
    elif tim:
        rec = f"TimeSeriesSplit(5) after sorting by '{dt_cols[0]}'"
        reason = "the data is temporal and test comes later"
    honest = score_clean
    for f in grp + tim:
        s = f.evidence.get("group_cv_score") or f.evidence.get("time_cv_score")
        if s is not None:
            honest = min(honest, s)
    cv_sec["recommended_split"] = rec
    cv_sec["reason"] = reason
    cv_sec["honest_cv_estimate"] = round(float(honest), 4)
    sections["cv"] = cv_sec

    sections["baseline"] = {"model": "HistGradientBoosting (sklearn), defaults, no tuning", "metric": metric,
                            "random_cv_all_features": round(score_full, 4), "random_cv_without_leaks": round(score_clean, 4),
                            "honest_estimate": round(float(honest), 4), "sample_rows_used": nS}

    # ---------------- validation design: how the test set was sampled ----------------
    if test is not None:
        tick("checking how the test set was sampled")
        sections["design"] = _validation_design(train, test, feature_cols, kinds, findings)
    else:
        sections["design"] = None

    # ---------------- score ----------------
    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], -f.penalty))
    penalty = sum(f.penalty for f in findings)
    score = int(max(0, min(100, round(100 - penalty))))
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 65 else "D" if score >= 50 else "F"
    n_by_sev = {s: sum(1 for f in findings if f.severity == s) for s in SEVERITY_ORDER}

    report = {
        "version": 1,
        "score": score, "grade": grade,
        "meta": {"n_rows": n, "n_cols": len(train.columns), "n_test_rows": int(len(test)) if test is not None else None,
                 "target": cfg.target, "task": task, "metric": metric, "runtime_s": round(time.time() - t0, 1),
                 "findings_by_severity": n_by_sev, "sample_rows": nS},
        "headline": _headline(findings, score, grade, task, metric, score_full, honest),
        "findings": [f.to_dict() for f in findings],
        "sections": sections,
    }
    return _py(report)


def _nearest_km(a_lat, a_lon, b_lat, b_lon, exclude_self=False, chunk=2000) -> np.ndarray:
    a1, a2 = np.radians(a_lat)[:, None], np.radians(a_lon)[:, None]
    b1, b2 = np.radians(b_lat)[None, :], np.radians(b_lon)[None, :]
    out = np.full(len(a_lat), np.inf)
    for i in range(0, len(a_lat), chunk):
        la, lo = a1[i:i + chunk], a2[i:i + chunk]
        h = np.sin((b1 - la) / 2) ** 2 + np.cos(la) * np.cos(b1) * np.sin((b2 - lo) / 2) ** 2
        d = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        if exclude_self:
            idx = np.arange(i, min(i + chunk, len(a_lat)))
            d[np.arange(len(idx)), idx] = np.inf
        out[i:i + chunk] = d.min(1) if d.shape[1] else np.inf
    return out


def _validation_design(train: pd.DataFrame, test: pd.DataFrame, feature_cols: List[str], kinds: Dict[str, str],
                       findings: List[Finding], max_rows: int = 20000) -> Dict[str, Any]:
    """Compare how test rows relate to train rows (per group, in space) with how train rows relate to each other.
    A validation split that does not reproduce the same relation measures a different, usually easier, task."""
    out: Dict[str, Any] = {"groups": [], "spatial": None}
    shared = [c for c in feature_cols if c in test.columns]
    for c in shared:
        if kinds.get(c) in ("constant", "text", "datetime"):
            continue
        tr, te = train[c].astype(str), test[c].astype(str)
        nu = tr.nunique()
        if nu < 2 or nu > 0.7 * len(train):
            continue
        if _is_num(train[c]) and not (train[c].dropna() % 1 == 0).all() and nu > 50:
            continue
        vc_tr, vc_te = tr.value_counts(), te.value_counts()
        unseen = float((~te.isin(set(vc_tr.index))).mean())
        g = {"column": c, "train_groups": int(len(vc_tr)), "train_rows_per_group": float(vc_tr.median()),
             "test_groups": int(len(vc_te)), "test_rows_per_group": float(vc_te.median()), "test_rows_unseen": round(unseen, 4)}
        out["groups"].append(g)
        if g["test_rows_per_group"] <= 1.0 and g["train_rows_per_group"] >= 2 and g["test_groups"] >= 20 and _name_has(c, GROUP_NAME_HINTS + ID_NAME_HINTS):
            findings.append(Finding(
                id=f"design.unit.{c}", category="validation_design", severity="medium",
                title=f"Test holds about one row per '{c}', train holds {g['train_rows_per_group']:.0f}",
                detail="The test set samples one row per group, so every group counts once in the score. Training and validating "
                       "on all rows weights big groups more than the test does, and a model tuned that way optimises the wrong "
                       "average.",
                penalty=3, evidence=g,
                fix=f"w = 1.0 / train.groupby('{c}')['{c}'].transform('size')   # sample_weight for fitting\n"
                    f"val = val.groupby('{c}').sample(1, random_state=42)          # validate one row per group"))
        if 0.1 <= unseen < 0.9 and g["test_groups"] >= 10:
            findings.append(Finding(
                id=f"design.mixed.{c}", category="validation_design", severity="medium",
                title=f"{unseen:.0%} of test rows have a '{c}' never seen in train; the rest are seen",
                detail="A split that holds out whole groups tests only the unseen part, and a random split tests only the seen "
                       "part. Validation should hold out about the same share of groups as the test does, plus random rows "
                       "from the remaining groups, or one model will win on validation and lose on the test.",
                penalty=2, evidence=g,
                fix=f"hold out ~{unseen:.0%} of '{c}' values entirely and add random rows of the other groups to each fold"))
    lat = next((c for c in test.columns if c.lower() in ("lat", "latitude")), None)
    lon = next((c for c in test.columns if c.lower() in ("lon", "lng", "long", "longitude")), None)
    if lat and lon and lat in train.columns and lon in train.columns:
        tr = train.dropna(subset=[lat, lon])
        te = test.dropna(subset=[lat, lon])
        tr = tr.sample(min(len(tr), max_rows), random_state=42)
        te = te.sample(min(len(te), max_rows), random_state=42)
        tr_u = tr.drop_duplicates(subset=[lat, lon])
        q = lambda v: {k: round(float(np.quantile(v[np.isfinite(v)], x)), 1) for k, x in (("p05", .05), ("median", .5))} if np.isfinite(v).any() else {}
        d_te = _nearest_km(te[lat].to_numpy(float), te[lon].to_numpy(float), tr_u[lat].to_numpy(float), tr_u[lon].to_numpy(float))
        d_tr = _nearest_km(tr_u[lat].to_numpy(float), tr_u[lon].to_numpy(float), tr_u[lat].to_numpy(float), tr_u[lon].to_numpy(float), exclude_self=True)
        sp = {"lat": lat, "lon": lon, "test_to_train_km": q(d_te), "train_to_train_km": q(d_tr), "by_category": {}}
        cats = [c for c in shared if c not in (lat, lon) and not _is_num(train[c]) and 2 <= train[c].nunique() <= 0.5 * len(train)]
        for c in cats[:5]:
            d_a, d_b = [], []
            for v, gte in te.groupby(te[c].astype(str)):
                ref = tr_u[tr_u[c].astype(str) == v] if c in tr_u.columns else tr_u.iloc[:0]
                if len(ref):
                    d_a.append(_nearest_km(gte[lat].to_numpy(float), gte[lon].to_numpy(float), ref[lat].to_numpy(float), ref[lon].to_numpy(float)))
            tr_c = tr.drop_duplicates(subset=[lat, lon, c])
            for v, gtr in tr_c.groupby(tr_c[c].astype(str)):
                if len(gtr) > 1:
                    d_b.append(_nearest_km(gtr[lat].to_numpy(float), gtr[lon].to_numpy(float), gtr[lat].to_numpy(float), gtr[lon].to_numpy(float), exclude_self=True))
            if d_a and d_b:
                a, b = np.concatenate(d_a), np.concatenate(d_b)
                b = b[np.isfinite(b)]
                if len(b) == 0:
                    continue
                sp["by_category"][c] = {"test_min_km": round(float(a.min()), 1), "test_median_km": round(float(np.median(a)), 1),
                                        "train_min_km": round(float(b.min()), 1), "train_median_km": round(float(np.median(b)), 1)}
                tm, rm = float(a.min()), float(b.min())
                if tm > 3 * max(rm, 25.0) and len(a) >= 20:
                    findings.append(Finding(
                        id=f"design.spatial.{c}", category="validation_design", severity="high",
                        title=f"No test row has a same-'{c}' training row closer than {tm:.0f} km (train-to-train: {rm:.0f} km)",
                        detail="The test set was held out in space within each group. A random or group split lets validation rows "
                               "see same-group neighbours a few km away, which the test never does, so validation is optimistic.",
                        penalty=6, evidence=sp["by_category"][c],
                        fix=f"for validation rows, drop same-'{c}' training rows within {tm:.0f} km (or use spatial blocks of that size)"))
        t5, r5 = sp["test_to_train_km"].get("p05"), sp["train_to_train_km"].get("p05")
        if t5 is not None and r5 is not None and t5 > 2 * max(r5, 10.0) and len(te) >= 20:
            findings.append(Finding(
                id="design.spatial", category="validation_design", severity="high",
                title=f"Test locations sit far from every training location (5th percentile {t5:.0f} km vs {r5:.0f} km within train)",
                detail="The test covers new places. Random folds keep near-duplicate locations on both sides of the split.",
                penalty=6, evidence=sp, fix="validate with spatial blocks (e.g. GroupKFold on a lat/lon grid cell at that distance)"))
        out["spatial"] = sp
    return out


def _headline(findings: List[Finding], score: int, grade: str, task: str, metric: str, naive: float, honest: float) -> str:
    crit = [f for f in findings if f.severity == "critical"]
    high = [f for f in findings if f.severity == "high"]
    if crit:
        return f"Grade {grade}: {len(crit)} critical issue(s). Your naive CV {metric} {naive:.3f} is not real; honest estimate ≈ {honest:.3f}."
    if high:
        return f"Grade {grade}: {len(high)} high-severity issue(s). Expect your CV {metric} {naive:.3f} to fall toward {honest:.3f} without them."
    if score >= 90:
        return f"Grade {grade}: no leakage or split problems detected. Naive CV {metric} {naive:.3f} looks trustworthy."
    return f"Grade {grade}: minor issues only. Honest CV estimate ≈ {honest:.3f} {metric}."
