import json

import numpy as np
import pandas as pd
import pytest

from leakproof.demo import make_churn_demo, make_clean_demo
from leakproof.engine import AuditConfig, audit
from leakproof.render import render_html


@pytest.fixture(scope="module")
def churn_report():
    train, test = make_churn_demo()
    return audit(train, test, AuditConfig(target="churned"))


def ids(report):
    return {f["id"] for f in report["findings"]}


def test_planted_problems_are_found(churn_report):
    r = churn_report
    found = ids(r)
    assert "leak.last_contact_outcome" in found
    assert "leak.cancellation_date" in found
    assert "overlap.exact" in found
    assert "cv.group.customer_id" in found
    assert "dup.exact" in found
    assert "shift.adversarial" in found
    assert r["grade"] == "F" and r["score"] < 40


def test_honest_estimate_below_naive(churn_report):
    b = churn_report["sections"]["baseline"]
    assert b["honest_estimate"] < b["random_cv_without_leaks"] < b["random_cv_all_features"]
    assert "customer_id" in churn_report["sections"]["cv"]["recommended_split"]


def test_report_is_json_serialisable(churn_report):
    json.dumps(churn_report)


def test_html_render(churn_report):
    html = render_html(churn_report, title="t")
    assert "Leakproof" in html and "cancellation_date" in html


def test_clean_dataset_gets_an_a():
    train, test = make_clean_demo()
    r = audit(train, test, AuditConfig(target="y"))
    assert r["grade"] == "A", r["findings"]
    assert not {f["id"] for f in r["findings"] if f["severity"] in ("critical", "high")}


def test_regression_task():
    rng = np.random.RandomState(0)
    n = 3000
    X = rng.normal(size=(n, 5))
    y = X[:, 0] * 2 + X[:, 1] + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(X, columns=list("abcde"))
    df["y"] = y
    df["y_plus_noise"] = y + rng.normal(scale=0.01, size=n)  # planted leak
    r = audit(df, None, AuditConfig(target="y"))
    assert r["meta"]["task"] == "regression"
    assert "leak.y_plus_noise" in ids(r)


def test_multiclass_and_no_test():
    rng = np.random.RandomState(1)
    n = 2000
    X = rng.normal(size=(n, 4))
    y = np.argmax(X[:, :3] + rng.normal(scale=0.7, size=(n, 3)), axis=1)
    df = pd.DataFrame(X, columns=list("abcd"))
    df["label"] = np.array(["cat", "dog", "fox"])[y]
    r = audit(df, None, AuditConfig(target="label"))
    assert r["meta"]["task"] == "multiclass"
    assert r["sections"]["overlap"] is None
    assert r["score"] >= 80


def test_missing_target_raises():
    train, _ = make_clean_demo()
    with pytest.raises(ValueError):
        audit(train, None, AuditConfig(target="nope"))


def test_validation_design_checks():
    """Sampling unit (one row per group in test), mixed seen/unseen groups, and a spatial hold-out within a category."""
    from leakproof.engine import _validation_design
    rng = np.random.RandomState(0)
    # train: 200 users x 5 rows; test: 120 users x 1 row, 40% of them unseen
    tr = pd.DataFrame({"user_id": np.repeat([f"u{i}" for i in range(200)], 5), "x": rng.randn(1000)})
    te_users = [f"u{i}" for i in range(72)] + [f"n{i}" for i in range(48)]
    te = pd.DataFrame({"user_id": te_users, "x": rng.randn(120)})
    F = []
    _validation_design(tr, te, ["user_id", "x"], {"user_id": "categorical", "x": "numeric"}, F)
    ids = {f.id for f in F}
    assert "design.unit.user_id" in ids and "design.mixed.user_id" in ids
    # spatial: two species on a grid; test sites of species A sit >= 600 km from any training site of A
    sites = []
    for i in range(40):
        for j in range(40):
            sites.append((30 + i * 0.5, -120 + j * 0.5, "A" if (i + j) % 2 else "B"))
    df = pd.DataFrame(sites, columns=["latitude", "longitude", "species"])
    far = (df.latitude > 42) & (df.longitude > -110)
    test = df[far & (df.species == "A")].sample(40, random_state=0)
    zone = (df.latitude > 36) & (df.longitude > -116)          # no species-A training site near the held-out region
    train = df[~(zone & (df.species == "A"))]
    F2 = []
    out = _validation_design(train, test, ["latitude", "longitude", "species"],
                             {"latitude": "numeric", "longitude": "numeric", "species": "categorical"}, F2)
    assert any(f.id == "design.spatial.species" for f in F2), [f.title for f in F2]
    assert out["spatial"]["by_category"]["species"]["test_min_km"] > 300
    # a random split of the same table must not raise either flag
    F3 = []
    shuf = df.sample(frac=1, random_state=1)
    _validation_design(shuf.iloc[:700], shuf.iloc[700:], ["latitude", "longitude", "species"],
                       {"latitude": "numeric", "longitude": "numeric", "species": "categorical"}, F3)
    assert not [f for f in F3 if f.id.startswith("design.spatial")], [f.title for f in F3]
