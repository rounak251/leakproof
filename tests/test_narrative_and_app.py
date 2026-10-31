import json
import os
import time
import types

import pytest


# ----------------------------------------------------------------------------- narrative (mocked Claude)

def test_narrative_without_key_returns_none(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from leakproof.narrative import write_narrative
    assert write_narrative({"findings": []}) is None


def test_narrative_parses_structured_output(monkeypatch):
    import anthropic
    from leakproof import narrative

    payload = {"headline": "h", "summary": "s", "root_cause": "none", "steps": [], "validation_recipe": "cv", "what_to_tell_stakeholders": "ok", "confidence": "high"}

    class FakeResp:
        stop_reason = "end_turn"
        model = "claude-opus-5-5"
        content = [types.SimpleNamespace(type="text", text=json.dumps(payload))]
        usage = types.SimpleNamespace(input_tokens=10, output_tokens=5)

    captured = {}

    class FakeMessages:
        def create(self, **kw):
            captured.update(kw)
            return FakeResp()

    class FakeClient:
        def __init__(self, api_key=None):
            self.beta = types.SimpleNamespace(messages=FakeMessages())
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    out = narrative.write_narrative({"findings": [], "sections": {}}, api_key="k")
    assert out["headline"] == "h" and out["model"] == "claude-opus-5-5"
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert captured["fallbacks"] == "default"


# ----------------------------------------------------------------------------- app

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["LEAKPROOF_DATA_DIR"] = str(tmp_path_factory.mktemp("data"))
    os.environ.pop("DATABASE_URL", None)
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app, follow_redirects=False)


def _wait(client, url, headers=None):
    for _ in range(600):
        s = client.get(url, headers=headers).json()
        if s["status"] in ("done", "failed"):
            return s
        time.sleep(0.5)
    raise AssertionError("audit did not finish")


def test_auth_and_demo_flow(client):
    assert client.get("/").status_code == 200
    assert client.get("/app").status_code == 303  # logged out
    r = client.post("/signup", data={"email": "a@example.com", "password": "password123"})
    assert r.status_code == 303
    r = client.post("/app/demo")
    aid = int(r.headers["location"].rsplit("/", 1)[1])
    s = _wait(client, f"/app/audits/{aid}/status")
    assert s["status"] == "done" and s["grade"] == "F", s
    assert client.get(f"/app/audits/{aid}").status_code == 200
    assert client.get("/app/audits/99999").status_code == 404
    # free plan: exports are paywalled, demo share link works
    assert client.get(f"/app/audits/{aid}/report.html").status_code == 402
    from app.db import Audit, SessionLocal
    db = SessionLocal()
    token = db.get(Audit, aid).share_token
    db.close()
    assert client.get(f"/r/{token}").status_code == 200
    assert client.get(f"/r/{token}/badge.svg").headers["content-type"].startswith("image/svg")
    assert client.get("/r/not-a-token").status_code == 404


def test_upload_validation_and_api(client):
    r = client.post("/app/new", files={"train": ("t.csv", b"a,b,y\n1,2,0\n3,4,1\n", "text/csv")}, data={"target": "nope"})
    assert r.status_code == 200 and b"not found" in r.content
    # API needs the Team plan
    from app.db import SessionLocal, User
    db = SessionLocal()
    u = db.query(User).filter(User.email == "a@example.com").first()
    key = u.api_key
    assert client.get("/api/v1/audits", headers={"Authorization": f"Bearer {key}"}).status_code == 402
    u.plan = "team"
    db.commit()
    db.close()
    from leakproof.demo import make_clean_demo
    train, test = make_clean_demo(n=1500)
    files = {"train": ("train.csv", train.to_csv(index=False).encode(), "text/csv"), "test": ("test.csv", test.to_csv(index=False).encode(), "text/csv")}
    r = client.post("/api/v1/audits", files=files, data={"target": "y"}, headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200, r.text
    s = _wait(client, f"/api/v1/audits/{r.json()['id']}", headers={"Authorization": f"Bearer {key}"})
    assert s["status"] == "done" and s["grade"] in ("A", "B"), s
    assert client.get("/api/v1/audits", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_users_cannot_see_each_others_audits(client):
    client.get("/logout")
    client.post("/signup", data={"email": "b@example.com", "password": "password123"})
    assert client.get("/app/audits/1").status_code == 404
    assert client.get("/app/audits/1/report.json").status_code == 404


def test_lemonsqueezy_billing(client, monkeypatch):
    """Checkout links carry the user id; signed subscription webhooks move the plan; bad signatures are refused."""
    import hashlib, hmac, json as _json
    import app.main as m
    from app.db import SessionLocal, User
    monkeypatch.setattr(m, "PAYMENTS", "lemonsqueezy")
    monkeypatch.setattr(m, "LEMON_WEBHOOK_SECRET", "whsec_test")
    monkeypatch.setitem(m.LEMON_CHECKOUT, "pro", "https://leakproof.lemonsqueezy.com/checkout/buy/abc-123")
    client.get("/logout")
    client.post("/signup", data={"email": "payer@example.com", "password": "password123"})
    r = client.post("/billing/checkout", data={"plan": "pro"})
    assert r.status_code == 303
    loc = r.headers["location"]
    assert loc.startswith("https://leakproof.lemonsqueezy.com/checkout/buy/abc-123?")
    assert "checkout%5Bcustom%5D%5Buser_id%5D=" in loc and "payer%40example.com" in loc
    db = SessionLocal()
    uid = db.query(User).filter(User.email == "payer@example.com").first().id
    db.close()

    def send(event, status, secret="whsec_test"):
        body = _json.dumps({"meta": {"event_name": event, "custom_data": {"user_id": str(uid), "plan": "pro"}},
                            "data": {"type": "subscriptions", "id": "77",
                                     "attributes": {"status": status, "customer_id": 9001, "variant_id": 5,
                                                    "urls": {"customer_portal": "https://leakproof.lemonsqueezy.com/billing?x=1"}}}}).encode()
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        return client.post("/billing/lemon-webhook", content=body, headers={"X-Signature": sig, "Content-Type": "application/json"})

    assert send("subscription_created", "active", secret="wrong").status_code == 400
    r = send("subscription_created", "active")
    assert r.status_code == 200 and r.json()["plan"] == "pro"
    assert send("subscription_cancelled", "cancelled").json()["plan"] == "pro"     # paid up until it expires
    assert send("subscription_expired", "expired").json()["plan"] == "free"
    db = SessionLocal()
    u = db.get(User, uid)
    assert u.lemon_customer_id == "9001" and u.billing_portal_url.startswith("https://leakproof.lemonsqueezy.com/billing")
    db.close()
    r = client.post("/billing/portal")
    assert r.status_code == 303 and r.headers["location"].startswith("https://leakproof.lemonsqueezy.com/billing")


def test_lint_page_and_api(client):
    import app.main as m
    from app.db import SessionLocal, User
    client.get("/logout")
    client.post("/signup", data={"email": "linter@example.com", "password": "password123"})
    leaky = ("import pandas as pd\nfrom sklearn.preprocessing import StandardScaler\n"
             "train = pd.read_csv('train.csv')\ntest = pd.read_csv('test.csv')\n"
             "sc = StandardScaler().fit(pd.concat([train, test]))\n")
    r = client.post("/app/lint", data={"code": leaky})
    assert r.status_code == 200 and b"leak found" in r.content and b"TRAIN+TEST" in r.content
    r = client.post("/app/lint", data={"code": "x = 1\n"})
    assert r.status_code == 200 and b"No test-set leakage found" in r.content
    db = SessionLocal()
    u = db.query(User).filter(User.email == "linter@example.com").first()
    u.plan = "team"
    key = u.api_key
    db.commit()
    db.close()
    r = client.post("/api/v1/lint", files={"file": ("train.py", leaky.encode(), "text/x-python")},
                    headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200 and r.json()["errors"] == 1
