# Leakproof

**Catch data leakage, label noise and broken CV splits before you train.**

Upload `train.csv` (and optionally `test.csv`), name the target column, and in under a minute you get a
Leakproof Score (0–100, grade A–F), a ranked list of findings with exact code fixes, and a fix plan
written by Claude. The same engine runs locally for free from the CLI and as a CI gate.

What it checks:

| Check | How | Typical finding |
|---|---|---|
| Target leakage | every feature fitted alone vs. the target; deterministic maps; missingness-vs-label | `'cancellation_date' looks like target leakage` |
| Train/test contamination | exact + fuzzy row hashing across splits; shared IDs | `40 test rows (2.35%) are exact copies of train rows` |
| Duplicates | exact + near-duplicate rows; conflicting labels | `150 exact duplicate rows (1.5% of train)` |
| Label noise | confident-learning on out-of-fold probabilities | `6.2% of labels contradict a confident model` |
| Distribution shift | adversarial validation + per-feature KS / PSI; missingness drift | `Train and test are distinguishable (AUC 0.91)` |
| Wrong cross-validation | discovers group and time structure; KFold vs GroupKFold / TimeSeriesSplit | `Random KFold is inflated by 0.061 AUC because rows share 'customer_id'` |
| Honest estimate | baseline model under the recommended split | `naive 0.919 → honest 0.681` |
| Test sampling design | one row per group in test, seen vs unseen groups, spatial hold-outs by lat/lon | `No test row has a same-species training row closer than 454 km` |

## Check training code too: `leakproof lint`

```bash
leakproof lint train.py notebook.ipynb            # which fit sees test rows, on which line / notebook cell
leakproof lint train.py --fail-on error --json    # CI: exit 1 when a scaler / PCA / vocab / model is fitted on test rows
```

Static data-flow analysis (nothing runs, nothing is uploaded): every value is tagged with where its rows come from,
through functions, returns, loops, comprehensions, lambdas, classes and containers. It flags fits on test rows,
train+test joins, imputation from test statistics, gradient steps on test batches, and (optionally) unseeded
randomness. Also in the web app ("Check code") and the API (`POST /api/v1/lint`).

## Quick start (CLI, free, nothing leaves your machine)

```bash
pip install "git+https://github.com/rounak251/leakproof"
leakproof demo                                   # writes a planted-leak dataset
leakproof audit leakproof_demo/train.csv --test leakproof_demo/test.csv --target churned --html report.html
leakproof gate  train.csv --target y --min-score 80 --fail-on critical   # exit 1 → fails CI
```

Python:

```python
from leakproof import audit, AuditConfig
report = audit(train_df, test_df, AuditConfig(target="y"))
report["score"], report["grade"], report["findings"]
```

Set `ANTHROPIC_API_KEY` and the CLI also prints Claude's fix plan.

## Run the web app

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[server,dev]"
cp .env.example .env            # fill SECRET_KEY, ANTHROPIC_API_KEY, STRIPE_* when you have them
set -a; . ./.env; set +a
uvicorn app.main:app --reload   # http://localhost:8000
pytest -q                       # engine + app tests
```

Pages: `/` landing · `/signup` · `/app` dashboard · `/app/new` upload · `/app/audits/{id}` report (live progress) ·
`/r/{token}` public share link + `/r/{token}/badge.svg` · `/app/audits/{id}/evidence.html` validation evidence ·
`/app/settings` plan, API key, team, audit log · `/api/docs` REST docs.

## Plans

| | Free | Pro $29/mo ($290/yr) | Team $99/mo | Enterprise |
|---|---|---|---|---|
| Audits / month | 3 | unlimited | unlimited | unlimited |
| File size | 20 MB | 500 MB | 2 GB | custom |
| Claude fix plan | – | ✓ | ✓ | ✓ |
| Share links, HTML/PDF, evidence export | – | ✓ | ✓ | ✓ |
| REST API + CI gate | – | – | ✓ (5 seats) | ✓ |
| On-prem Docker, SSO, audit log | – | – | – | ✓ |

Plan limits live in `app/db.py` (`PLANS`). Billing works with either provider, chosen by which variables are set:

- **Lemon Squeezy** (recommended from India: merchant of record, handles global sales tax, pays out to Indian banks).
  Set `LEMON_CHECKOUT_PRO / _PRO_YEAR / _TEAM` to the hosted checkout links, `LEMON_WEBHOOK_SECRET`, `LEMON_STORE_URL`, and point
  a webhook (subscription events) at `/billing/lemon-webhook`. Checkout links carry the user id as custom data.
- **Stripe** Checkout + Customer Portal; the webhook at `/billing/webhook` handles `customer.subscription.*`.

## Deploy

```bash
docker build -t leakproof . && docker run -p 8000:8000 -v leakproof_data:/data -e SECRET_KEY=... leakproof
# or
fly launch --copy-config --no-deploy && fly volumes create leakproof_data --size 10
fly secrets set SECRET_KEY=... ANTHROPIC_API_KEY=... STRIPE_SECRET_KEY=... && fly deploy
```

The same image is the on-prem build for Enterprise customers.

## Layout

```
leakproof/engine.py      the audit engine (pandas / sklearn, no network)
leakproof/narrative.py   Claude fix plan (structured output, refusal fallback)
leakproof/cli.py         audit / gate / demo commands
leakproof/render.py      standalone HTML report
leakproof/demo.py        synthetic datasets with planted problems
app/main.py              FastAPI app: auth, uploads, worker, share, API, Stripe
app/templates/           landing + app pages (Tailwind)
tests/                   engine + app tests
```

## Data handling

Uploaded files are deleted the moment the report is generated. Only statistics, column names and the short
lists of suspect rows shown in the report are stored. Enterprise customers run the Docker image inside their
own network. Nothing is used to train models.
