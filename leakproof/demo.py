"""Synthetic datasets with planted problems, used for tests and the in-app demo.

make_churn_demo() plants, deliberately:
  * a target-leak column       ('last_contact_outcome' is derived from churn)
  * a post-outcome date column ('cancellation_date', only filled when churned)
  * exact duplicate rows
  * group structure            (several rows per customer_id; test customers unseen)
  * label noise                (5% of labels flipped)
  * train/test shift           (test skews to a newer plan tier and later dates)
  * a handful of test rows copied from train
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_churn_demo(n_customers: int = 3000, rows_per_customer: int = 4, seed: int = 7):
    rng = np.random.RandomState(seed)
    n = n_customers * rows_per_customer
    cust = np.repeat(np.arange(100000, 100000 + n_customers), rows_per_customer)
    plan = rng.choice(["basic", "plus", "pro"], size=n_customers, p=[0.5, 0.35, 0.15])
    plan = np.repeat(plan, rows_per_customer)
    region = np.repeat(rng.choice(["north", "south", "east", "west"], size=n_customers), rows_per_customer)
    tenure = np.repeat(rng.gamma(2.0, 12.0, size=n_customers), rows_per_customer) + rng.normal(0, 1, n)
    # per-customer latent risk drives churn -> rows from the same customer are correlated
    latent = np.repeat(rng.normal(0, 1, n_customers), rows_per_customer)
    monthly_usage = np.clip(rng.normal(50, 20, n) - 8 * latent, 0, None)
    support_tickets = rng.poisson(np.clip(1.0 + latent, 0.1, None))
    late_payments = rng.binomial(3, np.clip(0.15 + 0.1 * latent, 0.01, 0.9))
    discount_pct = rng.choice([0, 5, 10, 20], size=n)
    signup = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.randint(0, 500, n), unit="D")
    # churn is decided per customer (the realistic case: one outcome, several rows per entity)
    lat_c = latent[::rows_per_customer]
    tickets_c = support_tickets.reshape(n_customers, rows_per_customer).mean(1)
    late_c = late_payments.reshape(n_customers, rows_per_customer).mean(1)
    usage_c = monthly_usage.reshape(n_customers, rows_per_customer).mean(1)
    logit_c = -0.8 + 1.6 * lat_c + 0.3 * (tickets_c > 2) + 0.5 * late_c - 0.01 * usage_c
    p_c = 1 / (1 + np.exp(-logit_c))
    churn_c = (rng.rand(n_customers) < p_c).astype(int)
    churn = np.repeat(churn_c, rows_per_customer)
    df = pd.DataFrame({
        "customer_id": cust,
        "plan": plan,
        "region": region,
        "signup_date": signup,
        "tenure_months": tenure.round(1),
        "monthly_usage_gb": monthly_usage.round(2),
        "support_tickets": support_tickets,
        "late_payments": late_payments,
        "discount_pct": discount_pct,
        "churned": churn,
    })
    # planted leaks
    df["last_contact_outcome"] = np.where(churn == 1, rng.choice(["cancel_request", "complaint"], size=n),
                                          rng.choice(["upsell", "no_action", "question"], size=n))
    df["cancellation_date"] = pd.Series(np.where(churn == 1, (signup + pd.Timedelta(days=200)).strftime("%Y-%m-%d"), None))
    # label noise
    flip = rng.rand(n) < 0.08
    df.loc[flip, "churned"] = 1 - df.loc[flip, "churned"]
    # split by customer: test customers are unseen, skewed to pro plan and later signups
    cust_ids = np.unique(cust)
    rng.shuffle(cust_ids)
    test_c = set(cust_ids[: n_customers // 5])
    is_test = df["customer_id"].isin(test_c)
    train = df[~is_test].copy()
    test = df[is_test].copy()
    test = test[(test["plan"] != "basic") | (rng.rand(len(test)) < 0.4)]
    test["signup_date"] = test["signup_date"] + pd.Timedelta(days=120)
    # duplicates in train
    dup = train.sample(min(150, len(train)), random_state=seed)
    train = pd.concat([train, dup], ignore_index=True)
    # contamination: a few train rows copied into test
    leak = train.sample(min(40, len(train)), random_state=seed + 1)
    test = pd.concat([test, leak], ignore_index=True)
    test = test.drop(columns=["churned"])
    train["row_id"] = np.arange(len(train))
    test["row_id"] = np.arange(len(test)) + 10_000_000
    train = train.sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, test.reset_index(drop=True)


def make_clean_demo(n: int = 4000, seed: int = 3):
    """A dataset with no planted problems: the engine should give it an A."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] ** 2 - X[:, 2] + rng.normal(scale=0.4, size=n) > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["cat"] = rng.choice(list("abcde"), size=n)
    df["y"] = y
    train, test = df.iloc[: int(0.8 * n)].copy(), df.iloc[int(0.8 * n):].drop(columns=["y"]).copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)
