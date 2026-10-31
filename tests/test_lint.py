"""leakproof lint: leakage patterns must be caught, compliant patterns must stay quiet."""
import json
import os
import subprocess
import sys

import pytest

from leakproof.lint import lint_path, lint_source

CASES = os.path.join(os.path.dirname(__file__), "lint_cases")

LEAKS = {   # case -> rule that must be an ERROR
    "fn_fillna_staged": "test_impute", "fn_torch_dataset": "test_gradient", "fn_class_setter": "test_fit",
    "fn_class_fit": "test_fit", "fn_comprehension": "test_fit", "fn_cluster": "test_fit", "fn_nanmean": "test_impute",
    "fn_numpy_r": "test_fit", "fn_generator": "test_fit", "fn_lambda": "test_fit", "fn_partial": "test_fit",
    "fn_bound_fit": "test_fit", "fn_global": "test_fit", "fn_default": "test_fit", "fn_fillna_keyword": "test_impute",
    "fn_read_path_generator": "test_fit", "fn_star_args": "test_fit", "fn_append_alias": "test_fit",
    "fn_dict_frames_control": "test_fit", "fn_list_loop_control": "test_fit", "fn_merge_control": "test_fit",
    "fp_safe_slice": "test_fit",          # a slice of a train+test concat stays an error: keep the arrays separate
}
CLEAN = ["fp_dict_get", "fp_dynamic_dict_index", "fp_len", "fp_param_name", "fp_test_column"]


@pytest.mark.parametrize("case,rule", sorted(LEAKS.items()))
def test_leak_caught(case, rule):
    rep = lint_path(os.path.join(CASES, case + ".py"))
    assert rule in {f.rule for f in rep.errors}, [(f.level, f.rule, f.msg[:80]) for f in rep.findings]


@pytest.mark.parametrize("case", CLEAN)
def test_clean_quiet(case):
    rep = lint_path(os.path.join(CASES, case + ".py"))
    assert not rep.errors, [f.msg[:100] for f in rep.errors]


GOOD = """
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
train = pd.read_csv(os.path.join("data", "train.csv"))
test = pd.read_csv(os.path.join("data", "test.csv"))
cols = ["a", "b"]
scaler = StandardScaler().fit(train[cols])
X_train, X_test = scaler.transform(train[cols]), scaler.transform(test[cols])
model = HistGradientBoostingClassifier(random_state=0).fit(X_train, train["y"])
pred = model.predict_proba(X_test)[:, 1]
"""


def test_good_script_is_clean():
    rep = lint_source(GOOD)
    assert not rep.errors and not rep.warnings
    assert all(r.sees == "train rows only" for r in rep.fits) and len(rep.fits) == 2


def test_classic_leak_message():
    bad = GOOD.replace('StandardScaler().fit(train[cols])', 'StandardScaler().fit(pd.concat([train[cols], test[cols]]))')
    rep = lint_source(bad)
    err = [f for f in rep.errors if f.rule == "test_fit"]
    assert err and "TRAIN+TEST" in err[0].msg
    assert any(f.rule == "joint" for f in rep.warnings)


def test_notebook_locations(tmp_path):
    nb = {"cells": [
        {"cell_type": "markdown", "source": ["# EDA"]},
        {"cell_type": "code", "source": ["import pandas as pd\n", "%matplotlib inline\n",
                                         "train = pd.read_csv('train.csv')\n", "test = pd.read_csv('test.csv')"]},
        {"cell_type": "code", "source": ["from sklearn.preprocessing import StandardScaler\n",
                                         "sc = StandardScaler().fit(test[['a']])"]},
    ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = tmp_path / "eda.ipynb"
    p.write_text(json.dumps(nb))
    rep = lint_path(str(p))
    assert rep.errors and rep.errors[0].where == "cell 2, line 2"


def test_cli_exit_codes(tmp_path):
    good, bad = tmp_path / "good.py", tmp_path / "bad.py"
    good.write_text(GOOD)
    bad.write_text(GOOD.replace("fit(train[cols])", "fit(test[cols])"))
    run = lambda *a: subprocess.run([sys.executable, "-m", "leakproof.cli", "lint", *a], capture_output=True, text=True,
                                    cwd=os.path.dirname(os.path.dirname(__file__)))
    assert run(str(good)).returncode == 0
    r = run(str(bad))
    assert r.returncode == 1 and "TEST rows" in r.stdout
    assert run(str(bad), "--fail-on", "never").returncode == 0
