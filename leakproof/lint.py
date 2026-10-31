"""
leakproof lint: find test-set leakage in training code, before it costs you a model.

Static data-flow analysis of a Python script or Jupyter notebook. Every value is tagged with where its rows come
from (training rows, test rows, or a train+test join), through assignments, function calls and returns, loops,
comprehensions, lambdas, classes and containers. Then every fit is checked:

  ERROR  a scaler / PCA / vocabulary / tokenizer / clustering / model .fit sees test rows
  ERROR  missing values filled with statistics computed on test rows
  ERROR  a training loop takes gradient steps on batches from test data
  WARN   train and test rows joined into one array (a fit on a slice of it needs proof the slice is train-only)
  WARN   aggregations or pairwise statistics across test rows (fine only when a group is one sample)
  WARN   reproducibility: unseeded randomness, estimators / splitters without random_state, unseeded DataLoader shuffling

Nothing is executed and nothing leaves your machine.
    leakproof lint train.py [--json] [--fail-on error|warn] [--no-repro]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

APPLY_METHODS = {"transform", "predict", "predict_proba", "predict_log_proba", "decision_function", "score_samples",
                 "inverse_transform", "kneighbors", "staged_predict"}

# ----------------------------------------------------------------------------- data-flow engine

CONCAT_CALLS = {"concatenate", "vstack", "hstack", "row_stack", "concat", "cat", "stack", "append", "r_", "chain"}


FIT_ATTRS = {"fit", "fit_transform", "partial_fit", "fit_predict", "fit_resample", "train_from_iterator",
             "train_new_from_iterator", "build_vocab", "fit_on_texts", "adapt"}


UNSUP_CTORS = {"Word2Vec", "FastText", "Doc2Vec", "LdaModel", "Dictionary"}


NAN_STATS = {"numpy.nanmean", "numpy.nanmedian", "numpy.nanstd", "numpy.nanvar", "numpy.nanpercentile", "numpy.nanquantile"}


PAIRWISE = {"cdist", "pairwise_distances", "cosine_similarity", "euclidean_distances", "linear_kernel", "rbf_kernel"}


SEED_KW = {"random_state", "seed", "random_seed", "rng", "generator"}


NEEDS_RS = {
    "RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor", "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor", "IsolationForest", "MLPClassifier", "MLPRegressor", "SGDClassifier",
    "SGDRegressor", "KMeans", "MiniBatchKMeans", "TruncatedSVD", "NMF", "FastICA", "TSNE", "GaussianMixture",
    "BayesianGaussianMixture", "SpectralClustering", "RandomizedSearchCV", "ShuffleSplit", "StratifiedShuffleSplit",
    "GroupShuffleSplit", "train_test_split", "BaggingClassifier", "BaggingRegressor", "AdaBoostClassifier",
    "AdaBoostRegressor", "DecisionTreeClassifier", "DecisionTreeRegressor", "RBFSampler", "Nystroem",
    "SparseRandomProjection", "GaussianRandomProjection", "permutation_importance", "RepeatedKFold",
    "RepeatedStratifiedKFold", "LGBMClassifier", "LGBMRegressor", "LGBMRanker", "XGBClassifier", "XGBRegressor",
    "XGBRanker", "CatBoostClassifier", "CatBoostRegressor", "CatBoostRanker", "UMAP", "LatentDirichletAllocation",
    "MiniBatchDictionaryLearning", "IterativeImputer", "RandomTreesEmbedding", "kmeans_plusplus", "make_blobs",
}


SHUFFLE_SPLITTERS = {"KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold"}


def dotted(n) -> Optional[str]:
    parts = []
    while True:
        if isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        elif isinstance(n, ast.Name):
            parts.append(n.id)
            break
        elif isinstance(n, ast.Call):
            n = n.func
        elif isinstance(n, ast.Subscript):
            n = n.value
        else:
            return None
    return ".".join(reversed(parts))


class Aliases:
    """Import aliases, plus names bound exactly once to another name / attribute (env = os.environ,
    clock = time.perf_counter, learn = scaler.fit) and names bound exactly once to a constant."""

    def __init__(self, tree):
        self.map: Dict[str, str] = {}
        self.local: Dict[str, str] = {}
        self.consts: Dict[str, object] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        self.map[a.asname] = a.name
                    else:
                        top = a.name.split(".")[0]
                        self.map.setdefault(top, top)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for a in node.names:
                    self.map[a.asname or a.name] = f"{node.module}.{a.name}"
        binds: Dict[str, List[ast.AST]] = {}

        def note(t, value):
            if isinstance(t, ast.Name):
                binds.setdefault(t.id, []).append(value)
            elif isinstance(t, (ast.Tuple, ast.List)):
                for e in t.elts:
                    note(e, None)
            elif isinstance(t, ast.Starred):
                note(t.value, None)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    note(t, node.value if isinstance(t, ast.Name) else None)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                note(node.target, None)
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                note(node.target, None)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for it in node.items:
                    if it.optional_vars is not None:
                        note(it.optional_vars, None)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                a = node.args
                for x in a.posonlyargs + a.args + a.kwonlyargs + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else []):
                    binds.setdefault(x.arg, []).append(None)
        for name, vals in binds.items():
            if len(vals) != 1 or vals[0] is None or name in self.map:
                continue
            v = vals[0]
            if isinstance(v, ast.Constant):
                self.consts[name] = v.value
            elif isinstance(v, (ast.Name, ast.Attribute)) and dotted(v):
                self.local[name] = dotted(v)

    def canon(self, d: Optional[str], depth: int = 0) -> str:
        if not d:
            return ""
        head, _, rest = d.partition(".")
        if head in self.local and depth < 5:
            return self.canon(self.local[head] + ("." + rest if rest else ""), depth + 1)
        base = self.map.get(head, head)
        return base + ("." + rest if rest else "")

    def const(self, n):
        """Constant value of a literal, or of a name bound once to a literal; else a sentinel."""
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name) and n.id in self.consts:
            return self.consts[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant) \
                and isinstance(n.operand.value, (int, float)):
            return -n.operand.value
        return _NOCONST

    def fold_str(self, n) -> Optional[str]:
        """String value of a constant / concatenation / f-string built from constants and constant names."""
        v = self.const(n)
        if isinstance(v, str):
            return v
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            a, b = self.fold_str(n.left), self.fold_str(n.right)
            return a + b if a is not None and b is not None else None
        if isinstance(n, ast.JoinedStr):
            out = []
            for part in n.values:
                if isinstance(part, ast.Constant):
                    out.append(str(part.value))
                elif isinstance(part, ast.FormattedValue):
                    v = self.fold_str(part.value)
                    out.append(v if v is not None else "\x00")
            return "".join(out)
        return None


_NOCONST = object()


def kwarg(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def const_true(n) -> bool:
    return isinstance(n, ast.Constant) and n.value is True


def is_const(n) -> bool:
    return isinstance(n, ast.Constant) or (isinstance(n, ast.UnaryOp) and isinstance(n.operand, ast.Constant))


def name_tokens(ident: str) -> List[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", ident)
    return [t for t in re.split(r"[_\W]+", s.lower()) if t]


NON_DATA_TOKENS = {"size", "ratio", "frac", "fraction", "pct", "percent", "mode", "flag", "split", "time", "every"}


def looks_test(ident: str) -> bool:
    toks = name_tokens(ident)
    if any(t in NON_DATA_TOKENS for t in toks):
        return False
    for t in toks:
        t2 = t.rstrip("0123456789")
        if t2 in ("test", "te", "tst", "tests", "testing") or (t2.endswith("test") and len(t2) <= 5) or t2.startswith("test"):
            return True
    return False


TEST_LIT = re.compile(r"(?i)(^|[/\\_.\-])test([/\\_.\-]|s?$)|^test")


def _is_dirty(lab: str) -> bool:
    return lab == "T" or lab[0] in "JS"


class Binding:
    __slots__ = ("line", "labs", "mut", "block", "elems")

    def __init__(self, line, labs, mut, block, elems=None):
        self.line, self.labs, self.mut, self.block, self.elems = line, set(labs), mut, block, elems


class Scope:
    def __init__(self, node, parent: Optional["Scope"], name: str):
        self.node, self.parent, self.name = node, parent, name
        self.params: List[str] = []
        self.env: Dict[str, List[Binding]] = {}
        self.funcs: Dict[str, "Scope"] = {}
        self.stmts: List[ast.AST] = []
        self.blocks: List[Tuple[int, int]] = []      # statement-list spans (if/else/for/while/try/with bodies)
        self.loops: List[Tuple[int, int]] = []
        start = getattr(node, "lineno", 0) if node is not None else 0
        end = getattr(node, "end_lineno", 10 ** 9) if node is not None else 10 ** 9
        self.span = (start if not isinstance(node, ast.Module) else 0, end if not isinstance(node, ast.Module) else 10 ** 9)
        self.globals: Set[str] = set()
        self.is_method = False
        self.called = False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            self.params = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
            if a.vararg:
                self.params.append(a.vararg.arg)
            for p in self.params:
                self.env[p] = [Binding(0, {f"P:{name}:{p}"} | ({"T"} if looks_test(p) else set()), False, self.span)]

    def block_of(self, line: int) -> Tuple[int, int]:
        best = self.span
        for b in self.blocks:
            if b[0] <= line <= b[1] and (b[1] - b[0]) < (best[1] - best[0]):
                best = b
        return best

    def in_same_loop(self, a: int, b: int) -> bool:
        return any(lo <= a <= hi and lo <= b <= hi for lo, hi in self.loops)


COLUMN_AXIS = ("hstack", "column_stack", "dstack")


META_ATTRS = {"shape", "size", "ndim", "dtype", "dtypes", "columns", "nbytes"}


ALLOC = {"zeros", "ones", "empty", "full", "zeros_like", "ones_like", "empty_like", "full_like", "arange", "linspace",
         "eye", "identity", "randn", "rand", "randint", "normal", "uniform", "permutation"}


COMPS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)


class Taint:
    def __init__(self, tree, aliases: Aliases):
        self.al = aliases
        self.scopes: List[Scope] = []
        self.classes: Dict[str, List[Scope]] = {}
        self.methods: Dict[str, List[Scope]] = {}
        self.module = self._build(tree, None, "<module>")
        self._class_cache: Dict[str, Set[str]] = {}
        # a parameter whose NAME reads as test data counts as test data only for functions nobody calls by name
        called_names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        called_attrs = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        for sc in self.scopes:
            nm = getattr(sc.node, "name", None) or sc.name.rsplit(".", 1)[-1]
            if (sc.is_method and nm in called_attrs) or (not sc.is_method and nm in called_names):
                sc.called = True
                for p in sc.params:
                    for b_ in sc.env.get(p, []):
                        b_.labs.discard("T")
        self.summary: Dict[str, List[Set[str]]] = {}
        self.call_join_lines: Set[int] = set()
        self.param_state: Dict[str, Set[str]] = {}      # "P:f:p[#i]" -> concrete dirty labels
        self.param_via: Dict[str, int] = {}

    def _build(self, node, parent, name):
        sc = Scope(node, parent, name)
        self.scopes.append(sc)

        def span(stmts):
            if stmts:
                sc.blocks.append((stmts[0].lineno, getattr(stmts[-1], "end_lineno", stmts[-1].lineno)))

        def collect(stmts):
            for st in stmts:
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    child = self._build(st, sc, st.name if sc.name == "<module>" else f"{sc.name}.{st.name}")
                    sc.funcs[st.name] = child
                elif isinstance(st, ast.ClassDef):
                    for b in st.body:
                        if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            m = self._build(b, sc, f"{st.name}.{b.name}")
                            m.is_method = True
                            self.classes.setdefault(st.name, []).append(m)
                            self.methods.setdefault(b.name, []).append(m)
                    sc.funcs.setdefault("__class__" + st.name, None)
                elif isinstance(st, ast.Global):
                    sc.globals.update(st.names)
                else:
                    if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) \
                            and isinstance(st.value, ast.Lambda):
                        lam = self._build_lambda(st.value, sc, st.targets[0].id)
                        sc.funcs[st.targets[0].id] = lam
                    sc.stmts.append(st)
                    if isinstance(st, (ast.For, ast.AsyncFor, ast.While)):
                        sc.loops.append((st.lineno, getattr(st, "end_lineno", st.lineno)))
                    for fld in ("body", "orelse", "finalbody"):
                        sub = getattr(st, fld, None)
                        if isinstance(sub, list):
                            span(sub)
                            collect(sub)
                    for h in getattr(st, "handlers", []) or []:
                        span(h.body)
                        collect(h.body)
                    if hasattr(ast, "Match") and isinstance(st, ast.Match):
                        for c in st.cases:
                            span(c.body)
                            collect(c.body)
        collect(node.body if hasattr(node, "body") else [])
        return sc

    def _build_lambda(self, lam: ast.Lambda, parent: Scope, name: str) -> Scope:
        sc = Scope(lam, parent, f"{parent.name}.<lambda {name}>" if parent.name != "<module>" else f"<lambda {name}>")
        ret = ast.Return(value=lam.body)
        ast.copy_location(ret, lam)
        sc.stmts.append(ret)
        self.scopes.append(sc)
        return sc

    # ---------------- lookups (flow-aware inside a scope, flow-insensitive across scopes)
    def _visible_binds(self, sc: Scope, binds: List[Binding], line: Optional[int]) -> List[Binding]:
        if line is None:
            return list(binds)
        out: List[Binding] = []
        before = sorted((b for b in binds if b.line <= line), key=lambda b: -b.line)
        for b in before:
            out.append(b)
            if not b.mut and b.block[0] <= line <= b.block[1] and b.line > 0:
                break
        loop_carried = [b for b in binds if b.line > line and sc.in_same_loop(b.line, line)]
        out += loop_carried
        if not before and not loop_carried:
            out = list(binds)
        return out

    def _visible(self, sc: Scope, binds: List[Binding], line: Optional[int]) -> Set[str]:
        out: Set[str] = set()
        for b in self._visible_binds(sc, binds, line):
            out |= b.labs
        return out

    def elem_lookup(self, sc: Scope, key: str, idx, line: Optional[int]) -> Optional[Set[str]]:
        """Labels of key[idx] when key was bound to a literal list / tuple / dict, e.g. the required-paths list
        [train.csv, test.csv] or {k: path(k) for k in ("train", "test")}: paths["train"] is not test data."""
        s, first = sc, True
        while s is not None:
            if key in s.env:
                vis = self._visible_binds(s, s.env[key], line if first else None)
                out: Set[str] = set()
                for b in vis:
                    if b.elems is None or b.mut:
                        return None
                    i = idx
                    if isinstance(i, int) and i < 0:
                        i += b.elems.get("__len__", 0)
                    if i not in b.elems:
                        return None
                    out |= b.elems[i]
                return out if vis else None
            s, first = s.parent, False
        return None

    def _const_scope(self, target, const_node, sc: Scope) -> Scope:
        cs = Scope(None, sc, sc.name)
        for k, _ in self._tkeys(target):
            cs.env[k] = [Binding(0, self.labels(const_node, sc), False, (0, 10 ** 9))]
        return cs

    def elem_labels(self, value, sc: Scope) -> Optional[Dict]:
        if isinstance(value, (ast.List, ast.Tuple)) and not any(isinstance(e, ast.Starred) for e in value.elts):
            d = {i: self.labels(e, sc) for i, e in enumerate(value.elts)}
            d["__len__"] = len(value.elts)
            return d
        if isinstance(value, ast.Dict) and all(isinstance(k, ast.Constant) for k in value.keys):
            return {k.value: self.labels(v, sc) for k, v in zip(value.keys, value.values)}
        if isinstance(value, (ast.ListComp, ast.DictComp)) and len(value.generators) == 1 and not value.generators[0].ifs:
            g = value.generators[0]
            if isinstance(g.iter, (ast.List, ast.Tuple)) and all(isinstance(e, ast.Constant) for e in g.iter.elts):
                if isinstance(value, ast.ListComp):
                    d = {i: self.labels(value.elt, self._const_scope(g.target, e, sc)) for i, e in enumerate(g.iter.elts)}
                    d["__len__"] = len(g.iter.elts)
                    return d
                if isinstance(value.key, ast.Name) and isinstance(g.target, ast.Name) and value.key.id == g.target.id:
                    return {e.value: self.labels(value.value, self._const_scope(g.target, e, sc)) for e in g.iter.elts}
        return None

    def lookup(self, sc: Scope, key: str, line: Optional[int] = None) -> Set[str]:
        parts = key.split(".")
        for i in range(len(parts), 0, -1):
            p = ".".join(parts[:i])
            s, first = sc, True
            while s is not None:
                if p in s.env:
                    return self._visible(s, s.env[p], line if first else None)
                s, first = s.parent, False
            if looks_test(parts[i - 1]):
                return {"T"}
        return set()

    def callee(self, fn, sc: Scope) -> Optional[Scope]:
        if isinstance(fn, ast.Attribute):
            cands = self.methods.get(fn.attr, [])
            return cands[0] if len(cands) == 1 and fn.attr != "__init__" else None
        if not isinstance(fn, ast.Name):
            return None
        s = sc
        while s is not None:
            if fn.id in s.funcs and s.funcs[fn.id] is not None:
                return s.funcs[fn.id]
            s = s.parent
        return None

    def class_labels(self, cname: str) -> Set[str]:
        """Labels a local class's instances carry from the module data its methods read (e.g. a torch Dataset whose
        __getitem__ indexes the test frame)."""
        if cname in self._class_cache:
            return self._class_cache[cname]
        self._class_cache[cname] = set()
        out: Set[str] = set()
        for m in self.classes.get(cname, []):
            for st in m.stmts:
                for nd in ast.walk(st):
                    if isinstance(nd, ast.Name) and isinstance(nd.ctx, ast.Load) and nd.id not in m.env and nd.id != "self":
                        out |= {l for l in self.lookup(m, nd.id) if _is_dirty(l)}
        self._class_cache[cname] = out
        return out

    @staticmethod
    def _select(labs: Set[str], sub: ast.Subscript) -> Set[str]:
        sl = sub.slice
        col = (isinstance(sl, ast.Constant) and isinstance(sl.value, str)) or \
              (isinstance(sl, ast.List) and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in sl.elts)) or \
              (isinstance(sl, ast.Tuple) and sl.elts and isinstance(sl.elts[0], ast.Slice) and sl.elts[0].lower is None and sl.elts[0].upper is None)
        if col:
            return labs
        return {("S" + l[1:]) if l.startswith("J") else l for l in labs}

    def _comp_scope(self, n, sc: Scope) -> Scope:
        cs = Scope(None, sc, sc.name)
        cs.funcs = {}
        for g in n.generators:
            elts = g.target.elts if isinstance(g.target, (ast.Tuple, ast.List)) else [g.target]
            per = self._iter_labels(g.iter, len(elts), cs)
            for i, e in enumerate(elts):
                for k, _ in self._tkeys(e):
                    cs.env.setdefault(k, []).append(Binding(0, per[i] if i < len(per) else per[-1], False, (0, 10 ** 9)))
        return cs

    def labels(self, n, sc: Scope) -> Set[str]:
        if n is None:
            return set()
        if isinstance(n, COMPS):
            cs = self._comp_scope(n, sc)
            out: Set[str] = set()
            for g in n.generators:
                out |= self.labels(g.iter, cs)
            parts = [n.key, n.value] if isinstance(n, ast.DictComp) else [n.elt]
            for x in parts:
                out |= self.labels(x, cs)
            return out
        if isinstance(n, ast.Call):
            fname = dotted(n.func) or ""
            if fname in ("len", "range", "id", "type", "isinstance", "hash"):
                return set()
            short = fname.split(".")[-1]
            if short in ALLOC and (fname.startswith(("np.", "numpy.", "torch.", "jnp.")) or "." not in fname or short.endswith("_like")):
                fill = kwarg(n, "fill_value")
                if fill is None and short in ("full", "full_like") and len(n.args) > 1:
                    fill = n.args[1]
                return self.labels(fill, sc) if fill is not None else set()   # shapes are not data
            if isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args and isinstance(n.func.value, (ast.Name, ast.Attribute)):
                k = self.al.const(n.args[0])
                if k is not _NOCONST and isinstance(k, (str, int)):
                    el = self.elem_lookup(sc, dotted(n.func.value) or "", k, getattr(n, "lineno", None))
                    if el is not None:
                        return el | (self.labels(n.args[1], sc) if len(n.args) > 1 else set())
            if isinstance(n.func, ast.Name) and n.func.id in self.classes:
                out = self.class_labels(n.func.id)
                for a in n.args:
                    out = out | self.labels(a, sc)
                for k in n.keywords:
                    out = out | self.labels(k.value, sc)
                return out
            cal = self.callee(n.func, sc)
            if cal is not None:
                out = set()
                for s_ in self.resolve(cal, n, sc):
                    out |= s_
                if isinstance(n.func, ast.Attribute):
                    out |= self.labels(n.func.value, sc)
                return out
            out = set()
            if isinstance(n.func, ast.Attribute) and n.func.attr not in APPLY_METHODS:
                out |= self.labels(n.func.value, sc)       # a fitted object's own contamination is reported at its fit
            for a in n.args:
                out |= self.labels(a, sc)
            for k in n.keywords:
                out |= self.labels(k.value, sc)
            j = self._join_node(n, sc)
            if j:
                out = {l for l in out if l != "T"} | {f"J{j}"}
            return out
        if isinstance(n, ast.Attribute) and n.attr in META_ATTRS:
            return set()
        if isinstance(n, (ast.Name, ast.Attribute)):
            key = dotted(n)
            out = self.lookup(sc, key, getattr(n, "lineno", None)) if key else set()
            if isinstance(n, ast.Attribute) and not isinstance(n.value, (ast.Name, ast.Attribute)):
                out |= self.labels(n.value, sc)
            return out
        if isinstance(n, ast.Constant):
            return {"T"} if isinstance(n.value, str) and len(n.value) < 200 and TEST_LIT.search(n.value) else set()
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) and any(
                isinstance(x, (ast.List, ast.Tuple)) and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in x.elts)
                for x in (n.left, n.right)):
            return set()                                   # ["test"] * len(train): a label column, not test rows
        if isinstance(n, ast.Compare) and all(isinstance(c, ast.Constant) for c in n.comparators):
            return self.labels(n.left, sc)                 # y == "test" selects by a label value
        if isinstance(n, ast.Subscript):
            sl = n.slice
            base = dotted(n.value) or ""
            if base.endswith(("r_", "c_")) and base.split(".")[-1] in ("r_", "c_"):
                elems = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
                labs_ = [self.labels(e, sc) for e in elems]
                out = set().union(*labs_) if labs_ else set()
                if base.endswith("r_") and self._mixes(labs_, elems):
                    self.call_join_lines.add(n.lineno)
                    out = {x for x in out if x != "T"} | {f"J{n.lineno}"}
                return out
            if isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.USub) and isinstance(sl.operand, ast.Constant):
                sl = ast.Constant(-sl.operand.value)
            if isinstance(sl, ast.Name):
                cv = self.al.const(sl)
                if cv is not _NOCONST and isinstance(cv, (str, int)):
                    sl = ast.Constant(cv)
            if isinstance(sl, ast.Constant) and isinstance(sl.value, (int, str)) and isinstance(n.value, (ast.Name, ast.Attribute)):
                el = self.elem_lookup(sc, dotted(n.value) or "", sl.value, getattr(n, "lineno", None))
                if el is not None:
                    return el
            return self._select(self.labels(n.value, sc), n)
        if isinstance(n, ast.Lambda):
            return set()
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.BitOr)):
            l, r = self.labels(n.left, sc), self.labels(n.right, sc)
            out = l | r
            if self._mixes([l, r], [n.left, n.right]):
                out = {x for x in out if x != "T"} | {f"J{n.lineno}"}
            return out
        out = set()
        for ch in ast.iter_child_nodes(n):
            out |= self.labels(ch, sc)
        return out

    @staticmethod
    def _mixes(label_sets, nodes) -> bool:
        dirty = clean = False
        for labs, nd in zip(label_sets, nodes):
            if is_const(nd):
                continue
            if any(_is_dirty(x) for x in labs):
                dirty = True
            elif not any(x.startswith("P:") for x in labs):
                clean = True
        return dirty and clean

    def _join_node(self, n: ast.Call, sc) -> Optional[int]:
        last = (dotted(n.func) or "").split(".")[-1]
        if last not in CONCAT_CALLS or last in COLUMN_AXIS:
            return None
        ax = kwarg(n, "axis") or kwarg(n, "dim")
        if ax is None and len(n.args) >= 2 and last in ("concatenate", "cat", "stack", "concat"):
            ax = n.args[1]
        if ax is not None:
            v = ax.value if isinstance(ax, ast.Constant) else (-ax.operand.value if isinstance(ax, ast.UnaryOp) and isinstance(ax.operand, ast.Constant) else None)
            if v in (1, -1, "columns"):
                return None
        elems: List[ast.AST] = []
        for a in n.args[:1]:
            elems += a.elts if isinstance(a, (ast.List, ast.Tuple)) else [a]
        if last in ("append", "chain") and isinstance(n.func, ast.Attribute):
            elems = list(n.args) + [n.func.value]
        if len(elems) < 2:
            return None
        if self._mixes([self.labels(e, sc) for e in elems], elems):
            self.call_join_lines.add(n.lineno)
            return n.lineno
        return None

    @staticmethod
    def _params_for(cal: Scope, call: ast.Call) -> List[str]:
        return cal.params[1:] if cal.is_method and isinstance(call.func, ast.Attribute) and cal.params else cal.params

    @staticmethod
    def _expand_args(call: ast.Call):
        out, star = [], []
        for a in call.args:
            if isinstance(a, ast.Starred):
                if isinstance(a.value, (ast.List, ast.Tuple)):
                    out += list(a.value.elts)
                else:
                    star.append(a.value)
            else:
                out.append(a)
        return out, star

    def resolve(self, cal: Scope, call: ast.Call, sc: Scope) -> List[Set[str]]:
        summ = self.summary.get(cal.name) or [set()]
        argmap: Dict[str, ast.AST] = {}
        params = self._params_for(cal, call)
        args, star = self._expand_args(call)
        for i, a in enumerate(args):
            if i < len(params):
                argmap[params[i]] = a
        for k in call.keywords:
            if k.arg in cal.params:
                argmap[k.arg] = k.value
            elif k.arg is None:
                star.append(k.value)
        pre = f"P:{cal.name}:"
        out = []
        for labs in summ:
            r: Set[str] = set()
            for lab in labs:
                if lab.startswith(pre):
                    pname, _, idx = lab[len(pre):].partition("#")
                    a = argmap.get(pname)
                    if a is None:
                        for st in star:
                            r |= self.labels(st, sc)
                        continue
                    if idx and isinstance(a, (ast.Tuple, ast.List)) and int(idx) < len(a.elts):
                        r |= self.labels(a.elts[int(idx)], sc)
                    else:
                        r |= self.labels(a, sc)
                else:
                    r.add(lab)
            out.append(r)
        return out

    # ---------------- statements
    def _bind(self, sc, key: str, labs: Set[str], line: int, mut=False, elems=None) -> bool:
        if sc.globals and key.split(".")[0] in sc.globals:
            sc = self.module
        binds = sc.env.setdefault(key, [])
        for b in binds:
            if b.line == line and b.mut == mut:
                changed = False
                if labs - b.labs:
                    b.labs |= labs
                    changed = True
                if elems is not None:
                    if b.elems is None:
                        b.elems = dict(elems)
                        changed = True
                    else:
                        for k2, v2 in elems.items():
                            if k2 == "__len__":
                                b.elems[k2] = v2
                            elif k2 not in b.elems or (v2 - b.elems[k2]):
                                b.elems[k2] = set(v2) | b.elems.get(k2, set())
                                changed = True
                return changed
        binds.append(Binding(line, labs, mut, sc.block_of(line), dict(elems) if elems is not None else None))
        return True

    def _tkeys(self, tgt) -> List[Tuple[str, bool]]:
        if isinstance(tgt, (ast.Name, ast.Attribute)):
            k = dotted(tgt)
            return [(k, False)] if k else []
        if isinstance(tgt, ast.Subscript):
            k = dotted(tgt.value)
            return [(k, True)] if k else []
        if isinstance(tgt, ast.Starred):
            return self._tkeys(tgt.value)
        out = []
        if isinstance(tgt, (ast.Tuple, ast.List)):
            for e in tgt.elts:
                out += self._tkeys(e)
        return out

    def _assign(self, sc, tgt, value, line) -> bool:
        ch = False
        if isinstance(tgt, (ast.Tuple, ast.List)):
            per: Optional[List[Set[str]]] = None
            cal = self.callee(value.func, sc) if isinstance(value, ast.Call) else None
            if cal is not None:
                per = self.resolve(cal, value, sc)
            elif isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(tgt.elts):
                per = [self.labels(e, sc) for e in value.elts]
            elif isinstance(value, ast.Name) and value.id in sc.params and len(sc.env.get(value.id, [])) == 1:
                per = [{f"P:{sc.name}:{value.id}#{i}"} for i in range(len(tgt.elts))]
            elif isinstance(value, (ast.Name, ast.Attribute)):
                els = [self.elem_lookup(sc, dotted(value) or "", i, line) for i in range(len(tgt.elts))]
                if all(e is not None for e in els):
                    per = els
            if per is not None:
                allu = set().union(*per) if per else set()
                for i, e in enumerate(tgt.elts):
                    labs = per[i] if len(per) == len(tgt.elts) else allu
                    for k, mut in self._tkeys(e):
                        ch |= self._bind(sc, k, labs, line, mut)
                return ch
        labs = self.labels(value, sc)
        elems = self.elem_labels(value, sc) if isinstance(tgt, (ast.Name, ast.Attribute)) else None
        for k, mut in self._tkeys(tgt):
            ch |= self._bind(sc, k, labs, line, mut, elems if not mut else None)
        return ch

    def _iter_labels(self, it, n_targets: int, sc) -> List[Set[str]]:
        last = (dotted(it.func) or "").split(".")[-1] if isinstance(it, ast.Call) else ""
        if last == "zip" and n_targets == len(it.args):
            return [self.labels(a, sc) for a in it.args]
        if last == "enumerate" and n_targets == 2 and it.args:
            return [set(), self.labels(it.args[0], sc)]
        if isinstance(it, (ast.List, ast.Tuple)) and it.elts and all(isinstance(e, (ast.Tuple, ast.List)) and len(e.elts) == n_targets for e in it.elts):
            per = [set() for _ in range(n_targets)]
            for e in it.elts:
                for i, x in enumerate(e.elts):
                    per[i] |= self.labels(x, sc)
            return per
        labs = self.labels(it, sc)
        return [labs] * max(n_targets, 1)

    @staticmethod
    def _split_iter(it) -> bool:
        for n in ast.walk(it):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "split":
                return True
        return False

    def _scope_pass(self, sc: Scope) -> bool:
        ch = False
        node = sc.node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and sc.parent is not None:
            a = node.args
            pos = a.posonlyargs + a.args
            for p_, d_ in zip(pos[len(pos) - len(a.defaults):], a.defaults):
                ch |= self._bind(sc, p_.arg, self.labels(d_, sc.parent), 0)
            for p_, d_ in zip(a.kwonlyargs, a.kw_defaults):
                if d_ is not None:
                    ch |= self._bind(sc, p_.arg, self.labels(d_, sc.parent), 0)
        for st in sc.stmts:
            line = getattr(st, "lineno", 0)
            if isinstance(st, ast.Assign):
                for tg in st.targets:
                    ch |= self._assign(sc, tg, st.value, line)
            elif isinstance(st, ast.AnnAssign) and st.value is not None:
                ch |= self._assign(sc, st.target, st.value, line)
            elif isinstance(st, ast.AugAssign):
                labs = self.labels(st.value, sc)
                for k, _ in self._tkeys(st.target):
                    cur = self.lookup(sc, k, line)
                    if self._mixes([cur, labs], [st.target, st.value]):
                        labs = {x for x in labs if x != "T"} | {f"J{line}"}
                    ch |= self._bind(sc, k, labs, line, True)
            elif isinstance(st, (ast.For, ast.AsyncFor)):
                if self._split_iter(st.iter):
                    for k, _ in self._tkeys(st.target):
                        ch |= self._bind(sc, k, set(), line)
                else:
                    elts = st.target.elts if isinstance(st.target, (ast.Tuple, ast.List)) else [st.target]
                    per = self._iter_labels(st.iter, len(elts), sc)
                    for i, e in enumerate(elts):
                        for k, mut in self._tkeys(e):
                            ch |= self._bind(sc, k, per[i] if i < len(per) else per[-1], line, mut)
            elif isinstance(st, (ast.With, ast.AsyncWith)):
                for item in st.items:
                    if item.optional_vars is not None:
                        ch |= self._assign(sc, item.optional_vars, item.context_expr, line)
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
                c = st.value
                recv = None
                if isinstance(c.func, ast.Name) and "." in self.al.canon(c.func.id):
                    full = self.al.canon(c.func.id)
                    if full.rsplit(".", 1)[1] in ("append", "extend", "update", "insert", "add"):
                        recv = full.rsplit(".", 1)[0]
                if isinstance(c.func, ast.Attribute) and c.func.attr in self.methods and isinstance(c.func.value, (ast.Name, ast.Attribute)):
                    labs = set()
                    for a_ in c.args:
                        labs |= self.labels(a_, sc)
                    for k_ in c.keywords:
                        labs |= self.labels(k_.value, sc)
                    if labs:
                        ch |= self._bind(sc, dotted(c.func.value), labs, line, True)
                if recv is not None or (isinstance(c.func, ast.Attribute) and c.func.attr in ("append", "extend", "update", "insert", "add")):
                    recv = recv or dotted(c.func.value)
                    if recv:
                        labs = set()
                        for a in c.args:
                            labs |= self.labels(a, sc)
                        cur = self.lookup(sc, recv, line)
                        recv_node = c.func.value if isinstance(c.func, ast.Attribute) else c.func
                        if self._mixes([cur, labs], [recv_node] + list(c.args[:1])):
                            labs = {x for x in labs if x != "T"} | {f"J{line}"}
                        ch |= self._bind(sc, recv, labs, line, True)
            if not isinstance(st, (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try)):
                for n in ast.walk(st):
                    if isinstance(n, ast.NamedExpr):
                        ch |= self._assign(sc, n.target, n.value, line)
        return ch

    def _summaries(self) -> bool:
        ch = False
        for sc in self.scopes:
            if sc is self.module or not isinstance(sc.node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            rets = [st.value for st in sc.stmts if isinstance(st, ast.Return) and st.value is not None]
            for st in sc.stmts:
                for nd in ast.walk(st):
                    if isinstance(nd, (ast.Yield, ast.YieldFrom)) and nd.value is not None:
                        rets.append(nd.value)
            if not rets:
                continue
            width = max(len(r.elts) if isinstance(r, ast.Tuple) else 1 for r in rets)
            per = [set() for _ in range(width)]
            for r in rets:
                if isinstance(r, ast.Tuple) and len(r.elts) == width:
                    for i, e in enumerate(r.elts):
                        per[i] |= self.labels(e, sc)
                else:
                    labs = self.labels(r, sc)
                    for i in range(width):
                        per[i] |= labs
            old = self.summary.get(sc.name)
            if old is None or len(old) != len(per) or any(a != b for a, b in zip(old, per)):
                self.summary[sc.name] = per
                ch = True
        return ch

    def _concretise(self) -> bool:
        ch = False
        for sc in self.scopes:
            for st in sc.stmts:
                for n in ast.walk(st):
                    if not isinstance(n, ast.Call):
                        continue
                    cal = self.callee(n.func, sc)
                    if cal is None:
                        continue
                    params = self._params_for(cal, n)
                    args, _ = self._expand_args(n)
                    pairs = [(params[i], a) for i, a in enumerate(args) if i < len(params)]
                    pairs += [(k.arg, k.value) for k in n.keywords if k.arg in params]
                    for p, a in pairs:
                        keys = [(f"P:{cal.name}:{p}", a)]
                        if isinstance(a, (ast.Tuple, ast.List)):
                            keys += [(f"P:{cal.name}:{p}#{i}", e) for i, e in enumerate(a.elts)]
                        for key, expr in keys:
                            conc = self.concrete(self.labels(expr, sc))
                            cur = self.param_state.setdefault(key, set())
                            if conc - cur:
                                cur |= conc
                                self.param_via.setdefault(key, n.lineno)
                                ch = True
        return ch

    def concrete(self, labs: Set[str]) -> Set[str]:
        out = {l for l in labs if _is_dirty(l)}
        for l in labs:
            if l.startswith("P:"):
                out |= self.param_state.get(l, set())
                if "#" not in l:
                    for k, v in self.param_state.items():
                        if k.startswith(l + "#"):
                            out |= v
        return out

    def run(self):
        for _ in range(16):
            c = False
            for sc in self.scopes:
                c |= self._scope_pass(sc)
            c |= self._summaries()
            if not c:
                break
        for _ in range(16):
            if not self._concretise():
                break
        return self

    def scope_of_line(self, line: int) -> Scope:
        best = self.module
        for sc in self.scopes:
            n = sc.node
            if hasattr(n, "lineno") and n.lineno <= line <= getattr(n, "end_lineno", n.lineno):
                if best is self.module or n.lineno >= best.node.lineno:
                    best = sc
        return best

    def contributors(self, expr, sc) -> List[str]:
        out = []
        for n in ast.walk(expr):
            if isinstance(n, (ast.Name, ast.Attribute)) and isinstance(getattr(n, "ctx", None), ast.Load):
                k = dotted(n)
                if k and self.concrete(self.lookup(sc, k, getattr(n, "lineno", None))):
                    out.append(k)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 200 and TEST_LIT.search(n.value):
                out.append(repr(n.value))
        return sorted(set(out))


# ----------------------------------------------------------------------------- findings

WHY = {
    "test_fit": "A transform, vocabulary, clustering or model fitted on test rows learns from the data it is later scored "
                "on. Validation numbers stop meaning anything, and production will not have those statistics.",
    "test_impute": "Filling missing values with test-set statistics (mean, median) leaks test information into features.",
    "test_gradient": "Training steps on test batches (pseudo-labelling, test-time training) fit the model to the test set.",
    "joint": "Train and test rows are combined into one array. Any fit on it, or on a slice of it, may see test rows; "
             "keeping X_train and X_test separate makes that impossible.",
    "cross_test": "A statistic computed across test rows makes each prediction depend on the other test rows.",
    "seed": "Without fixed seeds two runs give different models and different validation scores; experiment comparisons become noise.",
    "random_state": "This estimator / splitter is random; without random_state its results change from run to run.",
    "dataloader": "A shuffling DataLoader without a seeded generator gives a different training order every run.",
}

FIXES = {
    "test_fit": "fit on the training rows only, then call .transform() on the test rows",
    "test_impute": "compute the fill values on the training rows (train.median()) and reuse them for test",
    "test_gradient": "train only on training batches; use the test loader only inside torch.no_grad() for predictions",
    "joint": "keep separate X_train / X_test arrays and fit transforms on X_train",
    "cross_test": "compute group statistics on training rows, or per sample",
    "seed": "seed random, numpy and torch once at the top (random.seed(42); np.random.seed(42); torch.manual_seed(42))",
    "random_state": "pass random_state=SEED",
    "dataloader": "DataLoader(..., generator=torch.Generator().manual_seed(SEED))",
}

CLUSTERERS = {"KMeans", "MiniBatchKMeans", "SpectralClustering", "AgglomerativeClustering", "DBSCAN", "HDBSCAN",
              "OPTICS", "Birch", "AffinityPropagation", "MeanShift", "GaussianMixture"}


@dataclass
class LintFinding:
    level: str
    rule: str
    line: int
    msg: str
    where: str = ""

    def to_dict(self):
        return {"level": self.level, "rule": self.rule, "line": self.line, "where": self.where or f"line {self.line}",
                "msg": self.msg, "why": WHY.get(self.rule, ""), "fix": FIXES.get(self.rule, "")}


@dataclass
class FitRow:
    line: int
    call: str
    sees: str
    where: str = ""


@dataclass
class LintReport:
    path: str
    findings: List[LintFinding] = field(default_factory=list)
    fits: List[FitRow] = field(default_factory=list)
    _seen: Set[Tuple] = field(default_factory=set)

    def add(self, level, rule, line, msg):
        k = (level, rule, line, msg)
        if k not in self._seen:
            self._seen.add(k)
            self.findings.append(LintFinding(level, rule, line, msg))

    @property
    def errors(self):
        return [f for f in self.findings if f.level == "ERROR"]

    @property
    def warnings(self):
        return [f for f in self.findings if f.level == "WARN"]


# ----------------------------------------------------------------------------- notebooks

def notebook_source(nb_json: str) -> Tuple[str, Dict[int, str]]:
    """Code cells joined into one script (magics blanked so line numbers stay aligned) and a line -> 'cell N, line M' map."""
    nb = json.loads(nb_json)
    lines, where = [], {}
    k = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        k += 1
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        for j, ln in enumerate(src.splitlines(), 1):
            stripped = ln.lstrip()
            if stripped.startswith(("%", "!")) or stripped.startswith("?"):
                ln = ""
            lines.append(ln)
            where[len(lines)] = f"cell {k}, line {j}"
        lines.append("")
    return "\n".join(lines) + "\n", where


# ----------------------------------------------------------------------------- checks

def lint_source(src: str, path: str = "<script>", repro: bool = True, where: Optional[Dict[int, str]] = None) -> LintReport:
    rep = LintReport(path=path)
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        rep.add("ERROR", "syntax", e.lineno or 0, f"cannot parse: {e.msg}")
        return rep
    al = Aliases(tree)
    parent: Dict[int, ast.AST] = {}
    for nd in ast.walk(tree):
        for ch in ast.iter_child_nodes(nd):
            parent[id(ch)] = nd
    tnt = Taint(tree, al).run()

    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        raw = dotted(n.func) or ""
        cname = al.canon(raw)
        last = cname.split(".")[-1]
        sc = tnt.scope_of_line(n.lineno)
        comps, cur = [], parent.get(id(n))
        while cur is not None and not isinstance(cur, ast.stmt):
            if isinstance(cur, COMPS):
                comps.append(cur)
            cur = parent.get(id(cur))
        for comp in reversed(comps):
            sc = tnt._comp_scope(comp, sc)
        skip_kw = ("sample_weight", "eval_set", "callbacks", "valid_sets", "valid_names", "eval_names", "init_model")
        args = list(n.args) + [k.value for k in n.keywords if k.arg not in skip_kw]
        alias_full = al.canon(n.func.id) if isinstance(n.func, ast.Name) else ""
        sink, rule = None, "test_fit"
        if isinstance(n.func, ast.Attribute) and n.func.attr in FIT_ATTRS:
            sink = f".{n.func.attr}()"
        elif "." in alias_full and alias_full.rsplit(".", 1)[1] in FIT_ATTRS:
            sink = f"{n.func.id}() = {alias_full}()"
        elif last == "partial" and n.args and (al.canon(dotted(n.args[0]) or "").split(".")[-1] in FIT_ATTRS):
            sink, args = f"partial({dotted(n.args[0])})", list(n.args[1:]) + [k.value for k in n.keywords]
        elif isinstance(n.func, ast.Attribute) and n.func.attr == "backward":
            sink, args, rule = ".backward()", [n.func.value], "test_gradient"
        elif isinstance(n.func, ast.Attribute) and n.func.attr == "train" and n.args:
            sink = ".train(...)"
        elif last in UNSUP_CTORS and args:
            sink = f"{last}(...)"
        elif cname in NAN_STATS:
            ax = kwarg(n, "axis") if kwarg(n, "axis") is not None else (n.args[1] if len(n.args) > 1 else None)
            axv = al.const(ax) if ax is not None else None
            a0 = n.args[0] if n.args else None
            if axv in (1, -1) or a0 is None:
                continue
            direct = any(_is_dirty(l) for l in tnt.labels(a0, sc))
            sink, args, rule = f"{last}()" + ("" if direct and axv in (0, None) else "~"), [a0], "test_impute"
        elif isinstance(n.func, ast.Attribute) and n.func.attr == "fillna" and (n.args or kwarg(n, "value") is not None):
            val = kwarg(n, "value") if kwarg(n, "value") is not None else n.args[0]
            if isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute) and val.func.attr in ("mean", "median", "mode", "quantile"):
                sink, args, rule = "fillna(<statistic>)", [val.func.value], "test_impute"
            elif not is_const(val):
                sink, args, rule = "fillna(<value>)", [val], "test_impute"
        if sink is None:
            if last in PAIRWISE and len(n.args) >= 2 and all(tnt.concrete(tnt.labels(a, sc)) for a in n.args[:2]):
                rep.add("WARN", "cross_test", n.lineno, f"`{last}` between test rows")
            if isinstance(n.func, ast.Attribute) and n.func.attr == "groupby" and "T" in tnt.concrete(tnt.labels(n.func.value, sc)):
                up = parent.get(id(n))
                if isinstance(up, ast.Subscript):
                    up = parent.get(id(up))
                if isinstance(up, ast.Attribute) and up.attr in ("transform", "mean", "median", "std", "var", "sum", "agg",
                                                                 "aggregate", "count", "size", "nunique", "rank", "quantile",
                                                                 "min", "max", "cumsum", "first", "last", "apply"):
                    rep.add("WARN", "cross_test", n.lineno, f"groupby(...).{up.attr} over test rows")
            continue
        labs: Set[str] = set()
        contrib: List[str] = []
        for a in args:
            labs |= tnt.labels(a, sc)
            contrib += tnt.contributors(a, sc)
        conc = tnt.concrete(labs)
        seg = re.sub(r"\s+", " ", ast.get_source_segment(src, n) or raw)
        seg = seg if len(seg) <= 100 else seg[:97] + "..."
        if not conc:
            rep.fits.append(FitRow(n.lineno, seg, "train rows only"))
            continue
        joins = sorted({int(l[1:]) for l in conc if l[0] == "J"})
        sels = sorted({int(l[1:]) for l in conc if l[0] == "S"})
        if "T" in conc and not joins:
            sees = "TEST rows"
        elif joins:
            sees = f"TRAIN+TEST rows (joined at line {joins[0]})"
        else:
            sees = f"a slice of a TRAIN+TEST array (joined at line {sels[0]})"
        level = "ERROR"
        msg = f"{sink.rstrip('~')} sees {sees}"
        if sink.endswith("~"):
            level = "WARN"
            msg += "; fine if the statistic stays inside one sample"
        if rule == "test_gradient":
            direct = set()
            for a in args:
                direct |= tnt.labels(a, sc)
            loops_over_test, cur = False, parent.get(id(n))
            while cur is not None and not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                if isinstance(cur, (ast.For, ast.AsyncFor)) and "T" in tnt.labels(cur.iter, tnt.scope_of_line(cur.lineno)):
                    loops_over_test = True
                cur = parent.get(id(cur))
            pure = "T" in direct and not any(l[0] in "JS" for l in direct if l != "T")
            if not (pure and loops_over_test):
                level, msg = "WARN", msg + " (indirect: check that the loss uses training batches only)"
        if sink == ".fit_predict()" and sees == "TEST rows":
            looped, cur = False, parent.get(id(n))
            while cur is not None and not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                if isinstance(cur, (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.GeneratorExp)):
                    looped = True
                cur = parent.get(id(cur))
            if looped:
                level, msg = "WARN", msg + "; clustering inside one test group is fine only if that group is one sample"
        if sels and not joins and "T" not in conc:
            msg += "; prove the index keeps training rows only, or keep X_train / X_test separate"
        vias = set()
        for k in labs:
            if k.startswith("P:") and tnt.concrete({k}):
                for kk in tnt.param_via:
                    if (kk == k or kk.startswith(k + "#")) and tnt.param_state.get(kk):
                        vias.add(f"{kk.split(':', 2)[2]} <- line {tnt.param_via[kk]}")
        if vias:
            msg += f"; via {sc.name}() called with test data ({', '.join(sorted(vias))})"
        if contrib:
            msg += f"; from {', '.join(sorted(set(contrib)))[:100]}"
        rep.fits.append(FitRow(n.lineno, seg, sees))
        rep.add(level, rule, n.lineno, msg)
    for ln in sorted(tnt.call_join_lines):
        rep.add("WARN", "joint", ln, "train and test rows are joined into one array here")

    if repro:
        _repro_checks(tree, al, src, rep)
    if where:
        for f in rep.findings:
            f.where = where.get(f.line, "")
        for r in rep.fits:
            r.where = where.get(r.line, "")
    rep.findings.sort(key=lambda f: ({"ERROR": 0, "WARN": 1, "INFO": 2}[f.level], f.line))
    return rep


def _repro_checks(tree, al, src, rep):
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    calls = set()
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        cname = al.canon(dotted(n.func) or "")
        calls.add(cname)
        last = cname.split(".")[-1]
        if last in NEEDS_RS and not any(k.arg in SEED_KW for k in n.keywords) and not any(k.arg is None for k in n.keywords):
            if not (last == "train_test_split" and isinstance(kwarg(n, "shuffle"), ast.Constant) and kwarg(n, "shuffle").value is False):
                rep.add("WARN", "random_state", n.lineno, f"`{last}(...)` without random_state")
        if last in SHUFFLE_SPLITTERS and const_true(kwarg(n, "shuffle")) and kwarg(n, "random_state") is None:
            rep.add("WARN", "random_state", n.lineno, f"`{last}(shuffle=True)` without random_state")
        if last == "DataLoader":
            sh = kwarg(n, "shuffle")
            if sh is not None and al.const(sh) is True and kwarg(n, "generator") is None and kwarg(n, "sampler") is None:
                rep.add("WARN", "dataloader", n.lineno, "shuffle=True without a seeded generator")
    uses_np_random = bool(re.search(r"\bnp\.random\.|numpy\.random\.", src))
    if "torch" in imported and "torch.manual_seed" not in calls and "lightning.seed_everything" not in calls \
            and not any(c.endswith("seed_everything") for c in calls):
        rep.add("WARN", "seed", 0, "torch is used but torch.manual_seed(...) is never called")
    if uses_np_random and "numpy.random.seed" not in calls and not re.search(r"default_rng\(\s*[^)\s]", src):
        rep.add("WARN", "seed", 0, "numpy randomness is used but never seeded (np.random.seed or default_rng(seed))")


# ----------------------------------------------------------------------------- output

def lint_path(path: str, repro: bool = True) -> LintReport:
    raw = open(path, encoding="utf-8", errors="replace").read()
    if path.endswith(".ipynb"):
        src, where = notebook_source(raw)
        return lint_source(src, path, repro, where)
    return lint_source(raw, path, repro)


def format_report(rep: LintReport, color: bool = False) -> str:
    C = {"ERROR": "\033[91m", "WARN": "\033[93m", "INFO": "\033[90m"} if color else {}
    R = "\033[0m" if color else ""
    L = [f"leakproof lint: {rep.path}  {len(rep.errors)} error(s), {len(rep.warnings)} warning(s), {len(rep.fits)} fit(s) checked"]
    if not rep.findings:
        L.append("no test-set leakage found in the fits this script makes")
    for f in rep.findings:
        loc = f.where or (f"line {f.line}" if f.line else "file")
        L.append(f"{C.get(f.level, '')}{f.level:<5}{R} {loc:>16}  [{f.rule}] {f.msg}")
    rules = sorted({f.rule for f in rep.findings})
    if rules:
        L.append("")
        for r in rules:
            L.append(f"  [{r}] {WHY.get(r, '')}")
            if FIXES.get(r):
                L.append(f"        fix: {FIXES[r]}")
    if rep.fits:
        L.append("")
        L.append("fits and the rows they see:")
        for r in sorted(rep.fits, key=lambda x: x.line):
            L.append(f"  {(r.where or 'line ' + str(r.line)):>16}  {r.call}")
            L.append(f"  {'':>16}    -> {r.sees}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="leakproof lint", description="Find test-set leakage in a training script or notebook")
    ap.add_argument("paths", nargs="+", help=".py or .ipynb files")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fail-on", choices=["error", "warn", "never"], default="error")
    ap.add_argument("--no-repro", action="store_true", help="skip the reproducibility (seed / random_state) checks")
    a = ap.parse_args(argv)
    reports = [lint_path(p, repro=not a.no_repro) for p in a.paths]
    if a.json:
        print(json.dumps([{"path": r.path, "errors": len(r.errors), "warnings": len(r.warnings),
                           "findings": [f.to_dict() for f in r.findings],
                           "fits": [{"line": x.line, "where": x.where, "call": x.call, "sees": x.sees} for x in r.fits]}
                          for r in reports], indent=1))
    else:
        print("\n\n".join(format_report(r, color=sys.stdout.isatty()) for r in reports))
    bad = sum(len(r.errors) + (len(r.warnings) if a.fail_on == "warn" else 0) for r in reports)
    return 1 if (bad and a.fail_on != "never") else 0


if __name__ == "__main__":
    sys.exit(main())
