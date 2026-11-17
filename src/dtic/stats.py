"""The tests behind each question, and the one function that runs them all.

Rank-based where no adjustment is asked for: Spearman for two continuous variables,
Mann-Whitney for two groups. The one comparison that is adjusted for age uses a
linear model, with a rank-based check beside it. Every test is two-sided.
Benjamini-Hochberg is applied across functional connectivity outcomes only, within
one exposure and one bladder state at a time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from dtic.config import Config
from dtic.schemas import IOAB, MS

STATES = ("full", "empty", "change")
# (column suffix, label, role): FA is the primary measure of integrity, MD the secondary
METRICS = (("fa", "FA", "primary"), ("md", "MD", "secondary"))
SYMPTOM_OUTCOMES = [
    ("nbss_total", "primary"),
    ("diary_voids_per_day", "secondary"),
    ("diary_urgency_per_day", "secondary"),
    ("diary_incontinence_per_day", "secondary"),
]


def _rank_rows(a: np.ndarray) -> np.ndarray:
    return sps.rankdata(a, axis=1)


def _row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (a * b).sum(axis=1) / denom, np.nan)


def spearman(x, y, n_boot: int = 2000, seed: int = 0, min_n: int = 6) -> dict:
    """Spearman rho with a percentile bootstrap 95% interval.

    Pairs with a missing value are dropped. Too few pairs, or a variable with no
    variation, gives status 'skipped' and no numbers rather than a made-up result.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = int(len(x))
    out = {"test": "spearman", "estimate_name": "rho", "n": n,
           "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
           "statistic": np.nan, "p": np.nan, "status": "ok"}
    if n < min_n:
        out["status"] = f"skipped: n below {min_n}"
        return out
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        out["status"] = "skipped: no variation"
        return out
    res = sps.spearmanr(x, y)
    out["estimate"] = out["statistic"] = float(res.statistic)
    out["p"] = float(res.pvalue)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = _row_pearson(_rank_rows(x[idx]), _rank_rows(y[idx]))
    boot = boot[np.isfinite(boot)]
    if len(boot) >= n_boot // 2:
        out["ci_low"], out["ci_high"] = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    return out


def mann_whitney(group1, group0, n_boot: int = 2000, seed: int = 0, min_n: int = 6) -> dict:
    """Mann-Whitney U for group1 against group0.

    estimate is the rank-biserial correlation: +1 when every group1 value is above
    every group0 value, -1 the other way round, 0 for no separation. shift is the
    Hodges-Lehmann estimate, the median of all group1 minus group0 differences, in
    the units of the variable, with a bootstrap 95% interval.
    """
    a = np.asarray(group1, dtype=float)
    b = np.asarray(group0, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n0 = int(len(a)), int(len(b))
    out = {"test": "mann_whitney", "estimate_name": "rank_biserial", "n": n1 + n0,
           "n_group1": n1, "n_group0": n0,
           "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
           "shift": np.nan, "shift_ci_low": np.nan, "shift_ci_high": np.nan,
           "statistic": np.nan, "p": np.nan, "status": "ok"}
    if n1 + n0 < min_n or min(n1, n0) < 3:
        out["status"] = f"skipped: needs {min_n} in total and 3 per group"
        return out
    if np.ptp(np.concatenate([a, b])) == 0:
        out["status"] = "skipped: no variation"
        return out
    res = sps.mannwhitneyu(a, b, alternative="two-sided")
    u1 = float(res.statistic)
    out["statistic"], out["p"] = u1, float(res.pvalue)
    out["estimate"] = 2 * u1 / (n1 * n0) - 1
    out["shift"] = float(np.median(a[:, None] - b[None, :]))

    rng = np.random.default_rng(seed)
    ia = rng.integers(0, n1, size=(n_boot, n1))
    ib = rng.integers(0, n0, size=(n_boot, n0))
    rb, hl = np.empty(n_boot), np.empty(n_boot)
    # resamples x n1 x n0 differences, a slab at a time so a big cohort does not eat the memory
    step = max(1, 4_000_000 // (n1 * n0))
    for lo in range(0, n_boot, step):
        hi = min(lo + step, n_boot)
        diffs = a[ia[lo:hi]][:, :, None] - b[ib[lo:hi]][:, None, :]
        wins = (diffs > 0).sum(axis=(1, 2)) + 0.5 * (diffs == 0).sum(axis=(1, 2))
        rb[lo:hi] = 2 * wins / (n1 * n0) - 1
        hl[lo:hi] = np.median(diffs.reshape(hi - lo, -1), axis=1)
    out["ci_low"], out["ci_high"] = (float(v) for v in np.percentile(rb, [2.5, 97.5]))
    out["shift_ci_low"], out["shift_ci_high"] = (float(v) for v in np.percentile(hl, [2.5, 97.5]))
    return out


def adjusted_difference(values, group, covariate, n_boot: int = 2000, seed: int = 0,
                        min_n: int = 6) -> dict:
    """Difference between group 1 and group 0 at the same value of a covariate (age).

    A linear model, values ~ group + covariate, fitted by least squares. estimate is
    the group coefficient: how much higher group 1 is than group 0 at equal age, in
    the units of the variable. Its interval is a percentile bootstrap that resamples
    subjects within each group, and p is the usual t-test of the coefficient.
    std_effect is the estimate over the residual standard deviation.

    p_rank is a check that does not lean on the values being normally distributed:
    Quade's rank analysis of covariance. Values and covariate are ranked, the value
    ranks are regressed on the covariate ranks, and the residuals are compared
    between the groups.
    """
    y = np.asarray(values, dtype=float)
    g = np.asarray(group, dtype=float)
    a = np.asarray(covariate, dtype=float)
    keep = np.isfinite(y) & np.isfinite(a) & np.isin(g, (0, 1))
    y, g, a = y[keep], g[keep], a[keep]
    n1, n0 = int((g == 1).sum()), int((g == 0).sum())
    out = {"test": "linear_model", "estimate_name": "adjusted_difference", "n": n1 + n0,
           "n_group1": n1, "n_group0": n0,
           "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "std_effect": np.nan,
           "statistic": np.nan, "p": np.nan, "p_rank": np.nan, "status": "ok"}
    if n1 + n0 < min_n or min(n1, n0) < 3:
        out["status"] = f"skipped: needs {min_n} in total and 3 per group"
        return out
    if np.ptp(y) == 0:
        out["status"] = "skipped: no variation"
        return out
    if np.ptp(a) == 0:
        out["status"] = "skipped: the covariate does not vary"
        return out

    def design(group_col, cov_col):
        return np.stack([np.ones_like(group_col), group_col, cov_col], axis=-1)

    x = design(g, a)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    dof = len(y) - 3
    resid = y - x @ beta
    s2 = float(resid @ resid) / dof
    se = np.sqrt(s2 * np.linalg.inv(x.T @ x)[1, 1])
    out["estimate"] = float(beta[1])
    if s2 > 0:
        out["statistic"] = float(beta[1] / se)
        out["p"] = float(2 * sps.t.sf(abs(out["statistic"]), dof))
        out["std_effect"] = float(beta[1] / np.sqrt(s2))

    ry, ra = sps.rankdata(y), sps.rankdata(a)
    slope = np.polyfit(ra, ry, 1)
    rank_resid = ry - np.polyval(slope, ra)
    if np.ptp(rank_resid) > 0:
        out["p_rank"] = float(sps.f_oneway(rank_resid[g == 1], rank_resid[g == 0]).pvalue)

    rng = np.random.default_rng(seed)
    i1, i0 = np.flatnonzero(g == 1), np.flatnonzero(g == 0)
    idx = np.concatenate([i1[rng.integers(0, n1, size=(n_boot, n1))],
                          i0[rng.integers(0, n0, size=(n_boot, n0))]], axis=1)
    xb = design(g[idx], a[idx])
    # a resample in which everyone has the same age cannot separate age from the intercept
    usable = np.ptp(a[idx], axis=1) > 0
    boot = (np.linalg.pinv(xb[usable]) @ y[idx][usable][..., None])[:, 1, 0]
    if len(boot) >= n_boot // 2:
        out["ci_low"], out["ci_high"] = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    return out


def fdr_bh(p) -> np.ndarray:
    """Benjamini-Hochberg q-values. Missing p-values stay missing and are not counted."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def describe(values, min_cell_n: int = 1) -> dict:
    """n, mean, SD, median and quartiles. Below min_cell_n only n is given."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    out = {"n": int(len(v)), "mean": np.nan, "sd": np.nan,
           "median": np.nan, "q1": np.nan, "q3": np.nan, "suppressed": False}
    if len(v) == 0:
        return out
    if len(v) < min_cell_n:
        out["suppressed"] = True
        return out
    out["mean"] = float(v.mean())
    out["sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
    out["median"], out["q1"], out["q3"] = (float(x) for x in np.percentile(v, [50, 25, 75]))
    return out


# ---- which columns and which subjects play which part ----------------------------------

def integrity_measures(cfg: Config) -> list[tuple[str, str, str]]:
    """(role, label, column) for FA and MD in each primary tract. FA is primary, MD secondary."""
    return [(role, f"{tract} {label}", f"tract_{tract}_{key}")
            for key, label, role in METRICS for tract in cfg.tracts.primary]


def in_cohort(table: pd.DataFrame, cohort: str) -> pd.Series:
    """True for the rows of one cohort. A subject with no clinical row is in neither."""
    if "cohort" not in table.columns:
        return pd.Series(False, index=table.index)
    return table["cohort"].astype("string").str.lower().eq(cohort).fillna(False).astype(bool)


def women(table: pd.DataFrame) -> pd.Series:
    if "sex" not in table.columns:
        return pd.Series(False, index=table.index)
    return table["sex"].astype("string").str.lower().eq("female").fillna(False).astype(bool)


def ms_vs_ioab(table: pd.DataFrame) -> pd.Series:
    """1 for women in MSBladder, 0 for women in iOAB, missing for everyone else."""
    out = pd.Series(np.nan, index=table.index)
    out[women(table) & in_cohort(table, MS)] = 1.0
    out[women(table) & in_cohort(table, IOAB)] = 0.0
    return out


def fc_outcomes(table: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(column, state, connection) for every FC column in the subject table."""
    out = []
    for col in table.columns:
        if col.startswith("fc__"):
            _, state, connection = col.split("__", 2)
            out.append((col, state, connection))
    return out


def _row(question, role, exposure, outcome, result, **extra) -> dict:
    row = {"question": question, "role": role, "exposure": exposure, "outcome": outcome,
           "state": "", "connection": "", "family": "", "q": np.nan, "p_adjustment": "none"}
    row.update(result)
    row.update(extra)
    return row


def fc_values(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The FC columns as they enter the tests.

    The FC question is about subjects scanned with a full and an empty bladder. With
    fc_require_both_states on, a subject's full or empty value for a connection only
    counts when the other state is there too. The change column needs no such rule: it
    is either computed from both states or supplied as a difference.
    """
    cols = [c for c, _, _ in fc_outcomes(table)]
    out = table[cols].apply(pd.to_numeric, errors="coerce")
    if not cfg.stats.fc_require_both_states:
        return out
    for col, state, connection in fc_outcomes(table):
        other = {"full": "empty", "empty": "full"}.get(state)
        if other is None:
            continue
        partner = f"fc__{other}__{connection}"
        if partner in out.columns:
            out[col] = out[col].where(table[partner].notna())
        else:
            out[col] = np.nan
    return out


def _fc_block(table, cfg, question) -> list[dict]:
    rows = []
    st = cfg.stats
    fc = fc_values(table, cfg)
    for role, label, col in integrity_measures(cfg):
        if col not in table.columns:
            continue
        for state in STATES:
            family = f"{question} | {label} | {state}"
            block = []
            for fc_col, fc_state, connection in fc_outcomes(table):
                if fc_state != state:
                    continue
                res = spearman(table[col], fc[fc_col], st.bootstrap_resamples, st.seed, st.min_n)
                block.append(_row(question, role, label, f"FC {state}: {connection}", res,
                                  state=state, connection=connection, family=family,
                                  p_adjustment="fdr_bh across connections"))
            for row, q in zip(block, fdr_bh([r["p"] for r in block])):
                row["q"] = float(q)
            rows.extend(block)
    return rows


def _group_block(table, cfg, question, group: pd.Series, labels: tuple[str, str]) -> list[dict]:
    rows = []
    st = cfg.stats
    for role, label, col in integrity_measures(cfg):
        if col not in table.columns:
            continue
        values = pd.to_numeric(table[col], errors="coerce")
        res = mann_whitney(values[group == 1], values[group == 0],
                           st.bootstrap_resamples, st.seed, st.min_n)
        rows.append(_row(question, role, f"{labels[0]} vs {labels[1]}", label, res))
    return rows


def responder(table: pd.DataFrame, cfg: Config) -> pd.Series:
    """1 responder, 0 non-responder, missing for anyone outside the active arm.

    Response is an improvement (a fall) in NBSS total of at least the configured
    number of points between baseline and the post-treatment score.
    """
    threshold = cfg.treatment_response.min_nbss_improvement
    improvement = pd.to_numeric(table["nbss_total"], errors="coerce") - pd.to_numeric(
        table["nbss_total_post"], errors="coerce")
    out = pd.Series(np.nan, index=table.index)
    active = table["arm"].astype("string").str.lower().eq("active").fillna(False) & improvement.notna()
    out[active] = (improvement[active] >= threshold).astype(float)
    return out


COLUMNS = ["question", "role", "exposure", "outcome", "state", "connection", "test",
           "n", "n_group1", "n_group0", "estimate_name", "estimate", "ci_low", "ci_high",
           "shift", "shift_ci_low", "shift_ci_high", "std_effect", "statistic",
           "p", "p_rank", "q", "p_adjustment", "family", "status"]


def run_analyses(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Every enabled test, one row each. Tests that cannot run are kept, marked skipped.

    Every question but q2 is asked of the MSBladder cohort alone.
    """
    rows: list[dict] = []
    on, st = cfg.analyses, cfg.stats
    ms = table[in_cohort(table, MS)]

    if on.q1_integrity_fc:
        rows += _fc_block(ms, cfg, "q1_integrity_fc")

    if on.q2_ms_vs_ioab and {"cohort", "sex", "age"} <= set(table.columns):
        group = ms_vs_ioab(table)
        age = pd.to_numeric(table["age"], errors="coerce")
        for role, label, col in integrity_measures(cfg):
            if col not in table.columns:
                continue
            res = adjusted_difference(pd.to_numeric(table[col], errors="coerce"), group, age,
                                      st.bootstrap_resamples, st.seed, st.min_n)
            rows.append(_row("q2_ms_vs_ioab", role, "MSBladder vs iOAB, women, at equal age",
                             label, res))

    if on.q3_detrusor_overactivity and "do_present" in ms.columns:
        group = pd.to_numeric(ms["do_present"], errors="coerce")
        rows += _group_block(ms, cfg, "q3_detrusor_overactivity", group,
                             ("DO present", "DO absent"))

    if on.q4_symptom_severity:
        for measure_role, label, col in integrity_measures(cfg):
            if col not in ms.columns:
                continue
            for outcome, outcome_role in SYMPTOM_OUTCOMES:
                if outcome not in ms.columns:
                    continue
                res = spearman(ms[col], pd.to_numeric(ms[outcome], errors="coerce"),
                               st.bootstrap_resamples, st.seed, st.min_n)
                # primary only when both sides are: FA against the NBSS total
                role = "primary" if measure_role == outcome_role == "primary" else "secondary"
                rows.append(_row("q4_symptom_severity", role, label, outcome, res))

    if on.q5_treatment_response and not [
            c for c in ("arm", "nbss_total", "nbss_total_post") if c not in ms.columns]:
        rows += _group_block(ms, cfg, "q5_treatment_response", responder(ms, cfg),
                             ("responder", "non-responder"))

    frame = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in frame.columns:
            frame[c] = np.nan
    return frame[COLUMNS]
