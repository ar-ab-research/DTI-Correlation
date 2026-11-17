import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from dtic import fakedata
from dtic.config import from_dict
from dtic.schemas import fc_column, fc_wide
from dtic.stats import (adjusted_difference, describe, fdr_bh, mann_whitney, responder,
                        run_analyses, spearman)

RNG = np.random.default_rng(42)


def test_spearman_matches_scipy_and_drops_missing_pairs():
    x = RNG.normal(size=30)
    y = 0.5 * x + RNG.normal(size=30)
    ref = sps.spearmanr(x, y)
    got = spearman(x, y)
    assert got["estimate"] == pytest.approx(ref.statistic)
    assert got["p"] == pytest.approx(ref.pvalue)
    assert got["n"] == 30 and got["status"] == "ok"
    x2, y2 = np.append(x, [np.nan, 1.0]), np.append(y, [1.0, np.nan])
    again = spearman(x2, y2)
    assert again["n"] == 30 and again["estimate"] == pytest.approx(ref.statistic)


def test_spearman_interval_brackets_rho_and_is_seeded():
    x = RNG.normal(size=40)
    y = x + RNG.normal(size=40)
    a, b, c = spearman(x, y, seed=1), spearman(x, y, seed=1), spearman(x, y, seed=2)
    assert a["ci_low"] < a["estimate"] < a["ci_high"]
    assert a["ci_low"] > 0                      # a strong effect at n=40 excludes zero
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])
    assert (a["ci_low"], a["ci_high"]) != (c["ci_low"], c["ci_high"])


def test_spearman_handles_ties_like_scipy():
    x = np.array([0, 0, 0, 0, 1, 2, 3, 5, 8, 8], float)       # zero-inflated, like tract loads
    y = np.array([1, 2, 2, 3, 3, 4, 6, 5, 9, 7], float)
    assert spearman(x, y)["estimate"] == pytest.approx(sps.spearmanr(x, y).statistic)


def test_spearman_skips_rather_than_inventing_a_result():
    few = spearman([1, 2, 3], [1, 2, 3])
    assert few["status"].startswith("skipped") and np.isnan(few["estimate"]) and np.isnan(few["p"])
    flat = spearman(np.zeros(10), np.arange(10))
    assert flat["status"] == "skipped: no variation" and np.isnan(flat["p"])


def test_mann_whitney_matches_scipy():
    a, b = RNG.normal(1, 1, 12), RNG.normal(0, 1, 9)
    ref = sps.mannwhitneyu(a, b, alternative="two-sided")
    got = mann_whitney(a, b)
    assert got["statistic"] == pytest.approx(ref.statistic)
    assert got["p"] == pytest.approx(ref.pvalue)
    assert (got["n_group1"], got["n_group0"], got["n"]) == (12, 9, 21)
    assert got["estimate"] == pytest.approx(2 * ref.statistic / (12 * 9) - 1)
    assert got["ci_low"] <= got["estimate"] <= got["ci_high"]
    assert got["shift_ci_low"] <= got["shift"] <= got["shift_ci_high"]


def test_rank_biserial_direction_and_extremes():
    low, high = np.arange(6.0), np.arange(6.0) + 100
    assert mann_whitney(high, low)["estimate"] == 1.0
    assert mann_whitney(low, high)["estimate"] == -1.0
    assert mann_whitney(high, low)["shift"] == pytest.approx(100.0)
    same = mann_whitney([1, 2, 3, 4], [1, 2, 3, 4])
    assert same["estimate"] == pytest.approx(0.0) and same["shift"] == 0.0


def test_mann_whitney_skips_small_or_flat_groups():
    assert mann_whitney([1, 2], [3, 4, 5, 6, 7])["status"].startswith("skipped")
    assert mann_whitney([0, 0, 0], [0, 0, 0])["status"] == "skipped: no variation"
    drops_nan = mann_whitney([1, 2, 3, np.nan], [4, 5, 6])
    assert drops_nan["n_group1"] == 3


def test_fdr_matches_statsmodels_and_a_hand_calculation():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.6])
    assert fdr_bh(p) == pytest.approx(multipletests(p, method="fdr_bh")[1])
    # smallest p times m over rank: 0.001 * 5 / 1
    assert fdr_bh(p)[0] == pytest.approx(0.005)
    assert fdr_bh([0.03])[0] == pytest.approx(0.03)          # a family of one is unchanged


def test_fdr_leaves_missing_p_missing_and_out_of_the_count():
    q = fdr_bh([0.01, np.nan, 0.04])
    assert np.isnan(q[1])
    assert q[[0, 2]] == pytest.approx(multipletests([0.01, 0.04], method="fdr_bh")[1])
    assert np.isnan(fdr_bh([np.nan, np.nan])).all()


def test_describe_and_suppression():
    d = describe([1, 2, 3, 4, np.nan])
    assert d["n"] == 4 and d["mean"] == 2.5 and d["sd"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert (d["q1"], d["median"], d["q3"]) == (1.75, 2.5, 3.25)
    assert "min" not in d and "max" not in d                 # extremes are individual values
    small = describe([1, 2], min_cell_n=3)
    assert small["n"] == 2 and small["suppressed"] and np.isnan(small["mean"])


# ---- the age-adjusted comparison -------------------------------------------------------

def two_groups(n=60, effect=-0.03, age_slope=-0.002, seed=3):
    """FA that falls with age, in two groups of different ages, one shifted by `effect`."""
    rng = np.random.default_rng(seed)
    group = (np.arange(n) < n // 2).astype(float)
    age = np.where(group == 1, rng.uniform(25, 60, n), rng.uniform(45, 80, n))   # group 1 is younger
    fa = 0.5 + age_slope * age + effect * group + rng.normal(0, 0.01, n)
    return fa, group, age


def test_adjusted_difference_matches_statsmodels():
    import statsmodels.api as sm

    fa, group, age = two_groups()
    ref = sm.OLS(fa, sm.add_constant(np.column_stack([group, age]))).fit()
    got = adjusted_difference(fa, group, age)
    assert got["estimate"] == pytest.approx(ref.params[1])
    assert got["statistic"] == pytest.approx(ref.tvalues[1])
    assert got["p"] == pytest.approx(ref.pvalues[1])
    assert got["std_effect"] == pytest.approx(ref.params[1] / np.sqrt(ref.scale))
    assert (got["n"], got["n_group1"], got["n_group0"]) == (60, 30, 30)
    assert got["ci_low"] < got["estimate"] < got["ci_high"]


def test_adjustment_recovers_an_effect_that_age_hides():
    # the younger group has the lower FA at equal age, but its youth cancels that in the raw values
    fa, group, age = two_groups(effect=-0.03, age_slope=-0.002)
    raw = fa[group == 1].mean() - fa[group == 0].mean()
    got = adjusted_difference(fa, group, age)
    assert abs(raw) < 0.015
    assert got["estimate"] == pytest.approx(-0.03, abs=0.008)
    assert got["ci_high"] < 0 and got["p"] < 0.001 and got["p_rank"] < 0.001


def test_adjustment_removes_a_difference_that_is_only_age():
    fa, group, age = two_groups(effect=0.0, age_slope=-0.003)
    assert fa[group == 1].mean() - fa[group == 0].mean() > 0.03       # the raw groups differ
    got = adjusted_difference(fa, group, age)
    assert got["ci_low"] < 0 < got["ci_high"] and got["p"] > 0.05


def test_adjusted_difference_is_seeded_and_ignores_rows_outside_the_two_groups():
    fa, group, age = two_groups()
    a, b = adjusted_difference(fa, group, age, seed=1), adjusted_difference(fa, group, age, seed=1)
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])
    more = adjusted_difference(np.append(fa, [0.9, np.nan]), np.append(group, [np.nan, 1]),
                               np.append(age, [50, 50]))
    assert more["n"] == 60 and more["estimate"] == pytest.approx(a["estimate"])


def test_adjusted_difference_skips_rather_than_inventing_a_result():
    fa, group, age = two_groups(n=5)
    assert adjusted_difference(fa, group, age)["status"].startswith("skipped: needs")
    fa, group, age = two_groups(n=20)
    lopsided = np.zeros(20)
    lopsided[:2] = 1
    assert adjusted_difference(fa, lopsided, age)["status"].startswith("skipped: needs")
    assert adjusted_difference(np.ones(20), group, age)["status"] == "skipped: no variation"
    same_age = adjusted_difference(fa, group, np.full(20, 50.0))
    assert same_age["status"] == "skipped: the covariate does not vary" and np.isnan(same_age["p"])


# ---- the questions ---------------------------------------------------------------------

def subject_table(n=60, seed=0):
    """A per-subject table shaped like the pipeline's, with the fake tables joined on."""
    ids = fakedata.default_ids(n)
    rng = np.random.default_rng(seed)
    fa = rng.normal(0.42, 0.03, n)
    clinical, fc = fakedata.make_fake(ids, seed=seed, integrity=dict(zip(ids, fa)))
    imaging = pd.DataFrame({
        "subject_id": ids,
        "tract_ATR_fa": fa, "tract_SLF_fa": fa + rng.normal(0, 0.004, n),
        "tract_ATR_md": 0.75 - (fa - 0.42) + rng.normal(0, 0.004, n),
        "tract_SLF_md": rng.normal(0.74, 0.03, n),
        "tract_ATR_L_fa": fa, "tract_SLF_temporal_fa": fa,
    })
    return imaging.merge(clinical, on="subject_id").merge(fc_wide(fc), on="subject_id", how="left")


def pick(res, **where):
    for key, value in where.items():
        res = res[res[key] == value]
    return res


def test_planted_effects_are_found_and_survive_fdr():
    res = run_analyses(subject_table(), from_dict({}))
    q1 = pick(res, question="q1_integrity_fc", exposure="ATR FA", state="full")
    planted = pick(q1, connection=fakedata.PLANTED_CONNECTION).iloc[0]
    assert planted["estimate"] > 0.5 and planted["q"] < 0.05 and planted["role"] == "primary"
    assert (q1[q1.connection != fakedata.PLANTED_CONNECTION]["q"] > 0.05).all()
    nbss = pick(res, question="q4_symptom_severity", outcome="nbss_total", exposure="ATR FA").iloc[0]
    assert nbss["estimate"] < -0.5 and nbss["p"] < 0.001 and nbss["role"] == "primary"
    # the MSBladder cohort was given the lower integrity: lower FA, and in the ATR higher MD
    q2 = pick(res, question="q2_ms_vs_ioab")
    assert pick(q2, outcome="ATR FA").iloc[0]["estimate"] < 0
    assert pick(q2, outcome="ATR FA").iloc[0]["p"] < 0.05
    assert pick(q2, outcome="ATR MD").iloc[0]["estimate"] > 0


def test_fa_is_primary_and_md_secondary_in_every_question():
    res = run_analyses(subject_table(), from_dict({}))
    for question, column in (("q1_integrity_fc", "exposure"), ("q2_ms_vs_ioab", "outcome"),
                             ("q3_detrusor_overactivity", "outcome")):
        block = pick(res, question=question)
        assert set(block[column]) == {"ATR FA", "SLF FA", "ATR MD", "SLF MD"}, question
        assert (block[block[column].str.endswith("FA")]["role"] == "primary").all()
        assert (block[block[column].str.endswith("MD")]["role"] == "secondary").all()
    q4 = pick(res, question="q4_symptom_severity")
    assert len(q4) == 16                                   # four measures by four outcomes
    primary = q4[q4.role == "primary"]
    assert set(primary.exposure) == {"ATR FA", "SLF FA"} and set(primary.outcome) == {"nbss_total"}
    # only the tested tracts: no side, no temporal part
    assert not res.apply(lambda r: "L" in f"{r.exposure} {r.outcome}".split(), axis=1).any()
    assert not res["exposure"].str.contains("temporal").any()


def test_fdr_runs_within_measure_and_state_across_connections_only():
    res = run_analyses(subject_table(), from_dict({}))
    fc_rows = pick(res, question="q1_integrity_fc")
    for (_, _), family in fc_rows.groupby(["exposure", "state"]):
        assert family["family"].nunique() == 1
        expected = multipletests(family["p"], method="fdr_bh")[1]
        assert np.allclose(family["q"], expected)
    assert fc_rows["family"].nunique() == 4 * 3              # four measures, three states
    assert res[res.question != "q1_integrity_fc"]["q"].isna().all()


def test_question_layout_and_q5_off_by_default():
    res = run_analyses(subject_table(), from_dict({}))
    assert set(res.question) == {"q1_integrity_fc", "q2_ms_vs_ioab", "q3_detrusor_overactivity",
                                 "q4_symptom_severity"}
    assert set(pick(res, question="q1_integrity_fc").test) == {"spearman"}
    assert set(pick(res, question="q2_ms_vs_ioab").test) == {"linear_model"}
    assert set(pick(res, question="q3_detrusor_overactivity").test) == {"mann_whitney"}
    assert set(pick(res, question="q4_symptom_severity").test) == {"spearman"}
    assert set(pick(res, question="q1_integrity_fc").state) == {"full", "empty", "change"}


def test_every_question_but_the_comparison_uses_msbladder_only():
    table = subject_table()
    n_ms = int((table["cohort"] == "msbladder").sum())
    res = run_analyses(table, from_dict({}))
    for question in ("q1_integrity_fc", "q3_detrusor_overactivity", "q4_symptom_severity"):
        assert (pick(res, question=question)["n"] == n_ms).all(), question
    # an iOAB subject who somehow has symptom scores must still be left out
    extra = table.copy()
    ioab = extra["cohort"] == "ioab"
    extra.loc[ioab, "nbss_total"] = 40
    extra.loc[ioab, "do_present"] = 1
    again = run_analyses(extra, from_dict({}))
    assert (pick(again, question="q4_symptom_severity")["n"] == n_ms).all()
    assert (pick(again, question="q3_detrusor_overactivity")["n"] == n_ms).all()


def test_the_cohort_comparison_is_women_only_and_uses_both_cohorts():
    table = subject_table()
    women = table["sex"] == "female"
    n_ms = int((women & (table["cohort"] == "msbladder")).sum())
    n_ioab = int((women & (table["cohort"] == "ioab")).sum())
    assert (table["sex"] == "male").any()                    # there are men to leave out
    q2 = pick(run_analyses(table, from_dict({})), question="q2_ms_vs_ioab")
    assert (q2["n_group1"] == n_ms).all() and (q2["n_group0"] == n_ioab).all()
    assert (q2["n"] < len(table)).all()
    # the same numbers as fitting the model by hand on the women
    w = table[women]
    by_hand = adjusted_difference(w["tract_ATR_fa"], (w["cohort"] == "msbladder").astype(float), w["age"])
    assert pick(q2, outcome="ATR FA").iloc[0]["estimate"] == pytest.approx(by_hand["estimate"])


def test_the_group_comparison_reads_in_the_right_direction():
    table = subject_table()
    ms = table["cohort"] == "msbladder"
    table.loc[ms, "do_present"] = (table.loc[ms, "tract_ATR_fa"] < table.loc[ms, "tract_ATR_fa"].median()).astype(int)
    q3 = pick(run_analyses(table, from_dict({})), question="q3_detrusor_overactivity")
    row = pick(q3, outcome="ATR FA").iloc[0]
    assert row["exposure"] == "DO present vs DO absent"
    assert row["estimate"] == -1 and row["shift"] < 0        # DO present was given the lower FA


def test_q5_uses_the_active_arm_and_the_configured_threshold():
    cfg = from_dict({"analyses": {"q5_treatment_response": True},
                     "treatment_response": {"min_nbss_improvement": 5}})
    table = subject_table()
    resp = responder(table, cfg)
    active = table["arm"].eq("active").fillna(False).to_numpy(dtype=bool)
    assert resp[~active].isna().all() and resp[active].notna().all()
    improvement = (table["nbss_total"] - table["nbss_total_post"])[active].astype(float)
    assert ((improvement >= 5).astype(float) == resp[active]).all()
    res = run_analyses(table, cfg)
    q5 = pick(res, question="q5_treatment_response")
    assert len(q5) == 4 and (q5["n"] == int(active.sum())).all()
    assert set(q5.exposure) == {"responder vs non-responder"}
    assert "q5_treatment_response" not in set(run_analyses(table, from_dict({})).question)


def test_a_table_without_optional_columns_still_runs():
    table = subject_table().drop(columns=["do_present", "sex", "age", "arm", "nbss_total_post"])
    res = run_analyses(table, from_dict({}))
    assert set(res.question) == {"q1_integrity_fc", "q4_symptom_severity"}


def test_subjects_without_a_clinical_row_enter_no_test():
    table = subject_table()
    table.loc[:9, ["cohort", "sex", "age"]] = np.nan
    n_ms = int((table["cohort"] == "msbladder").sum())
    res = run_analyses(table, from_dict({}))
    assert (pick(res, question="q4_symptom_severity")["n"] <= n_ms).all()
    assert (pick(res, question="q2_ms_vs_ioab")["n"] <= 50).all()


def test_small_cohort_reports_skipped_tests_instead_of_dropping_them():
    res = run_analyses(subject_table(n=6), from_dict({}))
    assert len(res) > 0 and res["status"].str.startswith("skipped").all()
    assert res["p"].isna().all()


def test_fc_tests_use_only_subjects_scanned_in_both_states():
    table = subject_table()
    ms = table.index[table["cohort"] == "msbladder"]
    conn = fakedata.PLANTED_CONNECTION
    table.loc[ms[:5], fc_column("empty", conn)] = np.nan
    table.loc[ms[:5], fc_column("change", conn)] = np.nan
    res = run_analyses(table, from_dict({}))
    row = pick(res, question="q1_integrity_fc", exposure="ATR FA", state="full", connection=conn).iloc[0]
    assert row["n"] == len(ms) - 5
    loose = run_analyses(table, from_dict({"stats": {"fc_require_both_states": False}}))
    row = pick(loose, question="q1_integrity_fc", exposure="ATR FA", state="full", connection=conn).iloc[0]
    assert row["n"] == len(ms)


def test_a_supplied_change_value_counts_without_the_two_states():
    table = subject_table()
    ms = table.index[table["cohort"] == "msbladder"]
    conn = fakedata.PLANTED_CONNECTION
    table.loc[ms[:5], [fc_column("full", conn), fc_column("empty", conn)]] = np.nan
    res = run_analyses(table, from_dict({}))
    change = pick(res, question="q1_integrity_fc", exposure="ATR FA", state="change", connection=conn).iloc[0]
    assert change["n"] == len(ms)
