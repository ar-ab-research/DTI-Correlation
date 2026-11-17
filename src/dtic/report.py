"""The shareable group summary: counts, descriptives and test results, no subject rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dtic import stats
from dtic.config import Config
from dtic.fakedata import FAKE_COLUMN
from dtic.schemas import IOAB, MS

TITLES = {
    "q1_integrity_fc": "Q1. White matter integrity and functional connectivity",
    "q2_ms_vs_ioab": "Q2. MS against idiopathic OAB",
    "q3_detrusor_overactivity": "Q3. White matter integrity by detrusor overactivity",
    "q4_symptom_severity": "Q4. White matter integrity and symptom severity",
    "q5_treatment_response": "Q5. White matter integrity by treatment response",
}
COHORT_NAMES = {MS: "MSBladder", IOAB: "iOAB"}
QC_COLUMNS = ["motion_mean_mm", "motion_max_mm", "reg_similarity"]
# what group 1 and group 0 are in each question that compares two groups
GROUPS = {
    "q2_ms_vs_ioab": ("women, MSBladder", "women, iOAB"),
    "q3_detrusor_overactivity": ("DO present", "DO absent"),
    "q5_treatment_response": ("responder", "non-responder"),
}


def _tract_columns(table: pd.DataFrame, cfg: Config, primary_only: bool = False) -> list[str]:
    names = list(cfg.tracts.primary) + ([] if primary_only else list(cfg.tracts.secondary))
    cols = [f"tract_{name}_{key}" for key, _, _ in stats.METRICS for name in names]
    return [c for c in cols if c in table.columns]


def _clinical_columns(table: pd.DataFrame) -> list[str]:
    cols = ["age"] + [name for name, _ in stats.SYMPTOM_OUTCOMES] + ["nbss_total_post"]
    return [c for c in cols if c in table.columns]


def group_masks(table: pd.DataFrame, cfg: Config) -> dict[str, pd.Series]:
    """The groups the summary describes, each as a true/false column over the subject table."""
    ms = stats.in_cohort(table, MS)
    masks = {"all": pd.Series(True, index=table.index),
             COHORT_NAMES[MS]: ms, COHORT_NAMES[IOAB]: stats.in_cohort(table, IOAB)}
    if cfg.analyses.q2_ms_vs_ioab and {"sex", "age"} <= set(table.columns):
        compared = stats.ms_vs_ioab(table)
        masks["women, MSBladder"], masks["women, iOAB"] = compared == 1, compared == 0
    if cfg.analyses.q3_detrusor_overactivity and "do_present" in table.columns:
        do = pd.to_numeric(table["do_present"], errors="coerce")
        masks["DO present"], masks["DO absent"] = ms & (do == 1), ms & (do == 0)
    if cfg.analyses.q5_treatment_response and {"arm", "nbss_total", "nbss_total_post"} <= set(table.columns):
        resp = stats.responder(table, cfg)
        masks["responder"], masks["non-responder"] = ms & (resp == 1), ms & (resp == 0)
    return masks


def descriptives(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """n, mean, SD, median and quartiles: overall, by cohort, and by group where a question compares groups."""
    min_n = cfg.report.min_cell_n
    imaging = _tract_columns(table, cfg)
    clinical = _clinical_columns(table)
    fc_cols = [c for c, _, _ in stats.fc_outcomes(table)]
    qc = [c for c in QC_COLUMNS if c in table.columns]
    rows = []
    for group, mask in group_masks(table, cfg).items():
        cols = imaging + qc if group == "all" else imaging + clinical
        if group == COHORT_NAMES[MS]:
            cols = cols + fc_cols
        frame = table[mask]
        for col in cols:
            values = pd.to_numeric(frame[col], errors="coerce")
            rows.append({"group": group, "variable": col, **stats.describe(values, min_n)})
    return pd.DataFrame(rows)


def acquisition_table(table: pd.DataFrame) -> pd.DataFrame:
    """How many scans in each cohort were acquired in each way. Scanner settings, not subject data."""
    keys = ["n_directions", "b_value", "n_b0", "voxel_mm"]
    frame = table[keys].copy()
    cohort = table["cohort"].astype("string").str.lower() if "cohort" in table.columns else pd.Series(
        pd.NA, index=table.index, dtype="string")
    frame.insert(0, "cohort", cohort.map(COHORT_NAMES).fillna("no clinical row"))
    out = frame.groupby(["cohort", *keys], dropna=False).size().reset_index(name="scans")
    return out.sort_values(["cohort", "scans"], ascending=[True, False]).reset_index(drop=True)


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    # three significant figures, written out in full: 1183, not 1.18e+03
    if abs(value) >= 10 ** digits:
        return f"{value:.0f}"
    return f"{value:.{digits}g}"


def _p(value) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


DESCRIBE_HEADER = ["Variable", "n", "Mean ± SD", "Median (IQR)"]


def _describe_rows(desc: pd.DataFrame, group: str, variables: list[str]) -> list[list[str]]:
    rows = []
    for var in variables:
        hit = desc[(desc["group"] == group) & (desc["variable"] == var)]
        if hit.empty:
            continue
        d = hit.iloc[0]
        if d["n"] == 0:
            continue
        if d["suppressed"]:
            rows.append([var, _fmt(int(d["n"])), "withheld", ""])
        else:
            rows.append([var, _fmt(int(d["n"])), f"{_fmt(d['mean'])} ± {_fmt(d['sd'])}",
                         f"{_fmt(d['median'])} ({_fmt(d['q1'])} to {_fmt(d['q3'])})"])
    return rows


def _estimate(r) -> str:
    if r.status != "ok":
        return r.status
    return f"{_fmt(r.estimate)} ({_fmt(r.ci_low)} to {_fmt(r.ci_high)})"


def _fc_section(tests: pd.DataFrame, alpha: float) -> list[str]:
    lines = _table(
        ["Measure", "State", "Connections tested", f"q < {alpha:g}", "Smallest q"],
        [[exposure, state, _fmt(int(fam["status"].eq("ok").sum())),
          _fmt(int((fam["q"] < alpha).sum())), _p(fam["q"].min())]
         for (exposure, state), fam in tests.groupby(["exposure", "state"], sort=False)],
    )
    hits = tests[tests["q"] < alpha].sort_values("q")
    shown, caption = hits, f"Connections with q < {alpha:g}:"
    if hits.empty:
        shown = tests[tests["status"] == "ok"].nsmallest(5, "p")
        caption = (f"No connection reached q < {alpha:g}. For orientation only, the five smallest "
                   "p-values (none is a finding):")
    if not shown.empty:
        lines += ["", caption, ""]
        lines += _table(
            ["Measure", "State", "Connection", "n", "rho (95% CI)", "p", "q"],
            [[r.exposure, r.state, r.connection, _fmt(int(r.n)), _estimate(r), _p(r.p), _p(r.q)]
             for r in shown.itertuples()],
        )
    return lines


def _corr_section(tests: pd.DataFrame) -> list[str]:
    return _table(
        ["Measure", "Outcome", "Role", "n", "rho (95% CI)", "p (unadjusted)"],
        [[r.exposure, r.outcome, r.role, _fmt(int(r.n)), _estimate(r), _p(r.p)]
         for r in tests.itertuples()],
    )


def _group_section(tests: pd.DataFrame) -> list[str]:
    return _table(
        ["Comparison", "Measure", "Role", "n (group 1 / group 0)", "Rank-biserial r (95% CI)",
         "Median shift (95% CI)", "p (unadjusted)"],
        [[r.exposure, r.outcome, r.role, f"{_fmt(int(r.n_group1))} / {_fmt(int(r.n_group0))}",
          _estimate(r),
          f"{_fmt(r.shift)} ({_fmt(r.shift_ci_low)} to {_fmt(r.shift_ci_high)})" if r.status == "ok" else "",
          _p(r.p)] for r in tests.itertuples()],
    )


def _adjusted_section(tests: pd.DataFrame) -> list[str]:
    return _table(
        ["Measure", "Role", "n (MSBladder / iOAB)", "Difference at equal age (95% CI)",
         "Standardized", "p", "p (rank check)"],
        [[r.outcome, r.role, f"{_fmt(int(r.n_group1))} / {_fmt(int(r.n_group0))}", _estimate(r),
          _fmt(r.std_effect), _p(r.p), _p(r.p_rank)] for r in tests.itertuples()],
    )


def uses_fake_data(table: pd.DataFrame) -> bool:
    """True when any row came from `dtic make-fake-data`."""
    if FAKE_COLUMN not in table.columns:
        return False
    return bool((pd.to_numeric(table[FAKE_COLUMN], errors="coerce") == 1).any())


FAKE_BANNER = ("**FAKE DATA. The clinical and FC tables behind this summary were made up by "
               "`dtic make-fake-data` to test the pipeline. Nothing below is a finding.**")


def markdown(table: pd.DataFrame, tests: pd.DataFrame, desc: pd.DataFrame, acquisition: pd.DataFrame,
             cfg: Config, manifest: dict) -> str:
    counts, alpha, acq = manifest["counts"], cfg.stats.alpha, cfg.acquisition
    kept = (f"voxels inside the brain with FA of at least {cfg.tracts.min_fa:g}"
            if cfg.tracts.min_fa > 0 else "every voxel inside the brain")
    lines = [
        "# Group summary",
        "",
        *([FAKE_BANNER, ""] if uses_fake_data(table) else []),
        "Group-level results only. This file holds no subject IDs and no per-subject values,",
        "so it can be shared. How to read each number: `docs/interpreting-results.md`.",
        "",
        f"- Scans processed: {counts['processed']} of {counts['scans']}"
        + (f" ({counts['failed']} failed)" if counts["failed"] else ""),
        f"- MSBladder: {counts[MS]}. iOAB: {counts[IOAB]}. "
        f"With clinical data: {counts['with_clinical']}. With FC data: {counts['with_fc']}.",
        f"- Registration QC failures (no tract values): {counts['registration_qc_failed']}",
        f"- Scans that are not {acq.directions} directions at b={acq.b_value:g} with "
        f"{acq.b0_volumes} b0 volumes: {counts['acquisition_not_as_expected']}",
        "- FA has no unit and runs from 0 to 1. MD is in 10⁻³ mm²/s. A tract's value is the mean over",
        f"  {kept}, left and right sides together.",
        "- Lower FA and higher MD both mean lower white matter integrity.",
        "- All tests are two-sided. FC results carry Benjamini-Hochberg q-values, computed across",
        "  connections within one measure and one bladder state. Other p-values are unadjusted.",
        "- Every question but Q2 uses the MSBladder cohort only.",
        f"- Group statistics are withheld for groups smaller than {cfg.report.min_cell_n}.",
        f"- dtic {manifest['versions']['dtic']}, DIPY {manifest['versions']['dipy']}, "
        f"config {manifest['config_digest']}, finished {manifest['finished']}.",
        "",
        "## Cohort",
        "",
        "All scans:",
        "",
    ]
    imaging = _tract_columns(table, cfg)
    primary = _tract_columns(table, cfg, primary_only=True)
    lines += _table(DESCRIBE_HEADER, _describe_rows(desc, "all", imaging + QC_COLUMNS))
    for cohort in COHORT_NAMES.values():
        rows = _describe_rows(desc, cohort, primary + _clinical_columns(table))
        if rows:
            lines += ["", f"{cohort}:", ""] + _table(DESCRIBE_HEADER, rows)
    lines += ["", "How the scans were acquired:", ""]
    lines += _table(
        ["Cohort", "Directions", "b-value", "b0 volumes", "Voxel, mm", "Scans"],
        [[r.cohort, _fmt(int(r.n_directions)), _fmt(float(r.b_value)), _fmt(int(r.n_b0)),
          str(r.voxel_mm), _fmt(int(r.scans))] for r in acquisition.itertuples()],
    )
    lines.append("")

    for question, title in TITLES.items():
        block = tests[tests["question"] == question]
        if block.empty:
            continue
        lines += [f"## {title}", ""]
        skipped = int(block["status"].ne("ok").sum())
        if question == "q1_integrity_fc":
            lines += _fc_section(block, alpha)
        elif question == "q4_symptom_severity":
            lines += _corr_section(block)
        else:
            groups = GROUPS[question]
            if question == "q2_ms_vs_ioab":
                lines += _adjusted_section(block)
                shown = ["age"] + primary
                note = ("Women only. The difference is MSBladder minus iOAB at the same age, from a "
                        "linear model. A negative difference in FA, or a positive one in MD, means "
                        "lower integrity in MSBladder. The rank check is Quade's rank analysis of "
                        "covariance. The group values below are not adjusted for age.")
            else:
                lines += _group_section(block)
                shown = primary
                note = (f"Group 1 is {groups[0]}, group 0 is {groups[1]}. A positive rank-biserial r "
                        "or shift means group 1 has the higher values.")
            for g in groups:
                rows = _describe_rows(desc, g, [c for c in shown if c in table.columns])
                if rows:
                    lines += ["", f"{g}:", ""] + _table(DESCRIBE_HEADER, rows)
            lines += ["", note]
        if skipped:
            lines += ["", f"{skipped} test(s) could not be run (too few subjects or no variation) "
                          "and are listed as skipped in the csv."]
        lines.append("")
    lines += ["Every test, including all FC connections, is in `group_summary.csv`. "
              "Descriptives for every variable are in `descriptives.csv`.", ""]
    return "\n".join(lines)


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def write(table: pd.DataFrame, tests: pd.DataFrame, cfg: Config, manifest: dict) -> list[Path]:
    """Write the shareable files. Returns their paths."""
    out = cfg.paths.shareable_dir
    out.mkdir(parents=True, exist_ok=True)
    desc = descriptives(table, cfg)
    acquisition = acquisition_table(table)
    paths = [out / "group_summary.md", out / "group_summary.csv", out / "descriptives.csv",
             out / "group_summary.json"]
    paths[0].write_text(markdown(table, tests, desc, acquisition, cfg, manifest), encoding="utf-8")
    tests.to_csv(paths[1], index=False)
    desc.to_csv(paths[2], index=False)
    paths[3].write_text(json.dumps(
        {"fake_data": uses_fake_data(table), "run": manifest, "acquisition": _records(acquisition),
         "descriptives": _records(desc), "tests": _records(tests)}, indent=2),
        encoding="utf-8")
    return paths
