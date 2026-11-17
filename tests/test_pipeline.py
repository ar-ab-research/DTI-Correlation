import json

import numpy as np
import pandas as pd
import pytest

from dtic import atlas, fakedata, pipeline
from dtic.cli import main
from dtic.config import from_dict, load
from dtic.errors import PipelineError

IDS = [f"case{i:02d}" for i in range(40)]
TRACTS = ("ATR", "SLF", "ATR_L", "ATR_R", "SLF_L", "SLF_R", "SLF_temporal")


def imaging_row(sid, rng, reg_ok=True):
    fa = float(rng.normal(0.42, 0.03))
    row = {"subject_id": sid, "n_volumes": 69, "n_b0": 5, "n_directions": 64, "b_value": 1000.0,
           "n_other_volumes": 0, "acq_as_expected": True, "voxel_mm": "1.97 x 1.97 x 2.5",
           "motion_mean_mm": float(rng.uniform(0.2, 1.5)), "motion_max_mm": 2.0,
           "brain_volume_ml": 1200.0, "reg_similarity": 0.8, "reg_ok": reg_ok}
    for name in TRACTS:
        row[f"tract_{name}_fa"] = fa + float(rng.normal(0, 0.004))
        row[f"tract_{name}_md"] = 0.75 - (fa - 0.42) + float(rng.normal(0, 0.004))
        row[f"tract_{name}_voxels"] = 2000
        row[f"tract_{name}_kept_pct"] = 90.0
    return row


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Scans (empty files), fake tables, and a stand-in for the slow imaging steps."""
    config = tmp_path / "config.yaml"
    config.write_text("stats: {bootstrap_resamples: 200}\n")
    cfg = load(config)
    for sid in IDS:
        folder = cfg.paths.dwi_dir / sid
        folder.mkdir(parents=True)
        for name in ("dwi.nii.gz", "dwi.bval", "dwi.bvec"):
            (folder / name).write_bytes(b"")
    rng = np.random.default_rng(0)
    rows = {sid: imaging_row(sid, rng) for sid in IDS}
    integrity = {sid: rows[sid]["tract_ATR_fa"] for sid in IDS}
    clinical, fc = fakedata.make_fake(IDS, seed=0, integrity=integrity)
    clinical.to_csv(cfg.paths.clinical_csv, index=False)
    fc.to_csv(cfg.paths.fc_csv, index=False)

    def fake_process(sid, scan, cfg, force):
        if rows[sid] is None:
            raise PipelineError(f"{sid}: unreadable scan")
        return dict(rows[sid])

    monkeypatch.setattr(pipeline, "process_subject", fake_process)
    monkeypatch.setattr(atlas, "jhu_paths", lambda cfg, log=print: {})
    return config, cfg, rows


def read_tests(cfg):
    return pd.read_csv(cfg.paths.shareable_dir / "group_summary.csv")


def test_run_writes_local_and_shareable_outputs(project):
    config, cfg, _ = project
    assert main(["run", "-c", str(config)]) == 0
    subjects = pd.read_csv(cfg.paths.local_dir / "subjects.csv")
    assert len(subjects) == 40
    assert {"tract_ATR_fa", "tract_ATR_md", "tract_SLF_fa", "tract_SLF_md", "tract_SLF_temporal_fa",
            "cohort", "sex", "age", "nbss_total", "do_present", "motion_mean_mm"} <= set(subjects)
    assert any(c.startswith("fc__change__") for c in subjects)
    for name in ("group_summary.md", "group_summary.csv", "descriptives.csv", "group_summary.json"):
        assert (cfg.paths.shareable_dir / name).is_file()
    assert (cfg.paths.local_dir / "run_manifest.json").is_file()
    assert (cfg.paths.local_dir / "imaging.csv").is_file()


def test_shareable_files_hold_no_subject_ids_or_subject_values(project):
    config, cfg, rows = project
    main(["run", "-c", str(config)])
    for path in cfg.paths.shareable_dir.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "case" not in text, path.name                     # no ids
        assert "subject_id" not in text, path.name
        # no row of the per-subject table either: spot-check one subject's exact FA
        assert f"{rows['case03']['tract_ATR_fa']:.6f}"[:9] not in text, path.name
    desc = pd.read_csv(cfg.paths.shareable_dir / "descriptives.csv")
    assert not {"min", "max"} & set(desc.columns)                # an extreme is one person's value


def test_summary_reports_the_planted_effects(project):
    config, cfg, _ = project
    main(["run", "-c", str(config)])
    tests = read_tests(cfg)
    hit = tests[(tests.question == "q1_integrity_fc") & (tests.exposure == "ATR FA")
                & (tests.state == "full") & (tests.connection == fakedata.PLANTED_CONNECTION)].iloc[0]
    assert hit["q"] < 0.05 and hit["estimate"] > 0
    q2 = tests[(tests.question == "q2_ms_vs_ioab") & (tests.outcome == "ATR FA")].iloc[0]
    assert q2["estimate"] < 0                                    # MSBladder was given the lower FA
    md = (cfg.paths.shareable_dir / "group_summary.md").read_text(encoding="utf-8")
    for heading in ("## Cohort", "## Q1.", "## Q2.", "## Q3.", "## Q4."):
        assert heading in md
    assert "## Q5." not in md
    assert fakedata.PLANTED_CONNECTION in md
    assert "Difference at equal age" in md and "women, MSBladder:" in md
    summary = json.loads((cfg.paths.shareable_dir / "group_summary.json").read_text())
    assert summary["run"]["counts"]["processed"] == 40
    assert summary["run"]["counts"]["msbladder"] + summary["run"]["counts"]["ioab"] == 40
    assert "paths" not in summary["run"]["settings"]             # local folder names stay local


def test_a_failed_subject_is_left_out_and_counted_not_fatal(project, capsys):
    config, cfg, rows = project
    rows["case05"] = None
    assert main(["run", "-c", str(config)]) == 0
    assert "case05  FAILED" in capsys.readouterr().out
    assert len(pd.read_csv(cfg.paths.local_dir / "subjects.csv")) == 39
    manifest = json.loads((cfg.paths.local_dir / "run_manifest.json").read_text())
    assert manifest["counts"]["failed"] == 1 and "case05" in manifest["failed_subjects"]
    md = (cfg.paths.shareable_dir / "group_summary.md").read_text(encoding="utf-8")
    assert "39 of 40" in md and "1 failed" in md


def test_registration_qc_failure_blanks_the_tract_values(project):
    config, cfg, rows = project
    clinical = pd.read_csv(cfg.paths.clinical_csv, dtype={"subject_id": str})
    sid = clinical.loc[clinical["cohort"] == "msbladder", "subject_id"].iloc[0]
    n_ms = int((clinical["cohort"] == "msbladder").sum())
    rows[sid]["reg_ok"] = False
    main(["run", "-c", str(config)])
    subjects = pd.read_csv(cfg.paths.local_dir / "subjects.csv", dtype={"subject_id": str}).set_index("subject_id")
    bad = subjects.loc[sid]
    assert np.isnan(bad["tract_ATR_fa"]) and np.isnan(bad["tract_SLF_md"]) and np.isnan(bad["tract_ATR_L_fa"])
    assert bad["tract_ATR_voxels"] == 2000 and bad["motion_mean_mm"] > 0     # what was measured stays
    tests = read_tests(cfg)
    q4 = tests[(tests.question == "q4_symptom_severity") & (tests.outcome == "nbss_total")]
    assert (q4["n"] == n_ms - 1).all()
    md = (cfg.paths.shareable_dir / "group_summary.md").read_text(encoding="utf-8")
    assert "Registration QC failures (no tract values): 1" in md


def test_a_scan_that_is_not_the_expected_acquisition_is_kept_and_counted(project, capsys):
    config, cfg, rows = project
    rows["case02"].update({"n_directions": 60, "n_b0": 7, "acq_as_expected": False})
    main(["run", "-c", str(config)])
    assert "not 64 directions at b=1000 with 5 b0 (still processed): case02" in capsys.readouterr().out
    summary = json.loads((cfg.paths.shareable_dir / "group_summary.json").read_text())
    assert summary["run"]["counts"]["acquisition_not_as_expected"] == 1
    acquisition = pd.DataFrame(summary["acquisition"])
    assert acquisition["scans"].sum() == 40
    assert 1 in set(acquisition.loc[acquisition["n_directions"] == 60, "scans"])
    subjects = pd.read_csv(cfg.paths.local_dir / "subjects.csv", dtype={"subject_id": str}).set_index("subject_id")
    assert subjects.loc["case02", "tract_ATR_fa"] > 0


def test_small_groups_are_withheld_from_the_summary(project):
    config, cfg, _ = project
    clinical = pd.read_csv(cfg.paths.clinical_csv, dtype={"subject_id": str})
    ms = clinical.index[clinical["cohort"] == "msbladder"]
    clinical.loc[ms, "do_present"] = 0
    clinical.loc[ms[:2], "do_present"] = 1                # a group of two, under min_cell_n 3
    clinical.to_csv(cfg.paths.clinical_csv, index=False)
    main(["run", "-c", str(config)])
    desc = pd.read_csv(cfg.paths.shareable_dir / "descriptives.csv")
    small = desc[desc.group == "DO present"]
    assert (small["n"] == 2).all() and small["suppressed"].all() and small["mean"].isna().all()
    md = (cfg.paths.shareable_dir / "group_summary.md").read_text(encoding="utf-8")
    assert "withheld" in md
    q3 = read_tests(cfg)
    q3 = q3[q3.question == "q3_detrusor_overactivity"]
    assert q3["status"].str.startswith("skipped").all()


def test_q5_appears_only_when_turned_on(project):
    config, cfg, _ = project
    main(["run", "-c", str(config)])
    assert "q5_treatment_response" not in set(read_tests(cfg).question)
    config.write_text("stats: {bootstrap_resamples: 200}\nanalyses: {q5_treatment_response: true}\n"
                      "treatment_response: {min_nbss_improvement: 5}\n")
    main(["run", "-c", str(config)])
    assert "q5_treatment_response" in set(read_tests(cfg).question)
    md = (cfg.paths.shareable_dir / "group_summary.md").read_text(encoding="utf-8")
    assert "## Q5." in md and "Group 1 is responder" in md


def test_bad_table_stops_the_run_before_any_imaging(project, capsys, monkeypatch):
    config, cfg, _ = project
    monkeypatch.setattr(pipeline, "process_subject", lambda *a: pytest.fail("imaging started"))
    clinical = pd.read_csv(cfg.paths.clinical_csv, dtype={"subject_id": str})
    clinical.loc[0, "cohort"] = "controls"
    clinical.to_csv(cfg.paths.clinical_csv, index=False)
    assert main(["run", "-c", str(config)]) == 2
    assert "cohort must be one of msbladder | ioab" in capsys.readouterr().err


def test_a_scan_with_no_clinical_row_is_measured_but_enters_no_test(project):
    config, cfg, _ = project
    clinical = pd.read_csv(cfg.paths.clinical_csv, dtype={"subject_id": str})
    clinical.iloc[1:].to_csv(cfg.paths.clinical_csv, index=False)
    main(["run", "-c", str(config)])
    subjects = pd.read_csv(cfg.paths.local_dir / "subjects.csv", dtype={"subject_id": str}).set_index("subject_id")
    assert subjects.loc[clinical.loc[0, "subject_id"], "tract_ATR_fa"] > 0
    counts = json.loads((cfg.paths.shareable_dir / "group_summary.json").read_text())["run"]["counts"]
    assert counts["with_clinical"] == 39 and counts["msbladder"] + counts["ioab"] == 39


def test_several_workers_give_the_same_table_in_scan_order(project, monkeypatch):
    config, cfg, rows = project

    def in_reverse(scans, cfg, force):
        for sid in reversed(list(scans)):
            yield sid, dict(rows[sid]), None, 0.0

    monkeypatch.setattr(pipeline, "_process_all", in_reverse)
    main(["run", "-c", str(config)])
    subjects = pd.read_csv(cfg.paths.local_dir / "subjects.csv", dtype={"subject_id": str})
    assert list(subjects["subject_id"]) == IDS


def test_blank_failed_directly():
    table = pd.DataFrame({"subject_id": ["a", "b"], "reg_ok": [True, False],
                          "tract_ATR_fa": [0.4, 0.4], "tract_ATR_md": [0.7, 0.7],
                          "tract_ATR_voxels": [10, 10], "motion_mean_mm": [1.0, 1.0]})
    out = pipeline.blank_failed(table)
    assert out.loc[0, "tract_ATR_fa"] == 0.4
    assert np.isnan(out.loc[1, "tract_ATR_fa"]) and np.isnan(out.loc[1, "tract_ATR_md"])
    assert out.loc[1, "tract_ATR_voxels"] == 10 and out.loc[1, "motion_mean_mm"] == 1.0
    assert table.loc[1, "tract_ATR_fa"] == 0.4                   # the input is not changed


def test_numbers_in_the_summary_are_written_out_not_in_exponent_form():
    from dtic.report import _fmt, _p

    assert _fmt(1183.4) == "1183" and _fmt(0.41234) == "0.412" and _fmt(12) == "12"
    assert _fmt(float("nan")) == "" and _fmt(None) == ""
    assert _p(0.0004) == "<0.001" and _p(0.0312) == "0.031"


def test_a_summary_built_on_fake_tables_says_so_and_a_real_one_does_not(project):
    config, cfg, _ = project
    main(["run", "-c", str(config)])
    md = cfg.paths.shareable_dir / "group_summary.md"
    summary = cfg.paths.shareable_dir / "group_summary.json"
    assert "FAKE DATA" in md.read_text(encoding="utf-8")
    assert json.loads(summary.read_text())["fake_data"] is True
    clinical = pd.read_csv(cfg.paths.clinical_csv, dtype={"subject_id": str})
    clinical.drop(columns=[fakedata.FAKE_COLUMN]).to_csv(cfg.paths.clinical_csv, index=False)
    main(["run", "-c", str(config)])
    assert "FAKE DATA" not in md.read_text(encoding="utf-8")
    assert json.loads(summary.read_text())["fake_data"] is False


def test_defaults_ask_for_every_question_but_q5():
    assert from_dict({}).enabled_analyses() == [
        "q1_integrity_fc", "q2_ms_vs_ioab", "q3_detrusor_overactivity", "q4_symptom_severity"]
