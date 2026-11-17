"""The whole run, start to finish: scans and tables in, group summary out."""
from __future__ import annotations

import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib import metadata

import numpy as np
import pandas as pd

from dtic import __version__, atlas, dwi, inputs, register, report, schemas, stats, tracts
from dtic.config import Config
from dtic.errors import PipelineError

IMAGING_CSV = "imaging.csv"
SUBJECTS_CSV = "subjects.csv"

ACQUISITION_COLUMNS = ["n_volumes", "n_b0", "n_directions", "b_value", "n_other_volumes",
                       "acq_as_expected", "voxel_mm"]
QC_COLUMNS = ["motion_mean_mm", "motion_max_mm", "brain_volume_ml"]
# how many threads numpy, dipy and ants may each start
THREAD_SETTINGS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")


def tract_names(cfg: Config) -> list[str]:
    return list(cfg.tracts.primary) + list(cfg.tracts.secondary)


def blank_failed(table: pd.DataFrame) -> pd.DataFrame:
    """Rows that failed registration QC get no tract values.

    A registration that failed puts the tracts in the wrong place, so those values
    are blanked rather than carried into the statistics.
    """
    out = table.copy()
    bad = ~out["reg_ok"].astype(bool)
    means = [c for c in out.columns if c.startswith("tract_") and c.endswith(("_fa", "_md"))]
    out.loc[bad, means] = np.nan
    return out


def subject_table(imaging: pd.DataFrame, clinical: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """One row per scanned subject, with whatever the two tables hold for them."""
    table = imaging.merge(clinical, on="subject_id", how="left")
    return table.merge(schemas.fc_wide(fc), on="subject_id", how="left")


def process_subject(sid: str, scan: inputs.Scan, cfg: Config, force: bool) -> dict:
    """Preprocess, register and measure one subject. Returns one row of the imaging table."""
    inputs.check_dwi(scan.image)
    row = {"subject_id": sid}
    pre = dwi.preprocess_subject(sid, scan, cfg, force=force)
    row.update({k: pre[k] for k in ACQUISITION_COLUMNS + QC_COLUMNS})
    reg = register.register_subject(sid, cfg, force=force)
    row.update({k: reg[k] for k in ("reg_similarity", "reg_ok")})
    means = tracts.tract_subject(sid, cfg)
    row.update({k: v for k, v in means.items() if k != "subject_id"})
    return row


def _process_one(sid: str, scan: inputs.Scan, cfg: Config, force: bool):
    """(sid, row, error, seconds). One unusable scan should not cost the rest of the cohort."""
    t0 = time.time()
    try:
        return sid, process_subject(sid, scan, cfg, force), None, time.time() - t0
    except PipelineError as exc:
        return sid, None, str(exc), time.time() - t0


def _process_all(scans: dict[str, inputs.Scan], cfg: Config, force: bool):
    """Process every scan, several at a time when run.workers says so. Yields as each finishes."""
    workers = min(cfg.run.workers, len(scans))
    if workers <= 1:
        for sid, scan in scans.items():
            yield _process_one(sid, scan, cfg, force)
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # each worker would otherwise start a thread per core, and several workers doing
    # that at once run far slower than one. the workers inherit these when they start
    share = str(max(1, (os.cpu_count() or 1) // workers))
    before = {name: os.environ.get(name) for name in THREAD_SETTINGS}
    os.environ.update({name: share for name in THREAD_SETTINGS})
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(_process_one, sid, scan, cfg, force) for sid, scan in scans.items()]
            for job in as_completed(jobs):
                yield job.result()
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def versions() -> dict:
    out = {"dtic": __version__, "python": platform.python_version()}
    for name in ("dipy", "antspyx", "numpy", "scipy", "pandas", "statsmodels", "nibabel"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def run(cfg: Config, force: bool = False, log=print) -> dict:
    """Run every stage. Returns the manifest that is also written to the output folder."""
    started = time.time()
    timings = {}

    def stage(name: str) -> None:
        timings[name] = round(time.time() - stage.t0, 1)
        stage.t0 = time.time()

    stage.t0 = started

    # read and check everything that can be wrong before any slow step
    scans = inputs.discover_dwi(cfg.paths.dwi_dir)
    if not scans:
        raise PipelineError(f"no diffusion scans found in {cfg.paths.dwi_dir}")
    clinical = schemas.load_clinical(cfg.paths.clinical_csv, cfg)
    fc = schemas.load_fc(cfg.paths.fc_csv)
    matched = inputs.match_ids(set(scans), set(clinical["subject_id"]), set(fc["subject_id"]))
    log(f"inputs    {len(scans)} scans, {len(clinical)} clinical rows, "
        f"{fc['subject_id'].nunique()} subjects with FC, {len(matched['complete'])} with all three")
    for key in ("scan_without_clinical", "scan_without_fc", "clinical_without_scan", "fc_without_scan"):
        if matched[key]:
            log(f"  note: {key.replace('_', ' ')}: {', '.join(matched[key])}")
    atlas.jhu_paths(cfg, log=log)
    stage("inputs")

    rows, failed = {}, {}
    for i, (sid, row, error, seconds) in enumerate(_process_all(scans, cfg, force), 1):
        if error is None:
            rows[sid] = row
            log(f"subject   {i}/{len(scans)} {sid}  {seconds:.0f} s")
        else:
            failed[sid] = error
            log(f"subject   {i}/{len(scans)} {sid}  FAILED: {error}")
    # in scan order, whatever order they finished in
    rows = [rows[sid] for sid in scans if sid in rows]
    if not rows:
        raise PipelineError("every subject failed, see the messages above")
    stage("subjects")

    imaging = blank_failed(pd.DataFrame(rows))
    qc_failed = sorted(imaging.loc[~imaging["reg_ok"].astype(bool), "subject_id"])
    if qc_failed:
        log(f"  note: registration QC failed, no tract values: {', '.join(qc_failed)}")
    unexpected = sorted(imaging.loc[~imaging["acq_as_expected"].astype(bool), "subject_id"])
    if unexpected:
        acq = cfg.acquisition
        log(f"  note: not {acq.directions} directions at b={acq.b_value:g} with {acq.b0_volumes} b0 "
            f"(still processed): {', '.join(unexpected)}")

    local = cfg.paths.local_dir
    local.mkdir(parents=True, exist_ok=True)
    imaging.to_csv(local / IMAGING_CSV, index=False)
    table = subject_table(imaging, clinical, fc)
    table.to_csv(local / SUBJECTS_CSV, index=False)
    log(f"wrote     {local / SUBJECTS_CSV}  (per subject, stays on this machine)")
    stage("tables")

    tests = stats.run_analyses(table, cfg)
    counts = {
        "scans": len(scans), "processed": len(rows), "failed": len(failed),
        "registration_qc_failed": len(qc_failed),
        "acquisition_not_as_expected": len(unexpected),
        "with_clinical": int(table["subject_id"].isin(clinical["subject_id"]).sum()),
        "with_fc": int(table["subject_id"].isin(fc["subject_id"]).sum()),
        **{cohort: int(stats.in_cohort(table, cohort).sum()) for cohort in schemas.COHORTS},
    }
    manifest = {
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_digest": cfg.digest(),
        "settings": {k: v for k, v in cfg.as_dict().items() if k != "paths"},
        "versions": versions(),
        "counts": counts,
        "seconds": timings,
    }
    written = report.write(table, tests, cfg, manifest)
    stage("stats")
    manifest["seconds"] = timings
    manifest["seconds"]["total"] = round(time.time() - started, 1)
    # the manifest with subject ids of failures stays local. the shareable copy has counts only
    (local / "run_manifest.json").write_text(
        json.dumps({**manifest, "failed_subjects": failed, "registration_qc_failed": qc_failed,
                    "acquisition_not_as_expected": unexpected, "id_matching": matched}, indent=2),
        encoding="utf-8")
    for path in written:
        log(f"wrote     {path}")
    return manifest
