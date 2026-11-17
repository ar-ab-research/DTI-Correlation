import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from dtic import fakedata, inputs, schemas
from dtic.cli import main
from dtic.config import from_dict, load
from dtic.inputs import InputError, discover_dwi, match_ids, subject_id


def touch_scan(folder, stem="dwi", sidecars=(".bval", ".bvec"), ext=".nii.gz"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}{ext}").write_bytes(b"")
    for suffix in sidecars:
        (folder / f"{stem}{suffix}").write_text("")
    return folder / f"{stem}{ext}"


def test_flat_folder_and_nested_layouts(tmp_path):
    touch_scan(tmp_path, "sub-01_dwi")
    touch_scan(tmp_path, "sub-02", ext=".nii")
    touch_scan(tmp_path / "sub-03")
    touch_scan(tmp_path / "sub-04" / "ses-01" / "dwi", "sub-04_ses-01_dwi")     # a BIDS folder
    touch_scan(tmp_path / "sub-05", "scan", sidecars=(".bvals", ".bvecs"))      # the only image there
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / ".hidden").mkdir()
    found = discover_dwi(tmp_path)
    assert list(found) == ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]
    assert found["sub-01"].bval.name == "sub-01_dwi.bval"
    assert found["sub-04"].image.name == "sub-04_ses-01_dwi.nii.gz"
    assert found["sub-05"].bvec.name == "scan.bvecs"


def test_a_folder_with_other_images_uses_the_one_named_dwi(tmp_path):
    touch_scan(tmp_path / "sub-01")
    (tmp_path / "sub-01" / "T1w.nii.gz").write_bytes(b"")
    assert discover_dwi(tmp_path)["sub-01"].image.name == "dwi.nii.gz"


def test_a_scan_without_its_bval_and_bvec_is_an_error_that_says_so(tmp_path):
    touch_scan(tmp_path / "sub-01", sidecars=(".bval",))
    with pytest.raises(InputError, match="no dwi.bvec beside it"):
        discover_dwi(tmp_path)
    # a lone T1 is caught the same way: it has no b-values
    other = tmp_path / "other"
    touch_scan(other / "sub-02", "T1w", sidecars=())
    with pytest.raises(InputError, match="T1w.bval or T1w.bvec"):
        discover_dwi(other)


def test_ambiguous_folder_and_duplicate_subject_are_errors(tmp_path):
    touch_scan(tmp_path / "sub-01", "a")
    touch_scan(tmp_path / "sub-01", "b")
    with pytest.raises(InputError, match="cannot tell which image"):
        discover_dwi(tmp_path)
    other = tmp_path / "other"
    touch_scan(other, "sub-02_dwi")
    touch_scan(other / "sub-02")
    with pytest.raises(InputError, match="two scans for subject sub-02"):
        discover_dwi(other)
    with pytest.raises(InputError, match="not a folder"):
        discover_dwi(tmp_path / "missing")


def test_subject_id_strips_only_the_dwi_tag():
    assert subject_id("sub-01_dwi") == "sub-01"
    assert subject_id("sub-01-DTI") == "sub-01"
    assert subject_id("dwi") == "dwi"
    assert subject_id("sub-dwi-01") == "sub-dwi-01"


def test_a_single_volume_is_refused_with_a_reason(tmp_path):
    import nibabel as nib

    path = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.int16), np.eye(4)), str(path))
    with pytest.raises(InputError, match="expected a 4D diffusion series"):
        inputs.check_dwi(path)
    series = tmp_path / "dwi.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8, 12), np.int16), np.diag([2.0, 2.0, 2.5, 1.0])), str(series))
    assert inputs.check_dwi(series)["voxel_mm"] == (2.0, 2.0, 2.5)


def test_match_ids():
    got = match_ids({"a", "b", "c"}, {"a", "b", "x"}, {"a", "c"})
    assert got["complete"] == ["a"]
    assert got["scan_without_clinical"] == ["c"] and got["scan_without_fc"] == ["b"]
    assert got["clinical_without_scan"] == ["x"] and got["fc_without_scan"] == []


# ---- fake tables -----------------------------------------------------------------------

IDS = fakedata.default_ids(60)


def test_fake_data_fits_the_schemas(tmp_path):
    clinical, fc = fakedata.make_fake(IDS)
    clinical.to_csv(tmp_path / "clinical.csv", index=False)
    fc.to_csv(tmp_path / "fc.csv", index=False)
    q5 = from_dict({"analyses": {"q5_treatment_response": True},
                    "treatment_response": {"min_nbss_improvement": 5}})
    loaded = schemas.load_clinical(tmp_path / "clinical.csv", q5)
    assert len(loaded) == 60 and (loaded[fakedata.FAKE_COLUMN] == "1").all()
    assert set(loaded["cohort"]) == {"msbladder", "ioab"}
    assert len(schemas.load_fc(tmp_path / "fc.csv")) == len(fc)


def test_fake_cohorts_look_like_the_studies():
    clinical, fc = fakedata.make_fake(IDS)
    ms, ioab = clinical[clinical.cohort == "msbladder"], clinical[clinical.cohort == "ioab"]
    assert (ms["sex"] == "female").all()                        # the MS study enrols women
    assert {"female", "male"} == set(ioab["sex"])
    assert ioab["age"].between(40, 80).all() and ms["age"].between(18, 75).all()
    # symptom, urodynamic and FC values exist for the MS cohort only
    assert ioab[["do_present", "nbss_total", "diary_voids_per_day", "arm"]].isna().all().all()
    assert ms[["do_present", "nbss_total", "arm", "nbss_total_post"]].notna().all().all()
    assert set(fc["subject_id"]) == set(ms["subject_id"])
    assert set(ms["do_present"]) == {0, 1} and set(ms["arm"]) == {"active", "sham"}


def test_fake_data_is_seeded():
    a, b, c = fakedata.make_fake(IDS, seed=1), fakedata.make_fake(IDS, seed=1), fakedata.make_fake(IDS, seed=2)
    assert a[0].equals(b[0]) and a[1].equals(b[1])
    assert not a[0].equals(c[0])


def test_planted_effects_follow_the_supplied_integrity_and_nothing_else_does():
    rng = np.random.default_rng(5)
    integrity = dict(zip(IDS, rng.normal(0.42, 0.03, 60)))
    clinical, fc = fakedata.make_fake(IDS, seed=0, integrity=integrity)
    fa = clinical["subject_id"].map(integrity)
    ms = clinical["cohort"] == "msbladder"
    assert fa[ms].mean() < fa[~ms].mean()                       # MSBladder has the lower integrity
    assert sps.spearmanr(fa[ms], clinical.loc[ms, "nbss_total"].astype(float)).statistic < -0.5
    assert abs(sps.spearmanr(fa[ms], clinical.loc[ms, "diary_voids_per_day"].astype(float)).statistic) < 0.4
    wide = schemas.fc_wide(fc).set_index("subject_id")
    planted = wide[schemas.fc_column("full", fakedata.PLANTED_CONNECTION)]
    assert sps.spearmanr(planted.index.map(integrity), planted).statistic > 0.5
    other = wide[schemas.fc_column("full", "net03__net04")]
    assert abs(sps.spearmanr(other.index.map(integrity), other).statistic) < 0.4


def test_fake_data_needs_distinct_ids():
    with pytest.raises(ValueError):
        fakedata.make_fake(["a", "a", "b", "c"])
    with pytest.raises(ValueError):
        fakedata.make_fake(["a", "b"])


def test_cli_writes_fake_tables_and_will_not_overwrite(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    assert main(["make-fake-data", "-c", str(config), "--n", "12"]) == 0
    cfg = load(config)
    assert len(pd.read_csv(cfg.paths.clinical_csv)) == 12
    assert "FAKE" in capsys.readouterr().out
    assert main(["make-fake-data", "-c", str(config)]) == 2
    assert "not overwriting" in capsys.readouterr().err
    assert main(["make-fake-data", "-c", str(config), "--n", "8", "--force"]) == 0
    assert len(pd.read_csv(cfg.paths.clinical_csv)) == 8


def test_cli_uses_scan_ids_and_the_measured_fa_when_there_is_a_run(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    cfg = load(config)
    ids = [f"p{i:02d}" for i in range(20)]
    for sid in ids:
        touch_scan(cfg.paths.dwi_dir / sid)
    assert main(["make-fake-data", "-c", str(config)]) == 0
    assert "random stand-in" in capsys.readouterr().out
    assert list(pd.read_csv(cfg.paths.clinical_csv, dtype=str)["subject_id"]) == ids
    # after a run, the planted effects follow the FA that was measured
    fa = np.linspace(0.35, 0.5, 20)
    cfg.paths.local_dir.mkdir(parents=True)
    pd.DataFrame({"subject_id": ids, "tract_ATR_fa": fa, "tract_SLF_fa": fa}).to_csv(
        cfg.paths.local_dir / "imaging.csv", index=False)
    assert main(["make-fake-data", "-c", str(config), "--force"]) == 0
    assert "follow the FA measured" in capsys.readouterr().out
    clinical = pd.read_csv(cfg.paths.clinical_csv, dtype={"subject_id": str})
    by_id = dict(zip(ids, fa))
    ms = clinical["cohort"] == "msbladder"
    assert clinical.loc[ms, "subject_id"].map(by_id).mean() < clinical.loc[~ms, "subject_id"].map(by_id).mean()


def test_fake_scans_are_never_written_beside_existing_scans(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    touch_scan(load(config).paths.dwi_dir / "real-01")
    assert main(["make-fake-data", "-c", str(config), "--with-scans"]) == 2
    assert "already holds scans" in capsys.readouterr().err


def test_directions_are_unit_vectors_spread_over_a_half_sphere():
    g = fakedata.directions(64)
    assert np.allclose(np.linalg.norm(g, axis=1), 1)
    assert (g[:, 2] > 0).all()
    # no two directions nearly the same
    dots = np.abs(g @ g.T) - np.eye(64)
    assert dots.max() < 0.99
