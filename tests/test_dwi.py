import json
import os

import nibabel as nib
import numpy as np
import pytest
from dipy.core.gradients import gradient_table

from dtic import dwi, fakedata
from dtic.config import from_dict
from dtic.inputs import Scan

CFG = from_dict({})


def bvals(n_b0=5, n_dir=64, b=1000.0, extra=()):
    return np.concatenate([np.zeros(n_b0), np.full(n_dir, b), np.asarray(extra, float)])


# ---- reading the scan --------------------------------------------------------------------

def write_gradients(tmp_path, vals, vecs):
    np.savetxt(tmp_path / "dwi.bval", np.atleast_2d(vals), fmt="%g")
    np.savetxt(tmp_path / "dwi.bvec", vecs, fmt="%.6f")
    return Scan(tmp_path / "dwi.nii.gz", tmp_path / "dwi.bval", tmp_path / "dwi.bvec")


def test_b_vectors_are_read_as_three_rows_or_three_columns(tmp_path):
    vecs = np.vstack([np.zeros((1, 3)), fakedata.directions(7)])
    for layout in (vecs.T, vecs):
        scan = write_gradients(tmp_path, bvals(1, 7), layout)
        got_vals, got_vecs = dwi.read_gradients(scan, 8)
        assert got_vals.shape == (8,) and got_vecs.shape == (8, 3)
        assert np.allclose(got_vecs, vecs, atol=1e-5)


def test_a_count_that_does_not_match_the_image_is_an_error(tmp_path):
    scan = write_gradients(tmp_path, bvals(1, 7), np.zeros((3, 8)))
    with pytest.raises(dwi.DwiError, match="9 volumes, but 8 b-values"):
        dwi.read_gradients(scan, 9)
    (tmp_path / "dwi.bval").write_text("0 1000 oops\n")
    with pytest.raises(dwi.DwiError, match="cannot be read"):
        dwi.read_gradients(scan, 3)


def test_the_protocol_acquisition_is_recognised():
    acq = dwi.describe_acquisition(bvals(), CFG)
    assert (acq["n_volumes"], acq["n_b0"], acq["n_directions"], acq["b_value"]) == (69, 5, 64, 1000)
    assert acq["acq_as_expected"] and acq["n_other_volumes"] == 0 and acq["keep"].all()


def test_another_acquisition_is_kept_and_flagged():
    one_b0 = dwi.describe_acquisition(bvals(n_b0=1), CFG)
    assert one_b0["n_b0"] == 1 and not one_b0["acq_as_expected"]
    # nominal values the scanner reports a little off still count: b0 at 5, shell at 995 to 1005
    jitter = np.concatenate([np.full(7, 5.0), np.full(20, 995.0), np.full(40, 1005.0)])
    got = dwi.describe_acquisition(jitter, CFG)
    assert (got["n_b0"], got["n_directions"], got["b_value"]) == (7, 60, 1005)
    assert not got["acq_as_expected"]


def test_only_the_shell_near_the_configured_b_value_is_fitted():
    vals = bvals(1, 64, extra=np.r_[np.full(64, 2000.0), np.full(64, 3500.0)])
    acq = dwi.describe_acquisition(vals, CFG)
    assert acq["n_directions"] == 64 and acq["n_other_volumes"] == 128 and not acq["acq_as_expected"]
    assert acq["keep"].sum() == 65 and not acq["keep"][65:].any()
    other = dwi.describe_acquisition(vals, from_dict({"acquisition": {"b_value": 2000}}))
    assert other["b_value"] == 2000 and other["keep"][65:129].all()


def test_a_scan_a_tensor_cannot_be_fitted_to_is_refused():
    with pytest.raises(dwi.DwiError, match="no b0 volume"):
        dwi.describe_acquisition(np.full(64, 1000.0), CFG)
    with pytest.raises(dwi.DwiError, match="3 volumes near b=1000.*2000 x30"):
        dwi.describe_acquisition(np.r_[0, np.full(3, 1000.0), np.full(30, 2000.0)], CFG)


# ---- motion --------------------------------------------------------------------------------

def transforms(n, shift=(0, 0, 0), scale=1.0):
    out = np.repeat(np.eye(4)[:, :, None], n, axis=2)
    out[:3, :3] *= scale
    out[:3, 3] = np.asarray(shift, float)[:, None]
    return out


def test_displacement_is_measured_from_the_usual_position_of_each_kind_of_volume():
    b0s = np.array([True] + [False] * 9)
    centre = np.array([10.0, -5.0, 20.0])
    assert np.allclose(dwi.displacement_mm(transforms(10), centre, b0s), 0)
    # an offset every diffusion volume shares is contrast, not movement
    shared = transforms(10, shift=(0, 0, 3))
    shared[..., 0] = np.eye(4)
    assert np.allclose(dwi.displacement_mm(shared, centre, b0s), 0, atol=1e-9)
    # one volume that really moved stands out by the distance it moved
    moved = shared.copy()
    moved[:3, 3, 4] += [2.0, 0, 0]
    got = dwi.displacement_mm(moved, centre, b0s)
    assert got[4] == pytest.approx(2.0) and np.allclose(np.delete(got, 4), 0, atol=1e-9)


def test_stretch_is_not_counted_as_movement_but_a_turn_is():
    b0s = np.zeros(6, bool)
    centre = np.zeros(3)
    stretched = transforms(6)
    stretched[:3, :3, 2] *= 1.05
    assert np.allclose(dwi.displacement_mm(stretched, centre, b0s), 0, atol=1e-9)
    turned = transforms(6)
    a = np.deg2rad(2)
    turned[:3, :3, 2] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    got = dwi.displacement_mm(turned, centre, b0s)
    # a 2 degree turn of an 80 mm sphere: R * sqrt(2/5 * 2 * (1 - cos a))
    assert got[2] == pytest.approx(80 * np.sqrt(0.8 * (1 - np.cos(a))), rel=1e-6)


# ---- the tensor ------------------------------------------------------------------------------

def test_the_fit_recovers_a_known_tensor_and_reports_md_in_the_stated_unit():
    g = np.vstack([np.zeros((2, 3)), fakedata.directions(30)])
    gtab = gradient_table(bvals(2, 30), bvecs=g)
    fa_true, md_true = 0.6, 0.8e-3
    root = np.sqrt(3 - 2 * fa_true ** 2)
    lam_par, lam_perp = md_true * (1 + 2 * fa_true / root), md_true * (1 - fa_true / root)
    fibre = np.array([0.6, 0.0, 0.8])
    signal = 1000 * np.exp(-gtab.bvals * (lam_perp + (lam_par - lam_perp) * (g @ fibre) ** 2))
    data = np.tile(signal, (4, 4, 4, 1)).astype(np.float32)
    mask = np.zeros((4, 4, 4), bool)
    mask[1:3, 1:3, 1:3] = True
    fa, md = dwi.fit_tensor(data, gtab, mask)
    assert np.allclose(fa[mask], fa_true, atol=1e-3)
    assert np.allclose(md[mask], 0.8, atol=1e-3)                 # 10^-3 mm2/s
    assert (fa[~mask] == 0).all() and (md[~mask] == 0).all()


def small_template():
    """A blob with a bright core, standing in for the FA template."""
    x, y, z = np.meshgrid(*(np.arange(s) - (s - 1) / 2 for s in (22, 26, 20)), indexing="ij")
    r = np.sqrt((x / 9) ** 2 + (y / 11) ** 2 + (z / 8) ** 2)
    return np.where(r < 1, 0.2 + 0.5 * np.clip(1 - r, 0, 1), 0).astype(np.float32)


def test_a_fake_scan_carries_the_fa_it_was_built_with():
    template = small_template()
    core = template > 0.4
    means = {}
    for factor in (0.8, 1.0):
        data, vals, vecs = fakedata.fake_scan(template, factor, np.random.default_rng(0), 30, 3)
        assert data.dtype == np.int16 and data.shape == template.shape + (33,)
        assert (vals == 0).sum() == 3 and (vals == 1000).sum() == 30
        gtab = gradient_table(vals, bvecs=vecs)
        fa, md = dwi.fit_tensor(data.astype(np.float32), gtab, core)
        means[factor] = (fa[core].mean(), md[core].mean())
        assert fa[core].mean() == pytest.approx(template[core].mean() * factor, abs=0.03)
    assert means[0.8][0] < means[1.0][0] and means[0.8][1] > means[1.0][1]


# ---- one subject, start to finish ---------------------------------------------------------------

@pytest.fixture
def subject(tmp_path):
    cfg = from_dict({"preprocessing": {"motion_correction": False},
                     "acquisition": {"directions": 30, "b0_volumes": 3}}, base=tmp_path)
    data, vals, vecs = fakedata.fake_scan(small_template(), 0.9, np.random.default_rng(1), 30, 3)
    folder = cfg.paths.dwi_dir / "s1"
    folder.mkdir(parents=True)
    nib.save(nib.Nifti1Image(data, np.diag([3.0, 3.0, 3.0, 1.0])), str(folder / "dwi.nii.gz"))
    np.savetxt(folder / "dwi.bval", vals[None], fmt="%g")
    np.savetxt(folder / "dwi.bvec", vecs.T, fmt="%.6f")
    return cfg, Scan(folder / "dwi.nii.gz", folder / "dwi.bval", folder / "dwi.bvec")


def test_preprocessing_writes_the_maps_and_the_numbers(subject):
    cfg, scan = subject
    got = dwi.preprocess_subject("s1", scan, cfg)
    out = dwi.subject_dir(cfg, "s1")
    assert out == cfg.paths.local_dir / "subjects" / "s1"
    fa = nib.load(str(out / dwi.FA_FILE))
    mask = np.asarray(nib.load(str(out / dwi.MASK_FILE)).dataobj) > 0
    assert fa.shape == (22, 26, 20) and np.allclose(fa.affine, np.diag([3.0, 3.0, 3.0, 1.0]))
    values = np.asarray(fa.dataobj)
    assert 0 <= values.min() and values.max() <= 1 and (values[~mask] == 0).all()
    md = np.asarray(nib.load(str(out / dwi.MD_FILE)).dataobj)
    assert 0.5 < md[mask].mean() < 1.5                           # 10^-3 mm2/s
    assert got["processed_now"] and got["acq_as_expected"] and got["voxel_mm"] == "3 x 3 x 3"
    assert got["motion_mean_mm"] is None                         # correction was off
    assert got["brain_volume_ml"] == pytest.approx(mask.sum() * 27 / 1000)
    assert json.loads((out / dwi.DWI_JSON).read_text())["n_directions"] == 30


def test_a_finished_subject_is_reused_until_the_scan_or_the_settings_change(subject):
    cfg, scan = subject
    dwi.preprocess_subject("s1", scan, cfg)
    assert dwi.preprocess_subject("s1", scan, cfg)["processed_now"] is False
    assert dwi.preprocess_subject("s1", scan, cfg, force=True)["processed_now"] is True
    other = from_dict({"preprocessing": {"motion_correction": False, "fit_method": "OLS"},
                       "acquisition": {"directions": 30, "b0_volumes": 3}}, base=cfg.paths.dwi_dir.parents[1])
    assert dwi.preprocess_subject("s1", scan, other)["processed_now"] is True
    assert dwi.preprocess_subject("s1", scan, other)["processed_now"] is False
    later = (dwi.subject_dir(cfg, "s1") / dwi.DWI_JSON).stat().st_mtime + 10
    os.utime(scan.bvec, (later, later))                          # new b-vectors are a new scan
    assert dwi.preprocess_subject("s1", scan, other)["processed_now"] is True


def test_an_unusable_scan_names_the_subject(subject):
    cfg, scan = subject
    np.savetxt(scan.bval, np.full((1, 33), 2000.0), fmt="%g")
    with pytest.raises(dwi.DwiError, match="s1: no b0 volume"):
        dwi.preprocess_subject("s1", scan, cfg)


def test_b_vectors_follow_the_head_when_a_volume_is_turned_back(monkeypatch):
    """The head turns by R after the b0. A volume then measures along R^T g in the head's frame.

    dipy hands back, for each volume, the transform from a point in the reference to the
    matching point in that volume: for a head turned by R, that is R itself. This was
    checked against dipy's real registration on a scan turned by a known angle; here the
    registration is replaced by its answer, so the test is exact and takes no time.
    """
    import dipy.align

    a = np.deg2rad(6)
    turn = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    g = fakedata.directions(5)
    # b0 volumes first, in the middle and last: only the diffusion volumes are rotated
    is_b0 = np.array([True, False, False, True, False, False, False, True])
    vecs = np.zeros((8, 3))
    vecs[~is_b0] = g
    gtab = gradient_table(np.where(is_b0, 0.0, 1000.0), bvecs=vecs)
    found = np.repeat(np.eye(4)[:, :, None], 8, axis=2)
    found[:3, :3, [1, 2, 4]] = turn[:, :, None]          # three volumes were taken with the head turned
    data = np.zeros((4, 4, 4, 8), np.float32)

    def stand_in(volumes, table, affine=None, pipeline=None):
        assert pipeline == ["center_of_mass", "translation", "rigid"]
        return nib.Nifti1Image(volumes, affine), found

    monkeypatch.setattr(dipy.align, "motion_correction", stand_in)
    fixed, rotated, moves = dwi.correct_motion(data, gtab, np.eye(4), "rigid")
    assert fixed.shape == data.shape and moves.shape == (4, 4, 8)
    got = rotated.bvecs[~is_b0]
    assert np.allclose(got[[0, 1, 2]], (g @ turn)[[0, 1, 2]], atol=1e-6)     # rows of g @ R are R^T g
    assert not np.allclose(got[0], (g @ turn.T)[0], atol=1e-2)               # and not the other way round
    assert np.allclose(got[[3, 4]], g[[3, 4]], atol=1e-6)                    # a still head keeps its vectors
    assert np.allclose(rotated.bvals, gtab.bvals)


def test_the_affine_setting_adds_the_step_that_takes_out_eddy_distortion(monkeypatch):
    import dipy.align

    asked = {}

    def stand_in(volumes, table, affine=None, pipeline=None):
        asked["pipeline"] = pipeline
        return nib.Nifti1Image(volumes, affine), np.repeat(np.eye(4)[:, :, None], 3, axis=2)

    monkeypatch.setattr(dipy.align, "motion_correction", stand_in)
    gtab = gradient_table(np.array([0.0, 1000.0, 1000.0]), bvecs=np.vstack([np.zeros(3), fakedata.directions(2)]))
    dwi.correct_motion(np.zeros((2, 2, 2, 3), np.float32), gtab, np.eye(4), "affine")
    assert asked["pipeline"] == ["center_of_mass", "translation", "rigid", "affine"]
