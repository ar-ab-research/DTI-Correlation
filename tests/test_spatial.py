import json

import nibabel as nib
import numpy as np
import pytest
from nibabel.processing import resample_to_output
from scipy import ndimage

from dtic import atlas, dwi, register
from dtic.config import from_dict
from dtic.tracts import TRACTS_FILE, tract_means, tract_subject

XML = """<?xml version="1.0" encoding="ISO-8859-1"?>
<atlas version="1.0"><header><name>JHU White-Matter Tractography Atlas</name></header><data>
<label index="0" x="98" y="117" z="75">Anterior thalamic radiation L</label>
<label index="1" x="83" y="117" z="76">Anterior thalamic radiation R</label>
<label index="2" x="96" y="99" z="36">Corticospinal tract L</label>
<label index="14" x="129" y="112" z="102">Superior longitudinal fasciculus L</label>
<label index="15" x="40" y="121" z="99">Superior longitudinal fasciculus R</label>
<label index="18" x="140" y="89" z="61">Superior longitudinal fasciculus (temporal part) L</label>
<label index="19" x="52" y="116" z="103">Superior longitudinal fasciculus (temporal part) R</label>
</data></atlas>"""


@pytest.fixture
def names(tmp_path):
    path = tmp_path / "JHU-tracts.xml"
    path.write_text(XML, encoding="iso-8859-1")
    return atlas.read_tract_names(path)


def test_label_value_is_xml_index_plus_one(names):
    assert names["Anterior thalamic radiation L"] == 1
    assert names["Superior longitudinal fasciculus R"] == 16
    assert names["Superior longitudinal fasciculus (temporal part) R"] == 20


def test_default_config_tracts_resolve_and_the_main_slf_excludes_the_temporal_part(names):
    cfg = from_dict({})
    primary = atlas.tract_values(cfg.tracts.primary, names)
    assert primary == {"ATR": [1, 2], "SLF": [15, 16]}
    secondary = atlas.tract_values(cfg.tracts.secondary, names)
    assert secondary["SLF_temporal"] == [19, 20] and secondary["ATR_L"] == [1]


def test_unknown_tract_is_an_error_that_lists_what_exists(names):
    with pytest.raises(atlas.AtlasError, match="Fornix.*available: Anterior thalamic"):
        atlas.tract_values({"X": ["Fornix"]}, names)


def test_unreadable_atlas_xml(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<atlas><data>")
    with pytest.raises(atlas.AtlasError):
        atlas.read_tract_names(bad)
    empty = tmp_path / "empty.xml"
    empty.write_text("<atlas><data></data></atlas>")
    with pytest.raises(atlas.AtlasError, match="no tract labels"):
        atlas.read_tract_names(empty)


def test_the_fa_template_is_fetched_with_the_tract_labels(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setattr(atlas, "fetch", lambda url, dest: fetched.append((url, dest.name)))
    cfg = from_dict({}, base=tmp_path)
    paths = atlas.jhu_paths(cfg, log=lambda message: None)
    assert set(paths) == {"labels", "xml", "fa"}
    assert paths["fa"] == cfg.paths.cache_dir / "jhu" / "JHU-ICBM-FA-1mm.nii.gz"
    assert sorted(name for _, name in fetched) == [
        "JHU-ICBM-FA-1mm.nii.gz", "JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz", "JHU-tracts.xml"]
    assert all(url.startswith("https://git.fmrib.ox.ac.uk/fsl/data_atlases/") for url, _ in fetched)


# ---- tract means -------------------------------------------------------------------------

def volume():
    """Two labels of different size and FA, a strip of low FA, and one voxel outside the mask."""
    labels = np.zeros((6, 6, 6), np.int16)
    labels[0:2, :, :] = 1          # 72 voxels, left
    labels[4:5, :, :] = 2          # 36 voxels, right
    fa = np.zeros(labels.shape, np.float32)
    fa[labels == 1] = 0.4
    fa[labels == 2] = 0.7
    md = np.where(labels == 1, 0.8, 0.6).astype(np.float32)
    mask = np.ones(labels.shape, bool)
    return labels, fa, md, mask


def test_a_bilateral_tract_is_one_pool_of_voxels():
    labels, fa, md, mask = volume()
    got = tract_means(labels, fa, md, mask, {"T": [1, 2], "T_L": [1], "T_R": [2]})
    assert got["tract_T_L_fa"] == pytest.approx(0.4) and got["tract_T_R_fa"] == pytest.approx(0.7)
    # the larger side counts for more: (72 * 0.4 + 36 * 0.7) / 108, not the mean of the two sides
    assert got["tract_T_fa"] == pytest.approx(0.5)
    assert got["tract_T_md"] == pytest.approx((72 * 0.8 + 36 * 0.6) / 108)
    assert got["tract_T_voxels"] == 108 and got["tract_T_kept_pct"] == 100


def test_only_voxels_with_enough_fa_inside_the_mask_count():
    labels, fa, md, mask = volume()
    fa[0, :, 0] = 0.1              # six voxels of the left label that look like fluid
    md[0, :, 0] = 3.0
    mask[1, 0, 0] = False          # and one outside the brain mask
    fa[1, 0, 0] = 0.9
    kept = tract_means(labels, fa, md, mask, {"L": [1]}, min_fa=0.2)
    assert kept["tract_L_voxels"] == 72 - 6 - 1
    assert kept["tract_L_fa"] == pytest.approx(0.4) and kept["tract_L_md"] == pytest.approx(0.8)
    assert kept["tract_L_kept_pct"] == pytest.approx(100 * 65 / 71)      # of the voxels inside the mask
    everything = tract_means(labels, fa, md, mask, {"L": [1]}, min_fa=0.0)
    assert everything["tract_L_voxels"] == 71
    assert everything["tract_L_fa"] == pytest.approx((65 * 0.4 + 6 * 0.1) / 71)
    assert everything["tract_L_md"] == pytest.approx((65 * 0.8 + 6 * 3.0) / 71)


def test_the_threshold_is_inclusive():
    labels, fa, md, mask = volume()
    assert tract_means(labels, fa, md, mask, {"R": [2]}, min_fa=0.7)["tract_R_voxels"] == 36
    assert tract_means(labels, fa, md, mask, {"R": [2]}, min_fa=0.71)["tract_R_voxels"] == 0


def test_a_tract_with_too_few_voxels_gets_no_value_rather_than_a_noisy_one():
    labels, fa, md, mask = volume()
    got = tract_means(labels, fa, md, mask, {"R": [2], "absent": [9]}, min_voxels=50)
    assert np.isnan(got["tract_R_fa"]) and np.isnan(got["tract_R_md"]) and got["tract_R_voxels"] == 36
    assert got["tract_absent_voxels"] == 0 and np.isnan(got["tract_absent_kept_pct"])
    assert np.isnan(got["tract_absent_fa"])


# ---- the real registration, on a made-up template ----------------------------------------------

def blob_template():
    """An FA-like image: a bright ring and two bars, so that a wrong pose scores badly."""
    shape = (64, 76, 60)
    x, y, z = np.meshgrid(*(np.arange(s) - (s - 1) / 2 for s in shape), indexing="ij")
    r = np.sqrt((x / 24) ** 2 + (y / 30) ** 2 + (z / 22) ** 2)
    fa = np.where(r < 1, 0.15, 0.0)
    fa[(r > 0.55) & (r < 0.8)] = 0.45
    fa[(np.abs(x + 9) < 3) & (np.abs(y) < 16) & (np.abs(z) < 6)] = 0.7     # left bar
    fa[(np.abs(x - 9) < 3) & (np.abs(y - 4) < 12) & (np.abs(z) < 6)] = 0.6  # right bar
    labels = np.zeros(shape, np.int16)
    labels[(np.abs(x + 9) < 3) & (np.abs(y) < 16) & (np.abs(z) < 6)] = 1
    labels[(np.abs(x - 9) < 3) & (np.abs(y - 4) < 12) & (np.abs(z) < 6)] = 2
    return ndimage.gaussian_filter(fa, 0.8).astype(np.float32), labels


@pytest.mark.slow
def test_registration_puts_the_tracts_on_a_moved_and_coarser_subject(tmp_path):
    cfg = from_dict({"tracts": {"primary": {"ATR": ["Anterior thalamic radiation L",
                                                    "Anterior thalamic radiation R"]},
                                "secondary": {"ATR_L": ["Anterior thalamic radiation L"],
                                              "ATR_R": ["Anterior thalamic radiation R"]},
                                "min_fa": 0.2}}, base=tmp_path)
    fa, labels = blob_template()
    space = np.diag([2.0, 2.0, 2.0, 1.0])
    jhu = cfg.paths.cache_dir / "jhu"
    jhu.mkdir(parents=True)
    nib.save(nib.Nifti1Image(fa, space), str(jhu / "JHU-ICBM-FA-1mm.nii.gz"))
    nib.save(nib.Nifti1Image(labels, space), str(jhu / "JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz"))
    (jhu / "JHU-tracts.xml").write_text(XML, encoding="iso-8859-1")

    # the subject: the same head on a coarser grid, turned a little and shifted
    a = np.deg2rad(5)
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    pose[:3, 3] = [6.0, -4.0, 3.0]
    coarse = resample_to_output(nib.Nifti1Image(fa, space), voxel_sizes=2.5, order=1)
    sub = dwi.subject_dir(cfg, "s1")
    sub.mkdir(parents=True)
    native_fa = np.asarray(coarse.dataobj, dtype=np.float32)
    affine = pose @ coarse.affine
    nib.save(nib.Nifti1Image(native_fa, affine), str(sub / dwi.FA_FILE))
    nib.save(nib.Nifti1Image((native_fa * 1.2).astype(np.float32), affine), str(sub / dwi.MD_FILE))
    nib.save(nib.Nifti1Image((native_fa > 0.05).astype(np.uint8), affine), str(sub / dwi.MASK_FILE))

    reg = register.register_subject("s1", cfg)
    assert reg["reg_ok"] and reg["reg_similarity"] > 0.9 and reg["registered_now"]
    assert register.register_subject("s1", cfg)["registered_now"] is False       # cached
    got = tract_subject("s1", cfg)
    # each bar keeps the FA it has in the template (its edges are blurred there too), so left
    # and right were not swapped and each tract sits on its own bar
    assert got["tract_ATR_L_fa"] == pytest.approx(fa[labels == 1].mean(), abs=0.04)
    assert got["tract_ATR_R_fa"] == pytest.approx(fa[labels == 2].mean(), abs=0.04)
    assert got["tract_ATR_L_fa"] - got["tract_ATR_R_fa"] > 0.04
    assert got["tract_ATR_md"] == pytest.approx(got["tract_ATR_fa"] * 1.2, rel=1e-3)
    assert got["tract_ATR_kept_pct"] > 90
    warped = nib.load(str(sub / TRACTS_FILE))
    assert warped.shape == native_fa.shape and np.allclose(warped.affine, affine, atol=1e-4)
    assert json.loads((sub / "tracts.json").read_text())["tract_ATR_voxels"] == got["tract_ATR_voxels"]


@pytest.mark.slow
def test_a_registration_that_cannot_be_right_fails_the_check(tmp_path):
    cfg = from_dict({}, base=tmp_path)
    fa, _ = blob_template()
    jhu = cfg.paths.cache_dir / "jhu"
    jhu.mkdir(parents=True)
    nib.save(nib.Nifti1Image(fa, np.diag([2.0, 2.0, 2.0, 1.0])), str(jhu / "JHU-ICBM-FA-1mm.nii.gz"))
    sub = dwi.subject_dir(cfg, "noise")
    sub.mkdir(parents=True)
    noise = np.random.default_rng(0).random((40, 40, 40)).astype(np.float32)
    nib.save(nib.Nifti1Image(noise, np.diag([3.0, 3.0, 3.0, 1.0])), str(sub / dwi.FA_FILE))
    reg = register.register_subject("noise", cfg)
    assert reg["reg_similarity"] < register.MIN_SIMILARITY and reg["reg_ok"] is False


def test_similarity_is_a_correlation_inside_the_mask():
    a = np.arange(27, dtype=float).reshape(3, 3, 3)
    mask = np.ones((3, 3, 3), bool)
    assert register.similarity(a, 2 * a + 1, mask) == pytest.approx(1.0)
    assert register.similarity(a, -a, mask) == pytest.approx(-1.0)
    assert register.similarity(np.zeros((3, 3, 3)), a, mask) == 0.0
