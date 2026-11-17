import gzip
import json
import os

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from dtic import dwi, tracts, viewer_export
from dtic.config import from_dict
from dtic.errors import PipelineError

XML = """<?xml version="1.0" encoding="ISO-8859-1"?>
<atlas version="1.0"><header><name>JHU White-Matter Tractography Atlas</name></header><data>
<label index="0">Anterior thalamic radiation L</label>
<label index="1">Anterior thalamic radiation R</label>
<label index="14">Superior longitudinal fasciculus L</label>
<label index="15">Superior longitudinal fasciculus R</label>
<label index="18">Superior longitudinal fasciculus (temporal part) L</label>
<label index="19">Superior longitudinal fasciculus (temporal part) R</label>
</data></atlas>"""

# 2 x 2 x 2.5 mm voxels, first axis running right to left as scanners often store it
AFFINE = np.array([[-2.0, 0, 0, 40], [0, 2.0, 0, -50], [0, 0, 2.5, -30], [0, 0, 0, 1]])
SHAPE = (40, 50, 24)


def write_subject(cfg, sid):
    """A box of brain with the left ATR, both SLF and their FA and MD maps."""
    sub = dwi.subject_dir(cfg, sid)
    sub.mkdir(parents=True)
    mask = np.zeros(SHAPE, np.uint8)
    mask[6:34, 8:42, 4:20] = 1
    labels = np.zeros(SHAPE, np.int16)
    labels[24:28, 20:30, 8:14] = 1      # ATR L. high first index is the subject's left here
    labels[30:33, 15:35, 10:16] = 15    # SLF L
    labels[7:10, 15:35, 10:16] = 16     # SLF R
    fa = np.where(mask, 0.3, 0).astype(np.float32)
    fa[labels > 0] = 0.6
    md = np.where(mask, 0.9, 0).astype(np.float32)
    md[12:16, 20:26, 8:12] = 3.5        # fluid, above what the viewer draws as white
    for name, data in ((dwi.FA_FILE, fa), (dwi.MD_FILE, md), (dwi.MASK_FILE, mask),
                       (tracts.TRACTS_FILE, labels)):
        nib.save(nib.Nifti1Image(data, AFFINE), str(sub / name))
    return sub


@pytest.fixture
def exported(tmp_path):
    cfg = from_dict({}, base=tmp_path)
    jhu = cfg.paths.cache_dir / "jhu"
    jhu.mkdir(parents=True)
    (jhu / "JHU-tracts.xml").write_text(XML, encoding="iso-8859-1")
    for name in ("JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz", "JHU-ICBM-FA-1mm.nii.gz"):
        (jhu / name).write_bytes(b"x")
    write_subject(cfg, "s1")
    row = {"subject_id": "s1", "tract_ATR_fa": 0.41234567, "tract_ATR_md": 0.78, "tract_ATR_voxels": 480,
           "tract_ATR_kept_pct": 92.5, "tract_SLF_fa": 0.44, "tract_SLF_md": float("nan"),
           "tract_SLF_voxels": 900, "tract_SLF_kept_pct": 88.0, "tract_ATR_L_fa": 0.4, "tract_ATR_L_md": 0.8,
           "n_directions": 64, "b_value": 1000.0, "n_b0": 5, "voxel_mm": "2 x 2 x 2.5",
           "acq_as_expected": True, "motion_mean_mm": 0.61, "motion_max_mm": 1.4,
           "reg_similarity": 0.81, "reg_ok": True}
    out = tmp_path / "viewer_data"
    folder = viewer_export.export_subject("s1", cfg, out, row)
    return cfg, out, folder, json.loads((folder / "meta.json").read_text())


def read(folder, meta, layer):
    raw = gzip.decompress((folder / meta["layers"][layer]).read_bytes())
    nx, ny, nz = meta["dims"]
    return np.frombuffer(raw, np.uint8).reshape(nz, ny, nx).transpose(2, 1, 0)


def test_volumes_are_one_mm_cropped_to_the_brain_and_the_right_size(exported):
    _, _, folder, meta = exported
    assert meta["voxel_mm"] == 1.0 and meta["orientation"] == "RAS"
    assert set(meta["layers"]) == {"fa", "md", "tracts"}
    # the brain box is 56 x 68 x 40 mm, plus a margin of 4 on each side
    assert meta["dims"] == pytest.approx([64, 76, 48], abs=3)
    for layer in meta["layers"]:
        assert read(folder, meta, layer).shape == tuple(meta["dims"])


def test_maps_are_stored_on_a_fixed_scale_the_viewer_can_turn_back(exported):
    _, _, folder, meta = exported
    assert meta["white"] == {"fa": 1.0, "md": 3.0}
    fa, md, codes = (read(folder, meta, layer) for layer in ("fa", "md", "tracts"))
    assert fa.max() == round(0.6 * 255)                         # byte / 255 is the FA
    inside = fa[fa > 0]
    assert np.median(inside) == round(0.3 * 255)
    assert md.max() == 255                                      # fluid above 3 is clipped to white
    assert np.median(md[md > 0]) == round(0.9 / 3 * 255)
    assert np.median(fa[codes > 0]) == round(0.6 * 255)         # the tracts lie on the bright FA


def test_each_tested_tract_gets_one_code_and_lands_on_the_correct_side(exported):
    _, _, folder, meta = exported
    codes = read(folder, meta, "tracts")
    assert meta["tracts"] == {"1": "ATR", "2": "SLF"}
    assert set(np.unique(codes)) == {0, 1, 2}
    mid = meta["dims"][0] / 2
    # in RAS the first axis runs towards the subject's right, so left is the low half
    atr = np.argwhere(codes == 1)
    assert (atr[:, 0] < mid).all()
    slf = np.argwhere(codes == 2)
    assert (slf[:, 0] < mid).any() and (slf[:, 0] > mid).any()   # both sides share the SLF code
    # 4 x 10 x 6 voxels of 2 x 2 x 2.5 mm is 2400 mm3, so 2400 voxels of 1 mm
    assert (codes == 1).sum() == pytest.approx(2400, rel=0.1)


def test_jump_points_are_inside_their_tract_one_per_side(exported):
    _, _, folder, meta = exported
    codes = read(folder, meta, "tracts")
    places = meta["places"]
    assert [(p["tract"], p["side"]) for p in places] == [("ATR", "left"), ("SLF", "left"), ("SLF", "right")]
    for place in places:
        assert codes[tuple(place["voxel"])] == {"ATR": 1, "SLF": 2}[place["tract"]]
    mid = meta["dims"][0] / 2
    assert places[1]["voxel"][0] < mid < places[2]["voxel"][0]


def test_the_jump_point_of_a_curved_tract_is_inside_it():
    labels = np.zeros((12, 12, 3), np.int16)
    labels[2:10, 2:10, 1] = 7
    labels[4:10, 4:10, 1] = 0          # an L shape whose centre of mass is in the hollow
    place = viewer_export.tract_places(labels, {"T": {"left": 7, "right": 8}})
    assert len(place) == 1 and place[0]["side"] == "left"      # the absent side is left out
    assert labels[tuple(place[0]["voxel"])] == 7


def test_meta_carries_the_numbers_and_leaves_missing_ones_empty(exported):
    _, _, _, meta = exported
    stats = meta["stats"]
    assert stats["tracts"]["ATR"] == {"fa": 0.4123, "md": 0.78, "voxels": 480.0, "kept_pct": 92.5}
    assert stats["tracts"]["SLF"]["md"] is None                 # not a number, so not a number in json
    assert stats["parts"]["ATR_L"]["fa"] == 0.4 and stats["parts"]["SLF_temporal"]["fa"] is None
    assert (stats["n_directions"], stats["b_value"], stats["n_b0"]) == (64, 1000, 5)
    assert stats["voxel_mm"] == "2 x 2 x 2.5" and stats["motion_mean_mm"] == 0.61
    assert stats["reg_ok"] is True and stats["acq_as_expected"] is True
    assert meta["min_fa"] == 0.2 and meta["attribution"] == ""
    assert "cohort" not in json.dumps(meta) and "age" not in stats    # no clinical values in the viewer files


def test_public_dev_scans_carry_their_credit(exported):
    cfg, out, _, _ = exported
    write_subject(cfg, "sub-10159")
    folder = viewer_export.export_subject("sub-10159", cfg, out)
    meta = json.loads((folder / "meta.json").read_text())
    assert "OpenNeuro ds000030" in meta["attribution"] and "CC0" in meta["attribution"]
    assert meta["stats"]["tracts"]["ATR"]["fa"] is None         # no table yet, no numbers
    given = viewer_export.export_subject("sub-10159", cfg, out, attribution="Shown with permission.")
    assert json.loads((given / "meta.json").read_text())["attribution"] == "Shown with permission."


def test_index_lists_what_is_there_and_unprocessed_subjects_are_refused(exported):
    cfg, out, _, _ = exported
    assert viewer_export.write_index(out) == ["s1"]
    index = json.loads((out / "index.json").read_text())
    assert index["measurements"][0]["id"] == "s1"
    assert index["measurements"][0]["tracts"]["ATR"]["fa"] == 0.4123
    with pytest.raises(PipelineError, match="not processed yet"):
        viewer_export.export_subject("s2", cfg, out)
    with pytest.raises(PipelineError, match="no scan for: ghost"):
        viewer_export.export(cfg, ["s1"], out, subjects=["ghost"])


def test_export_skips_what_is_up_to_date_and_copies_the_group_summary(exported):
    cfg, out, folder, _ = exported
    cfg.paths.local_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"subject_id": "s1", "tract_ATR_fa": 0.5, "reg_ok": True}]).to_csv(
        cfg.paths.local_dir / "imaging.csv", index=False)
    logged = []
    assert viewer_export.export(cfg, ["s1", "s2"], out, log=logged.append) == ["s1"]
    assert logged == ["viewer    s1"]                            # the table is newer, so it is redone
    assert json.loads((folder / "meta.json").read_text())["stats"]["tracts"]["ATR"]["fa"] == 0.5
    logged.clear()
    viewer_export.export(cfg, ["s1", "s2"], out, log=logged.append)
    assert logged == []                                          # nothing changed since
    later = (folder / "meta.json").stat().st_mtime + 10
    os.utime(dwi.subject_dir(cfg, "s1") / dwi.FA_FILE, (later, later))
    viewer_export.export(cfg, ["s1"], out, log=logged.append)
    assert logged == ["viewer    s1"]
    assert not (out / "summary.json").exists()
    cfg.paths.shareable_dir.mkdir(parents=True)
    (cfg.paths.shareable_dir / "group_summary.json").write_text('{"fake_data": true, "tests": []}')
    (cfg.paths.shareable_dir / "group_summary.md").write_text("# Group summary")
    viewer_export.export(cfg, ["s1"], out, log=logged.append)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["fake_data"] is True and summary["markdown"] == "# Group summary"


def test_a_scan_from_an_earlier_run_is_not_left_in_the_viewer(exported):
    cfg, out, _, _ = exported
    write_subject(cfg, "old-01")
    assert viewer_export.export(cfg, ["s1", "old-01"], out, log=lambda m: None) == ["old-01", "s1"]
    # the scans folder now holds other people: the earlier one must go from the menu and from disk
    assert viewer_export.export(cfg, ["s1"], out, log=lambda m: None) == ["s1"]
    assert not (out / "old-01").exists()
