"""Mean FA and MD inside each white-matter tract.

The tract atlas is brought into the subject's own space, so the means are taken
over the subject's own voxels and the FA and MD maps are never resampled.
"""
from __future__ import annotations

import json

import numpy as np

from dtic import atlas, dwi, register
from dtic.config import Config
from dtic.errors import PipelineError

TRACTS_FILE = "tracts.nii.gz"    # atlas labels on the diffusion grid
TRACTS_JSON = "tracts.json"


def tract_means(labels: np.ndarray, fa: np.ndarray, md: np.ndarray, mask: np.ndarray,
                groups: dict[str, list[int]], min_fa: float = 0.0, min_voxels: int = 1) -> dict:
    """For each group of atlas labels: mean FA, mean MD, and how many voxels they rest on.

    A group that names a left and a right tract is one pool of voxels, so the mean
    is over both sides together. Only voxels inside the brain mask with FA of at
    least min_fa count. kept_pct is the share of the tract's voxels that passed,
    and a group left with fewer than min_voxels gets no mean.
    """
    mask = mask.astype(bool)
    out = {}
    for name, values in groups.items():
        tract = np.isin(labels, values) & mask
        kept = tract & (fa >= min_fa)
        n_tract, n_kept = int(tract.sum()), int(kept.sum())
        enough = n_kept >= min_voxels
        out[f"tract_{name}_fa"] = float(fa[kept].mean()) if enough else np.nan
        out[f"tract_{name}_md"] = float(md[kept].mean()) if enough else np.nan
        out[f"tract_{name}_voxels"] = n_kept
        out[f"tract_{name}_kept_pct"] = 100 * n_kept / n_tract if n_tract else np.nan
    return out


def tract_subject(sid: str, cfg: Config) -> dict:
    """Warp the atlas to one subject and take the tract means."""
    import nibabel as nib

    folder = dwi.subject_dir(cfg, sid)
    paths = atlas.jhu_paths(cfg)
    names = atlas.read_tract_names(paths["xml"])
    groups = atlas.tract_values({**cfg.tracts.primary, **cfg.tracts.secondary}, names)

    # the warped atlas only changes when the registration does
    native = folder / TRACTS_FILE
    affine = register.reg_dir(cfg, sid) / register.AFFINE
    if not native.is_file() or native.stat().st_mtime < affine.stat().st_mtime:
        register.to_native(sid, paths["labels"], native, cfg)
    labels_img = nib.load(str(native))
    fa_img = nib.load(str(folder / dwi.FA_FILE))
    if labels_img.shape != fa_img.shape or not np.allclose(
            labels_img.affine, fa_img.affine, atol=1e-3):
        raise PipelineError(f"{sid}: the warped atlas and the FA map are not on the same grid")
    labels = np.asarray(labels_img.dataobj).astype(np.int16)
    fa = np.asarray(fa_img.dataobj, dtype=np.float32)
    md = np.asarray(nib.load(str(folder / dwi.MD_FILE)).dataobj, dtype=np.float32)
    mask = np.asarray(nib.load(str(folder / dwi.MASK_FILE)).dataobj) > 0
    out = {"subject_id": sid}
    out.update(tract_means(labels, fa, md, mask, groups, cfg.tracts.min_fa, cfg.tracts.min_voxels))
    (folder / TRACTS_JSON).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
