"""Diffusion preprocessing with DIPY: read the scan, correct it for motion, fit the tensor.

Each volume is registered to the b0 with an affine transform, which takes out head
motion and the linear part of eddy-current distortion, and the b-vectors are rotated
to match. There is no model-based eddy correction and no susceptibility correction:
DIPY has neither, and the protocol has no reversed phase-encode scan to drive one.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from dtic.config import Config
from dtic.errors import PipelineError
from dtic.inputs import Scan

FA_FILE = "fa.nii.gz"
MD_FILE = "md.nii.gz"
MASK_FILE = "mask.nii.gz"
DWI_JSON = "dwi.json"

# mean diffusivity is written in units of 10^-3 mm2/s, where white matter sits near 0.8
MD_SCALE = 1000.0
# radius of the sphere a transform's displacement is averaged over, as in FSL's rmsdiff
HEAD_RADIUS_MM = 80.0


class DwiError(PipelineError):
    """A diffusion scan cannot be read or does not hold what a tensor fit needs."""


def subject_dir(cfg: Config, sid: str) -> Path:
    """Where one subject's derived files live. Under outputs/local, so never shared."""
    return cfg.paths.local_dir / "subjects" / sid


def read_gradients(scan: Scan, n_volumes: int) -> tuple[np.ndarray, np.ndarray]:
    """b-values (n,) and b-vectors (n, 3) from the two text files."""
    try:
        bvals = np.atleast_1d(np.loadtxt(scan.bval, dtype=float)).ravel()
        bvecs = np.atleast_2d(np.loadtxt(scan.bvec, dtype=float))
    except (OSError, ValueError) as exc:
        raise DwiError(f"{scan.bval.parent}: b-values or b-vectors cannot be read ({exc})") from exc
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    if bvecs.ndim != 2 or bvecs.shape[1] != 3:
        raise DwiError(f"{scan.bvec}: expected three rows or three columns, got shape {bvecs.shape}")
    if len(bvals) != n_volumes or len(bvecs) != n_volumes:
        raise DwiError(
            f"{scan.image}: {n_volumes} volumes, but {len(bvals)} b-values and {len(bvecs)} b-vectors"
        )
    if not (np.isfinite(bvals).all() and np.isfinite(bvecs).all()):
        raise DwiError(f"{scan.bval.parent}: b-values or b-vectors hold a value that is not a number")
    return bvals, bvecs


def describe_acquisition(bvals: np.ndarray, cfg: Config) -> dict:
    """Which volumes are b0, which form the shell, and whether that is what was expected."""
    acq = cfg.acquisition
    b0 = bvals <= acq.b0_threshold
    shell = np.abs(bvals - acq.b_value) <= acq.b_tolerance
    found = ", ".join(f"{int(b)} x{int(n)}" for b, n in
                      zip(*np.unique(np.round(bvals / 50) * 50, return_counts=True)))
    if not b0.any():
        raise DwiError(f"no b0 volume (b-values found: {found})")
    if shell.sum() < 6:
        raise DwiError(
            f"{int(shell.sum())} volumes near b={acq.b_value:g}, a tensor needs at least 6 "
            f"(b-values found: {found})"
        )
    other = int((~(b0 | shell)).sum())
    return {
        "n_volumes": int(len(bvals)),
        "n_b0": int(b0.sum()),
        "n_directions": int(shell.sum()),
        "b_value": float(np.round(np.median(bvals[shell]))),
        # volumes on other shells are left out of the fit
        "n_other_volumes": other,
        "acq_as_expected": bool(b0.sum() == acq.b0_volumes and shell.sum() == acq.directions
                                and other == 0),
        "keep": b0 | shell,
    }


def displacement_mm(affines: np.ndarray, centre: np.ndarray, b0s_mask: np.ndarray) -> np.ndarray:
    """How far each volume sits from the scan's usual head position, in mm.

    Root mean square over a head-sized sphere, the closed form of Jenkinson (1999),
    applied to the turn and the shift in each volume's transform.

    Two things in a transform are not movement and are kept out. Stretch and shear
    hold eddy distortion. And registering a diffusion-weighted volume to a b0 gives
    every such volume much the same offset, because the two differ in contrast: that
    shared part is removed by measuring each volume against the median transform of
    its own kind (b0, or diffusion-weighted).
    """
    out = np.empty(affines.shape[-1])
    b0s_mask = np.asarray(b0s_mask, dtype=bool)
    for kind in (b0s_mask, ~b0s_mask):
        if not kind.any():
            continue
        usual = np.linalg.inv(np.median(affines[..., kind], axis=-1))
        for k in np.flatnonzero(kind):
            relative = affines[..., k] @ usual
            linear = relative[:3, :3]
            u, _, vt = np.linalg.svd(linear)
            turn = u @ vt - np.eye(3)
            shift = linear @ centre + relative[:3, 3] - centre
            out[k] = np.sqrt(HEAD_RADIUS_MM ** 2 / 5 * np.trace(turn.T @ turn) + shift @ shift)
    return out


def correct_motion(data: np.ndarray, gtab, affine: np.ndarray, transform: str):
    """Register every volume to the b0. Returns (data, gradient table, one 4x4 per volume)."""
    from dipy.align import motion_correction
    from dipy.core.gradients import reorient_bvecs

    # dipy reports every registration step, with characters a windows console cannot print
    logging.getLogger("dipy").setLevel(logging.ERROR)
    pipeline =["center_of_mass", "translation", "rigid"]
    if transform == "affine":
        pipeline.append("affine")
    try:
        img, transforms = motion_correction(data, gtab, affine=affine, pipeline=pipeline)
    except Exception as exc:  # noqa: BLE001  the optimiser raises several unrelated types
        raise DwiError(f"motion correction failed ({exc})") from exc
    transforms = np.asarray(transforms)
    # a head turned by R after the b0 was measured along R^T g in its own frame, and that
    # is what reorient_bvecs returns for dipy's own transforms, passed as they come.
    # tests/test_dwi.py holds this to a known rotation
    rotated = reorient_bvecs(gtab, [transforms[..., k] for k in np.flatnonzero(~gtab.b0s_mask)])
    return np.asarray(img.dataobj), rotated, transforms


def brain_mask(b0: np.ndarray) -> np.ndarray:
    from dipy.segment.mask import median_otsu

    _, mask = median_otsu(b0, median_radius=2, numpass=1)
    return mask


def fit_tensor(data: np.ndarray, gtab, mask: np.ndarray, method: str = "WLS"):
    """FA (0 to 1) and MD (10^-3 mm2/s) inside the mask, zero outside."""
    from dipy.reconst.dti import TensorModel

    fit = TensorModel(gtab, fit_method=method).fit(data, mask=mask)
    fa = np.clip(np.nan_to_num(fit.fa, nan=0.0, posinf=0.0, neginf=0.0), 0, 1)
    md = np.clip(np.nan_to_num(fit.md, nan=0.0, posinf=0.0, neginf=0.0), 0, None) * MD_SCALE
    fa[~mask] = 0
    md[~mask] = 0
    return fa.astype(np.float32), md.astype(np.float32)


def _settings(cfg: Config) -> dict:
    acq, pre = cfg.acquisition, cfg.preprocessing
    return {"b_value": acq.b_value, "b_tolerance": acq.b_tolerance, "b0_threshold": acq.b0_threshold,
            "directions": acq.directions, "b0_volumes": acq.b0_volumes,
            "motion_correction": pre.motion_correction, "transform": pre.transform,
            "fit_method": pre.fit_method}


def _cached(folder: Path, scan: Scan, cfg: Config) -> dict | None:
    record = folder / DWI_JSON
    files = [folder / FA_FILE, folder / MD_FILE, folder / MASK_FILE, record]
    if not all(f.is_file() for f in files):
        return None
    if record.stat().st_mtime < scan.newest():
        return None
    saved = json.loads(record.read_text(encoding="utf-8"))
    return saved if saved.get("settings") == _settings(cfg) else None


def preprocess_subject(sid: str, scan: Scan, cfg: Config, force: bool = False) -> dict:
    """Motion-correct one scan and fit the tensor. Writes the FA and MD maps, returns the numbers."""
    import nibabel as nib
    from dipy.core.gradients import gradient_table

    folder = subject_dir(cfg, sid)
    if not force:
        saved = _cached(folder, scan, cfg)
        if saved is not None:
            saved["processed_now"] = False
            return saved

    img = nib.load(str(scan.image))
    bvals, bvecs = read_gradients(scan, img.shape[3])
    try:
        acq = describe_acquisition(bvals, cfg)
    except DwiError as exc:
        raise DwiError(f"{sid}: {exc}") from exc
    keep = acq.pop("keep")
    data = np.asarray(img.dataobj, dtype=np.float32)[..., keep]
    try:
        gtab = gradient_table(bvals[keep], bvecs=bvecs[keep], b0_threshold=cfg.acquisition.b0_threshold)
    except ValueError as exc:
        raise DwiError(f"{sid}: the b-vectors are not usable ({exc})") from exc

    motion = {"motion_mean_mm": None, "motion_max_mm": None}
    if cfg.preprocessing.motion_correction:
        try:
            data, gtab, transforms = correct_motion(data, gtab, img.affine, cfg.preprocessing.transform)
        except DwiError as exc:
            raise DwiError(f"{sid}: {exc}") from exc
        centre = nib.affines.apply_affine(img.affine, (np.array(img.shape[:3]) - 1) / 2)
        moved = displacement_mm(transforms, centre, gtab.b0s_mask)
        motion = {"motion_mean_mm": float(moved.mean()), "motion_max_mm": float(moved.max())}

    mask = brain_mask(data[..., gtab.b0s_mask].mean(axis=-1))
    if mask.sum() < 1000:
        raise DwiError(f"{sid}: no brain found in the b0 image")
    fa, md = fit_tensor(data, gtab, mask, cfg.preprocessing.fit_method)

    folder.mkdir(parents=True, exist_ok=True)
    for name, volume in ((FA_FILE, fa), (MD_FILE, md), (MASK_FILE, mask.astype(np.uint8))):
        out = nib.Nifti1Image(volume, img.affine)
        out.set_data_dtype(volume.dtype)
        nib.save(out, str(folder / name))

    voxel = [float(z) for z in img.header.get_zooms()[:3]]
    result = {
        "subject_id": sid,
        **acq,
        "voxel_mm": " x ".join(f"{z:.3g}" for z in voxel),
        **motion,
        "brain_volume_ml": float(mask.sum() * abs(np.linalg.det(img.affine[:3, :3])) / 1000),
        "settings": _settings(cfg),
    }
    (folder / DWI_JSON).write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["processed_now"] = True
    return result
