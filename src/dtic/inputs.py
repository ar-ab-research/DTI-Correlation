"""Find the diffusion scans and line them up with the tables."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from dtic.errors import PipelineError


class InputError(PipelineError):
    """The scans folder cannot be read unambiguously, or a scan is unusable."""


_NIFTI = re.compile(r"\.nii(\.gz)?$", re.IGNORECASE)
_DWI_TAG = re.compile(r"(^|[_-])(dwi|dti)$", re.IGNORECASE)
_SIDECARS = {"bval": (".bval", ".bvals"), "bvec": (".bvec", ".bvecs")}


@dataclass(frozen=True)
class Scan:
    """One diffusion scan: the 4D image and the two text files that describe it."""
    image: Path
    bval: Path
    bvec: Path

    def newest(self) -> float:
        return max(p.stat().st_mtime for p in (self.image, self.bval, self.bvec))


def _stem(path: Path) -> str:
    return _NIFTI.sub("", path.name)


def _is_nifti(path: Path) -> bool:
    return path.is_file() and bool(_NIFTI.search(path.name))


def subject_id(stem: str) -> str:
    """sub-01_dwi and sub-01 are the same subject."""
    return re.sub(r"[_-](dwi|dti)$", "", stem, flags=re.IGNORECASE) or stem


def _sidecar(image: Path, kind: str) -> Path | None:
    for suffix in _SIDECARS[kind]:
        candidate = image.with_name(_stem(image) + suffix)
        if candidate.is_file():
            return candidate
    return None


def _scan(image: Path) -> Scan:
    bval, bvec = _sidecar(image, "bval"), _sidecar(image, "bvec")
    missing = [f"{_stem(image)}.{kind}" for kind, found in (("bval", bval), ("bvec", bvec)) if not found]
    if missing:
        raise InputError(
            f"{image}: no {' or '.join(missing)} beside it. a diffusion scan needs its "
            "b-values and b-vectors, which dcm2niix writes next to the image"
        )
    return Scan(image, bval, bvec)


def discover_dwi(dwi_dir: Path) -> dict[str, Scan]:
    """Subject ID to scan, sorted by ID.

    Two layouts are read, and may be mixed:
      <id>.nii.gz or <id>_dwi.nii.gz      directly in the folder
      <id>/dwi.nii.gz                      one folder per subject, searched to any depth
    Each image needs a .bval and a .bvec file with the same name beside it.

    A subject folder may hold other images. The scan is the image named dwi (or ending
    in _dwi); with no such file, a folder holding exactly one image uses that one.
    """
    dwi_dir = Path(dwi_dir)
    if not dwi_dir.is_dir():
        raise InputError(f"not a folder: {dwi_dir}")
    found: dict[str, Scan] = {}

    def add(sid: str, image: Path) -> None:
        if sid in found:
            raise InputError(f"two scans for subject {sid}: {found[sid].image} and {image}")
        found[sid] = _scan(image)

    for entry in sorted(dwi_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if _is_nifti(entry):
            add(subject_id(_stem(entry)), entry)
        elif entry.is_dir():
            images = sorted(p for p in entry.rglob("*") if _is_nifti(p))
            if not images:
                continue
            tagged = [p for p in images if _DWI_TAG.search(_stem(p))]
            if len(tagged) == 1:
                add(entry.name, tagged[0])
            elif not tagged and len(images) == 1:
                add(entry.name, images[0])
            else:
                names = ", ".join(p.name for p in (tagged or images))
                raise InputError(
                    f"cannot tell which image in {entry} is the diffusion scan ({names}). "
                    "name it dwi.nii.gz"
                )
    return dict(sorted(found.items()))


def check_dwi(path: Path) -> dict:
    """Header checks. Returns shape and voxel size, or raises InputError."""
    import nibabel as nib
    import numpy as np

    try:
        img = nib.load(str(path))
    except Exception as exc:  # noqa: BLE001  nibabel raises several unrelated types
        raise InputError(f"{path}: not a readable NIfTI ({exc})") from exc
    if img.ndim != 4 or img.shape[3] < 7:
        raise InputError(
            f"{path}: expected a 4D diffusion series, shape is {tuple(img.shape)}. "
            "a T1 or any other single volume cannot be used here"
        )
    affine = np.asarray(img.affine, dtype=float)
    det = float(np.linalg.det(affine[:3, :3]))
    if not np.isfinite(affine).all() or abs(det) < 1e-6:
        raise InputError(f"{path}: the affine is not usable")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if min(zooms) <= 0 or max(zooms) > 10:
        raise InputError(f"{path}: voxel size {zooms} mm is not plausible")
    return {"shape": tuple(int(s) for s in img.shape), "voxel_mm": zooms,
            "voxel_volume_mm3": abs(det)}


def match_ids(scans: set[str], clinical: set[str], fc: set[str]) -> dict[str, list[str]]:
    """Who is where. Every scan is processed; the tables decide who enters which test."""
    return {
        "scan_without_clinical": sorted(scans - clinical),
        "scan_without_fc": sorted(scans - fc),
        "clinical_without_scan": sorted(clinical - scans),
        "fc_without_scan": sorted(fc - scans),
        "complete": sorted(scans & clinical & fc),
    }
