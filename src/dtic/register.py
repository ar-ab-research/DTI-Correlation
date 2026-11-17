"""Registration of each subject's FA map to the FA template, with ANTs.

The transform is used one way only: to bring the tract atlas, which lives in the
template's space, onto the subject's own diffusion grid.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dtic import atlas, dwi
from dtic.config import Config
from dtic.errors import PipelineError

REG_DIR = "reg"
PREFIX = "fa_to_template_"
REGISTER_JSON = "register.json"
AFFINE = PREFIX + "0GenericAffine.mat"
WARP = PREFIX + "1Warp.nii.gz"
INVERSE_WARP = PREFIX + "1InverseWarp.nii.gz"

TRANSFORMS = {"fast": "antsRegistrationSyNQuick[s]", "accurate": "antsRegistrationSyN[s]"}

# correlation between the warped FA map and the template inside the template's brain.
# good registrations of an FA map to an FA template sit well above this
MIN_SIMILARITY = 0.6


class RegistrationError(PipelineError):
    """ANTs failed, or produced a transform that cannot be right."""


def reg_dir(cfg: Config, sid: str) -> Path:
    return dwi.subject_dir(cfg, sid) / REG_DIR


def _settings(cfg: Config) -> dict:
    r = cfg.registration
    return {"template": r.template, "speed": r.speed}


def _cached(folder: Path, fa: Path, cfg: Config) -> dict | None:
    record = folder / REGISTER_JSON
    files = [folder / AFFINE, folder / WARP, folder / INVERSE_WARP, record]
    if not all(f.is_file() for f in files):
        return None
    if record.stat().st_mtime < fa.stat().st_mtime:
        return None
    saved = json.loads(record.read_text(encoding="utf-8"))
    return saved if saved.get("settings") == _settings(cfg) else None


def similarity(warped: np.ndarray, template: np.ndarray, mask: np.ndarray) -> float:
    a, b = warped[mask], template[mask]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def register_subject(sid: str, cfg: Config, force: bool = False) -> dict:
    """Register one subject's FA map to the template. Returns a QC number."""
    folder = reg_dir(cfg, sid)
    fa = dwi.subject_dir(cfg, sid) / dwi.FA_FILE
    if not force:
        saved = _cached(folder, fa, cfg)
        if saved is not None:
            saved["registered_now"] = False
            return saved

    import ants

    folder.mkdir(parents=True, exist_ok=True)
    fixed = ants.image_read(str(atlas.jhu_paths(cfg)["fa"]))
    moving = ants.image_read(str(fa))
    try:
        reg = ants.registration(
            fixed=fixed, moving=moving, outprefix=str(folder / PREFIX),
            type_of_transform=TRANSFORMS[cfg.registration.speed],
        )
    except Exception as exc:  # noqa: BLE001  ants raises plain exceptions of several types
        raise RegistrationError(f"{sid}: registration failed ({exc})") from exc
    if not all((folder / f).is_file() for f in (AFFINE, WARP, INVERSE_WARP)):
        raise RegistrationError(f"{sid}: registration did not write its transforms")

    template = fixed.numpy()
    result = {
        "subject_id": sid,
        "reg_similarity": similarity(reg["warpedmovout"].numpy(), template, template > 0),
        "settings": _settings(cfg),
    }
    result["reg_ok"] = bool(result["reg_similarity"] >= MIN_SIMILARITY)
    (folder / REGISTER_JSON).write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["registered_now"] = True
    return result


def to_native(sid: str, moving_path: Path, out_path: Path, cfg: Config,
              interpolator: str = "genericLabel") -> Path:
    """Bring a template-space image onto a subject's diffusion grid."""
    import ants

    folder = reg_dir(cfg, sid)
    fa = dwi.subject_dir(cfg, sid) / dwi.FA_FILE
    warped = ants.apply_transforms(
        fixed=ants.image_read(str(fa)), moving=ants.image_read(str(moving_path)),
        transformlist=[str(folder / AFFINE), str(folder / INVERSE_WARP)],
        whichtoinvert=[True, False], interpolator=interpolator,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(warped, str(out_path))
    return out_path
