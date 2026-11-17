"""Write what the slice viewer shows: small 8-bit volumes and the numbers for each scan.

Everything written here is derived from a subject's scan. By default it goes under
outputs/local, with the other per-subject files, and is served to the browser from
there. Only the public demo scans are ever exported into viewer/public/data, the
folder the public site is built from.
"""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import numpy as np

from dtic import atlas, devdata, dwi, tracts
from dtic.config import Config
from dtic.errors import PipelineError

VOXEL_MM = 1.0          # the viewer's grid: isotropic, RAS
MARGIN_VOXELS = 4       # kept around the brain when cropping
MD_WHITE = 3.0          # the MD, in 10^-3 mm2/s, drawn as white. free water is about 3
INDEX_FILE = "index.json"
SUMMARY_FILE = "summary.json"

SCAN_NUMBERS = ("n_directions", "b_value", "n_b0", "motion_mean_mm", "motion_max_mm",
                "reg_similarity")


def default_out(cfg: Config) -> Path:
    """Beside the other per-subject outputs, which never leave the machine."""
    return cfg.paths.local_dir / "viewer"


def _resample(img, order: int):
    """Onto a 1 mm grid whose axes run left-right, back-front, bottom-top."""
    from nibabel.processing import resample_to_output

    return resample_to_output(img, voxel_sizes=VOXEL_MM, order=order, mode="constant", cval=0)


def _brain_box(mask: np.ndarray) -> tuple[slice, slice, slice]:
    """Bounding box of the brain mask, with a small margin."""
    if not mask.any():
        return tuple(slice(0, s) for s in mask.shape)
    box = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        hit = np.flatnonzero(mask.any(axis=other))
        box.append(slice(max(hit[0] - MARGIN_VOXELS, 0),
                         min(hit[-1] + 1 + MARGIN_VOXELS, mask.shape[axis])))
    return tuple(box)


def to_uint8(values: np.ndarray, white: float) -> np.ndarray:
    """0 to `white` onto 0..255, clipped. The viewer turns a byte back with byte / 255 * white."""
    return np.round(np.clip(values / white, 0, 1) * 255).astype(np.uint8)


def _write(path: Path, volume: np.ndarray) -> None:
    # x fastest, then y, then z: the order a 3D texture is uploaded in
    with gzip.open(path, "wb", compresslevel=6) as out:
        out.write(np.ascontiguousarray(volume.transpose(2, 1, 0)).tobytes())


def tract_places(labels: np.ndarray, wanted: dict[str, dict[str, int]]) -> list[dict]:
    """Where to jump to for each tract and side: the tract's own voxel nearest its centre.

    The centre of mass itself can fall outside a curved tract, so the nearest voxel
    that belongs to the tract is used. wanted maps a tract name to {side: label value}.
    """
    out = []
    for name, sides in wanted.items():
        for side, value in sides.items():
            inside = np.argwhere(labels == value)
            if not len(inside):
                continue
            nearest = inside[np.argmin(((inside - inside.mean(axis=0)) ** 2).sum(axis=1))]
            out.append({"tract": name, "side": side, "voxel": [int(v) for v in nearest]})
    return out


def _side(label_name: str) -> str:
    return {"L": "left", "R": "right"}.get(label_name.rsplit(" ", 1)[-1], "")


def export_subject(sid: str, cfg: Config, out_dir: Path, imaging_row: dict | None = None,
                   attribution: str = "") -> Path:
    """Write one subject's viewer assets. Returns the subject's folder."""
    import nibabel as nib

    sub = dwi.subject_dir(cfg, sid)
    fa_path, tract_path = sub / dwi.FA_FILE, sub / tracts.TRACTS_FILE
    if not fa_path.is_file() or not tract_path.is_file():
        raise PipelineError(f"{sid}: not processed yet, run the pipeline first")

    def volume(path: Path, order: int) -> np.ndarray:
        img = nib.load(str(path))
        data = np.asarray(img.dataobj, dtype=np.float32 if order else np.int16)
        return np.asarray(_resample(nib.Nifti1Image(data, img.affine), order=order).dataobj)

    box = _brain_box(volume(sub / dwi.MASK_FILE, order=1) > 0.5)
    fa = to_uint8(volume(fa_path, order=1)[box], 1.0)
    md = to_uint8(volume(sub / dwi.MD_FILE, order=1)[box], MD_WHITE)
    native = volume(tract_path, order=0)[box]

    # one code per tested tract, left and right alike, in the order the config lists them
    names = atlas.read_tract_names(atlas.jhu_paths(cfg)["xml"])
    coded = np.zeros(native.shape, np.uint8)
    legend, wanted = {}, {}
    for code, (name, labels) in enumerate(cfg.tracts.primary.items(), start=1):
        values = [names[label] for label in labels]
        coded[np.isin(native, values)] = code
        legend[str(code)] = name
        wanted[name] = {_side(label) or label: value for label, value in zip(labels, values)}

    folder = Path(out_dir) / sid
    folder.mkdir(parents=True, exist_ok=True)
    layers = {"fa": "fa.u8.gz", "md": "md.u8.gz", "tracts": "tracts.u8.gz"}
    for name, data in (("fa", fa), ("md", md), ("tracts", coded)):
        _write(folder / layers[name], data)

    row = imaging_row or {}

    def number(key: str):
        value = row.get(key)
        if value is None or isinstance(value, str) or not np.isfinite(value):
            return None
        return round(float(value), 4)

    def tract_numbers(name: str) -> dict:
        return {key: number(f"tract_{name}_{key}") for key in ("fa", "md", "voxels", "kept_pct")}

    def flag(key: str):
        return None if row.get(key) is None else bool(row[key])

    meta = {
        "id": sid,
        "dims": [int(s) for s in fa.shape],
        "voxel_mm": VOXEL_MM,
        "orientation": "RAS",
        "layers": layers,
        # the value a byte of 255 stands for in each map
        "white": {"fa": 1.0, "md": MD_WHITE},
        "tracts": legend,
        "places": tract_places(native, wanted),
        "min_fa": cfg.tracts.min_fa,
        "stats": {
            "tracts": {name: tract_numbers(name) for name in cfg.tracts.primary},
            "parts": {name: tract_numbers(name) for name in cfg.tracts.secondary},
            **{key: number(key) for key in SCAN_NUMBERS},
            "voxel_mm": row.get("voxel_mm") if isinstance(row.get("voxel_mm"), str) else None,
            "acq_as_expected": flag("acq_as_expected"),
            "reg_ok": flag("reg_ok"),
        },
        "attribution": attribution or devdata.attribution(sid),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return folder


def write_index(out_dir: Path) -> list[str]:
    """List every exported scan with its measurements, for the menu and the per-scan table."""
    out_dir = Path(out_dir)
    metas = sorted(out_dir.glob("*/meta.json"), key=lambda p: p.parent.name)
    rows = []
    for path in metas:
        meta = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"id": meta["id"], **meta["stats"]})
    index = {"subjects": [row["id"] for row in rows], "measurements": rows}
    (out_dir / INDEX_FILE).write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index["subjects"]


def write_summary(cfg: Config, out_dir: Path) -> bool:
    """Put the group summary where the viewer can read it. False when there is none yet."""
    source = cfg.paths.shareable_dir / "group_summary.json"
    target = Path(out_dir) / SUMMARY_FILE
    if not source.is_file():
        target.unlink(missing_ok=True)
        return False
    summary = json.loads(source.read_text(encoding="utf-8"))
    text = cfg.paths.shareable_dir / "group_summary.md"
    summary["markdown"] = text.read_text(encoding="utf-8") if text.is_file() else ""
    target.write_text(json.dumps(summary), encoding="utf-8")
    return True


def _up_to_date(folder: Path, sources: list[Path]) -> bool:
    meta = folder / "meta.json"
    if not meta.is_file():
        return False
    return all(meta.stat().st_mtime >= s.stat().st_mtime for s in sources if s.is_file())


def _processed(cfg: Config, sid: str) -> bool:
    sub = dwi.subject_dir(cfg, sid)
    return (sub / dwi.FA_FILE).is_file() and (sub / tracts.TRACTS_FILE).is_file()


def export(cfg: Config, scans: list[str], out_dir: Path, subjects=None,
           attribution: str = "", force: bool = False, log=print) -> list[str]:
    """Export the named subjects (default: all that have been processed).

    A subject whose files are newer than its maps and the subject table is left as
    it is. A folder left over from a scan that is no longer exported is removed, so
    the viewer never lists a scan from an earlier, different run. Returns the IDs
    the viewer will list.
    """
    import pandas as pd

    out_dir = Path(out_dir)
    rows = {}
    table_path = cfg.paths.local_dir / "imaging.csv"
    if table_path.is_file():
        table = pd.read_csv(table_path, dtype={"subject_id": str})
        rows = {r["subject_id"]: r for r in table.to_dict("records")}
    wanted = list(subjects) if subjects else [s for s in scans if _processed(cfg, s)]
    unknown = [s for s in wanted if s not in scans]
    if unknown:
        raise PipelineError(f"no scan for: {', '.join(unknown)}")
    for meta in out_dir.glob("*/meta.json"):
        if meta.parent.name not in wanted:
            shutil.rmtree(meta.parent)
    for sid in wanted:
        sub = dwi.subject_dir(cfg, sid)
        sources = [sub / dwi.FA_FILE, sub / tracts.TRACTS_FILE, table_path]
        if not force and _up_to_date(out_dir / sid, sources):
            continue
        export_subject(sid, cfg, out_dir, rows.get(sid), attribution)
        log(f"viewer    {sid}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(cfg, out_dir)
    return write_index(out_dir)
