"""Fake clinical and FC tables, and fake scans, for trying the pipeline without real data.

The numbers are random. Three associations are planted on purpose so a test run has
something to find, and so the tests can check that it is found:
  - the MSBladder cohort has lower white matter integrity than the iOAB cohort
  - within MSBladder, nbss_total rises as integrity falls
  - within MSBladder, the first connection's full-state value rises with integrity
Everything else is noise. When no integrity is passed in, a random one stands in, so
the planted effects only line up with the imaging if the measured FA is supplied.

The fake scans are built from the FA template: each fake subject is the template
with its FA scaled by a known factor, turned into diffusion signals, with noise and
a little motion added. They exercise every stage and hold no person's anatomy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dtic.schemas import IOAB, MS

PLANTED_CONNECTION = "net01__net02"
# every fake clinical table carries this column, and the summary says so loudly when it sees it
FAKE_COLUMN = "is_fake"
FAKE_VOXEL_MM = 3.0


def connection_names(n_networks: int = 4, n_seed_targets: int = 4) -> list[str]:
    """Generic placeholder labels: network pairs, then seed-to-region values."""
    nets = [f"net{i + 1:02d}" for i in range(n_networks)]
    pairs = [f"{a}__{b}" for i, a in enumerate(nets) for b in nets[i + 1:]]
    seeds = [f"seed01__roi{i + 1:02d}" for i in range(n_seed_targets)]
    return pairs + seeds


def _z(values: np.ndarray) -> np.ndarray:
    """Ranks, rescaled to mean 0 and unit variance."""
    order = np.argsort(np.argsort(values))
    return ((order + 0.5) / len(values) - 0.5) * np.sqrt(12)


def make_fake(subject_ids, seed: int = 0, integrity=None, strength: float = 0.8,
              with_treatment: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (clinical, fc) in the template layouts.

    integrity: optional mapping of subject ID to any number that is higher for
    healthier white matter, such as mean tract FA. Only its ranks are used. strength
    is the share of the planted variables that follows it, between 0 (none) and 1 (all).
    About three in five subjects are put in MSBladder. The symptom, urodynamic and
    FC values exist for MSBladder only, as in the studies.
    """
    ids = [str(s) for s in subject_ids]
    n = len(ids)
    if n < 4 or len(set(ids)) != n:
        raise ValueError("need at least four distinct subject IDs")
    rng = np.random.default_rng(seed)
    if integrity is None:
        latent = rng.normal(size=n)
    else:
        latent = np.array([float(integrity[s]) for s in ids])
        # ties are broken at random so ranks stay spread
        latent = latent + rng.uniform(0, 1e-9, size=n)
    noise_share = np.sqrt(1 - strength ** 2)

    # the lower a subject's integrity, the likelier the MSBladder label
    lean = -strength * _z(latent) + noise_share * rng.normal(size=n)
    n_ms = int(np.clip(round(0.6 * n), 2, n - 2))
    is_ms = np.zeros(n, dtype=bool)
    is_ms[np.argsort(-lean)[:n_ms]] = True

    clinical = pd.DataFrame({"subject_id": ids})
    clinical[FAKE_COLUMN] = 1
    clinical["cohort"] = np.where(is_ms, MS, IOAB)
    female = is_ms | (rng.random(n) < 0.75)
    # the comparison is made in women, so keep some in the iOAB group even in a tiny cohort
    female[np.flatnonzero(~is_ms)[:3]] = True
    clinical["sex"] = np.where(female, "female", "male")
    age = np.where(is_ms, rng.normal(46, 11, n), rng.normal(58, 10, n))
    clinical["age"] = np.clip(np.round(age), np.where(is_ms, 20, 40), np.where(is_ms, 75, 80)).astype(int)

    ms = np.flatnonzero(is_ms)
    z = _z(latent[ms])

    def mix() -> np.ndarray:
        return strength * z + noise_share * rng.normal(size=n_ms)

    def ms_only(values, dtype="Float64") -> pd.Series:
        out = pd.Series(pd.NA, index=clinical.index, dtype=dtype)
        out.iloc[ms] = values
        return out

    do = rng.integers(0, 2, size=n_ms)
    do[:2] = [1, 0]   # keep both groups populated
    clinical["do_present"] = ms_only(do, "Int64")
    nbss = np.clip(np.round(30 - 9 * mix()), 0, 74).astype(int)
    clinical["nbss_total"] = ms_only(nbss, "Int64")
    clinical["diary_voids_per_day"] = ms_only(np.round(np.clip(rng.normal(10, 3, n_ms), 3, None), 1))
    clinical["diary_urgency_per_day"] = ms_only(np.round(np.clip(rng.normal(5, 2.5, n_ms), 0, None), 1))
    clinical["diary_incontinence_per_day"] = ms_only(np.round(np.clip(rng.normal(1.5, 1.5, n_ms), 0, None), 1))
    if with_treatment:
        arm = np.where(rng.random(n_ms) < 2 / 3, "active", "sham")
        arm[:2] = ["active", "sham"]
        clinical["arm"] = ms_only(arm, "string")
        drop = np.where(arm == "active", rng.normal(7, 6, n_ms), rng.normal(3, 5, n_ms))
        clinical["nbss_total_post"] = ms_only(np.clip(np.round(nbss - drop), 0, 74).astype(int), "Int64")

    ms_ids = [ids[i] for i in ms]
    rows = []
    for conn in connection_names():
        shared = rng.normal(0, 0.15, n_ms)  # a subject's two states resemble each other
        for state in ("full", "empty"):
            values = 0.2 + shared + rng.normal(0, 0.12, n_ms)
            if conn == PLANTED_CONNECTION and state == "full":
                values = 0.2 + 0.2 * mix()
            rows.append(pd.DataFrame({
                "subject_id": ms_ids, "state": state, "connection": conn,
                "fc_z": np.round(values, 4),
            }))
    fc = pd.concat(rows, ignore_index=True)
    return clinical, fc


def default_ids(n: int) -> list[str]:
    return [f"sub-{i + 1:03d}" for i in range(n)]


# ---- fake scans ----------------------------------------------------------------------

def directions(n: int) -> np.ndarray:
    """n unit vectors spread evenly over a half sphere."""
    i = np.arange(n) + 0.5
    z = 1 - i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z ** 2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def fake_scan(template_fa: np.ndarray, factor: float, rng, n_directions: int = 64,
              n_b0: int = 5, b_value: float = 1000.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diffusion signals for the template with its FA multiplied by factor.

    Returns (int16 volumes, b-values, b-vectors). Every voxel holds one tensor with
    the scaled FA; mean diffusivity rises as the factor falls, as it does in damaged
    white matter. Volumes carry Rician noise and are each shifted by a fraction of
    a voxel, so that the motion correction has something to do.
    """
    from scipy import ndimage

    brain = template_fa > 0.02
    fa = np.clip(template_fa * factor, 0, 0.95)
    md = 0.8e-3 * (1 + 1.5 * (1 - factor))
    root = np.sqrt(3 - 2 * fa ** 2)
    lam_par = md * (1 + 2 * fa / root)
    lam_perp = md * (1 - fa / root)
    # a fibre direction per voxel that varies smoothly: away from the centre of the volume
    grid = np.stack(np.meshgrid(*(np.arange(s) - (s - 1) / 2 for s in fa.shape), indexing="ij"), -1)
    # the grid holds whole and half numbers, so a quarter keeps every vector away from zero
    fibre = (grid + 0.25) / np.linalg.norm(grid + 0.25, axis=-1, keepdims=True)
    s0 = 1000.0 * brain * (1.2 - 0.5 * fa)
    head = ndimage.binary_dilation(brain, iterations=2)

    g = directions(n_directions)
    b0_at = set(np.linspace(0, n_directions + n_b0 - 1, n_b0, dtype=int).tolist())
    volumes, bvals, bvecs = [], [], []
    k = 0
    for index in range(n_directions + n_b0):
        if index in b0_at:
            signal, bval, bvec = s0, 0.0, np.zeros(3)
        else:
            along = (fibre @ g[k]) ** 2
            signal = s0 * np.exp(-b_value * (lam_perp + (lam_par - lam_perp) * along))
            bval, bvec = b_value, g[k]
            k += 1
        if index:
            signal = ndimage.shift(signal, rng.normal(0, 0.15, 3), order=1)
        noisy = np.hypot(signal + rng.normal(0, 15, fa.shape), rng.normal(0, 15, fa.shape))
        volumes.append(np.where(head, noisy, 0))
        bvals.append(bval)
        bvecs.append(bvec)
    return np.stack(volumes, -1).round().astype(np.int16), np.array(bvals), np.array(bvecs)


def _pose(rng) -> np.ndarray:
    """A small random turn and shift, so no two fake heads lie the same way in the scanner."""
    angles = np.deg2rad(rng.uniform(-4, 4, 3))
    turn = np.eye(4)
    for axis, a in enumerate(angles):
        c, s = np.cos(a), np.sin(a)
        i, j = [(1, 2), (0, 2), (0, 1)][axis]
        step = np.eye(4)
        step[i, i], step[i, j], step[j, i], step[j, j] = c, -s, s, c
        turn = turn @ step
    turn[:3, 3] = rng.uniform(-4, 4, 3)
    return turn


def make_fake_scans(template_fa_path: Path, dwi_dir: Path, ids, seed: int = 0, n_directions: int = 64,
                    n_b0: int = 5, b_value: float = 1000.0, log=print) -> dict[str, float]:
    """Write one fake scan per ID into dwi_dir/<id>/. Returns each ID's true FA factor."""
    import nibabel as nib
    from nibabel.processing import resample_to_output

    rng = np.random.default_rng(seed)
    coarse = resample_to_output(nib.load(str(template_fa_path)), voxel_sizes=FAKE_VOXEL_MM, order=1)
    template = np.asarray(coarse.dataobj, dtype=np.float32)
    factors = {}
    for sid in ids:
        factor = float(rng.uniform(0.75, 1.05))
        data, bvals, bvecs = fake_scan(template, factor, rng, n_directions, n_b0, b_value)
        folder = Path(dwi_dir) / str(sid)
        folder.mkdir(parents=True, exist_ok=True)
        img = nib.Nifti1Image(data, _pose(rng) @ coarse.affine)
        img.set_data_dtype(np.int16)
        nib.save(img, str(folder / "dwi.nii.gz"))
        np.savetxt(folder / "dwi.bval", bvals[None], fmt="%g")
        np.savetxt(folder / "dwi.bvec", bvecs.T, fmt="%.6f")
        factors[str(sid)] = factor
        log(f"wrote     {folder}  (FAKE scan)")
    return factors
