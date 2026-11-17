"""Public diffusion scans for trying the pipeline. None of this is patient data.

Two OpenNeuro datasets, both CC0, both single shell at b=1000 like the study protocol:
  ds000030  64 directions, 1 b0, 3 T Siemens Trio      (the same directions and b-value)
  ds000221  60 directions, 7 b0, 3 T Siemens Verio     (several b0 volumes, as in the protocol)
Only the diffusion image and its b-values and b-vectors are fetched, for a fixed
handful of healthy adults.
"""
from __future__ import annotations

from pathlib import Path

from dtic.download import fetch

OPENNEURO = "https://s3.amazonaws.com/openneuro.org"

DEV_SETS = {
    "ds000030": {
        "name": "UCLA Consortium for Neuropsychiatric Phenomics LA5c Study",
        "citation": ("Poldrack RA, et al. A phenome-wide examination of neural and cognitive "
                     "function. Scientific Data 2016;3:160110. doi:10.1038/sdata.2016.110. "
                     "OpenNeuro ds000030. Licence: CC0."),
        "stem": "{sid}/dwi/{sid}_dwi",
        "subjects": ["sub-10159", "sub-10217", "sub-10227", "sub-10228", "sub-10271", "sub-10273",
                     "sub-10304", "sub-10325", "sub-10339", "sub-10347", "sub-10356", "sub-10361"],
    },
    "ds000221": {
        "name": "MPI-Leipzig Mind-Brain-Body",
        "citation": ("Babayan A, et al. A mind-brain-body dataset of MRI, EEG, cognition, emotion, "
                     "and peripheral physiology in young and old adults. Scientific Data "
                     "2019;6:180308. doi:10.1038/sdata.2018.308. OpenNeuro ds000221. Licence: CC0."),
        "stem": "{sid}/ses-01/dwi/{sid}_ses-01_dwi",
        "subjects": ["sub-010002", "sub-010003", "sub-010004"],
    },
}
SUFFIXES = (".nii.gz", ".bval", ".bvec")


def attribution(sid: str) -> str:
    """The credit line for one public scan, or an empty string for any other ID."""
    for key, spec in DEV_SETS.items():
        if sid in spec["subjects"]:
            return f"Scan from OpenNeuro {key} ({spec['name']}), CC0."
    return ""


def fetch_dev_data(dest: Path, sets=None, log=print) -> list[str]:
    """Download the public scans into dest/<id>/dwi.*. Returns the subject IDs."""
    dest = Path(dest)
    ids = []
    for key in sets or DEV_SETS:
        spec = DEV_SETS[key]
        for sid in spec["subjects"]:
            for suffix in SUFFIXES:
                target = dest / sid / f"dwi{suffix}"
                if not target.is_file():
                    log(f"fetching  {key} {sid}/dwi{suffix}")
                fetch(f"{OPENNEURO}/{key}/{spec['stem'].format(sid=sid)}{suffix}", target)
            ids.append(sid)
    return ids
