"""The FA template and the tract atlas. Both are downloaded on first use, never shipped.

The JHU white-matter tractography atlas and the JHU-ICBM FA template are distributed
with FSL under FSL's licence, which allows non-commercial use only. They are fetched
into the local cache and are not part of this repository. The two share one space
and one voxel grid, so registering a subject's FA map to the template places the
tracts. Tracts are found by the names in the atlas's own XML file, not by fixed numbers.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from dtic.config import Config
from dtic.download import fetch
from dtic.errors import PipelineError

JHU_BASE = "https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/master"
JHU_FILES = {
    "labels": ("JHU/JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz", "JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz"),
    "xml": ("JHU-tracts.xml", "JHU-tracts.xml"),
    "fa": ("JHU/JHU-ICBM-FA-1mm.nii.gz", "JHU-ICBM-FA-1mm.nii.gz"),
}
JHU_NOTICE = (
    "Downloading the JHU white-matter tractography atlas and FA template (Hua et al. 2008;\n"
    "Wakana et al. 2007; Mori et al. 2005) from the FSL atlas repository. They are released\n"
    "under the FSL licence: non-commercial use only. They stay in the local cache and are\n"
    "not redistributed here."
)


class AtlasError(PipelineError):
    """The atlas is missing a tract, or could not be read."""


def jhu_paths(cfg: Config, log=print) -> dict[str, Path]:
    """Local paths of the tract labels, their XML and the FA template, downloading them if needed."""
    folder = cfg.paths.cache_dir / "jhu"
    paths = {key: folder / name for key, (_, name) in JHU_FILES.items()}
    if not all(p.is_file() for p in paths.values()):
        log(JHU_NOTICE)
        for key, (remote, _) in JHU_FILES.items():
            fetch(f"{JHU_BASE}/{remote}", paths[key])
    return paths


def read_tract_names(xml_path: Path) -> dict[str, int]:
    """Tract name to its value in the max-probability label image.

    The XML numbers tracts from 0. The label image keeps 0 for background, so a
    tract's voxels carry its XML index plus one.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise AtlasError(f"{xml_path}: not a readable atlas description ({exc})") from exc
    names = {}
    for label in root.iter("label"):
        names[(label.text or "").strip()] = int(label.attrib["index"]) + 1
    if not names:
        raise AtlasError(f"{xml_path}: no tract labels found")
    return names


def tract_values(groups: dict[str, list[str]], names: dict[str, int]) -> dict[str, list[int]]:
    """Label values for each named group of tracts. Unknown names are an error."""
    out = {}
    for group, wanted in groups.items():
        missing = [w for w in wanted if w not in names]
        if missing:
            raise AtlasError(
                f"tract(s) not in the atlas: {', '.join(missing)}. "
                f"available: {', '.join(sorted(names))}"
            )
        out[group] = [names[w] for w in wanted]
    return out
