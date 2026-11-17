"""The one config file: load it, check it, resolve its paths.

Every section is a dataclass whose defaults match config.example.yaml, so a config
only needs the keys it changes. Unknown keys are an error rather than ignored, so a
typo cannot silently fall back to a default.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from dtic.errors import PipelineError


class ConfigError(PipelineError):
    """The config file is missing, malformed, or holds a value that is not allowed."""


# the JHU tract names are the ones in the atlas's own xml. tracts are looked up by
# name at run time, never by a hardcoded index
_ATR = ["Anterior thalamic radiation L", "Anterior thalamic radiation R"]
_SLF = ["Superior longitudinal fasciculus L", "Superior longitudinal fasciculus R"]
_SLF_TEMPORAL = [
    "Superior longitudinal fasciculus (temporal part) L",
    "Superior longitudinal fasciculus (temporal part) R",
]


@dataclass
class Paths:
    dwi_dir: Path = Path("data/dwi")
    clinical_csv: Path = Path("data/clinical.csv")
    fc_csv: Path = Path("data/fc.csv")
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path("cache")

    @property
    def local_dir(self) -> Path:
        """Per-subject outputs. Never leaves the machine."""
        return self.output_dir / "local"

    @property
    def shareable_dir(self) -> Path:
        """Group-level outputs with no subject rows."""
        return self.output_dir / "shareable"


@dataclass
class Acquisition:
    # what a scan is expected to be. a scan that differs is still processed, and flagged
    b_value: float = 1000.0
    directions: int = 64
    b0_volumes: int = 5
    # volumes within this many s/mm2 of b_value form the shell the tensor is fitted to
    b_tolerance: float = 100.0
    b0_threshold: float = 50.0


@dataclass
class Preprocessing:
    motion_correction: bool = True
    transform: str = "affine"
    fit_method: str = "WLS"


@dataclass
class Run:
    # scans processed at the same time. each one needs about 3 GB of memory
    workers: int = 1


@dataclass
class Registration:
    template: str = "JHU-ICBM-FA-1mm"
    speed: str = "fast"


@dataclass
class Tracts:
    atlas: str = "jhu_tracts_maxprob_thr25_1mm"
    # tested measures: each name maps to the atlas labels that are pooled into it
    primary: dict[str, list[str]] = field(
        default_factory=lambda: {"ATR": list(_ATR), "SLF": list(_SLF)}
    )
    # written to the per-subject table, not tested
    secondary: dict[str, list[str]] = field(
        default_factory=lambda: {
            "ATR_L": _ATR[:1],
            "ATR_R": _ATR[1:],
            "SLF_L": _SLF[:1],
            "SLF_R": _SLF[1:],
            "SLF_temporal": list(_SLF_TEMPORAL),
        }
    )
    # only voxels with at least this FA count towards a tract's mean. 0 keeps every voxel
    min_fa: float = 0.2
    # a tract left with fewer voxels than this gets no value
    min_voxels: int = 10


@dataclass
class Stats:
    alpha: float = 0.05
    fdr_method: str = "bh"
    bootstrap_resamples: int = 2000
    seed: int = 0
    min_n: int = 6
    # the FC question is about subjects scanned in both bladder states
    fc_require_both_states: bool = True


@dataclass
class Analyses:
    q1_integrity_fc: bool = True
    q2_ms_vs_ioab: bool = True
    q3_detrusor_overactivity: bool = True
    q4_symptom_severity: bool = True
    q5_treatment_response: bool = False


@dataclass
class TreatmentResponse:
    # points of improvement in nbss total that count as a response. there is no
    # default: it has to be set by whoever turns q5 on
    min_nbss_improvement: float | None = None


@dataclass
class Report:
    min_cell_n: int = 3


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    run: Run = field(default_factory=Run)
    acquisition: Acquisition = field(default_factory=Acquisition)
    preprocessing: Preprocessing = field(default_factory=Preprocessing)
    registration: Registration = field(default_factory=Registration)
    tracts: Tracts = field(default_factory=Tracts)
    stats: Stats = field(default_factory=Stats)
    analyses: Analyses = field(default_factory=Analyses)
    treatment_response: TreatmentResponse = field(default_factory=TreatmentResponse)
    report: Report = field(default_factory=Report)
    # where the config file lives. relative paths are resolved against it
    source: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        """Plain dict of the settings, paths as posix strings. `source` is left out."""
        out = dataclasses.asdict(self)
        out.pop("source")
        return _plain(out)

    def digest(self) -> str:
        """Short hash of the settings that decide the results, for the run manifest."""
        settings = self.as_dict()
        settings.pop("run")     # how many scans run at once changes nothing but the time taken
        blob = json.dumps(settings, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def enabled_analyses(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self.analyses) if getattr(self.analyses, f.name)]


_CHOICES = {
    ("preprocessing", "transform"): {"rigid", "affine"},
    ("preprocessing", "fit_method"): {"WLS", "OLS", "NLLS"},
    ("registration", "template"): {"JHU-ICBM-FA-1mm"},
    ("registration", "speed"): {"fast", "accurate"},
    ("tracts", "atlas"): {"jhu_tracts_maxprob_thr25_1mm"},
    ("stats", "fdr_method"): {"bh"},
}


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _coerce(value: Any, hint: Any, where: str) -> Any:
    """Check one value against its annotated type. Paths come in as strings."""
    name = str(hint)
    optional = "None" in name
    if value is None:
        if optional:
            return None
        raise ConfigError(f"{where}: a value is required")
    if "Path" in name:
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{where}: expected a path, got {value!r}")
        return Path(value)
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true or false, got {value!r}")
        return value
    if hint is int:
        # bool is an int in python, and `yes` in yaml is a bool
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected a whole number, got {value!r}")
        return value
    if hint is float or "float" in name:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected text, got {value!r}")
        return value
    if name.startswith("dict[str, list[str]]"):
        ok = isinstance(value, dict) and all(
            isinstance(k, str) and isinstance(v, list) and v and all(isinstance(s, str) for s in v)
            for k, v in value.items()
        )
        if not ok:
            raise ConfigError(f"{where}: expected a mapping of name to a list of atlas labels")
        return {k: list(v) for k, v in value.items()}
    raise ConfigError(f"{where}: unsupported type {name}")  # pragma: no cover


def _build_section(cls: type, data: Any, section: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{section}: expected a mapping of settings")
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"{section}: unknown setting(s) {', '.join(map(str, unknown))}. "
            f"allowed: {', '.join(sorted(known))}"
        )
    values = {}
    for key, raw in data.items():
        where = f"{section}.{key}"
        value = _coerce(raw, hints[key], where)
        allowed = _CHOICES.get((section, key))
        if allowed is not None and value not in allowed:
            raise ConfigError(f"{where}: {value!r} is not one of {', '.join(sorted(allowed))}")
        values[key] = value
    return cls(**values)


def _validate(cfg: Config) -> None:
    acq, st = cfg.acquisition, cfg.stats
    if cfg.run.workers < 1:
        raise ConfigError("run.workers: must be at least 1")
    if acq.b_value <= 0:
        raise ConfigError("acquisition.b_value: must be positive")
    if acq.directions < 6:
        raise ConfigError("acquisition.directions: a tensor needs at least 6")
    if acq.b0_volumes < 1:
        raise ConfigError("acquisition.b0_volumes: must be at least 1")
    if not 0 < acq.b_tolerance < acq.b_value:
        raise ConfigError("acquisition.b_tolerance: must be positive and below b_value")
    if not 0 <= acq.b0_threshold < acq.b_value - acq.b_tolerance:
        raise ConfigError("acquisition.b0_threshold: must lie below the diffusion shell")
    if not 0 <= cfg.tracts.min_fa < 1:
        raise ConfigError("tracts.min_fa: must be from 0 up to, not including, 1")
    if cfg.tracts.min_voxels < 1:
        raise ConfigError("tracts.min_voxels: must be at least 1")
    if not 0 < st.alpha < 1:
        raise ConfigError("stats.alpha: must be between 0 and 1")
    if st.bootstrap_resamples < 100:
        raise ConfigError("stats.bootstrap_resamples: must be at least 100")
    if st.min_n < 3:
        raise ConfigError("stats.min_n: must be at least 3")
    if cfg.report.min_cell_n < 1:
        raise ConfigError("report.min_cell_n: must be at least 1")
    if not cfg.tracts.primary:
        raise ConfigError("tracts.primary: needs at least one tract")
    clash = sorted(set(cfg.tracts.primary) & set(cfg.tracts.secondary))
    if clash:
        raise ConfigError(f"tracts: {', '.join(clash)} is in both primary and secondary")
    threshold = cfg.treatment_response.min_nbss_improvement
    if cfg.analyses.q5_treatment_response and threshold is None:
        raise ConfigError(
            "treatment_response.min_nbss_improvement: needed when analyses.q5_treatment_response is on"
        )
    if threshold is not None and threshold <= 0:
        raise ConfigError("treatment_response.min_nbss_improvement: must be positive")


def _resolve_paths(cfg: Config, base: Path) -> None:
    for f in dataclasses.fields(cfg.paths):
        p = getattr(cfg.paths, f.name)
        setattr(cfg.paths, f.name, p if p.is_absolute() else (base / p).resolve())


def from_dict(data: Any, base: Path | None = None, source: Path | None = None) -> Config:
    """Build a checked Config from parsed yaml. `base` anchors relative paths."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError("the config must be a mapping of sections")
    sections = {f.name: f for f in dataclasses.fields(Config) if f.name != "source"}
    unknown = sorted(set(data) - set(sections))
    if unknown:
        raise ConfigError(
            f"unknown section(s) {', '.join(map(str, unknown))}. allowed: {', '.join(sections)}"
        )
    hints = get_type_hints(Config)
    built = {name: _build_section(hints[name], data.get(name), name) for name in sections}
    cfg = Config(**built, source=source)
    _validate(cfg)
    _resolve_paths(cfg, (base or Path.cwd()).resolve())
    return cfg


def load(path: str | Path) -> Config:
    """Read a yaml config. Relative paths in it are relative to the file itself."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid yaml: {exc}") from exc
    return from_dict(data, base=path.parent, source=path.resolve())


def check_inputs(cfg: Config) -> list[str]:
    """Problems with the input paths, as messages. Empty when a run can start."""
    problems = []
    p = cfg.paths
    if not p.dwi_dir.is_dir():
        problems.append(f"paths.dwi_dir is not a folder: {p.dwi_dir}")
    for name in ("clinical_csv", "fc_csv"):
        if not getattr(p, name).is_file():
            problems.append(f"paths.{name} not found: {getattr(p, name)}")
    return problems
