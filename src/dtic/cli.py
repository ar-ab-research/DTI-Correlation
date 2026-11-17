"""Command line: `dtic run -c config.yaml` is the whole pipeline."""
from __future__ import annotations

import argparse
import sys

from dtic import __version__, datasafety
from dtic.config import check_inputs, load
from dtic.errors import PipelineError

# what a full run does, in order. each entry is (stage, what it produces)
STAGES = [
    ("inputs", "diffusion scans matched to the clinical and FC tables"),
    ("correct", "motion and eddy correction, b-vectors rotated to match"),
    ("tensor", "tensor fit, FA and MD maps"),
    ("register", "FA map to the FA template"),
    ("tracts", "mean FA and MD inside each tract"),
    ("tables", "per-subject table (local only)"),
    ("stats", "shareable group summary"),
]


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", default="config.yaml",
                        help="path to the config file (default: config.yaml)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtic",
        description="White matter integrity (FA and MD in the ATR and SLF) from diffusion MRI "
                    "and its associations with functional connectivity, urodynamic and symptom data.",
    )
    parser.add_argument("--version", action="version", version=f"dtic {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    run = sub.add_parser("run", help="run the whole pipeline")
    _add_config(run)
    run.add_argument("--force", action="store_true", help="redo stages that already have output")
    run.add_argument("--dry-run", action="store_true",
                     help="check the config and inputs, list the stages, and stop")
    run.add_argument("--view", action="store_true",
                     help="open the results in the viewer when the run is done")

    look = sub.add_parser("view", help="open the processed scans and the group results in the browser")
    _add_config(look)
    look.add_argument("--port", type=int, default=5380, help="port to serve on (default: 5380)")
    look.add_argument("--no-browser", action="store_true", help="serve without opening a browser")

    show = sub.add_parser("show-config", help="print the config with defaults filled in")
    _add_config(show)

    fake = sub.add_parser("make-fake-data", help="write fake clinical and FC tables for testing")
    _add_config(fake)
    fake.add_argument("--with-scans", action="store_true",
                      help="also write fake scans, built from the FA template, into an empty scans folder")
    fake.add_argument("--n", type=int, default=None,
                      help="number of subjects when the scans folder is empty "
                           "(default: 24, or 12 with --with-scans)")
    fake.add_argument("--seed", type=int, default=0)
    fake.add_argument("--force", action="store_true",
                      help="overwrite the clinical and FC files named in the config")

    dev = sub.add_parser("fetch-dev-data",
                         help="download public diffusion scans of healthy adults to try the pipeline on")
    _add_config(dev)
    dev.add_argument("--data-dir", default=None,
                     help="where the scans go (default: data/dev next to the config)")

    fetch = sub.add_parser("fetch", help="download the atlas and template into the cache")
    _add_config(fetch)

    export = sub.add_parser("export-viewer", help="write the viewer's files for chosen scans")
    _add_config(export)
    export.add_argument("--subjects", nargs="*", default=None,
                        help="subject IDs to export (default: every processed subject)")
    export.add_argument("--out", default=None,
                        help="output folder (default: outputs/local/viewer). the public site is "
                             "built from viewer/public/data")
    export.add_argument("--attribution", default="",
                        help="credit line shown in the viewer for these scans")

    check = sub.add_parser("check-data", help="check that no data files are tracked by git")
    check.add_argument("--staged", action="store_true", help="check staged files only")
    return parser


def _run(args) -> int:
    cfg = load(args.config)
    problems = check_inputs(cfg)
    print(f"config    {cfg.source}  ({cfg.digest()})")
    print(f"analyses  {', '.join(cfg.enabled_analyses()) or 'none'}")
    for name, what in STAGES:
        print(f"  {name:9s} {what}")
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        return 2
    if args.dry_run:
        print("dry run: config and inputs are fine")
        return 0
    from dtic import pipeline

    manifest = pipeline.run(cfg, force=args.force)
    print(f"done      {manifest['counts']['processed']} subjects in "
          f"{manifest['seconds']['total']:.0f} s")
    if args.view:
        from dtic import view

        view.view(cfg)
    else:
        print(f"next      dtic view -c {args.config}   to look at the scans and the results")
    return 0


def _view(args) -> int:
    from dtic import view

    view.view(load(args.config), port=args.port, open_browser=not args.no_browser)
    return 0


def _show_config(args) -> int:
    import yaml

    cfg = load(args.config)
    print(yaml.safe_dump(cfg.as_dict(), sort_keys=False), end="")
    return 0


def _measured_integrity(cfg, ids) -> dict | None:
    """Mean FA over the primary tracts, from the last run, when it covers every ID."""
    import pandas as pd

    imaging = cfg.paths.local_dir / "imaging.csv"
    if not imaging.is_file():
        return None
    table = pd.read_csv(imaging, dtype={"subject_id": str}).set_index("subject_id")
    cols = [f"tract_{name}_fa" for name in cfg.tracts.primary]
    if not set(ids) <= set(table.index) or not set(cols) <= set(table.columns):
        return None
    fa = table.loc[ids, cols].mean(axis=1)
    return None if fa.isna().any() else fa.to_dict()


def _make_fake_data(args) -> int:
    from dtic import atlas, fakedata, inputs

    cfg = load(args.config)
    targets = [cfg.paths.clinical_csv, cfg.paths.fc_csv]
    existing = [str(p) for p in targets if p.exists()]
    if existing and not args.force:
        # the same paths hold the real tables on a real run, so never replace them quietly
        print(f"error: not overwriting {', '.join(existing)}. pass --force if these are fake.",
              file=sys.stderr)
        return 2
    ids = list(inputs.discover_dwi(cfg.paths.dwi_dir)) if cfg.paths.dwi_dir.is_dir() else []
    integrity = None
    if args.with_scans:
        if ids:
            # fake scans never go in beside real ones
            print(f"error: {cfg.paths.dwi_dir} already holds scans. fake scans are only written "
                  "into an empty folder.", file=sys.stderr)
            return 2
        ids = fakedata.default_ids(args.n or 12)
        acq = cfg.acquisition
        integrity = fakedata.make_fake_scans(
            atlas.jhu_paths(cfg)["fa"], cfg.paths.dwi_dir, ids, seed=args.seed,
            n_directions=acq.directions, n_b0=acq.b0_volumes, b_value=acq.b_value)
        print(f"subjects  {len(ids)} fake scans in {cfg.paths.dwi_dir}")
        print("planted   effects follow each fake scan's known FA")
    elif ids:
        print(f"subjects  {len(ids)} scans found in {cfg.paths.dwi_dir}")
        integrity = _measured_integrity(cfg, ids)
        print("planted   effects follow the FA measured in the last run" if integrity else
              "planted   no imaging table for these scans yet, effects follow a random stand-in. "
              "run the pipeline, then make the fake tables again")
    else:
        ids = fakedata.default_ids(args.n or 24)
        print(f"subjects  {len(ids)} made-up IDs (no scans in {cfg.paths.dwi_dir})")
    clinical, fc = fakedata.make_fake(ids, seed=args.seed, integrity=integrity)
    for path, frame in zip(targets, (clinical, fc)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, lineterminator="\n")
        print(f"wrote     {path}  ({len(frame)} rows, FAKE)")
    return 0


def _fetch_dev_data(args) -> int:
    from pathlib import Path

    from dtic import devdata

    cfg = load(args.config)
    dest = Path(args.data_dir) if args.data_dir else cfg.source.parent / "data" / "dev"
    ids = devdata.fetch_dev_data(dest)
    print(f"scans     {len(ids)} public scans in {dest}")
    for spec in devdata.DEV_SETS.values():
        print(f"credit    {spec['citation']}")
    print("next      point paths.dwi_dir at that folder in the config")
    return 0


def _fetch(args) -> int:
    from dtic import atlas

    cfg = load(args.config)
    for name, path in atlas.jhu_paths(cfg).items():
        print(f"{name:14s} {path}")
    return 0


def _export_viewer(args) -> int:
    from pathlib import Path

    from dtic import inputs, viewer_export

    cfg = load(args.config)
    out = Path(args.out) if args.out else viewer_export.default_out(cfg)
    scans = inputs.discover_dwi(cfg.paths.dwi_dir)
    ids = viewer_export.export(cfg, list(scans), out, args.subjects, args.attribution, force=True)
    print(f"wrote     {out}  ({len(ids)} subjects listed)")
    return 0


def _check_data(args) -> int:
    return datasafety.run(staged=args.staged)


COMMANDS = {
    "run": _run,
    "view": _view,
    "show-config": _show_config,
    "make-fake-data": _make_fake_data,
    "fetch-dev-data": _fetch_dev_data,
    "fetch": _fetch,
    "export-viewer": _export_viewer,
    "check-data": _check_data,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
