# DTI-Correlation

Measure white matter integrity from a diffusion MRI scan and test how it relates to
functional connectivity (FC), urodynamic findings and bladder symptoms, and whether it
differs between women with MS-related bladder dysfunction and women with idiopathic
overactive bladder. Each scan is corrected for motion and a diffusion tensor is fitted
with DIPY. Fractional anisotropy (FA) and mean diffusivity (MD) are then averaged
inside the anterior thalamic radiation (ATR) and the superior longitudinal fasciculus
(SLF) of the JHU white-matter atlas. One config file, one command.

A run writes a per-subject table that stays on your machine and a group summary with
no subject rows that can be shared. A viewer then shows each scan's FA and MD maps with
the two tracts outlined in three linked slice views, every scan's measurements, and the
results for each question.

Research use only. Nothing here is a medical device or meant for clinical decisions.

## Install

Python 3.11 to 3.13, and Node.js 20 or newer for the viewer.

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
git config core.hooksPath scripts/hooks
```

macOS and Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
git config core.hooksPath scripts/hooks
```

The `git config` line turns on the commit check that keeps data out of the repository.
No GPU is needed. The first run downloads the JHU atlas and FA template (a few MB) into
a local cache.

## Run

1. Copy `config.example.yaml` to `config.yaml`. The default paths work as they are.
2. Put the diffusion scans in `data/dwi/`, one folder per subject: `<id>/dwi.nii.gz` with
   `dwi.bval` and `dwi.bvec` beside it. Convert DICOM to NIfTI first; `dcm2niix` writes
   all three files. The pipeline takes the diffusion series, not a T1.
3. Fill `data/clinical.csv` and `data/fc.csv` from the templates in `templates/`.
   `data_dictionary.csv` explains every column. The subject IDs must match the scan
   folder names. `cohort` says which study each subject is from, `msbladder` or `ioab`.
4. Run it, then look at it:

```bash
dtic run -c config.yaml
dtic view -c config.yaml
```

A scan takes about 5 to 10 minutes, nearly all of it motion correction. To process
several at once, set `run.workers` in the config; each worker needs about 3 GB of memory.

`dtic view` opens the viewer in your browser, served from your own machine only. The
first time, it builds the viewer, which takes a minute.

- **Scans** shows the FA or MD map with the ATR and SLF outlined and a crosshair, with
  the scan's measurements beside it: FA and MD in each tract, the same for each side,
  how the scan was acquired, head motion and the registration check. The scan menu and
  its two arrows switch between subjects.
- **Group results** lists those measurements for every scan and the result for each
  question. Both can be copied as text from the buttons at the top.

The same results are on disk:

| Where | What | May it leave your machine? |
|---|---|---|
| `outputs/local/subjects.csv` | one row per subject: FA and MD in each tract, acquisition, motion and registration checks, plus your clinical and FC columns | No |
| `outputs/local/subjects/<id>/` | FA and MD maps, brain mask, registration, tract labels | No |
| `outputs/shareable/group_summary.md` | counts, means and SDs, and the test results for each question | Yes |
| `outputs/shareable/group_summary.csv`, `descriptives.csv`, `group_summary.json` | the same, in full, for further use | Yes |

[How to read the results](docs/interpreting-results.md) explains every number. A second
run reuses finished scans; `--force` redoes them.

Other commands:

| Command | What it does |
|---|---|
| `dtic run -c config.yaml --dry-run` | check the config and input paths, then stop |
| `dtic run -c config.yaml --view` | run, then open the viewer |
| `dtic show-config -c config.yaml` | print the config with every default filled in |
| `dtic make-fake-data -c config.yaml` | write fake clinical and FC tables for the scans in the folder, to try the pipeline. A summary built on them is stamped FAKE DATA |
| `dtic make-fake-data -c config.yaml --with-scans` | also write fake scans, built from the FA template, when the scans folder is empty |
| `dtic fetch-dev-data -c config.yaml` | download 15 public diffusion scans of healthy adults (about 650 MB) into `data/dev` |
| `dtic fetch -c config.yaml` | download the atlas and the template ahead of time |
| `dtic check-data` | check that no data file is tracked by git |

To try everything without data of your own, on real public scans:

```bash
dtic fetch-dev-data -c config.yaml
```

Then set `paths.dwi_dir` to `data/dev` in `config.yaml` and run:

```bash
dtic make-fake-data -c config.yaml
dtic run -c config.yaml
dtic make-fake-data -c config.yaml --force
dtic run -c config.yaml --view
```

The first run measures the scans. The fake tables are then made again around those
measurements, so the planted effects line up with the real FA, and the second run, which
reuses the finished scans and takes seconds, tests them. The public scans are 64 or 60
directions at b=1000 with one or seven b0 volumes, so they are flagged as not matching the
protocol, and processed all the same.

Or with no download at all, on fake scans built from the FA template:

```bash
dtic make-fake-data -c config.yaml --with-scans
dtic run -c config.yaml --view
```

Either way the first run takes a while: 15 public scans or 12 fake ones at 5 to 10
minutes each, divided by `run.workers`.

### Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
npm --prefix viewer ci
npm --prefix viewer test
```

### Keeping data out of git

Scans, tables and everything derived from them live in `data/`, `outputs/` and `cache/`,
all git-ignored, and the viewer reads its files from `outputs/local/`. The commit check
also refuses imaging files, spreadsheets, CSV files outside `templates/`, the local
`config.yaml` and files over 1 MB, and refuses direct commits on `main`.

### The public demo site

The viewer on GitHub Pages shows a few of the public scans, with group results computed
on fake tables. It lives on the `gh-pages` branch, which holds only the built site.
`main` never contains the demo scans, so cloning `main` to work on patient data does not
bring them along (`git clone --single-branch` skips the `gh-pages` branch entirely).

To rebuild it, export only the scans named in `viewer/public-subjects.json` into
`viewer/public/data/`, which git ignores, and deploy:

```bash
dtic export-viewer -c config.yaml --out viewer/public/data --subjects sub-10159 sub-10228 sub-10339 sub-010002
npm --prefix viewer run deploy
```

The deploy refuses to run if that folder holds a scan that is not on the list, or group
results that were not computed on fake tables. Patient scans never belong on the list.

## Licences and attributions

The code in this repository is under the [MIT licence](LICENSE).

- **DIPY** (BSD 3-Clause) does the motion correction and the tensor fit. Garyfallidis E,
  Brett M, Amirbekian B, et al. Dipy, a library for the analysis of diffusion MRI data.
  *Frontiers in Neuroinformatics* 2014;8:8. doi:10.3389/fninf.2014.00008.
- **ANTs**, through ANTsPy (Apache 2.0), does the registration.
- **JHU white-matter tractography atlas and JHU-ICBM FA template** supply the ATR, the
  SLF and the image they are registered through. Hua K, et al. *NeuroImage*
  2008;39:336-347. Wakana S, et al. *NeuroImage* 2007;36:630-644. Mori S, et al. *MRI
  Atlas of Human White Matter*, Elsevier, 2005. Both are distributed with FSL under the
  [FSL licence](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html), which allows
  **non-commercial use only**. They are not included in this repository: the pipeline
  downloads them into your local cache on first use.
- **UCLA Consortium for Neuropsychiatric Phenomics LA5c Study** (OpenNeuro ds000030, CC0)
  is used for development and for the public demo site. Poldrack RA, et al. A
  phenome-wide examination of neural and cognitive function. *Scientific Data*
  2016;3:160110. doi:10.1038/sdata.2016.110.
- **MPI-Leipzig Mind-Brain-Body** (OpenNeuro ds000221, CC0) is used the same way.
  Babayan A, et al. A mind-brain-body dataset of MRI, EEG, cognition, emotion, and
  peripheral physiology in young and old adults. *Scientific Data* 2019;6:180308.
  doi:10.1038/sdata.2018.308.
- Head motion is summarised with the root-mean-square deviation of Jenkinson M.
  *Measuring transformation error by RMS deviation.* FMRIB technical report TR99MJ1, 1999.
- The rank-based check on the age-adjusted comparison is Quade D. Rank analysis of
  covariance. *Journal of the American Statistical Association* 1967;62:1187-1200.
- The viewer uses three.js (MIT) and IBM Plex Sans (SIL Open Font License).
