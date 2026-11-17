# How to read the results

Everything below refers to the files in `outputs/shareable/` unless it says otherwise.

## What is measured

| Column | Meaning |
|---|---|
| `tract_ATR_fa`, `tract_SLF_fa` | Mean fractional anisotropy inside the tract, left and right sides together. FA has no unit and runs from 0 (water moves equally in every direction) to 1 (along one direction only). Healthy white matter in these tracts sits around 0.4 to 0.5. |
| `tract_ATR_md`, `tract_SLF_md` | Mean diffusivity inside the tract, in 10⁻³ mm²/s. Healthy white matter sits around 0.7 to 0.8, free fluid around 3. |
| `tract_*_L`, `_R`, `SLF_temporal` | The same per side, and for the temporal part of the SLF. Reported, not tested. |
| `tract_*_voxels`, `_kept_pct` | How many voxels the mean rests on, and what share of the tract's voxels that is. |
| `n_directions`, `b_value`, `n_b0`, `voxel_mm`, `acq_as_expected` | How the scan was acquired. `acq_as_expected` is false when the scan is not the configured protocol (64 directions at b=1000 with 5 b0 by default). Such a scan is still processed. |
| `motion_mean_mm`, `motion_max_mm` | How far the head sat from its usual position during the scan, averaged over the volumes and for the worst volume. |
| `reg_similarity`, `reg_ok` | A check on the registration: the correlation between the subject's FA map, once warped, and the template. When `reg_ok` is false the tract values for that subject are left empty and the subject drops out of every test. |

Lower FA and higher MD both mean lower white matter integrity.

**Which voxels count.** A tract's value is the mean over the voxels of the atlas tract
that fall inside the brain mask and have FA of at least `tracts.min_fa` (0.2 by default).
The threshold keeps cerebrospinal fluid and grey matter at the tract's edge out of the
mean, which matters most where the ventricles are enlarged. It also leaves out the most
damaged voxels, so differences between groups are, if anything, understated. Set
`tracts.min_fa: 0` to average every voxel instead. Left and right are one pool of
voxels, so the larger side counts for more.

## The tests

Every test is two-sided. Subjects with a missing value are left out of that one test,
and the `n` column says how many went in. FA is the primary measure and MD the
secondary one throughout. Every question but Q2 uses the MSBladder cohort only.

**Spearman correlation** (FA and MD against FC, and against symptom scores).
`estimate` is rho, from -1 to +1. Positive means the two rise together. The interval is
a 95% bootstrap interval: if it spans zero, the data are compatible with no association.
The p-value uses the usual t approximation, which is rough below about ten subjects.

**Mann-Whitney test** (FA and MD in two groups). `estimate` is the rank-biserial
correlation: take every pair of one subject from each group, and it is the share of pairs
where group 1 has the larger value minus the share where group 0 does. +1 means every
group 1 subject is above every group 0 subject, 0 means no separation. `shift` is the
Hodges-Lehmann estimate, the typical difference between the groups in the measure's own
unit (group 1 minus group 0), with a 95% bootstrap interval. Group 1 is "DO present" or
"responder".

**Age-adjusted difference** (MS against iOAB). A linear model of the measure on cohort
and age. `estimate` is how much higher the MSBladder women are than the iOAB women of
the same age, in the measure's own unit, with a 95% bootstrap interval. `std_effect` is
that difference divided by the spread left after the model, so 0.5 is half a standard
deviation. `p` is the model's p-value. `p_rank` is a check that does not assume the
values are normally distributed (Quade's rank analysis of covariance): when the two
p-values disagree badly, look at the data before trusting either. A negative difference
in FA, or a positive one in MD, means lower integrity in MSBladder, which is what the
hypothesis expects.

**p and q.** `p` is the unadjusted p-value. For the FC question many connections are
tested at once, so `q` is the Benjamini-Hochberg false discovery rate, computed across
connections within one measure and one bladder state (for example: ATR FA, full
bladder). Read FC results by `q`: among results with q below 0.05, about 5% are
expected to be false alarms. The other questions report `p` only, with no adjustment
across measures or outcomes, so a single p just under 0.05 among a dozen tests is weak
evidence.

**Status.** A test that could not be run says why (`skipped: n below 6`, `skipped: no
variation`) and has no numbers. It is listed so that nothing disappears silently.

## The questions

| | Who | Exposure or grouping | Outcome | Test |
|---|---|---|---|---|
| Q1 | MSBladder | FA (primary) and MD (secondary) in the ATR and SLF | each FC connection, in the full state, the empty state, and the change | Spearman, FDR across connections |
| Q2 | women, MSBladder and iOAB | MSBladder or iOAB | FA and MD in the ATR and SLF | linear model adjusted for age, with a rank check |
| Q3 | MSBladder | detrusor overactivity present or absent | FA and MD in the ATR and SLF | Mann-Whitney |
| Q4 | MSBladder | FA and MD in the ATR and SLF | NBSS total (primary), three bladder-diary measures (secondary) | Spearman |
| Q5 | MSBladder, active arm | responder or non-responder | baseline FA and MD in the ATR and SLF | Mann-Whitney |

FC change is the full-bladder value minus the empty-bladder value. The FC question is
about subjects scanned in both bladder states, so a subject enters an FC test for a
connection only when both their full and their empty value are there (or a change value
was supplied directly). `stats.fc_require_both_states: false` lifts that rule. Q5 is off
unless the config turns it on and sets the response threshold.

In Q4 a row is marked primary only when both sides are: FA against the NBSS total.

## Where to find things

`dtic view` shows all of this in the browser. The **Scans** side has one scan's
measurements beside its images: FA and MD in the ATR and the SLF at the top, then each
side, then how the scan was acquired and the two checks. **Group results** has the table
of every scan's measurements, the cohort summary, and one section per question, in the
same order as the table above. Both can be copied as text.

## What the summary can and cannot say

- These are associations in one sample. They do not show that white matter damage
  causes anything.
- With a small sample the intervals are wide. An interval that spans zero is not evidence
  of no effect, only of not enough data to tell.
- The tracts are atlas regions warped onto each subject, not that subject's own tracts.
  In a brain with large lesions or marked atrophy the fit is less sure. `reg_similarity`
  flags a registration that failed outright, not one that is slightly off, so look at
  the outlines in the viewer.
- MS lesions inside a tract lower its FA and raise its MD. The tract means do not
  separate lesioned from normal-appearing white matter.
- The MS against iOAB comparison is between two studies. Age is adjusted for, and the
  summary shows how each cohort's scans were acquired, but a difference in scanner,
  protocol or anything else that separates the two studies cannot be told apart from a
  difference between the conditions. If the two cohorts were not scanned the same way,
  read Q2 with that in mind. The age adjustment also assumes that FA and MD change with
  age along a straight line over the ages in the sample, and it works best when the two
  cohorts' ages overlap.
- Motion correction registers each volume to the b0 with an affine transform. That takes
  out head motion and the linear part of eddy-current distortion. Distortion from
  magnetic susceptibility is not corrected, because that needs a second scan with the
  phase encoding reversed.

## What may be shared

`outputs/shareable/` holds counts, means, SDs, medians, quartiles and test results. It has
no subject IDs, no per-subject values and no minimum or maximum (those are one person's
value). Statistics for a group smaller than `report.min_cell_n` are withheld. Everything
in `outputs/local/` is per subject and stays where it is.
