import assert from "node:assert/strict";
import { test } from "node:test";

import {
  MD_UNIT, label, num, pvalue, reportHtml, scanAsText, scanRows, scansAsText, tractNames,
} from "../src/report.mjs";

const measurements = [
  {
    id: "sub-01",
    tracts: { ATR: { fa: 0.4123, md: 0.7811, voxels: 2100, kept_pct: 91.2 }, SLF: { fa: 0.4456, md: 0.7402, voxels: 3300, kept_pct: 95 } },
    parts: { ATR_L: { fa: 0.41, md: 0.78 } },
    n_directions: 64, b_value: 1000, n_b0: 5, voxel_mm: "1.97 x 1.97 x 2.5",
    motion_mean_mm: 0.42, motion_max_mm: 1.1, reg_similarity: 0.81, reg_ok: true,
  },
  {
    id: "sub-02",
    tracts: { ATR: { fa: null, md: null }, SLF: { fa: null, md: null } },
    n_directions: 60, b_value: 1000, n_b0: 7, motion_mean_mm: 0.6, reg_similarity: 0.31, reg_ok: false,
  },
];

const row = (extra) => ({
  role: "primary", state: "", connection: "", test: "spearman", n: 20, estimate: -0.41,
  ci_low: -0.7, ci_high: 0.02, p: 0.07, q: null, status: "ok", ...extra,
});

const summary = {
  fake_data: true,
  run: {
    versions: { dtic: "0.1.0", dipy: "1.12.1" },
    counts: { scans: 2, processed: 2, msbladder: 1, ioab: 1, with_clinical: 2, with_fc: 1, registration_qc_failed: 1 },
    settings: { stats: { alpha: 0.05 }, tracts: { min_fa: 0.2 } },
  },
  acquisition: [{ cohort: "MSBladder", n_directions: 64, b_value: 1000, n_b0: 5, voxel_mm: "2 x 2 x 2", scans: 1 }],
  descriptives: [
    { group: "all", variable: "tract_ATR_fa", n: 2, mean: 0.41, sd: 0.02, median: 0.41, q1: 0.4, q3: 0.42, suppressed: false },
    { group: "all", variable: "tract_ATR_L_fa", n: 2, mean: 0.4, sd: 0.02, median: 0.4, q1: 0.39, q3: 0.41, suppressed: false },
    { group: "MSBladder", variable: "fc__full__a<b", n: 1, mean: 0.2, sd: null, median: 0.2, q1: 0.2, q3: 0.2, suppressed: false },
    { group: "women, MSBladder", variable: "age", n: 1, mean: null, sd: null, median: null, q1: null, q3: null, suppressed: true },
    { group: "DO present", variable: "tract_SLF_md", n: 4, mean: 0.74, sd: 0.03, median: 0.74, q1: 0.72, q3: 0.76, suppressed: false },
  ],
  tests: [
    row({ question: "q1_integrity_fc", exposure: "ATR FA", outcome: "FC full: a<b", state: "full", connection: "a<b", q: 0.01 }),
    row({ question: "q1_integrity_fc", exposure: "ATR MD", role: "secondary", outcome: "FC full: c", state: "full", connection: "c", q: 0.6 }),
    row({ question: "q2_ms_vs_ioab", exposure: "MSBladder vs iOAB, women, at equal age", outcome: "ATR FA", test: "linear_model",
      n_group1: 12, n_group0: 9, estimate: -0.021, std_effect: -0.9, p_rank: 0.04 }),
    row({ question: "q3_detrusor_overactivity", exposure: "DO present vs DO absent", outcome: "SLF MD", role: "secondary",
      test: "mann_whitney", n_group1: 4, n_group0: 5, shift: 0.02, shift_ci_low: -0.01, shift_ci_high: 0.05 }),
    row({ question: "q4_symptom_severity", exposure: "ATR FA", outcome: "nbss_total", status: "skipped: n below 6", estimate: null }),
  ],
};

test("numbers are written to three significant figures, in full", () => {
  assert.equal(num(0.41234), "0.412");
  assert.equal(num(1183.4), "1183");
  assert.equal(num(64), "64");
  assert.equal(num(null), "");
  assert.equal(pvalue(0.0004), "<0.001");
  assert.equal(pvalue(0.0712), "0.071");
});

test("columns of the subject table get readable names", () => {
  assert.equal(label("tract_ATR_fa"), "ATR FA");
  assert.equal(label("tract_SLF_temporal_md"), `SLF temporal MD (${MD_UNIT})`);
  assert.equal(label("nbss_total"), "NBSS total");
  assert.equal(label("age"), "Age (years)");
});

test("the scan table has FA and MD for every tested tract", () => {
  assert.deepEqual(tractNames(measurements), ["ATR", "SLF"]);
  const { head, rows } = scanRows(measurements);
  assert.deepEqual(head.slice(0, 5), ["Scan", "ATR FA", "SLF FA", `ATR MD (${MD_UNIT})`, `SLF MD (${MD_UNIT})`]);
  assert.deepEqual(rows[0].slice(0, 5), ["sub-01", 0.4123, 0.4456, 0.7811, 0.7402]);
  assert.equal(rows[1].at(-1), "0.31, failed");
});

test("the scan table copies as tab-separated text, with blanks for missing values", () => {
  const lines = scansAsText(measurements).split("\n");
  assert.equal(lines.length, 3);
  assert.ok(lines[0].startsWith("Scan\tATR FA\tSLF FA"));
  assert.ok(lines[1].startsWith("sub-01\t0.4123\t0.4456\t0.7811\t0.7402\t64\t1000\t5"));
  assert.ok(lines[2].startsWith("sub-02\t\t\t\t\t60"));
});

test("one scan's measurements copy with every tract, side and check", () => {
  const text = scanAsText({ id: "sub-01", stats: measurements[0] });
  assert.ok(text.includes("ATR FA\t0.4123"));
  assert.ok(text.includes(`SLF MD (${MD_UNIT})\t0.7402`));
  assert.ok(text.includes("ATR L FA\t0.41"));
  assert.ok(text.includes("Head motion, mean (mm)\t0.42"));
  assert.ok(scanAsText({ id: "sub-02", stats: measurements[1] }).includes("failed, tract values withheld"));
});

test("the results page says loudly when the tables were fake", () => {
  assert.ok(reportHtml(summary, measurements).includes("Fake data."));
  assert.ok(!reportHtml({ ...summary, fake_data: false }, measurements).includes("Fake data."));
});

test("every question that has tests gets its own section, in order", () => {
  const html = reportHtml(summary, measurements);
  const at = (text) => html.indexOf(text);
  assert.ok(at("Question 1. White matter integrity and functional connectivity") > 0);
  assert.ok(at("Question 2. MS against idiopathic OAB") > at("Question 1."));
  assert.ok(at("Question 3. White matter integrity by detrusor overactivity") > at("Question 2."));
  assert.ok(at("Question 4. White matter integrity and symptom severity") > at("Question 3."));
  assert.equal(at("Question 5."), -1);
  assert.ok(html.includes("Difference at equal age"));
  assert.ok(html.includes("skipped: n below 6"));
  assert.ok(html.includes("How the scans were acquired"));
});

test("text from the tables is escaped, and small groups are withheld", () => {
  const html = reportHtml(summary, measurements);
  assert.ok(html.includes("a&lt;b"));
  assert.ok(!html.includes("a<b"));
  assert.ok(html.includes("withheld, group too small"));
});

test("per-side values stay out of the cohort table", () => {
  const html = reportHtml(summary, measurements);
  const cohort = html.slice(html.indexOf("<h3>Cohort</h3>"), html.indexOf("Question 1."));
  assert.ok(cohort.includes("ATR FA"));
  assert.ok(!cohort.includes("ATR L FA"));
});

test("with no summary yet the page still lists the scans", () => {
  const html = reportHtml(null, measurements);
  assert.ok(html.includes('data-scan="sub-01"'));
  assert.ok(html.includes("no group results yet"));
});
