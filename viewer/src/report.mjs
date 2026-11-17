// builds the group results page from the pipeline's summary. plain functions that return
// text, so they can be tested without a browser. everything that came from a file is
// escaped before it goes into html: connection names are whatever the user typed.

export const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));

export const known = (value) => value !== null && value !== undefined && !Number.isNaN(value);

// three significant figures, written out in full
export function num(value, digits = 3) {
  if (!known(value)) return "";
  if (Number.isInteger(value)) return String(value);
  if (Math.abs(value) >= 10 ** digits) return value.toFixed(0);
  return String(Number(value.toPrecision(digits)));
}

export function pvalue(value) {
  if (!known(value)) return "";
  return value < 0.001 ? "<0.001" : value.toFixed(3);
}

export const MD_UNIT = "10⁻³ mm²/s";

const NAMES = {
  age: "Age (years)",
  motion_mean_mm: "Head motion, mean (mm)",
  motion_max_mm: "Head motion, largest (mm)",
  reg_similarity: "Registration check (0 to 1)",
  nbss_total: "NBSS total",
  nbss_total_post: "NBSS total after treatment",
  diary_voids_per_day: "Voids per day",
  diary_urgency_per_day: "Urgency episodes per day",
  diary_incontinence_per_day: "Incontinence episodes per day",
};

// a readable name for a column of the subject table
export function label(variable) {
  if (NAMES[variable]) return NAMES[variable];
  const tract = /^tract_(.+)_(fa|md)$/.exec(variable);
  if (tract) {
    const name = tract[1].replace(/_/g, " ");
    return tract[2] === "fa" ? `${name} FA` : `${name} MD (${MD_UNIT})`;
  }
  return variable.replace(/_/g, " ");
}

export const QUESTIONS = {
  q1_integrity_fc: "White matter integrity and functional connectivity",
  q2_ms_vs_ioab: "MS against idiopathic OAB",
  q3_detrusor_overactivity: "White matter integrity by detrusor overactivity",
  q4_symptom_severity: "White matter integrity and symptom severity",
  q5_treatment_response: "White matter integrity by treatment response",
};
const GROUPS = {
  q2_ms_vs_ioab: ["women, MSBladder", "women, iOAB"],
  q3_detrusor_overactivity: ["DO present", "DO absent"],
  q5_treatment_response: ["responder", "non-responder"],
};
// the per-side and temporal values are reported for each scan, not tested
const isPart = (variable) => /^tract_.+_(L|R|temporal)_(fa|md)$/.test(variable);

function table(head, rows, numeric = []) {
  const cls = (i) => (numeric.includes(i) ? ' class="n"' : "");
  const th = head.map((h, i) => `<th scope="col"${cls(i)}>${h}</th>`).join("");
  const body = rows.map((row) => `<tr>${row.map((cell, i) => `<td${cls(i)}>${cell}</td>`).join("")}</tr>`).join("");
  return `<div class="tablewrap"><table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table></div>`;
}

const interval = (r) => `${num(r.estimate)} <span class="ci">(${num(r.ci_low)} to ${num(r.ci_high)})</span>`;
const estimate = (r) => (r.status === "ok" ? interval(r) : esc(r.status));

// ---- every scan's measurements ------------------------------------------------------

export function tractNames(measurements) {
  const names = new Set();
  measurements.forEach((row) => Object.keys(row.tracts ?? {}).forEach((name) => names.add(name)));
  return [...names];
}

export function scanRows(measurements) {
  const tracts = tractNames(measurements);
  const head = ["Scan", ...tracts.map((t) => `${t} FA`), ...tracts.map((t) => `${t} MD (${MD_UNIT})`),
    "Directions", "b-value", "b0 volumes", "Head motion, mean (mm)", "Registration check"];
  const rows = measurements.map((m) => [
    m.id, ...tracts.map((t) => m.tracts?.[t]?.fa), ...tracts.map((t) => m.tracts?.[t]?.md),
    m.n_directions, m.b_value, m.n_b0, m.motion_mean_mm,
    known(m.reg_similarity) ? `${num(m.reg_similarity)}${m.reg_ok === false ? ", failed" : ""}` : null,
  ]);
  return { head, rows };
}

// tab-separated, so it pastes straight into a spreadsheet or an email
export function scansAsText(measurements) {
  const { head, rows } = scanRows(measurements);
  const line = (cells) => cells.map((c) => (known(c) ? String(c) : "")).join("\t");
  return [line(head), ...rows.map(line)].join("\n");
}

function scanTable(measurements) {
  const { head, rows } = scanRows(measurements);
  const body = rows.map(([id, ...values]) => [
    `<button type="button" class="open" data-scan="${esc(id)}">${esc(id)}</button>`,
    ...values.map((v) => (typeof v === "string" ? esc(v) : num(v))),
  ]);
  return table(head.map(esc), body, head.map((_, i) => i).slice(1));
}

// ---- the group summary ---------------------------------------------------------------

function describeTable(descriptives, group, variables) {
  const rows = [];
  for (const variable of variables) {
    const d = descriptives.find((x) => x.group === group && x.variable === variable);
    if (!d || !d.n) continue;
    rows.push(d.suppressed
      ? [esc(label(variable)), num(d.n), "withheld, group too small", ""]
      : [esc(label(variable)), num(d.n), `${num(d.mean)} ± ${num(d.sd)}`,
        `${num(d.median)} <span class="ci">(${num(d.q1)} to ${num(d.q3)})</span>`]);
  }
  return rows.length ? table(["Measure", "n", "Mean ± SD", "Median (IQR)"], rows, [1, 2, 3]) : "";
}

const variablesOf = (descriptives, group, keep) => [...new Set(
  descriptives.filter((d) => d.group === group && keep(d.variable)).map((d) => d.variable),
)];

function fcSection(tests, alpha) {
  const families = new Map();
  for (const r of tests) {
    const key = `${r.exposure}|${r.state}`;
    if (!families.has(key)) families.set(key, []);
    families.get(key).push(r);
  }
  const overview = [...families.values()].map((rows) => {
    const ok = rows.filter((r) => r.status === "ok");
    const qs = ok.map((r) => r.q).filter(known);
    return [esc(rows[0].exposure), esc(rows[0].role), esc(rows[0].state), num(ok.length),
      num(ok.filter((r) => known(r.q) && r.q < alpha).length), qs.length ? pvalue(Math.min(...qs)) : ""];
  });
  let html = table(["Measure", "Role", "Bladder state", "Connections tested", `q below ${alpha}`, "Smallest q"],
    overview, [3, 4, 5]);
  const hits = tests.filter((r) => known(r.q) && r.q < alpha).sort((a, b) => a.q - b.q);
  if (hits.length) {
    html += `<p class="note">Connections with q below ${alpha}:</p>`;
    html += table(["Measure", "Bladder state", "Connection", "n", "rho (95% CI)", "p", "q"],
      hits.map((r) => [esc(r.exposure), esc(r.state), esc(r.connection), num(r.n), interval(r),
        pvalue(r.p), pvalue(r.q)]), [3, 4, 5, 6]);
  } else {
    html += `<p class="note">No connection reached q below ${alpha}.</p>`;
  }
  return html;
}

function correlationSection(tests) {
  return table(["Measure", "Outcome", "Role", "n", "rho (95% CI)", "p, unadjusted"],
    tests.map((r) => [esc(r.exposure), esc(label(r.outcome)), esc(r.role), num(r.n),
      estimate(r), pvalue(r.p)]), [3, 4, 5]);
}

function groupTables(question, descriptives, keep) {
  let html = "";
  for (const group of GROUPS[question]) {
    const block = describeTable(descriptives, group, variablesOf(descriptives, group, keep));
    if (block) html += `<h4>${esc(group)}</h4>${block}`;
  }
  return html;
}

const isMeasure = (variable) => /^tract_.+_(fa|md)$/.test(variable) && !isPart(variable);

function groupSection(tests, question, descriptives) {
  const [one, zero] = GROUPS[question];
  let html = table(["Measure", "Role", `n, ${esc(one)} / ${esc(zero)}`, "Rank-biserial r (95% CI)",
    "Typical difference (95% CI)", "p, unadjusted"],
  tests.map((r) => [esc(r.outcome), esc(r.role), `${num(r.n_group1)} / ${num(r.n_group0)}`, estimate(r),
    r.status === "ok"
      ? `${num(r.shift)} <span class="ci">(${num(r.shift_ci_low)} to ${num(r.shift_ci_high)})</span>` : "",
    pvalue(r.p)]), [2, 3, 4, 5]);
  html += groupTables(question, descriptives, isMeasure);
  html += `<p class="note">A positive r or difference means the ${esc(one)} group has the higher values. `
    + `MD differences are in ${MD_UNIT}.</p>`;
  return html;
}

function adjustedSection(tests, descriptives) {
  let html = table(["Measure", "Role", "n, MSBladder / iOAB", "Difference at equal age (95% CI)",
    "Standardized", "p", "p, rank check"],
  tests.map((r) => [esc(r.outcome), esc(r.role), `${num(r.n_group1)} / ${num(r.n_group0)}`, estimate(r),
    num(r.std_effect), pvalue(r.p), pvalue(r.p_rank)]), [2, 3, 4, 5, 6]);
  html += groupTables("q2_ms_vs_ioab", descriptives, (v) => v === "age" || isMeasure(v));
  html += '<p class="note">Women only. The difference is MSBladder minus iOAB at the same age, from a '
    + "linear model. A negative difference in FA, or a positive one in MD, means lower integrity in "
    + "MSBladder. The rank check is Quade's rank analysis of covariance. The group values above are "
    + "not adjusted for age.</p>";
  return html;
}

function acquisitionTable(acquisition) {
  if (!acquisition?.length) return "";
  return `<h4>How the scans were acquired</h4>${table(
    ["Cohort", "Directions", "b-value", "b0 volumes", "Voxel (mm)", "Scans"],
    acquisition.map((a) => [esc(a.cohort), num(a.n_directions), num(a.b_value), num(a.n_b0),
      esc(a.voxel_mm), num(a.scans)]), [1, 2, 3, 5])}`;
}

export function reportHtml(summary, measurements) {
  const parts = [];
  if (summary?.fake_data) {
    parts.push('<p class="banner"><b>Fake data.</b> The clinical and FC tables behind these results '
      + "were made up to test the pipeline. Nothing in the group results is a finding.</p>");
  }
  if (measurements.length) {
    parts.push(`<section><h3>Each scan</h3>
      <p class="note">Select a scan to open it. FA has no unit and runs from 0 to 1. MD is in ${MD_UNIT}.
      Each value is the mean over the tract, left and right sides together.</p>${scanTable(measurements)}</section>`);
  }
  if (!summary) {
    parts.push('<section><h3>Group results</h3><p class="note">There are no group results yet. '
      + "They appear here after <code>dtic run</code> has finished.</p></section>");
    return parts.join("");
  }
  const { run, descriptives = [], tests = [], acquisition = [] } = summary;
  const alpha = run?.settings?.stats?.alpha ?? 0.05;
  const counts = run?.counts ?? {};
  const minFa = run?.settings?.tracts?.min_fa ?? 0;
  const rule = minFa > 0 ? `voxels inside the brain with FA of at least ${num(minFa)}` : "every voxel inside the brain";
  const cohorts = ["MSBladder", "iOAB"].map((cohort) => {
    const block = describeTable(descriptives, cohort,
      variablesOf(descriptives, cohort, (v) => !v.startsWith("fc__") && !isPart(v)));
    return block ? `<h4>${cohort}</h4>${block}` : "";
  }).join("");
  parts.push(`<section><h3>Cohort</h3>
    <p class="note">${num(counts.processed)} of ${num(counts.scans)} scans processed:
    ${num(counts.msbladder)} MSBladder, ${num(counts.ioab)} iOAB.
    ${num(counts.with_clinical)} with clinical data, ${num(counts.with_fc)} with FC data.
    ${num(counts.registration_qc_failed)} failed the registration check and have no tract values.
    A tract's value is the mean over ${rule}. Lower FA and higher MD both mean lower white matter
    integrity. Every test is two-sided, and every question but the second uses MSBladder only.</p>
    <h4>All scans</h4>
    ${describeTable(descriptives, "all", variablesOf(descriptives, "all", (v) => !isPart(v)))}
    ${cohorts}${acquisitionTable(acquisition)}
    </section>`);

  let index = 0;
  for (const [question, title] of Object.entries(QUESTIONS)) {
    index += 1;
    const block = tests.filter((r) => r.question === question);
    if (!block.length) continue;
    let body;
    if (question === "q1_integrity_fc") body = fcSection(block, alpha);
    else if (question === "q2_ms_vs_ioab") body = adjustedSection(block, descriptives);
    else if (question === "q4_symptom_severity") body = correlationSection(block);
    else body = groupSection(block, question, descriptives);
    parts.push(`<section><h3>Question ${index}. ${esc(title)}</h3>${body}</section>`);
  }
  return parts.join("");
}

// one scan's numbers as text, for pasting into a message
export function scanAsText(meta) {
  const s = meta.stats ?? {};
  const lines = [`Scan\t${meta.id}`];
  const add = (name, value) => { if (known(value)) lines.push(`${name}\t${value}`); };
  const tract = (name, values) => {
    add(`${name} FA`, values.fa);
    add(`${name} MD (${MD_UNIT})`, values.md);
  };
  for (const [name, values] of Object.entries(s.tracts ?? {})) tract(name, values);
  for (const [name, values] of Object.entries(s.parts ?? {})) tract(name.replace(/_/g, " "), values);
  for (const [name, values] of Object.entries(s.tracts ?? {})) {
    add(`${name}, voxels in the mean`, values.voxels);
    add(`${name}, share of the tract kept (%)`, values.kept_pct);
  }
  add("Directions", s.n_directions);
  add("b-value (s/mm²)", s.b_value);
  add("b0 volumes", s.n_b0);
  add("Voxel (mm)", s.voxel_mm);
  add("Head motion, mean (mm)", s.motion_mean_mm);
  add("Head motion, largest (mm)", s.motion_max_mm);
  add("Registration check (0 to 1)", s.reg_similarity);
  if (s.reg_ok === false) lines.push("Registration check\tfailed, tract values withheld");
  return lines.join("\n");
}
