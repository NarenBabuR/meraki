const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.title = "RAG Pipeline: Build, Break, Fix";

// ---- palette ----
const NAVY = "0B1F3A", TEAL = "0D9488", TEAL2 = "14B8A6", MINT = "5EEAD4";
const LIGHT = "F4F7FB", WHITE = "FFFFFF", INK = "1E293B", MUTE = "64748B";
const AMBER = "D97706", GREEN = "10B981", RED = "DC4C4C", CARD = "FFFFFF";
const HF = "Trebuchet MS", BF = "Calibri", MONO = "Consolas";
const W = 10, H = 5.625;

const shadow = () => ({ type: "outer", color: "0B1F3A", blur: 7, offset: 3, angle: 135, opacity: 0.12 });

function footer(slide, n) {
  slide.addText("RAG · Build / Break / Fix", { x: 0.5, y: 5.28, w: 4, h: 0.25, fontFace: BF, fontSize: 9, color: MUTE, align: "left" });
  slide.addText(String(n), { x: 9.2, y: 5.28, w: 0.3, h: 0.25, fontFace: BF, fontSize: 9, color: MUTE, align: "right" });
}

function header(slide, kicker, title, kColor = TEAL) {
  slide.background = { color: LIGHT };
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.16, h: H, fill: { color: kColor } });
  slide.addText(kicker.toUpperCase(), { x: 0.5, y: 0.34, w: 9, h: 0.3, fontFace: BF, fontSize: 12, bold: true, color: kColor, charSpacing: 2 });
  slide.addText(title, { x: 0.5, y: 0.62, w: 9, h: 0.7, fontFace: HF, fontSize: 28, bold: true, color: INK });
}

function card(slide, x, y, w, h, fill = CARD) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: "E2E8F0", width: 1 }, shadow: shadow() });
}

function stat(slide, x, y, w, big, label, color) {
  slide.addText(big, { x, y, w, h: 0.8, fontFace: HF, fontSize: 46, bold: true, color, align: "center", margin: 0 });
  slide.addText(label, { x, y: y + 0.78, w, h: 0.5, fontFace: BF, fontSize: 12, color: MUTE, align: "center", margin: 0 });
}

function pipe(slide, x, y, items, boxW, color) {
  // horizontal flow of small boxes with arrows
  const gap = 0.24, boxH = 0.62;
  let cx = x;
  items.forEach((it, i) => {
    card(slide, cx, y, boxW, boxH, WHITE);
    slide.addShape(pres.shapes.RECTANGLE, { x: cx, y, w: 0.07, h: boxH, fill: { color } });
    slide.addText(it, { x: cx + 0.12, y, w: boxW - 0.18, h: boxH, fontFace: BF, fontSize: 11.5, bold: true, color: INK, valign: "middle", margin: 2 });
    cx += boxW;
    if (i < items.length - 1) {
      slide.addText("›", { x: cx, y, w: gap, h: boxH, fontFace: HF, fontSize: 20, bold: true, color: color, align: "center", valign: "middle" });
      cx += gap;
    }
  });
}

// ============================================================ 1. TITLE
let s = pres.addSlide();
s.background = { color: NAVY };
s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.55, w: W, h: 0.05, fill: { color: TEAL2 } });
s.addText("MERAKI LABS · WORK TRIAL · PS1", { x: 0.7, y: 1.15, w: 9, h: 0.4, fontFace: BF, fontSize: 14, color: MINT, charSpacing: 3, bold: true });
s.addText("RAG Pipeline:", { x: 0.65, y: 1.65, w: 9, h: 0.9, fontFace: HF, fontSize: 50, bold: true, color: WHITE });
s.addText([
  { text: "Build", options: { color: TEAL2 } }, { text: "  ·  ", options: { color: MUTE } },
  { text: "Break", options: { color: AMBER } }, { text: "  ·  ", options: { color: MUTE } },
  { text: "Fix", options: { color: GREEN } },
], { x: 0.65, y: 2.55, w: 9, h: 0.9, fontFace: HF, fontSize: 50, bold: true });
s.addText("A retrieval-augmented QA system over ~20 arXiv ML papers — stress-tested, measured, and hardened.",
  { x: 0.7, y: 3.7, w: 8.6, h: 0.6, fontFace: BF, fontSize: 16, color: "CBD5E1" });
s.addText("github.com/NarenBabuR/meraki", { x: 0.7, y: 4.75, w: 9, h: 0.4, fontFace: BF, fontSize: 13, color: "94A3B8" });

// ============================================================ 2. WHAT I BUILT
s = pres.addSlide();
header(s, "The brief", "Build a RAG pipeline. Break it. Fix it. Prove it with eval.");
s.addText("Ask a question over a domain-rich corpus → get a grounded, cited answer — or an honest “I don’t know.”",
  { x: 0.5, y: 1.45, w: 9, h: 0.4, fontFace: BF, fontSize: 14, italic: true, color: MUTE });
const deliv = [
  ["Working pipeline", "Streamlit chat + Python API over 20 arXiv ML papers (7.3k chunks)"],
  ["Eval framework", "Ragas (4 metrics) + a deterministic abstention metric, 26-question gold set"],
  ["Build / Break / Fix", "2 failure modes documented, 1 fixed — with before/after eval scores"],
];
deliv.forEach((d, i) => {
  const x = 0.5 + i * 3.07;
  card(s, x, 2.1, 2.9, 2.4);
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: 2.35, w: 0.5, h: 0.5, fill: { color: TEAL } });
  s.addText(String(i + 1), { x: x + 0.25, y: 2.35, w: 0.5, h: 0.5, fontFace: HF, fontSize: 22, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText(d[0], { x: x + 0.25, y: 3.0, w: 2.45, h: 0.4, fontFace: HF, fontSize: 16, bold: true, color: INK, margin: 0 });
  s.addText(d[1], { x: x + 0.25, y: 3.45, w: 2.45, h: 1.0, fontFace: BF, fontSize: 12.5, color: MUTE, margin: 0 });
});
footer(s, 2);

// ============================================================ 3. ARCHITECTURE
s = pres.addSlide();
header(s, "Architecture", "Two workflows, one shared pipeline");
s.addText("INDEX-TIME  (once)", { x: 0.5, y: 1.5, w: 9, h: 0.3, fontFace: BF, fontSize: 12, bold: true, color: TEAL, charSpacing: 1 });
pipe(s, 0.5, 1.85, ["arXiv PDFs", "extract + section", "parent→child", "embed (BGE)", "ChromaDB"], 1.6, TEAL);
s.addText("QUERY-TIME  (per question)", { x: 0.5, y: 2.95, w: 9, h: 0.3, fontFace: BF, fontSize: 12, bold: true, color: NAVY, charSpacing: 1 });
pipe(s, 0.5, 3.3, ["question", "embed", "search 25", "rerank + gate", "merge→parents"], 1.6, NAVY);
// final box
card(s, 0.5, 4.35, 9, 0.75, "0B1F3A");
s.addText([
  { text: "Claude (Sonnet 4.6) ", options: { bold: true, color: MINT } },
  { text: "— answers only from retrieved context, cites sources, abstains when nothing is relevant.", options: { color: "E2E8F0" } },
], { x: 0.75, y: 4.35, w: 8.5, h: 0.75, fontFace: BF, fontSize: 13.5, valign: "middle" });
footer(s, 3);

// ============================================================ 4. KEY DECISIONS
s = pres.addSlide();
header(s, "Decisions & tradeoffs", "Runnable beats fancy: one key, no cloud infra");
const dec = [
  ["Local embeddings + rerank", "BGE-small + cross-encoder run on-device. Only Claude needs a key — free, reproducible, offline retrieval."],
  ["ChromaDB, embedded", "A folder on disk. No server, no Docker. The right altitude for a single-node demo."],
  ["Claude for gen + judge", "Sonnet 4.6 generates and judges eval. One vendor, one secret."],
  ["No cloud dependencies", "No AWS, Redis, SQS, S3, or paid PDF libraries — everything heavy runs locally and free, so it boots anywhere with one key."],
];
dec.forEach((d, i) => {
  const x = 0.5 + (i % 2) * 4.6, y = 1.5 + Math.floor(i / 2) * 1.75;
  card(s, x, y, 4.4, 1.55);
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.09, h: 1.55, fill: { color: TEAL } });
  s.addText(d[0], { x: x + 0.3, y: y + 0.18, w: 4.0, h: 0.4, fontFace: HF, fontSize: 15.5, bold: true, color: INK, margin: 0 });
  s.addText(d[1], { x: x + 0.3, y: y + 0.62, w: 4.0, h: 0.85, fontFace: BF, fontSize: 12, color: MUTE, margin: 0 });
});
footer(s, 4);

// ============================================================ 5. CHUNKING small-to-big
s = pres.addSlide();
header(s, "Chunking strategy", "Small-to-big: match small, answer big");
s.addText("Embed small children for precise matching — return the larger parent to the LLM for context. Every chunk tagged with its paper section.",
  { x: 0.5, y: 1.45, w: 9, h: 0.55, fontFace: BF, fontSize: 13.5, color: MUTE });
// parent box with children
card(s, 0.6, 2.25, 4.2, 2.6, WHITE);
s.addText("PARENT  ~1024 chars  (returned to LLM)", { x: 0.8, y: 2.4, w: 3.8, h: 0.3, fontFace: BF, fontSize: 11, bold: true, color: TEAL });
["child ~256  (embedded + searched)", "child ~256", "child ~256", "child ~256"].forEach((c, i) => {
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 2.78 + i * 0.48, w: 3.8, h: 0.4, fill: { color: i === 0 ? MINT : "E6FBF6" }, line: { color: TEAL2, width: 1 } });
  s.addText(c, { x: 0.95, y: 2.78 + i * 0.48, w: 3.6, h: 0.4, fontFace: MONO, fontSize: 10.5, color: INK, valign: "middle", margin: 0 });
});
// why
card(s, 5.1, 2.25, 4.3, 2.6, WHITE);
s.addText("Why", { x: 5.35, y: 2.4, w: 3.8, h: 0.3, fontFace: HF, fontSize: 15, bold: true, color: INK });
s.addText([
  { text: "Precision: ", options: { bold: true, color: TEAL } }, { text: "short children embed cleanly — less noise per vector.", options: { color: INK } },
  { text: "\n\nContext: ", options: { bold: true, color: TEAL } }, { text: "the LLM still sees the full surrounding passage.", options: { color: INK } },
  { text: "\n\nCitations: ", options: { bold: true, color: TEAL } }, { text: "section metadata (Abstract, §3 Method, …).", options: { color: INK } },
  { text: "\n\nToggle: ", options: { bold: true, color: MUTE } }, { text: "SMALL_TO_BIG; flat chunking still available.", options: { color: MUTE } },
], { x: 5.35, y: 2.75, w: 3.85, h: 2.0, fontFace: BF, fontSize: 12.5, valign: "top", lineSpacingMultiple: 1.0 });
footer(s, 5);

// ============================================================ 6. RETRIEVAL + GATE
s = pres.addSlide();
header(s, "Retrieval", "Two stages + a relevance gate that knows when to stop");
s.addText([
  { text: "Wide then narrow: ", options: { bold: true, color: INK } },
  { text: "embed → cosine search (25) → cross-encoder rerank → keep what clears the bar.", options: { color: MUTE } },
], { x: 0.5, y: 1.45, w: 9, h: 0.4, fontFace: BF, fontSize: 13.5 });
// the gate insight
card(s, 0.5, 2.05, 9, 1.05, WHITE);
s.addText("The cross-encoder cleanly separates relevant from off-topic — so it doubles as an abstention gate:", { x: 0.75, y: 2.2, w: 8.5, h: 0.35, fontFace: BF, fontSize: 12.5, bold: true, color: INK, margin: 0 });
s.addText([
  { text: "in-domain  +5 to +8", options: { color: GREEN, bold: true } },
  { text: "      vs      ", options: { color: MUTE } },
  { text: "out-of-domain  −7 to −10", options: { color: RED, bold: true } },
  { text: "      →  gate at −3.0", options: { color: NAVY, bold: true } },
], { x: 0.75, y: 2.55, w: 8.5, h: 0.5, fontFace: MONO, fontSize: 15, valign: "middle", margin: 0 });
stat(s, 0.5, 3.5, 3, "25 → 5", "candidates → final", NAVY);
stat(s, 3.5, 3.5, 3, "142 ms", "mean retrieval (p95 229)", TEAL);
stat(s, 6.5, 3.5, 3, "0 calls", "to the LLM when it abstains", GREEN);
footer(s, 6);

// ============================================================ 7. EVAL FRAMEWORK
s = pres.addSlide();
header(s, "Evaluation", "“A system with no eval is one you cannot improve.”");
const metrics = [
  ["Context Precision", "are retrieved chunks relevant & ranked?"],
  ["Context Recall", "did we find the chunks the answer needs?"],
  ["Faithfulness", "is the answer grounded — no hallucination?"],
  ["Answer Relevancy", "does it actually address the question?"],
];
metrics.forEach((m, i) => {
  const y = 1.55 + i * 0.62;
  card(s, 0.5, y, 5.1, 0.54, WHITE);
  s.addShape(pres.shapes.OVAL, { x: 0.65, y: y + 0.13, w: 0.28, h: 0.28, fill: { color: TEAL } });
  s.addText(m[0], { x: 1.05, y, w: 1.9, h: 0.54, fontFace: HF, fontSize: 12.5, bold: true, color: INK, valign: "middle", margin: 0 });
  s.addText(m[1], { x: 2.95, y, w: 2.55, h: 0.54, fontFace: BF, fontSize: 10.5, color: MUTE, valign: "middle", margin: 0 });
});
card(s, 5.9, 1.55, 3.6, 2.5, "0B1F3A");
s.addText("+ a 5th, deterministic metric", { x: 6.1, y: 1.7, w: 3.2, h: 0.35, fontFace: HF, fontSize: 14, bold: true, color: MINT });
s.addText("Abstention / hallucination rate — no LLM judge, fully reproducible. The right tool for “did it refuse the unanswerable?” (Ragas wasn’t — slide 12).",
  { x: 6.1, y: 2.1, w: 3.2, h: 1.9, fontFace: BF, fontSize: 12, color: "E2E8F0", valign: "top" });
s.addText([
  { text: "Gold set: ", options: { bold: true, color: INK } },
  { text: "26 hand-curated Q&A  —  18 in-domain · 4 multi-hop · 4 out-of-domain (unanswerable).   Judge: Claude Sonnet 4.6.", options: { color: MUTE } },
], { x: 0.5, y: 4.35, w: 9, h: 0.6, fontFace: BF, fontSize: 12.5 });
footer(s, 7);

// ============================================================ 8. BASELINE RESULTS
s = pres.addSlide();
header(s, "Baseline results", "Strong in-domain; multi-hop is the weak spot");
s.addChart(pres.charts.BAR, [
  { name: "Precision", labels: ["In-domain", "Multi-hop"], values: [0.76, 0.46] },
  { name: "Recall", labels: ["In-domain", "Multi-hop"], values: [0.89, 0.75] },
  { name: "Faithfulness", labels: ["In-domain", "Multi-hop"], values: [0.98, 0.78] },
], {
  x: 0.5, y: 1.55, w: 5.7, h: 3.4, barDir: "col",
  chartColors: [TEAL, TEAL2, NAVY], chartArea: { fill: { color: LIGHT } },
  valAxisMinVal: 0, valAxisMaxVal: 1, catAxisLabelColor: INK, catAxisLabelFontSize: 11,
  valAxisLabelColor: MUTE, valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 9,
  showLegend: true, legendPos: "b", legendColor: MUTE, legendFontSize: 10,
});
card(s, 6.5, 1.55, 3, 1.55, WHITE);
s.addText("Out-of-domain", { x: 6.7, y: 1.68, w: 2.6, h: 0.3, fontFace: HF, fontSize: 13, bold: true, color: INK });
s.addText("scores ~0 by design — the system abstains. Those are good zeros.", { x: 6.7, y: 2.0, w: 2.6, h: 1.0, fontFace: BF, fontSize: 11.5, color: MUTE });
card(s, 6.5, 3.3, 3, 1.65, "0B1F3A");
s.addText("Honest read", { x: 6.7, y: 3.42, w: 2.6, h: 0.3, fontFace: HF, fontSize: 13, bold: true, color: MINT });
s.addText("Multi-hop precision 0.46 is the clearest gap — retrieval finds one hop, not both.", { x: 6.7, y: 3.74, w: 2.6, h: 1.1, fontFace: BF, fontSize: 11.5, color: "E2E8F0" });
footer(s, 8);

// ============================================================ 9. CHUNKING EXPERIMENT
s = pres.addSlide();
header(s, "Experiment", "Small-to-big vs flat: a measured tradeoff, not a free win");
// two columns
const cmp = [
  ["Faithfulness", "0.91", "0.98", GREEN, "+0.07"],
  ["Retrieval latency", "327 ms", "142 ms", GREEN, "−57%"],
  ["Context precision", "0.85", "0.76", AMBER, "−0.09"],
];
s.addText("FLAT 1024", { x: 3.7, y: 1.5, w: 2.0, h: 0.3, fontFace: BF, fontSize: 11, bold: true, color: MUTE, align: "center" });
s.addText("SMALL-TO-BIG", { x: 5.8, y: 1.5, w: 2.2, h: 0.3, fontFace: BF, fontSize: 11, bold: true, color: TEAL, align: "center" });
cmp.forEach((r, i) => {
  const y = 1.85 + i * 0.62;
  card(s, 0.5, y, 9, 0.54, WHITE);
  s.addText(r[0], { x: 0.7, y, w: 2.9, h: 0.54, fontFace: HF, fontSize: 13, bold: true, color: INK, valign: "middle", margin: 0 });
  s.addText(r[1], { x: 3.7, y, w: 2.0, h: 0.54, fontFace: MONO, fontSize: 14, color: MUTE, align: "center", valign: "middle", margin: 0 });
  s.addText(r[2], { x: 5.8, y, w: 2.2, h: 0.54, fontFace: MONO, fontSize: 14, bold: true, color: INK, align: "center", valign: "middle", margin: 0 });
  s.addText(r[4], { x: 8.0, y, w: 1.3, h: 0.54, fontFace: HF, fontSize: 13, bold: true, color: r[3], align: "center", valign: "middle", margin: 0 });
});
card(s, 0.5, 3.85, 9, 1.1, "FFF7ED");
s.addText([
  { text: "The precision drop is a metric artifact. ", options: { bold: true, color: AMBER } },
  { text: "Ragas scores each returned unit; a 1024-char parent holds the answer plus context, so it reads as “less precise” even though retrieval found the answer and faithfulness went up. Shipped small-to-big as default — faithfulness + citations matter most.", options: { color: INK } },
], { x: 0.75, y: 3.95, w: 8.5, h: 0.9, fontFace: BF, fontSize: 12.5, valign: "middle" });
footer(s, 9);

// ============================================================ 10. BREAK #1
s = pres.addSlide();
header(s, "Break #1", "Turn off reranking → answers get less grounded", AMBER);
s.addText("Without the cross-encoder, worse children are selected → worse parents → the LLM’s answer is less faithful.",
  { x: 0.5, y: 1.45, w: 9, h: 0.5, fontFace: BF, fontSize: 13.5, color: MUTE });
stat(s, 1.2, 2.3, 3, "0.98", "faithfulness — rerank ON", GREEN);
s.addText("→", { x: 4.2, y: 2.35, w: 1.4, h: 0.8, fontFace: HF, fontSize: 40, bold: true, color: AMBER, align: "center" });
stat(s, 5.6, 2.3, 3, "0.85", "faithfulness — rerank OFF", AMBER);
card(s, 0.5, 3.95, 9, 1.0, WHITE);
s.addText([
  { text: "Also: ", options: { bold: true, color: INK } },
  { text: "precision 0.76 → 0.71.  ", options: { color: MUTE } },
  { text: "On the flat index the same flag hit precision far harder (0.85→0.63) — parent-merging absorbs it and the damage resurfaces as faithfulness. Same cause, different metric.", options: { color: INK } },
], { x: 0.75, y: 4.05, w: 8.5, h: 0.8, fontFace: BF, fontSize: 12.5, valign: "middle" });
footer(s, 10);

// ============================================================ 11. BREAK #2 (key insight)
s = pres.addSlide();
header(s, "Break #2  ·  the key insight", "Failure modes are config-specific", AMBER);
s.addText("BGE wants an instruction prefix on queries only. Forgetting it is an invisible bug — and its impact flips with the chunking strategy:",
  { x: 0.5, y: 1.45, w: 9, h: 0.55, fontFace: BF, fontSize: 13.5, color: MUTE });
card(s, 0.6, 2.2, 4.25, 1.5, "FEF2F2");
s.addText("Flat 1024 chunks", { x: 0.8, y: 2.32, w: 3.8, h: 0.3, fontFace: HF, fontSize: 14, bold: true, color: INK });
s.addText("recall  0.93 → 0.83", { x: 0.8, y: 2.66, w: 3.8, h: 0.4, fontFace: MONO, fontSize: 16, bold: true, color: RED });
s.addText("a real −10 pt regression", { x: 0.8, y: 3.12, w: 3.8, h: 0.4, fontFace: BF, fontSize: 12, color: MUTE });
card(s, 5.15, 2.2, 4.25, 1.5, "ECFDF5");
s.addText("Small-to-big (shipped)", { x: 5.35, y: 2.32, w: 3.8, h: 0.3, fontFace: HF, fontSize: 14, bold: true, color: INK });
s.addText("recall  0.89 → 0.94", { x: 5.35, y: 2.66, w: 3.8, h: 0.4, fontFace: MONO, fontSize: 16, bold: true, color: GREEN });
s.addText("effect vanished (within noise)", { x: 5.35, y: 3.12, w: 3.8, h: 0.4, fontFace: BF, fontSize: 12, color: MUTE });
card(s, 0.5, 3.95, 9, 1.0, "0B1F3A");
s.addText([
  { text: "Takeaway:  ", options: { bold: true, color: MINT } },
  { text: "a failure mode is a property of a configuration, not of “RAG.” A chunking change masked a 10-point bug. Only re-running the eval revealed it — the case for an eval gate in CI.", options: { color: "E2E8F0" } },
], { x: 0.75, y: 4.05, w: 8.5, h: 0.8, fontFace: BF, fontSize: 13, valign: "middle" });
footer(s, 11);

// ============================================================ 12. THE FIX
s = pres.addSlide();
header(s, "The Fix", "Abstention: refuse what the corpus can’t answer", GREEN);
s.addText("Relevance gate returns no context on out-of-domain queries + a strict “answer only from context” prompt.",
  { x: 0.5, y: 1.45, w: 9, h: 0.45, fontFace: BF, fontSize: 13.5, color: MUTE });
card(s, 1.0, 2.15, 3.7, 2.0, "FEF2F2");
s.addText("FIX OFF", { x: 1.0, y: 2.3, w: 3.7, h: 0.35, fontFace: BF, fontSize: 12, bold: true, color: RED, align: "center" });
s.addText("75%", { x: 1.0, y: 2.62, w: 3.7, h: 0.85, fontFace: HF, fontSize: 54, bold: true, color: RED, align: "center", margin: 0 });
s.addText("of unanswerable questions get confident, fabricated answers", { x: 1.15, y: 3.5, w: 3.4, h: 0.6, fontFace: BF, fontSize: 11.5, color: INK, align: "center" });
card(s, 5.3, 2.15, 3.7, 2.0, "ECFDF5");
s.addText("FIX ON", { x: 5.3, y: 2.3, w: 3.7, h: 0.35, fontFace: BF, fontSize: 12, bold: true, color: GREEN, align: "center" });
s.addText("0%", { x: 5.3, y: 2.62, w: 3.7, h: 0.85, fontFace: HF, fontSize: 54, bold: true, color: GREEN, align: "center", margin: 0 });
s.addText("hallucination on out-of-domain — declines all four", { x: 5.45, y: 3.5, w: 3.4, h: 0.6, fontFace: BF, fontSize: 11.5, color: INK, align: "center" });
s.addText([
  { text: "Cost, named honestly: ", options: { bold: true, color: INK } },
  { text: "1 of 18 in-domain questions (6%) is now over-refused. The safety/coverage tradeoff, made measurable.", options: { color: MUTE } },
], { x: 0.5, y: 4.45, w: 9, h: 0.5, fontFace: BF, fontSize: 12.5, align: "center" });
footer(s, 12);

// ============================================================ 13. EVAL DISAGREES
s = pres.addSlide();
header(s, "Where eval disagrees with judgment", "Why I built a second metric", NAVY);
s.addText("Measuring abstention with Ragas faithfulness gave a number that swung with judge AND config — same behavior, different signal:",
  { x: 0.5, y: 1.45, w: 9, h: 0.55, fontFace: BF, fontSize: 13, color: MUTE });
const fr = [["Haiku judge · flat", "0.50 → 0.18", "big drop", GREEN], ["Sonnet judge · flat", "0.50 → 0.50", "no signal", RED], ["Sonnet judge · small-to-big", "0.75 → 0.21", "big drop", GREEN]];
fr.forEach((r, i) => {
  const y = 2.1 + i * 0.5;
  card(s, 0.5, y, 5.6, 0.44, WHITE);
  s.addText(r[0], { x: 0.65, y, w: 2.7, h: 0.44, fontFace: BF, fontSize: 11.5, color: INK, valign: "middle", margin: 0 });
  s.addText(r[1], { x: 3.35, y, w: 1.6, h: 0.44, fontFace: MONO, fontSize: 12, bold: true, color: NAVY, valign: "middle", align: "center", margin: 0 });
  s.addText(r[2], { x: 4.95, y, w: 1.1, h: 0.44, fontFace: BF, fontSize: 10.5, italic: true, color: r[3], valign: "middle", align: "center", margin: 0 });
});
card(s, 6.3, 2.1, 3.2, 1.44, "0B1F3A");
s.addText("The trap", { x: 6.5, y: 2.2, w: 2.8, h: 0.3, fontFace: HF, fontSize: 13, bold: true, color: MINT });
s.addText("Faithfulness scores answers vs context. Gated OOD has no context → “faithful to nothing” is degenerate.", { x: 6.5, y: 2.52, w: 2.85, h: 1.0, fontFace: BF, fontSize: 11, color: "E2E8F0" });
card(s, 0.5, 3.75, 9, 1.2, "EFF6FF");
s.addText([
  { text: "And relevancy rewards the wrong thing:  ", options: { bold: true, color: NAVY } },
  { text: "turning the fix OFF raises Answer Relevancy 0.00 → 0.50 — a confident “Canberra” reads as relevant; an honest “I don’t know” scores zero. A single metric would rank the unsafe system higher. ", options: { color: INK } },
  { text: "Match the metric to the question.", options: { bold: true, color: NAVY } },
], { x: 0.75, y: 3.85, w: 8.5, h: 1.0, fontFace: BF, fontSize: 12.5, valign: "middle" });
footer(s, 13);

// ============================================================ 14. PRODUCTION
s = pres.addSlide();
header(s, "Production-first", "What breaks at 100k+ users — and what I’d fix first");
const prod = [
  ["Local embeddings", "In-process, single-thread CPU → blocks under load.", "Dedicated embed service (TEI/GPU) + batching + query cache."],
  ["ChromaDB", "Single-node, file-backed, no replication.", "Qdrant / pgvector with HNSW + read replicas."],
  ["Anthropic API", "RPM/TPM limits; spikes → 429s; cost.", "Concurrency limiter + backoff, prompt cache, Haiku routing."],
  ["Streamlit", "Full rerun per click; no auth; in-process state.", "Stateless FastAPI backend + real frontend."],
];
prod.forEach((p, i) => {
  const y = 1.5 + i * 0.72;
  card(s, 0.5, y, 9, 0.64, WHITE);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 0.08, h: 0.64, fill: { color: NAVY } });
  s.addText(p[0], { x: 0.7, y, w: 2.0, h: 0.64, fontFace: HF, fontSize: 12.5, bold: true, color: INK, valign: "middle", margin: 0 });
  s.addText(p[1], { x: 2.75, y, w: 3.3, h: 0.64, fontFace: BF, fontSize: 10.5, color: RED, valign: "middle", margin: 2 });
  s.addText(p[2], { x: 6.1, y, w: 3.3, h: 0.64, fontFace: BF, fontSize: 10.5, color: GREEN, valign: "middle", margin: 2 });
});
s.addText([
  { text: "Highest-leverage fix is organizational: ", options: { bold: true, color: TEAL } },
  { text: "put this eval in CI so quality can’t silently regress (see Break #2).", options: { color: MUTE } },
], { x: 0.5, y: 4.5, w: 9, h: 0.5, fontFace: BF, fontSize: 12.5, align: "center" });
footer(s, 14);

// ============================================================ 15. SCOPE / NEXT
s = pres.addSlide();
header(s, "Scope & next", "What I cut, and where I’d invest next");
card(s, 0.5, 1.55, 4.4, 3.3, WHITE);
s.addText("Deliberately not built", { x: 0.7, y: 1.7, w: 4.0, h: 0.35, fontFace: HF, fontSize: 15, bold: true, color: NAVY });
s.addText([
  "Auth / accounts / chat history",
  "Docker / k8s / CI",
  "Hybrid BM25 + dense search",
  "Multi-hop query decomposition",
  "Synthetic eval data (hand-curated instead)",
  "Multimodal / tables / figures",
].map((t, i, a) => ({ text: t, options: { bullet: true, breakLine: true, color: MUTE } })),
  { x: 0.75, y: 2.1, w: 4.0, h: 2.6, fontFace: BF, fontSize: 12.5, paraSpaceAfter: 6 });
card(s, 5.1, 1.55, 4.4, 3.3, "0B1F3A");
s.addText("What I’d tackle next", { x: 5.3, y: 1.7, w: 4.0, h: 0.35, fontFace: HF, fontSize: 15, bold: true, color: MINT });
s.addText([
  "Grow gold set to 100+ (firmer subsets)",
  "Multi-hop retrieval (lift 0.46 precision)",
  "Sweep child size + learn the gate threshold",
  "Hybrid retrieval for exact-term queries",
  "Eval gate in CI",
].map((t) => ({ text: t, options: { bullet: { type: "number" }, breakLine: true, color: "E2E8F0" } })),
  { x: 5.35, y: 2.1, w: 4.0, h: 2.6, fontFace: BF, fontSize: 12.5, paraSpaceAfter: 7 });
footer(s, 15);

// ============================================================ 16. CLOSING
s = pres.addSlide();
s.background = { color: NAVY };
s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.05, w: 0.16, h: 3.5, fill: { color: TEAL2 } });
s.addText("What I’d defend", { x: 0.7, y: 0.85, w: 9, h: 0.5, fontFace: BF, fontSize: 14, bold: true, color: MINT, charSpacing: 2 });
s.addText([
  { text: "Eval is the product. ", options: { bold: true, color: WHITE } },
  { text: "It caught a 10-pt bug a chunking change had hidden.", options: { color: "CBD5E1" } },
  { text: "\n\nNo single number. ", options: { bold: true, color: WHITE } },
  { text: "Relevancy ranked the unsafe system higher; only slicing + multiple metrics told the truth.", options: { color: "CBD5E1" } },
  { text: "\n\nMatch the metric to the question. ", options: { bold: true, color: WHITE } },
  { text: "Abstention needed a behavioral metric, not faithfulness.", options: { color: "CBD5E1" } },
  { text: "\n\nEvery improvement is a tradeoff. ", options: { bold: true, color: WHITE } },
  { text: "Small-to-big bought faithfulness + speed for precision; abstention bought safety for 6% coverage.", options: { color: "CBD5E1" } },
], { x: 0.7, y: 1.4, w: 8.7, h: 3.1, fontFace: BF, fontSize: 16, valign: "top", lineSpacingMultiple: 1.05 });
s.addText("github.com/NarenBabuR/meraki      ·      Thank you — questions?", { x: 0.7, y: 4.8, w: 9, h: 0.4, fontFace: BF, fontSize: 13, color: "94A3B8" });

pres.writeFile({ fileName: "RAG_Build_Break_Fix.pptx" }).then(f => console.log("WROTE", f));
