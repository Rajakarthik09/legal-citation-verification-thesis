#!/usr/bin/env python3
"""
Generate a Cursor Canvas file for the professor pipeline demo.

Canvas files must embed data inline (no runtime fetch). This script reads the
pre-bundled `code/demo/demo_bundle.json` and writes a generated `.canvas.tsx`
source file under `code/demo/` and into Cursor's managed canvas directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build demo canvas source.")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "demo" / "demo_bundle.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "demo" / "generated_demo_canvas.tsx",
    )
    parser.add_argument(
        "--managed-out",
        type=Path,
        default=None,
        help="Absolute path for Cursor-managed canvas .tsx (auto-detected when omitted).",
    )
    args = parser.parse_args()

    if not args.bundle.exists():
        raise FileNotFoundError(f"Missing demo bundle: {args.bundle}")

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    bundle_literal = json.dumps(bundle, ensure_ascii=False)

    ts_template = r"""import React from "react";
import {
  Stack,
  Row,
  Grid,
  Divider,
  H1,
  H2,
  H3,
  Text,
  Card,
  CardHeader,
  CardBody,
  Stat,
  Select,
  Table,
  BarChart,
  PieChart,
  UsageBar,
  Swatch,
  TodoList,
  Callout,
  CollapsibleSection,
  DiffView,
  DiffStats,
  useCanvasState,
  useHostTheme,
  colorPalette,
  Pill,
} from "cursor/canvas";

const DEMO_BUNDLE = __BUNDLE__ as const;

function pct(v: number | null | undefined) {
  const n = typeof v === "number" ? v : 0;
  return Math.round(n * 1000) / 10;
}

function hrTone(v: number | null | undefined) {
  const n = typeof v === "number" ? v : 0;
  if (n <= 0.2) return "success";
  if (n <= 0.6) return "warning";
  return "danger";
}

function labelToColor(label: string | null | undefined) {
  const l = (label || "").toLowerCase();
  if (l === "verified") return colorPalette.green;
  if (l === "mismatch") return colorPalette.orange;
  if (l === "ambiguous") return colorPalette.yellow;
  if (l === "notfound" || l === "malformed") return colorPalette.red;
  return colorPalette.gray;
}

function labelToRowTone(label: string | null | undefined) {
  const l = (label || "").toLowerCase();
  if (l === "verified") return "success";
  if (l === "mismatch" || l === "notfound" || l === "malformed") return "danger";
  if (l === "ambiguous") return "warning";
  return "neutral";
}

function labelSwatchColor(label: string | null | undefined) {
  const l = (label || "").toLowerCase();
  if (l === "verified") return "green";
  if (l === "mismatch") return "orange";
  if (l === "ambiguous") return "yellow";
  if (l === "notfound" || l === "malformed") return "red";
  return "gray";
}

function outcomeTone(status: string | null | undefined) {
  const s = (status || "").toLowerCase();
  if (s === "verified" || s === "replaced_verified") return "success";
  if (s === "still_bad" || s === "garbled") return "warning";
  if (s === "removed") return "info";
  return "neutral";
}

function highlightText(text: string, citations: readonly any[]) {
  if (!text) return null;
  const spans = (citations || [])
    .filter((c) => Array.isArray(c.span) && c.span.length === 2)
    .map((c) => ({
      label: c.label as string | undefined,
      start: Number(c.span[0]),
      end: Number(c.span[1]),
    }))
    .filter((s) => Number.isFinite(s.start) && Number.isFinite(s.end) && s.end > s.start)
    .sort((a, b) => a.start - b.start);

  if (spans.length === 0) {
    return (
      <Text as="p" style={{ whiteSpace: "pre-wrap", fontFamily: "monospace", fontSize: 12, lineHeight: 1.5 }}>
        {text}
      </Text>
    );
  }

  const parts: React.ReactNode[] = [];
  let cursor = 0;
  for (const sp of spans) {
    if (sp.start >= text.length) break;
    const start = Math.max(0, sp.start);
    const end = Math.min(text.length, Math.max(start, sp.end));
    if (start < cursor) continue;
    if (cursor < start) {
      parts.push(<Text as="span">{text.slice(cursor, start)}</Text>);
    }
    const color = labelToColor(sp.label);
    const isAmb = (sp.label || "").toLowerCase() === "ambiguous";
    parts.push(
      <Text
        as="span"
        style={{
          color,
          textDecoration: "underline",
          textDecorationThickness: isAmb ? 3 : 2,
          textUnderlineOffset: 3,
          backgroundColor: isAmb ? "rgba(234, 179, 8, 0.18)" : undefined,
          fontWeight: isAmb ? 600 : undefined,
        }}
      >
        {text.slice(start, end)}
      </Text>
    );
    cursor = end;
  }
  if (cursor < text.length) {
    parts.push(<Text as="span">{text.slice(cursor)}</Text>);
  }
  return (
    <Text as="p" style={{ whiteSpace: "pre-wrap", fontFamily: "monospace", fontSize: 12, lineHeight: 1.5 }}>
      {parts}
    </Text>
  );
}

function buildDiffLines(draft: string, corrected: string) {
  const a = (draft || "").split("\n");
  const b = (corrected || "").split("\n");
  const n = Math.max(a.length, b.length);
  const lines: { type: "added" | "removed" | "unchanged"; content: string; lineNumber: number }[] = [];
  let shown = 0;
  for (let i = 0; i < n && shown < 40; i++) {
    const left = a[i] ?? "";
    const right = b[i] ?? "";
    if (left === right) {
      if (left.length === 0) continue;
      lines.push({ type: "unchanged", content: left.slice(0, 160), lineNumber: i + 1 });
      shown += 1;
    } else {
      if (left) {
        lines.push({ type: "removed", content: left.slice(0, 160), lineNumber: i + 1 });
        shown += 1;
      }
      if (right) {
        lines.push({ type: "added", content: right.slice(0, 160), lineNumber: i + 1 });
        shown += 1;
      }
    }
  }
  return lines;
}

function LabelGlossary() {
  const rows: { label: string; tone: string; color: string; meaning: string; action: string }[] = [
    {
      label: "Verified",
      tone: "success",
      color: "green",
      meaning:
        "CourtListener found the citation (status 200) and the draft’s parties/year are compatible with that case cluster.",
      action: "Kept as-is — treated as citation-faithful.",
    },
    {
      label: "Mismatch",
      tone: "danger",
      color: "orange",
      meaning:
        "The citation string resolves to a cluster, but metadata does not fully agree (e.g. abbreviated party names or year drift) — without a hard party-name contradiction.",
      action: "Symbolic correction may expand/replace with the CourtListener case name when the identity is safe.",
    },
    {
      label: "Ambiguous",
      tone: "warning",
      color: "yellow",
      meaning:
        "Either CourtListener returned multiple possible clusters (status 300), or the cite resolves to a cluster whose parties clearly contradict the draft case name (wrong identity, e.g. Brown vs an unrelated LEXIS hit).",
      action:
        "Left highlighted in the text (counts as still_bad). If another Verified same-case cite exists in the summary, we re-anchor to that cite — we never substitute the contradicting CL name.",
    },
    {
      label: "NotFound",
      tone: "danger",
      color: "red",
      meaning: "CourtListener returned 404 — no matching citation in the database (often a fabricated reporter cite).",
      action: "Redacted as [unverifiable citation removed].",
    },
    {
      label: "Malformed",
      tone: "danger",
      color: "red",
      meaning: "CourtListener could not parse the string as a valid reporter citation (status 400) or the response was unexpected.",
      action: "Redacted like NotFound.",
    },
  ];

  return (
    <Card>
      <CardHeader>What each verification label means</CardHeader>
      <CardBody>
        <Stack gap={10}>
          <Text size="small" tone="secondary">
            Module B (eyecite) extracts citations; Module C looks each one up on CourtListener and assigns a label.
            Hallucination rate treats everything except Verified as bad.
          </Text>
          {rows.map((r) => (
            <Stack key={r.label} gap={4} style={{ paddingTop: 4, paddingBottom: 4 }}>
              <Row gap={8} style={{ alignItems: "center" }}>
                <Swatch color={r.color as any} />
                <Pill size="sm" tone={r.tone as any} active>
                  {r.label}
                </Pill>
              </Row>
              <Text size="small">
                <Text weight="medium">Meaning: </Text>
                {r.meaning}
              </Text>
              <Text size="small" tone="secondary">
                <Text weight="medium">In this pipeline: </Text>
                {r.action}
              </Text>
            </Stack>
          ))}
        </Stack>
      </CardBody>
    </Card>
  );
}

function Dashboard() {
  const metrics = DEMO_BUNDLE.aggregate.metrics_by_condition;
  const rq = DEMO_BUNDLE.aggregate.rq || {};
  const rq2 = rq.RQ2 || {};
  const rq3 = rq.RQ3 || {};
  const baseline = metrics.baseline?.hr_paired_micro ?? metrics.baseline?.hr_micro ?? 0;
  const symbolic = metrics.symbolic?.hr_paired_micro ?? metrics.symbolic?.hr_micro ?? 0;
  const llm = metrics.llm_rewrite?.hr_paired_micro ?? metrics.llm_rewrite?.hr_micro ?? 0;
  const rel = rq2.reduction?.relative_reduction_pct ?? ((baseline - symbolic) / (baseline || 1)) * 100;
  const sym = metrics.symbolic || {};

  return (
    <Stack gap={14}>
      <Callout tone="info" title={String(DEMO_BUNDLE.meta.presentation_label || "Pipeline demo")}>
        Dual-system legal summarization: PRIMERA draft → eyecite extraction → CourtListener verify → symbolic correction.
        Primary metric is paired draft-citation HR. Ambiguous cites are left unresolved and highlighted (not redacted),
        unless a verified same-case cite can re-anchor them.
      </Callout>

      <LabelGlossary />

      <Grid columns={3} gap={12}>
        <Stat value={`${pct(baseline)}%`} label="Draft paired HR" tone={hrTone(baseline) as any} />
        <Stat value={`${pct(symbolic)}%`} label="Symbolic paired HR" tone={hrTone(symbolic) as any} />
        <Stat
          value={`${Math.round(rel * 10) / 10}%`}
          label="Relative HR reduction"
          tone={rel >= 30 ? "success" : "warning"}
        />
      </Grid>

      <Card>
        <CardHeader>Paired hallucination rate by condition</CardHeader>
        <CardBody>
          <BarChart
            categories={["Draft", "Symbolic", "LLM rewrite"]}
            series={[
              {
                name: "Paired HR (%)",
                data: [pct(baseline), pct(symbolic), pct(llm)],
                tone: "danger",
              },
            ]}
            valueSuffix="%"
            showValues
            beginAtZero
            yMax={100}
            referenceLines={[{ value: 30, label: "RQ3 target cut", tone: "info" }]}
          />
          <Text size="small" tone="tertiary" style={{ marginTop: 8 }}>
            Source: full_pipeline_results · n={DEMO_BUNDLE.meta.n_cases} citation-rich validation cases · D→L
          </Text>
        </CardBody>
      </Card>

      <Card>
        <CardHeader>Correction outcome mix (symbolic)</CardHeader>
        <CardBody>
          <UsageBar
            segments={[
              { id: "fixed", value: Math.round((sym.fix_rate || 0) * 1000) / 10, color: "green" },
              { id: "still_bad", value: Math.round((sym.still_bad_rate || 0) * 1000) / 10, color: "yellow" },
              { id: "removed", value: Math.round((sym.removal_rate || 0) * 1000) / 10, color: "blue" },
              { id: "garbled", value: Math.round((sym.garbled_rate || 0) * 1000) / 10, color: "red" },
            ]}
            total={100}
            topLeftLabel="Share of draft citations"
            topRightLabel={`${sym.total_draft_citations ?? 7} draft cites`}
          />
          <Row gap={8} wrap style={{ marginTop: 10 }}>
            <Pill size="sm" tone="success" active>fix {pct(sym.fix_rate)}%</Pill>
            <Pill size="sm" tone="warning" active>still_bad {pct(sym.still_bad_rate)}%</Pill>
            <Pill size="sm" tone="info">removed {pct(sym.removal_rate)}%</Pill>
            <Pill size="sm" tone="danger">garbled {pct(sym.garbled_rate)}%</Pill>
          </Row>
        </CardBody>
      </Card>
    </Stack>
  );
}

function ResearchQuestions() {
  const rq = DEMO_BUNDLE.aggregate.rq || {};
  const rq1 = rq.RQ1 || {};
  const rq2 = rq.RQ2 || {};
  const rq3 = rq.RQ3 || {};
  const labels = rq1.label_counts || {};
  const pieData = Object.keys(labels).map((k) => ({ label: k, value: Number(labels[k]) || 0 }));
  const mcnemar = rq2.mcnemar_test || {};

  return (
    <Stack gap={12}>
      <H2>How we answer the research questions</H2>

      <Card>
        <CardHeader trailing={<Pill size="sm" tone="danger" active>HR {pct(rq1.hallucination_rate)}%</Pill>}>
          RQ1 — What hallucinations does the baseline produce?
        </CardHeader>
        <CardBody>
          <Text weight="medium">{rq1.question}</Text>
          <Text size="small" tone="secondary" style={{ marginTop: 6 }}>
            Method: extract FullCaseCitation spans with eyecite → CourtListener citation-lookup → label
            Verified / Mismatch / Ambiguous / NotFound / Malformed (see label glossary above).
            HR = bad labels / all draft cites.
          </Text>
          <Grid columns={2} gap={16} style={{ marginTop: 12 }}>
            {pieData.length > 0 ? (
              <PieChart data={pieData} donut size={200} />
            ) : null}
            <Stack gap={8}>
              <Stat value={String(rq1.total_citations ?? 0)} label="Draft citations (n=10 cases)" />
              <Text size="small">{rq1.summary}</Text>
              <Row gap={8} wrap>
                {Object.entries(labels).map(([k, v]) => (
                  <Pill size="sm" tone={labelToRowTone(k) as any}>
                    {k}: {String(v)}
                  </Pill>
                ))}
              </Row>
            </Stack>
          </Grid>
        </CardBody>
      </Card>

      <Card>
        <CardHeader trailing={<Pill size="sm" tone="success" active>−{pct((rq2.reduction?.relative_reduction_pct || 0) / 100)}% rel</Pill>}>
          RQ2 — Does symbolic verify+correct reduce HR?
        </CardHeader>
        <CardBody>
          <Text weight="medium">{rq2.question}</Text>
          <Text size="small" tone="secondary" style={{ marginTop: 6 }}>
            Method: paired draft-citation HR — track each draft cite after correction
            (replaced_verified / still_bad / removed / verified). Ambiguous cites are left in text and count as still_bad.
            McNemar pairs by citation_id across all draft cites.
          </Text>
          <Grid columns={3} gap={12} style={{ marginTop: 12 }}>
            <Stat value={`${pct(rq2.hr_baseline_paired_micro)}%`} label="Baseline paired HR" tone={hrTone(rq2.hr_baseline_paired_micro) as any} />
            <Stat value={`${pct(rq2.hr_symbolic_paired_micro)}%`} label="Symbolic paired HR" tone={hrTone(rq2.hr_symbolic_paired_micro) as any} />
            <Stat value={`${pct(rq2.hr_symbolic_surviving_micro)}%`} label="Surviving micro HR" tone="info" />
          </Grid>
          <Callout tone="warning" title="McNemar (underpowered at n=7 cites)" style={{ marginTop: 12 }}>
            Paired citations: {String(mcnemar.n_paired_citations ?? 0)} ·
            draft→success: {String(mcnemar.draft_hallucinated_final_success ?? mcnemar.draft_only_hallucinated ?? 0)} ·
            p={String(mcnemar.p_value)} · {String(mcnemar.interpretation || "")}
          </Callout>
          <Text size="small" style={{ marginTop: 8 }}>{rq2.summary}</Text>
        </CardBody>
      </Card>

      <Card>
        <CardHeader trailing={
          <Pill size="sm" tone={rq3.hr_target_met && rq3.rouge_target_met ? "success" : "warning"} active>
            {rq3.hr_target_met && rq3.rouge_target_met ? "PASS" : "CHECK"}
          </Pill>
        }>
          RQ3 — ≥30% HR cut without hurting ROUGE-L?
        </CardHeader>
        <CardBody>
          <Text weight="medium">{rq3.question}</Text>
          <TodoList
            style={{ marginTop: 10 }}
            todos={[
              {
                id: "hr",
                title: `HR relative reduction ≥ ${rq3.hr_target_pct}% (got ${rq3.hr_relative_reduction_pct}%)`,
                status: rq3.hr_target_met ? "completed" : "cancelled",
              },
              {
                id: "rouge",
                title: `ROUGE-L within ±${rq3.rouge_tolerance_pct} pp (delta ${rq3.rougeL_delta_pct_points} pp)`,
                status: rq3.rouge_target_met ? "completed" : "cancelled",
              },
              {
                id: "llm",
                title: "Symbolic vs LLM rewrite: HR tied; ROUGE nearly identical on this n=10 set",
                status: "completed",
              },
            ]}
          />
          <Grid columns={2} gap={12} style={{ marginTop: 12 }}>
            <Stat
              value={`${rq3.hr_relative_reduction_pct}%`}
              label="Paired HR relative reduction"
              tone={rq3.hr_target_met ? "success" : "danger"}
            />
            <Stat
              value={`${rq3.rougeL_delta_pct_points} pp`}
              label="ROUGE-L delta (symbolic − draft)"
              tone={rq3.rouge_target_met ? "success" : "warning"}
            />
          </Grid>
          <Text size="small" style={{ marginTop: 8 }}>{rq3.summary}</Text>
        </CardBody>
      </Card>
    </Stack>
  );
}

function CaseExplorer() {
  const cases = DEMO_BUNDLE.cases as readonly any[];
  const defaultId =
    cases.find((c) => c.case_id === "SD-KS-0001")?.case_id ||
    cases[0]?.case_id ||
    "";
  const [caseId, setCaseId] = useCanvasState("caseId", defaultId);
  const selected = cases.find((c) => c.case_id === caseId) || cases[0];
  const citationsDraft = selected?.citations_draft || [];
  const citationsSymbolic = selected?.citations_symbolic || [];
  const outcomes = selected?.correction_outcomes || [];

  const caseOptions = cases
    .map((c: any) => ({
      value: c.case_id,
      label: `${c.case_id} · ${pct(c.hr_draft)}% → ${pct(c.hr_symbolic)}%`,
    }))
    .sort((a: any, b: any) => a.label.localeCompare(b.label));

  const outcomeById: Record<string, string> = {};
  for (const o of outcomes) {
    if (o?.citation_id) outcomeById[o.citation_id] = o.final_status || "";
  }

  const draftRows = citationsDraft.map((c: any) => [
    c.raw_text || "",
    (
      <Row gap={8} align="center">
        <Swatch color={labelSwatchColor(c.label) as any} />
        <Text size="small">{c.label || ""}</Text>
      </Row>
    ),
    c.matched_case_name || "",
    outcomeById[c.citation_id] || "—",
    <Text size="small" truncate="end" style={{ maxWidth: 420 }}>{String(c.explanation || "")}</Text>,
  ]);
  const draftTone = citationsDraft.map((c: any) => labelToRowTone(c.label) as any);

  const symRows = citationsSymbolic.map((c: any) => [
    c.raw_text || "",
    (
      <Row gap={8} align="center">
        <Swatch color={labelSwatchColor(c.label) as any} />
        <Text size="small">{c.label || ""}</Text>
      </Row>
    ),
    c.matched_case_name || "",
    "—",
    <Text size="small" truncate="end" style={{ maxWidth: 420 }}>{String(c.explanation || "")}</Text>,
  ]);
  const symTone = citationsSymbolic.map((c: any) => labelToRowTone(c.label) as any);

  const headers = ["Citation", "Label", "Matched case", "Outcome", "Explanation"];
  const improved =
    typeof selected?.hr_draft === "number" &&
    typeof selected?.hr_symbolic === "number" &&
    selected.hr_symbolic < selected.hr_draft;

  const hasAmbiguous = citationsSymbolic.some((c: any) => c.label === "Ambiguous") ||
    citationsDraft.some((c: any) => c.label === "Ambiguous" && outcomeById[c.citation_id] === "still_bad");

  const diffLines = buildDiffLines(selected?.draft_summary || "", selected?.symbolic_summary || "");
  const additions = diffLines.filter((l) => l.type === "added").length;
  const deletions = diffLines.filter((l) => l.type === "removed").length;

  return (
    <Stack gap={14}>
      <H2>Case explorer</H2>
      <Row gap={8} wrap>
        <Pill size="sm" active>Default: SD-KS-0001</Pill>
        <Pill size="sm" tone="warning">PB-CA-0016: Ambiguous left highlighted</Pill>
        <Pill size="sm" tone="info">Yellow = Ambiguous (unresolved)</Pill>
      </Row>

      <Select
        value={caseId}
        onChange={(v: string) => setCaseId(v)}
        options={caseOptions as any}
      />

      {selected?.case_id === "PB-CA-0016" ? (
        <Callout tone="warning" title="Hard case — incomplete repair">
          Mismatches get expanded to full CourtListener case names; Ambiguous reporter cites stay in the
          summary and keep paired HR above zero for this case.
        </Callout>
      ) : null}
      {selected?.case_id === "SD-KS-0001" ? (
        <Callout tone="info" title="Showcase case">
          Reporter-only Brown cite verifies cleanly; LEXIS mismatch is replaced. Live Streamlit app runs
          extract → verify → correct for this case on demand.
        </Callout>
      ) : null}

      <Grid columns={3} gap={12}>
        <Stat value={`${pct(selected?.hr_draft)}%`} label="Draft paired HR" tone={hrTone(selected?.hr_draft) as any} />
        <Stat value={`${pct(selected?.hr_symbolic)}%`} label="Symbolic paired HR" tone={hrTone(selected?.hr_symbolic) as any} />
        <Stat
          value={improved ? "Improved" : "No drop"}
          label="Correction effect"
          tone={improved ? "success" : "neutral"}
        />
      </Grid>

      <TodoList
        todos={[
          { id: "gen", title: "Generate draft (PRIMERA D→L)", status: "completed" },
          { id: "ext", title: `Extract citations (${citationsDraft.length} found)`, status: "completed" },
          { id: "ver", title: "Verify via CourtListener", status: "completed" },
          {
            id: "cor",
            title: hasAmbiguous
              ? "Correct mismatches; leave Ambiguous highlighted"
              : "Symbolic correction",
            status: "completed",
          },
          { id: "eval", title: "Evaluate paired HR + ROUGE", status: "completed" },
        ]}
      />

      <Row gap={8} wrap>
        <Text size="small" tone="tertiary">Highlight colors:</Text>
        <Pill size="sm" tone="success"><Swatch color="green" /> Verified — cite + metadata OK</Pill>
        <Pill size="sm" tone="danger"><Swatch color="orange" /> Mismatch — resolves, metadata off</Pill>
        <Pill size="sm" tone="warning"><Swatch color="yellow" /> Ambiguous — multi-hit or wrong identity</Pill>
        <Pill size="sm" tone="danger"><Swatch color="red" /> NotFound / Malformed — redact</Pill>
      </Row>
      <Text size="small" tone="tertiary">
        Full definitions are in “What each verification label means” at the top of this canvas.
      </Text>

      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader>Draft summary (highlighted)</CardHeader>
          <CardBody>{highlightText(selected?.draft_summary || "", citationsDraft)}</CardBody>
        </Card>
        <Card>
          <CardHeader>Symbolic summary (highlighted)</CardHeader>
          <CardBody>{highlightText(selected?.symbolic_summary || "", citationsSymbolic)}</CardBody>
        </Card>
      </Grid>

      {diffLines.length > 0 && additions + deletions > 0 ? (
        <Card collapsible defaultOpen={selected?.case_id === "PB-CA-0016" || selected?.case_id === "SD-KS-0001"}>
          <CardHeader trailing={<DiffStats additions={additions} deletions={deletions} />}>
            Draft → symbolic line diff (excerpt)
          </CardHeader>
          <CardBody style={{ padding: 0 }}>
            <DiffView path="summary.txt" language="text" lines={diffLines as any} />
          </CardBody>
        </Card>
      ) : null}

      {citationsDraft.length > 0 ? (
        <Card>
          <CardHeader>Draft citations + correction outcomes</CardHeader>
          <CardBody>
            <Table headers={headers as any} rows={draftRows} rowTone={draftTone as any} framed stickyHeader={false} />
          </CardBody>
        </Card>
      ) : (
        <Callout tone="neutral" title="No draft citations">
          This case produced no eyecite FullCaseCitation spans (common for under-citing drafts).
        </Callout>
      )}

      {citationsSymbolic.length > 0 ? (
        <Card>
          <CardHeader>Symbolic final citations</CardHeader>
          <CardBody>
            <Table headers={headers as any} rows={symRows} rowTone={symTone as any} framed stickyHeader={false} />
          </CardBody>
        </Card>
      ) : null}
    </Stack>
  );
}

export default function ThesisPipelineDemo() {
  const theme = useHostTheme();
  const limitations = (DEMO_BUNDLE.aggregate.rq?.limitations || []) as string[];

  return (
    <Stack gap={20} style={{ padding: 16, color: theme.text.primary }}>
      <Stack gap={6}>
        <H1>Citation-faithful summarization demo</H1>
        <Text tone="secondary">
          Professor walkthrough · dual-system verify + correct · n=10 citation-rich validation
        </Text>
      </Stack>

      <Dashboard />
      <Divider />
      <ResearchQuestions />
      <Divider />
      <CaseExplorer />
      <Divider />

      <CollapsibleSection title="Limitations & honest caveats" count={limitations.length} defaultOpen>
        <Stack gap={8}>
          <Callout tone="neutral" title="Read before claiming 0% HR">
            Ambiguous citations are intentionally left in the summary and counted as still_bad.
            Removals only apply to NotFound/Malformed. One LEXIS mismatch can still be “repaired”
            to a wrong CourtListener cluster (see SD-KS-0001) — treat as a known failure mode.
          </Callout>
          {limitations.map((lim) => (
            <Text size="small">• {lim}</Text>
          ))}
        </Stack>
      </CollapsibleSection>
    </Stack>
  );
}
"""

    ts = ts_template.replace("__BUNDLE__", bundle_literal)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(ts, encoding="utf-8")
    print(f"Wrote generated canvas source: {args.out}")

    managed_out: Path | None = args.managed_out
    if managed_out is None:
        projects_root = Path.home() / ".cursor" / "projects"
        candidates = list(projects_root.glob("**/Desktop-Thesis/canvases/tsconfig.json"))
        if not candidates:
            candidates = list(projects_root.glob("**/*Desktop-Thesis*/canvases/tsconfig.json"))
        if candidates:
            managed_out = candidates[0].parent / "thesis-pipeline-demo.canvas.tsx"

    if managed_out is not None:
        managed_out.parent.mkdir(parents=True, exist_ok=True)
        managed_out.write_text(ts, encoding="utf-8")
        print(f"Wrote managed canvas: {managed_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
