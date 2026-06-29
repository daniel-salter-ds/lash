//! GEPA harness adapter for the OBLIQ-Bench `math` subset.
//!
//! Drives `lash-oblique run-batch` as a subprocess per (candidate, example) pair.
//! Reads the per-task JSON output for the fitness signal and the `.trace.jsonl`
//! for ASI construction.
//!
//! Raw document text is intentionally omitted from the reflective record because
//! including doc snippets would exceed the token budget. Query strings and gold
//! coverage statistics fully distinguish all four failure modes.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use lash_trace::{TraceContext, TraceEvent, TraceRecord};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::{
    Candidate, ComponentConstraints, ComponentValue, EvaluationResult, ExampleRun,
    HarnessExample, HarnessOptError, HarnessOptStore, HarnessProject, MutableComponent,
    OptimizationRun, Result, RunArtifacts, Split, TraceBundle,
};
use crate::gepa_summary::write_gepa_summary;

// ---------------------------------------------------------------------------
// Dataset split
// ---------------------------------------------------------------------------

const TRAIN_IDS: &[&str] = &[
    "q02193", "q01486", "q01066", "q02834", "q01757", "q02979", "q01298",
];

const VAL_IDS: &[&str] = &["q00844", "q00847", "q01488"];

// ---------------------------------------------------------------------------
// Component ID and seed prompt
// ---------------------------------------------------------------------------

const OBLIQ_TASK_INSTRUCTIONS_COMPONENT_ID: &str = "obliq.task_instructions";

const OBLIQ_SEED_PROMPT: &str = r#"# Task
Find documents that share the same abstract relationship, structure, or latent pattern as the query, even when surface topics or vocabulary differ.

Use multiple independent search angles to build a broad candidate pool. Calibrate relevance before finalising the ranking.

# Tools
- `search` — retrieve candidates by keyword, semantic, or hybrid query (pass 2–6 independent query strings per call)
- `discover_docs` — find similar documents given positive/negative example pairs (use after finding at least one likely positive)
- `judge_candidates` — assess 10–50 candidate document IDs against a verifier predicate; returns likely positives, distractors, and refined queries
- `tournament_rerank` — merge all candidate pools into a final ranked list (call once, with all pools, at the end)

# Workflow
1. Formulate a verifier predicate: what property must a relevant document demonstrate?
2. Search from several independent angles. Keep each result set as a separate pool.
3. Sample 20–50 candidate IDs across pools and call `judge_candidates` to identify likely positives.
4. Optionally use `discover_docs` if you have clear positive/negative text examples.
5. Build `candidate_pools` from all retrieval calls and call `tournament_rerank` once to produce the final ranking."#;

// ---------------------------------------------------------------------------
// Binary path detection
// ---------------------------------------------------------------------------

/// Locate the lash-oblique release binary.
///
/// Checks `CARGO_TARGET_DIR` first (set by some IDEs to redirect build output),
/// then falls back to the conventional `target/release/` path relative to the
/// lash-oblique workspace directory.
pub fn detect_lash_oblique_bin(lash_oblique_dir: &Path) -> PathBuf {
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        PathBuf::from(target_dir).join("release").join("lash-oblique")
    } else {
        lash_oblique_dir
            .join("target")
            .join("release")
            .join("lash-oblique")
    }
}

// ---------------------------------------------------------------------------
// Qrels TSV parsing
// ---------------------------------------------------------------------------

/// Parse a qrels TSV string into a nested map of `query_id → doc_id → relevance`.
///
/// Accepts 3-column format: `query-id\tdoc-id\tscore`. Header lines (where the
/// first column is literally `"query-id"`) and comment lines (starting with `#`)
/// are silently skipped.
pub fn parse_qrels_tsv(content: &str) -> HashMap<String, HashMap<String, f64>> {
    let mut map: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 3 {
            continue;
        }
        let query_id = cols[0].trim();
        if query_id == "query-id" {
            continue;
        }
        let doc_id = cols[1].trim();
        let Ok(rel) = cols[2].trim().parse::<f64>() else {
            continue;
        };
        map.entry(query_id.to_string())
            .or_default()
            .insert(doc_id.to_string(), rel);
    }
    map
}

// ---------------------------------------------------------------------------
// Gold coverage metric
// ---------------------------------------------------------------------------

/// Fraction of gold documents found in the top-`k` ranked results.
///
/// Returns `0.0` when `gold` is empty (no gold documents means no coverage
/// to measure) or when `ranked_ids` is empty.
pub fn gold_coverage(ranked_ids: &[String], gold: &HashMap<String, f64>, k: usize) -> f64 {
    if gold.is_empty() {
        return 0.0;
    }
    let found = ranked_ids.iter().take(k).filter(|id| gold.contains_key(*id)).count();
    found as f64 / gold.len() as f64
}

// ---------------------------------------------------------------------------
// ASI feedback builder
// ---------------------------------------------------------------------------

/// Tool names the Lashlang agent may invoke in OBLIQ-Bench traces.
const LASHLANG_TOOL_NAMES: &[&str] =
    &["search", "judge_candidates", "tournament_rerank", "discover_docs"];

/// Build a structured feedback string for the reflection LM from one evaluated example.
///
/// Scans `ProtocolStep` records emitted by the `runtime` plugin. Each such record contains
/// the fully assembled Lashlang code that was executed in one protocol iteration. Tool
/// invocations appear as `tools.TOOLNAME(` patterns in this code, and search queries are
/// extracted from the `queries: [...]` array literals within each `tools.search(` call.
///
/// Tool-level LLM failures are additionally detected from `LlmCallFailed` events whose
/// `caused_by.call_id` context-metadata encodes the tool name in the format
/// `lashlang:<session>:resource:tool:<TOOL_NAME>:resource_operation:...`.
///
/// Computes gold document coverage, and formats a concise diagnostic string stored in
/// `EvaluationResult.feedback` and surfaced to the reflection LLM via
/// `render_reflective_evidence`.
///
/// Token budget: search queries are truncated to the first 8; submitted doc IDs to the first 10;
/// final agent text to 300 characters.
pub fn build_asi_feedback(
    trace_records: &[TraceRecord],
    ranked_doc_ids: &[String],
    gold: &HashMap<String, f64>,
    ndcg_at_10: f64,
    final_agent_text: &str,
) -> String {
    let mut search_queries: Vec<String> = Vec::new();
    let mut tool_call_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut tool_failure_counts: BTreeMap<String, usize> = BTreeMap::new();

    for record in trace_records {
        match &record.event {
            // Primary detection path for Lashlang (RLM mode) traces.
            //
            // ProtocolStep records from the runtime plugin carry the fully assembled
            // Lashlang code block in payload["diagnostic"]["payload"]["code"].
            // Tool invocations appear literally as `tools.TOOLNAME(` in this text.
            TraceEvent::ProtocolStep { plugin_id, payload } if plugin_id == "runtime" => {
                if let Some(code) = payload
                    .get("diagnostic")
                    .and_then(|d| d.get("payload"))
                    .and_then(|p| p.get("code"))
                    .and_then(|c| c.as_str())
                {
                    for tool in LASHLANG_TOOL_NAMES {
                        let pattern = format!("tools.{tool}(");
                        let count = code.matches(pattern.as_str()).count();
                        if count > 0 {
                            *tool_call_counts.entry((*tool).to_string()).or_insert(0) += count;
                        }
                    }
                    if search_queries.len() < 8 {
                        extract_search_queries_from_lashlang(code, &mut search_queries);
                    }
                }
            }

            // Tool-level failure detection.
            //
            // When a tool's internal LLM calls fail, `LlmCallFailed` events are emitted
            // with a `caused_by.call_id` in context metadata encoded as:
            // `lashlang:<session>:resource:tool:<TOOL_NAME>:resource_operation:<hash>:<n>`
            TraceEvent::LlmCallFailed { .. } => {
                if let Some(call_id) = record
                    .context
                    .metadata
                    .get("caused_by")
                    .and_then(|c| c.get("call_id"))
                    .and_then(|c| c.as_str())
                {
                    if let Some(tool_name) = extract_tool_from_caused_by_call_id(call_id) {
                        *tool_failure_counts.entry(tool_name).or_insert(0) += 1;
                    }
                }
            }

            _ => {}
        }
    }

    let total_search_calls = tool_call_counts.get("search").copied().unwrap_or(0);

    // Format tool call summary line.
    let tool_summary = if tool_call_counts.is_empty() {
        "none — agent submitted without using retrieval tools".to_string()
    } else {
        tool_call_counts
            .iter()
            .map(|(name, count)| format!("{name}×{count}"))
            .collect::<Vec<_>>()
            .join(", ")
    };

    // Format optional tool-failure line (omitted when there are no failures).
    let failure_line = if tool_failure_counts.is_empty() {
        String::new()
    } else {
        let summary = tool_failure_counts
            .iter()
            .map(|(name, count)| format!("{name}×{count}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("Tool failures (internal LLM errors): {summary}\n")
    };

    // Format search queries section.
    let search_section = if total_search_calls == 0 {
        "  (none — agent made no search calls)".to_string()
    } else {
        let mut lines: Vec<String> = search_queries
            .iter()
            .enumerate()
            .map(|(i, q)| {
                let truncated: String = q.chars().take(100).collect();
                format!("  {}. \"{truncated}\"", i + 1)
            })
            .collect();
        if total_search_calls > 8 {
            lines.push(format!(
                "  ... ({total_search_calls} total, showing first 8)"
            ));
        }
        lines.join("\n")
    };

    // Gold coverage: how many gold docs appear in the top-100 submitted list.
    let gold_total = gold.len();
    let gold_in_top100 = ranked_doc_ids
        .iter()
        .take(100)
        .filter(|id| gold.contains_key(*id))
        .count();

    // Top-10 submitted doc IDs for quick inspection.
    let top_docs_str = if ranked_doc_ids.is_empty() {
        "(none submitted)".to_string()
    } else {
        let top: Vec<&str> = ranked_doc_ids.iter().take(10).map(String::as_str).collect();
        format!("[{}]", top.join(", "))
    };

    // Final agent text (last LLM response), truncated to 300 chars.
    let agent_text_display = if final_agent_text.is_empty() {
        "(empty)".to_string()
    } else {
        let char_count = final_agent_text.chars().count();
        let truncated: String = final_agent_text.chars().take(300).collect();
        if char_count > 300 {
            format!("{truncated}...")
        } else {
            truncated
        }
    };

    format!(
        "Score: NDCG@10={ndcg_at_10:.3}\n\
         Tool calls: {tool_summary}\n\
         {failure_line}\
         Search queries issued:\n{search_section}\n\
         Gold coverage: {gold_in_top100}/{gold_total} gold documents in top 100\n\
         Top submitted docs: {top_docs_str}\n\
         Final agent text: {agent_text_display}"
    )
}

/// Extract search queries from Lashlang code containing `tools.search({queries: [...]})` calls.
///
/// Splits on each `tools.search(` occurrence, finds the `queries: [...]` array in the
/// subsequent text, and extracts each double-quoted string. Appends to `out` until it
/// reaches 8 entries (the token-budget cap).
fn extract_search_queries_from_lashlang(code: &str, out: &mut Vec<String>) {
    // Each split segment starts right after a `tools.search(` occurrence.
    for part in code.split("tools.search(").skip(1) {
        if out.len() >= 8 {
            break;
        }
        // Locate the `queries:` key.
        let Some(queries_pos) = part.find("queries:") else {
            continue;
        };
        let after_key = &part[queries_pos + "queries:".len()..];
        let trimmed = after_key.trim_start_matches(|c: char| c.is_whitespace());
        if !trimmed.starts_with('[') {
            continue;
        }
        let bracket_content = &trimmed[1..];
        // The first `]` closes the queries array (query strings never contain `]`).
        let Some(end) = bracket_content.find(']') else {
            continue;
        };
        let array_str = &bracket_content[..end];
        // Extract every double-quoted string from the array literal.
        let mut s = array_str;
        while let Some(q_start) = s.find('"') {
            s = &s[q_start + 1..];
            let Some(q_end) = s.find('"') else { break };
            let query = &s[..q_end];
            if !query.is_empty() && out.len() < 8 {
                out.push(query.to_string());
            }
            s = &s[q_end + 1..];
        }
    }
}

/// Parse the tool name out of a Lashlang `caused_by.call_id` string.
///
/// Expected format:
/// `lashlang:<session_id>:resource:tool:<TOOL_NAME>:resource_operation:<hash>:<n>`
///
/// Returns `None` if the string does not match this pattern.
fn extract_tool_from_caused_by_call_id(call_id: &str) -> Option<String> {
    const MARKER: &str = ":resource:tool:";
    let pos = call_id.find(MARKER)?;
    let after = &call_id[pos + MARKER.len()..];
    let end = after.find(':').unwrap_or(after.len());
    let tool_name = &after[..end];
    if tool_name.is_empty() { None } else { Some(tool_name.to_string()) }
}

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// GEPA harness adapter for the OBLIQ-Bench `math` subset.
pub struct ObliqHarnessProject {
    /// Absolute path to the pre-compiled `lash-oblique` binary.
    pub lash_oblique_bin: PathBuf,
    /// Absolute path to the lash-oblique source directory (contains `scripts/`).
    /// The subprocess is run with this as its working directory so it can
    /// locate `scripts/query_math_qdrant.py`.
    pub lash_oblique_dir: PathBuf,
    /// Root data directory for qrels and queries (e.g. `.benchmarks/obliq/data`).
    pub data_dir: PathBuf,
    /// LLM model identifier passed to the subprocess (e.g. `"anthropic/claude-haiku-4.5"`).
    pub model: String,
    /// Optional model variant passed to the subprocess.
    pub variant: Option<String>,
    /// Pre-loaded qrels: `query_id → doc_id → relevance`.
    pub qrels: HashMap<String, HashMap<String, f64>>,
    /// Query texts pre-loaded from `queries.jsonl`: `query_id → text`.
    pub queries: HashMap<String, String>,
    /// Terms that must appear in every candidate's task instructions.
    pub preserve_terms: Vec<String>,
}

impl ObliqHarnessProject {
    /// Construct an `ObliqHarnessProject`, loading qrels and query texts from disk.
    ///
    /// - `lash_oblique_bin`: path to the pre-compiled binary (from `detect_lash_oblique_bin`)
    /// - `lash_oblique_dir`: path to the lash-oblique source directory (must contain `scripts/`)
    /// - `data_dir`: root data directory (e.g. `lash_oblique_dir/.benchmarks/obliq/data`)
    /// - `model`: model slug (e.g. `"anthropic/claude-haiku-4.5"`)
    /// - `variant`: optional variant string
    /// - `preserve_terms`: terms that every candidate's instructions must contain
    ///
    /// Reads `data_dir/math/queries.jsonl` and
    /// `data_dir/math/qrels_pool.tsv` (falling back to `qrels.tsv`).
    pub fn new(
        lash_oblique_bin: PathBuf,
        lash_oblique_dir: PathBuf,
        data_dir: PathBuf,
        model: String,
        variant: Option<String>,
        preserve_terms: Vec<String>,
    ) -> Result<Self> {
        let queries = load_queries(&data_dir)?;
        let qrels = load_qrels(&data_dir)?;
        Ok(Self {
            lash_oblique_bin,
            lash_oblique_dir,
            data_dir,
            model,
            variant,
            qrels,
            queries,
            preserve_terms,
        })
    }
}

fn load_queries(data_dir: &Path) -> Result<HashMap<String, String>> {
    let path = data_dir.join("math").join("queries.jsonl");
    let content = std::fs::read_to_string(&path)
        .map_err(|e| HarnessOptError::Harness(format!("read {}: {e}", path.display())))?;
    let mut queries: HashMap<String, String> = HashMap::new();
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: Value = serde_json::from_str(line).map_err(|e| {
            HarnessOptError::Harness(format!("parse queries.jsonl line {}: {e}", i + 1))
        })?;
        let id = record["_id"]
            .as_str()
            .ok_or_else(|| {
                HarnessOptError::Harness(format!(
                    "queries.jsonl line {}: missing '_id' field",
                    i + 1
                ))
            })?
            .to_string();
        let text = record["text"]
            .as_str()
            .ok_or_else(|| {
                HarnessOptError::Harness(format!(
                    "queries.jsonl line {}: missing 'text' field for id='{id}'",
                    i + 1
                ))
            })?
            .to_string();
        queries.insert(id, text);
    }
    Ok(queries)
}

fn load_qrels(data_dir: &Path) -> Result<HashMap<String, HashMap<String, f64>>> {
    let pool_path = data_dir.join("math").join("qrels_pool.tsv");
    let base_path = data_dir.join("math").join("qrels.tsv");
    let qrels_path = if pool_path.exists()
        && pool_path
            .metadata()
            .map(|m| m.len() > 0)
            .unwrap_or(false)
    {
        pool_path
    } else {
        base_path
    };
    let content = std::fs::read_to_string(&qrels_path)
        .map_err(|e| HarnessOptError::Harness(format!("read {}: {e}", qrels_path.display())))?;
    Ok(parse_qrels_tsv(&content))
}

// ---------------------------------------------------------------------------
// HarnessProject implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl HarnessProject for ObliqHarnessProject {
    async fn seed_candidate(&self) -> Result<Candidate> {
        let constraints = ComponentConstraints {
            max_chars: Some(4_000),
            preserve_terms: self.preserve_terms.clone(),
            forbidden_terms: vec![],
            format_hint: Some("Markdown retrieval strategy instructions".to_string()),
        };
        let component = MutableComponent {
            id: OBLIQ_TASK_INSTRUCTIONS_COMPONENT_ID.to_string(),
            description: Some(
                "Retrieval strategy instructions injected via --task-instructions-file".to_string(),
            ),
            value: ComponentValue::Text {
                text: OBLIQ_SEED_PROMPT.to_string(),
            },
            constraints,
        };
        let mut mutable_components = BTreeMap::new();
        mutable_components.insert(OBLIQ_TASK_INSTRUCTIONS_COMPONENT_ID.to_string(), component);
        Ok(Candidate {
            id: "seed".to_string(),
            parent_id: None,
            mutable_components,
            immutable_context: BTreeMap::new(),
            metadata: BTreeMap::from([("project".into(), json!("obliq-math"))]),
        })
    }

    async fn trainset(&self) -> Result<Vec<HarnessExample>> {
        build_examples(TRAIN_IDS, Split::Train, &self.queries)
    }

    async fn valset(&self) -> Result<Vec<HarnessExample>> {
        build_examples(VAL_IDS, Split::Val, &self.queries)
    }

    async fn evaluate_example(
        &self,
        run: &OptimizationRun,
        candidate: &Candidate,
        example: &HarnessExample,
        _context: TraceContext,
        _cancellation: CancellationToken,
    ) -> Result<ExampleRun> {
        // Step 1 — Extract task instructions text
        let component =
            match candidate
                .mutable_components
                .get(OBLIQ_TASK_INSTRUCTIONS_COMPONENT_ID)
            {
                Some(c) => c,
                None => {
                    return Ok(failure_run(
                        example,
                        format!(
                            "missing mutable component '{OBLIQ_TASK_INSTRUCTIONS_COMPONENT_ID}'"
                        ),
                        0,
                    ));
                }
            };
        let instructions = match &component.value {
            ComponentValue::Text { text } => text.clone(),
            _ => {
                return Ok(failure_run(
                    example,
                    format!(
                        "component '{OBLIQ_TASK_INSTRUCTIONS_COMPONENT_ID}' must be \
                         ComponentValue::Text"
                    ),
                    0,
                ));
            }
        };

        // Step 2 — Validate preserve_terms (before any file I/O)
        for term in &component.constraints.preserve_terms {
            if !instructions.contains(term.as_str()) {
                return Ok(ExampleRun {
                    example: example.clone(),
                    result: EvaluationResult {
                        example_id: example.id.clone(),
                        split: example.split.clone(),
                        score: 0.0,
                        passed: Some(false),
                        feedback: Some(format!(
                            "Constraint violation: missing required term '{term}'"
                        )),
                        metrics: BTreeMap::new(),
                        diagnostics: BTreeMap::from([(
                            "violated_term".into(),
                            json!(term.as_str()),
                        )]),
                    },
                    trace: None,
                    artifacts: RunArtifacts::default(),
                    metric_calls: 0,
                });
            }
        }

        // Step 3 — Write instructions to temp file (absolute path, passed to subprocess)
        let candidate_work_dir = run
            .run_dir
            .join("candidates")
            .join(&candidate.id);
        tokio::fs::create_dir_all(&candidate_work_dir).await?;
        let instructions_file = candidate_work_dir.join("obliq-instructions.txt");
        tokio::fs::write(&instructions_file, &instructions).await?;

        // Step 4 — Per-(candidate, example) output directory.
        // The subprocess writes to <output_dir>/<config_hash>/math/<task_id>.json and
        // updates _latest to point to the most recent config-hash directory.
        // Using a per-example subdirectory prevents concurrent evaluations for the same
        // candidate from overwriting each other's _latest pointer.
        let candidate_output_dir = candidate_work_dir
            .join(".benchmarks")
            .join("obliq")
            .join("runs")
            .join(&example.id);
        tokio::fs::create_dir_all(&candidate_output_dir).await?;

        // Step 5 — Invoke subprocess
        //
        // CWD = lash_oblique_dir so scripts/query_math_qdrant.py is reachable.
        // --output-dir uses an absolute path to isolate outputs per candidate.
        // --task-instructions-file uses an absolute path.
        // --data-dir uses an absolute path to the benchmark data.
        let mut cmd = tokio::process::Command::new(&self.lash_oblique_bin);
        cmd.current_dir(&self.lash_oblique_dir);
        cmd.arg("run-batch");
        cmd.arg("--subsets").arg("math");
        cmd.arg("--tasks").arg(&example.id);
        cmd.arg("--model").arg(&self.model);
        if let Some(v) = &self.variant {
            cmd.arg("--variant").arg(v);
        }
        cmd.arg("--task-instructions-file").arg(&instructions_file);
        cmd.arg("--concurrency").arg("1");
        cmd.arg("--data-dir").arg(&self.data_dir);
        cmd.arg("--output-dir").arg(&candidate_output_dir);

        let status = cmd.status().await.map_err(|e| {
            HarnessOptError::Harness(format!("failed to launch lash-oblique: {e}"))
        })?;

        // Step 6 — Handle non-zero exit
        if !status.success() {
            return Ok(failure_run(
                example,
                format!("lash-oblique exited with status: {status}"),
                1,
            ));
        }

        // Step 7 — Find output via `_latest` pointer
        let runs_dir = candidate_output_dir;
        let latest_content = tokio::fs::read_to_string(runs_dir.join("_latest"))
            .await
            .map_err(|e| HarnessOptError::Harness(format!("read _latest pointer: {e}")))?;
        let config_hash = latest_content.trim();
        let output_json = runs_dir
            .join(config_hash)
            .join("math")
            .join(format!("{}.json", example.id));
        let trace_jsonl = runs_dir
            .join(config_hash)
            .join("math")
            .join(format!("{}.trace.jsonl", example.id));

        // Step 8 — Parse JSON output and extract score
        let raw = tokio::fs::read(&output_json).await.map_err(|e| {
            HarnessOptError::Harness(format!("read output JSON {}: {e}", output_json.display()))
        })?;
        let run_output: Value = serde_json::from_slice(&raw)?;

        // When metrics is null the agent failed to submit a ranked list
        if run_output["metrics"].is_null() {
            let errors: Vec<String> = run_output["errors"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            return Ok(ExampleRun {
                example: example.clone(),
                result: EvaluationResult {
                    example_id: example.id.clone(),
                    split: example.split.clone(),
                    score: 0.0,
                    passed: Some(false),
                    feedback: Some(format!(
                        "agent did not submit a ranked list; errors: {}",
                        errors.join("; ")
                    )),
                    metrics: BTreeMap::new(),
                    diagnostics: BTreeMap::from([
                        ("turn_outcome".into(), json!("no_submission")),
                        ("tool_call_count".into(), json!(0)),
                        ("error_count".into(), json!(errors.len())),
                    ]),
                },
                trace: Some(TraceBundle {
                    example_id: example.id.clone(),
                    records: vec![],
                }),
                artifacts: RunArtifacts {
                    response_json: Some(output_json),
                    typed_trace_jsonl: Some(trace_jsonl),
                    ..Default::default()
                },
                metric_calls: 1,
            });
        }

        let score = run_output["metrics"]["pooled"]["ndcg_at_10"]
            .as_f64()
            .unwrap_or(0.0);
        let recall_at_10 = run_output["metrics"]["pooled"]["recall_at_10"]
            .as_f64()
            .unwrap_or(0.0);
        let recall_at_100 = run_output["metrics"]["pooled"]["recall_at_100"]
            .as_f64()
            .unwrap_or(0.0);
        let gold_count = run_output["metrics"]["pooled"]["gold_count"]
            .as_u64()
            .unwrap_or(0);
        let tool_calls = run_output["tool_calls"].as_u64().unwrap_or(0);
        let errors_count = run_output["errors"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0);
        let ranked_doc_ids: Vec<String> = run_output["ranked_doc_ids"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();

        // Step 9 — Parse trace JSONL into Vec<TraceRecord>
        let trace_records: Vec<TraceRecord> = if trace_jsonl.exists() {
            match tokio::fs::read_to_string(&trace_jsonl).await {
                Ok(content) => content
                    .lines()
                    .filter(|l| !l.is_empty())
                    .filter_map(|l| {
                        serde_json::from_str::<TraceRecord>(l)
                            .map_err(|e| {
                                eprintln!(
                                    "warn: skipping malformed trace record: {e}"
                                );
                            })
                            .ok()
                    })
                    .collect(),
                Err(e) => {
                    eprintln!(
                        "warn: could not read trace file {}: {e}",
                        trace_jsonl.display()
                    );
                    vec![]
                }
            }
        } else {
            vec![]
        };

        // Step 10 — Build rich feedback string for the reflection LM.
        // Extract the last non-empty LLM response text from the trace.
        let final_agent_text: String = {
            let mut last = String::new();
            for record in &trace_records {
                if let TraceEvent::LlmCallCompleted { response, .. } = &record.event {
                    if !response.text.is_empty() {
                        last.clone_from(&response.text);
                    }
                }
            }
            last
        };
        let gold_qrels = self.qrels.get(&example.id).cloned().unwrap_or_default();
        let feedback = build_asi_feedback(
            &trace_records,
            &ranked_doc_ids,
            &gold_qrels,
            score,
            &final_agent_text,
        );

        let mut metrics = BTreeMap::new();
        metrics.insert("recall_at_10".to_string(), recall_at_10);
        metrics.insert("recall_at_100".to_string(), recall_at_100);
        metrics.insert("gold_count".to_string(), gold_count as f64);

        let result = EvaluationResult {
            example_id: example.id.clone(),
            split: example.split.clone(),
            score,
            passed: Some(score > 0.0),
            feedback: Some(feedback),
            metrics,
            diagnostics: BTreeMap::from([
                ("turn_outcome".into(), json!("submitted")),
                ("tool_call_count".into(), json!(tool_calls)),
                ("error_count".into(), json!(errors_count)),
            ]),
        };

        Ok(ExampleRun {
            example: example.clone(),
            result,
            trace: Some(TraceBundle {
                example_id: example.id.clone(),
                records: trace_records,
            }),
            artifacts: RunArtifacts {
                response_json: Some(output_json),
                typed_trace_jsonl: Some(trace_jsonl),
                ..Default::default()
            },
            metric_calls: 1,
        })
    }

    async fn on_candidate_accepted(
        &self,
        run: &OptimizationRun,
        store: &dyn HarnessOptStore,
    ) -> Result<()> {
        write_gepa_summary(run, self, store).await
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn build_examples(
    ids: &[&str],
    split: Split,
    queries: &HashMap<String, String>,
) -> Result<Vec<HarnessExample>> {
    ids.iter()
        .map(|id| {
            let text = queries.get(*id).ok_or_else(|| {
                HarnessOptError::Harness(format!(
                    "missing query text for '{id}' in queries.jsonl"
                ))
            })?;
            Ok(HarnessExample {
                id: id.to_string(),
                split: split.clone(),
                input: json!({ "text": text }),
                expected: None,
                metadata: BTreeMap::from([("subset".into(), json!("math"))]),
            })
        })
        .collect()
}

/// Construct a failure `ExampleRun` with the given feedback message.
fn failure_run(example: &HarnessExample, feedback: String, metric_calls: u64) -> ExampleRun {
    ExampleRun {
        example: example.clone(),
        result: EvaluationResult {
            example_id: example.id.clone(),
            split: example.split.clone(),
            score: 0.0,
            passed: Some(false),
            feedback: Some(feedback),
            metrics: BTreeMap::new(),
            diagnostics: BTreeMap::from([
                ("turn_outcome".into(), json!("error")),
                ("tool_call_count".into(), json!(0)),
                ("error_count".into(), json!(1)),
            ]),
        },
        trace: None,
        artifacts: RunArtifacts::default(),
        metric_calls,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::OptimizationConfig;

    // ---- Test 1: qrels TSV parsing ----

    #[test]
    fn qrels_tsv_parsing_handles_header_and_3col_format() {
        let fixture = "query-id\tdoc-id\trelevance\n\
                       q02193\td00412\t2\n\
                       q02193\td00103\t1\n\
                       q01486\td00789\t1\n";
        let result = parse_qrels_tsv(fixture);
        assert_eq!(result["q02193"]["d00412"], 2.0);
        assert_eq!(result["q02193"]["d00103"], 1.0);
        assert_eq!(result["q01486"]["d00789"], 1.0);
        // Header row must not appear as a key
        assert!(!result.contains_key("query-id"));
    }

    // ---- Test 2: gold coverage computation ----

    #[test]
    fn gold_coverage_computation() {
        // 1 of 2 gold docs in top-3 → 0.5
        let ranked: Vec<String> = ["d1", "d2", "d3"].iter().map(|s| s.to_string()).collect();
        let mut gold: HashMap<String, f64> = HashMap::new();
        gold.insert("d1".to_string(), 1.0);
        gold.insert("d4".to_string(), 1.0);
        let cov = gold_coverage(&ranked, &gold, 3);
        assert!(
            (cov - 0.5).abs() < 1e-9,
            "expected 0.5 (1 of 2 gold in top-3), got {cov}"
        );

        // Empty ranked list → 0.0
        let cov2 = gold_coverage(&[], &gold, 10);
        assert_eq!(cov2, 0.0, "empty ranked list should give 0.0");

        // Empty gold set → 0.0
        let ranked2: Vec<String> = ["d1", "d2"].iter().map(|s| s.to_string()).collect();
        let cov3 = gold_coverage(&ranked2, &HashMap::new(), 10);
        assert_eq!(cov3, 0.0, "empty gold set should give 0.0");
    }

    // ---- Test 3: dataset split correctness ----

    #[test]
    fn dataset_split_correctness() {
        assert_eq!(TRAIN_IDS.len(), 7, "train set must have 7 query IDs");
        assert_eq!(VAL_IDS.len(), 3, "val set must have 3 query IDs");

        let mut seen: HashSet<&str> = HashSet::new();
        for id in TRAIN_IDS.iter().chain(VAL_IDS.iter()) {
            assert!(seen.insert(id), "duplicate query id: {id}");
        }
        assert_eq!(seen.len(), 10, "combined split must cover all 10 math-rep10 queries");
    }

    // ---- Test 4: build_asi_feedback — search query extraction ----
    //
    // Uses a ProtocolStep[runtime] fixture with assembled Lashlang code, matching the
    // format Lashlang (RLM mode) actually emits. The code block invokes both
    // tools.search (with a queries array) and tools.judge_candidates.

    #[test]
    fn build_asi_feedback_extracts_search_queries_and_tool_counts() {
        let lashlang_code = concat!(
            "@label(title: \"search and judge\")\n",
            "results = await tools.search({\n",
            "  queries: [\n",
            "    \"proof by induction algebra\"\n",
            "  ],\n",
            "  limit: 10\n",
            "})?\n",
            "judge = await tools.judge_candidates({\n",
            "  verifier_predicate: \"uses induction\",\n",
            "  candidates: [\"d1\", \"d2\"]\n",
            "})?",
        );
        let fixture = serde_json::json!({
            "schema_version": 2,
            "id": "test-uuid-1",
            "timestamp": "2026-01-01T00:00:00Z",
            "context": {},
            "type": "protocol_step",
            "plugin_id": "runtime",
            "payload": {
                "diagnostic": {
                    "payload": { "code": lashlang_code }
                }
            }
        });
        let record: lash_trace::TraceRecord = serde_json::from_value(fixture).unwrap();

        let gold: HashMap<String, f64> = HashMap::new();
        let feedback = build_asi_feedback(&[record], &[], &gold, 0.0, "");

        assert!(
            feedback.contains("proof by induction algebra"),
            "feedback should contain the search query; got: {feedback}"
        );
        assert!(
            feedback.contains("search×1"),
            "feedback should show search×1; got: {feedback}"
        );
        assert!(
            feedback.contains("judge_candidates×1"),
            "feedback should show judge_candidates×1; got: {feedback}"
        );
    }

    // ---- Test 5: build_asi_feedback — zero tool calls ----

    #[test]
    fn build_asi_feedback_zero_tool_calls_produces_explicit_message() {
        let gold: HashMap<String, f64> = HashMap::new();
        let feedback = build_asi_feedback(&[], &[], &gold, 0.0, "");
        assert!(
            feedback.contains("none"),
            "feedback for zero tool calls should contain 'none'; got: {feedback}"
        );
        // No panic is also verified by reaching this line.
    }

    // ---- Test 6: build_asi_feedback — gold coverage ----

    #[test]
    fn build_asi_feedback_gold_coverage_in_feedback_string() {
        let ranked_with_gold: Vec<String> =
            ["d_gold", "d_other1", "d_other2"].iter().map(|s| s.to_string()).collect();
        let mut gold: HashMap<String, f64> = HashMap::new();
        gold.insert("d_gold".to_string(), 1.0);

        let feedback_hit = build_asi_feedback(&[], &ranked_with_gold, &gold, 0.0, "");
        assert!(
            feedback_hit.contains("1/1"),
            "feedback should show 1/1 gold coverage; got: {feedback_hit}"
        );

        let ranked_no_gold: Vec<String> =
            ["d_other1", "d_other2"].iter().map(|s| s.to_string()).collect();
        let feedback_miss = build_asi_feedback(&[], &ranked_no_gold, &gold, 0.0, "");
        assert!(
            feedback_miss.contains("0/1"),
            "feedback should show 0/1 gold coverage; got: {feedback_miss}"
        );
    }

    // ---- Test 7: build_asi_feedback — token budget truncation ----
    //
    // Uses a single ProtocolStep[runtime] fixture whose code block contains 12
    // tools.search() calls. build_asi_feedback must cap displayed queries at 8 and
    // note the total count in the output.

    #[test]
    fn build_asi_feedback_truncates_to_eight_search_queries() {
        // Build a Lashlang code string with 12 sequential tools.search() calls,
        // each carrying a single query.
        let calls: String = (1u32..=12)
            .map(|i| {
                format!(
                    "r{i} = await tools.search({{\n  queries: [\n    \"search query number {i}\"\n  ],\n  limit: 10\n}})?\n"
                )
            })
            .collect();
        let fixture = serde_json::json!({
            "schema_version": 2,
            "id": "test-uuid-trunc",
            "timestamp": "2026-01-01T00:00:00Z",
            "context": {},
            "type": "protocol_step",
            "plugin_id": "runtime",
            "payload": {
                "diagnostic": {
                    "payload": { "code": calls }
                }
            }
        });
        let record: lash_trace::TraceRecord = serde_json::from_value(fixture).unwrap();

        let gold: HashMap<String, f64> = HashMap::new();
        let feedback = build_asi_feedback(&[record], &[], &gold, 0.0, "");

        // Query 8 should appear (1-indexed in the output), query 9 should not.
        assert!(
            feedback.contains("8."),
            "feedback should show query 8; got: {feedback}"
        );
        assert!(
            !feedback.contains("9."),
            "feedback should not show query 9; got: {feedback}"
        );
        // Total count (12) should be mentioned.
        assert!(
            feedback.contains("12"),
            "feedback should indicate 12 total queries; got: {feedback}"
        );
    }

    // ---- Test 8: real Sonnet trace — tool calls must be detected ----

    #[test]
    fn build_asi_feedback_real_sonnet_trace_detects_tool_calls() {
        // Load the real post-fix Sonnet 4.6 trace for q01066 (483 records).
        // This trace contains ProtocolStep records with assembled Lashlang code
        // that invokes search, judge_candidates, tournament_rerank, and discover_docs.
        // It contains NO ToolCallStarted events — only ProtocolStep + RuntimeStreamEvent.
        let fixture_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("q01066_sonnet_post_fix.trace.jsonl");

        let content = std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|e| panic!("could not read fixture {}: {e}", fixture_path.display()));

        let records: Vec<lash_trace::TraceRecord> = content
            .lines()
            .filter(|l| !l.is_empty())
            .enumerate()
            .filter_map(|(i, line)| {
                serde_json::from_str(line)
                    .map_err(|e| eprintln!("warn: skipping line {}: {e}", i + 1))
                    .ok()
            })
            .collect();

        assert!(!records.is_empty(), "fixture must contain at least one record");

        let gold: HashMap<String, f64> = HashMap::new();
        let feedback = build_asi_feedback(&records, &[], &gold, 0.0, "");

        // The old implementation always produced these strings for Lashlang traces
        // (because ToolCallStarted is never emitted). The new implementation must not.
        assert!(
            !feedback.contains("agent submitted without using retrieval tools"),
            "feedback must not claim agent used no tools; got:\n{feedback}"
        );
        assert!(
            !feedback.contains("none — agent made no search calls"),
            "feedback must not claim no search calls; got:\n{feedback}"
        );

        // Must detect at least one concrete tool from the real trace.
        assert!(
            feedback.contains("search×"),
            "feedback must show search call count; got:\n{feedback}"
        );
        assert!(
            feedback.contains("judge_candidates×"),
            "feedback must show judge_candidates call count; got:\n{feedback}"
        );

        // Must extract at least one real search query observed in Phase 1.
        assert!(
            feedback.contains("urn balls drawing expectation isolated white balls sequence"),
            "feedback must contain the observed search query; got:\n{feedback}"
        );
    }

    // ---- Smoke test (requires Qdrant and lash-oblique binary) ----

    #[tokio::test]
    #[ignore = "requires Qdrant running and lash-oblique binary"]
    async fn smoke_test_evaluate_example_q02193() {
        let lash_oblique_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap() // crates/
            .parent()
            .unwrap() // lash/ workspace root
            .parent()
            .unwrap() // gepa-obliq/
            .join("lash-oblique");

        let bin = detect_lash_oblique_bin(&lash_oblique_dir);
        assert!(
            bin.exists(),
            "lash-oblique binary not found at {:?}",
            bin
        );
        println!("lash-oblique binary: {:?}", bin);

        let data_dir = lash_oblique_dir
            .join(".benchmarks")
            .join("obliq")
            .join("data");

        let tmp = tempfile::tempdir().expect("create tempdir");
        let run_dir = tmp.path().to_path_buf();

        let project = ObliqHarnessProject::new(
            bin.clone(),
            lash_oblique_dir.clone(),
            data_dir,
            "anthropic/claude-haiku-4.5".to_string(),
            None,
            vec![],
        )
        .expect("ObliqHarnessProject::new");

        let run = crate::OptimizationRun {
            run_id: "smoke-test".to_string(),
            experiment_id: "obliq-math-smoke".to_string(),
            run_dir,
            config: OptimizationConfig::default(),
        };

        let candidate = project.seed_candidate().await.expect("seed_candidate");

        let example = project
            .trainset()
            .await
            .expect("trainset")
            .into_iter()
            .find(|e| e.id == "q02193")
            .expect("q02193 not found in trainset");

        let example_run = project
            .evaluate_example(
                &run,
                &candidate,
                &example,
                TraceContext::default(),
                CancellationToken::new(),
            )
            .await
            .expect("evaluate_example returned Err");

        let score = example_run.result.score;
        assert!(
            score >= 0.0 && score <= 1.0,
            "score {score} out of [0.0, 1.0]"
        );
        assert!(example_run.trace.is_some(), "trace should be Some");

        println!("score: {score}");
        println!("feedback: {:?}", example_run.result.feedback);
        println!(
            "trace records: {}",
            example_run.trace.as_ref().unwrap().records.len()
        );
        println!(
            "trace file: {:?}",
            example_run.artifacts.typed_trace_jsonl
        );
    }
}
