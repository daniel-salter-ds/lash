//! Writes `_gepa_summary.json` after each accepted candidate (Obliq) or at run completion (synthetic).
//! See `ARCHITECTURE.md §7` for the full specification.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::{
    CandidateRecord, ComponentValue, EvaluationRecord, HarnessOptStore, HarnessProject,
    OptimizationRun, ProposalRecord, Result, Split,
};

// ---------------------------------------------------------------------------
// Output structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaSummary {
    pub schema_version: String,
    pub run_id: String,
    pub experiment_id: String,
    pub completed_at: String,
    pub task_lm: String,
    pub reflection_lm: String,
    pub seed: u64,
    pub max_metric_calls: u64,
    pub total_metric_calls: u64,
    pub num_val_examples: usize,
    pub val_example_ids: Vec<String>,
    pub candidates: Vec<CandidateSummaryEntry>,
    pub pareto_front_per_val_example: BTreeMap<String, Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateSummaryEntry {
    pub candidate_idx: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_id: Option<String>,
    pub is_seed: bool,
    pub is_accepted: bool,
    pub parent_idxs: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub discovery_eval_count: Option<u64>,
    pub component_texts: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub component_mutated: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub val_aggregate_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub val_subscores: Option<BTreeMap<String, f64>>,
    pub is_on_pareto_front: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minibatch_score_before: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minibatch_score_after: Option<f64>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_text_value(value: &ComponentValue) -> String {
    match value {
        ComponentValue::Text { text } => text.clone(),
        ComponentValue::Json { value } => serde_json::to_string(value).unwrap_or_default(),
        ComponentValue::PromptTemplate { .. } => "<PromptTemplate>".to_string(),
        ComponentValue::PromptContribution { .. } => "<PromptContribution>".to_string(),
    }
}

fn apply_patches_to_candidate(
    candidate: &crate::Candidate,
    patches: &[crate::ComponentPatch],
) -> BTreeMap<String, String> {
    let mut candidate = candidate.clone();
    for patch in patches {
        let _ = crate::apply_patch(&mut candidate, patch);
    }
    candidate
        .mutable_components
        .iter()
        .map(|(k, v)| (k.clone(), extract_text_value(&v.value)))
        .collect()
}

/// Returns the indices (into `entries`) of candidates that are on the Pareto front.
///
/// A candidate is on the front if no other candidate beats it on every val example.
fn compute_pareto_front_idxs(
    entries: &[CandidateSummaryEntry],
    val_ids: &[String],
) -> Vec<usize> {
    let accepted: Vec<(usize, &BTreeMap<String, f64>)> = entries
        .iter()
        .filter(|e| e.is_accepted)
        .filter_map(|e| e.val_subscores.as_ref().map(|s| (e.candidate_idx, s)))
        .collect();

    accepted
        .iter()
        .filter(|(idx_i, scores_i)| {
            !accepted.iter().any(|(idx_j, scores_j)| {
                if idx_j == idx_i {
                    return false;
                }
                let mut strictly_better = false;
                for val_id in val_ids {
                    let si = scores_i.get(val_id).copied().unwrap_or(0.0);
                    let sj = scores_j.get(val_id).copied().unwrap_or(0.0);
                    if sj < si {
                        return false; // j is worse on this example — doesn't dominate
                    }
                    if sj > si {
                        strictly_better = true;
                    }
                }
                strictly_better // j dominates i
            })
        })
        .map(|(idx, _)| *idx)
        .collect()
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Write `_gepa_summary.json` to `run.run_dir`.
///
/// Called once at completion for the synthetic task, or after every accepted
/// candidate for `ObliqHarnessProject` (via `on_candidate_accepted`).
pub async fn write_gepa_summary(
    run: &OptimizationRun,
    project: &dyn HarnessProject,
    store: &dyn HarnessOptStore,
) -> Result<()> {
    // Step 1 — Load val ordering
    let val_examples = project.valset().await?;
    let val_ids: Vec<String> = val_examples.iter().map(|e| e.id.clone()).collect();

    // Step 2 — Load all accepted candidates
    let candidate_records: Vec<CandidateRecord> = store.candidates().await?;
    // Build lookup: candidate_id -> (idx, &CandidateRecord)
    let accepted_lookup: HashMap<String, (usize, &CandidateRecord)> = candidate_records
        .iter()
        .enumerate()
        .map(|(i, r)| (r.candidate.id.clone(), (i, r)))
        .collect();

    // Step 3 — Load all evaluations (val split only)
    let all_evals: Vec<EvaluationRecord> = store.evaluations().await?;
    let val_evals: HashMap<(String, String), f64> = all_evals
        .iter()
        .filter(|e| e.split == Split::Val)
        .map(|e| ((e.candidate_id.clone(), e.example_id.clone()), e.score))
        .collect();

    // Step 4 — Load all proposals
    let all_proposals: Vec<ProposalRecord> = store.proposals().await?;
    let accepted_proposals: HashMap<String, &ProposalRecord> = all_proposals
        .iter()
        .filter(|p| p.accepted && p.candidate_id.is_some())
        .map(|p| (p.candidate_id.clone().unwrap(), p))
        .collect();
    let rejected_proposals: Vec<&ProposalRecord> =
        all_proposals.iter().filter(|p| !p.accepted).collect();

    // Step 5 — Load run stats
    let stats = store.stats().await?;

    // Step 6 — Build accepted candidate entries (in order)
    let mut entries: Vec<CandidateSummaryEntry> = Vec::new();
    for (idx, record) in candidate_records.iter().enumerate() {
        let cid = &record.candidate.id;

        // Val subscores (in val_ids order)
        let val_subscores: BTreeMap<String, f64> = val_ids
            .iter()
            .map(|vid| {
                let score = val_evals
                    .get(&(cid.clone(), vid.clone()))
                    .copied()
                    .unwrap_or(0.0);
                (vid.clone(), score)
            })
            .collect();

        let val_aggregate = if val_subscores.is_empty() {
            0.0
        } else {
            val_subscores.values().sum::<f64>() / val_subscores.len() as f64
        };

        // Component texts
        let component_texts: BTreeMap<String, String> = record
            .candidate
            .mutable_components
            .iter()
            .map(|(k, v)| (k.clone(), extract_text_value(&v.value)))
            .collect();

        // Minibatch scores from proposal record
        let (mb_before, mb_after) = if let Some(prop) = accepted_proposals.get(cid) {
            (Some(prop.before_score), Some(prop.after_score))
        } else {
            (None, None) // seed has no ProposalRecord
        };

        // Component mutated
        let component_mutated = accepted_proposals
            .get(cid)
            .and_then(|p| p.selected_components.first().cloned());

        // Parent idxs
        let parent_idxs: Vec<usize> = record
            .parent_ids
            .iter()
            .filter_map(|pid| accepted_lookup.get(pid).map(|(i, _)| *i))
            .collect();

        // discovery_eval_count
        let discovery_eval_count = run
            .config
            .max_metric_calls
            .saturating_sub(record.discovery_budget);

        entries.push(CandidateSummaryEntry {
            candidate_idx: idx,
            candidate_id: Some(cid.clone()),
            is_seed: record.generation == 0,
            is_accepted: true,
            parent_idxs,
            discovery_eval_count: Some(discovery_eval_count),
            component_texts,
            component_mutated,
            val_aggregate_score: Some(val_aggregate),
            val_subscores: Some(val_subscores),
            is_on_pareto_front: false, // computed in step 8
            minibatch_score_before: mb_before,
            minibatch_score_after: mb_after,
        });
    }

    // Step 7 — Build rejected proposal entries
    let base_len = entries.len(); // accepted count before appending rejected
    for prop in &rejected_proposals {
        let parent_texts = if let Some(pid) = prop.parent_ids.first() {
            if let Some((_, parent_record)) = accepted_lookup.get(pid) {
                apply_patches_to_candidate(&parent_record.candidate, &prop.patches)
            } else {
                BTreeMap::new()
            }
        } else {
            BTreeMap::new()
        };

        let parent_idxs: Vec<usize> = prop
            .parent_ids
            .iter()
            .filter_map(|pid| accepted_lookup.get(pid).map(|(i, _)| *i))
            .collect();

        entries.push(CandidateSummaryEntry {
            candidate_idx: base_len + entries.len() - base_len,
            candidate_id: None,
            is_seed: false,
            is_accepted: false,
            parent_idxs,
            discovery_eval_count: None,
            component_texts: parent_texts,
            component_mutated: prop.selected_components.first().cloned(),
            val_aggregate_score: None,
            val_subscores: None,
            is_on_pareto_front: false,
            minibatch_score_before: Some(prop.before_score),
            minibatch_score_after: Some(prop.after_score),
        });
    }

    // Fix candidate_idx for rejected entries
    for (i, entry) in entries.iter_mut().enumerate() {
        entry.candidate_idx = i;
    }

    // Step 8 — Compute Pareto front
    let frontier_idxs = compute_pareto_front_idxs(&entries, &val_ids);
    for idx in frontier_idxs {
        if let Some(entry) = entries.get_mut(idx) {
            entry.is_on_pareto_front = true;
        }
    }

    // Step 9 — Compute pareto_front_per_val_example
    let mut pareto_front_per_val_example: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for val_id in &val_ids {
        let best_score: f64 = entries
            .iter()
            .filter(|e| e.is_accepted)
            .filter_map(|e| e.val_subscores.as_ref()?.get(val_id).copied())
            .fold(f64::NEG_INFINITY, f64::max);

        let best_candidates: Vec<usize> = entries
            .iter()
            .filter(|e| e.is_accepted)
            .filter(|e| {
                e.val_subscores
                    .as_ref()
                    .and_then(|s| s.get(val_id))
                    .copied()
                    .unwrap_or(f64::NEG_INFINITY)
                    >= best_score
            })
            .map(|e| e.candidate_idx)
            .collect();

        pareto_front_per_val_example.insert(val_id.clone(), best_candidates);
    }

    // Step 10 — Assemble and write JSON
    let completed_at = {
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        // Simple RFC 3339 UTC timestamp: YYYY-MM-DDTHH:MM:SSZ
        let s = secs;
        let sec = s % 60;
        let min = (s / 60) % 60;
        let hour = (s / 3600) % 24;
        let days = s / 86400; // days since unix epoch (1970-01-01)
        // Gregorian calendar computation
        let (year, month, day) = days_to_ymd(days);
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
    };

    let summary = GepaSummary {
        schema_version: "1".to_string(),
        run_id: run.run_id.clone(),
        experiment_id: run.experiment_id.clone(),
        completed_at,
        task_lm: run.config.task_lm.clone(),
        reflection_lm: run.config.reflection_lm.clone(),
        seed: 0,
        max_metric_calls: run.config.max_metric_calls,
        total_metric_calls: stats.metric_calls_used,
        num_val_examples: val_ids.len(),
        val_example_ids: val_ids,
        candidates: entries,
        pareto_front_per_val_example,
    };

    let json_bytes = serde_json::to_vec_pretty(&summary)
        .map_err(crate::HarnessOptError::Json)?;
    let path = run.run_dir.join("_gepa_summary.json");
    tokio::fs::write(&path, json_bytes).await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests — no LLM calls; uses in-process SQLite store
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use async_trait::async_trait;
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::{
        Candidate, CandidateRecord, ComponentConstraints, ComponentValue, EvaluationResult,
        ExampleRun, HarnessExample, HarnessProject, MutableComponent, OptimizationConfig,
        OptimizationRun, Result, RunArtifacts, Split,
        SqliteHarnessStore, candidate_fingerprint, record_example_run,
    };
    use lash_trace::TraceContext;

    // Minimal no-op project for valset queries.
    struct TwoValProject {
        val: Vec<HarnessExample>,
    }

    #[async_trait]
    impl HarnessProject for TwoValProject {
        async fn seed_candidate(&self) -> Result<Candidate> {
            unimplemented!()
        }
        async fn trainset(&self) -> Result<Vec<HarnessExample>> {
            Ok(vec![])
        }
        async fn valset(&self) -> Result<Vec<HarnessExample>> {
            Ok(self.val.clone())
        }
        async fn evaluate_example(
            &self, _: &OptimizationRun, _: &Candidate, _: &HarnessExample,
            _: TraceContext, _: CancellationToken,
        ) -> Result<ExampleRun> {
            unimplemented!()
        }
    }

    fn make_example(id: &str, split: Split) -> HarnessExample {
        HarnessExample {
            id: id.to_string(),
            split,
            input: json!({ "question": "test?" }),
            expected: Some(json!({ "answer": "yes" })),
            metadata: BTreeMap::new(),
        }
    }

    fn make_seed_candidate() -> Candidate {
        let mut comps = BTreeMap::new();
        comps.insert(
            "instr".to_string(),
            MutableComponent {
                id: "instr".to_string(),
                description: None,
                value: ComponentValue::Text { text: "Answer the question.".to_string() },
                constraints: ComponentConstraints {
                    max_chars: None,
                    preserve_terms: vec![],
                    forbidden_terms: vec![],
                    format_hint: None,
                },
            },
        );
        Candidate {
            id: "seed".to_string(),
            parent_id: None,
            mutable_components: comps,
            immutable_context: BTreeMap::new(),
            metadata: BTreeMap::new(),
        }
    }

    #[tokio::test]
    async fn write_gepa_summary_produces_valid_json_with_seed_and_val_scores() {
        let temp = std::env::temp_dir()
            .join(format!("gepa-summary-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp).unwrap();
        let store = SqliteHarnessStore::open(&temp).await.unwrap();
        let config = OptimizationConfig {
            max_metric_calls: 20,
            task_lm: "test-lm".to_string(),
            reflection_lm: "test-reflect-lm".to_string(),
            ..OptimizationConfig::default()
        };
        let run = OptimizationRun {
            run_id: "test-run".to_string(),
            experiment_id: "test-exp".to_string(),
            run_dir: temp.clone(),
            config: config.clone(),
        };
        store.init_run(&run).await.unwrap();

        // Register seed candidate.
        let seed = make_seed_candidate();
        let fp = candidate_fingerprint(&seed).unwrap();
        store.upsert_candidate(&CandidateRecord {
            candidate: seed.clone(),
            fingerprint: fp.clone(),
            parent_ids: vec![],
            generation: 0,
            source_strategy: "seed".to_string(),
            component_cursor: 0,
            discovery_budget: 20,
        }).await.unwrap();

        // Record two val evaluations.
        let val_a = make_example("V01", Split::Val);
        let val_b = make_example("V02", Split::Val);
        for (ex, score) in [(&val_a, 0.8_f64), (&val_b, 0.6_f64)] {
            let run_result = ExampleRun {
                example: ex.clone(),
                result: EvaluationResult {
                    example_id: ex.id.clone(),
                    split: Split::Val,
                    score,
                    passed: Some(score >= 1.0),
                    feedback: None,
                    metrics: BTreeMap::new(),
                    diagnostics: BTreeMap::new(),
                },
                trace: None,
                artifacts: RunArtifacts::default(),
                metric_calls: 1,
            };
            record_example_run(&store, &seed, &fp, &run_result, false).await.unwrap();
        }

        let project = TwoValProject { val: vec![val_a.clone(), val_b.clone()] };
        write_gepa_summary(&run, &project, &store).await.unwrap();

        let path = temp.join("_gepa_summary.json");
        assert!(path.exists(), "_gepa_summary.json must be written");

        let json_bytes = std::fs::read(&path).unwrap();
        let summary: GepaSummary = serde_json::from_slice(&json_bytes)
            .expect("_gepa_summary.json must be valid JSON matching GepaSummary schema");

        assert_eq!(summary.schema_version, "1");
        assert_eq!(summary.run_id, "test-run");
        assert_eq!(summary.experiment_id, "test-exp");
        assert_eq!(summary.task_lm, "test-lm");
        assert_eq!(summary.reflection_lm, "test-reflect-lm");
        assert_eq!(summary.num_val_examples, 2);
        assert_eq!(summary.val_example_ids, ["V01", "V02"]);
        assert_eq!(summary.max_metric_calls, 20);

        // Seed entry is present and marked is_seed = true
        assert_eq!(summary.candidates.len(), 1);
        let seed_entry = &summary.candidates[0];
        assert!(seed_entry.is_seed);
        assert!(seed_entry.is_accepted);

        // Val subscores match what we wrote
        let subscores = seed_entry.val_subscores.as_ref().unwrap();
        assert!((subscores["V01"] - 0.8).abs() < 1e-9);
        assert!((subscores["V02"] - 0.6).abs() < 1e-9);

        // Aggregate score
        let agg = seed_entry.val_aggregate_score.unwrap();
        assert!((agg - 0.7).abs() < 1e-9, "aggregate should be (0.8+0.6)/2 = 0.7, got {agg}");

        // Seed is on the Pareto front (only candidate)
        assert!(seed_entry.is_on_pareto_front);

        // pareto_front_per_val_example should map each val ID to [0]
        assert_eq!(summary.pareto_front_per_val_example["V01"], vec![0]);
        assert_eq!(summary.pareto_front_per_val_example["V02"], vec![0]);
    }
}

/// Convert days since Unix epoch (1970-01-01) to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z % 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
