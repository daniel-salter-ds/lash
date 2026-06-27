//! Synthetic validation task for GEPA harness testing.
//!
//! No external infrastructure required. Dataset is hardcoded (20 examples).
//! Scoring is exact substring match (case-sensitive). Uses Lash Standard mode
//! (one model turn per example, no tools).
//!
//! See `ARCHITECTURE.md §5` for the full specification.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use lash::durability::InlineEffectHost;
use lash::persistence::InMemoryAttachmentStore;
use lash::prompt::{PromptContribution, PromptSlot};
use lash::provider::ProviderHandle;
use lash::{ModelSpec, PromptLayerSink, StandardCore, TurnInput};
use lash_trace::TraceContext;

use crate::{
    Candidate, ComponentConstraints, ComponentValue, EvaluationResult, ExampleRun, HarnessExample,
    HarnessOptError, HarnessProject, MutableComponent, OptimizationRun, Result, RunArtifacts,
    Split,
};

// ---------------------------------------------------------------------------
// Component ID and seed prompt
// ---------------------------------------------------------------------------

pub const SYNTH_QA_COMPONENT_ID: &str = "synth_qa.system_instruction";
/// Deliberately brief seed (§5.5). Fails on format-sensitive examples (type S) where
/// a verbose response may not contain the exact expected substring (e.g. "2.72", "es").
const SYNTH_SEED_PROMPT: &str = "Answer the question.";

// ---------------------------------------------------------------------------
// Dataset — 20 examples (15 train, 5 val)
// ---------------------------------------------------------------------------

fn build_train_examples() -> Vec<HarnessExample> {
    vec![
        ex("T01", Split::Train, "What is the ISO 3166-1 alpha-2 country code for Japan?",      "JP",            "format_sensitive"),
        ex("T02", Split::Train, "What is the ISO 3166-1 alpha-2 country code for Germany?",    "DE",            "format_sensitive"),
        ex("T03", Split::Train, "What is the ISO 3166-1 alpha-2 country code for Brazil?",     "BR",            "format_sensitive"),
        ex("T04", Split::Train, "What is Euler's number e rounded to two decimal places?",     "2.72",          "format_sensitive"),
        ex("T05", Split::Train, "What is the square root of 5 rounded to two decimal places?", "2.24",          "format_sensitive"),
        ex("T06", Split::Train, "What is the ISO 639-1 two-letter language code for Spanish?", "es",            "format_sensitive"),
        ex("T07", Split::Train, "How many sides does a hexagon have?",                          "6",             "factual"),
        ex("T08", Split::Train, "What is the largest planet in the Solar System?",              "Jupiter",       "factual"),
        ex("T09", Split::Train, "In what year did World War I begin?",                          "1914",          "factual"),
        ex("T10", Split::Train, "What is the capital of France?",                               "Paris",         "factual"),
        ex("T11", Split::Train, "Who wrote the play Romeo and Juliet?",                         "Shakespeare",   "factual"),
        ex("T12", Split::Train, "How many bones are in the adult human body?",                  "206",           "factual"),
        ex("T13", Split::Train, "What is the smallest prime number?",                           "2",             "factual"),
        ex("T14", Split::Train, "In what year did Neil Armstrong first walk on the Moon?",      "1969",          "factual"),
        ex("T15", Split::Train, "What is the longest river in Africa?",                         "Nile",          "factual"),
    ]
}

fn build_val_examples() -> Vec<HarnessExample> {
    vec![
        ex("V01", Split::Val, "What is the ISO 3166-1 alpha-2 country code for France?",                                               "FR",            "format_sensitive"),
        ex("V02", Split::Val, "What is the square root of 7 rounded to two decimal places?",                                           "2.65",          "format_sensitive"),
        ex("V03", Split::Val, "How many players are on a standard association football team?",                                          "11",            "factual"),
        ex("V04", Split::Val, "What is the freezing point of water in degrees Celsius?",                                               "0",             "factual"),
        ex("V05", Split::Val, "What is the name of the biological process by which plants produce food using sunlight?", "photosynthesis","factual"),
    ]
}

fn ex(id: &str, split: Split, question: &str, answer: &str, ty: &str) -> HarnessExample {
    HarnessExample {
        id: id.to_string(),
        split,
        input: json!({ "question": question }),
        expected: Some(json!({ "answer": answer })),
        metadata: BTreeMap::from([("type".into(), json!(ty))]),
    }
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

fn score_synthetic_example(response: &str, expected: &Option<Value>, example_id: &str, split: &Split) -> EvaluationResult {
    let answer = expected
        .as_ref()
        .and_then(|v| v.get("answer"))
        .and_then(Value::as_str)
        .unwrap_or("");

    // Exact-match after whitespace trimming (case-sensitive).
    // This requires the model to output ONLY the expected value — no surrounding
    // prose. The seed "Answer the question." deliberately fails here because
    // modern LLMs produce full-sentence answers. GEPA must discover a more
    // precise instruction such as "Give only the exact answer value, nothing else."
    let passed = response.trim() == answer;
    let score = if passed { 1.0_f64 } else { 0.0_f64 };

    let feedback = if passed {
        format!("Correct. Trimmed response exactly equals the expected answer '{answer}'.")
    } else {
        format!(
            "Incorrect. Expected trimmed response to equal '{answer}' exactly \
             (case-sensitive), but got: '{}'",
            response.trim().chars().take(200).collect::<String>()
        )
    };

    EvaluationResult {
        example_id: example_id.to_string(),
        split: split.clone(),
        score,
        passed: Some(passed),
        feedback: Some(feedback),
        metrics: BTreeMap::new(),
        diagnostics: BTreeMap::from([
            ("answer_length".into(),   json!(answer.len())),
            ("response_length".into(), json!(response.len())),
            ("turn_outcome".into(),    json!(if passed { "correct" } else { "incorrect" })),
            ("tool_call_count".into(), json!(0)),
            ("error_count".into(),     json!(0)),
        ]),
    }
}

// ---------------------------------------------------------------------------
// SyntheticTask
// ---------------------------------------------------------------------------

/// GEPA harness validation task using Standard-mode Lash sessions.
///
/// Holds the active provider so sessions can be opened per `(candidate, example)` pair.
/// The model slug is taken from `run.config.task_lm` at evaluation time.
pub struct SyntheticTask {
    provider: ProviderHandle,
    train: Vec<HarnessExample>,
    val: Vec<HarnessExample>,
}

impl SyntheticTask {
    /// Construct a `SyntheticTask` from a pre-resolved provider handle.
    /// Call `resolve_provider()` in `main.rs` to obtain the handle, then pass it here.
    pub fn new(provider: ProviderHandle) -> Self {
        Self {
            provider,
            train: build_train_examples(),
            val: build_val_examples(),
        }
    }

    async fn run_standard_session(
        &self,
        instruction: &str,
        question: &str,
        model_slug: &str,
        _context: TraceContext,
        _cancellation: CancellationToken,
    ) -> Result<String> {
        let model_slug = if model_slug.is_empty() {
            "anthropic/claude-haiku-4-5"
        } else {
            model_slug
        };

        let model_spec = ModelSpec::from_token_limits(model_slug, None, 200_000, None)
            .map_err(|e| HarnessOptError::Harness(format!("invalid model spec: {e}")))?;

        let core = StandardCore::builder()
            .provider(self.provider.clone())
            .model(model_spec)
            .effect_host(Arc::new(InlineEffectHost::default()))
            .attachment_store(Arc::new(InMemoryAttachmentStore::new()))
            .process_env_store(Arc::new(
                lash::persistence::InMemoryProcessExecutionEnvStore::new(),
            ))
            .build()
            .map_err(|e| HarnessOptError::Harness(e.to_string()))?;

        let session = core
            .session("synth-qa-eval")
            .replace_prompt_slot(
                PromptSlot::Guidance,
                [PromptContribution::guidance("synth-qa", instruction)],
            )
            .open()
            .await
            .map_err(|e| HarnessOptError::Harness(e.to_string()))?;

        let output = session
            .turn(TurnInput::text(question))
            .run()
            .await
            .map_err(|e| HarnessOptError::Harness(e.to_string()))?;

        Ok(output
            .assistant_message()
            .unwrap_or_default()
            .to_string())
    }
}

#[async_trait]
impl HarnessProject for SyntheticTask {
    async fn seed_candidate(&self) -> Result<Candidate> {
        let constraints = ComponentConstraints {
            max_chars: Some(2000),
            preserve_terms: vec![],
            forbidden_terms: vec![],
            format_hint: Some("System instruction for question answering".to_string()),
        };
        let component = MutableComponent {
            id: SYNTH_QA_COMPONENT_ID.to_string(),
            description: Some(
                "System instruction injected as the guidance prompt slot".to_string(),
            ),
            value: ComponentValue::Text {
                text: SYNTH_SEED_PROMPT.to_string(),
            },
            constraints,
        };
        let mut mutable_components = BTreeMap::new();
        mutable_components.insert(SYNTH_QA_COMPONENT_ID.to_string(), component);
        Ok(Candidate {
            id: "seed".to_string(),
            parent_id: None,
            mutable_components,
            immutable_context: BTreeMap::new(),
            metadata: BTreeMap::from([("project".into(), json!("synth-qa"))]),
        })
    }

    async fn trainset(&self) -> Result<Vec<HarnessExample>> {
        Ok(self.train.clone())
    }

    async fn valset(&self) -> Result<Vec<HarnessExample>> {
        Ok(self.val.clone())
    }

    async fn evaluate_example(
        &self,
        run: &OptimizationRun,
        candidate: &Candidate,
        example: &HarnessExample,
        context: TraceContext,
        cancellation: CancellationToken,
    ) -> Result<ExampleRun> {
        // Step 1 — Extract system instruction
        let component = candidate
            .mutable_components
            .get(SYNTH_QA_COMPONENT_ID)
            .ok_or_else(|| {
                HarnessOptError::Harness(format!("missing component '{SYNTH_QA_COMPONENT_ID}'"))
            })?;
        let instruction = match &component.value {
            ComponentValue::Text { text } => text.as_str(),
            _ => {
                return Err(HarnessOptError::Harness(format!(
                    "component '{SYNTH_QA_COMPONENT_ID}' must be ComponentValue::Text"
                )))
            }
        };

        // Step 2 — Extract question text
        let question = example
            .input
            .get("question")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                HarnessOptError::Harness(format!(
                    "example '{}' is missing input['question']",
                    example.id
                ))
            })?;

        // Step 3 — Run Standard mode Lash session
        let model_slug = run.config.task_lm.as_str();
        let response_text = self
            .run_standard_session(instruction, question, model_slug, context, cancellation)
            .await?;

        // Step 4 — Score
        let result = score_synthetic_example(
            &response_text,
            &example.expected,
            &example.id,
            &example.split,
        );

        // Step 5 — Return ExampleRun
        Ok(ExampleRun {
            example: example.clone(),
            result,
            trace: None,
            artifacts: RunArtifacts::default(),
            metric_calls: 1,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests — no LLM calls required
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use async_trait::async_trait;
    use serde_json::{Value, json};
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::strategies::gepa::{ReflectiveGepaStrategy, ReflectiveProposalRequest, ReflectiveProposer};
    use crate::{
        ComponentValue, EvaluationResult, ExampleRun, HarnessExample, HarnessOptimizer,
        HarnessProject, MutableComponent, OptimizationConfig, OptimizationRun,
        ProjectHarnessRunner, Result, RunArtifacts, Split,
    };
    use lash_trace::TraceContext;

    // -----------------------------------------------------------------------
    // Scoring — pure unit tests (exact-match-trimmed semantics)
    // -----------------------------------------------------------------------

    #[test]
    fn score_passes_on_exact_trimmed_match() {
        let expected = Some(json!({ "answer": "JP" }));
        // Bare value with trailing newline → should pass after trim
        let result = score_synthetic_example("JP\n", &expected, "T01", &Split::Train);
        assert_eq!(result.score, 1.0);
        assert_eq!(result.passed, Some(true));
    }

    #[test]
    fn score_fails_when_response_is_verbose() {
        // This is the key property: a full-sentence answer fails, which is what
        // drives the score gap the GEPA loop must close.
        let expected = Some(json!({ "answer": "JP" }));
        let result = score_synthetic_example("The country code is JP.", &expected, "T01", &Split::Train);
        assert_eq!(result.score, 0.0);
        assert_eq!(result.passed, Some(false));
    }

    #[test]
    fn score_is_case_sensitive_exact_match() {
        let expected = Some(json!({ "answer": "es" }));
        // "ES" ≠ "es" even when trimmed
        let result = score_synthetic_example("ES", &expected, "T06", &Split::Train);
        assert_eq!(result.score, 0.0);
        // exact "es" passes
        let result2 = score_synthetic_example("es", &expected, "T06", &Split::Train);
        assert_eq!(result2.score, 1.0);
    }

    #[test]
    fn score_empty_answer_is_safe_fallback() {
        // answer = "" → "".trim() == "" → response.trim() == "" only if response is whitespace-only
        // Safe: any non-empty response → trim ≠ "" → 0.0. Empty response → trim == "" → 1.0.
        let result = score_synthetic_example("", &None, "X", &Split::Val);
        assert_eq!(result.score, 1.0, "empty response with empty answer should pass");
        let result2 = score_synthetic_example("something", &None, "X", &Split::Val);
        assert_eq!(result2.score, 0.0, "non-empty response should not equal empty answer");
    }

    #[test]
    fn score_populates_diagnostics() {
        let expected = Some(json!({ "answer": "Paris" }));
        let result = score_synthetic_example("Paris", &expected, "T10", &Split::Train);
        assert_eq!(result.diagnostics.get("answer_length"), Some(&json!(5)));
        assert_eq!(result.diagnostics.get("tool_call_count"), Some(&json!(0)));
    }

    // -----------------------------------------------------------------------
    // §5.5 seed candidate structure
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn seed_candidate_has_correct_component_id_and_seed_text() {
        // Build using a stub provider (not called); only testing the candidate structure.
        // We cannot construct a real ProviderHandle in tests without a provider, so
        // we test the helper functions directly.
        let train = build_train_examples();
        let val = build_val_examples();

        // Verify component: seed text matches §5.5
        // Build the candidate manually without needing a provider.
        let constraints = ComponentConstraints {
            max_chars: Some(2000),
            preserve_terms: vec![],
            forbidden_terms: vec![],
            format_hint: Some("System instruction for question answering".to_string()),
        };
        let mut comps = BTreeMap::new();
        comps.insert(
            SYNTH_QA_COMPONENT_ID.to_string(),
            MutableComponent {
                id: SYNTH_QA_COMPONENT_ID.to_string(),
                description: None,
                value: ComponentValue::Text { text: SYNTH_SEED_PROMPT.to_string() },
                constraints,
            },
        );

        let text = match &comps[SYNTH_QA_COMPONENT_ID].value {
            ComponentValue::Text { text } => text.as_str(),
            _ => panic!("wrong variant"),
        };
        assert_eq!(text, "Answer the question.");
        assert_eq!(train.len(), 15, "train set must have 15 examples");
        assert_eq!(val.len(), 5, "val set must have 5 examples");
    }

    // -----------------------------------------------------------------------
    // §5.3 dataset completeness
    // -----------------------------------------------------------------------

    #[test]
    fn dataset_has_correct_ids_and_splits() {
        let train = build_train_examples();
        let val = build_val_examples();

        let train_ids: Vec<&str> = train.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(
            train_ids,
            ["T01","T02","T03","T04","T05","T06","T07","T08","T09","T10","T11","T12","T13","T14","T15"]
        );
        let val_ids: Vec<&str> = val.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(val_ids, ["V01","V02","V03","V04","V05"]);

        assert!(train.iter().all(|e| e.split == Split::Train));
        assert!(val.iter().all(|e| e.split == Split::Val));
    }

    #[test]
    fn dataset_examples_have_question_and_answer_fields() {
        for ex in build_train_examples().iter().chain(build_val_examples().iter()) {
            assert!(ex.input.get("question").and_then(Value::as_str).is_some(),
                "example {} missing input['question']", ex.id);
            assert!(
                ex.expected.as_ref().and_then(|v| v.get("answer")).and_then(Value::as_str).is_some(),
                "example {} missing expected['answer']", ex.id
            );
        }
    }

    // -----------------------------------------------------------------------
    // Integration test — full optimization loop, no LLM calls
    //
    // Uses a mock project whose `evaluate_example` is deterministic:
    //   • Seed prompt "Answer the question." → format-sensitive examples score 0.0
    //   • Any prompt containing "concise" → all examples score 1.0
    // Uses a mock proposer that always patches the seed to "Be concise."
    // Verifies that one candidate is accepted and val scores improve.
    // -----------------------------------------------------------------------

    struct DeterministicSyntheticProject;

    #[async_trait]
    impl HarnessProject for DeterministicSyntheticProject {
        async fn seed_candidate(&self) -> Result<crate::Candidate> {
            let mut comps = BTreeMap::new();
            comps.insert(
                SYNTH_QA_COMPONENT_ID.to_string(),
                MutableComponent {
                    id: SYNTH_QA_COMPONENT_ID.to_string(),
                    description: None,
                    value: ComponentValue::Text { text: SYNTH_SEED_PROMPT.to_string() },
                    constraints: ComponentConstraints {
                        max_chars: Some(2000),
                        preserve_terms: vec![],
                        forbidden_terms: vec![],
                        format_hint: None,
                    },
                },
            );
            Ok(crate::Candidate {
                id: "seed".to_string(),
                parent_id: None,
                mutable_components: comps,
                immutable_context: BTreeMap::new(),
                metadata: BTreeMap::new(),
            })
        }

        async fn trainset(&self) -> Result<Vec<HarnessExample>> {
            Ok(build_train_examples())
        }

        async fn valset(&self) -> Result<Vec<HarnessExample>> {
            Ok(build_val_examples())
        }

        async fn evaluate_example(
            &self,
            _run: &OptimizationRun,
            candidate: &crate::Candidate,
            example: &HarnessExample,
            _ctx: TraceContext,
            _cancel: CancellationToken,
        ) -> Result<ExampleRun> {
            let instruction = match &candidate.mutable_components[SYNTH_QA_COMPONENT_ID].value {
                ComponentValue::Text { text } => text.clone(),
                _ => String::new(),
            };
            let ty = example.metadata.get("type").and_then(Value::as_str).unwrap_or("factual");
            // Simulate: format_sensitive examples fail with seed; pass when prompt says "concise"
            let score: f64 = if ty == "format_sensitive" && !instruction.contains("concise") {
                0.0
            } else {
                1.0
            };
            Ok(ExampleRun {
                example: example.clone(),
                result: EvaluationResult {
                    example_id: example.id.clone(),
                    split: example.split.clone(),
                    score,
                    passed: Some(score >= 1.0),
                    feedback: None,
                    metrics: BTreeMap::new(),
                    diagnostics: BTreeMap::new(),
                },
                trace: None,
                artifacts: RunArtifacts::default(),
                metric_calls: 1,
            })
        }
    }

    struct ConciseProposer;

    #[async_trait]
    impl ReflectiveProposer for ConciseProposer {
        async fn propose_json(
            &self,
            request: ReflectiveProposalRequest,
            _cancel: CancellationToken,
        ) -> crate::Result<Value> {
            Ok(json!({
                "proposals": [{
                    "parent_candidate_id": request.parent_candidate_id,
                    "patches": [{
                        "kind": "replace_value",
                        "component_id": SYNTH_QA_COMPONENT_ID,
                        "value": { "kind": "text", "text": "Be concise. Answer the question with the exact value only." }
                    }]
                }]
            }))
        }
    }

    #[tokio::test]
    async fn synthetic_task_optimization_accepts_better_prompt_without_llm() {
        let temp = std::env::temp_dir()
            .join(format!("synth-test-{}", uuid::Uuid::new_v4()));
        let project = Arc::new(DeterministicSyntheticProject);
        let runner = ProjectHarnessRunner::new(project.clone());
        let strategy = ReflectiveGepaStrategy::new(ConciseProposer);
        let optimizer = HarnessOptimizer::new(runner, strategy);
        let run = OptimizationRun {
            run_id: "test-run".to_string(),
            experiment_id: "synth-unit".to_string(),
            run_dir: temp,
            config: OptimizationConfig {
                max_metric_calls: 30,
                max_iterations: Some(3),
                minibatch_size: 3,
                max_concurrency: 1,
                skip_perfect_score: false,
                per_example_timeout_secs: None,
                ..OptimizationConfig::default()
            },
        };

        let state = optimizer
            .run(
                run,
                project.seed_candidate().await.unwrap(),
                project.trainset().await.unwrap(),
                project.valset().await.unwrap(),
                CancellationToken::new(),
            )
            .await
            .unwrap();

        // At least one accepted proposal (the "concise" prompt should beat seed on format-sensitive)
        assert!(state.accepted_proposals >= 1,
            "expected at least 1 accepted proposal, got {}", state.accepted_proposals);
        // Best candidate should score higher than seed (seed scores 0.0 on 6 format-sensitive train examples)
        let best = state.best().expect("must have a best candidate");
        assert!(best.mean_score() > 0.5,
            "best mean score should be > 0.5, got {}", best.mean_score());
    }
}
