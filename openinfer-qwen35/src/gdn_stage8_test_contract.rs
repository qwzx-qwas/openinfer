//! Source-level contracts for the Stage 8 HF integration seam.
//!
//! GPU/HF accuracy remains an ignored SM120 integration test. These guards
//! keep the production constructor Triton-only and make it difficult for the
//! explicit candidate test to silently lose backend identity or fallback.

#[test]
fn production_executor_remains_triton_only() {
    let executor = include_str!("executor.rs");
    let production = executor
        .find("pub fn from_runtime(")
        .expect("production executor constructor exists");
    let candidate = executor
        .find("pub fn from_runtime_with_flashinfer_gdn(")
        .expect("named FlashInfer test constructor exists");
    assert!(production < candidate);
    let production_body = &executor[production..candidate];
    assert!(production_body.contains("ExecutorGdnPrefillBackend::Triton"));
    assert!(!production_body.contains("install_flashinfer_gdn_for_benchmark"));

    let runtime = include_str!("lib.rs");
    assert!(!runtime.contains("pub use crate::flashinfer_gdn::GdnPrefillBackendSeam"));
    assert!(!executor.contains("OPENINFER_GDN_BACKEND"));
}

#[test]
fn candidate_executor_is_explicit_and_fail_closed() {
    let executor = include_str!("executor.rs");
    let candidate = executor
        .find("pub fn from_runtime_with_flashinfer_gdn(")
        .expect("named FlashInfer test constructor exists");
    let from_model = executor[candidate..]
        .find("    fn from_model(")
        .map(|offset| candidate + offset)
        .expect("candidate constructor has a bounded body");
    let candidate_body = &executor[candidate..from_model];
    assert!(candidate_body.contains("install_flashinfer_gdn_for_benchmark(manifest_path)?"));
    assert!(candidate_body.contains("ExecutorGdnPrefillBackend::FlashInfer"));
    assert!(!candidate_body.contains("unwrap_or"));
    assert!(!candidate_body.contains("Triton"));

    let prefill = include_str!("prefill.rs");
    assert!(prefill.contains("GdnPrefillRuntimeEvidence"));
    assert!(prefill.contains("GdnPrefillRuntimeEvidenceHandle"));
    assert!(prefill.contains("successful_launches.load(Ordering::Relaxed)"));
    let owner = include_str!("flashinfer_gdn.rs");
    let launch = owner
        .find("unsafe { launch.launch(config) }")
        .expect("FlashInfer driver launch exists");
    let count = owner
        .find("self.successful_launches.fetch_add(1, Ordering::Relaxed)")
        .expect("successful launch counter exists");
    assert!(launch < count, "only successful launches may be counted");
}

#[test]
fn hf_gate_requires_identity_and_paired_reporting() {
    let gate = include_str!("../tests/hf_golden_gate.rs");
    assert!(gate.contains("OPENINFER_QWEN35_FLASHINFER_GDN_MANIFEST"));
    assert!(gate.contains("flashinfer_gdn_and_triton_match_hf_short_golden"));
    assert!(gate.contains("flashinfer_gdn_and_triton_match_hf_long_golden"));
    assert!(gate.contains("require_flashinfer_launches"));
    assert!(gate.contains("FlashInfer-Triton delta"));
    assert!(gate.contains("evidence.successful_launches > previous_launches"));
}

#[test]
fn scheduler_handoff_gate_is_explicit_and_proves_launches() {
    let scheduler = include_str!("scheduler.rs");
    let production = scheduler
        .find("pub fn start_with_capacity(")
        .expect("production scheduler entry exists");
    let candidate = scheduler
        .find("fn start_with_capacity_flashinfer_gdn(")
        .expect("candidate scheduler entry exists");
    assert!(production < candidate);
    assert!(scheduler[production..candidate].contains("Qwen35SchedulerPolicy::Off"));
    assert!(scheduler[candidate..].contains("GdnPrefillBackendSeam::FlashInfer"));
    assert!(scheduler.contains("batch_prefill_logits_flashinfer"));
    assert!(scheduler.contains("unified_step_with_gdn_backend"));
    let runtime = include_str!("lib.rs");
    assert!(runtime.contains("pub fn start_engine_with_flashinfer_gdn_for_accuracy("));

    let chunked = include_str!("../tests/chunked_prefill.rs");
    assert!(chunked.contains("flashinfer_gdn_chunked_prefill_matches_unchunked_prefill"));
    assert!(chunked.contains("evidence.snapshot().successful_launches > 0"));

    let e2e = include_str!("../tests/e2e_scheduler.rs");
    assert!(e2e.contains("test_e2e_qwen35_scheduler_flashinfer_gdn"));
    assert!(e2e.contains("final_evidence.successful_launches > 0"));
}
