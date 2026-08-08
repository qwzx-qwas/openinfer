//! Dependency-free Stage 6 ownership/dispatch contract.
//!
//! This file is intentionally runnable with bare `rustc --test`: it guards the
//! chunk-level resource lifecycle and fail-closed seam even when the full CUDA
//! workspace build is unavailable on a host without the Qwen3.5 toolchain.

#![allow(dead_code)]

const BOUNDARY_TOKENS: [usize; 5] = [1, 63, 64, 65, 128];
const LINEAR_LAYERS_4B: usize = 24;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Backend {
    Triton,
    FlashInfer,
}

impl Default for Backend {
    fn default() -> Self {
        Self::Triton
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StateMode {
    Separate,
    InPlace,
}

fn validate_state_mode(initial: u64, final_state: u64, mode: StateMode) -> Result<(), String> {
    match mode {
        StateMode::Separate if initial == final_state => {
            Err("separate mode requires different pointers".into())
        }
        StateMode::InPlace if initial != final_state => {
            Err("in-place mode requires exact pointer alias".into())
        }
        _ => Ok(()),
    }
}

fn double_buffer_fallback_bytes(
    linear_layers: usize,
    value_heads: usize,
    head_dim: usize,
) -> usize {
    linear_layers * value_heads * head_dim * head_dim * std::mem::size_of::<f32>()
}

#[derive(Debug, Default)]
struct ChunkOwnerModel {
    metadata_uploads: usize,
    workspace_allocations: usize,
    tma_descriptor_builds: usize,
    layer_launches: usize,
    active_tokens: Option<usize>,
}

impl ChunkOwnerModel {
    fn begin_chunk(&mut self, tokens: usize) -> Result<(), String> {
        if tokens == 0 {
            return Err("T must be >= 1".into());
        }
        if self.active_tokens.is_some() {
            return Err("previous chunk resources are still active".into());
        }
        self.metadata_uploads += 1;
        self.workspace_allocations += 1;
        self.tma_descriptor_builds += 4;
        self.active_tokens = Some(tokens);
        Ok(())
    }

    fn launch_layer(&mut self, tokens: usize) -> Result<(), String> {
        if self.active_tokens != Some(tokens) {
            return Err("layer launch token count does not match chunk metadata".into());
        }
        self.layer_launches += 1;
        Ok(())
    }

    fn end_chunk(&mut self) -> Result<(), String> {
        if self.active_tokens.take().is_none() {
            return Err("no active chunk".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_default_is_triton() {
        assert_eq!(Backend::default(), Backend::Triton);
    }

    #[test]
    fn boundary_chunks_upload_metadata_and_allocate_workspace_once() {
        let mut owner = ChunkOwnerModel::default();
        for (chunk_index, &tokens) in BOUNDARY_TOKENS.iter().enumerate() {
            owner.begin_chunk(tokens).unwrap();
            for _ in 0..LINEAR_LAYERS_4B {
                owner.launch_layer(tokens).unwrap();
            }
            assert_eq!(owner.metadata_uploads, chunk_index + 1);
            assert_eq!(owner.workspace_allocations, chunk_index + 1);
            assert_eq!(owner.tma_descriptor_builds, (chunk_index + 1) * 4);
            owner.end_chunk().unwrap();
        }
        assert_eq!(
            owner.layer_launches,
            BOUNDARY_TOKENS.len() * LINEAR_LAYERS_4B
        );
    }

    #[test]
    fn token_mismatch_and_nested_chunk_fail_closed() {
        let mut owner = ChunkOwnerModel::default();
        assert!(owner.begin_chunk(0).is_err());
        owner.begin_chunk(64).unwrap();
        assert!(owner.begin_chunk(65).is_err());
        assert!(owner.launch_layer(65).is_err());
        owner.end_chunk().unwrap();
        assert!(owner.launch_layer(64).is_err());
    }

    #[test]
    fn separate_and_in_place_modes_enforce_pointer_identity() {
        validate_state_mode(0x1000, 0x2000, StateMode::Separate).unwrap();
        validate_state_mode(0x1000, 0x1000, StateMode::InPlace).unwrap();
        assert!(validate_state_mode(0x1000, 0x1000, StateMode::Separate).is_err());
        assert!(validate_state_mode(0x1000, 0x2000, StateMode::InPlace).is_err());
    }

    #[test]
    fn quantified_double_buffer_fallback_is_not_the_default() {
        assert_eq!(double_buffer_fallback_bytes(24, 32, 128), 48 * 1024 * 1024);
        assert_eq!(double_buffer_fallback_bytes(24, 48, 128), 72 * 1024 * 1024);
        assert_eq!(Backend::default(), Backend::Triton);
    }

    #[test]
    fn source_places_chunk_resources_outside_layer_loop_and_has_no_fallback() {
        let prefill = include_str!("prefill.rs");
        let owner = include_str!("flashinfer_gdn.rs");
        let production_entry = prefill
            .find("fn prefill_chunk_forward(")
            .expect("production prefill entry exists");
        let explicit_triton = prefill[production_entry..]
            .find("GdnPrefillBackendSeam::Triton")
            .expect("production entry explicitly selects Triton");
        let resource_creation = prefill
            .find("let mut gdn_scratch = match gdn_backend")
            .expect("chunk resource selection exists");
        let layer_loop = prefill
            .find("for (layer_idx, layer) in self.layers.iter().enumerate()")
            .expect("layer loop exists");
        assert!(explicit_triton > 0);
        assert!(resource_creation < layer_loop);
        assert_eq!(owner.matches("clone_htod(&[0_i64, cu_end])").count(), 1);
        assert!(prefill.contains("let backend = self.flashinfer_gdn()?;"));
        assert!(!prefill.contains("unwrap_or(GdnPrefillBackendSeam::Triton)"));
        assert!(!prefill.contains("std::env::var(\"OPENINFER_GDN_BACKEND\")"));
    }

    #[test]
    fn real_test_benchmark_seam_runs_both_named_backends_on_distinct_state() {
        let prefill = include_str!("prefill.rs");
        let runtime = include_str!("lib.rs");
        let triton_state = prefill
            .find("let mut triton_state = self.new_gdn_prefill_benchmark_state()?")
            .expect("comparison allocates an independent Triton request state");
        let flashinfer_state = prefill
            .find("let mut flashinfer_state = self.new_gdn_prefill_benchmark_state()?")
            .expect("comparison allocates an independent FlashInfer request state");
        assert_ne!(triton_state, flashinfer_state);
        assert!(prefill.contains("run_triton_gdn_prefill_benchmark_chunk("));
        assert!(prefill.contains("run_flashinfer_gdn_prefill_benchmark_chunk("));
        assert!(prefill.contains("pub fn compare_gdn_prefill_backends("));
        assert!(!prefill.contains("pub fn run_gdn_prefill_benchmark_chunk("));
        assert!(!runtime.contains("pub use crate::flashinfer_gdn::GdnPrefillBackendSeam;"));
        assert!(runtime.contains("pub use crate::prefill::GdnPrefillBenchmarkState;"));
    }

    #[test]
    fn native_prepare_status_is_consumed_once_at_chunk_boundary() {
        let prefill = include_str!("prefill.rs");
        let owner = include_str!("flashinfer_gdn.rs");
        let prepare = prefill
            .find("gated_delta_rule_prefill_native_prepare_into(")
            .expect("FlashInfer branch invokes native prepare");
        let launch = prefill[prepare..]
            .find("resources.launch_in_place(")
            .expect("FlashInfer branch launches after prepare");
        assert!(launch > 0);
        let layer_loop = prefill
            .find("for (layer_idx, layer) in self.layers.iter().enumerate()")
            .expect("model layer loop exists");
        let status_check = prefill
            .find("resources.ensure_prepare_inputs_finite(&self.ctx)?")
            .expect("chunk boundary consumes sticky prepare status");
        assert!(status_check > layer_loop);
        assert_eq!(
            owner
                .matches("clone_dtoh(&self.prepare.non_finite_status)")
                .count(),
            1
        );
        assert!(owner.contains("native GDN prepare rejected non-finite qkv/gate input"));
    }

    #[test]
    fn fixed_pointer_table_owner_is_not_part_of_prefill_seam() {
        let prefill = include_str!("prefill.rs");
        let recurrent_state = include_str!("recurrent_state.rs");
        assert!(!prefill.contains("LinearStatePointerTables"));
        assert!(recurrent_state.contains("pub(crate) struct LinearStatePointerTables"));
        assert!(recurrent_state.contains("from_recurrent_refs"));
    }
}
