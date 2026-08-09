//! Host contract and model-local owner for the experimental FlashInfer SM120
//! GDN prefill artifact.
//!
//! Production prefill still selects Triton. This module owns the crate-private
//! Stage 6 test/benchmark seam: one chunk-scoped metadata/workspace allocation,
//! explicit FlashInfer launch, and separate versus exact-pointer-alias state
//! endpoints. There is no environment-controlled dispatch or fallback.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use cudarc::driver::CudaFunction;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys;
use cudarc::nvrtc::Ptx;
use openinfer_core::tensor::DeviceContext;
use openinfer_core::tensor::HiddenStates;
use openinfer_kernels::ffi::FlashInferGdnPrefillArgs;
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;
use sha2::Digest;
use sha2::Sha256;

use crate::config::Config35;
use crate::prefill_buffers::GdnPrepareScratch35;
use crate::weights::Qwen35Model;

const SCHEMA_VERSION: u32 = 1;
const ARTIFACT_KIND: &str = "flashinfer_cute_gdn_prefill_ptx";
const TARGET_ARCH: &str = "sm_120a";
const DRIVER_JIT_TARGET: &str = "compute_120a";
const FLASHINFER_COMMIT: &str = "19f1a41e6b21f0c422d775e377b6fdf9a1fc9d23";
const PATCH_SHA256: &str = "c9ccea6881979c8bb21a29816cbe1e6782819c70567093ced76e475becca3d7a";
const KERNEL_SOURCE_SHA256: &str =
    "2ef4dcecf7c87ae1cc54bb1938d418af45dc47cd5eeaf1edd0cee2b977d0d5a0";
const PATCH_SET_SHA256: &str = "fbb15a0135095a3576d9c6439c0496bda5361d2af36028002bd264a3965ba992";
const REQUIREMENTS_LOCK_SHA256: &str =
    "2051b988e4ff3213f5115c688239d1271ea100f43646fa476e0148ed020a5a3f";
const GENERATOR_SHA256: &str = "1973974a91749e45e1bfcb7861d383e6b4c2a5940b4e777108fe3a17889499c7";
const ENTRY_SYMBOL: &str = "kernel_cutlass_kernel_flashinfergdn_kernelsdelta_rule_dsldelta_rule_sm120_FullyFusedDeltaRuleSm120_object_at__tensorptrf32gmemalign16o1_tensorptrf32gmemalign16o1_CopyAtom_ThrID10_TVLayout_0";
const ARTIFACT_SHA256: &str = "225646b26dab488cdfd64dcf3fe189ba4b7ccaf2ba735eb7b68a47d13db96b68";
const ARTIFACT_SIZE_BYTES: u64 = 549_690;
const WORKSPACE_BYTES_PER_SM: u64 = 128;
const WORKSPACE_ALIGNMENT: u64 = 128;
const THREADS_PER_BLOCK: u32 = 384;
// `cute.size_in_bytes(SharedStorage)` for the frozen Stage 3 specialization.
// The value is part of the naked PTX launch ABI and is checked against the
// frozen source shape by the Stage 6 contract tests.
const DYNAMIC_SHARED_MEMORY_BYTES: u32 = 100_864;
const TMA_TILE_TOKENS: u32 = 64;

/// Explicit internal seam. The production caller always passes `Triton`;
/// model-local tests and Criterion benches use backend-named methods instead
/// of exposing this enum as a user-selectable backend switch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GdnPrefillBackendSeam {
    Triton,
    FlashInfer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FlashInferStateMode {
    Separate,
    InPlace,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CompactTensorArg {
    pointer: u64,
    elements: i64,
}

// SAFETY: this is the frozen CuTe compact-tensor by-value kernel ABI: a CUDA
// device pointer followed by one signed dynamic extent.
unsafe impl DeviceRepr for CompactTensorArg {}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
struct TmaDescriptor {
    opaque: [u64; 16],
}

// SAFETY: CUDA 12.x `CUtensorMap` is a 128-byte, 64-byte-aligned by-value
// kernel argument. `encode_tma_descriptor` initializes every opaque byte.
unsafe impl DeviceRepr for TmaDescriptor {}

#[derive(Clone, Copy, Debug)]
struct GdnTensorMaps {
    q: TmaDescriptor,
    k: TmaDescriptor,
    v: TmaDescriptor,
    output: TmaDescriptor,
    q_pointer: u64,
    k_pointer: u64,
    v_pointer: u64,
    output_pointer: u64,
    tokens: u32,
    geometry: Geometry,
}

/// Chunk-scoped owner shared by every linear-attention layer in that chunk.
///
/// Q/K/V/output addresses are stable for the owner lifetime, so their TMA
/// descriptors, `[0,T]` metadata, and per-SM workspace are created exactly
/// once before the layer loop and reused by all 24 linear layers.
pub(crate) struct FlashInferGdnChunkResources {
    pub(crate) prepare: GdnPrepareScratch35,
    pub(crate) output: HiddenStates,
    workspace: CudaSlice<u8>,
    cu_seqlens: CudaSlice<i64>,
    tensor_maps: GdnTensorMaps,
    workspace_bytes: u64,
    tokens: usize,
    geometry: Geometry,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: u32,
    artifact_kind: String,
    variant: String,
    target: Target,
    dtypes: BTreeMap<String, String>,
    geometry: Geometry,
    tokens: Tokens,
    abi: Abi,
    artifact: Artifact,
    source: Source,
    workspace: Workspace,
    distribution: Distribution,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
struct Geometry {
    h_q: u32,
    h_k: u32,
    h_v: u32,
    head_dim: u32,
}

#[derive(Debug, Deserialize)]
struct Target {
    arch: String,
    driver_jit_target: String,
}

#[derive(Debug, Deserialize)]
struct Tokens {
    extent: Value,
    minimum: u32,
    divisibility: u32,
}

#[derive(Debug, Deserialize)]
struct Abi {
    entry_symbol: String,
    geometry_binding: String,
    q_view: Value,
    k_view: Value,
    v_view: Value,
    o_view: Value,
    state_layout: String,
}

#[derive(Debug, Deserialize)]
struct Artifact {
    file: String,
    format: String,
    sha256: String,
    size_bytes: u64,
    entry_symbols: Vec<String>,
    absolute_path_scan: String,
}

#[derive(Debug, Deserialize)]
struct Source {
    flashinfer_commit: String,
    hkv_state_index_patch_applied: bool,
    hkv_state_index_patch_sha256: String,
    kernel_source_sha256: String,
    patch_set_sha256: String,
    requirements_lock_sha256: String,
    generator_sha256: String,
}

#[derive(Debug, Deserialize)]
struct Workspace {
    kind: String,
    formula: String,
    bytes_per_sm: u64,
    alignment_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
struct Distribution {
    cuda_driver_jit_required: bool,
    serving_requires_cute_dsl: bool,
    serving_requires_python: bool,
    production_eligible: bool,
}

#[derive(Clone, Debug)]
struct ValidatedArtifact {
    manifest_path: PathBuf,
    ptx_path: PathBuf,
    geometry: Geometry,
    variant: String,
    entry_symbol: String,
    workspace_bytes_per_sm: u64,
    workspace_alignment: u64,
}

/// Opaque model-local owner. `CudaFunction` retains its `CudaModule`, and the
/// module retains the same `Arc<CudaContext>` as the model, so module unload
/// necessarily occurs before the last context reference is released.
#[derive(Debug)]
pub(super) struct FlashInferGdnBackend {
    function: CudaFunction,
    artifact: ValidatedArtifact,
    creation_context: usize,
    device_ordinal: usize,
    sm_count: u32,
}

#[derive(Clone, Copy, Debug)]
struct ValidatedLaunch {
    scale: f32,
    grid_x: u32,
    workspace_required: u64,
}

impl FlashInferGdnBackend {
    /// Load a pinned artifact into this model's CUDA context. This API remains
    /// crate-private until the GPU gates and full prefill integration pass.
    pub(super) fn load(ctx: &DeviceContext, manifest_path: &Path) -> Result<Self> {
        let (major, minor) = ctx.ctx.compute_capability()?;
        ensure!(
            (major, minor) == (12, 0),
            "FlashInfer GDN artifact requires SM120, device {} reports SM{major}{minor}",
            ctx.device_ordinal
        );

        ctx.ctx.bind_to_thread()?;
        let creation_context = current_context_identity()?;
        let expected_context = ctx.ctx.cu_ctx() as usize;
        ensure!(
            creation_context == expected_context,
            "CUDA current-context mismatch while loading GDN artifact: expected {expected_context:#x}, got {creation_context:#x}"
        );

        let (artifact, ptx) = load_and_validate_artifact(manifest_path)?;
        let module = ctx.ctx.load_module(Ptx::from_src(ptx))?;
        let function = module
            .load_function(&artifact.entry_symbol)
            .with_context(|| format!("missing PTX entry symbol {}", artifact.entry_symbol))?;
        function.set_attribute(
            sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            i32::try_from(DYNAMIC_SHARED_MEMORY_BYTES)
                .expect("frozen GDN dynamic shared-memory size fits i32"),
        )?;
        ensure!(
            function.max_threads_per_block()? >= THREADS_PER_BLOCK as i32,
            "GDN artifact cannot launch its frozen {THREADS_PER_BLOCK}-thread block"
        );
        let sm_count =
            u32::try_from(ctx.ctx.attribute(
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )?)
            .context("negative CUDA multiprocessor count")?;

        Ok(Self {
            function,
            artifact,
            creation_context,
            device_ordinal: ctx.device_ordinal,
            sm_count,
        })
    }

    /// Validate the complete naked-pointer call contract immediately before a
    /// future kernel launch. This does not bind or repair the current context:
    /// a wrong worker/context fails closed.
    fn validate_launch(
        &self,
        ctx: &DeviceContext,
        args: &FlashInferGdnPrefillArgs,
    ) -> Result<ValidatedLaunch> {
        ensure!(
            ctx.device_ordinal == self.device_ordinal,
            "GDN backend belongs to CUDA device {}, launch requested on device {}",
            self.device_ordinal,
            ctx.device_ordinal
        );
        let expected_context = ctx.ctx.cu_ctx() as usize;
        ensure!(
            expected_context == self.creation_context,
            "GDN backend/model CUDA context mismatch: loaded in {:#x}, model has {expected_context:#x}",
            self.creation_context
        );
        let current_context = current_context_identity()?;
        validate_launch_contract(
            args,
            self.artifact.geometry,
            self.artifact.workspace_bytes_per_sm,
            self.artifact.workspace_alignment,
            self.sm_count,
            self.creation_context,
            current_context,
            ctx.stream.cu_stream() as usize,
        )
    }

    fn workspace_required(&self) -> Result<u64> {
        u64::from(self.sm_count)
            .checked_mul(self.artifact.workspace_bytes_per_sm)
            .context("GDN workspace size overflow")
    }

    fn geometry(&self) -> Geometry {
        self.artifact.geometry
    }

    fn launch(
        &self,
        ctx: &DeviceContext,
        args: &FlashInferGdnPrefillArgs,
        maps: &GdnTensorMaps,
        state_mode: FlashInferStateMode,
    ) -> Result<()> {
        let validated = self.validate_launch(ctx, args)?;
        ensure!(
            maps.tokens == args.tokens
                && maps.geometry == self.artifact.geometry
                && maps.q_pointer == args.q
                && maps.k_pointer == args.k
                && maps.v_pointer == args.v
                && maps.output_pointer == args.output,
            "GDN TMA descriptors are stale or belong to another chunk"
        );
        validate_state_mode(args.initial_state, args.state, state_mode)?;

        let gate_elements = i64::from(args.tokens)
            .checked_mul(i64::from(args.h_v))
            .context("GDN alpha/beta extent overflow")?;
        let workspace_elements =
            i64::try_from(args.workspace_bytes).context("GDN workspace extent does not fit i64")?;
        let alpha = CompactTensorArg {
            pointer: args.alpha,
            elements: gate_elements,
        };
        let beta = CompactTensorArg {
            pointer: args.beta,
            elements: gate_elements,
        };
        let workspace = CompactTensorArg {
            pointer: args.workspace,
            elements: workspace_elements,
        };
        let cu_seqlens = CompactTensorArg {
            pointer: args.cu_seqlens,
            elements: i64::from(args.cu_seqlens_len),
        };
        let tokens = args.tokens;
        let state = args.state;
        let initial_state = args.initial_state;
        let scale = validated.scale;
        let h_q = args.h_q;
        let h_k = args.h_k;
        let h_v = args.h_v;
        let sab_heads = h_q.max(h_v);
        let num_sequences = 1_u32;
        let total_checkpoints = 1_u32;
        let checkpoint_every_n_tokens = 0_u32;

        let mut launch = ctx.stream.launch_builder(&self.function);
        launch
            .arg(&alpha)
            .arg(&beta)
            .arg(&maps.q)
            .arg(&tokens)
            .arg(&maps.k)
            .arg(&tokens)
            .arg(&maps.v)
            .arg(&tokens)
            .arg(&maps.output)
            .arg(&tokens)
            .arg(&state)
            .arg(&initial_state)
            .arg(&workspace)
            .arg(&cu_seqlens)
            .arg(&scale)
            .arg(&h_q)
            .arg(&h_k)
            .arg(&h_v)
            .arg(&sab_heads)
            .arg(&num_sequences)
            .arg(&total_checkpoints)
            .arg(&checkpoint_every_n_tokens);
        let config = LaunchConfig {
            grid_dim: (validated.grid_x, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: DYNAMIC_SHARED_MEMORY_BYTES,
        };
        // SAFETY: the exact 22-parameter CuTe ABI is frozen above; manifest,
        // geometry, pointers, context, stream, workspace, and TMA descriptor
        // ownership were all checked immediately before this async launch.
        unsafe { launch.launch(config) }
            .map_err(|error| anyhow::anyhow!("FlashInfer GDN launch failed: {error}"))?;
        Ok(())
    }
}

impl FlashInferGdnChunkResources {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &Config35,
        backend: &FlashInferGdnBackend,
        tokens: usize,
    ) -> Result<Self> {
        ensure!(tokens > 0, "FlashInfer GDN chunk requires T>=1");
        let tokens_u32 = u32::try_from(tokens).context("GDN token count exceeds u32")?;
        let geometry = Geometry {
            h_q: u32::try_from(config.linear_num_key_heads).context("Hq exceeds u32")?,
            h_k: u32::try_from(config.linear_num_key_heads).context("Hk exceeds u32")?,
            h_v: u32::try_from(config.linear_num_value_heads).context("Hv exceeds u32")?,
            head_dim: u32::try_from(config.linear_key_head_dim).context("D exceeds u32")?,
        };
        ensure!(
            geometry == backend.geometry(),
            "model GDN geometry {geometry:?} does not match installed artifact {:?}",
            backend.geometry()
        );
        ensure!(
            config.linear_value_head_dim == config.linear_key_head_dim,
            "FlashInfer GDN candidate requires equal K/V dimensions"
        );

        let mut prepare = GdnPrepareScratch35::new(ctx, config, tokens)?;
        let mut output = HiddenStates::zeros(
            ctx,
            config.linear_num_value_heads * config.linear_value_head_dim,
            tokens,
        )?;
        let workspace_bytes = backend.workspace_required()?;
        let workspace_len =
            usize::try_from(workspace_bytes).context("GDN workspace size exceeds usize")?;
        let mut workspace: CudaSlice<u8> = ctx
            .stream
            .alloc_zeros(workspace_len)
            .map_err(|error| anyhow::anyhow!("allocate GDN TMA workspace: {error}"))?;
        let cu_end = i64::try_from(tokens).context("GDN token count exceeds i64")?;
        let cu_seqlens = ctx
            .stream
            .clone_htod(&[0_i64, cu_end])
            .map_err(|error| anyhow::anyhow!("upload GDN cu_seqlens once for chunk: {error}"))?;

        let q_pointer = device_pointer_mut(&ctx.stream, &mut prepare.q.data);
        let k_pointer = device_pointer_mut(&ctx.stream, &mut prepare.k.data);
        let v_pointer = device_pointer_mut(&ctx.stream, &mut prepare.v.data);
        let output_pointer = device_pointer_mut(&ctx.stream, &mut output.data);
        let workspace_pointer = device_pointer_mut(&ctx.stream, &mut workspace);
        ensure!(
            workspace_pointer.is_multiple_of(backend.artifact.workspace_alignment),
            "GDN workspace pointer {workspace_pointer:#x} is not {}-byte aligned",
            backend.artifact.workspace_alignment
        );
        let tensor_maps = GdnTensorMaps {
            q: encode_tma_descriptor(q_pointer, tokens_u32, geometry.h_q, TmaSwizzle::B128)?,
            k: encode_tma_descriptor(k_pointer, tokens_u32, geometry.h_k, TmaSwizzle::B128)?,
            v: encode_tma_descriptor(v_pointer, tokens_u32, geometry.h_v, TmaSwizzle::B128)?,
            output: encode_tma_descriptor(
                output_pointer,
                tokens_u32,
                geometry.h_v,
                TmaSwizzle::B32,
            )?,
            q_pointer,
            k_pointer,
            v_pointer,
            output_pointer,
            tokens: tokens_u32,
            geometry,
        };

        Ok(Self {
            prepare,
            output,
            workspace,
            cu_seqlens,
            tensor_maps,
            workspace_bytes,
            tokens,
            geometry,
        })
    }

    /// Consume the sticky status written by every native-prepare launch in
    /// this chunk. This is intentionally one synchronization at the chunk
    /// boundary, not one per linear layer.
    pub(crate) fn ensure_prepare_inputs_finite(&self, ctx: &DeviceContext) -> Result<()> {
        let status = ctx
            .stream
            .clone_dtoh(&self.prepare.non_finite_status)
            .map_err(|error| anyhow::anyhow!("read native GDN finite-status failed: {error}"))?;
        ctx.sync()?;
        ensure!(
            status == [0],
            "native GDN prepare rejected non-finite qkv/gate input"
        );
        Ok(())
    }

    pub(crate) fn launch_in_place(
        &mut self,
        ctx: &DeviceContext,
        backend: &FlashInferGdnBackend,
        state: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let expected_state = state_elements(self.geometry)?;
        ensure!(
            state.len() == expected_state,
            "in-place GDN state length {}, expected {expected_state}",
            state.len()
        );
        let state_pointer = device_pointer_mut(&ctx.stream, state);
        self.launch_with_state_pointers(
            ctx,
            backend,
            state_pointer,
            state_pointer,
            FlashInferStateMode::InPlace,
        )
    }

    #[allow(dead_code)]
    pub(crate) fn launch_separate(
        &mut self,
        ctx: &DeviceContext,
        backend: &FlashInferGdnBackend,
        initial_state: &CudaSlice<f32>,
        final_state: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let expected_state = state_elements(self.geometry)?;
        ensure!(
            initial_state.len() == expected_state && final_state.len() == expected_state,
            "separate GDN state lengths initial={}, final={}, expected={expected_state}",
            initial_state.len(),
            final_state.len()
        );
        let initial_pointer = device_pointer(&ctx.stream, initial_state);
        let final_pointer = device_pointer_mut(&ctx.stream, final_state);
        self.launch_with_state_pointers(
            ctx,
            backend,
            initial_pointer,
            final_pointer,
            FlashInferStateMode::Separate,
        )
    }

    fn launch_with_state_pointers(
        &mut self,
        ctx: &DeviceContext,
        backend: &FlashInferGdnBackend,
        initial_state: u64,
        final_state: u64,
        state_mode: FlashInferStateMode,
    ) -> Result<()> {
        let args = self.args_for_state_pointers(ctx, initial_state, final_state);
        backend.launch(ctx, &args, &self.tensor_maps, state_mode)
    }

    fn args_for_state_pointers(
        &mut self,
        ctx: &DeviceContext,
        initial_state: u64,
        final_state: u64,
    ) -> FlashInferGdnPrefillArgs {
        FlashInferGdnPrefillArgs {
            q: device_pointer(&ctx.stream, &self.prepare.q.data),
            k: device_pointer(&ctx.stream, &self.prepare.k.data),
            v: device_pointer(&ctx.stream, &self.prepare.v.data),
            output: device_pointer_mut(&ctx.stream, &mut self.output.data),
            alpha: device_pointer(&ctx.stream, &self.prepare.alpha),
            beta: device_pointer(&ctx.stream, &self.prepare.beta),
            state: final_state,
            initial_state,
            workspace: device_pointer_mut(&ctx.stream, &mut self.workspace),
            workspace_bytes: self.workspace_bytes,
            cu_seqlens: device_pointer(&ctx.stream, &self.cu_seqlens),
            cu_seqlens_len: 2,
            tokens: u32::try_from(self.tokens).expect("validated GDN token count fits u32"),
            h_q: self.geometry.h_q,
            h_k: self.geometry.h_k,
            h_v: self.geometry.h_v,
            head_dim: self.geometry.head_dim,
            stream: ctx.stream.cu_stream(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum TmaSwizzle {
    B32,
    B128,
}

impl TmaSwizzle {
    const fn inner_box_elements(self) -> u32 {
        match self {
            Self::B32 => 16,
            Self::B128 => 64,
        }
    }

    const fn box_dimensions(self) -> [u32; 3] {
        // The PTX addresses every tensor map as `[D,T,H]`.  One TMA
        // operation spans part of D and one 64-token tile while remaining on
        // exactly one head.  Keeping the old `[D,H,T]` tile order here would
        // make independent head CTAs overlap the same TMA region.
        [self.inner_box_elements(), TMA_TILE_TOKENS, 1]
    }
}

fn tma_global_layout(tokens: u32, heads: u32) -> ([u64; 3], [u64; 2]) {
    let element_bytes = std::mem::size_of::<half::bf16>() as u64;
    (
        [128, u64::from(tokens), u64::from(heads)],
        [u64::from(heads) * 128 * element_bytes, 128 * element_bytes],
    )
}

fn encode_tma_descriptor(
    pointer: u64,
    tokens: u32,
    heads: u32,
    swizzle: TmaSwizzle,
) -> Result<TmaDescriptor> {
    ensure!(
        pointer != 0,
        "cannot encode a TMA descriptor for a null pointer"
    );
    ensure!(
        pointer.is_multiple_of(128),
        "swizzled TMA tensor pointer {pointer:#x} is not 128-byte aligned"
    );
    ensure!(
        tokens > 0 && heads > 0,
        "TMA tensor extents must be non-zero"
    );

    // The compiled CuTe TMA tensor emits coordinates as `[D,T,H]` (the PTX
    // operands are `{d, token, head}`). Preserve that logical axis order in
    // the descriptor while describing the token-major `[T,H,D]` allocation.
    // Sorting the axes by physical stride would silently turn head>0 into a
    // token coordinate and make those accesses OOB when T is small.
    let (global_dimensions, global_strides) = tma_global_layout(tokens, heads);
    // CuTe's K_SW128 atom covers D=128 with two 64-BF16 TMA operations;
    // MN_SW32 covers it with eight 16-BF16 operations.  `boxDim[0]` is the
    // inner dimension of one operation, not the full logical head dimension.
    // CUDA rejects an inner box wider than the selected swizzle span.
    let box_dimensions = swizzle.box_dimensions();
    let element_strides = [1_u32, 1, 1];
    let mut descriptor = TmaDescriptor { opaque: [0; 16] };
    let cuda_swizzle = match swizzle {
        TmaSwizzle::B32 => sys::CUtensorMapSwizzle_enum::CU_TENSOR_MAP_SWIZZLE_32B,
        TmaSwizzle::B128 => sys::CUtensorMapSwizzle_enum::CU_TENSOR_MAP_SWIZZLE_128B,
    };
    let result = unsafe {
        sys::cuTensorMapEncodeTiled(
            (&raw mut descriptor).cast::<sys::CUtensorMap>(),
            sys::CUtensorMapDataType_enum::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            3,
            pointer as usize as *mut std::ffi::c_void,
            global_dimensions.as_ptr(),
            global_strides.as_ptr(),
            box_dimensions.as_ptr(),
            element_strides.as_ptr(),
            sys::CUtensorMapInterleave_enum::CU_TENSOR_MAP_INTERLEAVE_NONE,
            cuda_swizzle,
            sys::CUtensorMapL2promotion_enum::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            sys::CUtensorMapFloatOOBfill_enum::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    result
        .result()
        .map_err(|error| anyhow::anyhow!("encode GDN TMA descriptor failed: {error}"))?;
    Ok(descriptor)
}

fn state_elements(geometry: Geometry) -> Result<usize> {
    usize::try_from(geometry.h_v)
        .context("Hv exceeds usize")?
        .checked_mul(usize::try_from(geometry.head_dim).context("D exceeds usize")?)
        .and_then(|elements| elements.checked_mul(usize::try_from(geometry.head_dim).ok()?))
        .context("GDN state length overflow")
}

fn validate_state_mode(
    initial_state: u64,
    final_state: u64,
    state_mode: FlashInferStateMode,
) -> Result<()> {
    match state_mode {
        FlashInferStateMode::Separate => ensure!(
            final_state != initial_state,
            "separate GDN state mode requires different initial/final pointers"
        ),
        FlashInferStateMode::InPlace => ensure!(
            final_state == initial_state,
            "in-place GDN state mode requires exact pointer alias"
        ),
    }
    Ok(())
}

fn device_pointer<T>(stream: &cudarc::driver::CudaStream, slice: &CudaSlice<T>) -> u64 {
    let (pointer, _guard) = slice.device_ptr(stream);
    pointer
}

fn device_pointer_mut<T>(stream: &cudarc::driver::CudaStream, slice: &mut CudaSlice<T>) -> u64 {
    let (pointer, _guard) = slice.device_ptr_mut(stream);
    pointer
}

impl Qwen35Model {
    /// Install exactly one backend owned by this model/context. There is no
    /// global cache and no fallback when validation or loading fails.
    pub(super) fn install_flashinfer_gdn(&mut self, manifest_path: &Path) -> Result<()> {
        ensure!(
            self.flashinfer_gdn.is_none(),
            "FlashInfer GDN backend is already installed for this model"
        );
        let backend = FlashInferGdnBackend::load(&self.ctx, manifest_path)?;
        install_once(&mut self.flashinfer_gdn, backend)
    }

    pub(super) fn flashinfer_gdn(&self) -> Result<&FlashInferGdnBackend> {
        self.flashinfer_gdn
            .as_ref()
            .context("FlashInfer GDN backend is not installed for this model")
    }
}

fn load_and_validate_artifact(manifest_path: &Path) -> Result<(ValidatedArtifact, String)> {
    let bytes = fs::read(manifest_path)
        .with_context(|| format!("read GDN manifest {}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse GDN manifest {}", manifest_path.display()))?;
    validate_manifest(&manifest)?;

    let parent = manifest_path
        .parent()
        .context("GDN manifest path has no parent directory")?;
    ensure!(
        manifest.artifact.file == "kernel.ptx",
        "artifact.file must be kernel.ptx"
    );
    let ptx_path = parent.join(&manifest.artifact.file);
    let ptx_bytes =
        fs::read(&ptx_path).with_context(|| format!("read GDN PTX {}", ptx_path.display()))?;
    ensure!(
        ptx_bytes.len() as u64 == manifest.artifact.size_bytes,
        "GDN PTX size mismatch: manifest {}, actual {}",
        manifest.artifact.size_bytes,
        ptx_bytes.len()
    );
    let actual_hash = hex_sha256(&ptx_bytes);
    ensure!(
        actual_hash == manifest.artifact.sha256 && actual_hash == ARTIFACT_SHA256,
        "GDN PTX SHA-256 mismatch: manifest {}, pinned {}, actual {actual_hash}",
        manifest.artifact.sha256,
        ARTIFACT_SHA256
    );
    let ptx = String::from_utf8(ptx_bytes).context("GDN artifact is not UTF-8 PTX")?;
    ensure!(
        ptx.contains(&format!(".entry {}(", manifest.abi.entry_symbol)),
        "GDN PTX does not define manifest entry symbol {}",
        manifest.abi.entry_symbol
    );
    validate_ptx_launch_abi(&ptx)?;
    let ptx = normalize_ptx_for_driver(ptx)?;

    Ok((
        ValidatedArtifact {
            manifest_path: manifest_path.to_owned(),
            ptx_path,
            geometry: manifest.geometry,
            variant: manifest.variant,
            entry_symbol: manifest.abi.entry_symbol,
            workspace_bytes_per_sm: manifest.workspace.bytes_per_sm,
            workspace_alignment: manifest.workspace.alignment_bytes,
        },
        ptx,
    ))
}

/// Normalize the verified PTX text before cudarc wraps it in a `CString`.
///
/// The frozen CUTLASS DSL artifact carries one C-string terminator followed by
/// a newline.  Those bytes remain part of the pinned file size and SHA-256,
/// but `Ptx::from_src` rejects the terminator as an interior NUL.  Permit that
/// exact trailing representation while continuing to fail closed for a NUL
/// followed by any non-whitespace PTX content or by another NUL.
fn normalize_ptx_for_driver(mut ptx: String) -> Result<String> {
    let Some(terminator) = ptx.find('\0') else {
        return Ok(ptx);
    };
    ensure!(
        ptx.as_bytes()[terminator + 1..]
            .iter()
            .all(u8::is_ascii_whitespace),
        "GDN PTX contains an interior NUL at byte {terminator}"
    );
    ptx.truncate(terminator);
    Ok(ptx)
}

fn validate_manifest(m: &Manifest) -> Result<()> {
    ensure!(
        m.schema_version == SCHEMA_VERSION,
        "unsupported GDN manifest schema {}",
        m.schema_version
    );
    ensure!(
        m.artifact_kind == ARTIFACT_KIND,
        "wrong GDN artifact_kind {}",
        m.artifact_kind
    );
    ensure!(
        m.target.arch == TARGET_ARCH,
        "wrong GDN target arch {}",
        m.target.arch
    );
    ensure!(
        m.target.driver_jit_target == DRIVER_JIT_TARGET,
        "wrong GDN JIT target {}",
        m.target.driver_jit_target
    );
    ensure!(
        m.source.flashinfer_commit == FLASHINFER_COMMIT,
        "unpinned FlashInfer commit {}",
        m.source.flashinfer_commit
    );
    ensure!(
        m.source.hkv_state_index_patch_applied,
        "required Hkv state-index patch is not applied"
    );
    ensure!(
        m.source.hkv_state_index_patch_sha256 == PATCH_SHA256,
        "wrong Hkv patch hash {}",
        m.source.hkv_state_index_patch_sha256
    );
    ensure!(
        m.source.kernel_source_sha256 == KERNEL_SOURCE_SHA256,
        "wrong patched kernel source hash {}",
        m.source.kernel_source_sha256
    );
    ensure!(
        m.source.patch_set_sha256 == PATCH_SET_SHA256,
        "wrong GDN patch-set hash {}",
        m.source.patch_set_sha256
    );
    ensure!(
        m.source.requirements_lock_sha256 == REQUIREMENTS_LOCK_SHA256,
        "wrong GDN requirements lock hash {}",
        m.source.requirements_lock_sha256
    );
    ensure!(
        m.source.generator_sha256 == GENERATOR_SHA256,
        "wrong GDN generator hash {}",
        m.source.generator_sha256
    );
    ensure!(
        m.artifact.format == "ptx",
        "unsupported GDN artifact format {}",
        m.artifact.format
    );
    ensure!(
        m.artifact.sha256 == ARTIFACT_SHA256,
        "unpinned GDN PTX hash {}",
        m.artifact.sha256
    );
    ensure!(
        m.artifact.size_bytes == ARTIFACT_SIZE_BYTES,
        "wrong GDN PTX size {}",
        m.artifact.size_bytes
    );
    ensure!(
        m.artifact.entry_symbols == [ENTRY_SYMBOL],
        "unexpected GDN entry_symbols"
    );
    ensure!(
        m.abi.entry_symbol == ENTRY_SYMBOL,
        "unexpected GDN ABI entry symbol {}",
        m.abi.entry_symbol
    );
    ensure!(
        m.artifact.absolute_path_scan == "passed",
        "artifact absolute-path scan did not pass"
    );
    ensure!(
        m.abi.geometry_binding == "manifest_guarded_runtime_head_parameters",
        "wrong geometry binding {}",
        m.abi.geometry_binding
    );
    ensure!(
        m.abi.state_layout == "openinfer_hkv_v_contiguous",
        "wrong state layout {}",
        m.abi.state_layout
    );

    let expected_variant = match m.geometry {
        Geometry {
            h_q: 16,
            h_k: 16,
            h_v: 32,
            head_dim: 128,
        } => "qwen35_4b_candidate",
        Geometry {
            h_q: 16,
            h_k: 16,
            h_v: 48,
            head_dim: 128,
        } => "operator_hv48",
        got => bail!("unsupported GDN head geometry {got:?}"),
    };
    ensure!(
        m.variant == expected_variant,
        "geometry {:?} requires variant {expected_variant}, got {}",
        m.geometry,
        m.variant
    );

    let expected_dtypes = BTreeMap::from([
        ("alpha".into(), "float32".into()),
        ("beta".into(), "float32".into()),
        ("cu_seqlens".into(), "int64".into()),
        ("k".into(), "bfloat16".into()),
        ("o".into(), "bfloat16".into()),
        ("q".into(), "bfloat16".into()),
        ("state".into(), "float32".into()),
        ("v".into(), "bfloat16".into()),
        ("workspace".into(), "uint8".into()),
    ]);
    ensure!(
        m.dtypes == expected_dtypes,
        "GDN dtype contract mismatch: {:?}",
        m.dtypes
    );
    ensure!(
        m.tokens.extent == json!("dynamic") && m.tokens.minimum == 1 && m.tokens.divisibility == 1,
        "unsupported token extent contract"
    );
    validate_views(m)?;
    ensure!(
        m.workspace.kind == "per_sm",
        "wrong workspace kind {}",
        m.workspace.kind
    );
    ensure!(
        m.workspace.formula == "sm_count * bytes_per_sm",
        "wrong workspace formula {}",
        m.workspace.formula
    );
    ensure!(
        m.workspace.bytes_per_sm == WORKSPACE_BYTES_PER_SM,
        "wrong workspace bytes/SM {}",
        m.workspace.bytes_per_sm
    );
    ensure!(
        m.workspace.alignment_bytes == WORKSPACE_ALIGNMENT,
        "wrong workspace alignment {}",
        m.workspace.alignment_bytes
    );
    ensure!(
        m.distribution.cuda_driver_jit_required,
        "PTX artifact must require CUDA driver JIT"
    );
    ensure!(
        !m.distribution.serving_requires_cute_dsl && !m.distribution.serving_requires_python,
        "serving artifact must not depend on Python/CuTe DSL"
    );
    ensure!(
        !m.distribution.production_eligible,
        "stage-4 loader only accepts the quarantined non-production artifact"
    );
    Ok(())
}

fn validate_views(manifest: &Manifest) -> Result<()> {
    let geometry = manifest.geometry;
    let q_view = json!({"shape": ["T", geometry.head_dim, geometry.h_q], "stride": [geometry.head_dim * geometry.h_q, 1, geometry.head_dim]});
    let k_view = json!({"shape": [geometry.head_dim, "T", geometry.h_k], "stride": [1, geometry.head_dim * geometry.h_k, geometry.head_dim]});
    let v_view = json!({"shape": [geometry.head_dim, "T", geometry.h_v], "stride": [1, geometry.head_dim * geometry.h_v, geometry.head_dim]});
    let output_view = json!({"shape": [geometry.head_dim, "T", geometry.h_v], "stride": [1, geometry.head_dim * geometry.h_v, geometry.head_dim]});
    ensure!(manifest.abi.q_view == q_view, "Q view mismatch");
    ensure!(manifest.abi.k_view == k_view, "K view mismatch");
    ensure!(manifest.abi.v_view == v_view, "V view mismatch");
    ensure!(manifest.abi.o_view == output_view, "O view mismatch");
    Ok(())
}

fn validate_ptx_launch_abi(ptx: &str) -> Result<()> {
    let expected = [
        (".align 8 .b8", "[16]"),
        (".align 8 .b8", "[16]"),
        (".align 64 .b8", "[128]"),
        (".align 4 .b8", "[4]"),
        (".align 64 .b8", "[128]"),
        (".align 4 .b8", "[4]"),
        (".align 64 .b8", "[128]"),
        (".align 4 .b8", "[4]"),
        (".align 64 .b8", "[128]"),
        (".align 4 .b8", "[4]"),
        (".align 8 .b8", "[8]"),
        (".align 8 .b8", "[8]"),
        (".align 8 .b8", "[16]"),
        (".align 8 .b8", "[16]"),
        (".f32", "param_14"),
        (".u32", "param_15"),
        (".u32", "param_16"),
        (".u32", "param_17"),
        (".u32", "param_18"),
        (".u32", "param_19"),
        (".u32", "param_20"),
        (".u32", "param_21"),
    ];
    let parameters: Vec<_> = ptx
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with(".param "))
        .collect();
    ensure!(
        parameters.len() == expected.len(),
        "GDN PTX parameter count mismatch: expected {}, got {}",
        expected.len(),
        parameters.len()
    );
    for (index, (line, (kind, extent_or_name))) in parameters.iter().zip(expected).enumerate() {
        ensure!(
            line.contains(kind) && line.contains(extent_or_name),
            "GDN PTX parameter {index} does not match frozen ABI: {line}"
        );
    }
    ensure!(
        ptx.contains(".maxntid 384, 1, 1"),
        "GDN PTX does not declare the frozen 384-thread block"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_launch_contract(
    args: &FlashInferGdnPrefillArgs,
    geometry: Geometry,
    bytes_per_sm: u64,
    workspace_alignment: u64,
    sm_count: u32,
    expected_context: usize,
    current_context: usize,
    expected_stream: usize,
) -> Result<ValidatedLaunch> {
    ensure!(
        current_context == expected_context,
        "CUDA current-context mismatch: backend {expected_context:#x}, current {current_context:#x}"
    );
    ensure!(
        args.stream as usize == expected_stream,
        "GDN launch stream does not belong to the model DeviceContext"
    );
    ensure!(args.tokens >= 1, "GDN token count must be >= 1");
    ensure!(
        (args.h_q, args.h_k, args.h_v, args.head_dim)
            == (geometry.h_q, geometry.h_k, geometry.h_v, geometry.head_dim),
        "GDN launch geometry {}/{}/{}/{} does not match artifact {}/{}/{}/{}",
        args.h_q,
        args.h_k,
        args.h_v,
        args.head_dim,
        geometry.h_q,
        geometry.h_k,
        geometry.h_v,
        geometry.head_dim
    );
    for (name, ptr, alignment) in [
        ("q", args.q, 16),
        ("k", args.k, 16),
        ("v", args.v, 16),
        ("output", args.output, 16),
        ("alpha", args.alpha, 16),
        ("beta", args.beta, 16),
        ("state", args.state, 16),
        ("initial_state", args.initial_state, 16),
        ("workspace", args.workspace, workspace_alignment),
        ("cu_seqlens", args.cu_seqlens, 8),
    ] {
        ensure!(ptr != 0, "GDN {name} device pointer is null");
        ensure!(
            ptr % alignment == 0,
            "GDN {name} device pointer {ptr:#x} is not {alignment}-byte aligned"
        );
    }
    ensure!(
        args.cu_seqlens_len == 2,
        "single-sequence GDN ABI requires cu_seqlens_len=2, got {}",
        args.cu_seqlens_len
    );
    let workspace_required = u64::from(sm_count)
        .checked_mul(bytes_per_sm)
        .context("GDN workspace size overflow")?;
    ensure!(
        args.workspace_bytes >= workspace_required,
        "GDN workspace too small: need {workspace_required}, got {}",
        args.workspace_bytes
    );
    let grid_x = geometry.h_v;
    Ok(ValidatedLaunch {
        scale: 1.0 / (geometry.head_dim as f32).sqrt(),
        grid_x,
        workspace_required,
    })
}

fn current_context_identity() -> Result<usize> {
    let mut current = std::ptr::null_mut();
    let status = unsafe { sys::cuCtxGetCurrent(&raw mut current) };
    ensure!(
        status == sys::CUresult::CUDA_SUCCESS,
        "cuCtxGetCurrent failed: {status:?}"
    );
    ensure!(
        !current.is_null(),
        "no CUDA context is current on the launch thread"
    );
    Ok(current as usize)
}

fn hex_sha256(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn install_once<T>(slot: &mut Option<T>, value: T) -> Result<()> {
    ensure!(
        slot.is_none(),
        "FlashInfer GDN backend is already installed for this model"
    );
    *slot = Some(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::mem::align_of;
    use std::mem::size_of;

    use half::bf16;
    use openinfer_core::tensor::DeviceVec;

    use super::*;
    use crate::config::LayerType;
    use crate::gdn_prepare_test_contract::Fixture;
    use crate::gdn_prepare_test_contract::Prepared;
    use crate::gdn_prepare_test_contract::bf16_to_f32;
    use crate::gdn_prepare_test_contract::deterministic_fixture;
    use crate::gdn_prepare_test_contract::prepare;
    use crate::gdn_stage7_test_support::CpuRunResult;
    use crate::gdn_stage7_test_support::DifferenceStats;
    use crate::gdn_stage7_test_support::PREPARE_GATE_TOLERANCE;
    use crate::gdn_stage7_test_support::PREPARE_QK_TOLERANCE;
    use crate::gdn_stage7_test_support::RECURRENCE_OUTPUT_TOLERANCE;
    use crate::gdn_stage7_test_support::RECURRENCE_STATE_TOLERANCE;
    use crate::gdn_stage7_test_support::asymmetric_hkv_state;
    use crate::gdn_stage7_test_support::cpu_decode_from_raw;
    use crate::gdn_stage7_test_support::cpu_stepwise;
    use crate::gdn_stage7_test_support::transpose_kv_as_wrong_hvk;
    use crate::prefill_buffers::GdrChunkwiseScratch35;

    fn candidate_config(h_v: usize) -> Config35 {
        Config35 {
            hidden_size: 2560,
            intermediate_size: 9216,
            num_hidden_layers: 32,
            vocab_size: 248_320,
            selection_vocab: 248_320,
            rms_norm_eps: 1e-6,
            eos_token_id: 151_645,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_num_value_heads: h_v,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            rope_theta: 10_000.0,
            rotary_dim: 64,
            max_position_embeddings: 262_144,
            tie_word_embeddings: true,
            layer_types: vec![LayerType::LinearAttention; 32],
        }
    }

    struct DeviceFixture {
        qkv: HiddenStates,
        b: HiddenStates,
        a: HiddenStates,
        dt_bias: DeviceVec,
        a_log: CudaSlice<f32>,
    }

    fn bf16_from_bits(values: &[u16]) -> Vec<bf16> {
        values.iter().copied().map(bf16::from_bits).collect()
    }

    fn f32_from_bits(values: &[u16]) -> Vec<f32> {
        values.iter().copied().map(bf16_to_f32).collect()
    }

    fn upload_fixture(ctx: &DeviceContext, fixture: &Fixture) -> Result<DeviceFixture> {
        Ok(DeviceFixture {
            qkv: HiddenStates::from_host(
                ctx,
                &bf16_from_bits(&fixture.qkv),
                fixture.offsets.total,
                fixture.geometry.tokens,
            )?,
            b: HiddenStates::from_host(
                ctx,
                &bf16_from_bits(&fixture.b),
                fixture.geometry.h_v,
                fixture.geometry.tokens,
            )?,
            a: HiddenStates::from_host(
                ctx,
                &bf16_from_bits(&fixture.a),
                fixture.geometry.h_v,
                fixture.geometry.tokens,
            )?,
            dt_bias: DeviceVec::from_host(ctx, &bf16_from_bits(&fixture.dt_bias))?,
            a_log: ctx.stream.clone_htod(&fixture.a_log)?,
        })
    }

    fn log_and_gate(
        label: &str,
        reference: &[f32],
        candidate: &[f32],
        tolerance: crate::gdn_stage7_test_support::NumericTolerance,
    ) -> Result<DifferenceStats> {
        let stats = log_difference_stats(label, reference, candidate, tolerance)?;
        stats.ensure_within(label).map_err(anyhow::Error::msg)?;
        Ok(stats)
    }

    fn log_difference_stats(
        label: &str,
        reference: &[f32],
        candidate: &[f32],
        tolerance: crate::gdn_stage7_test_support::NumericTolerance,
    ) -> Result<DifferenceStats> {
        let stats = DifferenceStats::compare(reference, candidate, tolerance)
            .map_err(anyhow::Error::msg)?;
        eprintln!("{label}: {stats:?}");
        Ok(stats)
    }

    fn validate_gpu_prepare(
        ctx: &DeviceContext,
        resources: &FlashInferGdnChunkResources,
        expected: &Prepared,
        tokens: usize,
        h_v: usize,
    ) -> Result<Prepared> {
        resources.ensure_prepare_inputs_finite(ctx)?;
        let q = ctx.stream.clone_dtoh(&resources.prepare.q.data)?;
        let k = ctx.stream.clone_dtoh(&resources.prepare.k.data)?;
        let v = ctx.stream.clone_dtoh(&resources.prepare.v.data)?;
        let alpha = ctx.stream.clone_dtoh(&resources.prepare.alpha)?;
        let beta = ctx.stream.clone_dtoh(&resources.prepare.beta)?;
        ctx.sync()?;

        let q_bits: Vec<u16> = q.iter().map(|value| value.to_bits()).collect();
        let k_bits: Vec<u16> = k.iter().map(|value| value.to_bits()).collect();
        let v_bits: Vec<u16> = v.iter().map(|value| value.to_bits()).collect();
        let q_f32: Vec<f32> = q.iter().map(|value| value.to_f32()).collect();
        let k_f32: Vec<f32> = k.iter().map(|value| value.to_f32()).collect();
        log_and_gate(
            &format!("prepare.q Hv={h_v} T={tokens}"),
            &f32_from_bits(&expected.q),
            &q_f32,
            PREPARE_QK_TOLERANCE,
        )?;
        log_and_gate(
            &format!("prepare.k Hv={h_v} T={tokens}"),
            &f32_from_bits(&expected.k),
            &k_f32,
            PREPARE_QK_TOLERANCE,
        )?;
        ensure!(
            v_bits == expected.v,
            "prepare.v must preserve BF16 bits exactly at Hv={h_v}, T={tokens}"
        );
        log_and_gate(
            &format!("prepare.alpha Hv={h_v} T={tokens}"),
            &expected.alpha,
            &alpha,
            PREPARE_GATE_TOLERANCE,
        )?;
        log_and_gate(
            &format!("prepare.beta Hv={h_v} T={tokens}"),
            &expected.beta,
            &beta,
            PREPARE_GATE_TOLERANCE,
        )?;
        Ok(Prepared {
            q: q_bits,
            k: k_bits,
            v: v_bits,
            alpha,
            beta,
        })
    }

    fn run_batched_decode_handoff(
        ctx: &DeviceContext,
        h_v: usize,
        cpu_prefill: &CpuRunResult,
        triton_state: &mut CudaSlice<f32>,
        flashinfer_state: &mut CudaSlice<f32>,
        tokens: usize,
        gate_triton_baseline: bool,
    ) -> Result<()> {
        let decode_fixture = deterministic_fixture(1, h_v);
        let cpu_decode = cpu_decode_from_raw(&decode_fixture, &cpu_prefill.final_state)
            .map_err(anyhow::Error::msg)?;

        let repeat_twice = |values: &[u16]| {
            values
                .iter()
                .chain(values.iter())
                .copied()
                .collect::<Vec<_>>()
        };
        let qkv = HiddenStates::from_host(
            ctx,
            &bf16_from_bits(&repeat_twice(&decode_fixture.qkv)),
            decode_fixture.offsets.total,
            2,
        )?;
        let b = HiddenStates::from_host(
            ctx,
            &bf16_from_bits(&repeat_twice(&decode_fixture.b)),
            h_v,
            2,
        )?;
        let a = HiddenStates::from_host(
            ctx,
            &bf16_from_bits(&repeat_twice(&decode_fixture.a)),
            h_v,
            2,
        )?;
        let dt_bias = DeviceVec::from_host(ctx, &bf16_from_bits(&decode_fixture.dt_bias))?;
        let a_log = ctx.stream.clone_htod(&decode_fixture.a_log)?;

        let state_ptrs = {
            let (triton_pointer, _triton_guard) = triton_state.device_ptr_mut(&ctx.stream);
            let (flashinfer_pointer, _flashinfer_guard) =
                flashinfer_state.device_ptr_mut(&ctx.stream);
            ctx.stream
                .clone_htod(&[triton_pointer, flashinfer_pointer])?
        };
        let mut output = HiddenStates::zeros(ctx, h_v * decode_fixture.geometry.d, 2)?;
        crate::ops::gated_delta_rule_decode_batch_into(
            ctx,
            &qkv,
            &b,
            &a,
            &dt_bias,
            &a_log,
            &state_ptrs,
            &mut output,
            2,
            decode_fixture.geometry.h_k,
            h_v,
            decode_fixture.geometry.d,
            decode_fixture.geometry.d,
        );

        let output = output.to_host(ctx)?;
        let triton_after_decode = ctx.stream.clone_dtoh(triton_state)?;
        let flashinfer_after_decode = ctx.stream.clone_dtoh(flashinfer_state)?;
        ctx.sync()?;
        let row = h_v * decode_fixture.geometry.d;
        let triton_output = &output[..row];
        let flashinfer_output = &output[row..];

        let cpu_triton_output_stats = log_difference_stats(
            &format!("first-decode CPU/Triton output Hv={h_v} after T={tokens}"),
            &cpu_decode.output,
            triton_output,
            RECURRENCE_OUTPUT_TOLERANCE,
        )?;
        let cpu_flashinfer_output_stats = log_difference_stats(
            &format!("first-decode CPU/FlashInfer output Hv={h_v} after T={tokens}"),
            &cpu_decode.output,
            flashinfer_output,
            RECURRENCE_OUTPUT_TOLERANCE,
        )?;
        let triton_flashinfer_output_stats = log_difference_stats(
            &format!("first-decode Triton/FlashInfer output Hv={h_v} after T={tokens}"),
            triton_output,
            flashinfer_output,
            RECURRENCE_OUTPUT_TOLERANCE,
        )?;
        let cpu_triton_state_stats = log_difference_stats(
            &format!("first-decode CPU/Triton state Hv={h_v} after T={tokens}"),
            &cpu_decode.final_state,
            &triton_after_decode,
            RECURRENCE_STATE_TOLERANCE,
        )?;
        let cpu_flashinfer_state_stats = log_difference_stats(
            &format!("first-decode CPU/FlashInfer state Hv={h_v} after T={tokens}"),
            &cpu_decode.final_state,
            &flashinfer_after_decode,
            RECURRENCE_STATE_TOLERANCE,
        )?;
        let triton_flashinfer_state_stats = log_difference_stats(
            &format!("first-decode Triton/FlashInfer state Hv={h_v} after T={tokens}"),
            &triton_after_decode,
            &flashinfer_after_decode,
            RECURRENCE_STATE_TOLERANCE,
        )?;

        for (label, stats) in [
            (
                format!("first-decode CPU/FlashInfer output Hv={h_v} after T={tokens}"),
                cpu_flashinfer_output_stats,
            ),
            (
                format!("first-decode CPU/FlashInfer state Hv={h_v} after T={tokens}"),
                cpu_flashinfer_state_stats,
            ),
        ] {
            stats.ensure_within(&label).map_err(anyhow::Error::msg)?;
        }
        if gate_triton_baseline {
            for (label, stats) in [
                (
                    format!("first-decode CPU/Triton output Hv={h_v} after T={tokens}"),
                    cpu_triton_output_stats,
                ),
                (
                    format!("first-decode Triton/FlashInfer output Hv={h_v} after T={tokens}"),
                    triton_flashinfer_output_stats,
                ),
                (
                    format!("first-decode CPU/Triton state Hv={h_v} after T={tokens}"),
                    cpu_triton_state_stats,
                ),
                (
                    format!("first-decode Triton/FlashInfer state Hv={h_v} after T={tokens}"),
                    triton_flashinfer_state_stats,
                ),
            ] {
                stats.ensure_within(&label).map_err(anyhow::Error::msg)?;
            }
        }
        Ok(())
    }

    fn validate_real_device_fail_closed(
        ctx: &DeviceContext,
        backend: &FlashInferGdnBackend,
        resources: &mut FlashInferGdnChunkResources,
        initial_state: &CudaSlice<f32>,
        final_state: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let initial_pointer = device_pointer(&ctx.stream, initial_state);
        let final_pointer = device_pointer_mut(&ctx.stream, final_state);
        let valid = resources.args_for_state_pointers(ctx, initial_pointer, final_pointer);
        backend.validate_launch(ctx, &valid)?;

        let mut short_workspace = valid;
        short_workspace.workspace_bytes = 1;
        ensure!(
            backend.validate_launch(ctx, &short_workspace).is_err(),
            "real-device launch contract accepted an undersized workspace"
        );
        ensure!(
            validate_state_mode(
                initial_pointer,
                initial_pointer,
                FlashInferStateMode::Separate
            )
            .is_err(),
            "real-device launch contract accepted an aliased separate state"
        );

        let clear_status = unsafe { sys::cuCtxSetCurrent(std::ptr::null_mut()) };
        ensure!(
            clear_status == sys::CUresult::CUDA_SUCCESS,
            "could not clear current CUDA context for negative gate: {clear_status:?}"
        );
        let wrong_context = backend.validate_launch(ctx, &valid);
        // Always restore the model context before inspecting the negative
        // result so a failed assertion cannot poison subsequent GPU gates.
        ctx.ctx.bind_to_thread()?;
        ensure!(
            wrong_context.is_err(),
            "real-device launch contract accepted a missing current context"
        );
        Ok(())
    }

    fn manifest_value(h_v: u32) -> Value {
        let variant = if h_v == 32 {
            "qwen35_4b_candidate"
        } else {
            "operator_hv48"
        };
        json!({
            "schema_version": 1,
            "artifact_kind": ARTIFACT_KIND,
            "variant": variant,
            "target": {"arch": TARGET_ARCH, "driver_jit_target": DRIVER_JIT_TARGET},
            "dtypes": {"alpha":"float32","beta":"float32","cu_seqlens":"int64","k":"bfloat16","o":"bfloat16","q":"bfloat16","state":"float32","v":"bfloat16","workspace":"uint8"},
            "geometry": {"h_q":16,"h_k":16,"h_v":h_v,"head_dim":128},
            "tokens": {"extent":"dynamic","minimum":1,"divisibility":1},
            "abi": {
                "entry_symbol": ENTRY_SYMBOL,
                "geometry_binding":"manifest_guarded_runtime_head_parameters",
                "q_view":{"shape":["T",128,16],"stride":[2048,1,128]},
                "k_view":{"shape":[128,"T",16],"stride":[1,2048,128]},
                "v_view":{"shape":[128,"T",h_v],"stride":[1,128*h_v,128]},
                "o_view":{"shape":[128,"T",h_v],"stride":[1,128*h_v,128]},
                "state_layout":"openinfer_hkv_v_contiguous"
            },
            "artifact":{"file":"kernel.ptx","format":"ptx","sha256":ARTIFACT_SHA256,"size_bytes":549690,"entry_symbols":[ENTRY_SYMBOL],"absolute_path_scan":"passed"},
            "source":{"flashinfer_commit":FLASHINFER_COMMIT,"hkv_state_index_patch_applied":true,"hkv_state_index_patch_sha256":PATCH_SHA256,"kernel_source_sha256":KERNEL_SOURCE_SHA256,"patch_set_sha256":PATCH_SET_SHA256,"requirements_lock_sha256":REQUIREMENTS_LOCK_SHA256,"generator_sha256":GENERATOR_SHA256},
            "workspace":{"kind":"per_sm","formula":"sm_count * bytes_per_sm","bytes_per_sm":128,"alignment_bytes":128},
            "distribution":{"cuda_driver_jit_required":true,"serving_requires_cute_dsl":false,"serving_requires_python":false,"production_eligible":false}
        })
    }

    fn parse(value: Value) -> Manifest {
        serde_json::from_value(value).unwrap()
    }

    fn valid_args(h_v: u32) -> FlashInferGdnPrefillArgs {
        FlashInferGdnPrefillArgs {
            q: 0x1000,
            k: 0x2000,
            v: 0x3000,
            output: 0x4000,
            alpha: 0x5000,
            beta: 0x6000,
            state: 0x7000,
            initial_state: 0x8000,
            workspace: 0x9000,
            workspace_bytes: 16_384,
            cu_seqlens: 0xa000,
            cu_seqlens_len: 2,
            tokens: 17,
            h_q: 16,
            h_k: 16,
            h_v,
            head_dim: 128,
            stream: 0xb000usize as sys::CUstream,
        }
    }

    #[test]
    fn c_abi_layout_is_stable() {
        assert_eq!(size_of::<FlashInferGdnPrefillArgs>(), 120);
        assert_eq!(align_of::<FlashInferGdnPrefillArgs>(), 8);
        assert_eq!(size_of::<CompactTensorArg>(), 16);
        assert_eq!(align_of::<CompactTensorArg>(), 8);
        assert_eq!(size_of::<TmaDescriptor>(), 128);
        assert_eq!(align_of::<TmaDescriptor>(), 64);
    }

    #[test]
    fn tma_inner_box_matches_frozen_smem_swizzle() {
        let bf16_bytes = size_of::<bf16>() as u32;
        assert_eq!(TmaSwizzle::B128.inner_box_elements() * bf16_bytes, 128);
        assert_eq!(TmaSwizzle::B32.inner_box_elements() * bf16_bytes, 32);
        assert_eq!(128 % TmaSwizzle::B128.inner_box_elements(), 0);
        assert_eq!(128 % TmaSwizzle::B32.inner_box_elements(), 0);
        assert_eq!(TmaSwizzle::B128.box_dimensions(), [64, 64, 1]);
        assert_eq!(TmaSwizzle::B32.box_dimensions(), [16, 64, 1]);
    }

    #[test]
    fn tma_global_layout_preserves_compiled_d_t_h_coordinates() {
        let (dimensions, strides) = tma_global_layout(1, 32);
        assert_eq!(dimensions, [128, 1, 32]);
        assert_eq!(strides, [32 * 128 * 2, 128 * 2]);

        let (dimensions, strides) = tma_global_layout(65, 48);
        assert_eq!(dimensions, [128, 65, 48]);
        assert_eq!(strides, [48 * 128 * 2, 128 * 2]);
    }

    #[test]
    fn state_modes_require_separate_or_exact_alias_pointers() {
        validate_state_mode(0x1000, 0x2000, FlashInferStateMode::Separate).unwrap();
        validate_state_mode(0x1000, 0x1000, FlashInferStateMode::InPlace).unwrap();
        assert!(validate_state_mode(0x1000, 0x1000, FlashInferStateMode::Separate).is_err());
        assert!(validate_state_mode(0x1000, 0x2000, FlashInferStateMode::InPlace).is_err());
    }

    #[test]
    fn accepts_hv32_and_hv48_variants() {
        validate_manifest(&parse(manifest_value(32))).unwrap();
        validate_manifest(&parse(manifest_value(48))).unwrap();
    }

    #[test]
    fn validates_real_stage3_artifact_when_requested() {
        let Some(path) = std::env::var_os("OPENINFER_GDN_STAGE3_MANIFEST") else {
            return;
        };
        let (artifact, ptx) = load_and_validate_artifact(Path::new(&path)).unwrap();
        assert_eq!(artifact.geometry.h_v, 32);
        assert!(!ptx.contains('\0'));
        assert!(ptx.ends_with("}\n"));
    }

    #[test]
    fn strips_verified_ptx_trailing_c_string_terminator() {
        assert_eq!(
            normalize_ptx_for_driver(".version 8.8\n.entry kernel() {\n}\n\0\n".to_owned())
                .unwrap(),
            ".version 8.8\n.entry kernel() {\n}\n"
        );
        assert_eq!(
            normalize_ptx_for_driver(".version 8.8\n".to_owned()).unwrap(),
            ".version 8.8\n"
        );
    }

    #[test]
    fn rejects_ptx_interior_or_repeated_nul() {
        assert!(
            normalize_ptx_for_driver(".version 8.8\n\0.entry kernel() {}\n".to_owned()).is_err()
        );
        assert!(normalize_ptx_for_driver(".version 8.8\n\0\0\n".to_owned()).is_err());
    }

    #[test]
    fn rejects_manifest_sm_hash_dtype_geometry_workspace_and_symbol() {
        let mutations: &[(&[&str], Value)] = &[
            (&["target", "arch"], json!("sm_90a")),
            (&["artifact", "sha256"], json!("00")),
            (&["dtypes", "q"], json!("float16")),
            (&["geometry", "h_k"], json!(32)),
            (&["workspace", "bytes_per_sm"], json!(64)),
            (&["abi", "entry_symbol"], json!("wrong")),
        ];
        for (path, replacement) in mutations {
            let mut value = manifest_value(32);
            let mut cursor = &mut value;
            for key in &path[..path.len() - 1] {
                cursor = &mut cursor[*key];
            }
            cursor[path[path.len() - 1]] = replacement.clone();
            assert!(
                validate_manifest(&parse(value)).is_err(),
                "mutation {path:?} was accepted"
            );
        }
    }

    #[test]
    fn validates_arguments_and_derives_scale_after_geometry() {
        let args = valid_args(32);
        let launch = validate_launch_contract(
            &args,
            Geometry {
                h_q: 16,
                h_k: 16,
                h_v: 32,
                head_dim: 128,
            },
            128,
            128,
            80,
            0xc000,
            0xc000,
            0xb000,
        )
        .unwrap();
        assert_eq!(launch.grid_x, 32);
        assert_eq!(launch.workspace_required, 10_240);
        assert!((launch.scale - 1.0 / 128.0_f32.sqrt()).abs() < f32::EPSILON);
    }

    #[test]
    fn rejects_bad_args_workspace_stream_and_context() {
        let geometry = Geometry {
            h_q: 16,
            h_k: 16,
            h_v: 32,
            head_dim: 128,
        };
        let mut cases = Vec::new();
        let mut a = valid_args(32);
        a.q = 0;
        cases.push((a, 0xc000, 0xb000));
        let mut a = valid_args(32);
        a.workspace_bytes = 1;
        cases.push((a, 0xc000, 0xb000));
        let mut a = valid_args(32);
        a.h_v = 48;
        cases.push((a, 0xc000, 0xb000));
        let mut a = valid_args(32);
        a.stream = 0xd000usize as sys::CUstream;
        cases.push((a, 0xc000, 0xb000));
        cases.push((valid_args(32), 0xd000, 0xb000));
        for (args, current, stream) in cases {
            assert!(
                validate_launch_contract(&args, geometry, 128, 128, 80, 0xc000, current, stream)
                    .is_err()
            );
        }
    }

    #[test]
    fn model_local_slots_reject_repeat_and_do_not_share() {
        let mut first = None;
        let mut second = None;
        install_once(&mut first, 1_u8).unwrap();
        assert!(install_once(&mut first, 2).is_err());
        install_once(&mut second, 3_u8).unwrap();
        assert_eq!(first, Some(1));
        assert_eq!(second, Some(3));
    }

    /// Complete Stage 7 operator/state gate for one manifest geometry.
    ///
    /// The runner script invokes this once with Hv32 and once with Hv48.  Each
    /// invocation compares native prepare against the CPU oracle, compares
    /// CPU/Triton/FlashInfer prefill output and final state, proves exact alias
    /// equivalence, then hands both GPU states to the real batched-decode
    /// kernel for one more token.
    #[test]
    #[ignore = "requires an SM120 GPU and OPENINFER_GDN_STAGE3_MANIFEST"]
    fn sm120_launch_smoke_covers_alias_separate_and_dynamic_t() -> Result<()> {
        let manifest = std::env::var_os("OPENINFER_GDN_STAGE3_MANIFEST")
            .context("set OPENINFER_GDN_STAGE3_MANIFEST to the Stage 3 manifest")?;
        let ctx = DeviceContext::new()?;
        let backend = FlashInferGdnBackend::load(&ctx, Path::new(&manifest))?;
        let config = candidate_config(usize::try_from(backend.geometry().h_v)?);
        let state_len = state_elements(backend.geometry())?;

        for tokens in [1_usize, 2, 63, 64, 65, 127, 128] {
            let mut resources = FlashInferGdnChunkResources::new(&ctx, &config, &backend, tokens)?;
            let h_q = usize::try_from(backend.geometry().h_q)?;
            let h_k = usize::try_from(backend.geometry().h_k)?;
            let h_v = usize::try_from(backend.geometry().h_v)?;
            let head_dim = usize::try_from(backend.geometry().head_dim)?;
            let fixture = deterministic_fixture(tokens, h_v);
            ensure!(
                fixture.geometry.h_q == h_q
                    && fixture.geometry.h_k == h_k
                    && fixture.geometry.d == head_dim,
                "Stage 7 fixture geometry does not match manifest"
            );
            let expected_prepare = prepare(&fixture).map_err(anyhow::Error::msg)?;
            let device = upload_fixture(&ctx, &fixture)?;
            crate::ops::gated_delta_rule_prefill_native_prepare_into(
                &ctx,
                &device.qkv,
                &device.b,
                &device.a,
                &device.dt_bias,
                &device.a_log,
                &mut resources.prepare,
                h_q,
                h_k,
                h_v,
                head_dim,
            )?;
            // Replay the verified native prepare outputs in the CPU recurrence so
            // the oracle and FlashInfer consume bit-identical prepared inputs.
            let actual_prepare =
                validate_gpu_prepare(&ctx, &resources, &expected_prepare, tokens, h_v)?;

            let initial_host = asymmetric_hkv_state(fixture.geometry);
            ensure!(
                initial_host.len() == state_len,
                "Stage 7 state length mismatch"
            );
            let cpu = cpu_stepwise(fixture.geometry, &actual_prepare, &initial_host)
                .map_err(anyhow::Error::msg)?;
            if tokens == 1 {
                let wrong_hvk = transpose_kv_as_wrong_hvk(fixture.geometry, &initial_host);
                let wrong_cpu = cpu_stepwise(fixture.geometry, &actual_prepare, &wrong_hvk)
                    .map_err(anyhow::Error::msg)?;
                let wrong_output = DifferenceStats::compare(
                    &cpu.output,
                    &wrong_cpu.output,
                    RECURRENCE_OUTPUT_TOLERANCE,
                )
                .map_err(anyhow::Error::msg)?;
                let wrong_state = DifferenceStats::compare(
                    &cpu.final_state,
                    &wrong_cpu.final_state,
                    RECURRENCE_STATE_TOLERANCE,
                )
                .map_err(anyhow::Error::msg)?;
                ensure!(
                    wrong_output.violations > 0 || wrong_state.violations > 0,
                    "wrong-HVK negative oracle was not detected at Hv={h_v}, T={tokens}"
                );
            }

            let mut triton_state = ctx.stream.clone_htod(&initial_host)?;
            let mut triton_scratch =
                GdrChunkwiseScratch35::from_dims(&ctx, h_v, head_dim, head_dim, tokens)?;
            let mut triton_output = HiddenStates::zeros(&ctx, h_v * head_dim, tokens)?;
            crate::ops::gated_delta_rule_prefill_chunkwise_into(
                &ctx,
                &device.qkv,
                &device.b,
                &device.a,
                &device.dt_bias,
                &device.a_log,
                &mut triton_state,
                &mut triton_scratch,
                &mut triton_output,
                h_k,
                h_v,
                head_dim,
                head_dim,
            )?;
            let triton_output_host = triton_output.to_host(&ctx)?;
            let triton_final = ctx.stream.clone_dtoh(&triton_state)?;
            ctx.sync()?;

            let mut alias_state = ctx.stream.clone_htod(&initial_host)?;
            resources.launch_in_place(&ctx, &backend, &mut alias_state)?;
            let alias_output = resources.output.to_host(&ctx)?;
            let alias_final = ctx.stream.clone_dtoh(&alias_state)?;
            ctx.sync()?;

            let initial_state = ctx.stream.clone_htod(&initial_host)?;
            let mut final_state: CudaSlice<f32> = ctx.stream.alloc_zeros(state_len)?;
            if tokens == 1 {
                validate_real_device_fail_closed(
                    &ctx,
                    &backend,
                    &mut resources,
                    &initial_state,
                    &mut final_state,
                )?;
            }
            resources.launch_separate(&ctx, &backend, &initial_state, &mut final_state)?;
            let separate_output = resources.output.to_host(&ctx)?;
            let separate_final = ctx.stream.clone_dtoh(&final_state)?;
            ctx.sync()?;

            ensure!(
                alias_output.iter().all(|value| value.is_finite())
                    && separate_output.iter().all(|value| value.is_finite())
                    && alias_final.iter().all(|value| value.is_finite())
                    && separate_final.iter().all(|value| value.is_finite()),
                "non-finite GDN smoke output at T={tokens}"
            );
            let alias_separate_output_stats = log_difference_stats(
                &format!("prefill alias/separate output Hv={h_v} T={tokens}"),
                &separate_output,
                &alias_output,
                RECURRENCE_OUTPUT_TOLERANCE,
            )?;
            let alias_separate_state_stats = log_difference_stats(
                &format!("prefill alias/separate state Hv={h_v} T={tokens}"),
                &separate_final,
                &alias_final,
                RECURRENCE_STATE_TOLERANCE,
            )?;
            if alias_output != separate_output || alias_final != separate_final {
                // The exact-alias gate is deliberately bitwise.  When it
                // fails, print both paths against the independent CPU and
                // Triton oracles before returning so a paid GPU rerun tells
                // us which state mode is wrong instead of only reporting that
                // the two modes differ.
                log_difference_stats(
                    &format!("diagnostic CPU/alias output Hv={h_v} T={tokens}"),
                    &cpu.output,
                    &alias_output,
                    RECURRENCE_OUTPUT_TOLERANCE,
                )?;
                log_difference_stats(
                    &format!("diagnostic CPU/separate output Hv={h_v} T={tokens}"),
                    &cpu.output,
                    &separate_output,
                    RECURRENCE_OUTPUT_TOLERANCE,
                )?;
                log_difference_stats(
                    &format!("diagnostic Triton/alias output Hv={h_v} T={tokens}"),
                    &triton_output_host,
                    &alias_output,
                    RECURRENCE_OUTPUT_TOLERANCE,
                )?;
                log_difference_stats(
                    &format!("diagnostic Triton/separate output Hv={h_v} T={tokens}"),
                    &triton_output_host,
                    &separate_output,
                    RECURRENCE_OUTPUT_TOLERANCE,
                )?;
                log_difference_stats(
                    &format!("diagnostic CPU/alias state Hv={h_v} T={tokens}"),
                    &cpu.final_state,
                    &alias_final,
                    RECURRENCE_STATE_TOLERANCE,
                )?;
                log_difference_stats(
                    &format!("diagnostic CPU/separate state Hv={h_v} T={tokens}"),
                    &cpu.final_state,
                    &separate_final,
                    RECURRENCE_STATE_TOLERANCE,
                )?;
                log_difference_stats(
                    &format!("diagnostic Triton/alias state Hv={h_v} T={tokens}"),
                    &triton_final,
                    &alias_final,
                    RECURRENCE_STATE_TOLERANCE,
                )?;
                log_difference_stats(
                    &format!("diagnostic Triton/separate state Hv={h_v} T={tokens}"),
                    &triton_final,
                    &separate_final,
                    RECURRENCE_STATE_TOLERANCE,
                )?;
            }
            ensure!(
                alias_output == separate_output,
                "alias/separate GDN outputs differ at Hv={h_v}, T={tokens}: {alias_separate_output_stats:?}"
            );
            ensure!(
                alias_final == separate_final,
                "alias/separate GDN final states differ at Hv={h_v}, T={tokens}: {alias_separate_state_stats:?}"
            );
            ensure!(
                alias_output.iter().any(|&value| value != 0.0),
                "GDN smoke output remained zero at T={tokens}"
            );
            ensure!(
                alias_final != initial_host,
                "GDN smoke state did not update at T={tokens}"
            );

            let cpu_triton_output_stats = log_difference_stats(
                &format!("prefill CPU/Triton output Hv={h_v} T={tokens}"),
                &cpu.output,
                &triton_output_host,
                RECURRENCE_OUTPUT_TOLERANCE,
            )?;
            let cpu_flashinfer_output_stats = log_difference_stats(
                &format!("prefill CPU/FlashInfer output Hv={h_v} T={tokens}"),
                &cpu.output,
                &alias_output,
                RECURRENCE_OUTPUT_TOLERANCE,
            )?;
            let triton_flashinfer_output_stats = log_difference_stats(
                &format!("prefill Triton/FlashInfer output Hv={h_v} T={tokens}"),
                &triton_output_host,
                &alias_output,
                RECURRENCE_OUTPUT_TOLERANCE,
            )?;
            let cpu_triton_state_stats = log_difference_stats(
                &format!("prefill CPU/Triton state Hv={h_v} T={tokens}"),
                &cpu.final_state,
                &triton_final,
                RECURRENCE_STATE_TOLERANCE,
            )?;
            let cpu_flashinfer_state_stats = log_difference_stats(
                &format!("prefill CPU/FlashInfer state Hv={h_v} T={tokens}"),
                &cpu.final_state,
                &alias_final,
                RECURRENCE_STATE_TOLERANCE,
            )?;
            let triton_flashinfer_state_stats = log_difference_stats(
                &format!("prefill Triton/FlashInfer state Hv={h_v} T={tokens}"),
                &triton_final,
                &alias_final,
                RECURRENCE_STATE_TOLERANCE,
            )?;

            for (label, stats) in [
                (
                    format!("prefill CPU/FlashInfer output Hv={h_v} T={tokens}"),
                    cpu_flashinfer_output_stats,
                ),
                (
                    format!("prefill CPU/FlashInfer state Hv={h_v} T={tokens}"),
                    cpu_flashinfer_state_stats,
                ),
            ] {
                stats.ensure_within(&label).map_err(anyhow::Error::msg)?;
            }

            // Hv32 is the Qwen3.5-4B candidate and must pass the complete
            // CPU/Triton/FlashInfer triangle. Hv48 is an operator-only future
            // geometry: its independent CPU/FlashInfer gates remain strict,
            // while the existing Triton chunk approximation is diagnostic.
            // The Hv48 baseline can accumulate a few state elements outside
            // the frozen bound even when FlashInfer remains within it.
            let gate_triton_baseline = h_v == 32;
            if gate_triton_baseline {
                for (label, stats) in [
                    (
                        format!("prefill CPU/Triton output Hv={h_v} T={tokens}"),
                        cpu_triton_output_stats,
                    ),
                    (
                        format!("prefill Triton/FlashInfer output Hv={h_v} T={tokens}"),
                        triton_flashinfer_output_stats,
                    ),
                    (
                        format!("prefill CPU/Triton state Hv={h_v} T={tokens}"),
                        cpu_triton_state_stats,
                    ),
                    (
                        format!("prefill Triton/FlashInfer state Hv={h_v} T={tokens}"),
                        triton_flashinfer_state_stats,
                    ),
                ] {
                    stats.ensure_within(&label).map_err(anyhow::Error::msg)?;
                }
            }

            run_batched_decode_handoff(
                &ctx,
                h_v,
                &cpu,
                &mut triton_state,
                &mut alias_state,
                tokens,
                gate_triton_baseline,
            )?;
        }
        Ok(())
    }
}
