//! Test-only contract and deterministic fixtures for GDN prefill backends.
//!
//! This module deliberately stays behind `cfg(test)`: it lets Triton, a future
//! FlashInfer backend, and a stepwise reference consume identical inputs without
//! adding a backend selector to the production API.

use std::fmt;

pub(crate) const BOUNDARY_TOKEN_LENGTHS: [usize; 7] = [1, 2, 63, 64, 65, 127, 128];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BackendIdentity {
    CpuStepwise,
    TritonChunkwise,
    FlashInferCute,
}

impl fmt::Display for BackendIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::CpuStepwise => "cpu-stepwise",
            Self::TritonChunkwise => "triton-chunkwise",
            Self::FlashInferCute => "flashinfer-cute",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ElementDType {
    Bf16,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BufferOwner {
    TestCaller,
    RequestRecurrentState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BufferLifetime {
    PrefillInvocation,
    Request,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TensorViewContract {
    pub(crate) logical_shape: [usize; 3],
    pub(crate) element_strides: [usize; 3],
    pub(crate) dtype: ElementDType,
    pub(crate) owner: BufferOwner,
    pub(crate) lifetime: BufferLifetime,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ObservedTensorView {
    pub(crate) logical_shape: [usize; 3],
    pub(crate) element_strides: [usize; 3],
    pub(crate) dtype: ElementDType,
}

impl TensorViewContract {
    pub(crate) fn validate_observed(
        &self,
        name: &str,
        observed: ObservedTensorView,
    ) -> Result<(), String> {
        if observed.logical_shape != self.logical_shape {
            return Err(format!(
                "{name} shape mismatch: observed {:?}, expected {:?}",
                observed.logical_shape, self.logical_shape
            ));
        }
        if observed.element_strides != self.element_strides {
            return Err(format!(
                "{name} stride mismatch: observed {:?}, expected {:?}",
                observed.element_strides, self.element_strides
            ));
        }
        if observed.dtype != self.dtype {
            return Err(format!(
                "{name} dtype mismatch: observed {:?}, expected {:?}",
                observed.dtype, self.dtype
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GdnGeometry {
    pub(crate) hq: usize,
    pub(crate) hk: usize,
    pub(crate) hv: usize,
    pub(crate) d: usize,
    pub(crate) tokens: usize,
}

impl GdnGeometry {
    pub(crate) fn new(
        hq: usize,
        hk: usize,
        hv: usize,
        d: usize,
        tokens: usize,
    ) -> Result<Self, String> {
        if hq == 0 || hk == 0 || hv == 0 || d == 0 || tokens == 0 {
            return Err(format!(
                "GDN geometry dimensions must be non-zero: Hq={hq}, Hk={hk}, Hv={hv}, D={d}, T={tokens}"
            ));
        }
        if !hv.is_multiple_of(hk) {
            return Err(format!(
                "GDN geometry requires Hv divisible by Hk: Hk={hk}, Hv={hv}"
            ));
        }
        Ok(Self {
            hq,
            hk,
            hv,
            d,
            tokens,
        })
    }

    pub(crate) fn q_len(self) -> usize {
        self.tokens * self.hq * self.d
    }

    pub(crate) fn k_len(self) -> usize {
        self.tokens * self.hk * self.d
    }

    pub(crate) fn value_len(self) -> usize {
        self.tokens * self.hv * self.d
    }

    pub(crate) fn gate_len(self) -> usize {
        self.tokens * self.hv
    }

    pub(crate) fn state_len(self) -> usize {
        self.hv * self.d * self.d
    }

    pub(crate) fn contracts(self) -> GdnTensorContracts {
        let invocation = BufferLifetime::PrefillInvocation;
        let caller = BufferOwner::TestCaller;
        let bf16 = ElementDType::Bf16;
        let f32 = ElementDType::F32;
        GdnTensorContracts {
            // Physical storage is token-major [T,H,D]. These CuTe views are
            // zero-copy reinterpretations of that storage.
            q: TensorViewContract {
                logical_shape: [self.tokens, self.d, self.hq],
                element_strides: [self.hq * self.d, 1, self.d],
                dtype: bf16,
                owner: caller,
                lifetime: invocation,
            },
            k: TensorViewContract {
                logical_shape: [self.d, self.tokens, self.hk],
                element_strides: [1, self.hk * self.d, self.d],
                dtype: bf16,
                owner: caller,
                lifetime: invocation,
            },
            v: TensorViewContract {
                logical_shape: [self.d, self.tokens, self.hv],
                element_strides: [1, self.hv * self.d, self.d],
                dtype: bf16,
                owner: caller,
                lifetime: invocation,
            },
            output: TensorViewContract {
                logical_shape: [self.d, self.tokens, self.hv],
                element_strides: [1, self.hv * self.d, self.d],
                dtype: bf16,
                owner: caller,
                lifetime: invocation,
            },
            // Alpha is the per-token decay multiplier. The current Triton
            // prepare stage produces log(alpha), then a separate stage takes a
            // chunk-local cumulative sum. A future backend must not treat that
            // cumulative Triton intermediate as per-token alpha and accumulate
            // it a second time.
            alpha: TensorViewContract {
                logical_shape: [self.tokens, self.hv, 1],
                element_strides: [self.hv, 1, 1],
                dtype: f32,
                owner: caller,
                lifetime: invocation,
            },
            beta: TensorViewContract {
                logical_shape: [self.tokens, self.hv, 1],
                element_strides: [self.hv, 1, 1],
                dtype: f32,
                owner: caller,
                lifetime: invocation,
            },
            // Both state endpoints use [H,K,V], with V contiguous. Whether the
            // two endpoints alias is a property of the test case, not a layout
            // change.
            initial_state: TensorViewContract {
                logical_shape: [self.hv, self.d, self.d],
                element_strides: [self.d * self.d, self.d, 1],
                dtype: f32,
                owner: BufferOwner::RequestRecurrentState,
                lifetime: BufferLifetime::Request,
            },
            final_state: TensorViewContract {
                logical_shape: [self.hv, self.d, self.d],
                element_strides: [self.d * self.d, self.d, 1],
                dtype: f32,
                owner: BufferOwner::RequestRecurrentState,
                lifetime: BufferLifetime::Request,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GdnTensorContracts {
    pub(crate) q: TensorViewContract,
    pub(crate) k: TensorViewContract,
    pub(crate) v: TensorViewContract,
    pub(crate) output: TensorViewContract,
    pub(crate) alpha: TensorViewContract,
    pub(crate) beta: TensorViewContract,
    pub(crate) initial_state: TensorViewContract,
    pub(crate) final_state: TensorViewContract,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StateAliasMode {
    Separate,
    InPlace,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GdnCaseSpec {
    pub(crate) label: String,
    pub(crate) geometry: GdnGeometry,
    pub(crate) base_pos: usize,
    /// Serial prefill segments. More than one segment means resumed prefill.
    pub(crate) segments: Vec<usize>,
    pub(crate) state_alias: StateAliasMode,
    pub(crate) seed: u32,
}

impl GdnCaseSpec {
    pub(crate) fn new(
        label: impl Into<String>,
        geometry: GdnGeometry,
        base_pos: usize,
        segments: Vec<usize>,
        state_alias: StateAliasMode,
        seed: u32,
    ) -> Result<Self, String> {
        base_pos.checked_add(geometry.tokens).ok_or_else(|| {
            format!(
                "GDN case base_pos overflow: base_pos={base_pos}, T={}",
                geometry.tokens
            )
        })?;
        if segments.is_empty() || segments.iter().any(|&segment| segment == 0) {
            return Err("GDN case segments must be non-empty and positive".to_string());
        }
        let segment_tokens = segments.iter().try_fold(0usize, |total, &segment| {
            total
                .checked_add(segment)
                .ok_or_else(|| "GDN case segment length overflow".to_string())
        })?;
        if segment_tokens != geometry.tokens {
            return Err(format!(
                "GDN case segments total {segment_tokens}, expected T={}",
                geometry.tokens
            ));
        }
        Ok(Self {
            label: label.into(),
            geometry,
            base_pos,
            segments,
            state_alias,
            seed,
        })
    }

    pub(crate) fn materialize(&self) -> GdnHostInputs {
        let g = self.geometry;
        let make_bf16_bits = |len: usize, salt: u32| {
            (0..len)
                .map(|index| f32_to_bf16_bits(deterministic_signal(index, self.seed, salt)))
                .collect()
        };
        let alpha = (0..g.gate_len())
            .map(|index| {
                let rate = 0.0025 * (1 + ((index + self.seed as usize) % 29)) as f32;
                (-rate).exp()
            })
            .collect();
        let beta = (0..g.gate_len())
            .map(|index| 0.05 + 0.9 * ((index * 17 + self.seed as usize) % 101) as f32 / 100.0)
            .collect();
        let initial_state = (0..g.state_len())
            .map(|index| {
                let v = index % g.d;
                let rest = index / g.d;
                let k = rest % g.d;
                let h = rest / g.d;
                asymmetric_state_value(h, k, v)
            })
            .collect();
        let first_decode = GdnDecodeHandoffInputs {
            position: self
                .base_pos
                .checked_add(g.tokens)
                .expect("case construction already checked end-position overflow"),
            q: make_bf16_bits(g.hq * g.d, 41),
            k: make_bf16_bits(g.hk * g.d, 43),
            v: make_bf16_bits(g.hv * g.d, 47),
            alpha: (0..g.hv)
                .map(|head| {
                    let rate = 0.0025 * (1 + ((head + self.seed as usize + 53) % 29)) as f32;
                    (-rate).exp()
                })
                .collect(),
            beta: (0..g.hv)
                .map(|head| {
                    0.05 + 0.9 * ((head * 17 + self.seed as usize + 59) % 101) as f32 / 100.0
                })
                .collect(),
        };
        GdnHostInputs {
            spec: self.clone(),
            q: make_bf16_bits(g.q_len(), 11),
            k: make_bf16_bits(g.k_len(), 23),
            v: make_bf16_bits(g.value_len(), 37),
            alpha,
            beta,
            initial_state,
            first_decode,
        }
    }
}

fn deterministic_signal(index: usize, seed: u32, salt: u32) -> f32 {
    let mixed = index
        .wrapping_mul(131)
        .wrapping_add(seed as usize * 17)
        .wrapping_add(salt as usize * 43);
    (mixed % 257) as f32 / 128.0 - 1.0
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let round_to_even = (bits >> 16) & 1;
    (bits.wrapping_add(0x7fff + round_to_even) >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub(crate) fn asymmetric_state_value(h: usize, k: usize, v: usize) -> f32 {
    (h * 100_000 + k * 100 + v) as f32
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GdnDecodeHandoffInputs {
    /// Absolute position of the first token decoded after this prefill.
    pub(crate) position: usize,
    /// Raw BF16 element bits for one native-Hq query row.
    pub(crate) q: Vec<u16>,
    /// Raw BF16 element bits for one native-Hk key row.
    pub(crate) k: Vec<u16>,
    /// Raw BF16 element bits for one Hv value row.
    pub(crate) v: Vec<u16>,
    pub(crate) alpha: Vec<f32>,
    pub(crate) beta: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GdnHostInputs {
    pub(crate) spec: GdnCaseSpec,
    /// Raw BF16 element bits in token-major `[T,H,D]` storage.
    pub(crate) q: Vec<u16>,
    /// Raw BF16 element bits in token-major `[T,H,D]` storage.
    pub(crate) k: Vec<u16>,
    /// Raw BF16 element bits in token-major `[T,H,D]` storage.
    pub(crate) v: Vec<u16>,
    /// Per-token decay multipliers, not the Triton chunk-local cumulative log gate.
    pub(crate) alpha: Vec<f32>,
    pub(crate) beta: Vec<f32>,
    pub(crate) initial_state: Vec<f32>,
    /// Deterministic next-token input for the prefill-to-decode state handoff.
    pub(crate) first_decode: GdnDecodeHandoffInputs,
}

impl GdnHostInputs {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let g = self.spec.geometry;
        for (name, actual, expected) in [
            ("q", self.q.len(), g.q_len()),
            ("k", self.k.len(), g.k_len()),
            ("v", self.v.len(), g.value_len()),
            ("alpha", self.alpha.len(), g.gate_len()),
            ("beta", self.beta.len(), g.gate_len()),
            ("initial_state", self.initial_state.len(), g.state_len()),
        ] {
            if actual != expected {
                return Err(format!(
                    "{} {name} length mismatch: observed {actual}, expected {expected}",
                    self.spec.label
                ));
            }
        }
        if let Some((index, value)) =
            self.alpha.iter().copied().enumerate().find(|(_, value)| {
                !value.is_finite() || !(0.0..=1.0).contains(value) || *value == 0.0
            })
        {
            return Err(format!(
                "{} alpha[{index}]={value} is not a finite value in (0, 1]",
                self.spec.label
            ));
        }
        if let Some((index, value)) = self
            .beta
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(format!(
                "{} beta[{index}]={value} is not a finite value in [0, 1]",
                self.spec.label
            ));
        }
        if let Some((index, value)) = self
            .initial_state
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "{} initial_state[{index}]={value} is not finite",
                self.spec.label
            ));
        }
        for (name, actual, expected) in [
            ("first_decode.q", self.first_decode.q.len(), g.hq * g.d),
            ("first_decode.k", self.first_decode.k.len(), g.hk * g.d),
            ("first_decode.v", self.first_decode.v.len(), g.hv * g.d),
            ("first_decode.alpha", self.first_decode.alpha.len(), g.hv),
            ("first_decode.beta", self.first_decode.beta.len(), g.hv),
        ] {
            if actual != expected {
                return Err(format!(
                    "{} {name} length mismatch: observed {actual}, expected {expected}",
                    self.spec.label
                ));
            }
        }
        let expected_decode_position = self
            .spec
            .base_pos
            .checked_add(g.tokens)
            .ok_or_else(|| "first decode position overflow".to_string())?;
        if self.first_decode.position != expected_decode_position {
            return Err(format!(
                "{} first decode position mismatch: observed {}, expected {expected_decode_position}",
                self.spec.label, self.first_decode.position
            ));
        }
        if let Some((index, value)) = self
            .first_decode
            .alpha
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite() || !(0.0..=1.0).contains(value) || *value == 0.0)
        {
            return Err(format!(
                "{} first_decode.alpha[{index}]={value} is not a finite value in (0, 1]",
                self.spec.label
            ));
        }
        if let Some((index, value)) = self
            .first_decode
            .beta
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(format!(
                "{} first_decode.beta[{index}]={value} is not a finite value in [0, 1]",
                self.spec.label
            ));
        }
        Ok(())
    }

    /// Raw log decay emitted by today's Triton prepare stage, before cumsum.
    pub(crate) fn triton_log_decay(&self) -> Vec<f32> {
        self.alpha.iter().map(|alpha| alpha.ln()).collect()
    }

    /// The chunk-local cumulative log decay consumed by later Triton stages.
    pub(crate) fn triton_chunk_local_cumsum(&self, chunk_size: usize) -> Result<Vec<f32>, String> {
        if chunk_size == 0 {
            return Err("Triton cumsum chunk_size must be positive".to_string());
        }
        let g = self.spec.geometry;
        let raw = self.triton_log_decay();
        let mut cumulative = vec![0.0; raw.len()];
        for head in 0..g.hv {
            let mut running = 0.0;
            for token in 0..g.tokens {
                if token % chunk_size == 0 {
                    running = 0.0;
                }
                let index = token * g.hv + head;
                running += raw[index];
                cumulative[index] = running;
            }
        }
        Ok(cumulative)
    }

    pub(crate) fn fingerprint(&self) -> u64 {
        fn mix(hash: &mut u64, bits: u64) {
            *hash ^= bits;
            *hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        let mut hash = 0xcbf2_9ce4_8422_2325;
        let g = self.spec.geometry;
        for value in [
            g.hq,
            g.hk,
            g.hv,
            g.d,
            g.tokens,
            self.spec.base_pos,
            self.spec.seed as usize,
            match self.spec.state_alias {
                StateAliasMode::Separate => 0,
                StateAliasMode::InPlace => 1,
            },
        ] {
            mix(&mut hash, value as u64);
        }
        for &segment in &self.spec.segments {
            mix(&mut hash, segment as u64);
        }
        for value in &self.q {
            mix(&mut hash, *value as u64);
        }
        for value in &self.k {
            mix(&mut hash, *value as u64);
        }
        for value in &self.v {
            mix(&mut hash, *value as u64);
        }
        for value in self
            .alpha
            .iter()
            .chain(&self.beta)
            .chain(&self.initial_state)
        {
            mix(&mut hash, value.to_bits() as u64);
        }
        mix(&mut hash, self.first_decode.position as u64);
        for value in self
            .first_decode
            .q
            .iter()
            .chain(&self.first_decode.k)
            .chain(&self.first_decode.v)
        {
            mix(&mut hash, *value as u64);
        }
        for value in self
            .first_decode
            .alpha
            .iter()
            .chain(&self.first_decode.beta)
        {
            mix(&mut hash, value.to_bits() as u64);
        }
        hash
    }
}

pub(crate) trait GdnPrefillRunner {
    fn identity(&self) -> BackendIdentity;
    fn run(&mut self, inputs: &GdnHostInputs) -> Result<GdnRunResult, String>;
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GdnRunResult {
    pub(crate) backend: BackendIdentity,
    pub(crate) input_fingerprint: u64,
    pub(crate) output: Vec<f32>,
    pub(crate) final_state: Vec<f32>,
    pub(crate) first_decode_output: Vec<f32>,
    pub(crate) state_after_first_decode: Vec<f32>,
}

pub(crate) fn run_checked(
    expected_backend: BackendIdentity,
    runner: &mut dyn GdnPrefillRunner,
    inputs: &GdnHostInputs,
) -> Result<GdnRunResult, String> {
    inputs.validate()?;
    let actual_backend = runner.identity();
    if actual_backend != expected_backend {
        return Err(format!(
            "backend identity mismatch before launch: requested {expected_backend}, runner is {actual_backend}; implicit fallback is forbidden"
        ));
    }
    let result = runner.run(inputs)?;
    if result.backend != expected_backend {
        return Err(format!(
            "backend identity mismatch after launch: requested {expected_backend}, result reports {}; implicit fallback is forbidden",
            result.backend
        ));
    }
    if result.input_fingerprint != inputs.fingerprint() {
        return Err(format!(
            "{} result input fingerprint {:016x} does not match fixture {:016x}",
            result.backend,
            result.input_fingerprint,
            inputs.fingerprint()
        ));
    }
    let g = inputs.spec.geometry;
    if result.output.len() != g.value_len() {
        return Err(format!(
            "{} output length mismatch: observed {}, expected {}",
            result.backend,
            result.output.len(),
            g.value_len()
        ));
    }
    if result.final_state.len() != g.state_len() {
        return Err(format!(
            "{} final-state length mismatch: observed {}, expected {}",
            result.backend,
            result.final_state.len(),
            g.state_len()
        ));
    }
    if result.first_decode_output.len() != g.hv * g.d {
        return Err(format!(
            "{} first-decode output length mismatch: observed {}, expected {}",
            result.backend,
            result.first_decode_output.len(),
            g.hv * g.d
        ));
    }
    if result.state_after_first_decode.len() != g.state_len() {
        return Err(format!(
            "{} state-after-first-decode length mismatch: observed {}, expected {}",
            result.backend,
            result.state_after_first_decode.len(),
            g.state_len()
        ));
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FirstDifference {
    pub(crate) index: usize,
    pub(crate) reference: f32,
    pub(crate) candidate: f32,
    pub(crate) abs_diff: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DifferenceStats {
    pub(crate) count: usize,
    pub(crate) first_difference: Option<FirstDifference>,
    pub(crate) max_abs: f32,
    pub(crate) mean_abs: f32,
    pub(crate) p99_abs: f32,
}

impl DifferenceStats {
    pub(crate) fn compare(reference: &[f32], candidate: &[f32]) -> Result<Self, String> {
        if reference.len() != candidate.len() {
            return Err(format!(
                "comparison length mismatch: reference={}, candidate={}",
                reference.len(),
                candidate.len()
            ));
        }
        if reference.is_empty() {
            return Err("comparison inputs must be non-empty".to_string());
        }
        let mut diffs = Vec::with_capacity(reference.len());
        let mut first_difference = None;
        let mut sum = 0.0_f64;
        let mut max_abs = 0.0_f32;
        for (index, (&reference, &candidate)) in reference.iter().zip(candidate).enumerate() {
            if !reference.is_finite() || !candidate.is_finite() {
                return Err(format!(
                    "comparison contains non-finite value at index {index}: reference={reference}, candidate={candidate}"
                ));
            }
            let abs_diff = (reference - candidate).abs();
            if abs_diff != 0.0 && first_difference.is_none() {
                first_difference = Some(FirstDifference {
                    index,
                    reference,
                    candidate,
                    abs_diff,
                });
            }
            max_abs = max_abs.max(abs_diff);
            sum += abs_diff as f64;
            diffs.push(abs_diff);
        }
        diffs.sort_by(f32::total_cmp);
        let p99_index = ((diffs.len() as f64 * 0.99).ceil() as usize)
            .saturating_sub(1)
            .min(diffs.len() - 1);
        Ok(Self {
            count: diffs.len(),
            first_difference,
            max_abs,
            mean_abs: (sum / diffs.len() as f64) as f32,
            p99_abs: diffs[p99_index],
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GdnComparisonReport {
    pub(crate) reference_backend: BackendIdentity,
    pub(crate) candidate_backend: BackendIdentity,
    pub(crate) output: DifferenceStats,
    pub(crate) final_state: DifferenceStats,
    pub(crate) first_decode_output: DifferenceStats,
    pub(crate) state_after_first_decode: DifferenceStats,
}

impl GdnComparisonReport {
    pub(crate) fn compare(
        reference: &GdnRunResult,
        candidate: &GdnRunResult,
    ) -> Result<Self, String> {
        if reference.input_fingerprint != candidate.input_fingerprint {
            return Err(format!(
                "cannot compare different fixtures: reference={:016x}, candidate={:016x}",
                reference.input_fingerprint, candidate.input_fingerprint
            ));
        }
        Ok(Self {
            reference_backend: reference.backend,
            candidate_backend: candidate.backend,
            output: DifferenceStats::compare(&reference.output, &candidate.output)?,
            final_state: DifferenceStats::compare(&reference.final_state, &candidate.final_state)?,
            first_decode_output: DifferenceStats::compare(
                &reference.first_decode_output,
                &candidate.first_decode_output,
            )?,
            state_after_first_decode: DifferenceStats::compare(
                &reference.state_after_first_decode,
                &candidate.state_after_first_decode,
            )?,
        })
    }
}

pub(crate) fn sm120_boundary_case_specs() -> Vec<GdnCaseSpec> {
    let mut cases = Vec::with_capacity(BOUNDARY_TOKEN_LENGTHS.len() * 2);
    for tokens in BOUNDARY_TOKEN_LENGTHS {
        let geometry =
            GdnGeometry::new(16, 16, 32, 128, tokens).expect("fixed SM120 GDN geometry is valid");
        let segments = if tokens == 1 {
            vec![1]
        } else {
            let first = if tokens > 64 { 64 } else { tokens - 1 };
            vec![first, tokens - first]
        };
        for state_alias in [StateAliasMode::Separate, StateAliasMode::InPlace] {
            cases.push(
                GdnCaseSpec::new(
                    format!("sm120-t{tokens}-{state_alias:?}"),
                    geometry,
                    37,
                    segments.clone(),
                    state_alias,
                    691,
                )
                .expect("fixed boundary case is valid"),
            );
        }
    }
    cases
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoRunner {
        identity: BackendIdentity,
        reported_identity: BackendIdentity,
        seen_fingerprints: Vec<u64>,
    }

    impl EchoRunner {
        fn new(identity: BackendIdentity) -> Self {
            Self {
                identity,
                reported_identity: identity,
                seen_fingerprints: Vec::new(),
            }
        }
    }

    impl GdnPrefillRunner for EchoRunner {
        fn identity(&self) -> BackendIdentity {
            self.identity
        }

        fn run(&mut self, inputs: &GdnHostInputs) -> Result<GdnRunResult, String> {
            let fingerprint = inputs.fingerprint();
            self.seen_fingerprints.push(fingerprint);
            Ok(GdnRunResult {
                backend: self.reported_identity,
                input_fingerprint: fingerprint,
                output: inputs
                    .v
                    .iter()
                    .map(|&value| bf16_bits_to_f32(value))
                    .collect(),
                final_state: inputs.initial_state.clone(),
                first_decode_output: inputs
                    .first_decode
                    .v
                    .iter()
                    .map(|&value| bf16_bits_to_f32(value))
                    .collect(),
                state_after_first_decode: inputs.initial_state.clone(),
            })
        }
    }

    #[test]
    fn sm120_contract_has_zero_copy_views_and_hkv_state() {
        let geometry = GdnGeometry::new(16, 16, 32, 128, 65).unwrap();
        let c = geometry.contracts();
        assert_eq!(c.q.logical_shape, [65, 128, 16]);
        assert_eq!(c.q.element_strides, [2048, 1, 128]);
        assert_eq!(c.k.logical_shape, [128, 65, 16]);
        assert_eq!(c.k.element_strides, [1, 2048, 128]);
        assert_eq!(c.v.logical_shape, [128, 65, 32]);
        assert_eq!(c.v.element_strides, [1, 4096, 128]);
        assert_eq!(c.output, c.v);
        assert_eq!(c.initial_state.logical_shape, [32, 128, 128]);
        assert_eq!(c.initial_state.element_strides, [16_384, 128, 1]);
        assert_eq!(c.initial_state.dtype, ElementDType::F32);
        assert_eq!(c.initial_state.owner, BufferOwner::RequestRecurrentState);
        assert_eq!(c.initial_state.lifetime, BufferLifetime::Request);

        let t = 7;
        let h = 5;
        let d = 11;
        let q_view_offset =
            t * c.q.element_strides[0] + d * c.q.element_strides[1] + h * c.q.element_strides[2];
        assert_eq!(q_view_offset, (t * geometry.hq + h) * geometry.d + d);

        let k_view_offset =
            d * c.k.element_strides[0] + t * c.k.element_strides[1] + h * c.k.element_strides[2];
        assert_eq!(k_view_offset, (t * geometry.hk + h) * geometry.d + d);

        let v_view_offset =
            d * c.v.element_strides[0] + t * c.v.element_strides[1] + h * c.v.element_strides[2];
        assert_eq!(v_view_offset, (t * geometry.hv + h) * geometry.d + d);
    }

    #[test]
    fn boundary_matrix_covers_dynamic_t_resume_base_pos_and_alias() {
        let cases = sm120_boundary_case_specs();
        assert_eq!(cases.len(), BOUNDARY_TOKEN_LENGTHS.len() * 2);
        for &tokens in &BOUNDARY_TOKEN_LENGTHS {
            let matching: Vec<_> = cases
                .iter()
                .filter(|case| case.geometry.tokens == tokens)
                .collect();
            assert_eq!(matching.len(), 2);
            assert!(matching.iter().all(|case| case.base_pos == 37));
            assert!(
                matching
                    .iter()
                    .all(|case| case.segments.iter().sum::<usize>() == tokens)
            );
            if tokens > 1 {
                assert!(matching.iter().all(|case| case.segments.len() == 2));
            }
            assert!(
                matching
                    .iter()
                    .any(|case| case.state_alias == StateAliasMode::Separate)
            );
            assert!(
                matching
                    .iter()
                    .any(|case| case.state_alias == StateAliasMode::InPlace)
            );
        }
    }

    #[test]
    fn deterministic_fixture_is_nonzero_asymmetric_and_tracks_gate_semantics() {
        let geometry = GdnGeometry::new(2, 2, 4, 3, 5).unwrap();
        let spec = GdnCaseSpec::new(
            "small-fixture",
            geometry,
            9,
            vec![2, 3],
            StateAliasMode::InPlace,
            691,
        )
        .unwrap();
        let inputs = spec.materialize();
        inputs.validate().unwrap();
        assert!(inputs.q.iter().any(|value| *value != 0));
        assert!(inputs.k.iter().any(|value| *value != 0));
        assert!(inputs.v.iter().any(|value| *value != 0));
        assert_eq!(inputs.first_decode.position, 14);
        assert!(inputs.first_decode.q.iter().any(|value| *value != 0));
        assert!(inputs.first_decode.k.iter().any(|value| *value != 0));
        assert!(inputs.first_decode.v.iter().any(|value| *value != 0));

        let index = (2 * geometry.d + 1) * geometry.d;
        assert_eq!(inputs.initial_state[index], asymmetric_state_value(2, 1, 0));
        assert_ne!(
            asymmetric_state_value(2, 1, 0),
            asymmetric_state_value(2, 0, 1),
            "fixture must detect a K/V axis swap"
        );

        let raw = inputs.triton_log_decay();
        let cumulative = inputs.triton_chunk_local_cumsum(4).unwrap();
        let head = 3;
        assert!((raw[head].exp() - inputs.alpha[head]).abs() < 1e-6);
        assert!(
            (cumulative[geometry.hv + head] - raw[head] - raw[geometry.hv + head]).abs() < 1e-6
        );
        assert!((cumulative[4 * geometry.hv + head] - raw[4 * geometry.hv + head]).abs() < 1e-6);

        let mut different_base_pos = inputs.clone();
        different_base_pos.spec.base_pos += 1;
        assert_ne!(inputs.fingerprint(), different_base_pos.fingerprint());
    }

    #[test]
    fn observed_contract_mismatches_fail_explicitly() {
        let q = GdnGeometry::new(16, 16, 32, 128, 63).unwrap().contracts().q;
        let exact = ObservedTensorView {
            logical_shape: q.logical_shape,
            element_strides: q.element_strides,
            dtype: q.dtype,
        };
        q.validate_observed("q", exact).unwrap();

        let mut wrong_stride = exact;
        wrong_stride.element_strides[0] += 1;
        assert!(
            q.validate_observed("q", wrong_stride)
                .unwrap_err()
                .contains("stride mismatch")
        );

        let mut wrong_shape = exact;
        wrong_shape.logical_shape[0] += 1;
        assert!(
            q.validate_observed("q", wrong_shape)
                .unwrap_err()
                .contains("shape mismatch")
        );

        let mut wrong_dtype = exact;
        wrong_dtype.dtype = ElementDType::F32;
        assert!(
            q.validate_observed("q", wrong_dtype)
                .unwrap_err()
                .contains("dtype mismatch")
        );
    }

    #[test]
    fn runners_share_one_fixture_and_cannot_implicitly_fallback() {
        let geometry = GdnGeometry::new(2, 2, 4, 3, 2).unwrap();
        let inputs = GdnCaseSpec::new(
            "runner-fixture",
            geometry,
            5,
            vec![1, 1],
            StateAliasMode::Separate,
            17,
        )
        .unwrap()
        .materialize();
        let mut reference = EchoRunner::new(BackendIdentity::CpuStepwise);
        let mut triton = EchoRunner::new(BackendIdentity::TritonChunkwise);

        let reference_result =
            run_checked(BackendIdentity::CpuStepwise, &mut reference, &inputs).unwrap();
        let triton_result =
            run_checked(BackendIdentity::TritonChunkwise, &mut triton, &inputs).unwrap();
        assert_eq!(reference.seen_fingerprints, triton.seen_fingerprints);
        let report = GdnComparisonReport::compare(&reference_result, &triton_result).unwrap();
        assert_eq!(report.output.max_abs, 0.0);
        assert_eq!(report.final_state.max_abs, 0.0);
        assert_eq!(report.first_decode_output.max_abs, 0.0);
        assert_eq!(report.state_after_first_decode.max_abs, 0.0);

        let err = run_checked(BackendIdentity::FlashInferCute, &mut triton, &inputs).unwrap_err();
        assert!(err.contains("implicit fallback is forbidden"));

        let mut lying_runner = EchoRunner::new(BackendIdentity::FlashInferCute);
        lying_runner.reported_identity = BackendIdentity::TritonChunkwise;
        let err =
            run_checked(BackendIdentity::FlashInferCute, &mut lying_runner, &inputs).unwrap_err();
        assert!(err.contains("after launch"));
    }

    #[test]
    fn error_report_separates_prefill_and_first_decode_statistics() {
        let reference = GdnRunResult {
            backend: BackendIdentity::CpuStepwise,
            input_fingerprint: 7,
            output: vec![0.0, 1.0, 2.0, 3.0],
            final_state: vec![10.0, 20.0, 30.0, 40.0],
            first_decode_output: vec![5.0, 6.0],
            state_after_first_decode: vec![11.0, 21.0, 31.0, 41.0],
        };
        let candidate = GdnRunResult {
            backend: BackendIdentity::FlashInferCute,
            input_fingerprint: 7,
            output: vec![0.0, 1.25, 2.0, 3.5],
            final_state: vec![9.0, 20.0, 30.0, 40.0],
            first_decode_output: vec![5.0, 6.75],
            state_after_first_decode: vec![11.0, 21.0, 29.0, 41.0],
        };
        let report = GdnComparisonReport::compare(&reference, &candidate).unwrap();
        assert_eq!(report.output.first_difference.unwrap().index, 1);
        assert_eq!(report.output.max_abs, 0.5);
        assert_eq!(report.output.mean_abs, 0.1875);
        assert_eq!(report.output.p99_abs, 0.5);
        assert_eq!(report.final_state.first_difference.unwrap().index, 0);
        assert_eq!(report.final_state.max_abs, 1.0);
        assert_eq!(report.final_state.mean_abs, 0.25);
        assert_eq!(report.final_state.p99_abs, 1.0);
        assert_eq!(
            report.first_decode_output.first_difference.unwrap().index,
            1
        );
        assert_eq!(report.first_decode_output.max_abs, 0.75);
        assert_eq!(
            report
                .state_after_first_decode
                .first_difference
                .unwrap()
                .index,
            2
        );
        assert_eq!(report.state_after_first_decode.max_abs, 2.0);
    }
}
