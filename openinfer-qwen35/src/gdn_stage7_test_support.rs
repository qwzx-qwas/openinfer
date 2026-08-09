//! CPU oracle and immutable numeric gates for the real-SM120 Stage 7 harness.
//!
//! This module is test-only.  In particular, none of these tolerances can be
//! changed through a serving or test environment variable during a paid GPU
//! session.

use half::f16;

use crate::gdn_prepare_test_contract::Fixture;
use crate::gdn_prepare_test_contract::Geometry;
use crate::gdn_prepare_test_contract::Prepared;
use crate::gdn_prepare_test_contract::bf16_to_f32;
use crate::gdn_prepare_test_contract::f32_to_bf16;
use crate::gdn_prepare_test_contract::prepare;

#[derive(Clone, Copy, Debug)]
pub(crate) struct NumericTolerance {
    pub(crate) atol: f32,
    pub(crate) rtol: f32,
}

/// Q/K are rounded to BF16 after an FP32 normalization reduction.  This
/// permits two BF16 steps around zero while remaining much narrower than the
/// operator tolerances used by the retired #709 candidate.
pub(crate) const PREPARE_QK_TOLERANCE: NumericTolerance = NumericTolerance {
    atol: 1.0 / 256.0,
    rtol: 0.0,
};

/// Alpha/beta stay FP32; only libdevice reduction/transcendental ordering may
/// differ between the scalar host oracle and the CUDA implementation.
pub(crate) const PREPARE_GATE_TOLERANCE: NumericTolerance = NumericTolerance {
    atol: 2.0e-6,
    rtol: 2.0e-6,
};

/// Prefill/decode outputs are stored as BF16.  State is accumulated in FP32.
/// The same fixed hybrid bound is applied to CPU↔Triton, CPU↔FlashInfer, and
/// Triton↔FlashInfer so no backend receives a looser gate.
pub(crate) const RECURRENCE_OUTPUT_TOLERANCE: NumericTolerance = NumericTolerance {
    atol: 1.0 / 64.0,
    rtol: 2.0e-3,
};
pub(crate) const RECURRENCE_STATE_TOLERANCE: NumericTolerance = NumericTolerance {
    atol: 5.0e-3,
    rtol: 2.0e-3,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FirstDifference {
    pub(crate) index: usize,
    pub(crate) reference: f32,
    pub(crate) candidate: f32,
    pub(crate) abs_diff: f32,
    pub(crate) allowed: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DifferenceStats {
    pub(crate) count: usize,
    pub(crate) first_difference: Option<FirstDifference>,
    pub(crate) first_violation: Option<FirstDifference>,
    pub(crate) max_abs: f32,
    pub(crate) mean_abs: f32,
    pub(crate) p99_abs: f32,
    pub(crate) max_rel: f32,
    pub(crate) violations: usize,
}

impl DifferenceStats {
    pub(crate) fn compare(
        reference: &[f32],
        candidate: &[f32],
        tolerance: NumericTolerance,
    ) -> Result<Self, String> {
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
        let mut first_violation = None;
        let mut sum = 0.0_f64;
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        let mut violations = 0;
        for (index, (&reference, &candidate)) in reference.iter().zip(candidate).enumerate() {
            if !reference.is_finite() || !candidate.is_finite() {
                return Err(format!(
                    "comparison contains non-finite value at index {index}: reference={reference}, candidate={candidate}"
                ));
            }
            let abs_diff = (reference - candidate).abs();
            let scale = reference.abs().max(candidate.abs());
            let allowed = tolerance.atol + tolerance.rtol * scale;
            let difference = FirstDifference {
                index,
                reference,
                candidate,
                abs_diff,
                allowed,
            };
            if abs_diff != 0.0 && first_difference.is_none() {
                first_difference = Some(difference);
            }
            if abs_diff > allowed {
                violations += 1;
                if first_violation.is_none() {
                    first_violation = Some(difference);
                }
            }
            max_abs = max_abs.max(abs_diff);
            max_rel = max_rel.max(abs_diff / scale.max(f32::MIN_POSITIVE));
            sum += f64::from(abs_diff);
            diffs.push(abs_diff);
        }
        diffs.sort_by(f32::total_cmp);
        let p99_index = ((diffs.len() as f64 * 0.99).ceil() as usize)
            .saturating_sub(1)
            .min(diffs.len() - 1);
        Ok(Self {
            count: diffs.len(),
            first_difference,
            first_violation,
            max_abs,
            mean_abs: (sum / diffs.len() as f64) as f32,
            p99_abs: diffs[p99_index],
            max_rel,
            violations,
        })
    }

    pub(crate) fn ensure_within(&self, label: &str) -> Result<(), String> {
        if self.violations == 0 {
            Ok(())
        } else {
            Err(format!(
                "{label} exceeded frozen tolerance at {}/{} elements; first violation {:?}; max_abs={}, mean_abs={}, p99_abs={}, max_rel={}",
                self.violations,
                self.count,
                self.first_violation,
                self.max_abs,
                self.mean_abs,
                self.p99_abs,
                self.max_rel
            ))
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CpuRunResult {
    pub(crate) output: Vec<f32>,
    pub(crate) final_state: Vec<f32>,
}

/// Serial Gated Delta Rule reference over already-prepared native Q/K/V and
/// per-token alpha/beta.  State is `[Hv,K,V]`, with V contiguous.
pub(crate) fn cpu_stepwise(
    geometry: Geometry,
    prepared: &Prepared,
    initial_state: &[f32],
) -> Result<CpuRunResult, String> {
    let expected_state = geometry.h_v * geometry.d * geometry.d;
    if initial_state.len() != expected_state
        || prepared.q.len() != geometry.q_len()
        || prepared.k.len() != geometry.k_len()
        || prepared.v.len() != geometry.v_len()
        || prepared.alpha.len() != geometry.gate_len()
        || prepared.beta.len() != geometry.gate_len()
    {
        return Err("CPU GDN reference input length mismatch".to_string());
    }
    if geometry.h_q != geometry.h_k || !geometry.h_v.is_multiple_of(geometry.h_k) {
        return Err("CPU GDN reference requires Hq=Hk and Hv divisible by Hk".to_string());
    }

    let mut state = initial_state.to_vec();
    let mut output = vec![0.0_f32; geometry.v_len()];
    let scale = 1.0_f32 / (geometry.d as f32).sqrt();
    for token in 0..geometry.tokens {
        for value_head in 0..geometry.h_v {
            let key_head = value_head * geometry.h_k / geometry.h_v;
            let q_base = (token * geometry.h_q + key_head) * geometry.d;
            let k_base = (token * geometry.h_k + key_head) * geometry.d;
            let v_base = (token * geometry.h_v + value_head) * geometry.d;
            let state_base = value_head * geometry.d * geometry.d;
            let alpha = prepared.alpha[token * geometry.h_v + value_head];
            let beta = prepared.beta[token * geometry.h_v + value_head];

            for key in 0..geometry.d {
                let row = state_base + key * geometry.d;
                for value in 0..geometry.d {
                    state[row + value] *= alpha;
                }
            }

            for value in 0..geometry.d {
                let mut memory = 0.0_f32;
                for key in 0..geometry.d {
                    memory += state[state_base + key * geometry.d + value]
                        * bf16_to_f32(prepared.k[k_base + key]);
                }
                let delta = (bf16_to_f32(prepared.v[v_base + value]) - memory) * beta;
                let mut out = 0.0_f32;
                for key in 0..geometry.d {
                    let index = state_base + key * geometry.d + value;
                    state[index] += delta * bf16_to_f32(prepared.k[k_base + key]);
                    out += state[index] * bf16_to_f32(prepared.q[q_base + key]) * scale;
                }
                // Both CUDA backends store the public operator output as BF16.
                output[v_base + value] = bf16_to_f32(f32_to_bf16(out));
            }
        }
    }
    Ok(CpuRunResult {
        output,
        final_state: state,
    })
}

fn round_to_bf16(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(value))
}

fn round_to_f16(value: f32) -> f32 {
    f16::from_f32(value).to_f32()
}

fn matmul_square(left: &[f32], right: &[f32], size: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; size * size];
    for row in 0..size {
        for col in 0..size {
            let mut value = 0.0_f32;
            for inner in 0..size {
                value += left[row * size + inner] * right[inner * size + col];
            }
            output[row * size + col] = value;
        }
    }
    output
}

/// Mirror FlashInfer's hierarchical FP16 `CollectiveInverse` for a unit
/// lower-triangular matrix.  Every level writes its inverse back to FP16, and
/// the off-diagonal block has an additional FP16 boundary between its two
/// HMMA products.
fn flashinfer_unit_lower_inverse(strict_lower_f16: &[f32], size: usize) -> Vec<f32> {
    debug_assert!(size.is_power_of_two() && size >= 8);
    debug_assert_eq!(strict_lower_f16.len(), size * size);

    if size == 8 {
        let mut inverse = vec![0.0_f32; size * size];
        for row in 0..size {
            inverse[row * size + row] = 1.0;
            for col in 0..row {
                let mut value = 0.0_f32;
                for inner in col..row {
                    value -= strict_lower_f16[row * size + inner] * inverse[inner * size + col];
                }
                inverse[row * size + col] = value;
            }
        }
        for value in &mut inverse {
            *value = round_to_f16(*value);
        }
        return inverse;
    }

    let half = size / 2;
    let mut a = vec![0.0_f32; half * half];
    let mut c = vec![0.0_f32; half * half];
    let mut d = vec![0.0_f32; half * half];
    for row in 0..half {
        for col in 0..half {
            a[row * half + col] = strict_lower_f16[row * size + col];
            c[row * half + col] = strict_lower_f16[(row + half) * size + col];
            d[row * half + col] = strict_lower_f16[(row + half) * size + col + half];
        }
    }
    let a_inverse = flashinfer_unit_lower_inverse(&a, half);
    let d_inverse = flashinfer_unit_lower_inverse(&d, half);

    let mut d_inverse_c = matmul_square(&d_inverse, &c, half);
    for value in &mut d_inverse_c {
        *value = round_to_f16(-*value);
    }
    let mut lower_inverse = matmul_square(&d_inverse_c, &a_inverse, half);
    for value in &mut lower_inverse {
        *value = round_to_f16(*value);
    }

    let mut inverse = vec![0.0_f32; size * size];
    for row in 0..half {
        for col in 0..half {
            inverse[row * size + col] = a_inverse[row * half + col];
            inverse[(row + half) * size + col] = lower_inverse[row * half + col];
            inverse[(row + half) * size + col + half] = d_inverse[row * half + col];
        }
    }
    inverse
}

/// Frozen CPU mirror of FlashInfer's 64-token SM120 final-state dataflow.
///
/// The semantic stepwise oracle remains the primary reference.  This mirror
/// additionally preserves the locked kernel's numeric boundaries: log2/exp2
/// alpha processing, FP16 hierarchical triangular inversion, BF16 T, BF16
/// state operands for SK, BF16 `(V-SK)`, and BF16 decayed NewV before the
/// final FP32-accumulating state GEMM.
pub(crate) fn cpu_blockwise_final_state(
    geometry: Geometry,
    prepared: &Prepared,
    initial_state: &[f32],
) -> Result<Vec<f32>, String> {
    const BLOCK_SIZE: usize = 64;

    let expected_state = geometry.h_v * geometry.d * geometry.d;
    if initial_state.len() != expected_state
        || prepared.k.len() != geometry.k_len()
        || prepared.v.len() != geometry.v_len()
        || prepared.alpha.len() != geometry.gate_len()
        || prepared.beta.len() != geometry.gate_len()
    {
        return Err("CPU blockwise GDN reference input length mismatch".to_string());
    }
    if geometry.h_q != geometry.h_k || !geometry.h_v.is_multiple_of(geometry.h_k) {
        return Err(
            "CPU blockwise GDN reference requires Hq=Hk and Hv divisible by Hk".to_string(),
        );
    }

    let d = geometry.d;
    let mut state = initial_state.to_vec();
    for value_head in 0..geometry.h_v {
        let key_head = value_head * geometry.h_k / geometry.h_v;
        let state_base = value_head * d * d;
        for block_start in (0..geometry.tokens).step_by(BLOCK_SIZE) {
            let block_len = (geometry.tokens - block_start).min(BLOCK_SIZE);
            let mut k = vec![0.0_f32; BLOCK_SIZE * d];
            let mut v = vec![0.0_f32; BLOCK_SIZE * d];
            let mut gamma_log2 = vec![0.0_f32; BLOCK_SIZE];
            let mut beta = vec![0.0_f32; BLOCK_SIZE];
            let mut cumulative_gamma_log2 = 0.0_f32;
            for token_in_block in 0..block_len {
                let token = block_start + token_in_block;
                let k_base = (token * geometry.h_k + key_head) * d;
                let v_base = (token * geometry.h_v + value_head) * d;
                for axis in 0..d {
                    k[token_in_block * d + axis] = bf16_to_f32(prepared.k[k_base + axis]);
                    v[token_in_block * d + axis] = bf16_to_f32(prepared.v[v_base + axis]);
                }
                cumulative_gamma_log2 +=
                    (prepared.alpha[token * geometry.h_v + value_head] + 1.0e-10).log2();
                gamma_log2[token_in_block] = cumulative_gamma_log2;
                beta[token_in_block] = prepared.beta[token * geometry.h_v + value_head];
            }
            for token_in_block in block_len..BLOCK_SIZE {
                gamma_log2[token_in_block] = cumulative_gamma_log2;
            }

            // KK is accumulated in FP32, scaled row-wise, and stored as FP16.
            // CollectiveInverse ignores the stored diagonal and supplies the
            // unit diagonal itself.
            let mut strict_lower_f16 = vec![0.0_f32; BLOCK_SIZE * BLOCK_SIZE];
            for row in 0..block_len {
                for col in 0..row {
                    let mut kk = 0.0_f32;
                    for axis in 0..d {
                        kk += k[row * d + axis] * k[col * d + axis];
                    }
                    strict_lower_f16[row * BLOCK_SIZE + col] =
                        round_to_f16(beta[row] * (gamma_log2[row] - gamma_log2[col]).exp2() * kk);
                }
            }

            // The inverse is reloaded from FP16, scaled column-wise by beta,
            // then written to the BF16 operand consumed by NewV HMMA.
            let inverse = flashinfer_unit_lower_inverse(&strict_lower_f16, BLOCK_SIZE);
            let mut t = vec![0.0_f32; BLOCK_SIZE * BLOCK_SIZE];
            for row in 0..block_len {
                for col in 0..=row {
                    t[row * BLOCK_SIZE + col] =
                        round_to_bf16(inverse[row * BLOCK_SIZE + col] * beta[col]);
                }
            }

            // The SM120 kernel converts the FP32 running state to BF16 before
            // SK, rounds scaled SK to BF16, and stores V-SK as BF16 before the
            // NewV×T HMMA.
            let state_operand: Vec<f32> = state[state_base..state_base + d * d]
                .iter()
                .copied()
                .map(round_to_bf16)
                .collect();
            let mut residual = vec![0.0_f32; BLOCK_SIZE * d];
            for token_in_block in 0..block_len {
                let gamma = gamma_log2[token_in_block].exp2();
                for value_axis in 0..d {
                    let mut memory = 0.0_f32;
                    for key_axis in 0..d {
                        memory += state_operand[key_axis * d + value_axis]
                            * k[token_in_block * d + key_axis];
                    }
                    residual[token_in_block * d + value_axis] = round_to_bf16(
                        v[token_in_block * d + value_axis] - round_to_bf16(gamma * memory),
                    );
                }
            }

            let mut new_v = vec![0.0_f32; BLOCK_SIZE * d];
            for row in 0..block_len {
                for value_axis in 0..d {
                    let mut value = 0.0_f32;
                    for col in 0..=row {
                        value += t[row * BLOCK_SIZE + col] * residual[col * d + value_axis];
                    }
                    new_v[row * d + value_axis] = value;
                }
            }

            let block_gamma_log2 = gamma_log2[block_len - 1];
            let block_decay = block_gamma_log2.exp2();
            let mut decayed_new_v = vec![0.0_f32; block_len * d];
            for token_in_block in 0..block_len {
                let decay = (block_gamma_log2 - gamma_log2[token_in_block]).exp2();
                for value_axis in 0..d {
                    decayed_new_v[token_in_block * d + value_axis] =
                        round_to_bf16(decay * new_v[token_in_block * d + value_axis]);
                }
            }
            for key_axis in 0..d {
                for value_axis in 0..d {
                    let mut increment = 0.0_f32;
                    for token_in_block in 0..block_len {
                        increment += k[token_in_block * d + key_axis]
                            * decayed_new_v[token_in_block * d + value_axis];
                    }
                    let index = state_base + key_axis * d + value_axis;
                    state[index] = block_decay * state[index] + increment;
                }
            }
        }
    }
    Ok(state)
}

/// One production-decode step from raw fused Q/K/V and gates.  Unlike the
/// prefill prepare path, the decode CUDA kernel keeps normalized Q/K in FP32
/// registers instead of rounding them through BF16 scratch.
pub(crate) fn cpu_decode_from_raw(
    fixture: &Fixture,
    initial_state: &[f32],
) -> Result<CpuRunResult, String> {
    let geometry = fixture.geometry;
    if geometry.tokens != 1 {
        return Err("CPU raw decode reference requires exactly one token".to_string());
    }
    let prepared = prepare(fixture)?;
    let expected_state = geometry.h_v * geometry.d * geometry.d;
    if initial_state.len() != expected_state {
        return Err("CPU raw decode state length mismatch".to_string());
    }

    let normalize = |bits: &[u16]| {
        let values: Vec<f32> = bits.iter().copied().map(bf16_to_f32).collect();
        let inv_norm = (values.iter().map(|value| value * value).sum::<f32>() + 1.0e-12)
            .sqrt()
            .recip();
        values
            .into_iter()
            .map(|value| value * inv_norm)
            .collect::<Vec<_>>()
    };
    let mut q = Vec::with_capacity(geometry.h_q * geometry.d);
    let mut k = Vec::with_capacity(geometry.h_k * geometry.d);
    for head in 0..geometry.h_q {
        let start = fixture.offsets.q + head * geometry.d;
        q.extend(normalize(&fixture.qkv[start..start + geometry.d]));
    }
    for head in 0..geometry.h_k {
        let start = fixture.offsets.k + head * geometry.d;
        k.extend(normalize(&fixture.qkv[start..start + geometry.d]));
    }

    let mut state = initial_state.to_vec();
    let mut output = vec![0.0_f32; geometry.h_v * geometry.d];
    let scale = 1.0_f32 / (geometry.d as f32).sqrt();
    for value_head in 0..geometry.h_v {
        let key_head = value_head * geometry.h_k / geometry.h_v;
        let q_base = key_head * geometry.d;
        let k_base = key_head * geometry.d;
        let v_base = value_head * geometry.d;
        let state_base = value_head * geometry.d * geometry.d;
        let alpha = prepared.alpha[value_head];
        let beta = prepared.beta[value_head];

        for key in 0..geometry.d {
            let row = state_base + key * geometry.d;
            for value in 0..geometry.d {
                state[row + value] *= alpha;
            }
        }
        for value in 0..geometry.d {
            let mut memory = 0.0_f32;
            for key_index in 0..geometry.d {
                memory +=
                    state[state_base + key_index * geometry.d + value] * k[k_base + key_index];
            }
            let delta = (bf16_to_f32(prepared.v[v_base + value]) - memory) * beta;
            let mut out = 0.0_f32;
            for key_index in 0..geometry.d {
                let index = state_base + key_index * geometry.d + value;
                state[index] += delta * k[k_base + key_index];
                out += state[index] * q[q_base + key_index] * scale;
            }
            output[v_base + value] = bf16_to_f32(f32_to_bf16(out));
        }
    }
    Ok(CpuRunResult {
        output,
        final_state: state,
    })
}

pub(crate) fn asymmetric_hkv_state(geometry: Geometry) -> Vec<f32> {
    (0..geometry.h_v * geometry.d * geometry.d)
        .map(|index| {
            let head = index / (geometry.d * geometry.d);
            let rem = index % (geometry.d * geometry.d);
            let key = rem / geometry.d;
            let value = rem % geometry.d;
            // A scaled version of h*100000+k*100+v keeps every axis
            // distinguishable without making BF16 output overflow dominate.
            (head * 100_000 + key * 100 + value) as f32 * 1.0e-6 - 0.2
        })
        .collect()
}

/// Deliberate K/V transpose used only to prove the asymmetric oracle would
/// reject the unpatched upstream HVK interpretation when K==V==128.
pub(crate) fn transpose_kv_as_wrong_hvk(geometry: Geometry, hkv: &[f32]) -> Vec<f32> {
    let mut wrong = vec![0.0_f32; hkv.len()];
    for head in 0..geometry.h_v {
        for key in 0..geometry.d {
            for value in 0..geometry.d {
                let destination = (head * geometry.d + key) * geometry.d + value;
                let source = (head * geometry.d + value) * geometry.d + key;
                wrong[destination] = hkv[source];
            }
        }
    }
    wrong
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn_prepare_test_contract::deterministic_fixture;
    use crate::gdn_prepare_test_contract::prepare;

    #[test]
    fn tolerance_report_identifies_first_violation() {
        let stats = DifferenceStats::compare(
            &[1.0, 2.0, 3.0],
            &[1.0, 2.01, 3.5],
            NumericTolerance {
                atol: 0.02,
                rtol: 0.0,
            },
        )
        .unwrap();
        assert_eq!(stats.violations, 1);
        assert_eq!(stats.first_violation.unwrap().index, 2);
        assert!(stats.ensure_within("negative-control").is_err());
    }

    #[test]
    fn cpu_stepwise_matches_hand_calculated_hkv_update() {
        let geometry = Geometry {
            h_q: 1,
            h_k: 1,
            h_v: 1,
            d: 2,
            tokens: 1,
        };
        let prepared = Prepared {
            q: vec![f32_to_bf16(1.0), f32_to_bf16(0.0)],
            k: vec![f32_to_bf16(1.0), f32_to_bf16(0.0)],
            v: vec![f32_to_bf16(2.0), f32_to_bf16(3.0)],
            alpha: vec![0.5],
            beta: vec![0.25],
        };
        let result = cpu_stepwise(geometry, &prepared, &[4.0, 5.0, 6.0, 7.0]).unwrap();
        assert_eq!(result.final_state, vec![2.0, 2.625, 3.0, 3.5]);
        let expected_output = vec![
            bf16_to_f32(f32_to_bf16(2.0 / 2.0_f32.sqrt())),
            bf16_to_f32(f32_to_bf16(2.625 / 2.0_f32.sqrt())),
        ];
        assert_eq!(result.output, expected_output);
    }

    #[test]
    fn cpu_stepwise_rejects_wrong_hvk_oracle() {
        let fixture = deterministic_fixture(2, 32);
        let prepared = prepare(&fixture).unwrap();
        let initial = asymmetric_hkv_state(fixture.geometry);
        let wrong = transpose_kv_as_wrong_hvk(fixture.geometry, &initial);
        let correct = cpu_stepwise(fixture.geometry, &prepared, &initial).unwrap();
        let wrong = cpu_stepwise(fixture.geometry, &prepared, &wrong).unwrap();
        let output =
            DifferenceStats::compare(&correct.output, &wrong.output, RECURRENCE_OUTPUT_TOLERANCE)
                .unwrap();
        let state = DifferenceStats::compare(
            &correct.final_state,
            &wrong.final_state,
            RECURRENCE_STATE_TOLERANCE,
        )
        .unwrap();
        assert!(output.violations > 0 || state.violations > 0);
    }

    #[test]
    fn cpu_blockwise_matches_stepwise_for_orthogonal_tokens() {
        let geometry = Geometry {
            h_q: 1,
            h_k: 1,
            h_v: 1,
            d: 2,
            tokens: 2,
        };
        let prepared = Prepared {
            q: vec![0; 4],
            k: vec![
                f32_to_bf16(1.0),
                f32_to_bf16(0.0),
                f32_to_bf16(0.0),
                f32_to_bf16(1.0),
            ],
            v: vec![
                f32_to_bf16(2.0),
                f32_to_bf16(3.0),
                f32_to_bf16(4.0),
                f32_to_bf16(5.0),
            ],
            alpha: vec![1.0, 1.0],
            beta: vec![1.0, 1.0],
        };
        let initial = vec![4.0, 5.0, 6.0, 7.0];
        let stepwise = cpu_stepwise(geometry, &prepared, &initial).unwrap();
        let blockwise = cpu_blockwise_final_state(geometry, &prepared, &initial).unwrap();
        assert_eq!(blockwise, stepwise.final_state);
    }

    #[test]
    fn cpu_blockwise_preserves_flashinfer_bf16_state_operand_boundary() {
        let geometry = Geometry {
            h_q: 1,
            h_k: 1,
            h_v: 1,
            d: 2,
            tokens: 1,
        };
        let prepared = Prepared {
            q: vec![0; 2],
            k: vec![f32_to_bf16(1.0), f32_to_bf16(0.0)],
            v: vec![0; 2],
            alpha: vec![1.0],
            beta: vec![1.0],
        };
        let initial = vec![1.001, 0.333, 0.0, 0.0];
        let stepwise = cpu_stepwise(geometry, &prepared, &initial).unwrap();
        let blockwise = cpu_blockwise_final_state(geometry, &prepared, &initial).unwrap();

        assert_eq!(stepwise.final_state[0], 0.0);
        assert_eq!(stepwise.final_state[1], 0.0);
        assert_eq!(blockwise[0], initial[0] - round_to_bf16(initial[0]));
        assert_eq!(blockwise[1], initial[1] - round_to_bf16(initial[1]));
    }
}
