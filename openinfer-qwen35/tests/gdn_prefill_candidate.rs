//! Explicit full-model Triton/FlashInfer GDN comparison harness.
//!
//! This is compiled with the Qwen3.5 test surface but remains ignored until
//! the SM120 correctness gates are run. Production dispatch is unaffected.

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use openinfer_qwen35::runtime::Qwen35Model;

#[test]
#[ignore = "requires Qwen3.5-4B weights, an SM120 GPU, and OPENINFER_GDN_STAGE3_MANIFEST"]
fn full_model_seam_compares_named_backends_at_boundary_lengths() -> Result<()> {
    let model_path = std::env::var("OPENINFER_TEST_MODEL_PATH")
        .context("set OPENINFER_TEST_MODEL_PATH to Qwen3.5-4B weights")?;
    let manifest = std::env::var("OPENINFER_GDN_STAGE3_MANIFEST")
        .context("set OPENINFER_GDN_STAGE3_MANIFEST to the Stage 3 manifest")?;
    let mut model = Qwen35Model::from_safetensors_with_options(&model_path, false)?;
    model.install_flashinfer_gdn_for_benchmark(Path::new(&manifest))?;

    for tokens in [1_usize, 2, 63, 64, 65, 127, 128] {
        let token_ids: Vec<u32> = (0..tokens).map(|index| 100 + index as u32).collect();
        let report = model.compare_gdn_prefill_backends(&token_ids)?;
        anyhow::ensure!(report.tokens == tokens, "comparison reported the wrong T");
        anyhow::ensure!(
            report.hidden_max_abs.is_finite()
                && report.recurrent_state_max_abs.is_finite()
                && report.conv_state_max_abs.is_finite(),
            "non-finite full-model comparison report at T={tokens}: {report:?}"
        );
        eprintln!("T={tokens}: {report:?}");
    }
    Ok(())
}
