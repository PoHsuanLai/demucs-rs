//! Wgpu E2E smoke: load htdemucs from the cache dir, run a 5-second
//! synthetic stereo signal through `Demucs::separate`, verify the
//! stems sum back to the original within SDR > 10 dB.
//!
//! The threshold is looser than `stem_sum.rs` (which uses 20 dB on real
//! audio) because synthetic content is out-of-distribution for the
//! model — it still reconstructs but with more leakage between stems.
//! The point isn't separation quality, just that the inference path
//! end-to-end on wgpu doesn't NaN, panic, or silently zero out.
//!
//! Cached weights: `~/Library/Caches/demucs-rs/htdemucs.safetensors`.
//! Run with: `cargo test -p demucs-core --features wgpu --test wgpu_smoke -- --ignored --nocapture`

use burn::backend::Wgpu;
use demucs_core::provider::fs::FsProvider;
use demucs_core::provider::ModelProvider;
use demucs_core::{Demucs, ModelOptions};

type B = Wgpu<f32, i32>;

const SR: u32 = 44_100;
const SECS: f32 = 5.0;

fn sdr_db(reference: &[f32], estimate: &[f32]) -> f64 {
    assert_eq!(reference.len(), estimate.len());
    let sig: f64 = reference.iter().map(|&x| (x as f64).powi(2)).sum();
    let noise: f64 = reference
        .iter()
        .zip(estimate.iter())
        .map(|(&r, &e)| ((r - e) as f64).powi(2))
        .sum();
    if noise < 1e-20 {
        return 100.0;
    }
    10.0 * (sig / noise).log10()
}

/// Synthetic "mix" — two sines + a low square mimicking
/// bass / lead / drum-ish content. Just enough variation that the
/// model doesn't trivially route everything to one stem.
fn synth_mix(n: usize, sr: u32) -> (Vec<f32>, Vec<f32>) {
    let dt = 1.0 / sr as f32;
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 * dt;
        let bass = (2.0 * std::f32::consts::PI * 110.0 * t).sin() * 0.3;
        let lead = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.2;
        let drum_env = (-((t % 0.5) * 30.0)).exp();
        let drum = (2.0 * std::f32::consts::PI * 60.0 * t).sin() * drum_env * 0.25;
        let mono = bass + lead + drum;
        // tiny stereo offset so left/right aren't bit-identical
        left.push(mono);
        right.push(mono * 0.95);
    }
    (left, right)
}

#[test]
#[ignore] // Requires cached weights + GPU
fn wgpu_separation_stem_sum() {
    eprintln!("→ loading cached htdemucs.safetensors");
    let provider = FsProvider::new().expect("no system cache dir");
    let info = ModelOptions::FourStem.model_info();
    let bytes = provider
        .load_cached(info)
        .expect("htdemucs not cached — download from huggingface.co/set-soft/audio_separation first");
    eprintln!("→ {} bytes of weights loaded", bytes.len());

    let device = Default::default();
    let demucs = Demucs::<B>::from_bytes(ModelOptions::FourStem, &bytes, device)
        .expect("model decode");
    eprintln!("→ model instantiated on Wgpu backend");

    let n = (SECS * SR as f32) as usize;
    let (left, right) = synth_mix(n, SR);
    eprintln!(
        "→ generated {} samples ({:.1}s @ {} Hz)",
        n, SECS, SR
    );

    eprintln!("→ running separate()");
    let t0 = std::time::Instant::now();
    let stems =
        pollster::block_on(demucs.separate(&left, &right, SR)).expect("separation failed");
    eprintln!(
        "→ separate done in {:.2?} ({} stems)",
        t0.elapsed(),
        stems.len()
    );
    assert_eq!(stems.len(), 4, "expected 4 stems");

    let mut sum_left = vec![0.0f32; n];
    let mut sum_right = vec![0.0f32; n];
    for (i, stem) in stems.iter().enumerate() {
        let len = stem.left.len().min(n);
        eprintln!(
            "  stem[{i}] len={} max_l={:.4} max_r={:.4}",
            stem.left.len(),
            stem.left.iter().copied().fold(0.0f32, f32::max),
            stem.right.iter().copied().fold(0.0f32, f32::max),
        );
        for j in 0..len {
            sum_left[j] += stem.left[j];
            sum_right[j] += stem.right[j];
        }
    }

    let sdr_l = sdr_db(&left, &sum_left);
    let sdr_r = sdr_db(&right, &sum_right);
    eprintln!("→ stem-sum SDR: L={sdr_l:.1} dB  R={sdr_r:.1} dB");

    // Sanity: no NaN / inf
    assert!(
        sum_left.iter().all(|x| x.is_finite()),
        "left sum has non-finite samples"
    );
    assert!(
        sum_right.iter().all(|x| x.is_finite()),
        "right sum has non-finite samples"
    );

    // Looser bound than the real-audio test — synthetic content is OOD.
    assert!(sdr_l > 10.0, "left SDR too low: {sdr_l:.1} dB");
    assert!(sdr_r > 10.0, "right SDR too low: {sdr_r:.1} dB");
}
