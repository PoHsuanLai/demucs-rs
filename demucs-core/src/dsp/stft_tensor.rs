//! GPU-resident STFT/iSTFT for the HTDemucs inference pipeline.
//!
//! Direct port of HTDemucs's Python `_spec` / `_ispec` (in
//! `demucs/htdemucs.py`) and `spectro` / `ispectro` (in `demucs/spec.py`)
//! onto burn 0.21's `burn::tensor::signal::{stft, istft, hann_window}`.
//! The reference Python:
//!
//! ```python
//! # spec.py
//! def spectro(x, n_fft=512, hop_length=None, pad=0):
//!     z = th.stft(x, n_fft * (1 + pad), hop_length or n_fft // 4,
//!                 window=th.hann_window(n_fft).to(x),
//!                 win_length=n_fft, normalized=True, center=True,
//!                 return_complex=True, pad_mode='reflect')
//!     return z  # shape: [..., n_fft/2 + 1, frames]
//!
//! def ispectro(z, hop_length=None, length=None, pad=0):
//!     n_fft = 2 * (freqs := z.shape[-2]) - 2
//!     win_length = n_fft // (1 + pad)
//!     x = th.istft(z, n_fft, hop_length,
//!                  window=th.hann_window(win_length).to(z.real),
//!                  win_length=win_length, normalized=True,
//!                  length=length, center=True)
//!     return x
//!
//! # htdemucs.py
//! def _spec(self, x):              # x: [B, C, T]
//!     hl, nfft = self.hop_length, self.nfft  # 1024, 4096
//!     le = ceil(x.shape[-1] / hl)
//!     pad = hl // 2 * 3            # 1536
//!     x = pad1d(x, (pad, pad + le*hl - x.shape[-1]), mode='reflect')
//!     z = spectro(x, nfft, hl)[..., :-1, :]   # drop Nyquist
//!     z = z[..., 2: 2+le]                      # drop border frames
//!     return z                     # [B, C, n_fft/2, le]
//!
//! def _ispec(self, z, length=None, scale=0):
//!     hl = self.hop_length // (4**scale)
//!     z = F.pad(z, (0, 0, 0, 1))   # re-add Nyquist as zero
//!     z = F.pad(z, (2, 2))         # re-add 2 zero frames each side
//!     pad = hl // 2 * 3
//!     le = hl * ceil(length / hl) + 2*pad
//!     x = ispectro(z, hl, length=le)
//!     return x[..., pad: pad+length]
//! ```
//!
//! Burn's `signal::stft` returns `[batch, frames, freqs, 2]` (real, imag
//! split). The CaC packing the model wants is `[batch, 2, freqs, frames]`
//! — we permute at the end. Burn doesn't normalize so we apply
//! `× 1/sqrt(n_fft)` manually to match `normalized=True`.
//!
//! Lives alongside the CPU `Stft` (used by the display spectrogram path)
//! rather than replacing it — display rendering doesn't cross the WIT
//! bridge so it has no perf reason to leave realfft.

use burn::tensor::ops::PadMode;
use burn::tensor::signal::{hann_window, istft, stft, StftOptions};
use burn::{prelude::Backend, Tensor};

use crate::{HOP_LENGTH, N_FFT};

/// HTDemucs `_spec` ported to tensor ops. Returns CaC-packed
/// `[B, 2*C, n_fft/2, le]` where the channel dim interleaves
/// `[real_ch0, real_ch1, ..., imag_ch0, imag_ch1, ...]`.
///
/// Input `audio`: `[B, C, T]` time-domain stereo (typically `[1, 2, T]`).
pub fn spec_cac<B: Backend>(audio: Tensor<B, 3>) -> Tensor<B, 4> {
    let [batch, channels, t] = audio.dims();
    let hl = HOP_LENGTH;
    let nfft = N_FFT;
    let le = t.div_ceil(hl);
    let pad = hl / 2 * 3;
    let right_pad = pad + le * hl - t;

    // [B, C, T] → [B, C, T + pad + right_pad] with reflect padding,
    // matching demucs's `pad1d(x, (pad, right_pad), mode='reflect')`.
    let padded = audio.pad([(0, 0), (0, 0), (pad, right_pad)], PadMode::Reflect);
    let new_t = t + pad + right_pad;

    // Flatten [B, C, T'] → [B*C, T'] for stft
    let flat = padded.reshape([batch * channels, new_t]);

    let win = hann_window::<B>(nfft, true, &flat.device());
    let opts = StftOptions {
        n_fft: nfft,
        hop_length: hl,
        win_length: Some(nfft),
        center: true,
        onesided: true,
    };

    // [B*C, frames, n_fft/2 + 1, 2]
    let z = stft(flat, Some(win), opts);

    // Match PyTorch normalized=True: multiply by 1/sqrt(n_fft).
    let norm = 1.0 / (nfft as f32).sqrt();
    let z = z.mul_scalar(norm);

    let [_, n_frames, n_freqs, _] = z.dims();
    debug_assert_eq!(n_freqs, nfft / 2 + 1);
    debug_assert_eq!(n_frames, le + 4, "expected le+4 frames after center pad");

    // Drop Nyquist: narrow freq dim from n_fft/2+1 → n_fft/2
    let z = z.narrow(2, 0, nfft / 2);

    // Drop border frames: keep [2 .. 2+le] on the frame dim
    let z = z.narrow(1, 2, le);

    // Reshape [B*C, le, n_fft/2, 2] → [B, C, le, n_fft/2, 2]
    let z = z.reshape([batch, channels, le, nfft / 2, 2]);

    // CaC packing the model expects: [B, 2*C, F, le] where the 2*C dim
    // iterates `(channel, real/imag)` — i.e. [real_c0, imag_c0,
    // real_c1, imag_c1, ...]. The old CPU path produced this same
    // layout via separate per-channel stft_to_cac + cat(dim=0).
    // Permute [B, C, le, F, 2] → [B, C, 2, F, le] then reshape collapses
    // the (C, 2) pair into 2*C.
    let z = z.permute([0, 1, 4, 3, 2]); // [B, C, 2, F, le]
    z.reshape([batch, 2 * channels, nfft / 2, le])
}

/// HTDemucs `_ispec` ported to tensor ops. Input is CaC-packed
/// `[B, 2*C, n_fft/2, frames]`, output is `[B, C, length]`.
pub fn ispec_cac<B: Backend>(cac: Tensor<B, 4>, length: usize) -> Tensor<B, 3> {
    let [batch, two_c, freqs_in, frames_in] = cac.dims();
    let channels = two_c / 2;
    let hl = HOP_LENGTH;
    let nfft = N_FFT;
    debug_assert_eq!(freqs_in, nfft / 2);
    debug_assert_eq!(two_c % 2, 0);

    // Undo the CaC packing: [B, 2*C, F, T] with (c, ri) interleaving
    // → [B, C, 2, F, T] → [B, C, T, F, 2]
    let z = cac.reshape([batch, channels, 2, freqs_in, frames_in]);
    let z = z.permute([0, 1, 4, 3, 2]); // [B, C, T, F, 2]

    // Re-add the Nyquist bin as zero (PyTorch `F.pad(z, (0, 0, 0, 1))`).
    // F dim is at index 3; pad from `freqs_in` → `freqs_in + 1`.
    let zeros_nyq = Tensor::<B, 5>::zeros([batch, channels, frames_in, 1, 2], &z.device());
    let z = Tensor::cat(vec![z, zeros_nyq], 3); // [B, C, T, F+1, 2]

    // Re-add 2 zero frames on each side (PyTorch `F.pad(z, (2, 2))` —
    // last dim is real/imag, second-to-last is freqs, so this pads frames).
    let frames_pad = Tensor::<B, 5>::zeros(
        [batch, channels, 2, freqs_in + 1, 2],
        &z.device(),
    );
    let z = Tensor::cat(vec![frames_pad.clone(), z, frames_pad], 2); // [B, C, T+4, F+1, 2]
    let frames_total = frames_in + 4;

    // ispectro target length: hl * ceil(length / hl) + 2 * pad
    let pad = hl / 2 * 3;
    let le = hl * length.div_ceil(hl) + 2 * pad;

    // Flatten batch+channels for istft. Burn wants [batch, frames, freqs, 2].
    let z = z.reshape([batch * channels, frames_total, freqs_in + 1, 2]);

    let win = hann_window::<B>(nfft, true, &z.device());
    let opts = StftOptions {
        n_fft: nfft,
        hop_length: hl,
        win_length: Some(nfft),
        center: true,
        onesided: true,
    };

    // istft returns [batch, length]
    let x = istft(z, Some(win), Some(le), opts);

    // Match normalized=True on the inverse side: multiply by sqrt(n_fft).
    // (Forward divided by sqrt; round-trip should preserve, so inverse
    // multiplies by the same factor.)
    let inv_norm = (nfft as f32).sqrt();
    let x = x.mul_scalar(inv_norm);

    // Reshape back to [B, C, le] and trim [pad .. pad+length]
    let x = x.reshape([batch, channels, le]);
    x.narrow(2, pad, length)
}

// Unit tests need a backend that implements `rfft`. As of burn 0.21
// that's Wgpu only — ndarray panics. Gated on the `wgpu` feature
// (already a dev-dep) so `cargo test -p demucs-core` without features
// doesn't fail on machines without a GPU.
#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type B = Wgpu<f32, i32>;

    #[test]
    fn spec_cac_output_shape() {
        // _spec output: [B, 2*C, n_fft/2, le]
        // For TRAINING_LENGTH=343980, hl=1024 → le = ceil(343980/1024) = 336
        let device = Default::default();
        let t = crate::TRAINING_LENGTH;
        let x = Tensor::<B, 3>::zeros([1, 2, t], &device);
        let z = spec_cac(x);
        assert_eq!(z.dims(), [1, 4, N_FFT / 2, 336]);
    }

    #[test]
    fn ispec_recovers_length() {
        let device = Default::default();
        let length = crate::TRAINING_LENGTH;
        let x = Tensor::<B, 3>::zeros([1, 2, length], &device);
        let z = spec_cac(x);
        let recon = ispec_cac(z, length);
        assert_eq!(recon.dims(), [1, 2, length]);
    }

    /// Round-trip a real signal through spec_cac → ispec_cac and check
    /// interior samples reconstruct to within 1e-3. Boundary regions
    /// are skipped — the same way the old CPU `stft.rs::round_trip_*`
    /// test does — because `_spec`'s outer reflect pad + frame trim
    /// leaves the edges imperfectly invertible.
    #[test]
    fn round_trip_interior_matches() {
        use core::f32::consts::PI;
        let device = Default::default();
        let length = crate::TRAINING_LENGTH;

        // Build a 2-channel signal with mismatched content so the test
        // detects accidental channel crossovers.
        let mut data = Vec::with_capacity(2 * length);
        for c in 0..2 {
            for i in 0..length {
                let t = i as f32;
                let base = (2.0 * PI * 0.01 * t).sin() + 0.5 * (2.0 * PI * 0.1 * t).cos();
                data.push(if c == 0 { base } else { 0.5 * base });
            }
        }
        let x = Tensor::<B, 3>::from_data(
            burn::tensor::TensorData::new(data.clone(), [1, 2, length]),
            &device,
        );
        let z = spec_cac(x);
        let recon = ispec_cac(z, length);
        let recon_data: Vec<f32> = recon.into_data().to_vec().unwrap();

        // Match the old CPU test's boundary skip: 6 * hop_length.
        let skip = 6 * crate::HOP_LENGTH;
        for c in 0..2 {
            let mut max_err = 0.0f32;
            for i in skip..(length - skip) {
                let orig = data[c * length + i];
                let got = recon_data[c * length + i];
                max_err = max_err.max((orig - got).abs());
            }
            assert!(
                max_err < 1e-3,
                "channel {c} interior reconstruction error too large: {max_err}",
            );
        }
    }
}
