//! STFT preprocessing for ONNX inference path.
//!
//! Computes the magnitude spectrogram that HTDemucs ONNX model expects as input.
//! Ported from dawai-demucs/src/inference.rs.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Demucs operates at 44100 Hz internally
pub const DEMUCS_SAMPLE_RATE: u32 = 44100;

/// Training segment length in samples
pub const TRAINING_LENGTH: usize = 343980;

/// STFT parameters (must match the ONNX export)
pub const NFFT: usize = 4096;
pub const HOP_LENGTH: usize = NFFT / 4; // 1024

/// Overlap between chunks in samples (~1 second)
pub const OVERLAP: usize = 44100;

pub struct MagnitudeSpectrogram {
    /// Flat data in [4, 2048, T] layout (stereo × real/imag)
    pub data: Vec<f32>,
    pub freq_bins: usize,
    pub time_frames: usize,
}

/// Compute the magnitude spectrogram that HTDemucs expects as its second input.
pub fn compute_magnitude_spectrogram(
    left: &[f32],
    right: &[f32],
) -> MagnitudeSpectrogram {
    let channels = [left, right];
    let n = left.len();
    let hl = HOP_LENGTH;
    let le = (n + hl - 1) / hl;
    let pad = hl / 2 * 3;
    let right_pad = pad + le * hl - n;

    let window = hann_window(NFFT);
    let norm_factor = 1.0 / (NFFT as f32).sqrt();

    let mut all_real = Vec::new();
    let mut all_imag = Vec::new();

    for ch_data in &channels {
        let padded = reflect_pad(ch_data, pad, right_pad);
        let center_pad = NFFT / 2;
        let centered = reflect_pad(&padded, center_pad, center_pad);

        let stft = stft_frames(&centered, NFFT, hl, &window, norm_factor);
        let freq_bins_full = NFFT / 2 + 1;
        let freq_bins = freq_bins_full - 1;
        let total_time_frames = stft.len() / freq_bins_full;

        let mut ch_real = Vec::with_capacity(freq_bins * le);
        let mut ch_imag = Vec::with_capacity(freq_bins * le);

        for f in 0..freq_bins {
            for t in 0..le {
                let src_t = t + 2;
                if src_t < total_time_frames {
                    let idx = f * total_time_frames + src_t;
                    ch_real.push(stft[idx].re);
                    ch_imag.push(stft[idx].im);
                } else {
                    ch_real.push(0.0);
                    ch_imag.push(0.0);
                }
            }
        }

        all_real.push(ch_real);
        all_imag.push(ch_imag);
    }

    let freq_bins = NFFT / 2;
    let time_frames = le;

    // Channel order: [L_real, L_imag, R_real, R_imag]
    let total_size = 4 * freq_bins * time_frames;
    let mut data = Vec::with_capacity(total_size);
    data.extend_from_slice(&all_real[0]);
    data.extend_from_slice(&all_imag[0]);
    data.extend_from_slice(&all_real[1]);
    data.extend_from_slice(&all_imag[1]);

    MagnitudeSpectrogram {
        data,
        freq_bins,
        time_frames,
    }
}

pub struct AudioChunk {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub start: usize,
    pub actual_len: usize,
}

pub fn build_chunks(
    left: &[f32],
    right: &[f32],
    chunk_size: usize,
    overlap: usize,
) -> Vec<AudioChunk> {
    let total = left.len();
    if total <= chunk_size {
        return vec![AudioChunk {
            left: left.to_vec(),
            right: right.to_vec(),
            start: 0,
            actual_len: total,
        }];
    }

    let step = chunk_size - overlap;
    let mut chunks = Vec::new();
    let mut pos = 0;

    while pos < total {
        let end = (pos + chunk_size).min(total);
        let actual_len = end - pos;
        chunks.push(AudioChunk {
            left: left[pos..end].to_vec(),
            right: right[pos..end].to_vec(),
            start: pos,
            actual_len,
        });
        if end == total {
            break;
        }
        pos += step;
    }

    chunks
}

fn stft_frames(
    signal: &[f32],
    nfft: usize,
    hop: usize,
    window: &[f32],
    norm_factor: f32,
) -> Vec<Complex<f32>> {
    let freq_bins = nfft / 2 + 1;
    let num_frames = (signal.len().saturating_sub(nfft)) / hop + 1;

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(nfft);

    let mut result = vec![Complex::new(0.0, 0.0); freq_bins * num_frames];
    let mut frame_buf = vec![Complex::new(0.0, 0.0); nfft];

    for t in 0..num_frames {
        let offset = t * hop;
        for i in 0..nfft {
            let sample = if offset + i < signal.len() {
                signal[offset + i]
            } else {
                0.0
            };
            frame_buf[i] = Complex::new(sample * window[i] * norm_factor, 0.0);
        }

        fft.process(&mut frame_buf);

        for f in 0..freq_bins {
            result[f * num_frames + t] = frame_buf[f];
        }
    }

    result
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f32::consts::PI * 2.0 * i as f32 / n as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect()
}

fn reflect_pad(signal: &[f32], left_pad: usize, right_pad: usize) -> Vec<f32> {
    let n = signal.len();
    let total = left_pad + n + right_pad;
    let mut out = Vec::with_capacity(total);

    for i in 0..left_pad {
        let idx = reflect_index((left_pad - i) as isize, n);
        out.push(signal[idx]);
    }

    out.extend_from_slice(signal);

    for i in 0..right_pad {
        let idx = reflect_index((n as isize) - 2 - (i as isize), n);
        out.push(signal[idx]);
    }

    out
}

fn reflect_index(idx: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = idx;
    loop {
        if i < 0 {
            i = -i;
        }
        if i >= n {
            i = 2 * (n - 1) - i;
        }
        if i >= 0 && i < n {
            return i as usize;
        }
    }
}
