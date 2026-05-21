wit_bindgen::generate!({
    world: "dawai-extension",
    path: "wit/world.wit",
});

mod dsp;

use std::cell::RefCell;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use exports::dawai::extension::extension::Guest;

struct DemucsExtension;

/// Tensor JSON format matching dawai-gpu's TensorJson
#[derive(Debug, Serialize, Deserialize)]
struct TensorJson {
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// HuggingFace URL for the HTDemucs ONNX model (~290 MB)
const MODEL_URL: &str =
    "https://huggingface.co/smank/htdemucs-onnx/resolve/main/htdemucs.onnx";

/// HTDemucs was trained on 44.1 kHz audio. Inputs at any other rate
/// get resampled to this before chunking; stems get resampled back out
/// so they line up with the original at its native rate.
const MODEL_SAMPLE_RATE: u32 = 44_100;

/// Relative path (under the extension's sandboxed storage root) where
/// the ONNX model is downloaded to and loaded from.
const MODEL_PATH: &str = "htdemucs.onnx";

/// Sinc-resample one channel of f32 PCM. Pass-through if `from_sr ==
/// to_sr`. Uses `rubato::SincFixedIn` with the high-quality preset —
/// 256-tap kernel, cubic interpolation. ~10–20× slower than linear,
/// but a separation pass is already minutes; the resample cost is
/// negligible and the spectral artifacts of a cheaper filter would
/// leak into the separation.
fn resample(input: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>, String> {
    use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

    if from_sr == to_sr || input.is_empty() {
        return Ok(input.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Cubic,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(
        to_sr as f64 / from_sr as f64,
        2.0,
        params,
        input.len(),
        1,
    )
    .map_err(|e| format!("rubato init ({from_sr} -> {to_sr}): {e}"))?;

    let waves_in = vec![input.to_vec()];
    let waves_out = resampler
        .process(&waves_in, None)
        .map_err(|e| format!("rubato process: {e}"))?;
    waves_out
        .into_iter()
        .next()
        .ok_or_else(|| "rubato produced no output channels".to_string())
}

/// Panel ID (must match extension.toml)
const PANEL_ID: &str = "dawai.demucs";

/// Stored ONNX session ID after loading
thread_local! {
    static SESSION_ID: RefCell<Option<String>> = const { RefCell::new(None) };
}

// =============================================================================
// Commands
// =============================================================================

/// Check if the model file exists on disk.
fn model_exists() -> bool {
    dawai::extension::storage::file_exists(MODEL_PATH)
        .unwrap_or_else(|_| "false".into()) == "true"
}

fn do_download_model() -> Result<String, String> {
    if model_exists() {
        return Ok("Model already downloaded".into());
    }

    notify_info("Downloading HTDemucs model (~290 MB)...");

    // download_file resolves `filename` against the extension's
    // sandboxed storage root — pass a relative name.
    dawai::extension::storage::download_file(MODEL_URL, MODEL_PATH)?;

    Ok("Model downloaded".into())
}

fn do_load(_args: &str) -> Result<String, String> {
    if !model_exists() {
        return Err("Model file not found. Please download first.".into());
    }

    notify_info("Loading ONNX model...");

    // load-model takes a sandboxed relative path under the extension's
    // storage root — same rules as the storage interface. The actual
    // load happens on Bevy's IoTaskPool; this returns immediately with
    // a handle. `run()` later waits for the asset to finish loading.
    let session_id = dawai::extension::gpu::load_model(MODEL_PATH)
        .map_err(|e| format!("GPU load failed: {e}"))?;

    SESSION_ID.with(|cell| {
        *cell.borrow_mut() = Some(session_id.clone());
    });

    notify_success("Model loaded successfully");

    // Enable the separate button
    let _ = dawai::extension::panel_ui::update_widget(
        PANEL_ID,
        "separate",
        &serde_json::json!({"enabled": true}).to_string(),
    );
    let _ = dawai::extension::panel_ui::update_widget(
        PANEL_ID,
        "load_model",
        &serde_json::json!({"label": "Model Loaded", "enabled": false}).to_string(),
    );

    Ok(format!("ONNX model loaded, session={session_id}"))
}

fn do_separate(args: &str) -> Result<String, String> {
    let parsed: serde_json::Value = serde_json::from_str(args)
        .map_err(|e| format!("Invalid args JSON: {e}"))?;
    let track_id = parsed["track_id"]
        .as_str()
        .ok_or("Missing track_id in args")?;
    let start_time = parsed["start_time"].as_f64().unwrap_or(0.0);

    let session_id = SESSION_ID.with(|cell| cell.borrow().clone())
        .ok_or("Model not loaded. Call demucs.load first.")?;

    // Show progress
    let progress_id = dawai::extension::progress::show("Separating stems...", true)
        .unwrap_or_default();

    // 1. Read audio samples from the source track. v0.5 renamed the
    //    pre-v2 `get-node-audio` to `get-track-audio` — the underlying
    //    semantics are the same (channels + sample rate of the track's
    //    audio source).
    let (channels, sample_rate) = dawai::extension::project::get_track_audio(track_id)
        .map_err(|e| format!("Failed to read audio: {e}"))?;

    if channels.is_empty() {
        return Err("No audio channels returned".into());
    }

    let raw_left = &channels[0];
    let raw_right = if channels.len() > 1 { &channels[1] } else { raw_left };

    // 2. Resample to MODEL_SAMPLE_RATE so the network sees what it was
    //    trained on. Owned buffers either way — we hand slices to the
    //    chunker below.
    let _ = dawai::extension::progress::update(
        &progress_id,
        &format!("Resampling {sample_rate} Hz -> {MODEL_SAMPLE_RATE} Hz"),
        0.0,
    );
    let left_owned = resample(raw_left, sample_rate, MODEL_SAMPLE_RATE)?;
    let right_owned = resample(raw_right, sample_rate, MODEL_SAMPLE_RATE)?;
    let left: &[f32] = &left_owned;
    let right: &[f32] = &right_owned;

    // 3. Build chunks with overlap
    let chunks = dsp::build_chunks(left, right, dsp::TRAINING_LENGTH, dsp::OVERLAP);
    let num_chunks = chunks.len();
    let total_frames = left.len();

    // 4. Accumulate stems: [4 stems][2 channels][total_frames]
    let mut stem_accum = vec![vec![vec![0.0f32; total_frames]; 2]; 4];
    let mut weight_accum = vec![0.0f32; total_frames];

    for (i, chunk) in chunks.iter().enumerate() {
        let _ = dawai::extension::progress::update(
            &progress_id,
            &format!("Processing chunk {}/{}", i + 1, num_chunks),
            i as f32 / num_chunks as f32,
        );

        // Pad chunk to TRAINING_LENGTH
        let mut chunk_left = chunk.left.clone();
        let mut chunk_right = chunk.right.clone();
        chunk_left.resize(dsp::TRAINING_LENGTH, 0.0);
        chunk_right.resize(dsp::TRAINING_LENGTH, 0.0);

        // Build waveform tensor [1, 2, TRAINING_LENGTH]
        let mut waveform_data = Vec::with_capacity(2 * dsp::TRAINING_LENGTH);
        waveform_data.extend_from_slice(&chunk_left);
        waveform_data.extend_from_slice(&chunk_right);

        // Compute magnitude spectrogram [1, 4, 2048, T]
        let magspec = dsp::compute_magnitude_spectrogram(&chunk_left, &chunk_right);

        // Build input tensors JSON
        let mut inputs: HashMap<String, TensorJson> = HashMap::new();
        inputs.insert("mix".into(), TensorJson {
            shape: vec![1, 2, dsp::TRAINING_LENGTH],
            data: waveform_data,
        });
        inputs.insert("spectrogram".into(), TensorJson {
            shape: vec![1, 4, magspec.freq_bins, magspec.time_frames],
            data: magspec.data,
        });

        let inputs_json = serde_json::to_string(&inputs)
            .map_err(|e| format!("Serialize inputs: {e}"))?;

        // Run ONNX inference
        let outputs_json = dawai::extension::gpu::run(&session_id, &inputs_json)
            .map_err(|e| format!("GPU run failed: {e}"))?;

        let outputs: HashMap<String, TensorJson> = serde_json::from_str(&outputs_json)
            .map_err(|e| format!("Parse outputs: {e}"))?;

        // The ONNX model outputs stems as output_1: [1, 4, 2, TRAINING_LENGTH]
        let stems_tensor = outputs.get("output_1")
            .ok_or("Missing output_1 in ONNX output")?;

        // Overlap-add into accumulator
        let actual_len = chunk.actual_len;
        let start = chunk.start;
        for stem_idx in 0..4 {
            for ch in 0..2 {
                for j in 0..actual_len {
                    if start + j < total_frames {
                        let flat_idx = ((stem_idx * 2 + ch) * dsp::TRAINING_LENGTH) + j;
                        if flat_idx < stems_tensor.data.len() {
                            stem_accum[stem_idx][ch][start + j] += stems_tensor.data[flat_idx];
                        }
                    }
                }
            }
        }

        for j in 0..actual_len {
            if start + j < total_frames {
                weight_accum[start + j] += 1.0;
            }
        }
    }

    // 5. Normalize by overlap weights
    for stem_idx in 0..4 {
        for ch in 0..2 {
            for j in 0..total_frames {
                if weight_accum[j] > 0.0 {
                    stem_accum[stem_idx][ch][j] /= weight_accum[j];
                }
            }
        }
    }

    // 6. Write stem WAVs and add an audio clip to a fresh track for
    //    each stem. v0.5 dropped the pre-v2 `add-node("sampler", json)`
    //    verb — we now add a track (document-track::add-track), then
    //    attach an audio clip to it (document-clip::add-clip) that
    //    points at the WAV path.
    use crate::dawai::extension::document_clip::{add_clip, AddClipPayload};
    use crate::dawai::extension::document_track::{add_track, AddTrackPayload};
    use crate::dawai::extension::types::{
        AudioClip, ClipKind, ColorRgb, TimelinePlacement, TrackKind, TrackSource,
    };

    let stem_names = ["drums", "bass", "vocals", "other"];
    let stem_colors = [
        ColorRgb { r: 220, g: 80,  b: 80  },   // drums  — red
        ColorRgb { r: 80,  g: 140, b: 220 },   // bass   — blue
        ColorRgb { r: 220, g: 180, b: 80  },   // vocals — gold
        ColorRgb { r: 140, g: 200, b: 140 },   // other  — green
    ];

    let _ = dawai::extension::progress::update(&progress_id, "Writing stems...", 0.9);

    for (idx, name) in stem_names.iter().enumerate() {
        // Resample each stem back to the source rate so the WAV plays
        // in sync with the rest of the project. No-op when source was
        // already 44.1k (the resample() shortcut handles that).
        let left_ch = resample(&stem_accum[idx][0], MODEL_SAMPLE_RATE, sample_rate)?;
        let right_ch = resample(&stem_accum[idx][1], MODEL_SAMPLE_RATE, sample_rate)?;
        let stem_len = left_ch.len().min(right_ch.len());

        // Interleave stereo for WAV
        let mut interleaved = Vec::with_capacity(stem_len * 2);
        for j in 0..stem_len {
            interleaved.push(left_ch[j]);
            interleaved.push(right_ch[j]);
        }

        // Encode WAV to bytes
        let wav_rel = format!("stems/{name}.wav");
        let wav_bytes = encode_wav(&interleaved, sample_rate, 2)
            .map_err(|e| format!("WAV encode: {e}"))?;

        // Write via binary storage API. Returns the absolute path so we
        // can point the clip at the real on-disk location.
        let wav_abs = dawai::extension::storage::write_bytes(&wav_rel, &wav_bytes)
            .map_err(|e| format!("Write stem: {e}"))?;

        // Length in beats — approximate from sample count (assume the
        // host's current tempo is roughly stable; the duration probe
        // will reconcile when the clip lands). For first cut, leave
        // length_beats = 0.0 as the "needs-probe" sentinel.
        let new_track_id = add_track(&AddTrackPayload {
            name: format!("{} ({name})", "stem"),
            kind: TrackKind::Audio,
            source: TrackSource::None,
            clips: Vec::new(),
            effects: Vec::new(),
            sends: Vec::new(),
            volume: Some(1.0),
            pan: Some(0.0),
            muted: Some(false),
            soloed: Some(false),
            color: Some(stem_colors[idx]),
            index: None,
        })
        .map_err(|e| format!("Add stem track: {e}"))?;

        let _clip_id = add_clip(&AddClipPayload {
            track: new_track_id,
            name: name.to_string(),
            placement: TimelinePlacement {
                start_time,
                length_beats: 0.0,
            },
            kind: ClipKind::Audio(AudioClip {
                sample_path: wav_abs,
                playback_rate: 1.0,
                loop_enabled: false,
                loop_start: 0.0,
                loop_end: 0.0,
            }),
        })
        .map_err(|e| format!("Add stem clip: {e}"))?;
    }

    let _ = dawai::extension::progress::complete(&progress_id, "Stem separation complete");

    Ok(format!("Separated into {} stems", stem_names.len()))
}

/// Encode interleaved f32 samples to WAV bytes in memory
fn encode_wav(samples: &[f32], sample_rate: u32, channels: u16) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    let cursor = std::io::Cursor::new(&mut buf);
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::new(cursor, spec)
        .map_err(|e| format!("WAV writer: {e}"))?;
    for &s in samples {
        writer.write_sample(s).map_err(|e| format!("WAV write: {e}"))?;
    }
    writer.finalize().map_err(|e| format!("WAV finalize: {e}"))?;
    Ok(buf)
}

// =============================================================================
// Toast helpers
// =============================================================================

fn notify_info(msg: &str) {
    let _ = dawai::extension::ui::notify(msg, dawai::extension::ui::NotifyLevel::Info);
}

fn notify_success(msg: &str) {
    let _ = dawai::extension::ui::notify(msg, dawai::extension::ui::NotifyLevel::Success);
}

fn notify_error(msg: &str) {
    let _ = dawai::extension::ui::notify(msg, dawai::extension::ui::NotifyLevel::Error);
}

// =============================================================================
// Extension interface
// =============================================================================

impl Guest for DemucsExtension {
    fn init() -> Result<String, String> {
        Ok("demucs extension initialized".into())
    }

    fn activate() -> Result<String, String> {
        // Check if model file exists on disk — update button accordingly
        if model_exists() {
            let _ = dawai::extension::panel_ui::update_widget(
                PANEL_ID,
                "load_model",
                &serde_json::json!({"label": "Load Model", "enabled": true}).to_string(),
            );
            let _ = dawai::extension::panel_ui::update_widget(
                PANEL_ID,
                "model_progress",
                &serde_json::json!({"value": 1.0, "label": "Model downloaded"}).to_string(),
            );
        }

        Ok("demucs extension activated".into())
    }

    fn deactivate() -> Result<String, String> {
        SESSION_ID.with(|cell| {
            if let Some(id) = cell.borrow().as_ref() {
                let _ = dawai::extension::gpu::unload_model(id);
            }
            *cell.borrow_mut() = None;
        });
        Ok("demucs extension deactivated".into())
    }

    fn execute_command(command_id: String, args: String) -> Result<String, String> {
        match command_id.as_str() {
            "demucs.download" => do_download_model(),
            "demucs.load" => do_load(&args),
            "demucs.separate" => do_separate(&args),
            _ => Err(format!("Unknown command: {command_id}")),
        }
    }

    fn handle_event(event_type: String, data: String) -> Result<String, String> {
        if event_type != "ui" {
            return Ok("".into());
        }

        let event: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| format!("Parse UI event: {e}"))?;

        let widget_id = event["widget_id"].as_str().unwrap_or("");
        let event_type = event["event_type"].as_str().unwrap_or("");

        match (widget_id, event_type) {
            ("load_model", "clicked") => {
                // Disable button while working
                let _ = dawai::extension::panel_ui::update_widget(
                    PANEL_ID,
                    "load_model",
                    &serde_json::json!({"label": "Loading...", "enabled": false}).to_string(),
                );

                // Download if needed, then load
                if !model_exists() {
                    let _ = dawai::extension::panel_ui::update_widget(
                        PANEL_ID,
                        "model_progress",
                        &serde_json::json!({"value": 0.1, "label": "Downloading..."}).to_string(),
                    );

                    match do_download_model() {
                        Ok(_) => {}
                        Err(e) => {
                            let _ = dawai::extension::panel_ui::update_widget(
                                PANEL_ID,
                                "load_model",
                                &serde_json::json!({"label": "Download & Load Model", "enabled": true}).to_string(),
                            );
                            return Err(e);
                        }
                    }
                }

                // Load the model into GPU runtime
                match do_load("{}") {
                    Ok(msg) => {
                        notify_success("Model loaded and ready");
                        Ok(msg)
                    }
                    Err(e) => {
                        let _ = dawai::extension::panel_ui::update_widget(
                            PANEL_ID,
                            "load_model",
                            &serde_json::json!({"label": "Retry Load", "enabled": true}).to_string(),
                        );
                        notify_error(&format!("Load failed: {e}"));
                        Err(e)
                    }
                }
            }

            ("separate", "clicked") => {
                // Get the currently selected track (v0.5 renamed
                // `get-selected-node-id` to `get-selected-track-id`).
                let track_id = match dawai::extension::project::get_selected_track_id() {
                    Ok(Some(id)) => id,
                    Ok(None) => {
                        notify_error("No track selected. Select an audio track first.");
                        return Ok("".into());
                    }
                    Err(e) => {
                        notify_error(&format!("Failed to get selection: {e}"));
                        return Err(e);
                    }
                };

                notify_info(&format!("Separating track {track_id}..."));

                let args = serde_json::json!({
                    "track_id": track_id,
                    "start_time": 0.0,
                });
                match do_separate(&args.to_string()) {
                    Ok(msg) => {
                        notify_success(&msg);
                        Ok(msg)
                    }
                    Err(e) => {
                        notify_error(&format!("Separation failed: {e}"));
                        Err(e)
                    }
                }
            }

            _ => Ok("".into()),
        }
    }
}

export!(DemucsExtension);
