//! Demucs source-separation extension — runs HTDemucs through the
//! host's burn runtime via the WIT `burn` interface.
//!
//! Inference path:
//!   guest                              host
//!   -----                              ----
//!   demucs_core::Demucs<Backend>::separate(left, right, sr)
//!     └─ Tensor ops emit `OperationIr` blobs
//!     └─ `WitClient` (router transport) calls host::register_op(blob)
//!     └─ `read_tensor_async` pulls the final stems back as raw bytes
//!   demucs_core stitches the stems and the extension writes WAVs +
//!   spawns one track per stem in the host project.

wit_bindgen::generate!({
    world: "dawai-extension",
    path: "wit/world.wit",
});

mod wit_channel;

use std::cell::RefCell;

use exports::dawai::extension::extension::Guest;

use burn::tensor::{Tensor, TensorData};
use burn_std::future::block_on;
use demucs_core::{Demucs, ModelOptions};

use crate::wit_channel::{Backend, WitDevice};

struct DemucsExtension;

/// HuggingFace URL for HTDemucs 4-stem safetensors. demucs-core
/// consumes the canonical Meta safetensors format directly — no
/// extra conversion step on the extension side.
///
/// `ai.adamprouduck.de` is a community mirror; substitute with your
/// preferred host. The 4-stem weights are ~80 MB.
const MODEL_URL: &str =
    "https://huggingface.co/smank/htdemucs-safetensors/resolve/main/htdemucs.safetensors";

/// HTDemucs was trained on 44.1 kHz audio. demucs-core takes the
/// source rate and resamples internally — we report it as a constant
/// only so the panel can display "44.1 kHz expected" if we ever wire
/// up a label.
#[allow(dead_code)]
const MODEL_SAMPLE_RATE: u32 = 44_100;

/// Relative path (under the extension's sandboxed storage root) where
/// the safetensors model is downloaded to and loaded from.
const MODEL_PATH: &str = "htdemucs.safetensors";

/// Panel ID (must match extension.toml)
const PANEL_ID: &str = "dawai.demucs";

// Cached model bytes after a successful `do_load`. We re-decode the
// model into the burn `Module` on each separate call — the decoded
// `Demucs` holds device-bound tensors so it can't outlive a single
// inference scope cleanly, but the raw bytes are cheap to keep.
thread_local! {
    static MODEL_BYTES: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };
}

// =============================================================================
// Commands
// =============================================================================

fn model_exists() -> bool {
    dawai::extension::storage::file_exists(MODEL_PATH)
        .unwrap_or_else(|_| "false".into())
        == "true"
}

fn do_download_model() -> Result<String, String> {
    if model_exists() {
        return Ok("Model already downloaded".into());
    }

    notify_info("Downloading HTDemucs model...");

    dawai::extension::storage::download_file(MODEL_URL, MODEL_PATH)?;

    Ok("Model downloaded".into())
}

fn do_load(_args: &str) -> Result<String, String> {
    if !model_exists() {
        return Err("Model file not found. Please download first.".into());
    }

    notify_info("Loading model bytes...");

    let bytes = dawai::extension::storage::read_bytes(MODEL_PATH)
        .map_err(|e| format!("Failed to read model bytes: {e}"))?;

    MODEL_BYTES.with(|cell| *cell.borrow_mut() = Some(bytes));

    notify_success("Model loaded successfully");

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

    Ok("Model bytes cached in extension".into())
}

fn do_separate(args: &str) -> Result<String, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(args).map_err(|e| format!("Invalid args JSON: {e}"))?;
    let track_id = parsed["track_id"]
        .as_str()
        .ok_or("Missing track_id in args")?;
    let start_time = parsed["start_time"].as_f64().unwrap_or(0.0);

    let bytes = MODEL_BYTES
        .with(|cell| cell.borrow().clone())
        .ok_or("Model not loaded. Click 'Load Model' first.")?;

    let progress_id = dawai::extension::progress::show("Separating stems...", true)
        .unwrap_or_default();

    // 1. Read audio samples from the source track.
    let (channels, sample_rate) = dawai::extension::project::get_track_audio(track_id)
        .map_err(|e| format!("Failed to read audio: {e}"))?;
    if channels.is_empty() {
        return Err("No audio channels returned".into());
    }
    let left = channels[0].clone();
    let right = if channels.len() > 1 {
        channels[1].clone()
    } else {
        left.clone()
    };

    // 2. Instantiate the Demucs model bound to the WitChannel router
    //    backend. The `from_bytes` parser de-serializes safetensors
    //    and copies parameters into router-allocated tensors —
    //    every parameter ends up as a `register-tensor-data` host
    //    call. This is one-shot per separation call; a follow-up
    //    can hoist this onto a per-app singleton.
    let _ = dawai::extension::progress::update(&progress_id, "Decoding model weights...", 0.05);
    let device = WitDevice::default();
    let demucs = Demucs::<Backend>::from_bytes(ModelOptions::FourStem, &bytes, device.clone())
        .map_err(|e| format!("Decode model: {e}"))?;

    // 3. Run separation. demucs-core handles resampling, chunking,
    //    STFT, and overlap-add internally — every tensor op flows
    //    through the router → WIT → host runner.
    let _ = dawai::extension::progress::update(&progress_id, "Running inference...", 0.15);
    let stems = block_on(demucs.separate(&left, &right, sample_rate))
        .map_err(|e| format!("Separation: {e}"))?;

    // 4. Write WAVs + add a track per stem.
    let _ = dawai::extension::progress::update(&progress_id, "Writing stems...", 0.9);
    use crate::dawai::extension::document_clip::{add_clip, AddClipPayload};
    use crate::dawai::extension::document_track::{add_track, AddTrackPayload};
    use crate::dawai::extension::types::{
        AudioClip, ClipKind, ColorRgb, TimelinePlacement, TrackKind, TrackSource,
    };

    let stem_colors: &[(&str, ColorRgb)] = &[
        ("drums",  ColorRgb { r: 220, g: 80,  b: 80  }),
        ("bass",   ColorRgb { r: 80,  g: 140, b: 220 }),
        ("vocals", ColorRgb { r: 220, g: 180, b: 80  }),
        ("other",  ColorRgb { r: 140, g: 200, b: 140 }),
    ];

    let n_stems = stems.len();
    for (idx, stem) in stems.iter().enumerate() {
        let (label, color) = stem_colors
            .get(idx)
            .copied()
            .unwrap_or(("stem", ColorRgb { r: 180, g: 180, b: 180 }));

        let stem_len = stem.left.len().min(stem.right.len());
        let mut interleaved = Vec::with_capacity(stem_len * 2);
        for j in 0..stem_len {
            interleaved.push(stem.left[j]);
            interleaved.push(stem.right[j]);
        }

        let wav_rel = format!("stems/{label}.wav");
        let wav_bytes = encode_wav(&interleaved, sample_rate, 2)
            .map_err(|e| format!("WAV encode: {e}"))?;

        let wav_abs = dawai::extension::storage::write_bytes(&wav_rel, &wav_bytes)
            .map_err(|e| format!("Write stem: {e}"))?;

        let new_track_id = add_track(&AddTrackPayload {
            name: format!("stem ({label})"),
            kind: TrackKind::Audio,
            source: TrackSource::None,
            clips: Vec::new(),
            effects: Vec::new(),
            sends: Vec::new(),
            volume: Some(1.0),
            pan: Some(0.0),
            muted: Some(false),
            soloed: Some(false),
            color: Some(color),
            index: None,
        })
        .map_err(|e| format!("Add stem track: {e}"))?;

        let _clip_id = add_clip(&AddClipPayload {
            track: new_track_id,
            name: label.to_string(),
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

    Ok(format!("Separated into {n_stems} stems"))
}

/// Minimal end-to-end exercise of the WIT `burn` interface: builds a
/// 3-element f32 tensor through the router, runs one op (`add_scalar`),
/// reads it back, and returns the result as JSON.
///
/// Used by the testkit smoke test to verify that
/// `register-tensor-data` / `register-op` / `read-tensor` all
/// round-trip through the host runner without involving the rest of
/// demucs. If this fails, no point trying the full separation path.
fn do_burn_smoke(_args: &str) -> Result<String, String> {
    let device = WitDevice::default();
    let input = TensorData::new(vec![1.0f32, 2.0, 3.0], [3]);
    let t: Tensor<Backend, 1> = Tensor::from_data(input, &device);
    let out = t.add_scalar(1.0);
    let data = block_on(out.into_data_async())
        .map_err(|e| format!("read tensor: {e}"))?;
    let bytes = data.into_bytes().to_vec();
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    serde_json::to_string(&floats).map_err(|e| format!("serialize burn_smoke: {e}"))
}

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
        writer
            .write_sample(s)
            .map_err(|e| format!("WAV write: {e}"))?;
    }
    writer
        .finalize()
        .map_err(|e| format!("WAV finalize: {e}"))?;
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
        MODEL_BYTES.with(|cell| *cell.borrow_mut() = None);
        Ok("demucs extension deactivated".into())
    }

    fn execute_command(command_id: String, args: String) -> Result<String, String> {
        match command_id.as_str() {
            "demucs.download" => do_download_model(),
            "demucs.load" => do_load(&args),
            "demucs.separate" => do_separate(&args),
            "demucs.burn_smoke" => do_burn_smoke(&args),
            _ => Err(format!("Unknown command: {command_id}")),
        }
    }

    fn handle_event(event_type: String, data: String) -> Result<String, String> {
        if event_type != "ui" {
            return Ok("".into());
        }

        let event: serde_json::Value =
            serde_json::from_str(&data).map_err(|e| format!("Parse UI event: {e}"))?;

        let widget_id = event["widget_id"].as_str().unwrap_or("");
        let event_type = event["event_type"].as_str().unwrap_or("");

        match (widget_id, event_type) {
            ("load_model", "clicked") => {
                let _ = dawai::extension::panel_ui::update_widget(
                    PANEL_ID,
                    "load_model",
                    &serde_json::json!({"label": "Loading...", "enabled": false}).to_string(),
                );

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
                                &serde_json::json!({"label": "Download & Load Model", "enabled": true})
                                    .to_string(),
                            );
                            return Err(e);
                        }
                    }
                }

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
