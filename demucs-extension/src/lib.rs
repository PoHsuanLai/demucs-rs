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

/// Panel ID (must match extension.toml)
const PANEL_ID: &str = "dawai.demucs";

/// Stored ONNX session ID after loading
thread_local! {
    static SESSION_ID: RefCell<Option<String>> = const { RefCell::new(None) };
}

// =============================================================================
// Commands
// =============================================================================

/// Get the on-disk path for the ONNX model.
fn model_path() -> Result<String, String> {
    let storage_path = dawai::extension::storage::get_storage_path()
        .unwrap_or_else(|_| ".".into());
    Ok(format!("{storage_path}/htdemucs.onnx"))
}

/// Check if the model file exists on disk.
fn model_exists() -> bool {
    let path = match model_path() {
        Ok(p) => p,
        Err(_) => return false,
    };
    dawai::extension::storage::file_exists(&path)
        .unwrap_or_else(|_| "false".into()) == "true"
}

fn do_download_model() -> Result<String, String> {
    if model_exists() {
        return Ok("Model already downloaded".into());
    }

    notify_info("Downloading HTDemucs model (~290 MB)...");

    let path = model_path()?;

    // Download via host-side streaming download
    dawai::extension::storage::download_file(MODEL_URL, &path)?;

    Ok("Model downloaded".into())
}

fn do_load(_args: &str) -> Result<String, String> {
    let path = model_path()?;

    if !model_exists() {
        return Err("Model file not found. Please download first.".into());
    }

    notify_info("Loading ONNX model...");

    // Load directly from file path (no 290MB IPC transfer)
    let session_id = dawai::extension::gpu::load_model_from_path("htdemucs", &path)
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
    let node_id = parsed["node_id"]
        .as_str()
        .ok_or("Missing node_id in args")?;
    let start_time = parsed["start_time"].as_f64().unwrap_or(0.0);

    let session_id = SESSION_ID.with(|cell| cell.borrow().clone())
        .ok_or("Model not loaded. Call demucs.load first.")?;

    // Show progress
    let progress_id = dawai::extension::progress::show("Separating stems...", true)
        .unwrap_or_default();

    // 1. Read audio samples from the sampler node
    let (channels, sample_rate) = dawai::extension::project::get_node_audio(node_id)
        .map_err(|e| format!("Failed to read audio: {e}"))?;

    if channels.is_empty() {
        return Err("No audio channels returned".into());
    }

    let left = &channels[0];
    let right = if channels.len() > 1 { &channels[1] } else { left };

    // 2. Resample to 44100 if needed
    // TODO: use rubato for resampling. For now, assume 44100.
    let _ = sample_rate;

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

    // 6. Write stem WAVs and emit document changes
    let stem_names = ["drums", "bass", "vocals", "other"];
    let storage_path = dawai::extension::storage::get_storage_path()
        .unwrap_or_else(|_| ".".into());
    let stems_dir = format!("{storage_path}/stems");

    let _ = dawai::extension::progress::update(&progress_id, "Writing stems...", 0.9);

    for (idx, name) in stem_names.iter().enumerate() {
        // Interleave stereo for WAV
        let left_ch = &stem_accum[idx][0];
        let right_ch = &stem_accum[idx][1];
        let mut interleaved = Vec::with_capacity(total_frames * 2);
        for j in 0..total_frames {
            interleaved.push(left_ch[j]);
            interleaved.push(right_ch[j]);
        }

        // Encode WAV to bytes
        let wav_path = format!("{stems_dir}/{name}.wav");
        let wav_bytes = encode_wav(&interleaved, 44100, 2)
            .map_err(|e| format!("WAV encode: {e}"))?;

        // Write via binary storage API
        dawai::extension::storage::write_bytes(&wav_path, &wav_bytes)
            .map_err(|e| format!("Write stem: {e}"))?;

        // Create sampler node pointing at the stem WAV
        let params = serde_json::json!({
            "sample_path": wav_path,
            "start_time": start_time,
        });
        let _node_id = dawai::extension::project::add_node("sampler", &params.to_string())
            .map_err(|e| format!("Add sampler node: {e}"))?;
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
                // Get the currently selected node
                let node_id = dawai::extension::project::get_selected_node_id()
                    .map_err(|e| format!("Failed to get selection: {e}"))?
                    .ok_or("No node selected. Select a sampler node first.")?;

                let args = serde_json::json!({
                    "node_id": node_id,
                    "start_time": 0.0,
                });
                do_separate(&args.to_string())
            }

            _ => Ok("".into()),
        }
    }
}

export!(DemucsExtension);
