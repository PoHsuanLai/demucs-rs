wit_bindgen::generate!({
    world: "dawai-extension",
    path: "wit/world.wit",
});

mod dsp;

use std::cell::RefCell;
use std::collections::HashMap;

use burn::backend::NdArray;
use demucs_core::listener::{ForwardEvent, ForwardListener};
use demucs_core::{Demucs, ModelOptions};
use serde::{Deserialize, Serialize};

use exports::dawai::extension::extension::Guest;

type B = NdArray;

struct DemucsExtension;

/// Tensor JSON format matching dawai-gpu's TensorJson
#[derive(Debug, Serialize, Deserialize)]
struct TensorJson {
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// Which inference backend is loaded
enum InferenceBackend {
    /// ONNX model loaded on host GPU via gpu.run()
    Onnx { session_id: String },
    /// Burn model loaded in-process on CPU
    Burn { model: Demucs<B> },
}

thread_local! {
    static BACKEND: RefCell<Option<InferenceBackend>> = const { RefCell::new(None) };
}

// =============================================================================
// Progress reporting
// =============================================================================

struct ProgressReporter {
    handle_id: Option<String>,
}

impl ProgressReporter {
    fn new() -> Self {
        Self { handle_id: None }
    }

    fn start(&mut self, title: &str) {
        if let Ok(id) = dawai::extension::progress::show(title, true) {
            self.handle_id = Some(id);
        }
    }

    fn update(&mut self, message: &str, pct: f32) {
        if let Some(ref id) = self.handle_id {
            let _ = dawai::extension::progress::update(id, message, pct);
        }
    }

    fn finish(&mut self, message: &str) {
        if let Some(id) = self.handle_id.take() {
            let _ = dawai::extension::progress::complete(&id, message);
        }
    }

    fn fail_progress(&mut self, message: &str) {
        if let Some(id) = self.handle_id.take() {
            let _ = dawai::extension::progress::fail(&id, message);
        }
    }
}

impl ForwardListener for ProgressReporter {
    fn on_event(&mut self, event: ForwardEvent) {
        let Some(ref id) = self.handle_id else { return };

        match event {
            ForwardEvent::ChunkStarted { index, total } => {
                let pct = index as f32 / total as f32;
                let _ = dawai::extension::progress::update(
                    id,
                    &format!("Processing chunk {}/{}", index + 1, total),
                    pct,
                );
            }
            ForwardEvent::ChunkDone { index, total } => {
                let pct = (index + 1) as f32 / total as f32;
                let _ = dawai::extension::progress::update(
                    id,
                    &format!("Chunk {}/{} done", index + 1, total),
                    pct,
                );
            }
            ForwardEvent::StemDone { index, total } => {
                let _ = dawai::extension::progress::update(
                    id,
                    &format!("Extracted stem {}/{}", index + 1, total),
                    0.9,
                );
            }
            _ => {}
        }
    }

    fn wants_stats(&self) -> bool {
        false
    }
}

// =============================================================================
// Commands
// =============================================================================

fn do_load(args: &str) -> Result<String, String> {
    let parsed: serde_json::Value = serde_json::from_str(args).unwrap_or_default();
    let format = parsed["format"].as_str().unwrap_or("safetensors");

    let bytes = dawai::extension::storage::get("model_bytes")
        .map_err(|e| format!("Failed to get model bytes: {e}"))?;

    if bytes.is_empty() {
        return Err("No model bytes in storage. Download the model first.".into());
    }

    match format {
        "onnx" => {
            // Load via host GPU API
            let session_id = dawai::extension::gpu::load_model(
                "htdemucs",
                "onnx",
                bytes.as_bytes(),
            )
            .map_err(|e| format!("GPU load failed: {e}"))?;

            BACKEND.with(|cell| {
                *cell.borrow_mut() = Some(InferenceBackend::Onnx { session_id: session_id.clone() });
            });

            Ok(format!("ONNX model loaded on GPU, session={session_id}"))
        }
        "safetensors" | _ => {
            // Burn CPU fallback
            let device = burn::backend::ndarray::NdArrayDevice::Cpu;
            let model = Demucs::<B>::from_bytes(ModelOptions::FourStem, bytes.as_bytes(), device)
                .map_err(|e| format!("Failed to load model: {e}"))?;

            BACKEND.with(|cell| {
                *cell.borrow_mut() = Some(InferenceBackend::Burn { model });
            });

            Ok("Burn model loaded on CPU".into())
        }
    }
}

fn do_separate(args: &str) -> Result<String, String> {
    let parsed: serde_json::Value = serde_json::from_str(args)
        .map_err(|e| format!("Invalid args JSON: {e}"))?;
    let sample_path = parsed["sample_path"]
        .as_str()
        .ok_or("Missing sample_path in args")?;
    let start_time = parsed["start_time"].as_f64().unwrap_or(0.0);

    // Read audio file via storage
    let audio_data = dawai::extension::storage::read_file(sample_path)
        .map_err(|e| format!("Failed to read audio: {e}"))?;

    // Parse WAV (simple: assume f32 PCM stereo for now)
    // TODO: proper WAV parsing
    let _ = &audio_data;

    BACKEND.with(|cell| {
        let backend_ref = cell.borrow();
        let backend = backend_ref
            .as_ref()
            .ok_or("Model not loaded. Call demucs.load first.")?;

        match backend {
            InferenceBackend::Onnx { session_id } => {
                separate_onnx(session_id, sample_path, start_time)
            }
            InferenceBackend::Burn { model } => {
                separate_burn(model, sample_path, start_time)
            }
        }
    })
}

fn separate_onnx(
    session_id: &str,
    _sample_path: &str,
    _start_time: f64,
) -> Result<String, String> {
    // TODO: Full ONNX pipeline:
    // 1. Read and decode audio file to f32 PCM
    // 2. Deinterleave to left/right channels
    // 3. Resample to 44100 if needed
    // 4. Build chunks with overlap
    // 5. For each chunk:
    //    a. Pad to TRAINING_LENGTH
    //    b. Build waveform tensor [1, 2, TRAINING_LENGTH]
    //    c. Compute magnitude spectrogram [1, 4, 2048, T]
    //    d. Call gpu.run(session_id, { "waveform": ..., "spectrogram": ... })
    //    e. Parse output stems [1, 4, 2, TRAINING_LENGTH]
    // 6. Overlap-add all chunks
    // 7. Write stem WAVs via storage
    // 8. Emit document changes (add sampler + volume nodes)

    let _ = session_id;
    Err("ONNX separation pipeline not yet implemented".into())
}

fn separate_burn(
    model: &Demucs<B>,
    _sample_path: &str,
    _start_time: f64,
) -> Result<String, String> {
    // TODO: Full Burn pipeline:
    // 1. Read and decode audio file
    // 2. Run model.separate_with_listener()
    // 3. Write stem WAVs via storage
    // 4. Emit document changes

    let _ = model;
    Err("Burn separation pipeline not yet implemented".into())
}

// =============================================================================
// Extension interface
// =============================================================================

impl Guest for DemucsExtension {
    fn init() -> Result<String, String> {
        Ok("demucs extension initialized".into())
    }

    fn activate() -> Result<String, String> {
        Ok("demucs extension activated".into())
    }

    fn deactivate() -> Result<String, String> {
        BACKEND.with(|cell| {
            if let Some(InferenceBackend::Onnx { session_id }) = cell.borrow().as_ref() {
                let _ = dawai::extension::gpu::unload_model(session_id);
            }
            *cell.borrow_mut() = None;
        });
        Ok("demucs extension deactivated".into())
    }

    fn execute_command(command_id: String, args: String) -> Result<String, String> {
        match command_id.as_str() {
            "demucs.load" => do_load(&args),
            "demucs.separate" => do_separate(&args),
            _ => Err(format!("Unknown command: {command_id}")),
        }
    }

    fn handle_event(_event_type: String, _data: String) -> Result<String, String> {
        Ok("".into())
    }
}

export!(DemucsExtension);
