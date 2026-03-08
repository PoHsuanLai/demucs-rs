wit_bindgen::generate!({
    world: "dawai-extension",
    path: "wit/world.wit",
});

use std::cell::RefCell;

use burn::backend::NdArray;
use demucs_core::listener::{ForwardEvent, ForwardListener};
use demucs_core::{Demucs, ModelOptions};

use exports::dawai::extension::extension::Guest;

type B = NdArray;

struct DemucsExtension;

thread_local! {
    static MODEL: RefCell<Option<Demucs<B>>> = const { RefCell::new(None) };
}

/// ForwardListener that reports progress via the WIT progress API.
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

#[derive(serde::Deserialize)]
struct SeparateArgs {
    left: Vec<f32>,
    right: Vec<f32>,
    sample_rate: u32,
}

#[derive(serde::Serialize)]
struct StemOutput {
    id: String,
    left: Vec<f32>,
    right: Vec<f32>,
}

#[derive(serde::Serialize)]
struct SeparateResult {
    stems: Vec<StemOutput>,
    sample_rate: u32,
}

fn do_separate(args: &str) -> Result<String, String> {
    let args: SeparateArgs =
        serde_json::from_str(args).map_err(|e| format!("Invalid args: {e}"))?;

    MODEL.with(|cell| {
        let model_ref = cell.borrow();
        let model = model_ref
            .as_ref()
            .ok_or("Model not loaded. Call demucs.load first.")?;

        let mut reporter = ProgressReporter::new();
        reporter.start("Separating stems...");

        // Run inference (blocking — ndarray backend is synchronous)
        let stems_result = pollster::block_on(model.separate_with_listener(
            &args.left,
            &args.right,
            args.sample_rate,
            &mut reporter,
        ));

        match stems_result {
            Ok(stems) => {
                reporter.finish("Stem separation complete");
                let result = SeparateResult {
                    stems: stems
                        .into_iter()
                        .map(|s| StemOutput {
                            id: s.id.as_str().to_string(),
                            left: s.left,
                            right: s.right,
                        })
                        .collect(),
                    sample_rate: args.sample_rate,
                };
                serde_json::to_string(&result).map_err(|e| format!("Serialize error: {e}"))
            }
            Err(e) => {
                reporter.fail_progress(&format!("Separation failed: {e}"));
                Err(format!("Separation failed: {e}"))
            }
        }
    })
}

fn do_load(_args: &str) -> Result<String, String> {
    let bytes = dawai::extension::storage::get("model_bytes")
        .map_err(|e| format!("Failed to get model bytes: {e}"))?;

    if bytes.is_empty() {
        return Err("No model bytes in storage. Download the model first.".into());
    }

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let model = Demucs::<B>::from_bytes(ModelOptions::FourStem, bytes.as_bytes(), device)
        .map_err(|e| format!("Failed to load model: {e}"))?;

    MODEL.with(|cell| {
        *cell.borrow_mut() = Some(model);
    });

    Ok("Model loaded".into())
}

impl Guest for DemucsExtension {
    fn init() -> Result<String, String> {
        Ok("demucs extension initialized".into())
    }

    fn activate() -> Result<String, String> {
        Ok("demucs extension activated".into())
    }

    fn deactivate() -> Result<String, String> {
        MODEL.with(|cell| {
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
