wit_bindgen::generate!({
    world: "dawai-audio-extension",
    path: "wit/world.wit",
});

use std::cell::RefCell;

use burn::backend::NdArray;
use demucs_core::listener::{ForwardEvent, ForwardListener};
use demucs_core::{Demucs, ModelOptions};

use exports::dawai::extension::extension::Guest;
use exports::dawai::extension::audio_extension::Guest as AudioGuest;
use dawai::extension::audio::{AudioBuffer, SeparationResult, Stem, StemId};

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

fn map_stem_id(id: demucs_core::model::metadata::StemId) -> StemId {
    match id {
        demucs_core::model::metadata::StemId::Drums => StemId::Drums,
        demucs_core::model::metadata::StemId::Bass => StemId::Bass,
        demucs_core::model::metadata::StemId::Other => StemId::Other,
        demucs_core::model::metadata::StemId::Vocals => StemId::Vocals,
        demucs_core::model::metadata::StemId::Guitar => StemId::Guitar,
        demucs_core::model::metadata::StemId::Piano => StemId::Piano,
    }
}

// Base extension interface
impl Guest for DemucsExtension {
    fn init() -> Result<String, String> {
        Ok("demucs extension initialized".into())
    }

    fn activate() -> Result<String, String> {
        // Register a panel so the extension shows in the toolbar
        let icon_svg = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>"#;
        let _ = dawai::extension::panel_ui::register_panel(
            "dawai.demucs",
            "Demucs",
            icon_svg,
            300.0,
            400.0,
        );
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
            _ => Err(format!("Unknown command: {command_id}")),
        }
    }

    fn handle_event(_event_type: String, _data: String) -> Result<String, String> {
        Ok("".into())
    }
}

// Typed audio extension interface — no JSON serialization
impl AudioGuest for DemucsExtension {
    fn separate_stems(input: AudioBuffer) -> Result<SeparationResult, String> {
        let left = input.channels.first()
            .ok_or("Input must have at least one channel")?;
        let right = input.channels.get(1).unwrap_or(left);

        MODEL.with(|cell| {
            let model_ref = cell.borrow();
            let model = model_ref
                .as_ref()
                .ok_or("Model not loaded. Call demucs.load first.")?;

            let mut reporter = ProgressReporter::new();
            reporter.start("Separating stems...");

            let stems_result = pollster::block_on(model.separate_with_listener(
                left,
                right,
                input.sample_rate,
                &mut reporter,
            ));

            match stems_result {
                Ok(stems) => {
                    reporter.finish("Stem separation complete");
                    Ok(SeparationResult {
                        stems: stems
                            .into_iter()
                            .map(|s| Stem {
                                id: map_stem_id(s.id),
                                audio: AudioBuffer {
                                    channels: vec![s.left, s.right],
                                    sample_rate: input.sample_rate,
                                },
                            })
                            .collect(),
                    })
                }
                Err(e) => {
                    reporter.fail_progress(&format!("Separation failed: {e}"));
                    Err(format!("Separation failed: {e}"))
                }
            }
        })
    }
}

export!(DemucsExtension);
