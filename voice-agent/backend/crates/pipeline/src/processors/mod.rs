//! Frame processors for the pipeline
//!
//! This module contains FrameProcessor implementations for:
//! - SentenceDetector: Detects sentence boundaries from LLM chunks
//! - TtsProcessor: Converts sentences to audio via streaming TTS
//! - InterruptHandler: Handles barge-in with configurable modes
//! - ProcessorChain: Channel-based chain connecting processors

mod chain;
mod interrupt_handler;
mod sentence_detector;
mod tts_processor;

pub use chain::{ProcessorChain, ProcessorChainBuilder};
// P2-2 FIX: Export generic processors for external use
pub use chain::{FilterProcessor, MapProcessor, PassthroughProcessor};
pub use interrupt_handler::{InterruptHandler, InterruptHandlerConfig, InterruptMode};
pub use sentence_detector::{SentenceDetector, SentenceDetectorConfig};
pub use tts_processor::{TtsProcessor, TtsProcessorConfig};
