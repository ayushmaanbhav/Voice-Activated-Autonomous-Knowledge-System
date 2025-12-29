//! Frame processors for the pipeline
//!
//! This module contains FrameProcessor implementations for:
//! - SentenceDetector: Detects sentence boundaries from LLM chunks
//! - TtsProcessor: Converts sentences to audio via streaming TTS
//! - InterruptHandler: Handles barge-in with configurable modes
//! - ProcessorChain: Channel-based chain connecting processors

mod sentence_detector;
mod tts_processor;
mod interrupt_handler;
mod chain;

pub use sentence_detector::{SentenceDetector, SentenceDetectorConfig};
pub use tts_processor::{TtsProcessor, TtsProcessorConfig};
pub use interrupt_handler::{InterruptHandler, InterruptMode, InterruptHandlerConfig};
pub use chain::{ProcessorChain, ProcessorChainBuilder};
