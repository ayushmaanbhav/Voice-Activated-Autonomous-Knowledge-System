//! Hybrid Turn Detection
//!
//! Combines VAD-based silence detection with semantic completeness analysis.
//! Architecture: Silence detector + Lightweight transformer classifier

mod semantic;
mod hybrid;

pub use semantic::SemanticTurnDetector;
pub use hybrid::{HybridTurnDetector, TurnDetectionConfig, TurnState, TurnDetectionResult};
