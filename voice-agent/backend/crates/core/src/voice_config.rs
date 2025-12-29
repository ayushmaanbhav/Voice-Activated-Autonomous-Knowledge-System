//! Voice configuration types for TTS

use serde::{Deserialize, Serialize};
use crate::language::Language;

/// Voice configuration for TTS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// Target language
    pub language: Language,
    /// Voice identifier
    pub voice_id: String,
    /// Speech speed (0.5 - 2.0, default 1.0)
    #[serde(default = "default_speed")]
    pub speed: f32,
    /// Voice pitch adjustment (-1.0 to 1.0, default 0.0)
    #[serde(default)]
    pub pitch: f32,
    /// Volume adjustment (0.0 - 2.0, default 1.0)
    #[serde(default = "default_volume")]
    pub volume: f32,
}

fn default_speed() -> f32 {
    1.0
}

fn default_volume() -> f32 {
    1.0
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            language: Language::Hindi,
            voice_id: "default".to_string(),
            speed: 1.0,
            pitch: 0.0,
            volume: 1.0,
        }
    }
}

impl VoiceConfig {
    /// Create a new voice config with the given language
    pub fn new(language: Language) -> Self {
        Self {
            language,
            ..Default::default()
        }
    }

    /// Set the voice ID
    pub fn with_voice_id(mut self, voice_id: impl Into<String>) -> Self {
        self.voice_id = voice_id.into();
        self
    }

    /// Set the speech speed
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed.clamp(0.5, 2.0);
        self
    }

    /// Set the pitch adjustment
    pub fn with_pitch(mut self, pitch: f32) -> Self {
        self.pitch = pitch.clamp(-1.0, 1.0);
        self
    }
}

/// Voice information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    /// Voice identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Supported language
    pub language: Language,
    /// Gender (optional)
    #[serde(default)]
    pub gender: Option<VoiceGender>,
    /// Sample audio URL (optional)
    #[serde(default)]
    pub sample_url: Option<String>,
    /// Description of the voice
    #[serde(default)]
    pub description: Option<String>,
}

/// Voice gender
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceGender {
    Male,
    Female,
    Neutral,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_config_default() {
        let config = VoiceConfig::default();
        assert_eq!(config.language, Language::Hindi);
        assert_eq!(config.speed, 1.0);
        assert_eq!(config.pitch, 0.0);
    }

    #[test]
    fn test_voice_config_builder() {
        let config = VoiceConfig::new(Language::Tamil)
            .with_voice_id("tamil_female_1")
            .with_speed(1.2)
            .with_pitch(0.1);

        assert_eq!(config.language, Language::Tamil);
        assert_eq!(config.voice_id, "tamil_female_1");
        assert_eq!(config.speed, 1.2);
        assert_eq!(config.pitch, 0.1);
    }

    #[test]
    fn test_speed_clamping() {
        let config = VoiceConfig::default().with_speed(5.0);
        assert_eq!(config.speed, 2.0);

        let config = VoiceConfig::default().with_speed(0.1);
        assert_eq!(config.speed, 0.5);
    }
}
