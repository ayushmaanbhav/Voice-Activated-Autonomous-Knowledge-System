//! Agent persona definitions
//!
//! Personas define the agent's communication style:
//! - Tone (formal/casual/professional)
//! - Warmth level (empathy, friendliness)
//! - Language style (simple/sophisticated)
//! - Response patterns

use serde::{Deserialize, Serialize};
use crate::CustomerSegment;

/// Communication tone
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tone {
    /// Highly formal, respectful (e.g., "Respected Sir/Madam")
    Formal,
    /// Professional but warm
    Professional,
    /// Friendly and approachable
    Friendly,
    /// Casual, conversational
    Casual,
}

impl Tone {
    /// Get greeting prefix for this tone
    pub fn greeting_prefix(&self) -> &'static str {
        match self {
            Tone::Formal => "Respected",
            Tone::Professional => "Dear",
            Tone::Friendly => "Hi",
            Tone::Casual => "Hey",
        }
    }

    /// Get closing phrase for this tone
    pub fn closing_phrase(&self) -> &'static str {
        match self {
            Tone::Formal => "Thank you for your valuable time.",
            Tone::Professional => "Thank you for considering us.",
            Tone::Friendly => "Thanks! Let me know if you need anything else.",
            Tone::Casual => "Cool, just ping me if you need help!",
        }
    }
}

impl Default for Tone {
    fn default() -> Self {
        Tone::Professional
    }
}

/// Language complexity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LanguageComplexity {
    /// Simple words, short sentences
    Simple,
    /// Moderate vocabulary, clear explanations
    Moderate,
    /// Technical terms acceptable, detailed explanations
    Sophisticated,
}

impl Default for LanguageComplexity {
    fn default() -> Self {
        LanguageComplexity::Moderate
    }
}

/// Response urgency level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseUrgency {
    /// Relaxed pace, no pressure
    Relaxed,
    /// Normal conversational pace
    Normal,
    /// Slightly faster, showing efficiency
    Efficient,
    /// Quick responses, highlighting urgency
    Urgent,
}

impl Default for ResponseUrgency {
    fn default() -> Self {
        ResponseUrgency::Normal
    }
}

/// Agent persona configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Persona {
    /// Persona name/identifier
    pub name: String,
    /// Communication tone
    pub tone: Tone,
    /// Warmth level (0.0 = cold/factual, 1.0 = very warm/empathetic)
    pub warmth: f32,
    /// Empathy level (0.0 = neutral, 1.0 = highly empathetic)
    pub empathy: f32,
    /// Language complexity
    pub language_complexity: LanguageComplexity,
    /// Response urgency
    pub urgency: ResponseUrgency,
    /// Whether to use customer's name frequently
    pub use_customer_name: bool,
    /// Whether to acknowledge emotions
    pub acknowledge_emotions: bool,
    /// Whether to use Hindi words/phrases in English
    pub use_hinglish: bool,
    /// Maximum response length preference (words)
    pub max_response_words: usize,
}

impl Default for Persona {
    fn default() -> Self {
        Self {
            name: "kotak_advisor".to_string(),
            tone: Tone::Professional,
            warmth: 0.8,
            empathy: 0.7,
            language_complexity: LanguageComplexity::Moderate,
            urgency: ResponseUrgency::Normal,
            use_customer_name: true,
            acknowledge_emotions: true,
            use_hinglish: false,
            max_response_words: 60,
        }
    }
}

impl Persona {
    /// Create a new persona
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Create persona optimized for a customer segment
    pub fn for_segment(segment: CustomerSegment) -> Self {
        match segment {
            CustomerSegment::HighValue => Self {
                name: "premium_advisor".to_string(),
                tone: Tone::Formal,
                warmth: 0.9,
                empathy: 0.8,
                language_complexity: LanguageComplexity::Sophisticated,
                urgency: ResponseUrgency::Efficient,
                use_customer_name: true,
                acknowledge_emotions: true,
                use_hinglish: false,
                max_response_words: 80,
            },
            CustomerSegment::TrustSeeker => Self {
                name: "trust_builder".to_string(),
                tone: Tone::Professional,
                warmth: 0.95,
                empathy: 0.95,
                language_complexity: LanguageComplexity::Moderate,
                urgency: ResponseUrgency::Relaxed,
                use_customer_name: true,
                acknowledge_emotions: true,
                use_hinglish: true,
                max_response_words: 70,
            },
            CustomerSegment::FirstTime => Self {
                name: "helpful_guide".to_string(),
                tone: Tone::Friendly,
                warmth: 0.9,
                empathy: 0.85,
                language_complexity: LanguageComplexity::Simple,
                urgency: ResponseUrgency::Relaxed,
                use_customer_name: true,
                acknowledge_emotions: true,
                use_hinglish: true,
                max_response_words: 50,
            },
            CustomerSegment::PriceSensitive => Self {
                name: "value_expert".to_string(),
                tone: Tone::Professional,
                warmth: 0.7,
                empathy: 0.6,
                language_complexity: LanguageComplexity::Moderate,
                urgency: ResponseUrgency::Normal,
                use_customer_name: true,
                acknowledge_emotions: false,
                use_hinglish: false,
                max_response_words: 55,
            },
            CustomerSegment::Women => Self {
                name: "shakti_advisor".to_string(),
                tone: Tone::Friendly,
                warmth: 0.95,
                empathy: 0.9,
                language_complexity: LanguageComplexity::Simple,
                urgency: ResponseUrgency::Relaxed,
                use_customer_name: true,
                acknowledge_emotions: true,
                use_hinglish: true,
                max_response_words: 55,
            },
            CustomerSegment::Professional => Self {
                name: "smart_advisor".to_string(),
                tone: Tone::Professional,
                warmth: 0.75,
                empathy: 0.65,
                language_complexity: LanguageComplexity::Moderate,
                urgency: ResponseUrgency::Efficient,
                use_customer_name: false,
                acknowledge_emotions: false,
                use_hinglish: false,
                max_response_words: 45,
            },
        }
    }

    /// Builder: set tone
    pub fn with_tone(mut self, tone: Tone) -> Self {
        self.tone = tone;
        self
    }

    /// Builder: set warmth
    pub fn with_warmth(mut self, warmth: f32) -> Self {
        self.warmth = warmth.clamp(0.0, 1.0);
        self
    }

    /// Builder: set empathy
    pub fn with_empathy(mut self, empathy: f32) -> Self {
        self.empathy = empathy.clamp(0.0, 1.0);
        self
    }

    /// Builder: set language complexity
    pub fn with_complexity(mut self, complexity: LanguageComplexity) -> Self {
        self.language_complexity = complexity;
        self
    }

    /// Builder: set urgency
    pub fn with_urgency(mut self, urgency: ResponseUrgency) -> Self {
        self.urgency = urgency;
        self
    }

    /// Builder: enable hinglish
    pub fn with_hinglish(mut self, enabled: bool) -> Self {
        self.use_hinglish = enabled;
        self
    }

    /// Get system prompt instructions for this persona
    pub fn system_prompt_instructions(&self) -> String {
        let mut instructions: Vec<String> = Vec::new();

        // Tone instructions
        instructions.push(match self.tone {
            Tone::Formal => "Use formal, respectful language. Address the customer with honorifics.".to_string(),
            Tone::Professional => "Use professional but warm language. Be clear and helpful.".to_string(),
            Tone::Friendly => "Use friendly, approachable language. Feel free to be conversational.".to_string(),
            Tone::Casual => "Use casual, relaxed language. Be natural and easy-going.".to_string(),
        });

        // Warmth instructions
        if self.warmth > 0.8 {
            instructions.push("Be very warm and welcoming in your responses.".to_string());
        } else if self.warmth > 0.6 {
            instructions.push("Maintain a warm and helpful tone.".to_string());
        } else {
            instructions.push("Keep responses focused and factual.".to_string());
        }

        // Empathy instructions
        if self.empathy > 0.8 {
            instructions.push("Show strong empathy. Acknowledge concerns and feelings explicitly.".to_string());
        } else if self.empathy > 0.5 {
            instructions.push("Show understanding when customer expresses concerns.".to_string());
        }

        // Complexity instructions
        instructions.push(match self.language_complexity {
            LanguageComplexity::Simple => "Use simple words and short sentences. Avoid jargon.".to_string(),
            LanguageComplexity::Moderate => "Use clear language. Explain any technical terms.".to_string(),
            LanguageComplexity::Sophisticated => "You can use industry terms. Assume customer understands finance.".to_string(),
        });

        // Urgency instructions
        instructions.push(match self.urgency {
            ResponseUrgency::Relaxed => "Take your time. Don't rush the customer.".to_string(),
            ResponseUrgency::Normal => "Maintain a natural pace.".to_string(),
            ResponseUrgency::Efficient => "Be efficient and value the customer's time.".to_string(),
            ResponseUrgency::Urgent => "Be quick and highlight time-sensitive benefits.".to_string(),
        });

        // Customer name usage
        if self.use_customer_name {
            instructions.push("Use the customer's name when appropriate to personalize the conversation.".to_string());
        }

        // Emotion acknowledgment
        if self.acknowledge_emotions {
            instructions.push("Acknowledge customer emotions before addressing their question.".to_string());
        }

        // Hinglish
        if self.use_hinglish {
            instructions.push("Feel free to use common Hindi words/phrases if the customer uses them (e.g., 'ji', 'bilkul', 'zaroor').".to_string());
        }

        // Response length
        instructions.push(format!(
            "Keep responses concise, ideally under {} words.",
            self.max_response_words
        ));

        instructions.join(" ")
    }

    /// Blend two personas based on a factor (0.0 = self, 1.0 = other)
    pub fn blend(&self, other: &Persona, factor: f32) -> Self {
        let factor = factor.clamp(0.0, 1.0);
        let inv = 1.0 - factor;

        Self {
            name: if factor > 0.5 {
                other.name.clone()
            } else {
                self.name.clone()
            },
            tone: if factor > 0.5 { other.tone } else { self.tone },
            warmth: self.warmth * inv + other.warmth * factor,
            empathy: self.empathy * inv + other.empathy * factor,
            language_complexity: if factor > 0.5 {
                other.language_complexity
            } else {
                self.language_complexity
            },
            urgency: if factor > 0.5 {
                other.urgency
            } else {
                self.urgency
            },
            use_customer_name: if factor > 0.5 {
                other.use_customer_name
            } else {
                self.use_customer_name
            },
            acknowledge_emotions: if factor > 0.5 {
                other.acknowledge_emotions
            } else {
                self.acknowledge_emotions
            },
            use_hinglish: self.use_hinglish || other.use_hinglish,
            max_response_words: ((self.max_response_words as f32) * inv
                + (other.max_response_words as f32) * factor) as usize,
        }
    }
}

/// Pre-defined persona templates
pub struct PersonaTemplates;

impl PersonaTemplates {
    /// Default Kotak advisor
    pub fn kotak_advisor() -> Persona {
        Persona::default()
    }

    /// Premium/high-value customer advisor
    pub fn premium_advisor() -> Persona {
        Persona::for_segment(CustomerSegment::HighValue)
    }

    /// Trust-building advisor (for switchers from NBFCs)
    pub fn trust_builder() -> Persona {
        Persona::for_segment(CustomerSegment::TrustSeeker)
    }

    /// First-time customer guide
    pub fn first_time_guide() -> Persona {
        Persona::for_segment(CustomerSegment::FirstTime)
    }

    /// Quick, efficient advisor for professionals
    pub fn smart_advisor() -> Persona {
        Persona::for_segment(CustomerSegment::Professional)
    }

    /// Empathetic advisor for sensitive situations
    pub fn empathetic_advisor() -> Persona {
        Persona::new("empathetic_advisor")
            .with_tone(Tone::Friendly)
            .with_warmth(0.98)
            .with_empathy(0.98)
            .with_complexity(LanguageComplexity::Simple)
            .with_urgency(ResponseUrgency::Relaxed)
    }

    /// Sales-focused closer
    pub fn closer() -> Persona {
        Persona::new("closer")
            .with_tone(Tone::Professional)
            .with_warmth(0.8)
            .with_empathy(0.7)
            .with_complexity(LanguageComplexity::Moderate)
            .with_urgency(ResponseUrgency::Urgent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persona_default() {
        let persona = Persona::default();
        assert_eq!(persona.tone, Tone::Professional);
        assert!(persona.warmth > 0.0);
        assert!(persona.empathy > 0.0);
    }

    #[test]
    fn test_persona_for_segment() {
        let persona = Persona::for_segment(CustomerSegment::HighValue);
        assert_eq!(persona.tone, Tone::Formal);
        assert!(persona.warmth >= 0.9);

        let persona = Persona::for_segment(CustomerSegment::FirstTime);
        assert_eq!(persona.tone, Tone::Friendly);
        assert_eq!(persona.language_complexity, LanguageComplexity::Simple);
    }

    #[test]
    fn test_persona_builder() {
        let persona = Persona::new("test")
            .with_tone(Tone::Casual)
            .with_warmth(0.5)
            .with_hinglish(true);

        assert_eq!(persona.tone, Tone::Casual);
        assert_eq!(persona.warmth, 0.5);
        assert!(persona.use_hinglish);
    }

    #[test]
    fn test_persona_blend() {
        let formal = Persona::new("formal")
            .with_tone(Tone::Formal)
            .with_warmth(0.3);
        let casual = Persona::new("casual")
            .with_tone(Tone::Casual)
            .with_warmth(0.9);

        let blended = formal.blend(&casual, 0.5);
        assert_eq!(blended.warmth, 0.6); // Average
    }

    #[test]
    fn test_system_prompt_instructions() {
        let persona = Persona::for_segment(CustomerSegment::TrustSeeker);
        let instructions = persona.system_prompt_instructions();

        assert!(instructions.contains("empathy") || instructions.contains("Acknowledge"));
        assert!(instructions.contains("Hindi") || instructions.contains("hindi"));
    }

    #[test]
    fn test_tone_phrases() {
        assert_eq!(Tone::Formal.greeting_prefix(), "Respected");
        assert_eq!(Tone::Casual.greeting_prefix(), "Hey");
    }

    #[test]
    fn test_templates() {
        let premium = PersonaTemplates::premium_advisor();
        assert_eq!(premium.tone, Tone::Formal);

        let empathetic = PersonaTemplates::empathetic_advisor();
        assert!(empathetic.empathy >= 0.95);
    }
}
