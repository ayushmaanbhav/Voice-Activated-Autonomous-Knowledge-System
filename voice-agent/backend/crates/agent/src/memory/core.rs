//! Core Memory Module
//!
//! Implements MemGPT-style core memory with two primary blocks:
//! - **Human Block**: Information about the customer (preferences, facts, context)
//! - **Persona Block**: Agent's self-concept, personality, behavioral guidelines
//!
//! Core memory is always included in the LLM context and can be modified
//! via explicit memory manipulation functions.
//!
//! Reference: MemGPT paper (arXiv:2310.08560)

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Maximum size for each memory block (in characters)
const DEFAULT_BLOCK_SIZE_LIMIT: usize = 2000;

/// Core memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMemoryConfig {
    /// Maximum characters for human block
    pub human_block_limit: usize,
    /// Maximum characters for persona block
    pub persona_block_limit: usize,
    /// Enable automatic fact extraction
    pub auto_extract_facts: bool,
}

impl Default for CoreMemoryConfig {
    fn default() -> Self {
        Self {
            human_block_limit: DEFAULT_BLOCK_SIZE_LIMIT,
            persona_block_limit: DEFAULT_BLOCK_SIZE_LIMIT,
            auto_extract_facts: true,
        }
    }
}

/// A single entry in a memory block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBlockEntry {
    /// Unique key for this entry (e.g., "name", "preferred_language")
    pub key: String,
    /// The value/content
    pub value: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// When this entry was created
    pub created_at: DateTime<Utc>,
    /// When this entry was last updated
    pub updated_at: DateTime<Utc>,
    /// Source of this information (e.g., "user_stated", "inferred", "system")
    pub source: EntrySource,
}

impl MemoryBlockEntry {
    pub fn new(key: impl Into<String>, value: impl Into<String>, source: EntrySource) -> Self {
        let now = Utc::now();
        Self {
            key: key.into(),
            value: value.into(),
            confidence: 1.0,
            created_at: now,
            updated_at: now,
            source,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Source of a memory entry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntrySource {
    /// User explicitly stated this information
    UserStated,
    /// Inferred from conversation context
    Inferred,
    /// System/configuration provided
    System,
    /// Extracted from external source (e.g., CRM)
    External,
}

/// Human memory block - stores information about the customer
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HumanBlock {
    /// Customer's name
    pub name: Option<String>,
    /// Preferred language
    pub preferred_language: Option<String>,
    /// Customer preferences and facts
    pub facts: HashMap<String, MemoryBlockEntry>,
    /// Conversation context notes
    pub context_notes: Vec<String>,
    /// Character count for limit checking
    char_count: usize,
}

impl HumanBlock {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set customer name
    pub fn set_name(&mut self, name: impl Into<String>) {
        let name = name.into();
        // Update char count
        if let Some(ref old_name) = self.name {
            self.char_count = self.char_count.saturating_sub(old_name.len());
        }
        self.char_count += name.len();
        self.name = Some(name);
    }

    /// Set preferred language
    pub fn set_language(&mut self, language: impl Into<String>) {
        let lang = language.into();
        // Update char count
        if let Some(ref old_lang) = self.preferred_language {
            self.char_count = self.char_count.saturating_sub(old_lang.len());
        }
        self.char_count += lang.len();
        self.preferred_language = Some(lang);
    }

    /// Add or update a fact
    pub fn set_fact(&mut self, key: impl Into<String>, value: impl Into<String>, source: EntrySource) {
        let key = key.into();
        let value = value.into();

        // Update char count
        if let Some(existing) = self.facts.get(&key) {
            self.char_count = self.char_count.saturating_sub(existing.key.len() + existing.value.len());
        }
        self.char_count += key.len() + value.len();

        let entry = MemoryBlockEntry::new(key.clone(), value, source);
        self.facts.insert(key, entry);
    }

    /// Get a fact by key
    pub fn get_fact(&self, key: &str) -> Option<&MemoryBlockEntry> {
        self.facts.get(key)
    }

    /// Add a context note
    pub fn add_context_note(&mut self, note: impl Into<String>) {
        let note = note.into();
        self.char_count += note.len();
        self.context_notes.push(note);
    }

    /// Get total character count
    pub fn char_count(&self) -> usize {
        self.char_count
    }

    /// Format block for LLM context
    pub fn format_for_context(&self) -> String {
        let mut output = String::new();

        if let Some(ref name) = self.name {
            output.push_str(&format!("Customer Name: {}\n", name));
        }

        if let Some(ref lang) = self.preferred_language {
            output.push_str(&format!("Preferred Language: {}\n", lang));
        }

        if !self.facts.is_empty() {
            output.push_str("\nKnown Facts:\n");
            for (key, entry) in &self.facts {
                output.push_str(&format!("- {}: {}\n", key, entry.value));
            }
        }

        if !self.context_notes.is_empty() {
            output.push_str("\nContext Notes:\n");
            for note in &self.context_notes {
                output.push_str(&format!("- {}\n", note));
            }
        }

        output
    }
}

/// Persona memory block - stores agent's self-concept and guidelines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaBlock {
    /// Agent's name
    pub name: String,
    /// Agent's role
    pub role: String,
    /// Personality traits
    pub personality: String,
    /// Behavioral guidelines
    pub guidelines: Vec<String>,
    /// Domain-specific knowledge pointers
    pub domain_expertise: Vec<String>,
    /// Current conversation goals
    pub current_goals: Vec<String>,
    /// Character count for limit checking
    char_count: usize,
}

impl Default for PersonaBlock {
    fn default() -> Self {
        let name = "Priya".to_string();
        let role = "Gold Loan Advisor at Kotak Mahindra Bank".to_string();
        let personality = "warm, professional, and helpful".to_string();
        let guidelines = vec![
            "Always be respectful and patient".to_string(),
            "Explain gold loan benefits clearly".to_string(),
            "Compare favorably with competitors when relevant".to_string(),
            "Offer to schedule branch visits".to_string(),
        ];
        let domain_expertise = vec![
            "Gold loan products and interest rates".to_string(),
            "Kotak Mahindra Bank services".to_string(),
            "Gold purity assessment".to_string(),
            "Loan documentation requirements".to_string(),
        ];

        // Calculate actual char count
        let char_count = name.len()
            + role.len()
            + personality.len()
            + guidelines.iter().map(|s| s.len()).sum::<usize>()
            + domain_expertise.iter().map(|s| s.len()).sum::<usize>();

        Self {
            name,
            role,
            personality,
            guidelines,
            domain_expertise,
            current_goals: Vec::new(),
            char_count,
        }
    }
}

impl PersonaBlock {
    pub fn new(name: impl Into<String>, role: impl Into<String>) -> Self {
        let name = name.into();
        let role = role.into();
        let char_count = name.len() + role.len();

        Self {
            name,
            role,
            personality: "professional and helpful".to_string(),
            guidelines: Vec::new(),
            domain_expertise: Vec::new(),
            current_goals: Vec::new(),
            char_count,
        }
    }

    /// Set personality description
    pub fn set_personality(&mut self, personality: impl Into<String>) {
        let personality = personality.into();
        self.char_count = self.char_count.saturating_sub(self.personality.len());
        self.char_count += personality.len();
        self.personality = personality;
    }

    /// Add a behavioral guideline
    pub fn add_guideline(&mut self, guideline: impl Into<String>) {
        let guideline = guideline.into();
        self.char_count += guideline.len();
        self.guidelines.push(guideline);
    }

    /// Set current conversation goals
    pub fn set_goals(&mut self, goals: Vec<String>) {
        // Remove old goals from char count
        for goal in &self.current_goals {
            self.char_count = self.char_count.saturating_sub(goal.len());
        }
        // Add new goals
        for goal in &goals {
            self.char_count += goal.len();
        }
        self.current_goals = goals;
    }

    /// Add a current goal
    pub fn add_goal(&mut self, goal: impl Into<String>) {
        let goal = goal.into();
        self.char_count += goal.len();
        self.current_goals.push(goal);
    }

    /// Clear current goals
    pub fn clear_goals(&mut self) {
        for goal in &self.current_goals {
            self.char_count = self.char_count.saturating_sub(goal.len());
        }
        self.current_goals.clear();
    }

    /// Get total character count
    pub fn char_count(&self) -> usize {
        self.char_count
    }

    /// Format block for LLM context
    pub fn format_for_context(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!("You are {}, a {}.\n", self.name, self.role));
        output.push_str(&format!("Personality: {}\n", self.personality));

        if !self.guidelines.is_empty() {
            output.push_str("\nGuidelines:\n");
            for guideline in &self.guidelines {
                output.push_str(&format!("- {}\n", guideline));
            }
        }

        if !self.domain_expertise.is_empty() {
            output.push_str("\nExpertise Areas:\n");
            for area in &self.domain_expertise {
                output.push_str(&format!("- {}\n", area));
            }
        }

        if !self.current_goals.is_empty() {
            output.push_str("\nCurrent Goals:\n");
            for goal in &self.current_goals {
                output.push_str(&format!("- {}\n", goal));
            }
        }

        output
    }
}

/// Core Memory - MemGPT-style in-context memory
///
/// Always included in the LLM's context window. Contains:
/// - Human block: Customer information
/// - Persona block: Agent self-concept
pub struct CoreMemory {
    config: CoreMemoryConfig,
    human: RwLock<HumanBlock>,
    persona: RwLock<PersonaBlock>,
}

impl CoreMemory {
    /// Create new core memory with default persona
    pub fn new(config: CoreMemoryConfig) -> Self {
        Self {
            config,
            human: RwLock::new(HumanBlock::new()),
            persona: RwLock::new(PersonaBlock::default()),
        }
    }

    /// Create with custom persona
    pub fn with_persona(config: CoreMemoryConfig, persona: PersonaBlock) -> Self {
        Self {
            config,
            human: RwLock::new(HumanBlock::new()),
            persona: RwLock::new(persona),
        }
    }

    // =========================================================================
    // Human Block Operations
    // =========================================================================

    /// Append to human block
    ///
    /// MemGPT function: core_memory_append
    pub fn human_append(&self, key: &str, value: &str) -> Result<(), CoreMemoryError> {
        let mut human = self.human.write();

        // Check size limit
        let new_size = human.char_count() + key.len() + value.len();
        if new_size > self.config.human_block_limit {
            return Err(CoreMemoryError::BlockSizeLimitExceeded {
                block: "human".to_string(),
                limit: self.config.human_block_limit,
                requested: new_size,
            });
        }

        human.set_fact(key, value, EntrySource::UserStated);
        Ok(())
    }

    /// Replace in human block
    ///
    /// MemGPT function: core_memory_replace
    pub fn human_replace(&self, key: &str, old_value: &str, new_value: &str) -> Result<(), CoreMemoryError> {
        let mut human = self.human.write();

        // Verify old value exists and matches
        let current_value = human.facts.get(key).map(|e| e.value.clone());
        match current_value {
            Some(ref val) if val != old_value => {
                return Err(CoreMemoryError::ValueMismatch {
                    key: key.to_string(),
                    expected: old_value.to_string(),
                    actual: val.clone(),
                });
            }
            None => {
                return Err(CoreMemoryError::KeyNotFound(key.to_string()));
            }
            _ => {}
        }

        // Check new size
        let size_diff = new_value.len() as i64 - old_value.len() as i64;
        let new_size = (human.char_count() as i64 + size_diff) as usize;
        if new_size > self.config.human_block_limit {
            return Err(CoreMemoryError::BlockSizeLimitExceeded {
                block: "human".to_string(),
                limit: self.config.human_block_limit,
                requested: new_size,
            });
        }

        // Update char count first
        human.char_count = (human.char_count as i64 + size_diff) as usize;

        // Now update the entry
        if let Some(entry) = human.facts.get_mut(key) {
            entry.value = new_value.to_string();
            entry.updated_at = Utc::now();
        }

        Ok(())
    }

    /// Set customer name
    pub fn set_customer_name(&self, name: &str) {
        self.human.write().set_name(name);
    }

    /// Set customer language
    pub fn set_customer_language(&self, language: &str) {
        self.human.write().set_language(language);
    }

    /// Add context note
    pub fn add_context_note(&self, note: &str) {
        self.human.write().add_context_note(note);
    }

    /// Get human block snapshot
    pub fn human_snapshot(&self) -> HumanBlock {
        self.human.read().clone()
    }

    // =========================================================================
    // Persona Block Operations
    // =========================================================================

    /// Set persona name
    pub fn set_persona_name(&self, name: &str) {
        let mut persona = self.persona.write();
        persona.char_count = persona.char_count.saturating_sub(persona.name.len());
        persona.name = name.to_string();
        persona.char_count += name.len();
    }

    /// Update persona goals
    pub fn set_persona_goals(&self, goals: Vec<String>) {
        self.persona.write().set_goals(goals);
    }

    /// Add a persona goal
    pub fn add_persona_goal(&self, goal: &str) {
        self.persona.write().add_goal(goal);
    }

    /// Clear persona goals
    pub fn clear_persona_goals(&self) {
        self.persona.write().clear_goals();
    }

    /// Get persona block snapshot
    pub fn persona_snapshot(&self) -> PersonaBlock {
        self.persona.read().clone()
    }

    // =========================================================================
    // Combined Operations
    // =========================================================================

    /// Get formatted context for LLM
    ///
    /// Returns the combined human and persona blocks formatted for inclusion
    /// in the LLM's system prompt.
    pub fn format_for_context(&self) -> String {
        let mut output = String::new();

        // Persona block first (agent identity)
        output.push_str("## Agent Identity\n");
        output.push_str(&self.persona.read().format_for_context());

        // Human block (customer context)
        let human = self.human.read();
        if human.name.is_some() || !human.facts.is_empty() {
            output.push_str("\n## Customer Context\n");
            output.push_str(&human.format_for_context());
        }

        output
    }

    /// Get total character count
    pub fn total_char_count(&self) -> usize {
        self.human.read().char_count() + self.persona.read().char_count()
    }

    /// Estimate token count (rough: 4 chars per token)
    pub fn estimated_tokens(&self) -> usize {
        self.total_char_count() / 4
    }

    /// Check if within limits
    pub fn is_within_limits(&self) -> bool {
        let human = self.human.read();
        let persona = self.persona.read();

        human.char_count() <= self.config.human_block_limit
            && persona.char_count() <= self.config.persona_block_limit
    }

    /// Clear all human block data (for new session)
    pub fn clear_human_block(&self) {
        *self.human.write() = HumanBlock::new();
    }

    /// Reset to default state
    pub fn reset(&self) {
        *self.human.write() = HumanBlock::new();
        *self.persona.write() = PersonaBlock::default();
    }
}

impl Default for CoreMemory {
    fn default() -> Self {
        Self::new(CoreMemoryConfig::default())
    }
}

/// Errors that can occur during core memory operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum CoreMemoryError {
    #[error("Block size limit exceeded: {block} block limit is {limit} chars, requested {requested}")]
    BlockSizeLimitExceeded {
        block: String,
        limit: usize,
        requested: usize,
    },

    #[error("Key not found: {0}")]
    KeyNotFound(String),

    #[error("Value mismatch for key '{key}': expected '{expected}', found '{actual}'")]
    ValueMismatch {
        key: String,
        expected: String,
        actual: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_human_block_basics() {
        let mut human = HumanBlock::new();

        human.set_name("Rajesh Kumar");
        human.set_language("Hindi");
        human.set_fact("gold_weight", "50 grams", EntrySource::UserStated);

        assert_eq!(human.name, Some("Rajesh Kumar".to_string()));
        assert_eq!(human.preferred_language, Some("Hindi".to_string()));
        assert!(human.get_fact("gold_weight").is_some());
        assert_eq!(human.get_fact("gold_weight").unwrap().value, "50 grams");
    }

    #[test]
    fn test_persona_block_basics() {
        let mut persona = PersonaBlock::new("Priya", "Gold Loan Advisor");

        persona.set_personality("warm and helpful");
        persona.add_guideline("Always greet customers warmly");
        persona.add_goal("Help customer understand gold loan benefits");

        assert_eq!(persona.name, "Priya");
        assert!(persona.guidelines.len() > 0);
        assert!(persona.current_goals.len() > 0);
    }

    #[test]
    fn test_core_memory_append() {
        let memory = CoreMemory::default();

        assert!(memory.human_append("loan_amount", "500000").is_ok());

        let human = memory.human_snapshot();
        assert!(human.get_fact("loan_amount").is_some());
    }

    #[test]
    fn test_core_memory_replace() {
        let memory = CoreMemory::default();

        memory.human_append("loan_amount", "500000").unwrap();
        assert!(memory.human_replace("loan_amount", "500000", "750000").is_ok());

        let human = memory.human_snapshot();
        assert_eq!(human.get_fact("loan_amount").unwrap().value, "750000");
    }

    #[test]
    fn test_core_memory_replace_mismatch() {
        let memory = CoreMemory::default();

        memory.human_append("loan_amount", "500000").unwrap();
        let result = memory.human_replace("loan_amount", "wrong_value", "750000");

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CoreMemoryError::ValueMismatch { .. }));
    }

    #[test]
    fn test_format_for_context() {
        let memory = CoreMemory::default();

        memory.set_customer_name("Rajesh");
        memory.human_append("gold_weight", "50 grams").unwrap();
        memory.add_persona_goal("Explain loan benefits");

        let context = memory.format_for_context();

        assert!(context.contains("Priya"));
        assert!(context.contains("Rajesh"));
        assert!(context.contains("50 grams"));
        assert!(context.contains("Explain loan benefits"));
    }

    #[test]
    fn test_size_limits() {
        let config = CoreMemoryConfig {
            human_block_limit: 100,
            persona_block_limit: 100,
            auto_extract_facts: false,
        };
        let memory = CoreMemory::new(config);

        // This should fail due to size limit
        let long_value = "x".repeat(150);
        let result = memory.human_append("key", &long_value);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CoreMemoryError::BlockSizeLimitExceeded { .. }));
    }
}
