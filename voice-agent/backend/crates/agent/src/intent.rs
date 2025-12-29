//! Intent Detection and Slot Filling
//!
//! Detects user intents and extracts relevant entities.

use std::collections::HashMap;
use parking_lot::RwLock;
use regex::Regex;
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

/// Intent definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    /// Intent name
    pub name: String,
    /// Description
    pub description: String,
    /// Required slots
    pub required_slots: Vec<String>,
    /// Optional slots
    pub optional_slots: Vec<String>,
    /// Example utterances
    pub examples: Vec<String>,
}

/// Slot/Entity definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Slot {
    /// Slot name
    pub name: String,
    /// Slot type
    pub slot_type: SlotType,
    /// Extracted value
    pub value: Option<String>,
    /// Confidence
    pub confidence: f32,
}

/// Slot types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlotType {
    Text,
    Number,
    Currency,
    Phone,
    Date,
    Time,
    Location,
    Enum(Vec<String>),
}

/// Detected intent with slots
#[derive(Debug, Clone)]
pub struct DetectedIntent {
    /// Intent name
    pub intent: String,
    /// Confidence score
    pub confidence: f32,
    /// Extracted slots
    pub slots: HashMap<String, Slot>,
    /// Alternative intents
    pub alternatives: Vec<(String, f32)>,
}

/// Compiled slot pattern with its regex
struct CompiledSlotPattern {
    name: String,
    regex: Regex,
    slot_type: SlotType,
    /// Multiplier for numeric values (e.g., 100000 for "lakh")
    multiplier: Option<f64>,
}

/// Intent detector
pub struct IntentDetector {
    intents: RwLock<Vec<Intent>>,
    /// P0 FIX: Compiled regex patterns for slot extraction
    compiled_patterns: HashMap<String, Vec<CompiledSlotPattern>>,
}

impl IntentDetector {
    /// Create a new intent detector with gold loan intents
    pub fn new() -> Self {
        let mut detector = Self {
            intents: RwLock::new(Vec::new()),
            compiled_patterns: HashMap::new(),
        };

        detector.register_gold_loan_intents();
        detector.compile_slot_patterns();

        detector
    }

    /// Register gold loan specific intents
    fn register_gold_loan_intents(&self) {
        let intents = vec![
            Intent {
                name: "loan_inquiry".to_string(),
                description: "User wants to know about gold loan".to_string(),
                required_slots: vec![],
                optional_slots: vec!["loan_amount".to_string(), "gold_weight".to_string()],
                examples: vec![
                    "I want a gold loan".to_string(),
                    "Tell me about gold loan".to_string(),
                    "Gold loan kaise milega".to_string(),
                ],
            },
            Intent {
                name: "interest_rate".to_string(),
                description: "User asking about interest rates".to_string(),
                required_slots: vec![],
                optional_slots: vec!["loan_amount".to_string()],
                examples: vec![
                    "What is the interest rate".to_string(),
                    "Interest rate kitna hai".to_string(),
                    "Rate of interest".to_string(),
                ],
            },
            Intent {
                name: "eligibility_check".to_string(),
                description: "User wants to check eligibility".to_string(),
                required_slots: vec!["gold_weight".to_string()],
                optional_slots: vec!["gold_purity".to_string()],
                examples: vec![
                    "Am I eligible".to_string(),
                    "Can I get a loan".to_string(),
                    "Kitna loan milega".to_string(),
                ],
            },
            Intent {
                name: "switch_lender".to_string(),
                description: "User wants to switch from current lender".to_string(),
                required_slots: vec!["current_lender".to_string()],
                optional_slots: vec!["current_rate".to_string(), "loan_amount".to_string()],
                examples: vec![
                    "I want to switch from Muthoot".to_string(),
                    "Transfer my loan".to_string(),
                    "Can I move my gold loan".to_string(),
                ],
            },
            Intent {
                name: "objection".to_string(),
                description: "User has concerns or objections".to_string(),
                required_slots: vec![],
                optional_slots: vec!["objection_type".to_string()],
                examples: vec![
                    "I'm not sure".to_string(),
                    "What if something goes wrong".to_string(),
                    "Is it safe".to_string(),
                    "Mujhe dar lagta hai".to_string(),
                ],
            },
            Intent {
                name: "schedule_visit".to_string(),
                description: "User wants to visit branch".to_string(),
                required_slots: vec![],
                optional_slots: vec!["location".to_string(), "date".to_string(), "time".to_string()],
                examples: vec![
                    "I want to visit".to_string(),
                    "Schedule appointment".to_string(),
                    "Kab aa sakte hain".to_string(),
                ],
            },
            Intent {
                name: "documentation".to_string(),
                description: "User asking about required documents".to_string(),
                required_slots: vec![],
                optional_slots: vec![],
                examples: vec![
                    "What documents needed".to_string(),
                    "Kya documents chahiye".to_string(),
                    "Paper work".to_string(),
                ],
            },
            Intent {
                name: "greeting".to_string(),
                description: "User greeting".to_string(),
                required_slots: vec![],
                optional_slots: vec![],
                examples: vec![
                    "Hello".to_string(),
                    "Hi".to_string(),
                    "Namaste".to_string(),
                ],
            },
            Intent {
                name: "farewell".to_string(),
                description: "User saying goodbye".to_string(),
                required_slots: vec![],
                optional_slots: vec![],
                examples: vec![
                    "Bye".to_string(),
                    "Thank you".to_string(),
                    "Dhanyavaad".to_string(),
                ],
            },
            Intent {
                name: "affirmative".to_string(),
                description: "User agreeing".to_string(),
                required_slots: vec![],
                optional_slots: vec![],
                examples: vec![
                    "Yes".to_string(),
                    "Sure".to_string(),
                    "Haan".to_string(),
                    "Okay".to_string(),
                ],
            },
            Intent {
                name: "negative".to_string(),
                description: "User declining".to_string(),
                required_slots: vec![],
                optional_slots: vec![],
                examples: vec![
                    "No".to_string(),
                    "Not now".to_string(),
                    "Nahi".to_string(),
                ],
            },
        ];

        *self.intents.write() = intents;
    }

    /// P0 FIX: Compile slot patterns into regex at startup
    ///
    /// This replaces the old register_slot_patterns() which stored patterns
    /// as strings but never used them. Now patterns are compiled once and
    /// reused for all extractions.
    ///
    /// P0 FIX (Dec 2025): Added Devanagari script support for Hindi users.
    /// Includes:
    /// - Devanagari numerals (०-९)
    /// - Hindi number words (पांच, दस, बीस, etc.)
    /// - Hindi multiplier words (लाख, करोड़, हज़ार)
    fn compile_slot_patterns(&mut self) {
        // Loan amount patterns - P0 FIX: Added Devanagari support
        let loan_patterns = vec![
            // === DEVANAGARI PATTERNS (Hindi) - Check first for proper Hindi support ===

            // Hindi: करोड़ (crore) with Devanagari numerals
            CompiledSlotPattern {
                name: "hindi_crore_devanagari".to_string(),
                regex: Regex::new(r"([०-९]+(?:\.[०-९]+)?)\s*(?:करोड़|करोड)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(10_000_000.0),
            },
            // Hindi: करोड़ (crore) with ASCII numerals
            CompiledSlotPattern {
                name: "hindi_crore_ascii".to_string(),
                regex: Regex::new(r"(\d+(?:\.\d+)?)\s*(?:करोड़|करोड)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(10_000_000.0),
            },
            // Hindi: लाख (lakh) with Devanagari numerals
            CompiledSlotPattern {
                name: "hindi_lakh_devanagari".to_string(),
                regex: Regex::new(r"([०-९]+(?:\.[०-९]+)?)\s*(?:लाख|लख)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(100_000.0),
            },
            // Hindi: लाख (lakh) with ASCII numerals
            CompiledSlotPattern {
                name: "hindi_lakh_ascii".to_string(),
                regex: Regex::new(r"(\d+(?:\.\d+)?)\s*(?:लाख|लख)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(100_000.0),
            },
            // Hindi: हज़ार (thousand) with Devanagari numerals
            CompiledSlotPattern {
                name: "hindi_hazar_devanagari".to_string(),
                regex: Regex::new(r"([०-९]+(?:\.[०-९]+)?)\s*(?:हज़ार|हजार)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(1_000.0),
            },
            // Hindi: हज़ार (thousand) with ASCII numerals
            CompiledSlotPattern {
                name: "hindi_hazar_ascii".to_string(),
                regex: Regex::new(r"(\d+(?:\.\d+)?)\s*(?:हज़ार|हजार)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(1_000.0),
            },
            // Hindi number words with लाख
            CompiledSlotPattern {
                name: "hindi_word_lakh".to_string(),
                regex: Regex::new(r"(एक|दो|तीन|चार|पांच|पाँच|छह|छे|सात|आठ|नौ|दस|बीस|पच्चीस|तीस|पैंतीस|चालीस|पचास|साठ|सत्तर|अस्सी|नब्बे|सौ)\s*(?:लाख|लख)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(100_000.0), // Will be multiplied by word value
            },
            // Hindi number words with करोड़
            CompiledSlotPattern {
                name: "hindi_word_crore".to_string(),
                regex: Regex::new(r"(एक|दो|तीन|चार|पांच|पाँच|छह|छे|सात|आठ|नौ|दस)\s*(?:करोड़|करोड)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(10_000_000.0), // Will be multiplied by word value
            },
            // Hindi: रुपये amount with Devanagari
            CompiledSlotPattern {
                name: "hindi_rupees".to_string(),
                regex: Regex::new(r"([०-९\d]+(?:,[०-९\d]+)*)\s*(?:रुपये|रूपये|रुपए|₹)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: None,
            },

            // === P3 FIX: OTHER INDIC LANGUAGE MULTIPLIER PATTERNS ===

            // Tamil: லட்சம் (lakh), கோடி (crore)
            CompiledSlotPattern {
                name: "tamil_lakh".to_string(),
                regex: Regex::new(r"([௦-௯\d]+(?:\.[௦-௯\d]+)?)\s*(?:லட்சம்|இலட்சம்)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(100_000.0),
            },
            CompiledSlotPattern {
                name: "tamil_crore".to_string(),
                regex: Regex::new(r"([௦-௯\d]+(?:\.[௦-௯\d]+)?)\s*(?:கோடி)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(10_000_000.0),
            },

            // Telugu: లక్ష (lakh), కోటి (crore)
            CompiledSlotPattern {
                name: "telugu_lakh".to_string(),
                regex: Regex::new(r"([౦-౯\d]+(?:\.[౦-౯\d]+)?)\s*(?:లక్ష|లక్షలు)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(100_000.0),
            },
            CompiledSlotPattern {
                name: "telugu_crore".to_string(),
                regex: Regex::new(r"([౦-౯\d]+(?:\.[౦-౯\d]+)?)\s*(?:కోటి|కోట్లు)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(10_000_000.0),
            },

            // Bengali: লক্ষ (lakh), কোটি (crore)
            CompiledSlotPattern {
                name: "bengali_lakh".to_string(),
                regex: Regex::new(r"([০-৯\d]+(?:\.[০-৯\d]+)?)\s*(?:লক্ষ|লাখ)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(100_000.0),
            },
            CompiledSlotPattern {
                name: "bengali_crore".to_string(),
                regex: Regex::new(r"([০-৯\d]+(?:\.[০-৯\d]+)?)\s*(?:কোটি)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(10_000_000.0),
            },

            // Gujarati: લાખ (lakh), કરોડ (crore)
            CompiledSlotPattern {
                name: "gujarati_lakh".to_string(),
                regex: Regex::new(r"([૦-૯\d]+(?:\.[૦-૯\d]+)?)\s*(?:લાખ)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(100_000.0),
            },
            CompiledSlotPattern {
                name: "gujarati_crore".to_string(),
                regex: Regex::new(r"([૦-૯\d]+(?:\.[૦-૯\d]+)?)\s*(?:કરોડ)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(10_000_000.0),
            },

            // Kannada: ಲಕ್ಷ (lakh), ಕೋಟಿ (crore)
            CompiledSlotPattern {
                name: "kannada_lakh".to_string(),
                regex: Regex::new(r"([೦-೯\d]+(?:\.[೦-೯\d]+)?)\s*(?:ಲಕ್ಷ)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(100_000.0),
            },
            CompiledSlotPattern {
                name: "kannada_crore".to_string(),
                regex: Regex::new(r"([೦-೯\d]+(?:\.[೦-೯\d]+)?)\s*(?:ಕೋಟಿ)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(10_000_000.0),
            },

            // Malayalam: ലക്ഷം (lakh), കോടി (crore)
            CompiledSlotPattern {
                name: "malayalam_lakh".to_string(),
                regex: Regex::new(r"([൦-൯\d]+(?:\.[൦-൯\d]+)?)\s*(?:ലക്ഷം)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(100_000.0),
            },
            CompiledSlotPattern {
                name: "malayalam_crore".to_string(),
                regex: Regex::new(r"([൦-൯\d]+(?:\.[൦-൯\d]+)?)\s*(?:കോടി)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(10_000_000.0),
            },

            // Marathi: लाख (same as Hindi), कोटी (crore)
            CompiledSlotPattern {
                name: "marathi_crore".to_string(),
                regex: Regex::new(r"([०-९\d]+(?:\.[०-९\d]+)?)\s*(?:कोटी)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(10_000_000.0),
            },

            // Odia: ଲକ୍ଷ (lakh), କୋଟି (crore)
            CompiledSlotPattern {
                name: "odia_lakh".to_string(),
                regex: Regex::new(r"([୦-୯\d]+(?:\.[୦-୯\d]+)?)\s*(?:ଲକ୍ଷ)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(100_000.0),
            },
            CompiledSlotPattern {
                name: "odia_crore".to_string(),
                regex: Regex::new(r"([୦-୯\d]+(?:\.[୦-୯\d]+)?)\s*(?:କୋଟି)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(10_000_000.0),
            },

            // Punjabi (Gurmukhi): ਲੱਖ (lakh), ਕਰੋੜ (crore)
            CompiledSlotPattern {
                name: "punjabi_lakh".to_string(),
                regex: Regex::new(r"([੦-੯\d]+(?:\.[੦-੯\d]+)?)\s*(?:ਲੱਖ|ਲਾਖ)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(100_000.0),
            },
            CompiledSlotPattern {
                name: "punjabi_crore".to_string(),
                regex: Regex::new(r"([੦-੯\d]+(?:\.[੦-੯\d]+)?)\s*(?:ਕਰੋੜ)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(10_000_000.0),
            },

            // === ENGLISH/ROMANIZED PATTERNS ===

            // Crore (10 million) - highest priority
            CompiledSlotPattern {
                name: "crore".to_string(),
                regex: Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(?:crore|cr)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(10_000_000.0),
            },
            // Lakh (100 thousand)
            CompiledSlotPattern {
                name: "lakh".to_string(),
                regex: Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(?:lakh|lac|lakhs)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(100_000.0),
            },
            // Thousand / Hazar
            CompiledSlotPattern {
                name: "thousand".to_string(),
                regex: Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(?:thousand|hazar|hazaar|k)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: Some(1_000.0),
            },
            // Rs/₹ amount (direct value)
            CompiledSlotPattern {
                name: "rs_amount".to_string(),
                regex: Regex::new(r"(?:Rs\.?|₹|INR)\s*(\d+(?:,\d+)*(?:\.\d+)?)").unwrap(),
                slot_type: SlotType::Currency,
                multiplier: None, // Parse as-is (remove commas)
            },
            // Plain large number (>=1000)
            CompiledSlotPattern {
                name: "plain_number".to_string(),
                regex: Regex::new(r"(\d{4,})").unwrap(), // 4+ digits
                slot_type: SlotType::Currency,
                multiplier: None,
            },
        ];
        self.compiled_patterns.insert("loan_amount".to_string(), loan_patterns);

        // Gold weight patterns
        let weight_patterns = vec![
            CompiledSlotPattern {
                name: "grams".to_string(),
                regex: Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(?:grams?|gms?|g\b)").unwrap(),
                slot_type: SlotType::Number,
                multiplier: None,
            },
            CompiledSlotPattern {
                name: "tola".to_string(),
                regex: Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(?:tola|tole)").unwrap(),
                slot_type: SlotType::Number,
                multiplier: Some(11.66), // 1 tola = 11.66 grams
            },
        ];
        self.compiled_patterns.insert("gold_weight".to_string(), weight_patterns);

        // Phone patterns
        let phone_patterns = vec![
            CompiledSlotPattern {
                name: "indian".to_string(),
                regex: Regex::new(r"(?:\+91)?([6-9]\d{9})").unwrap(),
                slot_type: SlotType::Phone,
                multiplier: None,
            },
        ];
        self.compiled_patterns.insert("phone".to_string(), phone_patterns);

        // Current lender patterns
        let lender_patterns = vec![
            CompiledSlotPattern {
                name: "muthoot".to_string(),
                regex: Regex::new(r"(?i)\b(muthoot)\b").unwrap(),
                slot_type: SlotType::Text,
                multiplier: None,
            },
            CompiledSlotPattern {
                name: "manappuram".to_string(),
                regex: Regex::new(r"(?i)\b(manappuram)\b").unwrap(),
                slot_type: SlotType::Text,
                multiplier: None,
            },
            CompiledSlotPattern {
                name: "iifl".to_string(),
                regex: Regex::new(r"(?i)\b(iifl|ii\s*fl)\b").unwrap(),
                slot_type: SlotType::Text,
                multiplier: None,
            },
        ];
        self.compiled_patterns.insert("current_lender".to_string(), lender_patterns);

        // Gold purity patterns
        let purity_patterns = vec![
            CompiledSlotPattern {
                name: "karat".to_string(),
                regex: Regex::new(r"(?i)(22|24|18)\s*(?:k|karat|carat|kt)").unwrap(),
                slot_type: SlotType::Enum(vec!["18K".into(), "22K".into(), "24K".into()]),
                multiplier: None,
            },
        ];
        self.compiled_patterns.insert("gold_purity".to_string(), purity_patterns);

        // Location/City patterns
        let location_patterns = vec![
            CompiledSlotPattern {
                name: "city".to_string(),
                regex: Regex::new(r"(?i)\b(mumbai|delhi|bangalore|chennai|hyderabad|kolkata|pune|ahmedabad|jaipur)\b").unwrap(),
                slot_type: SlotType::Location,
                multiplier: None,
            },
        ];
        self.compiled_patterns.insert("location".to_string(), location_patterns);

        tracing::debug!("Compiled {} slot pattern groups", self.compiled_patterns.len());
    }

    /// Detect intent from text
    pub fn detect(&self, text: &str) -> DetectedIntent {
        let intents = self.intents.read();
        let text_lower = text.to_lowercase();

        let mut scores: Vec<(String, f32)> = intents
            .iter()
            .map(|intent| {
                let score = self.calculate_intent_score(&text_lower, intent);
                (intent.name.clone(), score)
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let (best_intent, best_score) = scores.first()
            .cloned()
            .unwrap_or(("unknown".to_string(), 0.0));

        // Extract slots
        let slots = self.extract_slots(text);

        DetectedIntent {
            intent: best_intent,
            confidence: best_score,
            slots,
            alternatives: scores.into_iter().skip(1).take(3).collect(),
        }
    }

    /// Calculate intent match score
    ///
    /// P2 FIX: Uses unicode_segmentation for proper Hindi/Devanagari word boundaries
    /// instead of split_whitespace() which doesn't handle Indian scripts correctly.
    fn calculate_intent_score(&self, text: &str, intent: &Intent) -> f32 {
        let mut score: f32 = 0.0;

        // Check examples
        for example in &intent.examples {
            let example_lower = example.to_lowercase();

            // Exact match
            if text == example_lower {
                return 1.0;
            }

            // Contains check
            if text.contains(&example_lower) {
                score = score.max(0.9);
            }

            // Word overlap - P2 FIX: Use Unicode word boundaries for Hindi/Devanagari support
            let example_words: std::collections::HashSet<&str> = example_lower
                .unicode_words()
                .collect();
            let text_words: std::collections::HashSet<&str> = text
                .unicode_words()
                .collect();

            let overlap = example_words.intersection(&text_words).count();
            if overlap > 0 {
                let overlap_score = overlap as f32 / example_words.len().max(1) as f32;
                score = score.max(overlap_score * 0.8);
            }
        }

        score
    }

    /// P0 FIX: Extract slots from text using compiled regex patterns
    ///
    /// Iterates through all pattern groups and extracts matching slots
    /// with proper type inference and confidence scoring.
    pub fn extract_slots(&self, text: &str) -> HashMap<String, Slot> {
        let mut slots = HashMap::new();

        for (slot_name, patterns) in &self.compiled_patterns {
            if let Some((value, slot_type, confidence)) = self.extract_slot_with_patterns(text, patterns) {
                slots.insert(slot_name.clone(), Slot {
                    name: slot_name.clone(),
                    slot_type,
                    value: Some(value),
                    confidence,
                });
            }
        }

        slots
    }

    /// P3 FIX: Convert all Indic script numerals to ASCII digits
    ///
    /// Supports all 11 major Indic scripts:
    /// - Devanagari (Hindi, Marathi, Sanskrit, Nepali, Konkani, Maithili, Dogri, Bodo)
    /// - Bengali/Assamese (Bengali, Assamese, Manipuri)
    /// - Tamil
    /// - Telugu
    /// - Gujarati
    /// - Kannada
    /// - Malayalam
    /// - Odia
    /// - Gurmukhi (Punjabi)
    /// - Ol Chiki (Santali)
    /// - Arabic-Indic (Urdu, Sindhi, Kashmiri)
    pub fn indic_numerals_to_ascii(s: &str) -> String {
        s.chars().map(|c| {
            match c {
                // Devanagari (U+0966 - U+096F)
                '०' => '0', '१' => '1', '२' => '2', '३' => '3', '४' => '4',
                '५' => '5', '६' => '6', '७' => '7', '८' => '8', '९' => '9',

                // Bengali/Assamese (U+09E6 - U+09EF)
                '০' => '0', '১' => '1', '২' => '2', '৩' => '3', '৪' => '4',
                '৫' => '5', '৬' => '6', '৭' => '7', '৮' => '8', '৯' => '9',

                // Tamil (U+0BE6 - U+0BEF)
                '௦' => '0', '௧' => '1', '௨' => '2', '௩' => '3', '௪' => '4',
                '௫' => '5', '௬' => '6', '௭' => '7', '௮' => '8', '௯' => '9',

                // Telugu (U+0C66 - U+0C6F)
                '౦' => '0', '౧' => '1', '౨' => '2', '౩' => '3', '౪' => '4',
                '౫' => '5', '౬' => '6', '౭' => '7', '౮' => '8', '౯' => '9',

                // Gujarati (U+0AE6 - U+0AEF)
                '૦' => '0', '૧' => '1', '૨' => '2', '૩' => '3', '૪' => '4',
                '૫' => '5', '૬' => '6', '૭' => '7', '૮' => '8', '૯' => '9',

                // Kannada (U+0CE6 - U+0CEF)
                '೦' => '0', '೧' => '1', '೨' => '2', '೩' => '3', '೪' => '4',
                '೫' => '5', '೬' => '6', '೭' => '7', '೮' => '8', '೯' => '9',

                // Malayalam (U+0D66 - U+0D6F)
                '൦' => '0', '൧' => '1', '൨' => '2', '൩' => '3', '൪' => '4',
                '൫' => '5', '൬' => '6', '൭' => '7', '൮' => '8', '൯' => '9',

                // Odia (U+0B66 - U+0B6F)
                '୦' => '0', '୧' => '1', '୨' => '2', '୩' => '3', '୪' => '4',
                '୫' => '5', '୬' => '6', '୭' => '7', '୮' => '8', '୯' => '9',

                // Gurmukhi/Punjabi (U+0A66 - U+0A6F)
                '੦' => '0', '੧' => '1', '੨' => '2', '੩' => '3', '੪' => '4',
                '੫' => '5', '੬' => '6', '੭' => '7', '੮' => '8', '੯' => '9',

                // Ol Chiki/Santali (U+1C50 - U+1C59)
                '᱐' => '0', '᱑' => '1', '᱒' => '2', '᱓' => '3', '᱔' => '4',
                '᱕' => '5', '᱖' => '6', '᱗' => '7', '᱘' => '8', '᱙' => '9',

                // Extended Arabic-Indic (U+06F0 - U+06F9) - Used in Urdu, Sindhi, Kashmiri
                '۰' => '0', '۱' => '1', '۲' => '2', '۳' => '3', '۴' => '4',
                '۵' => '5', '۶' => '6', '۷' => '7', '۸' => '8', '۹' => '9',

                // Pass through non-numeral characters
                _ => c,
            }
        }).collect()
    }

    /// P3 FIX: Check if a character is an Indic numeral
    pub fn is_indic_numeral(c: char) -> bool {
        matches!(c,
            // Devanagari
            '०'..='९' |
            // Bengali/Assamese
            '০'..='৯' |
            // Tamil
            '௦'..='௯' |
            // Telugu
            '౦'..='౯' |
            // Gujarati
            '૦'..='૯' |
            // Kannada
            '೦'..='೯' |
            // Malayalam
            '൦'..='൯' |
            // Odia
            '୦'..='୯' |
            // Gurmukhi
            '੦'..='੯' |
            // Ol Chiki
            '᱐'..='᱙' |
            // Extended Arabic-Indic
            '۰'..='۹'
        )
    }

    /// P1 FIX: Convert Devanagari numerals to ASCII (alias for indic_numerals_to_ascii)
    ///
    /// This is a convenience function for tests and backward compatibility.
    /// Use `indic_numerals_to_ascii` for the full multi-script implementation.
    pub fn devanagari_to_ascii(s: &str) -> String {
        Self::indic_numerals_to_ascii(s)
    }

    /// P0 FIX: Convert Hindi number word to numeric value
    fn hindi_word_to_number(word: &str) -> Option<f64> {
        match word {
            "एक" => Some(1.0),
            "दो" => Some(2.0),
            "तीन" => Some(3.0),
            "चार" => Some(4.0),
            "पांच" | "पाँच" => Some(5.0),
            "छह" | "छे" => Some(6.0),
            "सात" => Some(7.0),
            "आठ" => Some(8.0),
            "नौ" => Some(9.0),
            "दस" => Some(10.0),
            "बीस" => Some(20.0),
            "पच्चीस" => Some(25.0),
            "तीस" => Some(30.0),
            "पैंतीस" => Some(35.0),
            "चालीस" => Some(40.0),
            "पचास" => Some(50.0),
            "साठ" => Some(60.0),
            "सत्तर" => Some(70.0),
            "अस्सी" => Some(80.0),
            "नब्बे" => Some(90.0),
            "सौ" => Some(100.0),
            _ => None,
        }
    }

    /// P0 FIX: Extract slot value using compiled regex patterns
    ///
    /// Tries each pattern in order (highest priority first) and returns
    /// the first match with its computed value and confidence.
    ///
    /// P0 FIX (Dec 2025): Added support for:
    /// - Devanagari numerals (०-९) conversion
    /// - Hindi number words (पांच, दस, etc.)
    fn extract_slot_with_patterns(
        &self,
        text: &str,
        patterns: &[CompiledSlotPattern],
    ) -> Option<(String, SlotType, f32)> {
        for pattern in patterns {
            if let Some(captures) = pattern.regex.captures(text) {
                // Get the first capturing group (the value)
                if let Some(matched) = captures.get(1) {
                    let raw_value = matched.as_str();

                    // Compute final value based on multiplier
                    let value = if let Some(multiplier) = pattern.multiplier {
                        // P3 FIX: Handle all Indic script numerals and number words
                        let numeric_value = if pattern.name.starts_with("hindi_word") {
                            // Hindi number word - convert to numeric value
                            Self::hindi_word_to_number(raw_value).unwrap_or(1.0)
                        } else if raw_value.chars().any(Self::is_indic_numeral) {
                            // P3 FIX: Contains any Indic script numerals - convert to ASCII first
                            let ascii_value = Self::indic_numerals_to_ascii(raw_value);
                            ascii_value.replace(",", "").parse::<f64>().unwrap_or(0.0)
                        } else {
                            // Regular ASCII number
                            raw_value.replace(",", "").parse::<f64>().unwrap_or(0.0)
                        };

                        if numeric_value > 0.0 {
                            format!("{}", (numeric_value * multiplier) as i64)
                        } else {
                            raw_value.to_string()
                        }
                    } else {
                        // Remove commas for currency, keep as-is for others
                        match pattern.slot_type {
                            SlotType::Currency => {
                                // P3 FIX: Also convert all Indic script numerals for direct amounts
                                let converted = Self::indic_numerals_to_ascii(raw_value);
                                converted.replace(",", "")
                            }
                            SlotType::Text => {
                                // Capitalize lender names
                                let s = raw_value.to_lowercase();
                                if s.contains("muthoot") {
                                    "Muthoot".to_string()
                                } else if s.contains("manappuram") {
                                    "Manappuram".to_string()
                                } else if s.contains("iifl") {
                                    "IIFL".to_string()
                                } else {
                                    raw_value.to_string()
                                }
                            }
                            SlotType::Enum(_) => {
                                // Normalize karat values
                                format!("{}K", raw_value)
                            }
                            _ => raw_value.to_string(),
                        }
                    };

                    // Calculate confidence based on pattern specificity
                    // P3 FIX: All Indic language patterns get same high confidence
                    let confidence = match pattern.name.as_str() {
                        // English patterns
                        "crore" | "lakh" | "rs_amount" => 0.95,
                        // Hindi patterns
                        "hindi_crore_devanagari" | "hindi_crore_ascii" => 0.95,
                        "hindi_lakh_devanagari" | "hindi_lakh_ascii" => 0.95,
                        "hindi_word_lakh" | "hindi_word_crore" => 0.93,
                        "hindi_hazar_devanagari" | "hindi_hazar_ascii" => 0.90,
                        "hindi_rupees" => 0.92,
                        // P3 FIX: Other Indic language patterns
                        "tamil_lakh" | "tamil_crore" => 0.95,
                        "telugu_lakh" | "telugu_crore" => 0.95,
                        "bengali_lakh" | "bengali_crore" => 0.95,
                        "gujarati_lakh" | "gujarati_crore" => 0.95,
                        "kannada_lakh" | "kannada_crore" => 0.95,
                        "malayalam_lakh" | "malayalam_crore" => 0.95,
                        "marathi_crore" => 0.95,
                        "odia_lakh" | "odia_crore" => 0.95,
                        "punjabi_lakh" | "punjabi_crore" => 0.95,
                        // General patterns
                        "thousand" | "grams" | "karat" => 0.90,
                        "plain_number" => 0.70,
                        _ => 0.85,
                    };

                    return Some((value, pattern.slot_type.clone(), confidence));
                }
            }
        }

        None
    }

    /// Get intent by name
    pub fn get_intent(&self, name: &str) -> Option<Intent> {
        self.intents.read()
            .iter()
            .find(|i| i.name == name)
            .cloned()
    }

    /// List all intents
    pub fn list_intents(&self) -> Vec<String> {
        self.intents.read()
            .iter()
            .map(|i| i.name.clone())
            .collect()
    }
}

impl Default for IntentDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_detection() {
        let detector = IntentDetector::new();

        let result = detector.detect("I want a gold loan");
        assert_eq!(result.intent, "loan_inquiry");
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_interest_rate_intent() {
        let detector = IntentDetector::new();

        let result = detector.detect("What is the interest rate");
        assert_eq!(result.intent, "interest_rate");
    }

    #[test]
    fn test_slot_extraction() {
        let detector = IntentDetector::new();

        let slots = detector.extract_slots("I have a loan from Muthoot");
        assert!(slots.contains_key("current_lender"));
        assert_eq!(slots.get("current_lender").unwrap().value, Some("Muthoot".to_string()));
    }

    #[test]
    fn test_greeting() {
        let detector = IntentDetector::new();

        let result = detector.detect("Hello");
        assert_eq!(result.intent, "greeting");
    }

    #[test]
    fn test_loan_amount_extraction_lakh() {
        let detector = IntentDetector::new();

        let slots = detector.extract_slots("I need a loan of 5 lakh rupees");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("500000".to_string()));
    }

    #[test]
    fn test_loan_amount_extraction_crore() {
        let detector = IntentDetector::new();

        let slots = detector.extract_slots("I want 1.5 crore loan");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("15000000".to_string()));
    }

    #[test]
    fn test_gold_weight_extraction() {
        let detector = IntentDetector::new();

        let slots = detector.extract_slots("I have 50 grams of gold");
        assert!(slots.contains_key("gold_weight"));
        assert_eq!(slots.get("gold_weight").unwrap().value, Some("50".to_string()));
    }

    #[test]
    fn test_gold_purity_extraction() {
        let detector = IntentDetector::new();

        let slots = detector.extract_slots("It is 22 karat gold");
        assert!(slots.contains_key("gold_purity"));
        assert_eq!(slots.get("gold_purity").unwrap().value, Some("22K".to_string()));
    }

    #[test]
    fn test_phone_extraction() {
        let detector = IntentDetector::new();

        let slots = detector.extract_slots("My number is 9876543210");
        assert!(slots.contains_key("phone"));
        assert_eq!(slots.get("phone").unwrap().value, Some("9876543210".to_string()));
    }

    #[test]
    fn test_location_extraction() {
        let detector = IntentDetector::new();

        let slots = detector.extract_slots("I am from Mumbai");
        assert!(slots.contains_key("location"));
        // Regex captures as-is from text
        assert_eq!(slots.get("location").unwrap().value, Some("Mumbai".to_string()));
    }

    #[test]
    fn test_multiple_lenders() {
        let detector = IntentDetector::new();

        let slots1 = detector.extract_slots("I have a loan from Manappuram");
        assert_eq!(slots1.get("current_lender").unwrap().value, Some("Manappuram".to_string()));

        let slots2 = detector.extract_slots("My loan is with IIFL");
        assert_eq!(slots2.get("current_lender").unwrap().value, Some("IIFL".to_string()));
    }

    #[test]
    fn test_tola_to_grams() {
        let detector = IntentDetector::new();

        let slots = detector.extract_slots("I have 10 tola gold");
        assert!(slots.contains_key("gold_weight"));
        // 10 tola = 116.6 grams (truncated to 116)
        assert_eq!(slots.get("gold_weight").unwrap().value, Some("116".to_string()));
    }

    // P0 FIX: Hindi/Devanagari slot extraction tests

    #[test]
    fn test_hindi_lakh_with_ascii() {
        let detector = IntentDetector::new();

        let slots = detector.extract_slots("5 लाख का लोन चाहिए");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("500000".to_string()));
    }

    #[test]
    fn test_hindi_lakh_with_devanagari_numerals() {
        let detector = IntentDetector::new();

        let slots = detector.extract_slots("५ लाख रुपये का लोन");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("500000".to_string()));
    }

    #[test]
    fn test_hindi_word_lakh() {
        let detector = IntentDetector::new();

        let slots = detector.extract_slots("पांच लाख रुपये का लोन चाहिए");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("500000".to_string()));
    }

    #[test]
    fn test_hindi_crore() {
        let detector = IntentDetector::new();

        let slots = detector.extract_slots("एक करोड़ का लोन");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("10000000".to_string()));
    }

    #[test]
    fn test_hindi_hazar() {
        let detector = IntentDetector::new();

        let slots = detector.extract_slots("50 हज़ार रुपये");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("50000".to_string()));
    }

    #[test]
    fn test_devanagari_numeral_conversion() {
        assert_eq!(IntentDetector::devanagari_to_ascii("५०"), "50");
        assert_eq!(IntentDetector::devanagari_to_ascii("१२३४५"), "12345");
        assert_eq!(IntentDetector::devanagari_to_ascii("mixed १२ and 34"), "mixed 12 and 34");
    }

    #[test]
    fn test_hindi_number_word_conversion() {
        assert_eq!(IntentDetector::hindi_word_to_number("पांच"), Some(5.0));
        assert_eq!(IntentDetector::hindi_word_to_number("दस"), Some(10.0));
        assert_eq!(IntentDetector::hindi_word_to_number("बीस"), Some(20.0));
        assert_eq!(IntentDetector::hindi_word_to_number("पचास"), Some(50.0));
        assert_eq!(IntentDetector::hindi_word_to_number("unknown"), None);
    }

    // P3 FIX: Tests for all 11 Indic script numerals

    #[test]
    fn test_indic_numerals_to_ascii_all_scripts() {
        // Devanagari (Hindi, Marathi, Sanskrit, Nepali)
        assert_eq!(IntentDetector::indic_numerals_to_ascii("०१२३४५६७८९"), "0123456789");

        // Bengali/Assamese
        assert_eq!(IntentDetector::indic_numerals_to_ascii("০১২৩৪৫৬৭৮৯"), "0123456789");

        // Tamil
        assert_eq!(IntentDetector::indic_numerals_to_ascii("௦௧௨௩௪௫௬௭௮௯"), "0123456789");

        // Telugu
        assert_eq!(IntentDetector::indic_numerals_to_ascii("౦౧౨౩౪౫౬౭౮౯"), "0123456789");

        // Gujarati
        assert_eq!(IntentDetector::indic_numerals_to_ascii("૦૧૨૩૪૫૬૭૮૯"), "0123456789");

        // Kannada
        assert_eq!(IntentDetector::indic_numerals_to_ascii("೦೧೨೩೪೫೬೭೮೯"), "0123456789");

        // Malayalam
        assert_eq!(IntentDetector::indic_numerals_to_ascii("൦൧൨൩൪൫൬൭൮൯"), "0123456789");

        // Odia
        assert_eq!(IntentDetector::indic_numerals_to_ascii("୦୧୨୩୪୫୬୭୮୯"), "0123456789");

        // Gurmukhi (Punjabi)
        assert_eq!(IntentDetector::indic_numerals_to_ascii("੦੧੨੩੪੫੬੭੮੯"), "0123456789");

        // Ol Chiki (Santali)
        assert_eq!(IntentDetector::indic_numerals_to_ascii("᱐᱑᱒᱓᱔᱕᱖᱗᱘᱙"), "0123456789");

        // Extended Arabic-Indic (Urdu, Sindhi, Kashmiri)
        assert_eq!(IntentDetector::indic_numerals_to_ascii("۰۱۲۳۴۵۶۷۸۹"), "0123456789");
    }

    #[test]
    fn test_indic_numerals_mixed_text() {
        // Mix of scripts should all convert
        assert_eq!(IntentDetector::indic_numerals_to_ascii("Amount: ५०"), "Amount: 50");
        assert_eq!(IntentDetector::indic_numerals_to_ascii("Price ৫০ rupees"), "Price 50 rupees");
        assert_eq!(IntentDetector::indic_numerals_to_ascii("கோடி ௫"), "கோடி 5");
    }

    #[test]
    fn test_is_indic_numeral() {
        // Test Devanagari
        assert!(IntentDetector::is_indic_numeral('५'));
        assert!(IntentDetector::is_indic_numeral('०'));

        // Test Bengali
        assert!(IntentDetector::is_indic_numeral('৫'));

        // Test Tamil
        assert!(IntentDetector::is_indic_numeral('௫'));

        // Test Telugu
        assert!(IntentDetector::is_indic_numeral('౫'));

        // Test that ASCII is not considered Indic
        assert!(!IntentDetector::is_indic_numeral('5'));
        assert!(!IntentDetector::is_indic_numeral('0'));

        // Test non-numerals
        assert!(!IntentDetector::is_indic_numeral('a'));
        assert!(!IntentDetector::is_indic_numeral('क'));
    }

    // P3 FIX: Tests for other Indic language multiplier patterns

    #[test]
    fn test_tamil_lakh_extraction() {
        let detector = IntentDetector::new();
        let slots = detector.extract_slots("5 லட்சம் கடன்");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("500000".to_string()));
    }

    #[test]
    fn test_telugu_lakh_extraction() {
        let detector = IntentDetector::new();
        let slots = detector.extract_slots("5 లక్ష రూపాయలు");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("500000".to_string()));
    }

    #[test]
    fn test_bengali_lakh_extraction() {
        let detector = IntentDetector::new();
        let slots = detector.extract_slots("৫ লাখ টাকা");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("500000".to_string()));
    }

    #[test]
    fn test_gujarati_lakh_extraction() {
        let detector = IntentDetector::new();
        let slots = detector.extract_slots("૫ લાખ રૂપિયા");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("500000".to_string()));
    }

    #[test]
    fn test_kannada_lakh_extraction() {
        let detector = IntentDetector::new();
        let slots = detector.extract_slots("5 ಲಕ್ಷ ರೂಪಾಯಿ");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("500000".to_string()));
    }

    #[test]
    fn test_malayalam_lakh_extraction() {
        let detector = IntentDetector::new();
        let slots = detector.extract_slots("5 ലക്ഷം രൂപ");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("500000".to_string()));
    }

    #[test]
    fn test_odia_lakh_extraction() {
        let detector = IntentDetector::new();
        let slots = detector.extract_slots("5 ଲକ୍ଷ ଟଙ୍କା");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("500000".to_string()));
    }

    #[test]
    fn test_punjabi_lakh_extraction() {
        let detector = IntentDetector::new();
        let slots = detector.extract_slots("5 ਲੱਖ ਰੁਪਏ");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("500000".to_string()));
    }

    #[test]
    fn test_bengali_crore_extraction() {
        let detector = IntentDetector::new();
        let slots = detector.extract_slots("১ কোটি টাকা");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("10000000".to_string()));
    }

    #[test]
    fn test_telugu_native_numerals() {
        let detector = IntentDetector::new();
        // 5 in Telugu script with lakh
        let slots = detector.extract_slots("౫ లక్ష");
        assert!(slots.contains_key("loan_amount"));
        assert_eq!(slots.get("loan_amount").unwrap().value, Some("500000".to_string()));
    }
}
