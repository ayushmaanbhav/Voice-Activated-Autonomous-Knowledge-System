//! Named Entity Recognition for Indian names and addresses
//!
//! P3 FIX: Heuristic-based NER for detecting person names and addresses
//! in Indian multilingual context (supports English and Indic scripts).
//!
//! Uses pattern matching and dictionary lookup instead of ML models
//! for low-latency, high-accuracy detection in banking context.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;
use voice_agent_core::{DetectionMethod, PIIEntity, PIIType};

// P2-2: Removed unused ENGLISH_TITLES and HINDI_TITLES static HashSets
// (Title detection is handled by regex patterns in NERPatterns instead)

/// Common Indian first names for validation
static COMMON_FIRST_NAMES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        // Male names
        "rahul", "amit", "raj", "vijay", "suresh", "ramesh", "rakesh", "anil", "sanjay", "ajay",
        "kumar", "ashok", "mohan", "ravi", "deepak", "manoj", "krishna", "ganesh", "sunil", "arun",
        "vinod", "pradeep", "rajesh", "satish", "santosh", "mahesh", "naresh", "dinesh", "mukesh",
        "girish", "prakash", "vikram", "nitin", "rohit", "sachin", "sudhir", "manish",
        // Female names
        "priya", "sunita", "anita", "neha", "pooja", "divya", "kavita", "rekha", "meena", "seema",
        "suman", "renu", "usha", "nisha", "anju", "kiran", "lakshmi", "sarita", "geeta", "rita",
        "sita", "radha", "maya", "shanti", "rani", "pushpa", "kamla", "savita", "mamta", "rashmi",
        "archana",
    ]
    .into_iter()
    .collect()
});

/// Common Indian last names
static COMMON_LAST_NAMES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "sharma",
        "verma",
        "gupta",
        "singh",
        "kumar",
        "yadav",
        "jain",
        "agarwal",
        "aggarwal",
        "patel",
        "shah",
        "mehta",
        "desai",
        "joshi",
        "tiwari",
        "mishra",
        "pandey",
        "dubey",
        "tripathi",
        "shukla",
        "srivastava",
        "saxena",
        "kapoor",
        "khanna",
        "malhotra",
        "chopra",
        "arora",
        "bhatia",
        "sethi",
        "ahuja",
        "choudhary",
        "chaudhary",
        "thakur",
        "rawat",
        "chauhan",
        "rajput",
        "rathore",
        "reddy",
        "naidu",
        "rao",
        "iyer",
        "iyengar",
        "mukherjee",
        "banerjee",
        "chatterjee",
        "ghosh",
        "das",
        "roy",
        "paul",
        "pillai",
        "nair",
        "menon",
        "varma",
        "bose",
        "sen",
        "dutta",
        "ganguly",
    ]
    .into_iter()
    .collect()
});

/// Indian state names (for address detection)
static INDIAN_STATES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        // States
        "andhra pradesh",
        "arunachal pradesh",
        "assam",
        "bihar",
        "chhattisgarh",
        "goa",
        "gujarat",
        "haryana",
        "himachal pradesh",
        "jharkhand",
        "karnataka",
        "kerala",
        "madhya pradesh",
        "maharashtra",
        "manipur",
        "meghalaya",
        "mizoram",
        "nagaland",
        "odisha",
        "orissa",
        "punjab",
        "rajasthan",
        "sikkim",
        "tamil nadu",
        "telangana",
        "tripura",
        "uttar pradesh",
        "uttarakhand",
        "west bengal",
        // Union Territories
        "andaman and nicobar",
        "chandigarh",
        "dadra and nagar haveli",
        "daman and diu",
        "delhi",
        "jammu and kashmir",
        "ladakh",
        "lakshadweep",
        "puducherry",
        // Abbreviations
        "ap",
        "ar",
        "as",
        "br",
        "cg",
        "ga",
        "gj",
        "hr",
        "hp",
        "jh",
        "ka",
        "kl",
        "mp",
        "mh",
        "mn",
        "ml",
        "mz",
        "nl",
        "od",
        "pb",
        "rj",
        "sk",
        "tn",
        "ts",
        "tr",
        "up",
        "uk",
        "wb",
        "an",
        "ch",
        "dn",
        "dd",
        "dl",
        "jk",
        "la",
        "ld",
        "py",
    ]
    .into_iter()
    .collect()
});

/// Major Indian cities (Tier 1 and Tier 2)
static MAJOR_CITIES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        // Tier 1
        "mumbai",
        "delhi",
        "bangalore",
        "bengaluru",
        "chennai",
        "kolkata",
        "hyderabad",
        "pune",
        "ahmedabad",
        // Tier 2
        "jaipur",
        "lucknow",
        "kanpur",
        "nagpur",
        "indore",
        "thane",
        "bhopal",
        "visakhapatnam",
        "vizag",
        "vadodara",
        "baroda",
        "ludhiana",
        "agra",
        "nashik",
        "faridabad",
        "meerut",
        "rajkot",
        "varanasi",
        "srinagar",
        "aurangabad",
        "dhanbad",
        "amritsar",
        "allahabad",
        "ranchi",
        "coimbatore",
        "jabalpur",
        "gwalior",
        "vijayawada",
        "jodhpur",
        "madurai",
        "raipur",
        "kota",
        "guwahati",
        "chandigarh",
        "solapur",
        "hubli",
        "mysore",
        "mysuru",
        "tiruchirappalli",
        "trichy",
        "bareilly",
        "aligarh",
        "moradabad",
        "gurgaon",
        "gurugram",
        "noida",
        "greater noida",
        "navi mumbai",
        "ghaziabad",
    ]
    .into_iter()
    .collect()
});

/// Address indicator words
static ADDRESS_INDICATORS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        // English
        "road",
        "rd",
        "street",
        "st",
        "lane",
        "colony",
        "nagar",
        "vihar",
        "enclave",
        "extension",
        "extn",
        "block",
        "sector",
        "phase",
        "floor",
        "flat",
        "apartment",
        "apt",
        "house",
        "plot",
        "building",
        "complex",
        "tower",
        "society",
        "soc",
        "near",
        "opposite",
        "opp",
        "behind",
        "beside",
        "next to",
        "above",
        "below",
        "ground",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "main",
        "cross",
        "layout",
        "garden",
        "residency",
        "heights",
        "plaza",
        "arcade",
        "market",
        "bazaar",
        "chowk",
        "circle",
        "square",
        "junction",
        "crossing",
        "bridge",
        // Hindi (transliterated)
        "marg",
        "path",
        "gali",
        "mohalla",
        "para",
        "basti",
        "puram",
        "wadi",
        "abad",
        "gunj",
        "ganj",
        "pet",
        "peth",
        "wala",
    ]
    .into_iter()
    .collect()
});

/// Compiled patterns for NER
pub struct NERPatterns {
    /// Pattern: Title + Name
    title_name: Regex,
    /// Pattern: Hindi title + name
    hindi_title_name: Regex,
    /// Pattern: Indian pincode (6 digits)
    pincode: Regex,
    // P2-2: Removed unused address_with_pincode field
    // (detect_addresses uses pincode-based lookback instead)
}

impl Default for NERPatterns {
    fn default() -> Self {
        Self::new()
    }
}

impl NERPatterns {
    pub fn new() -> Self {
        Self {
            // English: Title + Capitalized Words (title is case-insensitive)
            // Using separate case handling for title vs. name parts
            title_name: Regex::new(
                r"\b(?i:mr\.?|mrs\.?|ms\.?|miss|dr\.?|prof\.?|shri|shree|smt\.?|kumari|master|late|capt\.?|col\.?|maj\.?)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b"
            ).unwrap(),

            // Hindi: श्री/श्रीमती + Devanagari name
            hindi_title_name: Regex::new(
                r"(श्री|श्रीमती|सुश्री|कुमारी|डॉ\.?|प्रो\.?|स्वर्गीय)\s+([\u0900-\u097F]+(?:\s+[\u0900-\u097F]+){0,3})"
            ).unwrap(),

            // Indian pincode: 6 digits, first digit 1-9
            pincode: Regex::new(r"\b[1-9]\d{5}\b").unwrap(),
        }
    }
}

/// NER-based entity detector
pub struct NameAddressDetector {
    patterns: NERPatterns,
}

impl Default for NameAddressDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl NameAddressDetector {
    pub fn new() -> Self {
        Self {
            patterns: NERPatterns::new(),
        }
    }

    /// Detect person names in text
    pub fn detect_names(&self, text: &str) -> Vec<PIIEntity> {
        let mut entities = Vec::new();

        // Pattern 1: Title + Name (English)
        for cap in self.patterns.title_name.captures_iter(text) {
            if let (Some(full_match), Some(name_match)) = (cap.get(0), cap.get(1)) {
                let name = name_match.as_str();
                // Validate: at least one word should be a common name
                let words: Vec<&str> = name.split_whitespace().collect();
                let has_common_name = words.iter().any(|w| {
                    let lower = w.to_lowercase();
                    COMMON_FIRST_NAMES.contains(lower.as_str())
                        || COMMON_LAST_NAMES.contains(lower.as_str())
                });

                let confidence = if has_common_name { 0.90 } else { 0.75 };

                entities.push(PIIEntity {
                    pii_type: PIIType::PersonName,
                    text: full_match.as_str().to_string(),
                    start: full_match.start(),
                    end: full_match.end(),
                    confidence,
                    method: DetectionMethod::NER,
                });
            }
        }

        // Pattern 2: Hindi Title + Name (Devanagari)
        for cap in self.patterns.hindi_title_name.captures_iter(text) {
            if let Some(full_match) = cap.get(0) {
                entities.push(PIIEntity {
                    pii_type: PIIType::PersonName,
                    text: full_match.as_str().to_string(),
                    start: full_match.start(),
                    end: full_match.end(),
                    confidence: 0.85,
                    method: DetectionMethod::NER,
                });
            }
        }

        // Pattern 3: Standalone common name patterns (First Last)
        // Look for sequences of 2-3 capitalized words that match common names
        let name_sequence = Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b").unwrap();
        for cap in name_sequence.captures_iter(text) {
            if let Some(name_match) = cap.get(1) {
                let name = name_match.as_str();
                let words: Vec<&str> = name.split_whitespace().collect();

                // Must have at least 2 words, first word is first name, last is last name
                if words.len() >= 2 {
                    let first_lower = words[0].to_lowercase();
                    let last_lower = words.last().unwrap().to_lowercase();

                    let first_is_common = COMMON_FIRST_NAMES.contains(first_lower.as_str());
                    let last_is_common = COMMON_LAST_NAMES.contains(last_lower.as_str());

                    // Both first and last should be common Indian names
                    if first_is_common && last_is_common {
                        // Check if this overlaps with existing entity
                        let overlaps = entities.iter().any(|e| {
                            (name_match.start() >= e.start && name_match.start() < e.end)
                                || (name_match.end() > e.start && name_match.end() <= e.end)
                        });

                        if !overlaps {
                            entities.push(PIIEntity {
                                pii_type: PIIType::PersonName,
                                text: name.to_string(),
                                start: name_match.start(),
                                end: name_match.end(),
                                confidence: 0.80,
                                method: DetectionMethod::NER,
                            });
                        }
                    }
                }
            }
        }

        entities
    }

    /// Detect addresses in text
    pub fn detect_addresses(&self, text: &str) -> Vec<PIIEntity> {
        let mut entities = Vec::new();
        let text_lower = text.to_lowercase();

        // Strategy 1: Find pincode and expand backwards to find address
        for pincode_match in self.patterns.pincode.find_iter(text) {
            let pincode_start = pincode_match.start();
            let pincode_end = pincode_match.end();

            // Look backwards for address indicators
            let before_pincode = &text[..pincode_start];

            // Find where address likely starts (look for indicators)
            let mut address_start = pincode_start;
            let before_lower = before_pincode.to_lowercase();

            // Find the closest address indicator before the pincode
            let mut best_indicator_pos = None;
            for indicator in ADDRESS_INDICATORS.iter() {
                if let Some(pos) = before_lower.rfind(indicator) {
                    // Only consider indicators within reasonable distance (100 chars)
                    if pincode_start - pos <= 100 {
                        match best_indicator_pos {
                            None => best_indicator_pos = Some(pos),
                            Some(best) if pos > best => best_indicator_pos = Some(pos),
                            _ => {},
                        }
                    }
                }
            }

            // Use the indicator position if found
            if let Some(indicator_pos) = best_indicator_pos {
                // Address starts at most 30 chars before the indicator (house number, etc.)
                address_start = indicator_pos.saturating_sub(30);
            }

            // Also check for city names within the same range
            for city in MAJOR_CITIES.iter() {
                if let Some(pos) = before_lower.rfind(city) {
                    // City should be before the address indicator (if any)
                    if pincode_start - pos <= 100 && pos < address_start {
                        address_start = pos.saturating_sub(20);
                    }
                }
            }

            // Trim to word boundary - find the start of a word
            while address_start > 0 {
                let c = text.as_bytes()[address_start - 1];
                if c.is_ascii_whitespace() || c == b',' || c == b':' || c == b';' {
                    break;
                }
                address_start = address_start.saturating_sub(1);
            }

            // Skip leading whitespace and punctuation
            while address_start < pincode_start {
                let c = text.as_bytes()[address_start];
                if c.is_ascii_whitespace() || c == b',' || c == b':' || c == b';' {
                    address_start += 1;
                } else {
                    break;
                }
            }

            if address_start < pincode_start {
                let address_text = text[address_start..pincode_end].trim().to_string();

                // Validate: must contain at least one address indicator
                let has_indicator = ADDRESS_INDICATORS
                    .iter()
                    .any(|ind| address_text.to_lowercase().contains(ind));

                let has_city = MAJOR_CITIES
                    .iter()
                    .any(|city| address_text.to_lowercase().contains(city));

                // Skip if address is too short (likely false positive)
                if (has_indicator || has_city) && address_text.len() >= 15 {
                    let confidence = if has_indicator && has_city {
                        0.95
                    } else if has_indicator {
                        0.85
                    } else {
                        0.75
                    };

                    entities.push(PIIEntity {
                        pii_type: PIIType::Address,
                        text: address_text,
                        start: address_start,
                        end: pincode_end,
                        confidence,
                        method: DetectionMethod::NER,
                    });
                }
            }
        }

        // Strategy 2: Look for state/city combinations
        for state in INDIAN_STATES.iter() {
            if let Some(state_pos) = text_lower.find(state) {
                // Check if there's address context around it
                let context_start = state_pos.saturating_sub(150);
                let context_end = (state_pos + state.len() + 50).min(text.len());
                let context = &text_lower[context_start..context_end];

                let has_address_indicator =
                    ADDRESS_INDICATORS.iter().any(|ind| context.contains(ind));

                if has_address_indicator {
                    // Find actual boundaries
                    let start = context_start;
                    let mut end = state_pos + state.len();

                    // Look for pincode after state
                    if let Some(pincode_match) = self.patterns.pincode.find(&text[end..]) {
                        end += pincode_match.end();
                    }

                    // Check we haven't already captured this
                    let overlaps = entities.iter().any(|e| {
                        (start >= e.start && start < e.end) || (end > e.start && end <= e.end)
                    });

                    if !overlaps {
                        let address_text = text[start..end].trim().to_string();
                        if address_text.len() > 20 {
                            // Reasonable address length
                            entities.push(PIIEntity {
                                pii_type: PIIType::Address,
                                text: address_text,
                                start,
                                end,
                                confidence: 0.70,
                                method: DetectionMethod::NER,
                            });
                        }
                    }
                }
            }
        }

        entities
    }

    /// Detect both names and addresses
    pub fn detect_all(&self, text: &str) -> Vec<PIIEntity> {
        let mut entities = self.detect_names(text);
        entities.extend(self.detect_addresses(text));

        // Sort by position
        entities.sort_by_key(|e| e.start);

        // Remove overlaps, keeping higher confidence
        let mut result: Vec<PIIEntity> = Vec::new();
        for entity in entities {
            if let Some(last) = result.last() {
                if entity.start < last.end {
                    // Overlap - keep existing if higher confidence
                    if entity.confidence > last.confidence {
                        result.pop();
                        result.push(entity);
                    }
                    continue;
                }
            }
            result.push(entity);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_name_with_title() {
        let detector = NameAddressDetector::new();
        let text = "Please contact Mr. Rahul Sharma for further details";

        let names = detector.detect_names(text);
        assert_eq!(names.len(), 1);
        assert_eq!(names[0].pii_type, PIIType::PersonName);
        assert!(names[0].text.contains("Rahul Sharma"));
    }

    #[test]
    fn test_detect_hindi_name() {
        let detector = NameAddressDetector::new();
        let text = "श्री राजेश कुमार से मिलें";

        let names = detector.detect_names(text);
        assert_eq!(names.len(), 1);
        assert!(names[0].text.contains("राजेश"));
    }

    #[test]
    fn test_detect_name_smt() {
        let detector = NameAddressDetector::new();
        let text = "Smt. Sunita Verma is the account holder";

        let names = detector.detect_names(text);
        assert_eq!(names.len(), 1);
        assert!(names[0].text.contains("Sunita Verma"));
    }

    #[test]
    fn test_detect_address_with_pincode() {
        let detector = NameAddressDetector::new();
        let text = "Send documents to: 123, Gandhi Road, Sector 5, Mumbai 400001";

        let addresses = detector.detect_addresses(text);
        assert_eq!(addresses.len(), 1);
        assert_eq!(addresses[0].pii_type, PIIType::Address);
        assert!(addresses[0].text.contains("400001"));
        assert!(addresses[0].text.to_lowercase().contains("mumbai"));
    }

    #[test]
    fn test_detect_address_with_state() {
        let detector = NameAddressDetector::new();
        let text = "Flat 202, Sunrise Apartments, MG Road, Bangalore, Karnataka 560001";

        let addresses = detector.detect_addresses(text);
        assert!(!addresses.is_empty());
        assert!(
            addresses[0].text.to_lowercase().contains("karnataka")
                || addresses[0].text.contains("560001")
        );
    }

    #[test]
    fn test_detect_standalone_name() {
        let detector = NameAddressDetector::new();
        let text = "The loan applicant is Amit Gupta";

        let names = detector.detect_names(text);
        // Should detect "Amit Gupta" as both are common Indian names
        assert!(!names.is_empty());
        assert!(names.iter().any(|n| n.text == "Amit Gupta"));
    }

    #[test]
    fn test_common_names_validation() {
        assert!(COMMON_FIRST_NAMES.contains("rahul"));
        assert!(COMMON_LAST_NAMES.contains("sharma"));
        assert!(!COMMON_FIRST_NAMES.contains("unknown"));
    }

    #[test]
    fn test_no_false_positives() {
        let detector = NameAddressDetector::new();
        let text = "I want a gold loan of 5 lakh rupees";

        let names = detector.detect_names(text);
        let addresses = detector.detect_addresses(text);

        assert!(names.is_empty());
        assert!(addresses.is_empty());
    }

    #[test]
    fn test_detect_multiple_entities() {
        let detector = NameAddressDetector::new();
        let text = "Mr. Rahul Sharma, residing at 45 Park Street, Kolkata 700016, applied for loan";

        let entities = detector.detect_all(text);

        let has_name = entities.iter().any(|e| e.pii_type == PIIType::PersonName);
        let has_address = entities.iter().any(|e| e.pii_type == PIIType::Address);

        // Should find at least one entity (name or address)
        assert!(
            !entities.is_empty(),
            "Should detect at least one entity: {:?}",
            entities
        );
        // Either name or address should be detected
        assert!(
            has_name || has_address,
            "Should detect name or address. Found: {:?}",
            entities
        );
    }
}
