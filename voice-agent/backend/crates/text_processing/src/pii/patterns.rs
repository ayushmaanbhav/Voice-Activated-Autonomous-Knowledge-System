//! India-specific PII regex patterns

use regex::Regex;
use voice_agent_core::PIIType;
use std::collections::HashMap;
use once_cell::sync::Lazy;

/// Compiled PII patterns for India
pub static INDIAN_PII_PATTERNS: Lazy<HashMap<PIIType, CompiledPattern>> = Lazy::new(|| {
    let mut patterns = HashMap::new();

    // Aadhaar: 12 digits, often with spaces (XXXX XXXX XXXX)
    // First digit cannot be 0 or 1
    patterns.insert(
        PIIType::Aadhaar,
        CompiledPattern {
            regex: Regex::new(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b").unwrap(),
            description: "Aadhaar number (12 digits)",
            confidence: 0.95,
        },
    );

    // PAN: 5 letters, 4 digits, 1 letter (ABCDE1234F)
    // 4th char indicates holder type: P=Person, C=Company, etc.
    patterns.insert(
        PIIType::PAN,
        CompiledPattern {
            regex: Regex::new(r"\b[A-Z]{3}[ABCFGHLJPT][A-Z][0-9]{4}[A-Z]\b").unwrap(),
            description: "PAN card number",
            confidence: 0.98,
        },
    );

    // Indian phone: +91 or 0 followed by 10 digits starting with 6-9
    patterns.insert(
        PIIType::PhoneNumber,
        CompiledPattern {
            regex: Regex::new(r"(?:\+91[\-\s]?)?(?:0)?[6-9]\d{9}\b").unwrap(),
            description: "Indian phone number",
            confidence: 0.90,
        },
    );

    // Email
    patterns.insert(
        PIIType::Email,
        CompiledPattern {
            regex: Regex::new(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b").unwrap(),
            description: "Email address",
            confidence: 0.95,
        },
    );

    // IFSC: 4 letters, 0, 6 alphanumeric (SBIN0001234)
    patterns.insert(
        PIIType::IFSC,
        CompiledPattern {
            regex: Regex::new(r"\b[A-Z]{4}0[A-Z0-9]{6}\b").unwrap(),
            description: "IFSC code",
            confidence: 0.98,
        },
    );

    // Bank Account: 9-18 digits (varies by bank)
    // More restrictive to avoid false positives
    patterns.insert(
        PIIType::BankAccount,
        CompiledPattern {
            regex: Regex::new(r"\b\d{9,18}\b").unwrap(),
            description: "Bank account number",
            confidence: 0.60, // Lower confidence due to false positives
        },
    );

    // Voter ID (EPIC): 3 letters followed by 7 digits
    patterns.insert(
        PIIType::VoterId,
        CompiledPattern {
            regex: Regex::new(r"\b[A-Z]{3}\d{7}\b").unwrap(),
            description: "Voter ID (EPIC)",
            confidence: 0.85,
        },
    );

    // Driving License: State code (2 letters) + 13-15 alphanumeric
    // Format varies by state
    patterns.insert(
        PIIType::DrivingLicense,
        CompiledPattern {
            regex: Regex::new(r"\b[A-Z]{2}[\-\s]?\d{2}[\-\s]?\d{4}[\-\s]?\d{7}\b").unwrap(),
            description: "Driving license number",
            confidence: 0.80,
        },
    );

    // Passport: Letter + 7 digits (Indian passport)
    patterns.insert(
        PIIType::Passport,
        CompiledPattern {
            regex: Regex::new(r"\b[A-Z][0-9]{7}\b").unwrap(),
            description: "Indian passport number",
            confidence: 0.85,
        },
    );

    // UPI ID: name@upihandle
    patterns.insert(
        PIIType::UpiId,
        CompiledPattern {
            regex: Regex::new(r"\b[a-zA-Z0-9._-]+@[a-zA-Z]+\b").unwrap(),
            description: "UPI ID",
            confidence: 0.70, // Can have false positives with emails
        },
    );

    // Credit/Debit Card: 16 digits, often with spaces/dashes
    patterns.insert(
        PIIType::CardNumber,
        CompiledPattern {
            regex: Regex::new(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b").unwrap(),
            description: "Credit/Debit card number",
            confidence: 0.90,
        },
    );

    // GSTIN: 15 character alphanumeric
    patterns.insert(
        PIIType::GSTIN,
        CompiledPattern {
            regex: Regex::new(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b").unwrap(),
            description: "GSTIN number",
            confidence: 0.95,
        },
    );

    patterns
});

/// Compiled regex pattern with metadata
#[derive(Debug, Clone)]
pub struct CompiledPattern {
    pub regex: Regex,
    pub description: &'static str,
    pub confidence: f32,
}

/// Indian PII patterns helper
#[derive(Debug, Clone, Default)]
pub struct IndianPIIPatterns;

impl IndianPIIPatterns {
    /// Get pattern for a PII type
    pub fn get(pii_type: PIIType) -> Option<&'static CompiledPattern> {
        INDIAN_PII_PATTERNS.get(&pii_type)
    }

    /// Find all matches of a PII type in text
    pub fn find_matches(text: &str, pii_type: PIIType) -> Vec<PatternMatch> {
        if let Some(pattern) = INDIAN_PII_PATTERNS.get(&pii_type) {
            pattern.regex
                .find_iter(text)
                .map(|m| PatternMatch {
                    pii_type,
                    text: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                    confidence: pattern.confidence,
                })
                .collect()
        } else {
            vec![]
        }
    }

    /// Find all PII in text
    pub fn find_all(text: &str, types: &[PIIType]) -> Vec<PatternMatch> {
        let mut matches = Vec::new();

        for pii_type in types {
            matches.extend(Self::find_matches(text, *pii_type));
        }

        // Sort by start position
        matches.sort_by_key(|m| m.start);
        matches
    }

    /// P3 FIX: Validate Aadhaar checksum using Verhoeff algorithm
    ///
    /// Aadhaar uses the Verhoeff algorithm for check digit validation.
    /// The algorithm uses three tables: multiplication (d), permutation (p), and inverse (inv).
    /// Reference: https://en.wikibooks.org/wiki/Algorithm_Implementation/Checksums/Verhoeff_Algorithm
    pub fn validate_aadhaar(number: &str) -> bool {
        let digits: Vec<u32> = number
            .chars()
            .filter(|c| c.is_ascii_digit())
            .filter_map(|c| c.to_digit(10))
            .collect();

        if digits.len() != 12 {
            return false;
        }

        // First digit cannot be 0 or 1
        if digits[0] < 2 {
            return false;
        }

        // Verhoeff multiplication table
        #[rustfmt::skip]
        const D: [[u32; 10]; 10] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
            [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
            [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
            [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
            [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
            [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
            [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
            [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ];

        // Verhoeff permutation table
        #[rustfmt::skip]
        const P: [[u32; 10]; 8] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
            [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
            [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
            [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
            [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
            [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
            [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
        ];

        // Compute checksum
        let mut c: u32 = 0;
        for (i, &digit) in digits.iter().rev().enumerate() {
            let p_index = i % 8;
            let p_value = P[p_index][digit as usize];
            c = D[c as usize][p_value as usize];
        }

        // Valid if checksum is 0
        c == 0
    }

    /// Generate Verhoeff check digit for Aadhaar
    ///
    /// Given the first 11 digits, returns the check digit (12th digit)
    #[allow(dead_code)]
    pub fn generate_aadhaar_checksum(digits_11: &[u32]) -> Option<u32> {
        if digits_11.len() != 11 {
            return None;
        }

        // Verhoeff multiplication table
        #[rustfmt::skip]
        const D: [[u32; 10]; 10] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
            [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
            [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
            [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
            [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
            [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
            [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
            [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ];

        // Verhoeff permutation table
        #[rustfmt::skip]
        const P: [[u32; 10]; 8] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
            [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
            [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
            [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
            [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
            [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
            [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
        ];

        // Verhoeff inverse table
        #[rustfmt::skip]
        const INV: [u32; 10] = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9];

        // Compute intermediate checksum for 11 digits
        let mut c: u32 = 0;
        for (i, &digit) in digits_11.iter().rev().enumerate() {
            // Position is i+1 because we're computing for position 12 (check digit)
            let p_index = (i + 1) % 8;
            let p_value = P[p_index][digit as usize];
            c = D[c as usize][p_value as usize];
        }

        // The check digit is the inverse of c
        Some(INV[c as usize])
    }

    /// Validate PAN format
    pub fn validate_pan(pan: &str) -> bool {
        let pan = pan.to_uppercase();
        if pan.len() != 10 {
            return false;
        }

        // First 3 characters: letters
        // 4th character: holder type (P, C, H, A, B, G, J, L, F, T)
        // 5th character: first letter of last name
        // 6-9: digits
        // 10: check letter

        let chars: Vec<char> = pan.chars().collect();

        chars[0..3].iter().all(|c| c.is_ascii_uppercase())
            && ['P', 'C', 'H', 'A', 'B', 'G', 'J', 'L', 'F', 'T'].contains(&chars[3])
            && chars[4].is_ascii_uppercase()
            && chars[5..9].iter().all(|c| c.is_ascii_digit())
            && chars[9].is_ascii_uppercase()
    }
}

/// Pattern match result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pii_type: PIIType,
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aadhaar_pattern() {
        let text = "My Aadhaar is 2345 6789 0123";
        let matches = IndianPIIPatterns::find_matches(text, PIIType::Aadhaar);
        assert_eq!(matches.len(), 1);
        assert!(matches[0].text.contains("2345"));
    }

    #[test]
    fn test_pan_pattern() {
        let text = "My PAN is ABCPD1234E";
        let matches = IndianPIIPatterns::find_matches(text, PIIType::PAN);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].text, "ABCPD1234E");
    }

    #[test]
    fn test_pan_validation() {
        assert!(IndianPIIPatterns::validate_pan("ABCPD1234E"));
        assert!(!IndianPIIPatterns::validate_pan("12345")); // Too short
        assert!(!IndianPIIPatterns::validate_pan("ABCXD1234E")); // Invalid 4th char
    }

    #[test]
    fn test_phone_pattern() {
        let text = "Call me at +91 9876543210 or 9123456789";
        let matches = IndianPIIPatterns::find_matches(text, PIIType::PhoneNumber);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_email_pattern() {
        let text = "Email: user@example.com";
        let matches = IndianPIIPatterns::find_matches(text, PIIType::Email);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].text, "user@example.com");
    }

    #[test]
    fn test_ifsc_pattern() {
        let text = "IFSC: SBIN0001234";
        let matches = IndianPIIPatterns::find_matches(text, PIIType::IFSC);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].text, "SBIN0001234");
    }

    #[test]
    fn test_find_all() {
        let text = "PAN: ABCPD1234E, Phone: 9876543210, Email: test@example.com";
        let matches = IndianPIIPatterns::find_all(
            text,
            &[PIIType::PAN, PIIType::PhoneNumber, PIIType::Email],
        );
        assert_eq!(matches.len(), 3);
    }

    // P3 FIX: Verhoeff algorithm tests

    #[test]
    fn test_aadhaar_verhoeff_valid() {
        // Known valid Aadhaar number with correct Verhoeff checksum
        // Using test number: 234567890123 (starts with 2, 12 digits)
        // Actually let's compute a valid one using the algorithm
        // For testing, we'll use a number that passes the checksum

        // Test number generation: given 11 digits, compute check digit
        let digits_11: Vec<u32> = vec![2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2];
        let check_digit = IndianPIIPatterns::generate_aadhaar_checksum(&digits_11);
        assert!(check_digit.is_some());

        // Construct full number and verify
        let check = check_digit.unwrap();
        let full_number = format!("23456789012{}", check);
        assert!(IndianPIIPatterns::validate_aadhaar(&full_number), "Generated number should be valid");
    }

    #[test]
    fn test_aadhaar_verhoeff_invalid() {
        // Invalid Aadhaar numbers

        // Too short
        assert!(!IndianPIIPatterns::validate_aadhaar("1234567890"));

        // Starts with 0
        assert!(!IndianPIIPatterns::validate_aadhaar("012345678901"));

        // Starts with 1
        assert!(!IndianPIIPatterns::validate_aadhaar("123456789012"));
    }

    #[test]
    fn test_aadhaar_verhoeff_with_spaces() {
        // Aadhaar with spaces should be validated correctly
        let digits_11: Vec<u32> = vec![2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2];
        let check_digit = IndianPIIPatterns::generate_aadhaar_checksum(&digits_11).unwrap();

        // Format with spaces: XXXX XXXX XXXX
        let with_spaces = format!("2345 6789 012{}", check_digit);
        assert!(IndianPIIPatterns::validate_aadhaar(&with_spaces), "Number with spaces should be valid");
    }

    #[test]
    fn test_aadhaar_checksum_generation() {
        // Test checksum generation consistency
        let digits: Vec<u32> = vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9];
        let check = IndianPIIPatterns::generate_aadhaar_checksum(&digits);
        assert!(check.is_some());

        // Verify the generated check digit makes the number valid
        let check_digit = check.unwrap();
        assert!(check_digit < 10, "Check digit should be single digit");

        // Build full number and validate
        let mut full_digits = digits.clone();
        full_digits.push(check_digit);
        let number_str: String = full_digits.iter().map(|d| char::from_digit(*d, 10).unwrap()).collect();
        assert!(IndianPIIPatterns::validate_aadhaar(&number_str), "Number with generated checksum should be valid");
    }

    #[test]
    fn test_verhoeff_single_digit_error_detection() {
        // Verhoeff should detect single-digit errors
        let digits_11: Vec<u32> = vec![2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2];
        let check = IndianPIIPatterns::generate_aadhaar_checksum(&digits_11).unwrap();
        let valid_number = format!("23456789012{}", check);

        // Verify original is valid
        assert!(IndianPIIPatterns::validate_aadhaar(&valid_number));

        // Change one digit (not the first or check digit)
        let invalid_number = format!("23456789022{}", check);
        assert!(!IndianPIIPatterns::validate_aadhaar(&invalid_number), "Single digit change should invalidate");
    }
}
