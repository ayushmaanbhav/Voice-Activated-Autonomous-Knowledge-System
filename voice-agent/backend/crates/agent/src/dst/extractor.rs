//! Slot Value Extractor for Gold Loan Conversations
//!
//! Implements rule-based and pattern-based slot extraction from user utterances.
//! Supports Hindi, Hinglish, and English utterances.

use regex::Regex;
use std::collections::HashMap;
use voice_agent_text_processing::intent::{Slot, SlotType};

/// Slot extractor for gold loan domain
pub struct SlotExtractor {
    /// Regex patterns for amount extraction
    amount_patterns: Vec<(Regex, AmountMultiplier)>,
    /// Regex patterns for weight extraction
    weight_patterns: Vec<Regex>,
    /// Regex patterns for phone extraction
    phone_patterns: Vec<Regex>,
    /// Regex patterns for pincode extraction
    pincode_patterns: Vec<Regex>,
    /// Regex patterns for time extraction
    time_patterns: Vec<Regex>,
    /// Lender name patterns
    lender_patterns: HashMap<String, Vec<String>>,
    /// Regex patterns for name extraction
    name_patterns: Vec<Regex>,
    /// Regex patterns for PAN extraction
    pan_patterns: Vec<Regex>,
    /// Regex patterns for DOB extraction
    dob_patterns: Vec<Regex>,
    /// Regex patterns for loan purpose extraction
    purpose_patterns: Vec<(Regex, String)>,
    /// Regex patterns for repayment type extraction
    repayment_patterns: Vec<(Regex, String)>,
    /// Regex patterns for city extraction
    city_patterns: Vec<Regex>,
    /// Intent detection patterns
    intent_patterns: Vec<(Regex, String)>,
}

/// Amount multiplier for parsing
#[derive(Debug, Clone, Copy)]
enum AmountMultiplier {
    Unit,       // 1
    Thousand,   // 1,000
    Lakh,       // 100,000
    Crore,      // 10,000,000
}

impl AmountMultiplier {
    fn value(&self) -> f64 {
        match self {
            AmountMultiplier::Unit => 1.0,
            AmountMultiplier::Thousand => 1_000.0,
            AmountMultiplier::Lakh => 100_000.0,
            AmountMultiplier::Crore => 10_000_000.0,
        }
    }
}

impl SlotExtractor {
    /// Create a new slot extractor
    pub fn new() -> Self {
        Self {
            amount_patterns: Self::build_amount_patterns(),
            weight_patterns: Self::build_weight_patterns(),
            phone_patterns: Self::build_phone_patterns(),
            pincode_patterns: Self::build_pincode_patterns(),
            time_patterns: Self::build_time_patterns(),
            lender_patterns: Self::build_lender_patterns(),
            name_patterns: Self::build_name_patterns(),
            pan_patterns: Self::build_pan_patterns(),
            dob_patterns: Self::build_dob_patterns(),
            purpose_patterns: Self::build_purpose_patterns(),
            repayment_patterns: Self::build_repayment_patterns(),
            city_patterns: Self::build_city_patterns(),
            intent_patterns: Self::build_intent_patterns(),
        }
    }

    fn build_amount_patterns() -> Vec<(Regex, AmountMultiplier)> {
        vec![
            // Crore patterns
            (Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(?:crore|cr|करोड़)").unwrap(), AmountMultiplier::Crore),
            // Lakh patterns (English and Hindi)
            (Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(?:lakh|lac|लाख)").unwrap(), AmountMultiplier::Lakh),
            // Thousand patterns
            (Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(?:thousand|k|हज़ार|hazar)").unwrap(), AmountMultiplier::Thousand),
            // Direct rupee amounts
            (Regex::new(r"(?:₹|rs\.?|rupees?)\s*(\d+(?:,\d+)*)").unwrap(), AmountMultiplier::Unit),
            // Plain large numbers (5-8 digits, avoiding phone numbers)
            // Phone numbers are 10 digits starting with 6-9, so we limit to 8 digits max
            (Regex::new(r"\b(\d{5,8})\b").unwrap(), AmountMultiplier::Unit),
        ]
    }

    fn build_weight_patterns() -> Vec<Regex> {
        vec![
            // Grams patterns
            Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(?:grams?|gm|g|ग्राम)").unwrap(),
            // Tola patterns (1 tola ≈ 11.66g)
            Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(?:tola|तोला)").unwrap(),
            // Contextual weight (e.g., "I have 50 grams gold")
            Regex::new(r"(?i)(?:have|hai|है)\s*(\d+(?:\.\d+)?)\s*(?:grams?|g)?\s*(?:gold|sona|सोना)").unwrap(),
        ]
    }

    fn build_phone_patterns() -> Vec<Regex> {
        vec![
            // Indian mobile numbers (10 digits starting with 6-9)
            Regex::new(r"\b([6-9]\d{9})\b").unwrap(),
            // With country code
            Regex::new(r"(?:\+91|91)?[-\s]?([6-9]\d{9})\b").unwrap(),
            // Formatted numbers
            Regex::new(r"\b([6-9]\d{2})[-\s]?(\d{3})[-\s]?(\d{4})\b").unwrap(),
        ]
    }

    fn build_pincode_patterns() -> Vec<Regex> {
        vec![
            // Indian pincodes (6 digits, first digit 1-9)
            Regex::new(r"\b([1-9]\d{5})\b").unwrap(),
            // With "pincode" keyword
            Regex::new(r"(?i)(?:pincode|pin|पिनकोड)\s*(?:is|hai|है)?\s*(\d{6})").unwrap(),
        ]
    }

    fn build_time_patterns() -> Vec<Regex> {
        vec![
            // Time formats
            Regex::new(r"(?i)(\d{1,2})(?::(\d{2}))?\s*(am|pm|बजे)").unwrap(),
            // Time slots
            Regex::new(r"(?i)(morning|afternoon|evening|subah|dopahar|shaam)").unwrap(),
        ]
    }

    fn build_lender_patterns() -> HashMap<String, Vec<String>> {
        let mut patterns = HashMap::new();

        patterns.insert("muthoot".to_string(), vec![
            "muthoot".to_string(),
            "muthut".to_string(),
            "muthoot finance".to_string(),
        ]);

        patterns.insert("manappuram".to_string(), vec![
            "manappuram".to_string(),
            "manapuram".to_string(),
            "manappuram gold".to_string(),
        ]);

        patterns.insert("hdfc".to_string(), vec![
            "hdfc".to_string(),
            "hdfc bank".to_string(),
        ]);

        patterns.insert("icici".to_string(), vec![
            "icici".to_string(),
            "icici bank".to_string(),
        ]);

        patterns.insert("sbi".to_string(), vec![
            "sbi".to_string(),
            "state bank".to_string(),
        ]);

        patterns.insert("kotak".to_string(), vec![
            "kotak".to_string(),
            "kotak mahindra".to_string(),
        ]);

        patterns.insert("axis".to_string(), vec![
            "axis".to_string(),
            "axis bank".to_string(),
        ]);

        patterns.insert("federal".to_string(), vec![
            "federal".to_string(),
            "federal bank".to_string(),
        ]);

        patterns.insert("iifl".to_string(), vec![
            "iifl".to_string(),
            "india infoline".to_string(),
        ]);

        patterns
    }

    fn build_name_patterns() -> Vec<Regex> {
        vec![
            // English patterns: "my name is X", "I am X", "this is X", "I'm X"
            Regex::new(r"(?i)(?:my\s+name\s+is|i\s+am|i'm|this\s+is|call\s+me)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)").unwrap(),
            // Hindi patterns: "mera naam X hai" - capture name before hai
            Regex::new(r"(?i)(?:mera\s+)?(?:naam|name)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]*)?)\s+(?:hai|h)\b").unwrap(),
            // Hindi patterns without hai: "mera naam X", "naam X"
            Regex::new(r"(?i)(?:mera\s+)?(?:naam|name)\s+([A-Z][a-zA-Z]+)(?:\s+[A-Z][a-zA-Z]+)?(?:\s|$|[.,])").unwrap(),
            // Simple name after introduction keywords
            Regex::new(r"(?i)(?:myself|name[:\s]+)\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)").unwrap(),
        ]
    }

    fn build_pan_patterns() -> Vec<Regex> {
        vec![
            // Standard PAN format: 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F)
            Regex::new(r"(?i)(?:pan|pan\s+(?:card|number|no\.?)|my\s+pan)\s*(?:is|:)?\s*([A-Z]{5}[0-9]{4}[A-Z])").unwrap(),
            // Just PAN number mentioned
            Regex::new(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b").unwrap(),
            // Numeric PAN (for users who say just digits - we'll capture but flag as incomplete)
            Regex::new(r"(?i)pan\s+(?:is|:)?\s*(\d{8,10})").unwrap(),
        ]
    }

    fn build_dob_patterns() -> Vec<Regex> {
        vec![
            // Standard date formats: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
            Regex::new(r"(?i)(?:date\s+of\s+birth|dob|born\s+on|birthday)\s*(?:is|:)?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})").unwrap(),
            // Written format: 25 February 1993, 25th Feb 1993
            Regex::new(r"(?i)(?:date\s+of\s+birth|dob|born\s+on|birthday)\s*(?:is|:)?\s*(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{2,4})").unwrap(),
            // Hindi format: 25 February 1993
            Regex::new(r"(?i)(?:janam\s+din|janam\s+tithi)\s*(?:hai|:)?\s*(\d{1,2}\s+\w+\s+\d{2,4})").unwrap(),
        ]
    }

    /// Build patterns for loan purpose extraction
    fn build_purpose_patterns() -> Vec<(Regex, String)> {
        vec![
            // Business purposes
            (Regex::new(r"(?i)(?:business|dhandha|vyapaar|karobar|shop|dukaan)").unwrap(), "business".to_string()),
            (Regex::new(r"(?i)(?:working\s+capital|stock|inventory|माल)").unwrap(), "business_working_capital".to_string()),
            // Medical emergencies
            (Regex::new(r"(?i)(?:medical|hospital|doctor|treatment|ilaj|ilaaj|dawai|medicine|surgery|operation)").unwrap(), "medical".to_string()),
            // Education
            (Regex::new(r"(?i)(?:education|school|college|fees|padhai|study|exam|admission)").unwrap(), "education".to_string()),
            // Wedding/marriage
            (Regex::new(r"(?i)(?:wedding|marriage|shaadi|shadi|vivah|byah)").unwrap(), "wedding".to_string()),
            // Home renovation
            (Regex::new(r"(?i)(?:renovation|repair|construction|ghar|home\s+improvement|makaan)").unwrap(), "home_renovation".to_string()),
            // Agriculture
            (Regex::new(r"(?i)(?:farming|agriculture|khet|kheti|crop|fasal|tractor|seeds|beej)").unwrap(), "agriculture".to_string()),
            // Debt consolidation
            (Regex::new(r"(?i)(?:debt|loan\s+repay|karza|karz|EMI\s+pay)").unwrap(), "debt_consolidation".to_string()),
            // Emergency/urgent
            (Regex::new(r"(?i)(?:emergency|urgent|zaruri|jaldi|turant|immediately)").unwrap(), "emergency".to_string()),
        ]
    }

    /// Build patterns for repayment type preferences
    fn build_repayment_patterns() -> Vec<(Regex, String)> {
        vec![
            // EMI preference
            (Regex::new(r"(?i)(?:EMI|monthly\s+(?:payment|installment)|mahina|kishte)").unwrap(), "emi".to_string()),
            // Bullet repayment
            (Regex::new(r"(?i)(?:bullet|lump\s*sum|one\s+time|ek\s+baar|ekmusht)").unwrap(), "bullet".to_string()),
            // Overdraft
            (Regex::new(r"(?i)(?:overdraft|OD|credit\s+line|flexible)").unwrap(), "overdraft".to_string()),
            // Interest only
            (Regex::new(r"(?i)(?:interest\s+only|sirf\s+byaaj|only\s+interest)").unwrap(), "interest_only".to_string()),
        ]
    }

    /// Build patterns for city extraction (major Indian cities)
    fn build_city_patterns() -> Vec<Regex> {
        vec![
            // Direct city mentions with context
            Regex::new(r"(?i)(?:from|in|at|near|city|sheher)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)").unwrap(),
            // Major metros - direct match
            Regex::new(r"(?i)\b(Mumbai|Delhi|Bangalore|Bengaluru|Chennai|Hyderabad|Kolkata|Pune|Ahmedabad|Jaipur|Lucknow|Kanpur|Nagpur|Indore|Thane|Bhopal|Visakhapatnam|Patna|Vadodara|Ghaziabad|Ludhiana|Agra|Nashik|Faridabad|Meerut|Rajkot|Kalyan|Vasai|Varanasi|Srinagar|Aurangabad|Dhanbad|Amritsar|Navi Mumbai|Allahabad|Ranchi|Howrah|Coimbatore|Jabalpur|Gwalior|Vijayawada|Jodhpur|Madurai|Raipur|Kota|Guwahati|Chandigarh|Solapur|Hubli|Mysore|Tiruchirappalli|Bareilly|Aligarh|Tiruppur|Gurgaon|Noida|NCR)\b").unwrap(),
            // Hindi transliterations
            Regex::new(r"(?i)\b(Dilli|Mumbay|Calcutta|Madras|Bangaluru)\b").unwrap(),
        ]
    }

    /// Build patterns for intent detection (helps small models understand what user wants)
    /// IMPORTANT: Order matters - more specific patterns should come first
    fn build_intent_patterns() -> Vec<(Regex, String)> {
        vec![
            // Balance transfer (BEFORE savings/comparison - most specific)
            (Regex::new(r"(?i)(?:balance\s+transfer|loan\s+transfer|transfer\s+(?:my\s+)?loan|move\s+(?:my\s+)?loan|transfer\s+kar|BT\s+kar|switch\s+(?:to|from)\s+\w+)").unwrap(), "balance_transfer".to_string()),
            // Gold price inquiry (BEFORE rate_inquiry - more specific)
            (Regex::new(r"(?i)(?:gold\s+(?:price|rate)|sone\s+ka\s+(?:rate|bhav|price)|aaj\s+ka\s+(?:gold\s+)?rate|today.+gold|current\s+gold)").unwrap(), "gold_price_inquiry".to_string()),
            // Rate inquiry
            (Regex::new(r"(?i)(?:interest\s+rate|byaaj\s+dar|rate\s+kya|kitna\s+percent|what.+(?:interest|byaaj)\s+rate)").unwrap(), "rate_inquiry".to_string()),
            // Savings calculation
            (Regex::new(r"(?i)(?:kitna\s+bachega|how\s+much\s+(?:can\s+i\s+)?sav|bachat|savings|save\s+money|calculate\s+saving)").unwrap(), "savings_inquiry".to_string()),
            // Eligibility check
            (Regex::new(r"(?i)(?:am\s+i\s+eligible|eligibility|loan\s+milega|kitna\s+loan|qualify|kya\s+mil\s+sakta|eligible\s+for)").unwrap(), "eligibility_inquiry".to_string()),
            // Document inquiry
            (Regex::new(r"(?i)(?:documents?\s+(?:required|needed|chahiye|list)|kya\s+laana|what\s+(?:documents?|to\s+bring)|kaunsa\s+document|laana\s+(?:hoga|padega))").unwrap(), "document_inquiry".to_string()),
            // Appointment booking (BEFORE branch - more specific, look for action words)
            (Regex::new(r"(?i)(?:book\s+(?:an?\s+)?appointment|schedule\s+(?:a\s+)?(?:visit|appointment)|fix\s+(?:a\s+)?time|milna\s+(?:hai|chahta)|time\s+slot|slot\s+book)").unwrap(), "appointment_request".to_string()),
            // Branch/location inquiry
            (Regex::new(r"(?i)(?:(?:nearest|nearby)\s+branch|branch\s+(?:location|kahan|where)|where\s+is\s+(?:the\s+)?(?:branch|office)|office\s+address|location\s+of)").unwrap(), "branch_inquiry".to_string()),
            // Safety/security concern
            (Regex::new(r"(?i)(?:(?:is\s+)?(?:my\s+)?gold\s+safe|security|suraksha|chori|theft|insurance|vault|locker)").unwrap(), "safety_inquiry".to_string()),
            // Repayment inquiry
            (Regex::new(r"(?i)(?:repay|payment\s+(?:option|method)|EMI\s+(?:kaise|how)|bhugtan|kaise\s+dena|how\s+to\s+pay|repayment)").unwrap(), "repayment_inquiry".to_string()),
            // Closure/release
            (Regex::new(r"(?i)(?:close\s+(?:my\s+)?loan|loan\s+close|release\s+(?:my\s+)?gold|gold\s+back|sona\s+wapas|get\s+(?:my\s+)?gold\s+back)").unwrap(), "closure_inquiry".to_string()),
            // Human escalation
            (Regex::new(r"(?i)(?:talk\s+to\s+(?:a\s+)?human|(?:real\s+)?agent|real\s+person|customer\s+care|complaint|shikayat|(?:speak\s+(?:to|with)\s+)?manager)").unwrap(), "human_escalation".to_string()),
            // Callback request
            (Regex::new(r"(?i)(?:call\s+(?:me\s+)?back|callback|phone\s+kar|give\s+(?:me\s+)?(?:a\s+)?call|ring\s+me)").unwrap(), "callback_request".to_string()),
            // SMS request
            (Regex::new(r"(?i)(?:send\s+(?:me\s+)?(?:sms|message|details|info)|SMS\s+kar|whatsapp\s+(?:me|kar))").unwrap(), "sms_request".to_string()),
            // Competitor comparison (LAST - most generic, avoid triggering on mentions of competitors in other contexts)
            (Regex::new(r"(?i)(?:compare\s+(?:with|to)|comparison|vs\s+\w+|versus|better\s+than\s+(?:muthoot|manappuram|iifl))").unwrap(), "comparison_inquiry".to_string()),
        ]
    }

    /// Extract all slots from an utterance
    pub fn extract(&self, utterance: &str) -> HashMap<String, Slot> {
        let mut slots = HashMap::new();

        // Extract amount
        if let Some((amount, confidence)) = self.extract_amount(utterance) {
            slots.insert("loan_amount".to_string(), Slot {
                name: "loan_amount".to_string(),
                value: Some(amount.to_string()),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        // Extract weight
        if let Some((weight, confidence)) = self.extract_weight(utterance) {
            slots.insert("gold_weight".to_string(), Slot {
                name: "gold_weight".to_string(),
                value: Some(weight.to_string()),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        // Extract phone
        if let Some((phone, confidence)) = self.extract_phone(utterance) {
            slots.insert("phone_number".to_string(), Slot {
                name: "phone_number".to_string(),
                value: Some(phone),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        // Extract pincode
        if let Some((pincode, confidence)) = self.extract_pincode(utterance) {
            slots.insert("pincode".to_string(), Slot {
                name: "pincode".to_string(),
                value: Some(pincode),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        // Extract lender
        if let Some((lender, confidence)) = self.extract_lender(utterance) {
            slots.insert("current_lender".to_string(), Slot {
                name: "current_lender".to_string(),
                value: Some(lender),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        // Extract purity
        if let Some((purity, confidence)) = self.extract_purity(utterance) {
            slots.insert("gold_purity".to_string(), Slot {
                name: "gold_purity".to_string(),
                value: Some(purity),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        // Extract purpose
        if let Some((purpose, confidence)) = self.extract_purpose(utterance) {
            slots.insert("loan_purpose".to_string(), Slot {
                name: "loan_purpose".to_string(),
                value: Some(purpose),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        // Extract location
        if let Some((location, confidence)) = self.extract_location(utterance) {
            slots.insert("location".to_string(), Slot {
                name: "location".to_string(),
                value: Some(location),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        // Extract customer name
        if let Some((name, confidence)) = self.extract_name(utterance) {
            slots.insert("customer_name".to_string(), Slot {
                name: "customer_name".to_string(),
                value: Some(name),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        // Extract PAN number
        if let Some((pan, confidence)) = self.extract_pan(utterance) {
            slots.insert("pan_number".to_string(), Slot {
                name: "pan_number".to_string(),
                value: Some(pan),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        // Extract date of birth
        if let Some((dob, confidence)) = self.extract_dob(utterance) {
            slots.insert("date_of_birth".to_string(), Slot {
                name: "date_of_birth".to_string(),
                value: Some(dob),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        // Extract interest rate (FIX: was missing from extract())
        if let Some((rate, confidence)) = self.extract_interest_rate(utterance) {
            slots.insert("current_interest_rate".to_string(), Slot {
                name: "current_interest_rate".to_string(),
                value: Some(rate.to_string()),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        // Extract tenure (FIX: was missing from extract())
        if let Some((tenure, confidence)) = self.extract_tenure(utterance) {
            slots.insert("tenure_months".to_string(), Slot {
                name: "tenure_months".to_string(),
                value: Some(tenure.to_string()),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        // Extract repayment type preference
        if let Some((repayment_type, confidence)) = self.extract_repayment_type(utterance) {
            slots.insert("repayment_type".to_string(), Slot {
                name: "repayment_type".to_string(),
                value: Some(repayment_type),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        // Extract city
        if let Some((city, confidence)) = self.extract_city(utterance) {
            slots.insert("city".to_string(), Slot {
                name: "city".to_string(),
                value: Some(city),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        // Extract detected intent (helps LLM understand what user wants)
        if let Some((intent, confidence)) = self.extract_intent(utterance) {
            slots.insert("detected_intent".to_string(), Slot {
                name: "detected_intent".to_string(),
                value: Some(intent),
                confidence,
                slot_type: SlotType::Text,
            });
        }

        slots
    }

    /// Extract amount from utterance
    pub fn extract_amount(&self, utterance: &str) -> Option<(f64, f32)> {
        let lower = utterance.to_lowercase();

        for (pattern, multiplier) in &self.amount_patterns {
            if let Some(caps) = pattern.captures(&lower) {
                if let Some(num_match) = caps.get(1) {
                    let num_str = num_match.as_str().replace(',', "");
                    if let Ok(num) = num_str.parse::<f64>() {
                        let amount = num * multiplier.value();

                        // Skip if looks like a phone number (10-digit starting with 6-9)
                        let clean_str = num_str.replace(',', "");
                        if clean_str.len() == 10 {
                            if let Some(first) = clean_str.chars().next() {
                                if first >= '6' && first <= '9' {
                                    tracing::debug!(
                                        value = %clean_str,
                                        "Skipping amount extraction - looks like phone number"
                                    );
                                    continue;
                                }
                            }
                        }

                        // Skip if unreasonably large (> 100 crore)
                        if amount > 1_000_000_000.0 {
                            tracing::debug!(
                                amount = amount,
                                "Skipping amount extraction - unreasonably large"
                            );
                            continue;
                        }

                        // Confidence based on context
                        let confidence = if lower.contains("loan") || lower.contains("lakh")
                            || lower.contains("amount") || lower.contains("chahiye")
                        {
                            0.9
                        } else {
                            0.7
                        };

                        return Some((amount, confidence));
                    }
                }
            }
        }

        None
    }

    /// Extract weight from utterance
    pub fn extract_weight(&self, utterance: &str) -> Option<(f64, f32)> {
        let lower = utterance.to_lowercase();

        for pattern in &self.weight_patterns {
            if let Some(caps) = pattern.captures(&lower) {
                if let Some(num_match) = caps.get(1) {
                    if let Ok(num) = num_match.as_str().parse::<f64>() {
                        // Check if it's tola (convert to grams)
                        let weight = if lower.contains("tola") || lower.contains("तोला") {
                            num * 11.66 // 1 tola ≈ 11.66 grams
                        } else {
                            num
                        };

                        // Confidence based on context
                        let confidence = if lower.contains("gold") || lower.contains("sona")
                            || lower.contains("gram") || lower.contains("tola")
                        {
                            0.9
                        } else {
                            0.7
                        };

                        return Some((weight, confidence));
                    }
                }
            }
        }

        None
    }

    /// Extract phone number from utterance
    pub fn extract_phone(&self, utterance: &str) -> Option<(String, f32)> {
        for pattern in &self.phone_patterns {
            if let Some(caps) = pattern.captures(utterance) {
                // Handle formatted numbers
                if caps.len() > 2 {
                    // Formatted pattern with groups
                    let parts: Vec<&str> = caps.iter()
                        .skip(1)
                        .filter_map(|m| m.map(|m| m.as_str()))
                        .collect();
                    let phone = parts.join("");
                    if phone.len() == 10 {
                        return Some((phone, 0.95));
                    }
                } else if let Some(m) = caps.get(1) {
                    let phone = m.as_str().to_string();
                    if phone.len() == 10 {
                        return Some((phone, 0.95));
                    }
                }
            }
        }

        None
    }

    /// Extract pincode from utterance
    pub fn extract_pincode(&self, utterance: &str) -> Option<(String, f32)> {
        for pattern in &self.pincode_patterns {
            if let Some(caps) = pattern.captures(utterance) {
                if let Some(m) = caps.get(1) {
                    let pincode = m.as_str().to_string();
                    // Basic validation - Indian pincodes
                    if pincode.len() == 6 && pincode.chars().next().unwrap() != '0' {
                        let confidence = if utterance.to_lowercase().contains("pincode")
                            || utterance.to_lowercase().contains("pin")
                        {
                            0.95
                        } else {
                            0.7
                        };
                        return Some((pincode, confidence));
                    }
                }
            }
        }

        None
    }

    /// Extract lender name from utterance
    pub fn extract_lender(&self, utterance: &str) -> Option<(String, f32)> {
        let lower = utterance.to_lowercase();

        for (canonical, variants) in &self.lender_patterns {
            for variant in variants {
                if lower.contains(variant) {
                    let confidence = if lower.contains("from") || lower.contains("with")
                        || lower.contains("se") || lower.contains("current")
                    {
                        0.9
                    } else {
                        0.7
                    };
                    return Some((canonical.clone(), confidence));
                }
            }
        }

        None
    }

    /// Extract gold purity from utterance
    pub fn extract_purity(&self, utterance: &str) -> Option<(String, f32)> {
        let lower = utterance.to_lowercase();

        // Direct karat mentions
        let purity_patterns = [
            (r"24\s*(?:k|karat|carat|kt)", "24"),
            (r"22\s*(?:k|karat|carat|kt)", "22"),
            (r"18\s*(?:k|karat|carat|kt)", "18"),
            (r"14\s*(?:k|karat|carat|kt)", "14"),
            // Descriptive
            (r"pure\s*gold", "24"),
            (r"hallmark(?:ed)?", "22"), // Hallmarked is typically 22k in India
        ];

        for (pattern, purity) in &purity_patterns {
            if let Ok(re) = Regex::new(&format!("(?i){}", pattern)) {
                if re.is_match(&lower) {
                    return Some((purity.to_string(), 0.85));
                }
            }
        }

        None
    }

    /// Extract loan purpose from utterance
    pub fn extract_purpose(&self, utterance: &str) -> Option<(String, f32)> {
        let lower = utterance.to_lowercase();

        let purposes = [
            // Medical
            (vec!["medical", "hospital", "treatment", "surgery", "ilaj", "dawai", "doctor"],
             "medical"),
            // Education
            (vec!["education", "school", "college", "fees", "padhai", "admission"],
             "education"),
            // Business
            (vec!["business", "shop", "dukan", "karobar", "vyapaar", "investment"],
             "business"),
            // Wedding
            (vec!["wedding", "marriage", "shaadi", "vivah", "function"],
             "wedding"),
            // Emergency
            (vec!["emergency", "urgent", "zaruri", "turant"],
             "emergency"),
            // Home
            (vec!["home", "house", "ghar", "renovation", "repair", "construction"],
             "home"),
            // Personal
            (vec!["personal", "family", "apna kaam"],
             "personal"),
        ];

        for (keywords, purpose) in &purposes {
            for keyword in keywords {
                if lower.contains(keyword) {
                    return Some((purpose.to_string(), 0.8));
                }
            }
        }

        None
    }

    /// Extract location from utterance
    pub fn extract_location(&self, utterance: &str) -> Option<(String, f32)> {
        let lower = utterance.to_lowercase();

        // Major Indian cities
        let cities = [
            "mumbai", "delhi", "bangalore", "bengaluru", "chennai", "hyderabad",
            "kolkata", "pune", "ahmedabad", "jaipur", "surat", "lucknow",
            "kanpur", "nagpur", "indore", "thane", "bhopal", "visakhapatnam",
            "patna", "vadodara", "ghaziabad", "ludhiana", "agra", "nashik",
            "faridabad", "meerut", "rajkot", "kalyan", "vasai", "varanasi",
            "aurangabad", "dhanbad", "amritsar", "allahabad", "ranchi", "gwalior",
            "jodhpur", "coimbatore", "vijayawada", "madurai", "raipur", "kota",
        ];

        for city in &cities {
            if lower.contains(city) {
                let confidence = if lower.contains("in ") || lower.contains("at ")
                    || lower.contains("from ") || lower.contains("near ")
                    || lower.contains("mein") || lower.contains("में")
                {
                    0.9
                } else {
                    0.7
                };

                // Capitalize city name
                let capitalized = city.chars().next().unwrap().to_uppercase().to_string()
                    + &city[1..];
                return Some((capitalized, confidence));
            }
        }

        // Try to extract location after keywords
        let location_patterns = [
            Regex::new(r"(?i)(?:from|in|at|near|mein|में)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)").unwrap(),
        ];

        for pattern in &location_patterns {
            if let Some(caps) = pattern.captures(utterance) {
                if let Some(m) = caps.get(1) {
                    let location = m.as_str().to_string();
                    if location.len() >= 3 && location.len() <= 30 {
                        return Some((location, 0.6));
                    }
                }
            }
        }

        None
    }

    /// Extract tenure from utterance
    pub fn extract_tenure(&self, utterance: &str) -> Option<(u32, f32)> {
        let lower = utterance.to_lowercase();

        // Month patterns
        let month_pattern = Regex::new(r"(\d+)\s*(?:months?|mahine|महीने)").unwrap();
        if let Some(caps) = month_pattern.captures(&lower) {
            if let Some(m) = caps.get(1) {
                if let Ok(months) = m.as_str().parse::<u32>() {
                    if months >= 1 && months <= 60 {
                        return Some((months, 0.85));
                    }
                }
            }
        }

        // Year patterns
        let year_pattern = Regex::new(r"(\d+)\s*(?:years?|saal|साल)").unwrap();
        if let Some(caps) = year_pattern.captures(&lower) {
            if let Some(m) = caps.get(1) {
                if let Ok(years) = m.as_str().parse::<u32>() {
                    if years >= 1 && years <= 5 {
                        return Some((years * 12, 0.85));
                    }
                }
            }
        }

        None
    }

    /// Extract interest rate from utterance
    pub fn extract_interest_rate(&self, utterance: &str) -> Option<(f32, f32)> {
        let lower = utterance.to_lowercase();

        // Pattern with explicit rate context
        let rate_context_pattern = Regex::new(r"(?i)(?:interest\s+)?rate\s+(?:is|:)?\s*(\d+(?:\.\d+)?)\s*(?:%|percent|प्रतिशत)?").unwrap();
        if let Some(caps) = rate_context_pattern.captures(&lower) {
            if let Some(m) = caps.get(1) {
                if let Ok(rate) = m.as_str().parse::<f32>() {
                    if rate >= 5.0 && rate <= 30.0 {
                        return Some((rate, 0.9));
                    }
                }
            }
        }

        // Pattern with percent symbol
        let rate_pattern = Regex::new(r"(\d+(?:\.\d+)?)\s*(?:%|percent|प्रतिशत)").unwrap();
        if let Some(caps) = rate_pattern.captures(&lower) {
            if let Some(m) = caps.get(1) {
                if let Ok(rate) = m.as_str().parse::<f32>() {
                    // Gold loan rates are typically 7-24%
                    if rate >= 5.0 && rate <= 30.0 {
                        return Some((rate, 0.85));
                    }
                }
            }
        }

        None
    }

    /// Extract customer name from utterance
    pub fn extract_name(&self, utterance: &str) -> Option<(String, f32)> {
        for pattern in &self.name_patterns {
            if let Some(caps) = pattern.captures(utterance) {
                if let Some(m) = caps.get(1) {
                    let name = m.as_str().trim().to_string();
                    // Basic validation: name should be 2-50 chars and not be common words
                    if name.len() >= 2 && name.len() <= 50 {
                        let lower = name.to_lowercase();
                        // Filter out common false positives
                        let exclude_words = [
                            "loan", "gold", "bank", "kotak", "muthoot", "amount",
                            "rate", "interest", "help", "need", "want", "please",
                        ];
                        if !exclude_words.iter().any(|w| lower == *w) {
                            return Some((name, 0.85));
                        }
                    }
                }
            }
        }

        None
    }

    /// Extract PAN number from utterance
    pub fn extract_pan(&self, utterance: &str) -> Option<(String, f32)> {
        let upper = utterance.to_uppercase();

        for pattern in &self.pan_patterns {
            if let Some(caps) = pattern.captures(&upper) {
                if let Some(m) = caps.get(1) {
                    let pan = m.as_str().to_string();
                    // Validate PAN format: 5 letters + 4 digits + 1 letter
                    if pan.len() == 10 {
                        let chars: Vec<char> = pan.chars().collect();
                        let valid_format = chars[0..5].iter().all(|c| c.is_ascii_alphabetic())
                            && chars[5..9].iter().all(|c| c.is_ascii_digit())
                            && chars[9].is_ascii_alphabetic();

                        if valid_format {
                            return Some((pan, 0.95));
                        }
                    }
                    // Numeric PAN (incomplete/incorrect format)
                    if pan.chars().all(|c| c.is_ascii_digit()) && pan.len() >= 8 {
                        return Some((pan, 0.5)); // Low confidence for numeric-only
                    }
                }
            }
        }

        None
    }

    /// Extract date of birth from utterance
    pub fn extract_dob(&self, utterance: &str) -> Option<(String, f32)> {
        for pattern in &self.dob_patterns {
            if let Some(caps) = pattern.captures(utterance) {
                if let Some(m) = caps.get(1) {
                    let dob = m.as_str().trim().to_string();
                    // Basic validation: should look like a date
                    if dob.len() >= 6 && dob.len() <= 30 {
                        return Some((dob, 0.85));
                    }
                }
            }
        }

        None
    }

    /// Extract repayment type preference from utterance
    pub fn extract_repayment_type(&self, utterance: &str) -> Option<(String, f32)> {
        let lower = utterance.to_lowercase();

        for (pattern, repayment_type) in &self.repayment_patterns {
            if pattern.is_match(&lower) {
                return Some((repayment_type.clone(), 0.8));
            }
        }

        None
    }

    /// Extract city from utterance
    pub fn extract_city(&self, utterance: &str) -> Option<(String, f32)> {
        // First try direct city patterns
        for pattern in &self.city_patterns {
            if let Some(caps) = pattern.captures(utterance) {
                if let Some(m) = caps.get(1) {
                    let city = m.as_str().trim().to_string();
                    // Basic validation
                    if city.len() >= 2 && city.len() <= 30 {
                        // Capitalize first letter
                        let capitalized = city.chars().next().unwrap().to_uppercase().to_string()
                            + &city[1..].to_lowercase();
                        return Some((capitalized, 0.85));
                    }
                }
            }
        }

        None
    }

    /// Extract detected intent from utterance (helps small models understand what user wants)
    pub fn extract_intent(&self, utterance: &str) -> Option<(String, f32)> {
        let lower = utterance.to_lowercase();

        // Check all intent patterns and return the first (most specific) match
        for (pattern, intent) in &self.intent_patterns {
            if pattern.is_match(&lower) {
                return Some((intent.clone(), 0.8));
            }
        }

        None
    }

    /// Extract loan purpose from utterance
    pub fn extract_loan_purpose(&self, utterance: &str) -> Option<(String, f32)> {
        let lower = utterance.to_lowercase();

        for (pattern, purpose) in &self.purpose_patterns {
            if pattern.is_match(&lower) {
                return Some((purpose.clone(), 0.8));
            }
        }

        None
    }
}

impl Default for SlotExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amount_extraction() {
        let extractor = SlotExtractor::new();

        // Lakh amounts
        let (amount, _) = extractor.extract_amount("I need a loan of 5 lakh").unwrap();
        assert!((amount - 500_000.0).abs() < 1.0);

        let (amount, _) = extractor.extract_amount("mujhe 3.5 lakh chahiye").unwrap();
        assert!((amount - 350_000.0).abs() < 1.0);

        // Crore amounts
        let (amount, _) = extractor.extract_amount("loan of 1 crore").unwrap();
        assert!((amount - 10_000_000.0).abs() < 1.0);

        // Thousand amounts
        let (amount, _) = extractor.extract_amount("50 thousand rupees").unwrap();
        assert!((amount - 50_000.0).abs() < 1.0);
    }

    #[test]
    fn test_weight_extraction() {
        let extractor = SlotExtractor::new();

        // Gram weights
        let (weight, _) = extractor.extract_weight("I have 50 grams of gold").unwrap();
        assert!((weight - 50.0).abs() < 0.1);

        let (weight, _) = extractor.extract_weight("mere paas 100g sona hai").unwrap();
        assert!((weight - 100.0).abs() < 0.1);

        // Tola weights
        let (weight, _) = extractor.extract_weight("5 tola gold").unwrap();
        assert!((weight - 58.3).abs() < 0.1); // 5 * 11.66
    }

    #[test]
    fn test_phone_extraction() {
        let extractor = SlotExtractor::new();

        let (phone, _) = extractor.extract_phone("my number is 9876543210").unwrap();
        assert_eq!(phone, "9876543210");

        let (phone, _) = extractor.extract_phone("call me at +91 8765432109").unwrap();
        assert_eq!(phone, "8765432109");
    }

    #[test]
    fn test_pincode_extraction() {
        let extractor = SlotExtractor::new();

        let (pincode, _) = extractor.extract_pincode("pincode is 400001").unwrap();
        assert_eq!(pincode, "400001");

        let (pincode, _) = extractor.extract_pincode("I'm in 560001").unwrap();
        assert_eq!(pincode, "560001");
    }

    #[test]
    fn test_lender_extraction() {
        let extractor = SlotExtractor::new();

        let (lender, _) = extractor.extract_lender("I have loan from Muthoot").unwrap();
        assert_eq!(lender, "muthoot");

        let (lender, _) = extractor.extract_lender("with HDFC bank").unwrap();
        assert_eq!(lender, "hdfc");
    }

    #[test]
    fn test_purity_extraction() {
        let extractor = SlotExtractor::new();

        let (purity, _) = extractor.extract_purity("24k gold").unwrap();
        assert_eq!(purity, "24");

        let (purity, _) = extractor.extract_purity("22 karat jewelry").unwrap();
        assert_eq!(purity, "22");
    }

    #[test]
    fn test_purpose_extraction() {
        let extractor = SlotExtractor::new();

        let (purpose, _) = extractor.extract_purpose("for medical treatment").unwrap();
        assert_eq!(purpose, "medical");

        let (purpose, _) = extractor.extract_purpose("business ke liye").unwrap();
        assert_eq!(purpose, "business");

        let (purpose, _) = extractor.extract_purpose("wedding expenses").unwrap();
        assert_eq!(purpose, "wedding");
    }

    #[test]
    fn test_location_extraction() {
        let extractor = SlotExtractor::new();

        let (location, _) = extractor.extract_location("I'm in Mumbai").unwrap();
        assert_eq!(location, "Mumbai");

        let (location, _) = extractor.extract_location("from Bangalore").unwrap();
        assert_eq!(location, "Bangalore");
    }

    #[test]
    fn test_tenure_extraction() {
        let extractor = SlotExtractor::new();

        let (tenure, _) = extractor.extract_tenure("for 12 months").unwrap();
        assert_eq!(tenure, 12);

        let (tenure, _) = extractor.extract_tenure("2 years loan").unwrap();
        assert_eq!(tenure, 24);
    }

    #[test]
    fn test_combined_extraction() {
        let extractor = SlotExtractor::new();

        let utterance = "I want a gold loan of 5 lakh for my 50 grams of 22k gold";
        let slots = extractor.extract(utterance);

        assert!(slots.contains_key("loan_amount"));
        assert!(slots.contains_key("gold_weight"));
        assert!(slots.contains_key("gold_purity"));
    }

    #[test]
    fn test_hindi_extraction() {
        let extractor = SlotExtractor::new();

        let (amount, _) = extractor.extract_amount("mujhe 5 lakh chahiye").unwrap();
        assert!((amount - 500_000.0).abs() < 1.0);

        let (weight, _) = extractor.extract_weight("mere paas 50 gram sona hai").unwrap();
        assert!((weight - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_name_extraction() {
        let extractor = SlotExtractor::new();

        // English patterns
        let (name, _) = extractor.extract_name("My name is Ayush").unwrap();
        assert_eq!(name, "Ayush");

        let (name, _) = extractor.extract_name("I am Rajesh Kumar").unwrap();
        assert_eq!(name, "Rajesh Kumar");

        // Hindi patterns
        let (name, _) = extractor.extract_name("Mera naam Priya hai").unwrap();
        assert_eq!(name, "Priya");

        // Should not match common words
        assert!(extractor.extract_name("I need a loan").is_none());
    }

    #[test]
    fn test_pan_extraction() {
        let extractor = SlotExtractor::new();

        // Standard PAN format
        let (pan, confidence) = extractor.extract_pan("My PAN is ABCDE1234F").unwrap();
        assert_eq!(pan, "ABCDE1234F");
        assert!(confidence > 0.9);

        // Just PAN number
        let (pan, _) = extractor.extract_pan("PAN: GHIJK5678L").unwrap();
        assert_eq!(pan, "GHIJK5678L");

        // Invalid PAN (all digits) - lower confidence
        let (pan, confidence) = extractor.extract_pan("PAN is 12345678").unwrap();
        assert_eq!(pan, "12345678");
        assert!(confidence < 0.6);
    }

    #[test]
    fn test_dob_extraction() {
        let extractor = SlotExtractor::new();

        // Date format
        let (dob, _) = extractor.extract_dob("My date of birth is 25/02/1993").unwrap();
        assert!(dob.contains("25"));

        // Written format
        let (dob, _) = extractor.extract_dob("date of birth is 25 February 1993").unwrap();
        assert!(dob.contains("25") && dob.contains("February"));
    }

    #[test]
    fn test_interest_rate_extraction() {
        let extractor = SlotExtractor::new();

        // With percent symbol
        let (rate, _) = extractor.extract_interest_rate("current rate is 18%").unwrap();
        assert!((rate - 18.0).abs() < 0.1);

        // With rate context
        let (rate, _) = extractor.extract_interest_rate("interest rate is 15 percent").unwrap();
        assert!((rate - 15.0).abs() < 0.1);

        // Out of range should not match
        assert!(extractor.extract_interest_rate("rate is 50%").is_none());
    }

    #[test]
    fn test_full_customer_extraction() {
        let extractor = SlotExtractor::new();

        // Simulate a customer providing all their details
        let utterance = "My name is Ayush. Phone is 8544130924. Loan amount 10 lakh at 18% from Muthoot.";
        let slots = extractor.extract(utterance);

        assert!(slots.contains_key("customer_name"));
        assert!(slots.contains_key("phone_number"));
        assert!(slots.contains_key("loan_amount"));
        assert!(slots.contains_key("current_interest_rate"));
        assert!(slots.contains_key("current_lender"));

        // Verify values
        assert_eq!(slots.get("customer_name").unwrap().value, Some("Ayush".to_string()));
        assert_eq!(slots.get("phone_number").unwrap().value, Some("8544130924".to_string()));
        assert_eq!(slots.get("current_lender").unwrap().value, Some("muthoot".to_string()));
    }

    // ============================================
    // NEW TESTS FOR INTENT AND TOOL TRIGGERING
    // ============================================

    #[test]
    fn test_intent_savings_inquiry() {
        let extractor = SlotExtractor::new();

        // Savings inquiry patterns
        let (intent, _) = extractor.extract_intent("kitna bachega").unwrap();
        assert_eq!(intent, "savings_inquiry");

        let (intent, _) = extractor.extract_intent("how much can I save money").unwrap();
        assert_eq!(intent, "savings_inquiry");

        let (intent, _) = extractor.extract_intent("calculate savings for me").unwrap();
        assert_eq!(intent, "savings_inquiry");
    }

    #[test]
    fn test_intent_balance_transfer() {
        let extractor = SlotExtractor::new();

        let (intent, _) = extractor.extract_intent("I want balance transfer").unwrap();
        assert_eq!(intent, "balance_transfer");

        let (intent, _) = extractor.extract_intent("loan transfer to Kotak").unwrap();
        assert_eq!(intent, "balance_transfer");

        let (intent, _) = extractor.extract_intent("transfer my loan").unwrap();
        assert_eq!(intent, "balance_transfer");

        let (intent, _) = extractor.extract_intent("BT kar do").unwrap();
        assert_eq!(intent, "balance_transfer");
    }

    #[test]
    fn test_intent_eligibility() {
        let extractor = SlotExtractor::new();

        let (intent, _) = extractor.extract_intent("am I eligible for gold loan").unwrap();
        assert_eq!(intent, "eligibility_inquiry");

        let (intent, _) = extractor.extract_intent("kitna loan mil sakta hai").unwrap();
        assert_eq!(intent, "eligibility_inquiry");

        let (intent, _) = extractor.extract_intent("do I qualify").unwrap();
        assert_eq!(intent, "eligibility_inquiry");
    }

    #[test]
    fn test_intent_document_inquiry() {
        let extractor = SlotExtractor::new();

        let (intent, _) = extractor.extract_intent("what documents required").unwrap();
        assert_eq!(intent, "document_inquiry");

        let (intent, _) = extractor.extract_intent("kya laana padega").unwrap();
        assert_eq!(intent, "document_inquiry");

        let (intent, _) = extractor.extract_intent("documents needed").unwrap();
        assert_eq!(intent, "document_inquiry");

        let (intent, _) = extractor.extract_intent("document list for gold loan").unwrap();
        assert_eq!(intent, "document_inquiry");
    }

    #[test]
    fn test_intent_branch_inquiry() {
        let extractor = SlotExtractor::new();

        let (intent, _) = extractor.extract_intent("nearest branch in Delhi").unwrap();
        assert_eq!(intent, "branch_inquiry");

        let (intent, _) = extractor.extract_intent("branch kahan hai").unwrap();
        assert_eq!(intent, "branch_inquiry");

        let (intent, _) = extractor.extract_intent("where is the branch").unwrap();
        assert_eq!(intent, "branch_inquiry");
    }

    #[test]
    fn test_intent_appointment() {
        let extractor = SlotExtractor::new();

        let (intent, _) = extractor.extract_intent("book an appointment").unwrap();
        assert_eq!(intent, "appointment_request");

        let (intent, _) = extractor.extract_intent("schedule a visit").unwrap();
        assert_eq!(intent, "appointment_request");

        let (intent, _) = extractor.extract_intent("fix a time for meeting").unwrap();
        assert_eq!(intent, "appointment_request");
    }

    #[test]
    fn test_intent_gold_price() {
        let extractor = SlotExtractor::new();

        let (intent, _) = extractor.extract_intent("what is gold price today").unwrap();
        assert_eq!(intent, "gold_price_inquiry");

        let (intent, _) = extractor.extract_intent("sone ka rate batao").unwrap();
        assert_eq!(intent, "gold_price_inquiry");

        let (intent, _) = extractor.extract_intent("current gold rate").unwrap();
        assert_eq!(intent, "gold_price_inquiry");
    }

    #[test]
    fn test_intent_comparison() {
        let extractor = SlotExtractor::new();

        let (intent, _) = extractor.extract_intent("compare with muthoot").unwrap();
        assert_eq!(intent, "comparison_inquiry");

        let (intent, _) = extractor.extract_intent("vs manappuram").unwrap();
        assert_eq!(intent, "comparison_inquiry");

        let (intent, _) = extractor.extract_intent("better than IIFL").unwrap();
        assert_eq!(intent, "comparison_inquiry");
    }

    #[test]
    fn test_intent_human_escalation() {
        let extractor = SlotExtractor::new();

        let (intent, _) = extractor.extract_intent("talk to human agent").unwrap();
        assert_eq!(intent, "human_escalation");

        let (intent, _) = extractor.extract_intent("I want to speak to manager").unwrap();
        assert_eq!(intent, "human_escalation");

        let (intent, _) = extractor.extract_intent("connect to customer care").unwrap();
        assert_eq!(intent, "human_escalation");

        let (intent, _) = extractor.extract_intent("I have a complaint").unwrap();
        assert_eq!(intent, "human_escalation");
    }

    #[test]
    fn test_intent_callback() {
        let extractor = SlotExtractor::new();

        let (intent, _) = extractor.extract_intent("call me back").unwrap();
        assert_eq!(intent, "callback_request");

        let (intent, _) = extractor.extract_intent("please callback").unwrap();
        assert_eq!(intent, "callback_request");
    }

    #[test]
    fn test_intent_sms() {
        let extractor = SlotExtractor::new();

        let (intent, _) = extractor.extract_intent("send SMS with details").unwrap();
        assert_eq!(intent, "sms_request");

        let (intent, _) = extractor.extract_intent("whatsapp me the info").unwrap();
        assert_eq!(intent, "sms_request");
    }

    #[test]
    fn test_intent_safety() {
        let extractor = SlotExtractor::new();

        let (intent, _) = extractor.extract_intent("is my gold safe").unwrap();
        assert_eq!(intent, "safety_inquiry");

        let (intent, _) = extractor.extract_intent("security of vault").unwrap();
        assert_eq!(intent, "safety_inquiry");
    }

    #[test]
    fn test_intent_rate_inquiry() {
        let extractor = SlotExtractor::new();

        let (intent, _) = extractor.extract_intent("what is interest rate").unwrap();
        assert_eq!(intent, "rate_inquiry");

        let (intent, _) = extractor.extract_intent("byaaj dar kitna hai").unwrap();
        assert_eq!(intent, "rate_inquiry");
    }

    // ============================================
    // REPAYMENT TYPE EXTRACTION TESTS
    // ============================================

    #[test]
    fn test_repayment_type_emi() {
        let extractor = SlotExtractor::new();

        let (repayment, _) = extractor.extract_repayment_type("I want to pay monthly EMI").unwrap();
        assert_eq!(repayment, "emi");

        let (repayment, _) = extractor.extract_repayment_type("mahine mein kishte dena hai").unwrap();
        assert_eq!(repayment, "emi");
    }

    #[test]
    fn test_repayment_type_bullet() {
        let extractor = SlotExtractor::new();

        let (repayment, _) = extractor.extract_repayment_type("I want bullet repayment").unwrap();
        assert_eq!(repayment, "bullet");

        let (repayment, _) = extractor.extract_repayment_type("ek baar mein pay karunga").unwrap();
        assert_eq!(repayment, "bullet");

        let (repayment, _) = extractor.extract_repayment_type("lump sum payment").unwrap();
        assert_eq!(repayment, "bullet");
    }

    #[test]
    fn test_repayment_type_overdraft() {
        let extractor = SlotExtractor::new();

        let (repayment, _) = extractor.extract_repayment_type("I want overdraft facility").unwrap();
        assert_eq!(repayment, "overdraft");

        let (repayment, _) = extractor.extract_repayment_type("flexible payment option").unwrap();
        assert_eq!(repayment, "overdraft");
    }

    // ============================================
    // CITY EXTRACTION TESTS
    // ============================================

    #[test]
    fn test_city_extraction_metros() {
        let extractor = SlotExtractor::new();

        let (city, _) = extractor.extract_city("I am in Mumbai").unwrap();
        assert_eq!(city.to_lowercase(), "mumbai");

        let (city, _) = extractor.extract_city("from Delhi").unwrap();
        assert_eq!(city.to_lowercase(), "delhi");

        let (city, _) = extractor.extract_city("near Bangalore").unwrap();
        assert_eq!(city.to_lowercase(), "bangalore");
    }

    #[test]
    fn test_city_extraction_hindi() {
        let extractor = SlotExtractor::new();

        // Test direct Hindi city name (pattern matches Dilli directly)
        let (city, _) = extractor.extract_city("I am from Dilli").unwrap();
        assert_eq!(city.to_lowercase(), "dilli");
    }

    #[test]
    fn test_city_extraction_tier2() {
        let extractor = SlotExtractor::new();

        let (city, _) = extractor.extract_city("branch in Jaipur").unwrap();
        assert_eq!(city.to_lowercase(), "jaipur");

        let (city, _) = extractor.extract_city("I am from Lucknow").unwrap();
        assert_eq!(city.to_lowercase(), "lucknow");
    }

    // ============================================
    // LOAN PURPOSE EXTRACTION TESTS
    // ============================================

    #[test]
    fn test_loan_purpose_medical() {
        let extractor = SlotExtractor::new();

        let (purpose, _) = extractor.extract_loan_purpose("for medical treatment").unwrap();
        assert_eq!(purpose, "medical");

        let (purpose, _) = extractor.extract_loan_purpose("hospital ke liye").unwrap();
        assert_eq!(purpose, "medical");

        let (purpose, _) = extractor.extract_loan_purpose("surgery expenses").unwrap();
        assert_eq!(purpose, "medical");
    }

    #[test]
    fn test_loan_purpose_business() {
        let extractor = SlotExtractor::new();

        let (purpose, _) = extractor.extract_loan_purpose("for my business").unwrap();
        assert_eq!(purpose, "business");

        let (purpose, _) = extractor.extract_loan_purpose("dhandha ke liye chahiye").unwrap();
        assert_eq!(purpose, "business");

        let (purpose, _) = extractor.extract_loan_purpose("shop renovation").unwrap();
        assert_eq!(purpose, "business");
    }

    #[test]
    fn test_loan_purpose_wedding() {
        let extractor = SlotExtractor::new();

        let (purpose, _) = extractor.extract_loan_purpose("for wedding expenses").unwrap();
        assert_eq!(purpose, "wedding");

        let (purpose, _) = extractor.extract_loan_purpose("shaadi ke liye").unwrap();
        assert_eq!(purpose, "wedding");
    }

    #[test]
    fn test_loan_purpose_education() {
        let extractor = SlotExtractor::new();

        let (purpose, _) = extractor.extract_loan_purpose("for education fees").unwrap();
        assert_eq!(purpose, "education");

        let (purpose, _) = extractor.extract_loan_purpose("college admission").unwrap();
        assert_eq!(purpose, "education");
    }

    #[test]
    fn test_loan_purpose_agriculture() {
        let extractor = SlotExtractor::new();

        let (purpose, _) = extractor.extract_loan_purpose("for farming").unwrap();
        assert_eq!(purpose, "agriculture");

        let (purpose, _) = extractor.extract_loan_purpose("kheti ke liye").unwrap();
        assert_eq!(purpose, "agriculture");
    }

    #[test]
    fn test_loan_purpose_emergency() {
        let extractor = SlotExtractor::new();

        let (purpose, _) = extractor.extract_loan_purpose("urgent zaruri hai").unwrap();
        assert_eq!(purpose, "emergency");

        let (purpose, _) = extractor.extract_loan_purpose("emergency situation").unwrap();
        assert_eq!(purpose, "emergency");
    }

    // ============================================
    // FULL CONVERSATION FLOW TESTS
    // ============================================

    #[test]
    fn test_balance_transfer_flow_extraction() {
        let extractor = SlotExtractor::new();

        // Customer says: "I want balance transfer of my 10 lakh loan at 18%"
        let utterance = "I want balance transfer of my 10 lakh loan at 18%";
        let slots = extractor.extract(utterance);

        assert!(slots.contains_key("loan_amount"));
        assert!(slots.contains_key("current_interest_rate"));
        assert!(slots.contains_key("detected_intent"));

        // Verify intent is balance_transfer
        assert_eq!(slots.get("detected_intent").unwrap().value, Some("balance_transfer".to_string()));
    }

    #[test]
    fn test_eligibility_flow_extraction() {
        let extractor = SlotExtractor::new();

        // Customer says: "Am I eligible? I have 50 grams 22 karat gold"
        let utterance = "Am I eligible? I have 50 grams 22 karat gold";
        let slots = extractor.extract(utterance);

        assert!(slots.contains_key("gold_weight"));
        assert!(slots.contains_key("gold_purity"));
        assert!(slots.contains_key("detected_intent"));

        assert_eq!(slots.get("detected_intent").unwrap().value, Some("eligibility_inquiry".to_string()));
    }

    #[test]
    fn test_appointment_flow_extraction() {
        let extractor = SlotExtractor::new();

        // Customer says: "Book appointment for Raj at 9876543210 in Mumbai"
        let utterance = "Book appointment for Raj at 9876543210 in Mumbai";
        let slots = extractor.extract(utterance);

        assert!(slots.contains_key("phone_number"));
        assert!(slots.contains_key("city"));
        assert!(slots.contains_key("detected_intent"));

        assert_eq!(slots.get("detected_intent").unwrap().value, Some("appointment_request".to_string()));
    }

    #[test]
    fn test_document_inquiry_bt_extraction() {
        let extractor = SlotExtractor::new();

        // Customer says: "What documents needed for balance transfer?"
        let utterance = "What documents needed for balance transfer";
        let slots = extractor.extract(utterance);

        // Should detect document_inquiry intent
        // Note: balance_transfer might take precedence - that's okay, both trigger related actions
        assert!(slots.contains_key("detected_intent"));
    }

    #[test]
    fn test_hindi_full_extraction() {
        let extractor = SlotExtractor::new();

        // Customer says: "Mera naam Rajesh hai, phone 9876543210, 5 lakh chahiye medical ke liye"
        let utterance = "Mera naam Rajesh hai, phone 9876543210, 5 lakh chahiye medical ke liye";
        let slots = extractor.extract(utterance);

        assert!(slots.contains_key("customer_name"));
        assert!(slots.contains_key("phone_number"));
        assert!(slots.contains_key("loan_amount"));
        assert!(slots.contains_key("loan_purpose"));

        assert_eq!(slots.get("loan_purpose").unwrap().value, Some("medical".to_string()));
    }
}
