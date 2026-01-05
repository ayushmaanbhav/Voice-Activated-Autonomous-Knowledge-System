//! Number to Words Conversion
//!
//! Converts numeric values to spoken words using Indian numbering system.
//! Supports: integers, decimals, currency, percentages, dates, phone numbers.

use once_cell::sync::Lazy;
use regex::Regex;
use voice_agent_core::Language;

/// Indian numbering system converter
pub struct IndianNumberSystem;

impl IndianNumberSystem {
    /// Convert integer to Indian words (English)
    pub fn to_words_en(n: u64) -> String {
        if n == 0 {
            return "zero".to_string();
        }

        let mut result = String::new();
        let mut remaining = n;

        // Crores (10 million)
        if remaining >= 10_000_000 {
            let crores = remaining / 10_000_000;
            result.push_str(&Self::small_number_to_words_en(crores));
            result.push_str(" crore ");
            remaining %= 10_000_000;
        }

        // Lakhs (100 thousand)
        if remaining >= 100_000 {
            let lakhs = remaining / 100_000;
            result.push_str(&Self::small_number_to_words_en(lakhs));
            result.push_str(" lakh ");
            remaining %= 100_000;
        }

        // Thousands
        if remaining >= 1_000 {
            let thousands = remaining / 1_000;
            result.push_str(&Self::small_number_to_words_en(thousands));
            result.push_str(" thousand ");
            remaining %= 1_000;
        }

        // Hundreds
        if remaining >= 100 {
            let hundreds = remaining / 100;
            result.push_str(&Self::small_number_to_words_en(hundreds));
            result.push_str(" hundred ");
            remaining %= 100;
        }

        // Remaining (0-99)
        if remaining > 0 {
            result.push_str(&Self::small_number_to_words_en(remaining));
        }

        result.trim().to_string()
    }

    /// Convert integer to Indian words (Hindi transliteration)
    pub fn to_words_hi(n: u64) -> String {
        if n == 0 {
            return "shunya".to_string();
        }

        let mut result = String::new();
        let mut remaining = n;

        // Crores
        if remaining >= 10_000_000 {
            let crores = remaining / 10_000_000;
            result.push_str(&Self::small_number_to_words_hi(crores));
            result.push_str(" crore ");
            remaining %= 10_000_000;
        }

        // Lakhs
        if remaining >= 100_000 {
            let lakhs = remaining / 100_000;
            result.push_str(&Self::small_number_to_words_hi(lakhs));
            result.push_str(" lakh ");
            remaining %= 100_000;
        }

        // Hazaar (thousands)
        if remaining >= 1_000 {
            let thousands = remaining / 1_000;
            result.push_str(&Self::small_number_to_words_hi(thousands));
            result.push_str(" hazaar ");
            remaining %= 1_000;
        }

        // Sau (hundreds)
        if remaining >= 100 {
            let hundreds = remaining / 100;
            result.push_str(&Self::small_number_to_words_hi(hundreds));
            result.push_str(" sau ");
            remaining %= 100;
        }

        // Remaining
        if remaining > 0 {
            result.push_str(&Self::small_number_to_words_hi(remaining));
        }

        result.trim().to_string()
    }

    /// Convert small number (0-99) to English words
    fn small_number_to_words_en(n: u64) -> String {
        const ONES: &[&str] = &[
            "",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ];
        const TENS: &[&str] = &[
            "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        ];

        if n < 20 {
            ONES[n as usize].to_string()
        } else {
            let ten = (n / 10) as usize;
            let one = (n % 10) as usize;
            if one == 0 {
                TENS[ten].to_string()
            } else {
                format!("{} {}", TENS[ten], ONES[one])
            }
        }
    }

    /// Convert small number (0-99) to Hindi transliteration
    fn small_number_to_words_hi(n: u64) -> String {
        const HINDI_ONES: &[&str] = &[
            "", "ek", "do", "teen", "chaar", "paanch", "chheh", "saat", "aath", "nau", "das",
            "gyaarah", "baarah", "terah", "chaudah", "pandrah", "solah", "satrah", "athaarah",
            "unees",
        ];
        const HINDI_TENS: &[&str] = &[
            "", "", "bees", "tees", "chaalis", "pachaas", "saath", "sattar", "assi", "nabbe",
        ];

        if n < 20 {
            HINDI_ONES[n as usize].to_string()
        } else {
            let ten = (n / 10) as usize;
            let one = (n % 10) as usize;
            if one == 0 {
                HINDI_TENS[ten].to_string()
            } else {
                // Hindi has special forms for 21-99, simplified here
                format!("{} {}", HINDI_TENS[ten], HINDI_ONES[one])
            }
        }
    }
}

/// Number to words converter
pub struct NumberToWords {
    language: Language,
}

// Regex patterns for number detection
static CURRENCY_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"₹\s*(\d+(?:,\d+)*(?:\.\d{1,2})?)").unwrap());

static PERCENTAGE_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+(?:\.\d+)?)\s*%").unwrap());

static PHONE_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(\d{10})\b").unwrap());

static PLAIN_NUMBER_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b(\d+(?:,\d+)*(?:\.\d+)?)\b").unwrap());

static DATE_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b").unwrap());

impl NumberToWords {
    /// Create new converter for given language
    pub fn new(language: Language) -> Self {
        Self { language }
    }

    /// Convert all numbers in text to words
    pub fn convert(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Order matters: currency before plain numbers
        result = self.convert_currency(&result);
        result = self.convert_percentages(&result);
        result = self.convert_phones(&result);
        result = self.convert_dates(&result);
        result = self.convert_plain_numbers(&result);

        result
    }

    /// Convert currency (₹15000 → "fifteen thousand rupees")
    fn convert_currency(&self, text: &str) -> String {
        CURRENCY_PATTERN
            .replace_all(text, |caps: &regex::Captures| {
                let num_str = caps.get(1).unwrap().as_str().replace(',', "");
                if let Ok(amount) = num_str.parse::<f64>() {
                    let whole = amount.trunc() as u64;
                    let paise = ((amount.fract() * 100.0).round()) as u64;

                    let mut words = self.integer_to_words(whole);
                    words.push_str(" rupees");

                    if paise > 0 {
                        words.push_str(" and ");
                        words.push_str(&self.integer_to_words(paise));
                        words.push_str(" paise");
                    }

                    words
                } else {
                    caps.get(0).unwrap().as_str().to_string()
                }
            })
            .to_string()
    }

    /// Convert percentages (8.5% → "eight point five percent")
    fn convert_percentages(&self, text: &str) -> String {
        PERCENTAGE_PATTERN
            .replace_all(text, |caps: &regex::Captures| {
                let num_str = caps.get(1).unwrap().as_str();
                if let Ok(num) = num_str.parse::<f64>() {
                    let words = self.decimal_to_words(num);
                    format!("{} percent", words)
                } else {
                    caps.get(0).unwrap().as_str().to_string()
                }
            })
            .to_string()
    }

    /// Convert phone numbers (expand each digit for clarity)
    fn convert_phones(&self, text: &str) -> String {
        PHONE_PATTERN
            .replace_all(text, |caps: &regex::Captures| {
                let phone = caps.get(1).unwrap().as_str();
                phone
                    .chars()
                    .map(|c| self.digit_to_word(c))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .to_string()
    }

    /// Convert dates (15/01/2024 → "fifteenth January twenty twenty four")
    fn convert_dates(&self, text: &str) -> String {
        DATE_PATTERN
            .replace_all(text, |caps: &regex::Captures| {
                let day: u64 = caps.get(1).unwrap().as_str().parse().unwrap_or(1);
                let month: u64 = caps.get(2).unwrap().as_str().parse().unwrap_or(1);
                let year: u64 = caps.get(3).unwrap().as_str().parse().unwrap_or(2024);

                let day_word = self.ordinal(day);
                let month_word = self.month_name(month);
                let year_word = self.year_to_words(year);

                format!("{} {} {}", day_word, month_word, year_word)
            })
            .to_string()
    }

    /// Convert plain numbers
    fn convert_plain_numbers(&self, text: &str) -> String {
        PLAIN_NUMBER_PATTERN
            .replace_all(text, |caps: &regex::Captures| {
                let num_str = caps.get(1).unwrap().as_str().replace(',', "");
                if let Ok(num) = num_str.parse::<f64>() {
                    self.decimal_to_words(num)
                } else {
                    caps.get(0).unwrap().as_str().to_string()
                }
            })
            .to_string()
    }

    /// Convert integer to words
    fn integer_to_words(&self, n: u64) -> String {
        match self.language {
            Language::Hindi => IndianNumberSystem::to_words_hi(n),
            _ => IndianNumberSystem::to_words_en(n),
        }
    }

    /// Convert decimal to words
    fn decimal_to_words(&self, n: f64) -> String {
        let whole = n.trunc() as u64;
        let frac = n.fract();

        if frac.abs() < 0.001 {
            self.integer_to_words(whole)
        } else {
            // Convert fractional part digit by digit after "point"
            let frac_str = format!("{:.2}", frac);
            let frac_part = &frac_str[2..]; // Skip "0."

            // Strip trailing zeros to avoid "five zero" for 0.5
            let frac_part = frac_part.trim_end_matches('0');

            let mut result = self.integer_to_words(whole);
            result.push_str(" point ");

            for c in frac_part.chars() {
                // Skip leading zeros after "point" but keep interior zeros
                if c != '0' || !result.ends_with("point ") {
                    result.push_str(self.digit_to_word(c));
                    result.push(' ');
                }
            }

            result.trim().to_string()
        }
    }

    /// Convert single digit to word
    fn digit_to_word(&self, c: char) -> &'static str {
        match c {
            '0' => "zero",
            '1' => "one",
            '2' => "two",
            '3' => "three",
            '4' => "four",
            '5' => "five",
            '6' => "six",
            '7' => "seven",
            '8' => "eight",
            '9' => "nine",
            _ => "",
        }
    }

    /// Convert number to ordinal (1 → "first", 2 → "second")
    fn ordinal(&self, n: u64) -> String {
        match n {
            1 => "first".to_string(),
            2 => "second".to_string(),
            3 => "third".to_string(),
            4 => "fourth".to_string(),
            5 => "fifth".to_string(),
            6 => "sixth".to_string(),
            7 => "seventh".to_string(),
            8 => "eighth".to_string(),
            9 => "ninth".to_string(),
            10 => "tenth".to_string(),
            11 => "eleventh".to_string(),
            12 => "twelfth".to_string(),
            13 => "thirteenth".to_string(),
            14 => "fourteenth".to_string(),
            15 => "fifteenth".to_string(),
            16 => "sixteenth".to_string(),
            17 => "seventeenth".to_string(),
            18 => "eighteenth".to_string(),
            19 => "nineteenth".to_string(),
            20 => "twentieth".to_string(),
            21 => "twenty first".to_string(),
            22 => "twenty second".to_string(),
            23 => "twenty third".to_string(),
            24 => "twenty fourth".to_string(),
            25 => "twenty fifth".to_string(),
            26 => "twenty sixth".to_string(),
            27 => "twenty seventh".to_string(),
            28 => "twenty eighth".to_string(),
            29 => "twenty ninth".to_string(),
            30 => "thirtieth".to_string(),
            31 => "thirty first".to_string(),
            _ => format!("{}", n),
        }
    }

    /// Get month name
    fn month_name(&self, month: u64) -> &'static str {
        match month {
            1 => "January",
            2 => "February",
            3 => "March",
            4 => "April",
            5 => "May",
            6 => "June",
            7 => "July",
            8 => "August",
            9 => "September",
            10 => "October",
            11 => "November",
            12 => "December",
            _ => "Invalid",
        }
    }

    /// Convert year to words
    fn year_to_words(&self, year: u64) -> String {
        if year < 100 {
            // Two digit year: assume 2000s
            let full_year = 2000 + year;
            self.integer_to_words(full_year)
        } else if (2000..2100).contains(&year) {
            // 2024 → "twenty twenty four"
            let century = year / 100;
            let remainder = year % 100;
            if remainder == 0 {
                format!("{} hundred", self.integer_to_words(century))
            } else {
                format!(
                    "{} {}",
                    self.integer_to_words(century),
                    self.integer_to_words(remainder)
                )
            }
        } else {
            self.integer_to_words(year)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indian_number_system() {
        assert_eq!(IndianNumberSystem::to_words_en(0), "zero");
        assert_eq!(IndianNumberSystem::to_words_en(15), "fifteen");
        assert_eq!(IndianNumberSystem::to_words_en(100), "one hundred");
        assert_eq!(IndianNumberSystem::to_words_en(1000), "one thousand");
        assert_eq!(IndianNumberSystem::to_words_en(100000), "one lakh");
        assert_eq!(IndianNumberSystem::to_words_en(10000000), "one crore");
        assert_eq!(IndianNumberSystem::to_words_en(15000), "fifteen thousand");
        assert_eq!(
            IndianNumberSystem::to_words_en(150000),
            "one lakh fifty thousand"
        );
    }

    #[test]
    fn test_currency_conversion() {
        let converter = NumberToWords::new(Language::English);
        let result = converter.convert("₹15000");
        assert_eq!(result, "fifteen thousand rupees");
    }

    #[test]
    fn test_percentage_conversion() {
        let converter = NumberToWords::new(Language::English);
        let result = converter.convert("8.5%");
        assert!(result.contains("eight point five percent"));
    }

    #[test]
    fn test_phone_conversion() {
        let converter = NumberToWords::new(Language::English);
        let result = converter.convert("9876543210");
        assert!(result.contains("nine eight seven six"));
    }
}
