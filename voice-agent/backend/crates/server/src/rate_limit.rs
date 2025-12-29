//! Token bucket rate limiter for WebSocket connections
//!
//! Prevents DoS attacks by limiting messages and audio bytes per second.

use std::time::Instant;
use voice_agent_config::RateLimitConfig;

/// Token bucket rate limiter
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    /// Message tokens
    message_tokens: f32,
    /// Audio byte tokens
    audio_tokens: f32,
    /// Last refill time
    last_refill: Instant,
}

impl RateLimiter {
    /// Create a new rate limiter with the given config
    pub fn new(config: RateLimitConfig) -> Self {
        let burst_messages = config.messages_per_second as f32 * config.burst_multiplier;
        let burst_audio = config.audio_bytes_per_second as f32 * config.burst_multiplier;

        Self {
            config,
            message_tokens: burst_messages,
            audio_tokens: burst_audio,
            last_refill: Instant::now(),
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        let elapsed_secs = elapsed.as_secs_f32();

        if elapsed_secs > 0.0 {
            // Refill message tokens
            let message_refill = elapsed_secs * self.config.messages_per_second as f32;
            let max_messages = self.config.messages_per_second as f32 * self.config.burst_multiplier;
            self.message_tokens = (self.message_tokens + message_refill).min(max_messages);

            // Refill audio tokens
            let audio_refill = elapsed_secs * self.config.audio_bytes_per_second as f32;
            let max_audio = self.config.audio_bytes_per_second as f32 * self.config.burst_multiplier;
            self.audio_tokens = (self.audio_tokens + audio_refill).min(max_audio);

            self.last_refill = now;
        }
    }

    /// Check if a message can be sent (and consume a token if so)
    pub fn check_message(&mut self) -> Result<(), RateLimitError> {
        if !self.config.enabled {
            return Ok(());
        }

        self.refill();

        if self.message_tokens >= 1.0 {
            self.message_tokens -= 1.0;
            Ok(())
        } else {
            Err(RateLimitError::MessageRateExceeded)
        }
    }

    /// Check if audio bytes can be sent (and consume tokens if so)
    pub fn check_audio(&mut self, bytes: usize) -> Result<(), RateLimitError> {
        if !self.config.enabled {
            return Ok(());
        }

        self.refill();

        let bytes_f32 = bytes as f32;
        if self.audio_tokens >= bytes_f32 {
            self.audio_tokens -= bytes_f32;
            Ok(())
        } else {
            Err(RateLimitError::AudioRateExceeded)
        }
    }

    /// Get remaining message tokens (for diagnostics)
    pub fn remaining_message_tokens(&self) -> f32 {
        self.message_tokens
    }

    /// Get remaining audio tokens (for diagnostics)
    pub fn remaining_audio_tokens(&self) -> f32 {
        self.audio_tokens
    }
}

/// Rate limit errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitError {
    /// Too many messages per second
    MessageRateExceeded,
    /// Too much audio data per second
    AudioRateExceeded,
}

impl std::fmt::Display for RateLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RateLimitError::MessageRateExceeded => {
                write!(f, "Message rate limit exceeded")
            }
            RateLimitError::AudioRateExceeded => {
                write!(f, "Audio rate limit exceeded")
            }
        }
    }
}

impl std::error::Error for RateLimitError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_allows_under_limit() {
        let config = RateLimitConfig {
            enabled: true,
            messages_per_second: 10,
            audio_bytes_per_second: 1000,
            burst_multiplier: 2.0,
        };
        let mut limiter = RateLimiter::new(config);

        // Should allow up to burst limit (20 messages)
        for _ in 0..20 {
            assert!(limiter.check_message().is_ok());
        }
    }

    #[test]
    fn test_rate_limiter_blocks_over_limit() {
        let config = RateLimitConfig {
            enabled: true,
            messages_per_second: 10,
            audio_bytes_per_second: 1000,
            burst_multiplier: 1.0, // No burst
        };
        let mut limiter = RateLimiter::new(config);

        // Use up all tokens
        for _ in 0..10 {
            assert!(limiter.check_message().is_ok());
        }

        // Next one should fail
        assert!(limiter.check_message().is_err());
    }

    #[test]
    fn test_rate_limiter_disabled() {
        let config = RateLimitConfig {
            enabled: false,
            messages_per_second: 1,
            audio_bytes_per_second: 1,
            burst_multiplier: 1.0,
        };
        let mut limiter = RateLimiter::new(config);

        // Should always allow when disabled
        for _ in 0..1000 {
            assert!(limiter.check_message().is_ok());
        }
    }

    #[test]
    fn test_audio_rate_limiting() {
        let config = RateLimitConfig {
            enabled: true,
            messages_per_second: 100,
            audio_bytes_per_second: 1000,
            burst_multiplier: 1.0,
        };
        let mut limiter = RateLimiter::new(config);

        // Should allow 1000 bytes
        assert!(limiter.check_audio(500).is_ok());
        assert!(limiter.check_audio(500).is_ok());

        // Next should fail
        assert!(limiter.check_audio(100).is_err());
    }
}
