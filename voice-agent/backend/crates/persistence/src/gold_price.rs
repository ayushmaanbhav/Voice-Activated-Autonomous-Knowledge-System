//! Simulated gold price service with ScyllaDB persistence
//!
//! Provides realistic gold price simulation with:
//! - Daily fluctuation within configurable bounds
//! - Price caching in ScyllaDB
//! - Historical price tracking

use async_trait::async_trait;
use chrono::{DateTime, Utc, Datelike, Timelike, NaiveDate};
use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::{ScyllaClient, PersistenceError};

/// Gold price data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldPrice {
    /// Price per gram (default purity, usually 22k)
    pub price_per_gram: f64,
    /// 24 karat (99.9% pure) price per gram
    pub price_24k: f64,
    /// 22 karat (91.6% pure) price per gram
    pub price_22k: f64,
    /// 18 karat (75% pure) price per gram
    pub price_18k: f64,
    /// Source of the price
    pub source: String,
    /// When the price was last updated
    pub updated_at: DateTime<Utc>,
}

impl GoldPrice {
    /// Calculate maximum loan amount based on gold weight and LTV ratio
    pub fn calculate_max_loan(&self, gold_weight_grams: f64, purity: GoldPurity, ltv_ratio: f64) -> f64 {
        let price = match purity {
            GoldPurity::K24 => self.price_24k,
            GoldPurity::K22 => self.price_22k,
            GoldPurity::K18 => self.price_18k,
        };
        gold_weight_grams * price * ltv_ratio
    }
}

/// Gold purity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoldPurity {
    K24, // 99.9% pure
    K22, // 91.6% pure (most jewelry)
    K18, // 75% pure
}

/// Gold price service trait
#[async_trait]
pub trait GoldPriceService: Send + Sync {
    /// Get the current gold price
    async fn get_current_price(&self) -> Result<GoldPrice, PersistenceError>;

    /// Get historical price for a specific date
    async fn get_historical_price(&self, date: NaiveDate) -> Result<Option<GoldPrice>, PersistenceError>;

    /// Force refresh the price (even if cache is valid)
    async fn refresh_price(&self) -> Result<GoldPrice, PersistenceError>;
}

/// Simulated gold price service
#[derive(Clone)]
pub struct SimulatedGoldPriceService {
    client: ScyllaClient,
    base_price_24k: f64,
    fluctuation_percent: f64,
    cache_ttl_seconds: i64,
}

impl SimulatedGoldPriceService {
    /// Create a new simulated gold price service
    ///
    /// # Arguments
    /// * `client` - ScyllaDB client
    /// * `base_price_24k` - Base price for 24k gold in INR per gram (e.g., 7500.0)
    pub fn new(client: ScyllaClient, base_price_24k: f64) -> Self {
        Self {
            client,
            base_price_24k,
            fluctuation_percent: 2.0, // ±2% daily fluctuation
            cache_ttl_seconds: 300,   // 5 minute cache
        }
    }

    /// Set custom fluctuation percentage
    pub fn with_fluctuation(mut self, percent: f64) -> Self {
        self.fluctuation_percent = percent;
        self
    }

    /// Set custom cache TTL
    pub fn with_cache_ttl(mut self, seconds: i64) -> Self {
        self.cache_ttl_seconds = seconds;
        self
    }

    /// Generate a simulated price with realistic fluctuation
    fn generate_price(&self) -> GoldPrice {
        let mut rng = rand::thread_rng();

        // Generate fluctuation: -fluctuation_percent% to +fluctuation_percent%
        let fluctuation = (rng.gen::<f64>() - 0.5) * 2.0 * (self.fluctuation_percent / 100.0);
        let price_24k = self.base_price_24k * (1.0 + fluctuation);

        // Calculate other purities based on 24k
        let price_22k = price_24k * 0.916; // 22k = 91.6% pure
        let price_18k = price_24k * 0.75;  // 18k = 75% pure

        GoldPrice {
            price_per_gram: price_22k, // Default to 22k (most common for jewelry)
            price_24k,
            price_22k,
            price_18k,
            source: "simulated".to_string(),
            updated_at: Utc::now(),
        }
    }

    /// Get cached price from ScyllaDB
    async fn get_cached_price(&self) -> Result<Option<GoldPrice>, PersistenceError> {
        let query = format!(
            "SELECT price_per_gram, price_24k, price_22k, price_18k, updated_at, source
             FROM {}.gold_price_latest WHERE singleton = 1",
            self.client.keyspace()
        );

        let result = self.client.session()
            .query_unpaged(query, &[])
            .await?;

        if let Some(rows) = result.rows {
            if let Some(row) = rows.into_iter().next() {
                let (
                    price_per_gram,
                    price_24k,
                    price_22k,
                    price_18k,
                    updated_at,
                    source,
                ): (f64, f64, f64, f64, i64, String) = row.into_typed()
                    .map_err(|e| PersistenceError::InvalidData(e.to_string()))?;

                return Ok(Some(GoldPrice {
                    price_per_gram,
                    price_24k,
                    price_22k,
                    price_18k,
                    source,
                    updated_at: DateTime::from_timestamp_millis(updated_at).unwrap_or_else(Utc::now),
                }));
            }
        }

        Ok(None)
    }

    /// Update the latest price cache
    async fn update_cache(&self, price: &GoldPrice) -> Result<(), PersistenceError> {
        let query = format!(
            "INSERT INTO {}.gold_price_latest (
                singleton, price_per_gram, price_24k, price_22k, price_18k, updated_at, source
            ) VALUES (1, ?, ?, ?, ?, ?, ?)",
            self.client.keyspace()
        );

        self.client.session().query_unpaged(
            query,
            (
                price.price_per_gram,
                price.price_24k,
                price.price_22k,
                price.price_18k,
                price.updated_at.timestamp_millis(),
                &price.source,
            ),
        ).await?;

        Ok(())
    }

    /// Record price in history table
    async fn record_history(&self, price: &GoldPrice) -> Result<(), PersistenceError> {
        let now = Utc::now();
        let date = now.date_naive();
        let hour = now.hour() as i32;

        let query = format!(
            "INSERT INTO {}.gold_prices (
                date, hour, price_per_gram, price_24k, price_22k, price_18k, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            self.client.keyspace()
        );

        self.client.session().query_unpaged(
            query,
            (
                date.to_string(),
                hour,
                price.price_per_gram,
                price.price_24k,
                price.price_22k,
                price.price_18k,
                &price.source,
                now.timestamp_millis(),
            ),
        ).await?;

        Ok(())
    }
}

#[async_trait]
impl GoldPriceService for SimulatedGoldPriceService {
    async fn get_current_price(&self) -> Result<GoldPrice, PersistenceError> {
        // Check cache first
        if let Some(cached) = self.get_cached_price().await? {
            let age = Utc::now() - cached.updated_at;
            if age.num_seconds() < self.cache_ttl_seconds {
                tracing::debug!(
                    age_seconds = age.num_seconds(),
                    "Returning cached gold price"
                );
                return Ok(cached);
            }
        }

        // Generate new price
        let price = self.generate_price();

        // Update cache and history
        self.update_cache(&price).await?;
        self.record_history(&price).await?;

        tracing::info!(
            price_24k = price.price_24k,
            price_22k = price.price_22k,
            "Generated new simulated gold price"
        );

        Ok(price)
    }

    async fn get_historical_price(&self, date: NaiveDate) -> Result<Option<GoldPrice>, PersistenceError> {
        let query = format!(
            "SELECT price_per_gram, price_24k, price_22k, price_18k, source, created_at
             FROM {}.gold_prices WHERE date = ? LIMIT 1",
            self.client.keyspace()
        );

        let result = self.client.session()
            .query_unpaged(query, (date.to_string(),))
            .await?;

        if let Some(rows) = result.rows {
            if let Some(row) = rows.into_iter().next() {
                let (
                    price_per_gram,
                    price_24k,
                    price_22k,
                    price_18k,
                    source,
                    created_at,
                ): (f64, f64, f64, f64, String, i64) = row.into_typed()
                    .map_err(|e| PersistenceError::InvalidData(e.to_string()))?;

                return Ok(Some(GoldPrice {
                    price_per_gram,
                    price_24k,
                    price_22k,
                    price_18k,
                    source,
                    updated_at: DateTime::from_timestamp_millis(created_at).unwrap_or_else(Utc::now),
                }));
            }
        }

        Ok(None)
    }

    async fn refresh_price(&self) -> Result<GoldPrice, PersistenceError> {
        let price = self.generate_price();
        self.update_cache(&price).await?;
        self.record_history(&price).await?;
        Ok(price)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gold_price_calculation() {
        let price = GoldPrice {
            price_per_gram: 6870.0,
            price_24k: 7500.0,
            price_22k: 6870.0,
            price_18k: 5625.0,
            source: "test".to_string(),
            updated_at: Utc::now(),
        };

        // 100 grams of 22k gold at 75% LTV
        let max_loan = price.calculate_max_loan(100.0, GoldPurity::K22, 0.75);
        assert!((max_loan - 515250.0).abs() < 1.0); // 100 * 6870 * 0.75 = 515250
    }

    #[test]
    fn test_price_generation_bounds() {
        let service = SimulatedGoldPriceService {
            client: unsafe { std::mem::zeroed() }, // Won't be used in this test
            base_price_24k: 7500.0,
            fluctuation_percent: 2.0,
            cache_ttl_seconds: 300,
        };

        // Generate 100 prices and check they're within bounds
        for _ in 0..100 {
            let price = service.generate_price();
            assert!(price.price_24k >= 7350.0 && price.price_24k <= 7650.0); // ±2%
            assert!(price.price_22k < price.price_24k);
            assert!(price.price_18k < price.price_22k);
        }
    }
}
