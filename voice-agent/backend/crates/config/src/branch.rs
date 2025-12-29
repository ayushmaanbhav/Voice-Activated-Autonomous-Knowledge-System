//! Branch and location configuration
//!
//! Manages branch information for:
//! - Branch locator functionality
//! - Doorstep service availability
//! - Operating hours
//! - Special services (women-only, premium, etc.)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Branch configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchConfig {
    /// List of branches
    #[serde(default)]
    pub branches: Vec<Branch>,
    /// City-wise branch counts (for quick lookup)
    #[serde(default)]
    pub city_coverage: HashMap<String, usize>,
    /// States with coverage
    #[serde(default)]
    pub states: Vec<String>,
    /// Total branch count
    #[serde(default = "default_branch_count")]
    pub total_branches: usize,
    /// Doorstep service config
    #[serde(default)]
    pub doorstep_service: DoorstepServiceConfig,
}

fn default_branch_count() -> usize {
    1600
}

impl Default for BranchConfig {
    fn default() -> Self {
        let mut config = Self {
            branches: Vec::new(),
            city_coverage: HashMap::new(),
            states: vec![
                "Maharashtra".to_string(),
                "Gujarat".to_string(),
                "Karnataka".to_string(),
                "Tamil Nadu".to_string(),
                "Kerala".to_string(),
                "Andhra Pradesh".to_string(),
                "Telangana".to_string(),
                "Delhi".to_string(),
                "Uttar Pradesh".to_string(),
                "Rajasthan".to_string(),
                "Madhya Pradesh".to_string(),
                "West Bengal".to_string(),
                "Punjab".to_string(),
                "Haryana".to_string(),
            ],
            total_branches: default_branch_count(),
            doorstep_service: DoorstepServiceConfig::default(),
        };

        // Add sample metro city coverage
        config.city_coverage.insert("Mumbai".to_string(), 120);
        config.city_coverage.insert("Delhi".to_string(), 85);
        config.city_coverage.insert("Bangalore".to_string(), 65);
        config.city_coverage.insert("Chennai".to_string(), 55);
        config.city_coverage.insert("Hyderabad".to_string(), 50);
        config.city_coverage.insert("Pune".to_string(), 45);
        config.city_coverage.insert("Ahmedabad".to_string(), 40);
        config.city_coverage.insert("Kolkata".to_string(), 35);

        config
    }
}

/// Individual branch information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    /// Branch ID/code
    pub id: String,
    /// Branch name
    pub name: String,
    /// City
    pub city: String,
    /// State
    pub state: String,
    /// Pincode
    pub pincode: String,
    /// Full address
    pub address: String,
    /// Contact phone
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phone: Option<String>,
    /// Location coordinates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coordinates: Option<Coordinates>,
    /// Operating hours
    #[serde(default)]
    pub hours: OperatingHours,
    /// Branch features
    #[serde(default)]
    pub features: BranchFeatures,
    /// Is this branch active
    #[serde(default = "default_true")]
    pub active: bool,
}

fn default_true() -> bool {
    true
}

/// Geographic coordinates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coordinates {
    pub latitude: f64,
    pub longitude: f64,
}

/// Operating hours
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatingHours {
    /// Weekday opening time (HH:MM)
    #[serde(default = "default_open_time")]
    pub weekday_open: String,
    /// Weekday closing time (HH:MM)
    #[serde(default = "default_close_time")]
    pub weekday_close: String,
    /// Saturday opening time
    #[serde(default = "default_open_time")]
    pub saturday_open: String,
    /// Saturday closing time
    #[serde(default = "default_saturday_close")]
    pub saturday_close: String,
    /// Open on Sunday
    #[serde(default)]
    pub sunday_open: bool,
    /// Holiday schedule notes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub holiday_note: Option<String>,
}

fn default_open_time() -> String {
    "09:30".to_string()
}

fn default_close_time() -> String {
    "17:30".to_string()
}

fn default_saturday_close() -> String {
    "14:00".to_string()
}

impl Default for OperatingHours {
    fn default() -> Self {
        Self {
            weekday_open: default_open_time(),
            weekday_close: default_close_time(),
            saturday_open: default_open_time(),
            saturday_close: default_saturday_close(),
            sunday_open: false,
            holiday_note: None,
        }
    }
}

/// Branch features/capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BranchFeatures {
    /// Offers gold loan services
    #[serde(default = "default_true")]
    pub gold_loan: bool,
    /// Has locker facility
    #[serde(default)]
    pub locker: bool,
    /// Women-priority service
    #[serde(default)]
    pub women_priority: bool,
    /// Premium/priority banking
    #[serde(default)]
    pub premium_banking: bool,
    /// NRI services
    #[serde(default)]
    pub nri_services: bool,
    /// Doorstep service available from this branch
    #[serde(default)]
    pub doorstep_available: bool,
    /// Languages spoken at this branch
    #[serde(default)]
    pub languages: Vec<String>,
}

/// Doorstep service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoorstepServiceConfig {
    /// Doorstep service enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Minimum loan amount for doorstep service
    #[serde(default = "default_doorstep_min")]
    pub min_loan_amount: f64,
    /// Available in these cities
    #[serde(default)]
    pub available_cities: Vec<String>,
    /// Service hours
    #[serde(default)]
    pub hours: OperatingHours,
    /// Advance booking required (hours)
    #[serde(default = "default_booking_hours")]
    pub booking_advance_hours: u32,
}

fn default_doorstep_min() -> f64 {
    50000.0
}

fn default_booking_hours() -> u32 {
    24
}

impl Default for DoorstepServiceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_loan_amount: default_doorstep_min(),
            available_cities: vec![
                "Mumbai".to_string(),
                "Delhi".to_string(),
                "Bangalore".to_string(),
                "Chennai".to_string(),
                "Hyderabad".to_string(),
                "Pune".to_string(),
                "Ahmedabad".to_string(),
                "Kolkata".to_string(),
            ],
            hours: OperatingHours::default(),
            booking_advance_hours: default_booking_hours(),
        }
    }
}

impl BranchConfig {
    /// Find branches by city
    pub fn find_by_city(&self, city: &str) -> Vec<&Branch> {
        let city_lower = city.to_lowercase();
        self.branches
            .iter()
            .filter(|b| b.city.to_lowercase() == city_lower && b.active)
            .collect()
    }

    /// Find branches by pincode
    pub fn find_by_pincode(&self, pincode: &str) -> Vec<&Branch> {
        self.branches
            .iter()
            .filter(|b| b.pincode == pincode && b.active)
            .collect()
    }

    /// Find nearest branches by coordinates
    pub fn find_nearest(&self, lat: f64, lon: f64, limit: usize) -> Vec<(&Branch, f64)> {
        let mut with_distance: Vec<_> = self
            .branches
            .iter()
            .filter(|b| b.active && b.coordinates.is_some())
            .map(|b| {
                let coords = b.coordinates.as_ref().unwrap();
                let dist = haversine_distance(lat, lon, coords.latitude, coords.longitude);
                (b, dist)
            })
            .collect();

        with_distance.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        with_distance.into_iter().take(limit).collect()
    }

    /// Get branch count for city
    pub fn get_city_count(&self, city: &str) -> usize {
        self.city_coverage
            .get(&city.to_string())
            .copied()
            .unwrap_or(0)
    }

    /// Check if doorstep service available in city
    pub fn doorstep_available(&self, city: &str) -> bool {
        let city_lower = city.to_lowercase();
        self.doorstep_service.enabled
            && self.doorstep_service
                .available_cities
                .iter()
                .any(|c| c.to_lowercase() == city_lower)
    }

    /// Find branches with specific feature
    pub fn find_with_feature(&self, feature: &str) -> Vec<&Branch> {
        self.branches
            .iter()
            .filter(|b| {
                b.active
                    && match feature.to_lowercase().as_str() {
                        "women" | "women_priority" => b.features.women_priority,
                        "premium" | "priority" => b.features.premium_banking,
                        "locker" => b.features.locker,
                        "doorstep" => b.features.doorstep_available,
                        "nri" => b.features.nri_services,
                        _ => false,
                    }
            })
            .collect()
    }

    /// Get coverage summary text
    pub fn coverage_summary(&self) -> String {
        format!(
            "{} branches across {} states in {} cities",
            self.total_branches,
            self.states.len(),
            self.city_coverage.len()
        )
    }
}

/// Calculate distance between two coordinates using Haversine formula
fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS_KM: f64 = 6371.0;

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    EARTH_RADIUS_KM * c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BranchConfig::default();
        assert_eq!(config.total_branches, 1600);
        assert!(!config.states.is_empty());
        assert!(config.city_coverage.get("Mumbai").is_some());
    }

    #[test]
    fn test_doorstep_availability() {
        let config = BranchConfig::default();
        assert!(config.doorstep_available("Mumbai"));
        assert!(config.doorstep_available("MUMBAI"));
        assert!(!config.doorstep_available("SmallTown"));
    }

    #[test]
    fn test_coverage_summary() {
        let config = BranchConfig::default();
        let summary = config.coverage_summary();
        assert!(summary.contains("1600"));
        assert!(summary.contains("states"));
    }

    #[test]
    fn test_haversine_distance() {
        // Mumbai to Delhi approximately 1150 km
        let dist = haversine_distance(19.0760, 72.8777, 28.6139, 77.2090);
        assert!(dist > 1100.0 && dist < 1200.0);
    }

    #[test]
    fn test_operating_hours() {
        let hours = OperatingHours::default();
        assert_eq!(hours.weekday_open, "09:30");
        assert_eq!(hours.weekday_close, "17:30");
        assert!(!hours.sunday_open);
    }

    #[test]
    fn test_find_by_city() {
        let mut config = BranchConfig::default();
        config.branches.push(Branch {
            id: "MUM001".to_string(),
            name: "Andheri Branch".to_string(),
            city: "Mumbai".to_string(),
            state: "Maharashtra".to_string(),
            pincode: "400069".to_string(),
            address: "Andheri West, Mumbai".to_string(),
            phone: Some("022-12345678".to_string()),
            coordinates: Some(Coordinates {
                latitude: 19.1197,
                longitude: 72.8464,
            }),
            hours: OperatingHours::default(),
            features: BranchFeatures::default(),
            active: true,
        });

        let branches = config.find_by_city("Mumbai");
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].name, "Andheri Branch");
    }
}
