//! Location Data Management
//!
//! Handles loading and managing service location data for domain tools.
//! This is domain-agnostic - the actual locations come from domain config.

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Location/branch data structure for service locations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchData {
    pub branch_id: String,
    pub name: String,
    pub city: String,
    pub area: String,
    pub address: String,
    #[serde(default)]
    pub pincode: String,
    pub phone: String,
    /// Whether the primary service is available at this location
    #[serde(alias = "gold_loan_available")]
    pub service_available: bool,
    pub timing: String,
    #[serde(default)]
    pub facilities: Vec<String>,
}

/// Branch data file structure
#[derive(Debug, Deserialize)]
struct BranchDataFile {
    branches: Vec<BranchData>,
}

/// Get default paths for location data files.
/// Checks environment variable first, then falls back to common relative paths.
fn default_data_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // Environment variable override (highest priority)
    if let Ok(data_dir) = std::env::var("VOICE_AGENT_DATA_DIR") {
        paths.push(PathBuf::from(&data_dir).join("branches.json"));
    }

    // Config directory from environment
    if let Ok(config_dir) = std::env::var("VOICE_AGENT_CONFIG_DIR") {
        paths.push(PathBuf::from(&config_dir).join("data/branches.json"));
    }

    // Executable-relative path
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            paths.push(exe_dir.join("data/branches.json"));
        }
    }

    // Common relative paths (fallback)
    paths.extend([
        PathBuf::from("data/branches.json"),
        PathBuf::from("../data/branches.json"),
        PathBuf::from("../../data/branches.json"),
    ]);

    paths
}

/// Global location data loaded from config/JSON
/// Note: Locations should be loaded from domain config. This is a runtime cache.
static BRANCH_DATA: Lazy<RwLock<Vec<BranchData>>> = Lazy::new(|| {
    // Try to load from default paths (env var, exe-relative, then common paths)
    for path in default_data_paths() {
        if let Ok(data) = load_branches_from_file(&path) {
            tracing::info!("Loaded {} locations from {}", data.len(), path.display());
            return RwLock::new(data);
        }
    }

    // Return empty - locations must come from config
    tracing::warn!("No location data file found - locations should be loaded from domain config");
    RwLock::new(Vec::new())
});

/// Load branches from a JSON file
pub fn load_branches_from_file<P: AsRef<Path>>(path: P) -> Result<Vec<BranchData>, std::io::Error> {
    let content = std::fs::read_to_string(path)?;
    let file: BranchDataFile = serde_json::from_str(&content)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(file.branches)
}

/// Reload branches from a file (for runtime updates)
pub fn reload_branches<P: AsRef<Path>>(path: P) -> Result<usize, std::io::Error> {
    let branches = load_branches_from_file(path)?;
    let count = branches.len();
    *BRANCH_DATA.write() = branches;
    Ok(count)
}

/// Get all loaded branches
pub fn get_branches() -> Vec<BranchData> {
    BRANCH_DATA.read().clone()
}

/// Initialize locations from config data
///
/// This should be called during startup to populate locations from domain config.
pub fn initialize_locations(locations: Vec<BranchData>) {
    let count = locations.len();
    *BRANCH_DATA.write() = locations;
    tracing::info!("Initialized {} service locations from config", count);
}

/// Find service locations by criteria
///
/// Filters locations by city, pincode, and/or area.
/// Only returns locations where the service is available.
pub fn find_locations(
    city: Option<&str>,
    pincode: Option<&str>,
    area: Option<&str>,
) -> Vec<BranchData> {
    let all_locations = get_branches();

    all_locations
        .into_iter()
        .filter(|b| {
            // Only include locations where service is available
            if !b.service_available {
                return false;
            }

            // Filter by city
            if let Some(c) = city {
                if !b.city.to_lowercase().contains(&c.to_lowercase()) {
                    return false;
                }
            }

            // Filter by pincode
            if let Some(p) = pincode {
                if !b.pincode.starts_with(p) && !b.address.contains(p) {
                    return false;
                }
            }

            // Filter by area
            if let Some(a) = area {
                if !b.area.to_lowercase().contains(&a.to_lowercase())
                    && !b.address.to_lowercase().contains(&a.to_lowercase())
                {
                    return false;
                }
            }

            true
        })
        .collect()
}

/// Alias for backward compatibility
#[deprecated(note = "Use find_locations instead")]
pub fn get_mock_branches(
    city: Option<&str>,
    pincode: Option<&str>,
    area: Option<&str>,
) -> Vec<BranchData> {
    find_locations(city, pincode, area)
}
