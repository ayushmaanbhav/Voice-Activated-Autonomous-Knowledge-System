//! Vector Store using Qdrant
//!
//! Dense vector storage and similarity search.

use qdrant_client::{
    qdrant::{
        value::Kind, Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance,
        FieldCondition, Filter, Match, PointId, PointStruct, PointsIdsList, SearchPointsBuilder,
        UpsertPointsBuilder, VectorParamsBuilder,
    },
    Qdrant,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
// P1 FIX: Use centralized constants
use voice_agent_config::constants::endpoints;

use crate::RagError;

/// Vector store configuration
#[derive(Debug, Clone)]
pub struct VectorStoreConfig {
    /// Qdrant endpoint
    pub endpoint: String,
    /// Collection name
    pub collection: String,
    /// Vector dimension
    pub vector_dim: usize,
    /// Distance metric
    pub distance: VectorDistance,
    /// API key (optional)
    pub api_key: Option<String>,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            endpoint: endpoints::QDRANT_DEFAULT.to_string(), // P1 FIX: Use centralized constant
            collection: "gold_loan_knowledge".to_string(),
            // P6 FIX: Match qwen3-embedding:0.6b which produces 1024 dims
            vector_dim: 1024,
            distance: VectorDistance::Cosine,
            api_key: None,
        }
    }
}

/// Distance metric
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorDistance {
    Cosine,
    Euclidean,
    DotProduct,
}

impl From<VectorDistance> for Distance {
    fn from(d: VectorDistance) -> Self {
        match d {
            VectorDistance::Cosine => Distance::Cosine,
            VectorDistance::Euclidean => Distance::Euclid,
            VectorDistance::DotProduct => Distance::Dot,
        }
    }
}

/// Document with metadata
///
/// P2-2 FIX: Renamed `text` to `content` to match core::Document.
/// Serde alias "text" kept for backwards compatibility with existing Qdrant data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique ID
    pub id: String,
    /// Document content (P2-2 FIX: renamed from `text`)
    #[serde(alias = "text")]
    pub content: String,
    /// Document title/source
    pub title: Option<String>,
    /// Category/type
    pub category: Option<String>,
    /// Language
    pub language: Option<String>,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Search result from vector store
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    /// Document ID
    pub id: String,
    /// Similarity score
    pub score: f32,
    /// Document content (P2-2 FIX: renamed from `text`)
    pub content: String,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Vector store client
pub struct VectorStore {
    client: Qdrant,
    config: VectorStoreConfig,
}

impl VectorStore {
    /// Create a new vector store connection
    ///
    /// P0 FIX: Now uses api_key from config for authenticated Qdrant connections.
    pub async fn new(config: VectorStoreConfig) -> Result<Self, RagError> {
        // Use gRPC port (6334) for qdrant-client
        let grpc_endpoint = config.endpoint.replace(":6333", ":6334");
        tracing::info!("Connecting to Qdrant at {}", grpc_endpoint);

        let mut builder = Qdrant::from_url(&grpc_endpoint)
            // Skip version check to avoid connection issues
            .skip_compatibility_check()
            // Increase timeouts for slower connections
            .timeout(std::time::Duration::from_secs(30))
            .connect_timeout(std::time::Duration::from_secs(10));

        // P0 FIX: Apply API key if configured
        if let Some(ref api_key) = config.api_key {
            builder = builder.api_key(api_key.clone());
            tracing::info!("Qdrant connection using API key authentication");
        }

        let client = builder
            .build()
            .map_err(|e| RagError::Connection(e.to_string()))?;

        Ok(Self { client, config })
    }

    /// Create collection if not exists
    pub async fn ensure_collection(&self) -> Result<(), RagError> {
        let exists = self
            .client
            .collection_exists(&self.config.collection)
            .await
            .map_err(|e| RagError::VectorStore(e.to_string()))?;

        if !exists {
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.config.collection).vectors_config(
                        VectorParamsBuilder::new(
                            self.config.vector_dim as u64,
                            Distance::from(self.config.distance),
                        ),
                    ),
                )
                .await
                .map_err(|e| RagError::VectorStore(e.to_string()))?;
        }

        Ok(())
    }

    /// Insert documents with embeddings
    pub async fn upsert(
        &self,
        documents: &[Document],
        embeddings: &[Vec<f32>],
    ) -> Result<(), RagError> {
        if documents.len() != embeddings.len() {
            return Err(RagError::VectorStore(
                "Document and embedding count mismatch".to_string(),
            ));
        }

        let points: Vec<PointStruct> = documents
            .iter()
            .zip(embeddings.iter())
            .map(|(doc, emb)| {
                let mut payload: HashMap<String, qdrant_client::qdrant::Value> = HashMap::new();
                // P2-2 FIX: Store as "text" in Qdrant for backwards compatibility
                payload.insert("text".to_string(), doc.content.clone().into());

                if let Some(ref title) = doc.title {
                    payload.insert("title".to_string(), title.clone().into());
                }
                if let Some(ref category) = doc.category {
                    payload.insert("category".to_string(), category.clone().into());
                }
                if let Some(ref language) = doc.language {
                    payload.insert("language".to_string(), language.clone().into());
                }

                for (k, v) in &doc.metadata {
                    payload.insert(k.clone(), v.clone().into());
                }

                PointStruct::new(doc.id.clone(), emb.clone(), payload)
            })
            .collect();

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.config.collection, points))
            .await
            .map_err(|e| RagError::VectorStore(e.to_string()))?;

        Ok(())
    }

    /// Search by vector
    pub async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filter: Option<SearchFilter>,
    ) -> Result<Vec<VectorSearchResult>, RagError> {
        let qdrant_filter = filter.map(|f| f.into_qdrant());

        let mut search_builder = SearchPointsBuilder::new(
            &self.config.collection,
            query_embedding.to_vec(),
            top_k as u64,
        )
        .with_payload(true);

        if let Some(f) = qdrant_filter {
            search_builder = search_builder.filter(f);
        }

        let results = self
            .client
            .search_points(search_builder)
            .await
            .map_err(|e| RagError::Search(e.to_string()))?;

        let search_results: Vec<VectorSearchResult> = results
            .result
            .into_iter()
            .map(|point| {
                let mut metadata = HashMap::new();
                let mut content = String::new();

                for (k, v) in point.payload {
                    // P2-2 FIX: "text" key in Qdrant maps to `content` field
                    if k == "text" {
                        if let Some(Kind::StringValue(s)) = v.kind {
                            content = s;
                        }
                    } else if let Some(Kind::StringValue(s)) = v.kind {
                        metadata.insert(k, s);
                    }
                }

                let id = point
                    .id
                    .map(|pid| match pid.point_id_options {
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u)) => u,
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => {
                            n.to_string()
                        },
                        None => String::new(),
                    })
                    .unwrap_or_default();

                VectorSearchResult {
                    id,
                    score: point.score,
                    content,
                    metadata,
                }
            })
            .collect();

        Ok(search_results)
    }

    /// Delete by IDs
    pub async fn delete(&self, ids: &[String]) -> Result<(), RagError> {
        let points: Vec<PointId> = ids.iter().map(|id| PointId::from(id.clone())).collect();

        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.config.collection)
                    .points(PointsIdsList { ids: points }),
            )
            .await
            .map_err(|e| RagError::VectorStore(e.to_string()))?;

        Ok(())
    }

    /// Get collection info
    pub async fn collection_info(&self) -> Result<CollectionInfo, RagError> {
        let info = self
            .client
            .collection_info(&self.config.collection)
            .await
            .map_err(|e| RagError::VectorStore(e.to_string()))?;

        let points_count = info
            .result
            .map(|r| r.points_count.unwrap_or(0))
            .unwrap_or(0);

        Ok(CollectionInfo {
            name: self.config.collection.clone(),
            vectors_count: points_count,
            points_count,
        })
    }
}

/// Collection info
#[derive(Debug, Clone)]
pub struct CollectionInfo {
    pub name: String,
    pub vectors_count: u64,
    pub points_count: u64,
}

/// Search filter
#[derive(Debug, Clone)]
pub struct SearchFilter {
    pub category: Option<String>,
    pub language: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl SearchFilter {
    pub fn new() -> Self {
        Self {
            category: None,
            language: None,
            metadata: HashMap::new(),
        }
    }

    pub fn category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    fn into_qdrant(self) -> Filter {
        let mut conditions = Vec::new();

        if let Some(category) = self.category {
            conditions.push(Condition {
                condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Field(
                    FieldCondition {
                        key: "category".to_string(),
                        r#match: Some(Match {
                            match_value: Some(qdrant_client::qdrant::r#match::MatchValue::Keyword(
                                category,
                            )),
                        }),
                        ..Default::default()
                    },
                )),
            });
        }

        if let Some(language) = self.language {
            conditions.push(Condition {
                condition_one_of: Some(qdrant_client::qdrant::condition::ConditionOneOf::Field(
                    FieldCondition {
                        key: "language".to_string(),
                        r#match: Some(Match {
                            match_value: Some(qdrant_client::qdrant::r#match::MatchValue::Keyword(
                                language,
                            )),
                        }),
                        ..Default::default()
                    },
                )),
            });
        }

        Filter {
            must: conditions,
            ..Default::default()
        }
    }
}

impl Default for SearchFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = VectorStoreConfig::default();
        assert_eq!(config.vector_dim, 384);
        assert_eq!(config.distance, VectorDistance::Cosine);
    }

    #[test]
    fn test_search_filter() {
        let filter = SearchFilter::new().category("product").language("hi");

        assert_eq!(filter.category, Some("product".to_string()));
        assert_eq!(filter.language, Some("hi".to_string()));
    }
}
