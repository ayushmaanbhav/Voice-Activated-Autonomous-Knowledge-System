//! Knowledge Base Loader
//!
//! P2 FIX: Loads knowledge documents from YAML/JSON files and indexes them
//! in the vector store for RAG retrieval.

use std::path::Path;
use serde::{Deserialize, Serialize};

use crate::{VectorStore, RagError};
use crate::vector_store::Document;

/// Knowledge document format for YAML/JSON files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeDocument {
    /// Unique document ID
    pub id: String,
    /// Document title
    pub title: String,
    /// Document content (will be embedded)
    pub content: String,
    /// Category/type (e.g., "faq", "product", "policy")
    #[serde(default)]
    pub category: Option<String>,
    /// Language code (e.g., "en", "hi")
    #[serde(default = "default_language")]
    pub language: String,
    /// Keywords for boosting
    #[serde(default)]
    pub keywords: Vec<String>,
}

fn default_language() -> String {
    "en".to_string()
}

/// Knowledge base file format
#[derive(Debug, Serialize, Deserialize)]
pub struct KnowledgeFile {
    /// Version for format compatibility
    #[serde(default)]
    pub version: Option<String>,
    /// List of documents
    pub documents: Vec<KnowledgeDocument>,
}

/// Knowledge loader for populating vector store
pub struct KnowledgeLoader;

impl KnowledgeLoader {
    /// Load knowledge base from a directory
    ///
    /// Scans the directory for YAML and JSON files containing knowledge documents.
    /// Each file should have a `documents` array of `KnowledgeDocument` objects.
    ///
    /// # Arguments
    /// * `knowledge_dir` - Path to directory containing knowledge files
    /// * `vector_store` - Vector store to populate
    /// * `embedder` - Embedding function for vectorizing documents
    ///
    /// # Returns
    /// Number of documents loaded
    pub async fn load_directory<F, Fut>(
        knowledge_dir: &Path,
        vector_store: &VectorStore,
        embedder: F,
    ) -> Result<usize, RagError>
    where
        F: Fn(&str) -> Fut,
        Fut: std::future::Future<Output = Result<Vec<f32>, RagError>>,
    {
        if !knowledge_dir.exists() {
            tracing::warn!(
                path = %knowledge_dir.display(),
                "Knowledge directory does not exist"
            );
            return Ok(0);
        }

        let mut total_count = 0;

        // Iterate through files in directory
        let entries = std::fs::read_dir(knowledge_dir)
            .map_err(|e| RagError::Index(format!("Failed to read directory: {}", e)))?;

        for entry in entries {
            let entry = entry.map_err(|e| RagError::Index(format!("Failed to read entry: {}", e)))?;
            let path = entry.path();

            // Only process YAML and JSON files
            let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !matches!(extension, "yaml" | "yml" | "json") {
                continue;
            }

            match Self::load_file(&path, vector_store, &embedder).await {
                Ok(count) => {
                    tracing::info!(
                        file = %path.display(),
                        documents = count,
                        "Loaded knowledge file"
                    );
                    total_count += count;
                }
                Err(e) => {
                    tracing::error!(
                        file = %path.display(),
                        error = %e,
                        "Failed to load knowledge file"
                    );
                }
            }
        }

        tracing::info!(
            directory = %knowledge_dir.display(),
            total_documents = total_count,
            "Knowledge base loading complete"
        );

        Ok(total_count)
    }

    /// Load a single knowledge file
    async fn load_file<F, Fut>(
        path: &Path,
        vector_store: &VectorStore,
        embedder: F,
    ) -> Result<usize, RagError>
    where
        F: Fn(&str) -> Fut,
        Fut: std::future::Future<Output = Result<Vec<f32>, RagError>>,
    {
        let content = std::fs::read_to_string(path)
            .map_err(|e| RagError::Index(format!("Failed to read file: {}", e)))?;

        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let knowledge: KnowledgeFile = match extension {
            "json" => serde_json::from_str(&content)
                .map_err(|e| RagError::Index(format!("JSON parse error: {}", e)))?,
            "yaml" | "yml" => serde_yaml::from_str(&content)
                .map_err(|e| RagError::Index(format!("YAML parse error: {}", e)))?,
            _ => return Err(RagError::Index(format!("Unsupported file type: {}", extension))),
        };

        let mut documents = Vec::new();
        let mut embeddings = Vec::new();

        for doc in &knowledge.documents {
            // Create document for vector store
            let vs_doc = Document {
                id: doc.id.clone(),
                text: doc.content.clone(),
                title: Some(doc.title.clone()),
                category: doc.category.clone(),
                language: Some(doc.language.clone()),
                metadata: doc.keywords.iter()
                    .enumerate()
                    .map(|(i, k)| (format!("keyword_{}", i), k.clone()))
                    .collect(),
            };

            // Generate embedding
            let embedding = embedder(&doc.content).await?;

            documents.push(vs_doc);
            embeddings.push(embedding);
        }

        // Batch upsert to vector store
        if !documents.is_empty() {
            vector_store.upsert(&documents, &embeddings).await?;
        }

        Ok(documents.len())
    }

    /// Create a sample knowledge file for reference
    ///
    /// This creates an example YAML file showing the expected format.
    pub fn create_sample_file(path: &Path) -> Result<(), RagError> {
        let sample = KnowledgeFile {
            version: Some("1.0".to_string()),
            documents: vec![
                KnowledgeDocument {
                    id: "gold_loan_intro_001".to_string(),
                    title: "What is a Gold Loan?".to_string(),
                    content: "A gold loan is a secured loan where you pledge your gold ornaments \
                              or jewelry as collateral to borrow money. The loan amount is \
                              typically 75-90% of the gold's market value.".to_string(),
                    category: Some("faq".to_string()),
                    language: "en".to_string(),
                    keywords: vec![
                        "gold loan".to_string(),
                        "secured loan".to_string(),
                        "collateral".to_string(),
                    ],
                },
                KnowledgeDocument {
                    id: "gold_loan_benefits_001".to_string(),
                    title: "Benefits of Gold Loan".to_string(),
                    content: "Gold loans offer several benefits: quick disbursal (often within \
                              30 minutes), lower interest rates compared to personal loans, \
                              no credit score requirements, and flexible repayment options.".to_string(),
                    category: Some("product".to_string()),
                    language: "en".to_string(),
                    keywords: vec![
                        "benefits".to_string(),
                        "quick".to_string(),
                        "low interest".to_string(),
                    ],
                },
            ],
        };

        let yaml = serde_yaml::to_string(&sample)
            .map_err(|e| RagError::Index(format!("Failed to serialize: {}", e)))?;

        std::fs::write(path, yaml)
            .map_err(|e| RagError::Index(format!("Failed to write file: {}", e)))?;

        tracing::info!(path = %path.display(), "Created sample knowledge file");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_knowledge_document_serialization() {
        let doc = KnowledgeDocument {
            id: "test_001".to_string(),
            title: "Test Document".to_string(),
            content: "This is test content.".to_string(),
            category: Some("test".to_string()),
            language: "en".to_string(),
            keywords: vec!["test".to_string()],
        };

        let yaml = serde_yaml::to_string(&doc).unwrap();
        assert!(yaml.contains("test_001"));

        let parsed: KnowledgeDocument = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.id, "test_001");
    }

    #[test]
    fn test_create_sample_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sample_knowledge.yaml");

        KnowledgeLoader::create_sample_file(&path).unwrap();

        assert!(path.exists());

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: KnowledgeFile = serde_yaml::from_str(&content).unwrap();
        assert_eq!(parsed.documents.len(), 2);
    }
}
