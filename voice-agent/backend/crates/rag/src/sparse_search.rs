//! Sparse Search using Tantivy (BM25)
//!
//! Provides keyword-based search for hybrid retrieval.
//!
//! P0 FIX: Added Hindi/Devanagari tokenization support using SimpleTokenizer.
//! For production Hindi NLP, consider integrating ICU or language-specific
//! stemmers (e.g., NLTK's Hindi stemmer via Python bindings).

use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;
use tantivy::{
    collector::TopDocs,
    query::QueryParser,
    schema::{Field, OwnedValue, Schema, TextFieldIndexing, TextOptions, STORED, STRING},
    tokenizer::{Language, LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer, TextAnalyzer},
    Index, IndexReader, IndexWriter, TantivyDocument,
};

use crate::RagError;

/// Sparse search configuration
#[derive(Debug, Clone)]
pub struct SparseConfig {
    /// Index path (use RAM if None)
    pub index_path: Option<String>,
    /// Number of results to retrieve
    pub top_k: usize,
    /// Enable stemming
    pub stemming: bool,
    /// Language for analysis
    pub language: String,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            index_path: None,
            top_k: 20,
            stemming: true,
            language: "en".to_string(),
        }
    }
}

/// Sparse search result
#[derive(Debug, Clone)]
pub struct SparseResult {
    /// Document ID
    pub id: String,
    /// BM25 score
    pub score: f32,
    /// Document content (P2-2 FIX: renamed from `text` for consistency)
    pub content: String,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Sparse index for BM25 search
#[allow(dead_code)]
pub struct SparseIndex {
    index: Index,
    reader: IndexReader,
    writer: RwLock<Option<IndexWriter>>,
    schema: Schema,
    id_field: Field,
    text_field: Field,
    title_field: Field,
    category_field: Field,
    config: SparseConfig,
}

impl SparseIndex {
    /// Create a new sparse index
    ///
    /// P0 FIX: Now uses language-aware tokenization based on config.
    pub fn new(config: SparseConfig) -> Result<Self, RagError> {
        // Build schema with custom tokenizer
        let mut schema_builder = Schema::builder();

        // P0 FIX: Create text options with custom tokenizer
        let text_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("multilingual")
                    .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored();

        let id_field = schema_builder.add_text_field("id", STRING | STORED);
        let text_field = schema_builder.add_text_field("text", text_options.clone());
        let title_field = schema_builder.add_text_field("title", text_options);
        let category_field = schema_builder.add_text_field("category", STRING | STORED);

        let schema = schema_builder.build();

        // Create index
        let index = if let Some(ref path) = config.index_path {
            let dir = tantivy::directory::MmapDirectory::open(Path::new(path))
                .map_err(|e| RagError::Index(e.to_string()))?;
            Index::open_or_create(dir, schema.clone())
                .map_err(|e| RagError::Index(e.to_string()))?
        } else {
            Index::create_in_ram(schema.clone())
        };

        // P0 FIX: Register multilingual tokenizer
        // Uses SimpleTokenizer which handles Unicode/Devanagari properly
        let tokenizer = Self::build_tokenizer(&config);
        index.tokenizers().register("multilingual", tokenizer);

        let reader = index.reader().map_err(|e| RagError::Index(e.to_string()))?;

        let writer = index
            .writer(50_000_000) // 50MB buffer
            .map_err(|e| RagError::Index(e.to_string()))?;

        tracing::info!(
            "Sparse index created with language={}, stemming={}",
            config.language,
            config.stemming
        );

        Ok(Self {
            index,
            reader,
            writer: RwLock::new(Some(writer)),
            schema,
            id_field,
            text_field,
            title_field,
            category_field,
            config,
        })
    }

    /// Build tokenizer based on configuration
    ///
    /// P0 FIX: Supports English stemming; Hindi uses simple Unicode tokenization.
    fn build_tokenizer(config: &SparseConfig) -> TextAnalyzer {
        // SimpleTokenizer handles Unicode properly (including Devanagari)
        let base = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(RemoveLongFilter::limit(100))
            .filter(LowerCaser);

        // Add stemmer for supported languages
        if config.stemming && config.language == "en" {
            base.filter(Stemmer::new(Language::English)).build()
        } else {
            if config.language == "hi" || config.language == "hindi" {
                tracing::info!("Hindi: using SimpleTokenizer (no stemming available in Tantivy)");
            } else if config.language != "en" {
                tracing::warn!(
                    "Language '{}' has no stemmer, using simple tokenization",
                    config.language
                );
            }
            base.build()
        }
    }

    /// Index documents
    pub fn index_documents(
        &self,
        documents: &[super::vector_store::Document],
    ) -> Result<(), RagError> {
        let mut writer = self.writer.write();
        let writer = writer
            .as_mut()
            .ok_or_else(|| RagError::Index("Writer not available".to_string()))?;

        for doc in documents {
            let mut tantivy_doc = TantivyDocument::default();

            tantivy_doc.add_text(self.id_field, &doc.id);
            tantivy_doc.add_text(self.text_field, &doc.content);

            if let Some(ref title) = doc.title {
                tantivy_doc.add_text(self.title_field, title);
            }
            if let Some(ref category) = doc.category {
                tantivy_doc.add_text(self.category_field, category);
            }

            writer
                .add_document(tantivy_doc)
                .map_err(|e| RagError::Index(e.to_string()))?;
        }

        writer
            .commit()
            .map_err(|e| RagError::Index(e.to_string()))?;

        // Reload reader
        self.reader
            .reload()
            .map_err(|e| RagError::Index(e.to_string()))?;

        Ok(())
    }

    /// Search using BM25
    pub fn search(&self, query: &str, top_k: Option<usize>) -> Result<Vec<SparseResult>, RagError> {
        let k = top_k.unwrap_or(self.config.top_k);

        let searcher = self.reader.searcher();
        let query_parser =
            QueryParser::for_index(&self.index, vec![self.text_field, self.title_field]);

        let query = query_parser
            .parse_query(query)
            .map_err(|e| RagError::Search(e.to_string()))?;

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(k))
            .map_err(|e| RagError::Search(e.to_string()))?;

        let mut results = Vec::with_capacity(top_docs.len());

        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher
                .doc(doc_address)
                .map_err(|e| RagError::Search(e.to_string()))?;

            let id = doc
                .get_first(self.id_field)
                .and_then(|v| match v {
                    OwnedValue::Str(s) => Some(s.as_str()),
                    _ => None,
                })
                .unwrap_or("")
                .to_string();

            let content = doc
                .get_first(self.text_field)
                .and_then(|v| match v {
                    OwnedValue::Str(s) => Some(s.as_str()),
                    _ => None,
                })
                .unwrap_or("")
                .to_string();

            let mut metadata = HashMap::new();

            if let Some(OwnedValue::Str(title)) = doc.get_first(self.title_field) {
                metadata.insert("title".to_string(), title.to_string());
            }
            if let Some(OwnedValue::Str(category)) = doc.get_first(self.category_field) {
                metadata.insert("category".to_string(), category.to_string());
            }

            results.push(SparseResult {
                id,
                score,
                content,
                metadata,
            });
        }

        Ok(results)
    }

    /// Delete documents by ID
    pub fn delete(&self, ids: &[String]) -> Result<(), RagError> {
        let mut writer = self.writer.write();
        let writer = writer
            .as_mut()
            .ok_or_else(|| RagError::Index("Writer not available".to_string()))?;

        for id in ids {
            let term = tantivy::Term::from_field_text(self.id_field, id);
            writer.delete_term(term);
        }

        writer
            .commit()
            .map_err(|e| RagError::Index(e.to_string()))?;

        self.reader
            .reload()
            .map_err(|e| RagError::Index(e.to_string()))?;

        Ok(())
    }

    /// Get document count
    pub fn doc_count(&self) -> u64 {
        self.reader.searcher().num_docs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_store::Document;

    #[test]
    fn test_sparse_index_create() {
        let index = SparseIndex::new(SparseConfig::default()).unwrap();
        assert_eq!(index.doc_count(), 0);
    }

    #[test]
    fn test_index_and_search() {
        let index = SparseIndex::new(SparseConfig::default()).unwrap();

        let docs = vec![
            Document {
                id: "1".to_string(),
                content: "Gold loan interest rate is 10% per annum".to_string(),
                title: Some("Interest Rates".to_string()),
                category: Some("product".to_string()),
                language: Some("en".to_string()),
                metadata: HashMap::new(),
            },
            Document {
                id: "2".to_string(),
                content: "Apply for gold loan online easily".to_string(),
                title: Some("Application".to_string()),
                category: Some("process".to_string()),
                language: Some("en".to_string()),
                metadata: HashMap::new(),
            },
        ];

        index.index_documents(&docs).unwrap();
        assert_eq!(index.doc_count(), 2);

        let results = index.search("interest rate", None).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "1");
    }
}
