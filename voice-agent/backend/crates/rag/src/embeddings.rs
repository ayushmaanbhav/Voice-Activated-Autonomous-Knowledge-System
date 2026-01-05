//! Text Embeddings
//!
//! Generates dense embeddings for semantic search.

use std::path::Path;

#[cfg(feature = "onnx")]
use ndarray::Array2;
#[cfg(feature = "onnx")]
use ort::{GraphOptimizationLevel, Session};
#[cfg(feature = "onnx")]
use tokenizers::Tokenizer;

use crate::RagError;

/// Embedding configuration
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Normalize embeddings
    pub normalize: bool,
    /// Batch size for bulk embedding
    pub batch_size: usize,
    /// P2 FIX: ONNX output tensor name for embeddings
    /// Different models use different names: "last_hidden_state", "sentence_embedding", "output", etc.
    pub output_name: String,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 512,
            // P6 FIX: Default to 1024 to match qwen3-embedding:0.6b
            embedding_dim: 1024,
            normalize: true,
            batch_size: 32,
            output_name: "last_hidden_state".to_string(),
        }
    }
}

/// Text embedder using ONNX model
pub struct Embedder {
    #[cfg(feature = "onnx")]
    session: Session,
    #[cfg(feature = "onnx")]
    tokenizer: Tokenizer,
    config: EmbeddingConfig,
}

impl Embedder {
    /// Create a new embedder
    #[cfg(feature = "onnx")]
    pub fn new(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config: EmbeddingConfig,
    ) -> Result<Self, RagError> {
        let session = Session::builder()
            .map_err(|e| RagError::Model(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| RagError::Model(e.to_string()))?
            .with_intra_threads(2)
            .map_err(|e| RagError::Model(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| RagError::Model(e.to_string()))?;

        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| RagError::Model(e.to_string()))?;

        Ok(Self {
            session,
            tokenizer,
            config,
        })
    }

    /// Create a new embedder (stub when ONNX disabled)
    #[cfg(not(feature = "onnx"))]
    pub fn new(
        _model_path: impl AsRef<Path>,
        _tokenizer_path: impl AsRef<Path>,
        config: EmbeddingConfig,
    ) -> Result<Self, RagError> {
        Ok(Self { config })
    }

    /// Embed a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, RagError> {
        #[cfg(feature = "onnx")]
        {
            let embeddings = self.embed_batch(&[text])?;
            Ok(embeddings.into_iter().next().unwrap_or_default())
        }
        #[cfg(not(feature = "onnx"))]
        {
            Ok(SimpleEmbedder::new(self.config.clone()).embed(text))
        }
    }

    /// Embed multiple texts
    #[cfg(feature = "onnx")]
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, RagError> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.config.batch_size) {
            let batch_embeddings = self.embed_batch_internal(chunk)?;
            all_embeddings.extend(batch_embeddings);
        }

        Ok(all_embeddings)
    }

    /// Embed multiple texts (stub when ONNX disabled)
    #[cfg(not(feature = "onnx"))]
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, RagError> {
        let embedder = SimpleEmbedder::new(self.config.clone());
        Ok(texts.iter().map(|t| embedder.embed(t)).collect())
    }

    /// Internal batch embedding
    #[cfg(feature = "onnx")]
    fn embed_batch_internal(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, RagError> {
        let batch_size = texts.len();

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| RagError::Embedding(e.to_string()))?;

        let mut input_ids = vec![0i64; batch_size * self.config.max_seq_len];
        let mut attention_mask = vec![0i64; batch_size * self.config.max_seq_len];
        let mut token_type_ids = vec![0i64; batch_size * self.config.max_seq_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let types = encoding.get_type_ids();

            let len = ids.len().min(self.config.max_seq_len);
            let offset = i * self.config.max_seq_len;

            for j in 0..len {
                input_ids[offset + j] = ids[j] as i64;
                attention_mask[offset + j] = mask[j] as i64;
                token_type_ids[offset + j] = types[j] as i64;
            }
        }

        let input_ids = Array2::from_shape_vec((batch_size, self.config.max_seq_len), input_ids)
            .map_err(|e| RagError::Embedding(e.to_string()))?;

        let attention_mask =
            Array2::from_shape_vec((batch_size, self.config.max_seq_len), attention_mask)
                .map_err(|e| RagError::Embedding(e.to_string()))?;

        let token_type_ids =
            Array2::from_shape_vec((batch_size, self.config.max_seq_len), token_type_ids)
                .map_err(|e| RagError::Embedding(e.to_string()))?;

        let outputs = self
            .session
            .run(
                ort::inputs![
                    "input_ids" => input_ids.view(),
                    "attention_mask" => attention_mask.view(),
                    "token_type_ids" => token_type_ids.view(),
                ]
                .map_err(|e| RagError::Model(e.to_string()))?,
            )
            .map_err(|e| RagError::Model(e.to_string()))?;

        // P2 FIX: Use configurable output name instead of hardcoded "last_hidden_state"
        let last_hidden = outputs
            .get(&self.config.output_name)
            .ok_or_else(|| {
                RagError::Model(format!(
                    "Missing output tensor: {}",
                    self.config.output_name
                ))
            })?
            .try_extract_tensor::<f32>()
            .map_err(|e| RagError::Model(e.to_string()))?;

        let hidden_view = last_hidden.view();

        let mut embeddings = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let seq_len = encodings[i].get_ids().len().min(self.config.max_seq_len);
            let mut embedding = vec![0.0f32; self.config.embedding_dim];

            for j in 0..seq_len {
                for k in 0..self.config.embedding_dim {
                    embedding[k] += hidden_view[[i, j, k]];
                }
            }

            for v in &mut embedding {
                *v /= seq_len as f32;
            }

            if self.config.normalize {
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut embedding {
                        *v /= norm;
                    }
                }
            }

            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    /// Get embedding dimension
    pub fn dim(&self) -> usize {
        self.config.embedding_dim
    }
}

/// Simple embedder for testing (no model required)
pub struct SimpleEmbedder {
    config: EmbeddingConfig,
}

impl SimpleEmbedder {
    pub fn new(config: EmbeddingConfig) -> Self {
        Self { config }
    }

    /// Generate a simple hash-based embedding
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.config.embedding_dim];

        for (i, c) in text.chars().enumerate() {
            let idx = (c as usize + i) % self.config.embedding_dim;
            embedding[idx] += 1.0;
        }

        if self.config.normalize {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut embedding {
                    *v /= norm;
                }
            }
        }

        embedding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_embedder() {
        let embedder = SimpleEmbedder::new(EmbeddingConfig::default());
        let embedding = embedder.embed("Hello world");

        assert_eq!(embedding.len(), 384);

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.embedding_dim, 384);
        assert!(config.normalize);
    }
}
