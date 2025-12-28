//! Candle BERT Embeddings
//!
//! Native Rust BERT implementation using Candle for text embeddings.
//! Supports loading from SafeTensors and HuggingFace Hub.
//!
//! # Features
//!
//! - Pure Rust inference (no Python/ONNX dependencies)
//! - SafeTensors weight loading
//! - Mean pooling for sentence embeddings
//! - Support for multilingual models (e5-small, mBERT, etc.)

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
#[cfg(feature = "candle")]
use candle_nn::VarBuilder;
#[cfg(feature = "candle")]
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
#[cfg(feature = "candle")]
use tokenizers::Tokenizer;

#[cfg(feature = "candle")]
use std::path::Path;

use crate::RagError;

/// Configuration for Candle BERT embedder
#[derive(Debug, Clone)]
pub struct CandleEmbeddingConfig {
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Normalize embeddings to unit length
    pub normalize: bool,
    /// Batch size for bulk embedding
    pub batch_size: usize,
    /// Pooling strategy
    pub pooling: PoolingStrategy,
    /// Device to run on
    pub device: DeviceConfig,
}

/// Pooling strategy for sentence embeddings
#[derive(Debug, Clone, Copy, Default)]
pub enum PoolingStrategy {
    /// Mean of all token embeddings (weighted by attention mask)
    #[default]
    Mean,
    /// Use [CLS] token embedding
    Cls,
    /// Max pooling across tokens
    Max,
}

/// Device configuration
#[derive(Debug, Clone, Default)]
pub enum DeviceConfig {
    #[default]
    Cpu,
    #[cfg(feature = "candle")]
    Cuda(usize),
    #[cfg(feature = "candle")]
    Metal,
}

impl Default for CandleEmbeddingConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 512,
            embedding_dim: 384,
            normalize: true,
            batch_size: 32,
            pooling: PoolingStrategy::Mean,
            device: DeviceConfig::Cpu,
        }
    }
}

impl CandleEmbeddingConfig {
    /// Configuration for multilingual-e5-small
    pub fn e5_small() -> Self {
        Self {
            embedding_dim: 384,
            max_seq_len: 512,
            normalize: true,
            batch_size: 32,
            pooling: PoolingStrategy::Mean,
            device: DeviceConfig::Cpu,
        }
    }

    /// Configuration for mBERT
    pub fn mbert() -> Self {
        Self {
            embedding_dim: 768,
            max_seq_len: 512,
            normalize: true,
            batch_size: 16,
            pooling: PoolingStrategy::Mean,
            device: DeviceConfig::Cpu,
        }
    }
}

/// Candle BERT Embedder
///
/// Native Rust implementation for text embeddings using BERT-family models.
#[cfg(feature = "candle")]
pub struct CandleBertEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    config: CandleEmbeddingConfig,
    device: Device,
}

#[cfg(feature = "candle")]
impl CandleBertEmbedder {
    /// Load from local SafeTensors file
    pub fn from_safetensors<P: AsRef<Path>>(
        model_path: P,
        config_path: P,
        tokenizer_path: P,
        embed_config: CandleEmbeddingConfig,
    ) -> Result<Self, RagError> {
        let device = match embed_config.device {
            DeviceConfig::Cpu => Device::Cpu,
            DeviceConfig::Cuda(idx) => Device::new_cuda(idx)
                .map_err(|e| RagError::Model(format!("Failed to create CUDA device: {}", e)))?,
            DeviceConfig::Metal => Device::new_metal(0)
                .map_err(|e| RagError::Model(format!("Failed to create Metal device: {}", e)))?,
        };

        // Load BERT config
        let config_data = std::fs::read_to_string(config_path.as_ref())
            .map_err(|e| RagError::Model(format!("Failed to read config: {}", e)))?;
        let bert_config: BertConfig = serde_json::from_str(&config_data)
            .map_err(|e| RagError::Model(format!("Failed to parse config: {}", e)))?;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path.as_ref()], DType::F32, &device)
                .map_err(|e| RagError::Model(format!("Failed to load weights: {}", e)))?
        };

        let model = BertModel::load(vb, &bert_config)
            .map_err(|e| RagError::Model(format!("Failed to load BERT model: {}", e)))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| RagError::Model(format!("Failed to load tokenizer: {}", e)))?;

        Ok(Self {
            model,
            tokenizer,
            config: embed_config,
            device,
        })
    }

    /// Load from HuggingFace Hub
    pub fn from_hub(
        repo_id: &str,
        embed_config: CandleEmbeddingConfig,
    ) -> Result<Self, RagError> {
        use hf_hub::{api::sync::Api, Repo, RepoType};

        let device = match embed_config.device {
            DeviceConfig::Cpu => Device::Cpu,
            DeviceConfig::Cuda(idx) => Device::new_cuda(idx)
                .map_err(|e| RagError::Model(format!("Failed to create CUDA device: {}", e)))?,
            DeviceConfig::Metal => Device::new_metal(0)
                .map_err(|e| RagError::Model(format!("Failed to create Metal device: {}", e)))?,
        };

        let api = Api::new().map_err(|e| RagError::Model(e.to_string()))?;
        let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

        // Download files
        let config_path = repo.get("config.json")
            .map_err(|e| RagError::Model(format!("Failed to download config: {}", e)))?;
        let tokenizer_path = repo.get("tokenizer.json")
            .map_err(|e| RagError::Model(format!("Failed to download tokenizer: {}", e)))?;
        let weights_path = repo.get("model.safetensors")
            .map_err(|e| RagError::Model(format!("Failed to download weights: {}", e)))?;

        // Load config
        let config_data = std::fs::read_to_string(&config_path)
            .map_err(|e| RagError::Model(format!("Failed to read config: {}", e)))?;
        let bert_config: BertConfig = serde_json::from_str(&config_data)
            .map_err(|e| RagError::Model(format!("Failed to parse config: {}", e)))?;

        // Load model
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.as_path()], DType::F32, &device)
                .map_err(|e| RagError::Model(format!("Failed to load weights: {}", e)))?
        };

        let model = BertModel::load(vb, &bert_config)
            .map_err(|e| RagError::Model(format!("Failed to load BERT model: {}", e)))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| RagError::Model(format!("Failed to load tokenizer: {}", e)))?;

        Ok(Self {
            model,
            tokenizer,
            config: embed_config,
            device,
        })
    }

    /// Embed a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, RagError> {
        let embeddings = self.embed_batch(&[text])?;
        Ok(embeddings.into_iter().next().unwrap_or_default())
    }

    /// Embed multiple texts
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, RagError> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.config.batch_size) {
            let batch_embeddings = self.embed_batch_internal(chunk)?;
            all_embeddings.extend(batch_embeddings);
        }

        Ok(all_embeddings)
    }

    /// Internal batch embedding
    fn embed_batch_internal(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, RagError> {
        let batch_size = texts.len();

        // Tokenize
        let encodings = self.tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| RagError::Embedding(e.to_string()))?;

        // Prepare input tensors
        let max_len = encodings.iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(self.config.max_seq_len);

        let mut input_ids = vec![0u32; batch_size * max_len];
        let mut attention_mask = vec![0u32; batch_size * max_len];
        let mut token_type_ids = vec![0u32; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let types = encoding.get_type_ids();

            let len = ids.len().min(max_len);
            let offset = i * max_len;

            for j in 0..len {
                input_ids[offset + j] = ids[j];
                attention_mask[offset + j] = mask[j] as u32;
                token_type_ids[offset + j] = types[j];
            }
        }

        // Create tensors
        let input_ids = Tensor::from_vec(input_ids, (batch_size, max_len), &self.device)
            .map_err(|e| RagError::Embedding(e.to_string()))?;
        let attention_mask = Tensor::from_vec(attention_mask, (batch_size, max_len), &self.device)
            .map_err(|e| RagError::Embedding(e.to_string()))?;
        let token_type_ids = Tensor::from_vec(token_type_ids, (batch_size, max_len), &self.device)
            .map_err(|e| RagError::Embedding(e.to_string()))?;

        // Run model
        let output = self.model.forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| RagError::Embedding(format!("Model forward failed: {}", e)))?;

        // Pool embeddings
        let embeddings = self.pool(&output, &attention_mask)
            .map_err(|e| RagError::Embedding(format!("Pooling failed: {}", e)))?;

        // Convert to Vec<Vec<f32>>
        let embeddings_data: Vec<f32> = embeddings.to_vec2()
            .map_err(|e| RagError::Embedding(e.to_string()))?
            .into_iter()
            .flatten()
            .collect();

        let mut result = Vec::with_capacity(batch_size);
        let dim = self.config.embedding_dim;

        for i in 0..batch_size {
            let start = i * dim;
            let end = start + dim;
            let mut embedding: Vec<f32> = if end <= embeddings_data.len() {
                embeddings_data[start..end].to_vec()
            } else {
                vec![0.0; dim]
            };

            // Normalize if requested
            if self.config.normalize {
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut embedding {
                        *v /= norm;
                    }
                }
            }

            result.push(embedding);
        }

        Ok(result)
    }

    /// Pool token embeddings to sentence embedding
    fn pool(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
        match self.config.pooling {
            PoolingStrategy::Mean => {
                // Expand attention mask to hidden dimension
                let mask = attention_mask.unsqueeze(D::Minus1)?;
                let mask = mask.to_dtype(hidden_states.dtype())?;

                // Masked sum
                let sum = hidden_states.broadcast_mul(&mask)?.sum(1)?;

                // Count non-padding tokens
                let count = mask.sum(1)?;
                let count = count.maximum(&Tensor::new(1e-9f32, &self.device)?)?;

                sum.broadcast_div(&count)
            }
            PoolingStrategy::Cls => {
                // Take first token (CLS)
                hidden_states.narrow(1, 0, 1)?.squeeze(1)
            }
            PoolingStrategy::Max => {
                // Expand attention mask
                let mask = attention_mask.unsqueeze(D::Minus1)?;
                let mask = mask.to_dtype(hidden_states.dtype())?;

                // Apply mask (set padding to large negative)
                let large_neg = Tensor::new(-1e9f32, &self.device)?;
                let masked = hidden_states.broadcast_mul(&mask)?
                    .broadcast_add(&mask.broadcast_mul(&large_neg)?.broadcast_sub(&large_neg)?)?;

                masked.max(1)
            }
        }
    }

    /// Get embedding dimension
    pub fn dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

// Non-Candle stubs
#[cfg(not(feature = "candle"))]
pub struct CandleBertEmbedder {
    config: CandleEmbeddingConfig,
}

#[cfg(not(feature = "candle"))]
impl CandleBertEmbedder {
    pub fn new(config: CandleEmbeddingConfig) -> Self {
        Self { config }
    }

    pub fn embed(&self, _text: &str) -> Result<Vec<f32>, RagError> {
        Err(RagError::Model("Candle feature not enabled".to_string()))
    }

    pub fn embed_batch(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>, RagError> {
        Err(RagError::Model("Candle feature not enabled".to_string()))
    }

    pub fn dim(&self) -> usize {
        self.config.embedding_dim
    }
}

/// Unified embedder that can use either ONNX or Candle backend
pub enum UnifiedEmbedder {
    #[cfg(feature = "onnx")]
    Onnx(super::embeddings::Embedder),
    #[cfg(feature = "candle")]
    Candle(CandleBertEmbedder),
    /// Fallback simple embedder
    Simple(super::embeddings::SimpleEmbedder),
}

impl UnifiedEmbedder {
    /// Create from ONNX model
    #[cfg(feature = "onnx")]
    pub fn from_onnx<P: AsRef<std::path::Path>>(
        model_path: P,
        tokenizer_path: P,
        config: super::embeddings::EmbeddingConfig,
    ) -> Result<Self, RagError> {
        let embedder = super::embeddings::Embedder::new(model_path, tokenizer_path, config)?;
        Ok(Self::Onnx(embedder))
    }

    /// Create from Candle model
    #[cfg(feature = "candle")]
    pub fn from_candle<P: AsRef<std::path::Path>>(
        model_path: P,
        config_path: P,
        tokenizer_path: P,
        config: CandleEmbeddingConfig,
    ) -> Result<Self, RagError> {
        let embedder = CandleBertEmbedder::from_safetensors(
            model_path,
            config_path,
            tokenizer_path,
            config,
        )?;
        Ok(Self::Candle(embedder))
    }

    /// Create from HuggingFace Hub (Candle)
    #[cfg(feature = "candle")]
    pub fn from_hub(
        repo_id: &str,
        config: CandleEmbeddingConfig,
    ) -> Result<Self, RagError> {
        let embedder = CandleBertEmbedder::from_hub(repo_id, config)?;
        Ok(Self::Candle(embedder))
    }

    /// Create simple fallback embedder
    pub fn simple(config: super::embeddings::EmbeddingConfig) -> Self {
        Self::Simple(super::embeddings::SimpleEmbedder::new(config))
    }

    /// Embed a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, RagError> {
        match self {
            #[cfg(feature = "onnx")]
            Self::Onnx(e) => e.embed(text),
            #[cfg(feature = "candle")]
            Self::Candle(e) => e.embed(text),
            Self::Simple(e) => Ok(e.embed(text)),
        }
    }

    /// Embed multiple texts
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, RagError> {
        match self {
            #[cfg(feature = "onnx")]
            Self::Onnx(e) => e.embed_batch(texts),
            #[cfg(feature = "candle")]
            Self::Candle(e) => e.embed_batch(texts),
            Self::Simple(e) => Ok(texts.iter().map(|t| e.embed(t)).collect()),
        }
    }

    /// Get embedding dimension
    pub fn dim(&self) -> usize {
        match self {
            #[cfg(feature = "onnx")]
            Self::Onnx(e) => e.dim(),
            #[cfg(feature = "candle")]
            Self::Candle(e) => e.dim(),
            Self::Simple(_) => 384, // Default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = CandleEmbeddingConfig::default();
        assert_eq!(config.embedding_dim, 384);
        assert!(config.normalize);
        assert!(matches!(config.pooling, PoolingStrategy::Mean));
    }

    #[test]
    fn test_e5_config() {
        let config = CandleEmbeddingConfig::e5_small();
        assert_eq!(config.embedding_dim, 384);
    }

    #[cfg(not(feature = "candle"))]
    #[test]
    fn test_stub_returns_error() {
        let embedder = CandleBertEmbedder::new(CandleEmbeddingConfig::default());
        let result = embedder.embed("test");
        assert!(result.is_err());
    }
}
