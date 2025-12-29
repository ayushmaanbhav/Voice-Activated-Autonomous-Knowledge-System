//! Embedding Cache
//!
//! LRU cache for text embeddings to avoid redundant computation.
//! Significantly speeds up repeated queries and document re-embedding.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;

/// Cache statistics
#[derive(Debug, Default)]
pub struct CacheStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }
}

/// LRU node for doubly-linked list
struct LruNode {
    key_hash: u64,
    embedding: Vec<f32>,
    prev: Option<u64>,
    next: Option<u64>,
}

/// LRU Embedding Cache
///
/// Thread-safe LRU cache for embeddings with configurable size.
pub struct EmbeddingCache {
    /// Maximum number of entries
    capacity: usize,
    /// Hash -> Node mapping
    entries: RwLock<HashMap<u64, LruNode>>,
    /// Head of LRU list (most recently used)
    head: RwLock<Option<u64>>,
    /// Tail of LRU list (least recently used)
    tail: RwLock<Option<u64>>,
    /// Cache statistics
    pub stats: CacheStats,
}

impl EmbeddingCache {
    /// Create a new cache with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: RwLock::new(HashMap::with_capacity(capacity)),
            head: RwLock::new(None),
            tail: RwLock::new(None),
            stats: CacheStats::default(),
        }
    }

    /// Create a cache with default capacity (10,000 entries)
    pub fn default_capacity() -> Self {
        Self::new(10_000)
    }

    /// Compute hash for cache key
    fn hash_key(text: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Get embedding from cache
    pub fn get(&self, text: &str) -> Option<Vec<f32>> {
        let key_hash = Self::hash_key(text);

        let entries = self.entries.read();
        if let Some(node) = entries.get(&key_hash) {
            self.stats.record_hit();
            // Clone the embedding (we can't move it out while holding read lock)
            let embedding = node.embedding.clone();
            drop(entries);

            // Move to front (most recently used)
            self.move_to_front(key_hash);

            Some(embedding)
        } else {
            self.stats.record_miss();
            None
        }
    }

    /// Insert embedding into cache
    pub fn insert(&self, text: &str, embedding: Vec<f32>) {
        let key_hash = Self::hash_key(text);

        let mut entries = self.entries.write();

        // Check if already exists
        if entries.contains_key(&key_hash) {
            // Update existing entry
            if let Some(node) = entries.get_mut(&key_hash) {
                node.embedding = embedding;
            }
            drop(entries);
            self.move_to_front(key_hash);
            return;
        }

        // Evict if at capacity
        if entries.len() >= self.capacity {
            drop(entries);
            self.evict_lru();
            entries = self.entries.write();
        }

        // Get current head
        let old_head = *self.head.read();

        // Create new node
        let node = LruNode {
            key_hash,
            embedding,
            prev: None,
            next: old_head,
        };

        // Update old head's prev pointer
        if let Some(old_head_hash) = old_head {
            if let Some(old_head_node) = entries.get_mut(&old_head_hash) {
                old_head_node.prev = Some(key_hash);
            }
        }

        // Insert new node
        entries.insert(key_hash, node);

        // Update head
        *self.head.write() = Some(key_hash);

        // Update tail if this is the first entry
        if self.tail.read().is_none() {
            *self.tail.write() = Some(key_hash);
        }
    }

    /// Move entry to front of LRU list
    fn move_to_front(&self, key_hash: u64) {
        let mut entries = self.entries.write();

        // Get current position
        let (prev, next) = {
            if let Some(node) = entries.get(&key_hash) {
                (node.prev, node.next)
            } else {
                return;
            }
        };

        // Already at front?
        if prev.is_none() {
            return;
        }

        // Remove from current position
        if let Some(prev_hash) = prev {
            if let Some(prev_node) = entries.get_mut(&prev_hash) {
                prev_node.next = next;
            }
        }
        if let Some(next_hash) = next {
            if let Some(next_node) = entries.get_mut(&next_hash) {
                next_node.prev = prev;
            }
        }

        // Update tail if we're moving the tail
        if *self.tail.read() == Some(key_hash) {
            *self.tail.write() = prev;
        }

        // Move to front
        let old_head = *self.head.read();
        if let Some(node) = entries.get_mut(&key_hash) {
            node.prev = None;
            node.next = old_head;
        }

        if let Some(old_head_hash) = old_head {
            if let Some(old_head_node) = entries.get_mut(&old_head_hash) {
                old_head_node.prev = Some(key_hash);
            }
        }

        *self.head.write() = Some(key_hash);
    }

    /// Evict least recently used entry
    fn evict_lru(&self) {
        let tail_hash = *self.tail.read();

        if let Some(tail_hash) = tail_hash {
            let mut entries = self.entries.write();

            // Get tail's prev
            let new_tail = entries.get(&tail_hash).and_then(|n| n.prev);

            // Remove tail
            entries.remove(&tail_hash);
            self.stats.record_eviction();

            // Update tail pointer
            *self.tail.write() = new_tail;

            // Update new tail's next pointer
            if let Some(new_tail_hash) = new_tail {
                if let Some(new_tail_node) = entries.get_mut(&new_tail_hash) {
                    new_tail_node.next = None;
                }
            } else {
                // Cache is now empty
                *self.head.write() = None;
            }
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        let mut entries = self.entries.write();
        entries.clear();
        *self.head.write() = None;
        *self.tail.write() = None;
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Get cache capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Cached embedder wrapper
///
/// Wraps any embedder with an LRU cache for automatic caching.
pub struct CachedEmbedder<E> {
    embedder: E,
    cache: EmbeddingCache,
}

impl<E> CachedEmbedder<E> {
    /// Create a new cached embedder
    pub fn new(embedder: E, cache_capacity: usize) -> Self {
        Self {
            embedder,
            cache: EmbeddingCache::new(cache_capacity),
        }
    }

    /// Create with default cache capacity
    pub fn with_default_cache(embedder: E) -> Self {
        Self::new(embedder, 10_000)
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> &CacheStats {
        &self.cache.stats
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        self.cache.stats.hit_rate()
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Get inner embedder reference
    pub fn inner(&self) -> &E {
        &self.embedder
    }
}

// Implement for ONNX Embedder
impl CachedEmbedder<super::embeddings::Embedder> {
    /// Embed with caching
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, super::RagError> {
        // Check cache first
        if let Some(embedding) = self.cache.get(text) {
            return Ok(embedding);
        }

        // Compute embedding
        let embedding = self.embedder.embed(text)?;

        // Cache result
        self.cache.insert(text, embedding.clone());

        Ok(embedding)
    }

    /// Embed batch with caching
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, super::RagError> {
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached_texts = Vec::new();
        let mut uncached_indices = Vec::new();

        // Check cache for each text
        for (i, text) in texts.iter().enumerate() {
            if let Some(embedding) = self.cache.get(text) {
                results.push(Some(embedding));
            } else {
                results.push(None);
                uncached_texts.push(*text);
                uncached_indices.push(i);
            }
        }

        // Compute uncached embeddings
        if !uncached_texts.is_empty() {
            let uncached_embeddings = self.embedder.embed_batch(&uncached_texts)?;

            // Insert into cache and results
            for (idx, embedding) in uncached_indices.into_iter().zip(uncached_embeddings) {
                self.cache.insert(texts[idx], embedding.clone());
                results[idx] = Some(embedding);
            }
        }

        // Unwrap all results (all should be Some now)
        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    /// Get embedding dimension
    pub fn dim(&self) -> usize {
        self.embedder.dim()
    }
}

// Implement for Candle Embedder
#[cfg(feature = "candle")]
impl CachedEmbedder<super::candle_embeddings::CandleBertEmbedder> {
    /// Embed with caching
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, super::RagError> {
        if let Some(embedding) = self.cache.get(text) {
            return Ok(embedding);
        }

        let embedding = self.embedder.embed(text)?;
        self.cache.insert(text, embedding.clone());
        Ok(embedding)
    }

    /// Embed batch with caching
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, super::RagError> {
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached_texts = Vec::new();
        let mut uncached_indices = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            if let Some(embedding) = self.cache.get(text) {
                results.push(Some(embedding));
            } else {
                results.push(None);
                uncached_texts.push(*text);
                uncached_indices.push(i);
            }
        }

        if !uncached_texts.is_empty() {
            let uncached_embeddings = self.embedder.embed_batch(&uncached_texts)?;

            for (idx, embedding) in uncached_indices.into_iter().zip(uncached_embeddings) {
                self.cache.insert(texts[idx], embedding.clone());
                results[idx] = Some(embedding);
            }
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    /// Get embedding dimension
    pub fn dim(&self) -> usize {
        self.embedder.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let cache = EmbeddingCache::new(100);

        // Insert
        cache.insert("hello", vec![1.0, 2.0, 3.0]);
        assert_eq!(cache.len(), 1);

        // Get
        let result = cache.get("hello");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), vec![1.0, 2.0, 3.0]);

        // Miss
        let result = cache.get("world");
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_stats() {
        let cache = EmbeddingCache::new(100);

        cache.insert("test", vec![1.0]);

        // Hit
        cache.get("test");
        assert_eq!(cache.stats.hits.load(Ordering::Relaxed), 1);

        // Miss
        cache.get("missing");
        assert_eq!(cache.stats.misses.load(Ordering::Relaxed), 1);

        // Hit rate
        assert!((cache.stats.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = EmbeddingCache::new(3);

        cache.insert("a", vec![1.0]);
        cache.insert("b", vec![2.0]);
        cache.insert("c", vec![3.0]);
        assert_eq!(cache.len(), 3);

        // Insert fourth item, should evict "a" (LRU)
        cache.insert("d", vec![4.0]);
        assert_eq!(cache.len(), 3);

        // "a" should be evicted
        assert!(cache.get("a").is_none());
        // Others should still exist
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
        assert!(cache.get("d").is_some());
    }

    #[test]
    fn test_cache_lru_order() {
        let cache = EmbeddingCache::new(3);

        cache.insert("a", vec![1.0]);
        cache.insert("b", vec![2.0]);
        cache.insert("c", vec![3.0]);

        // Access "a" to make it recently used
        cache.get("a");

        // Insert "d", should evict "b" (now LRU)
        cache.insert("d", vec![4.0]);

        assert!(cache.get("a").is_some()); // Recently accessed
        assert!(cache.get("b").is_none()); // Evicted
        assert!(cache.get("c").is_some());
        assert!(cache.get("d").is_some());
    }

    #[test]
    fn test_cache_clear() {
        let cache = EmbeddingCache::new(100);

        cache.insert("a", vec![1.0]);
        cache.insert("b", vec![2.0]);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }
}
