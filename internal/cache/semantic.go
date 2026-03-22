package cache

import (
	"log"
	"math"
	"sync"
	"time"

	"llm-router/internal/types"
)

// SemanticEntry holds a cached embedding+response pair for semantic lookup.
type SemanticEntry struct {
	Embedding []float64
	Response  *types.ChatCompletionResponse
	CreatedAt time.Time
	LastUsed  time.Time
}

// SemanticCache provides an in-memory LRU cache with cosine-similarity lookup.
type SemanticCache struct {
	mu        sync.Mutex
	entries   []*SemanticEntry
	maxSize   int
	threshold float64
	ttl       time.Duration
}

// NewSemanticCache creates a new SemanticCache with the given max size and similarity threshold.
func NewSemanticCache(maxSize int, threshold float64) *SemanticCache {
	return &SemanticCache{
		maxSize:   maxSize,
		threshold: threshold,
		ttl:       5 * time.Minute,
	}
}

// Get finds the most similar cached entry above the threshold.
// Returns nil if no entry is similar enough or all are expired.
func (c *SemanticCache) Get(embedding []float64) *types.ChatCompletionResponse {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	bestSim := -1.0
	var bestEntry *SemanticEntry

	// Sweep expired entries and find best match
	alive := c.entries[:0]
	for _, e := range c.entries {
		if now.Sub(e.CreatedAt) > c.ttl {
			continue // expired
		}
		alive = append(alive, e)

		sim := cosineSimilarity(embedding, e.Embedding)
		if sim > bestSim {
			bestSim = sim
			bestEntry = e
		}
	}
	c.entries = alive

	if bestEntry != nil && bestSim >= c.threshold {
		bestEntry.LastUsed = now
		log.Printf("[SemanticCache] HIT with similarity %.4f", bestSim)
		return bestEntry.Response
	}

	return nil
}

// Set adds a new entry to the cache, evicting LRU entries if at capacity.
func (c *SemanticCache) Set(embedding []float64, resp *types.ChatCompletionResponse) {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()

	// Evict if at capacity — remove least recently used
	for len(c.entries) >= c.maxSize {
		lruIdx := 0
		for i, e := range c.entries {
			if e.LastUsed.Before(c.entries[lruIdx].LastUsed) {
				lruIdx = i
			}
		}
		c.entries = append(c.entries[:lruIdx], c.entries[lruIdx+1:]...)
	}

	c.entries = append(c.entries, &SemanticEntry{
		Embedding: embedding,
		Response:  resp,
		CreatedAt: now,
		LastUsed:  now,
	})
}

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
