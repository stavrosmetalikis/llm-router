package cache

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/redis/go-redis/v9"
	"llm-router/internal/types"
)

const exactCacheTTL = 5 * time.Minute

// ExactCache provides exact-match caching via Redis.
type ExactCache struct {
	client *redis.Client
}

// NewExactCache creates a new ExactCache connected to the given Redis address.
// Returns a no-op cache if Redis is unavailable.
func NewExactCache(addr string) *ExactCache {
	client := redis.NewClient(&redis.Options{
		Addr:        addr,
		DialTimeout: 2 * time.Second,
	})

	// Test connection — log warning but don't fail
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		log.Printf("[ExactCache] Redis unavailable at %s: %v (cache disabled)", addr, err)
	} else {
		log.Printf("[ExactCache] Connected to Redis at %s", addr)
	}

	return &ExactCache{client: client}
}

// hashRequest produces a SHA-256 hex digest of the serialized request.
func hashRequest(req *types.ChatCompletionRequest) (string, error) {
	data, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash), nil
}

// Get looks up a cached response for the given request.
// Returns nil if not found or if Redis is unavailable.
func (c *ExactCache) Get(ctx context.Context, req *types.ChatCompletionRequest) *types.ChatCompletionResponse {
	key, err := hashRequest(req)
	if err != nil {
		return nil
	}

	val, err := c.client.Get(ctx, "exact:"+key).Result()
	if err != nil {
		return nil
	}

	var resp types.ChatCompletionResponse
	if err := json.Unmarshal([]byte(val), &resp); err != nil {
		return nil
	}

	log.Printf("[ExactCache] HIT for key %s", key[:12])
	return &resp
}

// Set stores a response in the cache keyed by the request hash.
func (c *ExactCache) Set(ctx context.Context, req *types.ChatCompletionRequest, resp *types.ChatCompletionResponse) {
	key, err := hashRequest(req)
	if err != nil {
		return
	}

	data, err := json.Marshal(resp)
	if err != nil {
		return
	}

	if err := c.client.Set(ctx, "exact:"+key, string(data), exactCacheTTL).Err(); err != nil {
		log.Printf("[ExactCache] Failed to set cache: %v", err)
	}
}
