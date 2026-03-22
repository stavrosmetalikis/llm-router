package cache

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sync"

	"llm-router/internal/types"
)

// inflightResult holds the result of a shared in-flight request.
type inflightResult struct {
	Response *types.ChatCompletionResponse
	Err      error
}

// inflightCall tracks waiters for a single in-flight request.
type inflightCall struct {
	wg  sync.WaitGroup
	res *inflightResult
}

// InflightCache provides in-flight deduplication for identical concurrent requests.
type InflightCache struct {
	mu    sync.Mutex
	calls map[string]*inflightCall
}

// NewInflightCache creates a new InflightCache.
func NewInflightCache() *InflightCache {
	return &InflightCache{
		calls: make(map[string]*inflightCall),
	}
}

// requestKey produces a deterministic hash for a request.
func requestKey(req *types.ChatCompletionRequest) string {
	data, _ := json.Marshal(req)
	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash)
}

// Do ensures that only one execution of fn happens for duplicate concurrent requests.
// If a call is already in flight for this request, subsequent callers wait and receive the same result.
// Returns the response, error, and a bool indicating if this caller was the one that executed fn (shared=false means this caller did the work).
func (c *InflightCache) Do(req *types.ChatCompletionRequest, fn func() (*types.ChatCompletionResponse, error)) (*types.ChatCompletionResponse, error, bool) {
	key := requestKey(req)

	c.mu.Lock()
	if call, ok := c.calls[key]; ok {
		// Another goroutine is already processing this request — wait for it
		c.mu.Unlock()
		call.wg.Wait()
		return call.res.Response, call.res.Err, true
	}

	// This caller will execute the request
	call := &inflightCall{}
	call.wg.Add(1)
	c.calls[key] = call
	c.mu.Unlock()

	resp, err := fn()
	call.res = &inflightResult{Response: resp, Err: err}
	call.wg.Done()

	// Clean up
	c.mu.Lock()
	delete(c.calls, key)
	c.mu.Unlock()

	return resp, err, false
}
