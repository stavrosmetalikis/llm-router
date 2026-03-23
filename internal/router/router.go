package router

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"llm-router/internal/cache"
	"llm-router/internal/compressor"
	"llm-router/internal/embedding"
	"llm-router/internal/pool"
	"llm-router/internal/types"
)

// retryableStatusCodes are HTTP status codes that trigger failover to the next provider.
var retryableStatusCodes = map[int]bool{
	400: true,
	401: true,
	403: true,
	404: true,
	429: true,
	500: true,
}

// codeMarkers are strings whose presence suggest code or structured (JSON) content.
var codeMarkers = []string{"{", "}", "func ", "def ", "class ", "=>", "//", "/*", "import ", "return ", "\"type\":", "\"key\":"}

// looksLikeCode returns true if the text appears to contain code or JSON content.
func looksLikeCode(text string) bool {
	for _, marker := range codeMarkers {
		if strings.Contains(text, marker) {
			return true
		}
	}
	return false
}

// estimateTokens provides a rough token count estimate for a request.
// Uses ~3 characters per token for code/JSON content and ~4 for natural language.
func estimateTokens(req *types.ChatCompletionRequest) int {
	totalChars := 0
	var allContent strings.Builder
	for _, m := range req.Messages {
		text := flattenContent(m.Content)
		totalChars += len(text)
		allContent.WriteString(text)
		// Account for role, tool calls, etc.
		totalChars += 10
	}
	// Account for tools definitions if present
	if len(req.Tools) > 0 {
		for _, t := range req.Tools {
			totalChars += len(t.Function.Name) + len(t.Function.Description) + 50
			if t.Function.Parameters != nil {
				paraBytes, _ := json.Marshal(t.Function.Parameters)
				totalChars += len(paraBytes)
			}
		}
	}

	// Use chars/3 for code-heavy content, chars/4 for natural language
	if looksLikeCode(allContent.String()) {
		return totalChars / 3
	}
	return totalChars / 4
}

// Router is the core orchestrator that implements the full request pipeline.
type Router struct {
	KeyPool       *pool.KeyPool
	StickyStore   *pool.StickyStore
	ExactCache    *cache.ExactCache
	SemanticCache *cache.SemanticCache
	InflightCache *cache.InflightCache
	Embedder      *embedding.GeminiClient
	Compressor    *compressor.Compressor
	HTTPClient    *http.Client
}

// NewRouter creates a new Router with all components wired together.
func NewRouter(
	keyPool *pool.KeyPool,
	stickyStore *pool.StickyStore,
	exactCache *cache.ExactCache,
	semanticCache *cache.SemanticCache,
	inflightCache *cache.InflightCache,
	embedder *embedding.GeminiClient,
	comp *compressor.Compressor,
) *Router {
	return &Router{
		KeyPool:       keyPool,
		StickyStore:   stickyStore,
		ExactCache:    exactCache,
		SemanticCache: semanticCache,
		InflightCache: inflightCache,
		Embedder:      embedder,
		Compressor:    comp,
		HTTPClient:    &http.Client{Timeout: 120 * time.Second},
	}
}

// StreamingResponse holds raw SSE bytes from a streaming provider response.
type StreamingResponse struct {
	Body       io.ReadCloser
	StatusCode int
	Model      string
}

// sessionKeyFromRequest derives a session key from the request messages.
func sessionKeyFromRequest(messages []types.ChatMessage) string {
	converted := make([]pool.ChatMessage, len(messages))
	for i, m := range messages {
		converted[i] = pool.ChatMessage{Role: m.Role, Content: m.Content}
	}
	return pool.SessionKey(converted)
}

// orderKeysWithSticky reorders keys so the sticky preferred key comes first.
// Returns the keys in order and the session ID for later updates.
func (r *Router) orderKeysWithSticky(keys []*pool.APIKey, messages []types.ChatMessage) ([]*pool.APIKey, string) {
	sessionID := sessionKeyFromRequest(messages)
	preferred := r.StickyStore.Get(sessionID)
	if preferred == "" {
		return keys, sessionID
	}

	// Find the preferred key and move it to the front
	for i, k := range keys {
		if k.Name == preferred {
			reordered := make([]*pool.APIKey, 0, len(keys))
			reordered = append(reordered, k)
			reordered = append(reordered, keys[:i]...)
			reordered = append(reordered, keys[i+1:]...)
			log.Printf("[Sticky] Session %s: trying preferred provider %s first", sessionID[:8], preferred)
			return reordered, sessionID
		}
	}

	// Preferred key not available (in cooldown) — fall through normally
	log.Printf("[Sticky] Session %s: preferred provider %s not available, falling back to tier rotation", sessionID[:8], preferred)
	return keys, sessionID
}

// HandleRequest processes a non-streaming chat completion request through the full pipeline.
func (r *Router) HandleRequest(ctx context.Context, req *types.ChatCompletionRequest) (*types.ChatCompletionResponse, error) {
	// Step 1: In-flight deduplication (non-streaming only)
	resp, err, shared := r.InflightCache.Do(req, func() (*types.ChatCompletionResponse, error) {
		return r.executeNonStreamingPipeline(ctx, req)
	})

	if shared {
		log.Printf("[Router] Returned shared in-flight result")
	}

	return resp, err
}

// executeNonStreamingPipeline runs the full pipeline for non-streaming requests.
func (r *Router) executeNonStreamingPipeline(ctx context.Context, req *types.ChatCompletionRequest) (*types.ChatCompletionResponse, error) {
	// Step 2: Exact cache check
	if cached := r.ExactCache.Get(ctx, req); cached != nil {
		return cached, nil
	}

	// Step 3-4: Semantic cache check (if embedder is available)
	var currentEmbedding []float64
	if r.Embedder.Enabled() {
		lastUserMsg := getLastUserMessage(req.Messages)
		if lastUserMsg != "" {
			emb, err := r.Embedder.Embed(lastUserMsg)
			if err != nil {
				log.Printf("[Router] Embedding generation failed: %v", err)
			} else {
				currentEmbedding = emb
				if cached := r.SemanticCache.Get(emb); cached != nil {
					return cached, nil
				}
			}
		}
	}

	// Compress messages via claw-compactor sidecar (if enabled)
	req.Messages = r.Compressor.Compress(req.Messages)

	// Normalize messages (flatten content arrays to strings)
	normalizeMessages(req.Messages)

	// Step 7: Try providers (sticky preferred first, then tier rotation)
	resp, err := r.tryProviders(ctx, req)
	if err != nil {
		return nil, err
	}

	// Step 8: Cache the successful response
	r.ExactCache.Set(ctx, req, resp)
	if currentEmbedding != nil {
		r.SemanticCache.Set(currentEmbedding, resp)
	}

	return resp, nil
}

// HandleStreamingRequest processes a streaming chat completion request.
// Returns a StreamingResponse with the raw SSE body from the provider.
func (r *Router) HandleStreamingRequest(ctx context.Context, req *types.ChatCompletionRequest) (*StreamingResponse, error) {
	// Compress messages via claw-compactor sidecar (if enabled)
	req.Messages = r.Compressor.Compress(req.Messages)

	// Normalize messages (flatten content arrays to strings)
	normalizeMessages(req.Messages)

	// Get keys with sticky preference applied
	keys := r.KeyPool.GetAvailableKeys()
	if len(keys) == 0 {
		return nil, fmt.Errorf("no available providers (all in cooldown)")
	}
	keys, sessionID := r.orderKeysWithSticky(keys, req.Messages)

	// Estimate token count for context limit checking
	estimatedTokens := estimateTokens(req)

	// Pre-serialize request for 400 diagnostics
	reqSnapshot, _ := json.Marshal(req)

	var lastErr error
	for _, key := range keys {
		// Skip providers whose context window is too small
		if key.MaxContextTokens > 0 && estimatedTokens > key.MaxContextTokens {
			log.Printf("[Router] Skipping %s: estimated %d tokens exceeds limit %d", key.Name, estimatedTokens, key.MaxContextTokens)
			continue
		}

		log.Printf("[Router] Trying provider %s (%s) for streaming", key.Name, key.Provider)

		sresp, err := r.callProviderStreaming(ctx, req, key)
		if err != nil {
			log.Printf("[Router] Provider %s failed: %v", key.Name, err)
			r.KeyPool.MarkFailure(key)
			lastErr = err
			continue
		}

		// Check for retryable status codes
		if retryableStatusCodes[sresp.StatusCode] {
			// Drain the body before closing to prevent hanging on streaming connections
			errBody, _ := io.ReadAll(sresp.Body)
			sresp.Body.Close()

			if sresp.StatusCode == 400 {
				log.Printf("[Router] 400 from %s (streaming) — request body: %s — response: %s", key.Name, string(reqSnapshot), string(errBody))
			} else {
				log.Printf("[Router] Provider %s returned %d, trying next — response: %s", key.Name, sresp.StatusCode, string(errBody))
			}

			r.KeyPool.MarkFailure(key)
			lastErr = fmt.Errorf("provider %s returned status %d: %s", key.Name, sresp.StatusCode, string(errBody))
			continue
		}

		r.KeyPool.MarkSuccess(key)
		r.StickyStore.Set(sessionID, key.Name)
		sresp.Model = key.Model
		return sresp, nil
	}

	return nil, fmt.Errorf("all providers failed: %v", lastErr)
}

// tryProviders attempts each available provider in order for non-streaming requests.
// Sticky preferred provider is tried first, then falls back to tier rotation.
func (r *Router) tryProviders(ctx context.Context, req *types.ChatCompletionRequest) (*types.ChatCompletionResponse, error) {
	keys := r.KeyPool.GetAvailableKeys()
	if len(keys) == 0 {
		return nil, fmt.Errorf("no available providers (all in cooldown)")
	}
	keys, sessionID := r.orderKeysWithSticky(keys, req.Messages)
	estimatedTokens := estimateTokens(req)

	var lastErr error
	for _, key := range keys {
		// Skip providers whose context window is too small
		if key.MaxContextTokens > 0 && estimatedTokens > key.MaxContextTokens {
			log.Printf("[Router] Skipping %s: estimated %d tokens exceeds limit %d", key.Name, estimatedTokens, key.MaxContextTokens)
			continue
		}

		log.Printf("[Router] Trying provider %s (%s)", key.Name, key.Provider)

		resp, err := r.callProvider(ctx, req, key)
		if err != nil {
			log.Printf("[Router] Provider %s failed: %v", key.Name, err)
			r.KeyPool.MarkFailure(key)
			lastErr = err
			continue
		}

		r.KeyPool.MarkSuccess(key)
		r.StickyStore.Set(sessionID, key.Name)
		return resp, nil
	}

	return nil, fmt.Errorf("all providers failed: %v", lastErr)
}

// callProvider makes a non-streaming HTTP request to a single provider.
func (r *Router) callProvider(ctx context.Context, req *types.ChatCompletionRequest, key *pool.APIKey) (*types.ChatCompletionResponse, error) {
	// Build provider request — override model and disable streaming
	providerReq := *req
	providerReq.Model = key.Model
	providerReq.Stream = false

	reqData, err := json.Marshal(providerReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/chat/completions", strings.TrimRight(key.BaseURL, "/"))
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+key.Key)

	resp, err := r.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if retryableStatusCodes[resp.StatusCode] {
		if resp.StatusCode == 400 {
			log.Printf("[Router] 400 from %s — request body sent: %s", key.Name, string(reqData))
		}
		return nil, fmt.Errorf("provider returned %d: %s", resp.StatusCode, string(body))
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("provider returned %d: %s", resp.StatusCode, string(body))
	}

	var completionResp types.ChatCompletionResponse
	if err := json.Unmarshal(body, &completionResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Ensure required fields
	if completionResp.Object == "" {
		completionResp.Object = "chat.completion"
	}
	if completionResp.Created == 0 {
		completionResp.Created = time.Now().Unix()
	}
	if completionResp.ID == "" {
		completionResp.ID = fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	}

	return &completionResp, nil
}

// callProviderStreaming makes a streaming HTTP request to a single provider, returning the raw body.
func (r *Router) callProviderStreaming(ctx context.Context, req *types.ChatCompletionRequest, key *pool.APIKey) (*StreamingResponse, error) {
	providerReq := *req
	providerReq.Model = key.Model
	providerReq.Stream = true

	reqData, err := json.Marshal(providerReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/chat/completions", strings.TrimRight(key.BaseURL, "/"))
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+key.Key)

	resp, err := r.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("streaming request failed: %w", err)
	}

	return &StreamingResponse{
		Body:       resp.Body,
		StatusCode: resp.StatusCode,
	}, nil
}

// normalizeMessages flattens Content arrays to plain strings for all messages.
func normalizeMessages(messages []types.ChatMessage) {
	for i := range messages {
		messages[i].Content = flattenContent(messages[i].Content)
	}
}

// flattenContent converts Content (string or array of {type, text} objects) to a plain string.
func flattenContent(content interface{}) string {
	if content == nil {
		return ""
	}

	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		var parts []string
		for _, item := range v {
			if m, ok := item.(map[string]interface{}); ok {
				if text, ok := m["text"].(string); ok {
					parts = append(parts, text)
				}
			}
		}
		return strings.Join(parts, " ")
	default:
		return fmt.Sprintf("%v", v)
	}
}

// getLastUserMessage extracts the text content of the last user message.
func getLastUserMessage(messages []types.ChatMessage) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return flattenContent(messages[i].Content)
		}
	}
	return ""
}
