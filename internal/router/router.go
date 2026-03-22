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
	ctxengine "llm-router/internal/context"
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

// Router is the core orchestrator that implements the full request pipeline.
type Router struct {
	KeyPool       *pool.KeyPool
	ExactCache    *cache.ExactCache
	SemanticCache *cache.SemanticCache
	InflightCache *cache.InflightCache
	Embedder      *embedding.GeminiClient
	CtxEngine     *ctxengine.Engine
	HTTPClient    *http.Client
}

// NewRouter creates a new Router with all components wired together.
func NewRouter(
	keyPool *pool.KeyPool,
	exactCache *cache.ExactCache,
	semanticCache *cache.SemanticCache,
	inflightCache *cache.InflightCache,
	embedder *embedding.GeminiClient,
	ctxEngine *ctxengine.Engine,
) *Router {
	return &Router{
		KeyPool:       keyPool,
		ExactCache:    exactCache,
		SemanticCache: semanticCache,
		InflightCache: inflightCache,
		Embedder:      embedder,
		CtxEngine:     ctxEngine,
		HTTPClient:    &http.Client{Timeout: 120 * time.Second},
	}
}

// StreamingResponse holds raw SSE bytes from a streaming provider response.
type StreamingResponse struct {
	Body       io.ReadCloser
	StatusCode int
	Model      string
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

	// Step 5: Compress context if needed
	req.Messages = r.CtxEngine.CompressContext(req.Messages)

	// Step 6: Normalize messages
	normalizeMessages(req.Messages)

	// Step 7: Try providers in order
	resp, err := r.tryProviders(ctx, req, false)
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
	// Compress context if needed
	req.Messages = r.CtxEngine.CompressContext(req.Messages)

	// Normalize messages
	normalizeMessages(req.Messages)

	// Try providers in order for streaming
	keys := r.KeyPool.GetAvailableKeys()
	if len(keys) == 0 {
		return nil, fmt.Errorf("no available providers (all in cooldown)")
	}

	var lastErr error
	for _, key := range keys {
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
			log.Printf("[Router] Provider %s returned %d, trying next", key.Name, sresp.StatusCode)
			sresp.Body.Close()
			r.KeyPool.MarkFailure(key)
			lastErr = fmt.Errorf("provider %s returned status %d", key.Name, sresp.StatusCode)
			continue
		}

		r.KeyPool.MarkSuccess(key)
		sresp.Model = key.Model
		return sresp, nil
	}

	return nil, fmt.Errorf("all providers failed: %v", lastErr)
}

// tryProviders attempts each available provider in order for non-streaming requests.
func (r *Router) tryProviders(ctx context.Context, req *types.ChatCompletionRequest, stream bool) (*types.ChatCompletionResponse, error) {
	keys := r.KeyPool.GetAvailableKeys()
	if len(keys) == 0 {
		return nil, fmt.Errorf("no available providers (all in cooldown)")
	}

	var lastErr error
	for _, key := range keys {
		log.Printf("[Router] Trying provider %s (%s)", key.Name, key.Provider)

		resp, err := r.callProvider(ctx, req, key)
		if err != nil {
			log.Printf("[Router] Provider %s failed: %v", key.Name, err)
			r.KeyPool.MarkFailure(key)
			lastErr = err
			continue
		}

		r.KeyPool.MarkSuccess(key)
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
