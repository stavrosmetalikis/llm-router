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
	"llm-router/internal/prompt"
	"llm-router/internal/types"
)

var retryableStatusCodes = map[int]bool{
	400: true, 401: true, 403: true, 404: true, 429: true, 500: true,
}

var codeMarkers = []string{"{", "}", "func ", "def ", "class ", "=>", "//", "/*", "import ", "return ", "\"type\":", "\"key\":"}

func looksLikeCode(text string) bool {
	for _, marker := range codeMarkers {
		if strings.Contains(text, marker) {
			return true
		}
	}
	return false
}

func estimateTokens(req *types.ChatCompletionRequest) int {
	totalChars := 0
	var allContent strings.Builder
	for _, m := range req.Messages {
		text := flattenContent(m.Content)
		totalChars += len(text)
		allContent.WriteString(text)
		totalChars += 10
	}
	if len(req.Tools) > 0 {
		for _, t := range req.Tools {
			totalChars += len(t.Function.Name) + len(t.Function.Description) + 50
			if t.Function.Parameters != nil {
				paraBytes, _ := json.Marshal(t.Function.Parameters)
				totalChars += len(paraBytes)
			}
		}
	}
	if looksLikeCode(allContent.String()) {
		return totalChars / 3
	}
	return totalChars / 4
}

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
		KeyPool: keyPool, StickyStore: stickyStore,
		ExactCache: exactCache, SemanticCache: semanticCache,
		InflightCache: inflightCache, Embedder: embedder,
		Compressor: comp, HTTPClient: &http.Client{Timeout: 120 * time.Second},
	}
}

type StreamingResponse struct {
	Body       io.ReadCloser
	StatusCode int
	Model      string
	Usage      *types.Usage
}

func sessionKeyFromRequest(messages []types.ChatMessage) string {
	converted := make([]pool.ChatMessage, len(messages))
	for i, m := range messages {
		converted[i] = pool.ChatMessage{Role: m.Role, Content: m.Content}
	}
	return pool.SessionKey(converted)
}

func (r *Router) orderKeysWithSticky(keys []*pool.APIKey, messages []types.ChatMessage) ([]*pool.APIKey, string) {
	sessionID := sessionKeyFromRequest(messages)
	preferred := r.StickyStore.Get(sessionID)
	if preferred == "" {
		return keys, sessionID
	}
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
	log.Printf("[Sticky] Session %s: preferred provider %s not available, falling back", sessionID[:8], preferred)
	return keys, sessionID
}

// applyExecutionGuardrails injects the execution-forcing system prompt and,
// if the last assistant turn looks like stuck planning, appends a nudge.
// Applied to every request regardless of provider.
func applyExecutionGuardrails(messages []types.ChatMessage) []types.ChatMessage {
	msgs := prompt.InjectSystemPrompt(messages)
	if prompt.DetectStuckPlanning(msgs) {
		log.Printf("[Router] Stuck planning detected — injecting execution nudge")
		msgs = prompt.InjectNudge(msgs)
	}
	return msgs
}

func (r *Router) HandleRequest(ctx context.Context, req *types.ChatCompletionRequest) (*types.ChatCompletionResponse, error) {
	resp, err, shared := r.InflightCache.Do(req, func() (*types.ChatCompletionResponse, error) {
		return r.executeNonStreamingPipeline(ctx, req)
	})
	if shared {
		log.Printf("[Router] Returned shared in-flight result")
	}
	return resp, err
}

func (r *Router) executeNonStreamingPipeline(ctx context.Context, req *types.ChatCompletionRequest) (*types.ChatCompletionResponse, error) {
	isCompaction := compressor.IsCompactionRequest(req.Messages)

	if !isCompaction {
		if cached := r.ExactCache.Get(ctx, req); cached != nil {
			return cached, nil
		}
	}

	var currentEmbedding []float64
	if !isCompaction && r.Embedder.Enabled() {
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

	if isCompaction {
		log.Printf("[Router] Compaction request — bypassing cache and compression")
	}

	req.Messages = r.Compressor.Compress(req.Messages)
	normalizeMessages(req.Messages)

	resp, err := r.tryProviders(ctx, req)
	if err != nil {
		return nil, err
	}

	if !isCompaction {
		r.ExactCache.Set(ctx, req, resp)
		if currentEmbedding != nil {
			r.SemanticCache.Set(currentEmbedding, resp)
		}
	}

	return resp, nil
}

func (r *Router) HandleStreamingRequest(ctx context.Context, req *types.ChatCompletionRequest) (*StreamingResponse, error) {
	isCompaction := compressor.IsCompactionRequest(req.Messages)
	if isCompaction {
		log.Printf("[Router] Compaction request — bypassing compression")
	}

	req.Messages = r.Compressor.Compress(req.Messages)
	normalizeMessages(req.Messages)

	keys := r.KeyPool.GetAvailableKeys()
	if len(keys) == 0 {
		return nil, fmt.Errorf("no available providers (all in cooldown)")
	}
	keys, sessionID := r.orderKeysWithSticky(keys, req.Messages)

	estimatedTokens := estimateTokens(req)
	reqSnapshot, _ := json.Marshal(req)

	// Apply execution guardrails to the messages once, before trying any provider.
	// Skip for compaction — those must pass through unmodified.
	guardedMessages := req.Messages
	if !isCompaction {
		guardedMessages = applyExecutionGuardrails(req.Messages)
	}

	var lastErr error
	for _, key := range keys {
		if key.MaxContextTokens > 0 && estimatedTokens > key.MaxContextTokens {
			log.Printf("[Router] Skipping %s: estimated %d tokens exceeds limit %d", key.Name, estimatedTokens, key.MaxContextTokens)
			continue
		}

		log.Printf("[Router] Trying provider %s (%s) for streaming", key.Name, key.Provider)

		activeReq := *req
		activeReq.Messages = guardedMessages

		sresp, err := r.callProviderStreaming(ctx, &activeReq, key)
		if err != nil {
			log.Printf("[Router] Provider %s failed: %v", key.Name, err)
			r.KeyPool.MarkFailure(key)
			lastErr = err
			continue
		}

		if retryableStatusCodes[sresp.StatusCode] {
			errBody, _ := io.ReadAll(sresp.Body)
			sresp.Body.Close()
			if sresp.StatusCode == 400 {
				log.Printf("[Router] 400 from %s — request: %s — response: %s", key.Name, string(reqSnapshot), string(errBody))
			} else {
				log.Printf("[Router] Provider %s returned %d — response: %s", key.Name, sresp.StatusCode, string(errBody))
			}
			r.KeyPool.MarkFailure(key)
			lastErr = fmt.Errorf("provider %s returned status %d: %s", key.Name, sresp.StatusCode, string(errBody))
			continue
		}

		r.KeyPool.MarkSuccess(key)
		r.StickyStore.Set(sessionID, key.Name)
		sresp.Model = key.Model
		promptTokens := estimateTokens(req)
		sresp.Usage = &types.Usage{
			PromptTokens: promptTokens, CompletionTokens: 0, TotalTokens: promptTokens,
		}
		return sresp, nil
	}

	return nil, fmt.Errorf("all providers failed: %v", lastErr)
}

func (r *Router) tryProviders(ctx context.Context, req *types.ChatCompletionRequest) (*types.ChatCompletionResponse, error) {
	keys := r.KeyPool.GetAvailableKeys()
	if len(keys) == 0 {
		return nil, fmt.Errorf("no available providers (all in cooldown)")
	}
	keys, sessionID := r.orderKeysWithSticky(keys, req.Messages)
	estimatedTokens := estimateTokens(req)

	isCompaction := compressor.IsCompactionRequest(req.Messages)
	guardedMessages := req.Messages
	if !isCompaction {
		guardedMessages = applyExecutionGuardrails(req.Messages)
	}

	var lastErr error
	for _, key := range keys {
		if key.MaxContextTokens > 0 && estimatedTokens > key.MaxContextTokens {
			log.Printf("[Router] Skipping %s: estimated %d tokens exceeds limit %d", key.Name, estimatedTokens, key.MaxContextTokens)
			continue
		}

		log.Printf("[Router] Trying provider %s (%s)", key.Name, key.Provider)

		activeReq := *req
		activeReq.Messages = guardedMessages

		resp, err := r.callProvider(ctx, &activeReq, key)
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

func (r *Router) callProvider(ctx context.Context, req *types.ChatCompletionRequest, key *pool.APIKey) (*types.ChatCompletionResponse, error) {
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
			log.Printf("[Router] 400 from %s — request body: %s", key.Name, string(reqData))
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

	if completionResp.Object == "" {
		completionResp.Object = "chat.completion"
	}
	if completionResp.Created == 0 {
		completionResp.Created = time.Now().Unix()
	}
	if completionResp.ID == "" {
		completionResp.ID = fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	}
	if completionResp.Usage == nil {
		promptTokens := estimateTokens(req)
		completionTokens := 0
		if len(completionResp.Choices) > 0 {
			completionTokens = len(flattenContent(completionResp.Choices[0].Message.Content)) / 4
		}
		completionResp.Usage = &types.Usage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		}
	}

	return &completionResp, nil
}

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

	return &StreamingResponse{Body: resp.Body, StatusCode: resp.StatusCode}, nil
}

func normalizeMessages(messages []types.ChatMessage) {
	for i := range messages {
		messages[i].Content = flattenContent(messages[i].Content)
	}
}

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

func getLastUserMessage(messages []types.ChatMessage) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return flattenContent(messages[i].Content)
		}
	}
	return ""
}