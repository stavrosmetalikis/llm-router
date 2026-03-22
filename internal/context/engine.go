package context

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"llm-router/internal/pool"
	"llm-router/internal/types"
)

// Engine handles context compression and memory injection for conversations.
type Engine struct {
	maxMessages int
	keyPool     *pool.KeyPool
	client      *http.Client
}

// NewEngine creates a new context Engine.
func NewEngine(maxMessages int, keyPool *pool.KeyPool) *Engine {
	return &Engine{
		maxMessages: maxMessages,
		keyPool:     keyPool,
		client:      &http.Client{Timeout: 30 * time.Second},
	}
}

// CompressContext compresses conversation history if it exceeds maxMessages.
// It summarizes older messages and returns a reduced message list with a memory system message.
func (e *Engine) CompressContext(messages []types.ChatMessage) []types.ChatMessage {
	if len(messages) <= e.maxMessages {
		return messages
	}

	// Separate system prompt (if any), older messages, and recent messages
	var systemMsgs []types.ChatMessage
	var otherMsgs []types.ChatMessage

	for _, m := range messages {
		if m.Role == "system" {
			systemMsgs = append(systemMsgs, m)
		} else {
			otherMsgs = append(otherMsgs, m)
		}
	}

	// Keep last N non-system messages
	keepCount := e.maxMessages - len(systemMsgs)
	if keepCount < 2 {
		keepCount = 2
	}
	if keepCount > len(otherMsgs) {
		return messages // Not enough to compress
	}

	olderMsgs := otherMsgs[:len(otherMsgs)-keepCount]
	recentMsgs := otherMsgs[len(otherMsgs)-keepCount:]

	// Generate summary of older messages
	summary := e.summarize(olderMsgs)
	if summary == "" {
		return messages // Summarization failed, keep original
	}

	// Build compressed message list
	var result []types.ChatMessage
	result = append(result, systemMsgs...)
	result = append(result, types.ChatMessage{
		Role:    "system",
		Content: fmt.Sprintf("[MEMORY] Previous conversation summary: %s", summary),
	})
	result = append(result, recentMsgs...)

	log.Printf("[Context] Compressed %d messages down to %d", len(messages), len(result))
	return result
}

// InjectMemory prepends a memory system message at the start of the messages array.
func (e *Engine) InjectMemory(messages []types.ChatMessage, summary string) []types.ChatMessage {
	if summary == "" {
		return messages
	}

	memoryMsg := types.ChatMessage{
		Role:    "system",
		Content: fmt.Sprintf("[MEMORY] Previous conversation summary: %s", summary),
	}

	result := make([]types.ChatMessage, 0, len(messages)+1)
	result = append(result, memoryMsg)
	result = append(result, messages...)
	return result
}

// summarize uses a cheap/fast provider to generate a conversation summary.
func (e *Engine) summarize(messages []types.ChatMessage) string {
	keys := e.keyPool.GetAvailableKeys()
	if len(keys) == 0 {
		log.Printf("[Context] No available keys for summarization")
		return ""
	}

	// Build a prompt for summarization
	var sb strings.Builder
	for _, m := range messages {
		content := flattenContent(m.Content)
		sb.WriteString(fmt.Sprintf("%s: %s\n", m.Role, content))
	}

	summaryReq := types.ChatCompletionRequest{
		Model: keys[0].Model,
		Messages: []types.ChatMessage{
			{
				Role:    "system",
				Content: "Summarize the following conversation concisely in 2-3 sentences. Focus on key topics, decisions, and context that would be needed to continue the conversation.",
			},
			{
				Role:    "user",
				Content: sb.String(),
			},
		},
	}

	reqData, err := json.Marshal(summaryReq)
	if err != nil {
		log.Printf("[Context] Failed to marshal summary request: %v", err)
		return ""
	}

	// Try first available key
	key := keys[0]
	url := fmt.Sprintf("%s/chat/completions", strings.TrimRight(key.BaseURL, "/"))
	req, err := http.NewRequest("POST", url, bytes.NewReader(reqData))
	if err != nil {
		return ""
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key.Key)

	resp, err := e.client.Do(req)
	if err != nil {
		log.Printf("[Context] Summary request failed: %v", err)
		return ""
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ""
	}

	if resp.StatusCode != http.StatusOK {
		log.Printf("[Context] Summary API returned %d: %s", resp.StatusCode, string(body))
		return ""
	}

	var completionResp types.ChatCompletionResponse
	if err := json.Unmarshal(body, &completionResp); err != nil {
		return ""
	}

	if len(completionResp.Choices) > 0 {
		return completionResp.Choices[0].Message.Content
	}

	return ""
}

// flattenContent converts Content (string or array) to a plain string.
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
