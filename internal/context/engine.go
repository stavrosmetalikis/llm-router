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

	// Build compressed message list — merge ALL system messages and memory into
	// a single system message. Mistral and others reject multiple system messages.
	memoryBlock := fmt.Sprintf("\n\n[MEMORY] Previous conversation summary: %s", summary)

	var result []types.ChatMessage
	if len(systemMsgs) > 0 {
		// Merge all system messages + memory into a single system message
		var parts []string
		for _, sm := range systemMsgs {
			parts = append(parts, flattenContent(sm.Content))
		}
		merged := types.ChatMessage{
			Role:    "system",
			Content: strings.Join(parts, "\n\n") + memoryBlock,
		}
		result = append(result, merged)
	} else {
		// No system message exists — inject memory as a user message
		result = append(result, types.ChatMessage{
			Role:    "user",
			Content: fmt.Sprintf("[MEMORY] Previous conversation summary: %s", summary),
		})
	}
	result = append(result, recentMsgs...)

	log.Printf("[Context] Compressed %d messages down to %d", len(messages), len(result))
	return result
}

// InjectMemory merges a memory summary into the existing system message (index 0),
// or prepends it as a user message if no system message exists.
// This ensures only one system message is present (required by Mistral and others).
func (e *Engine) InjectMemory(messages []types.ChatMessage, summary string) []types.ChatMessage {
	if summary == "" {
		return messages
	}

	memoryBlock := fmt.Sprintf("\n\n[MEMORY] Previous conversation summary: %s", summary)

	result := make([]types.ChatMessage, len(messages))
	copy(result, messages)

	if len(result) > 0 && result[0].Role == "system" {
		// Merge into existing system message
		result[0].Content = flattenContent(result[0].Content) + memoryBlock
	} else {
		// No system message — prepend as a user message
		memoryMsg := types.ChatMessage{
			Role:    "user",
			Content: fmt.Sprintf("[MEMORY] Previous conversation summary: %s", summary),
		}
		result = append([]types.ChatMessage{memoryMsg}, result...)
	}

	return result
}

// summarize uses a cheap/fast provider to generate a conversation summary.
// Tries each available key in order until one succeeds.
func (e *Engine) summarize(messages []types.ChatMessage) string {
	keys := e.keyPool.GetAvailableKeys()
	if len(keys) == 0 {
		log.Printf("[Context] No available keys for summarization")
		return ""
	}

	// Build the conversation text to summarize
	var sb strings.Builder
	for _, m := range messages {
		content := flattenContent(m.Content)
		sb.WriteString(fmt.Sprintf("%s: %s\n", m.Role, content))
	}
	conversationText := sb.String()

	// Try each available key until one succeeds
	for _, key := range keys {
		summaryReq := types.ChatCompletionRequest{
			Model: key.Model,
			Messages: []types.ChatMessage{
				{
					Role:    "system",
					Content: "Summarize the following conversation concisely in 2-3 sentences. Focus on key topics, decisions, and context that would be needed to continue the conversation.",
				},
				{
					Role:    "user",
					Content: conversationText,
				},
			},
		}

		reqData, err := json.Marshal(summaryReq)
		if err != nil {
			log.Printf("[Context] Failed to marshal summary request: %v", err)
			continue
		}

		url := fmt.Sprintf("%s/chat/completions", strings.TrimRight(key.BaseURL, "/"))
		req, err := http.NewRequest("POST", url, bytes.NewReader(reqData))
		if err != nil {
			continue
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+key.Key)

		resp, err := e.client.Do(req)
		if err != nil {
			log.Printf("[Context] Summary request to %s failed: %v", key.Name, err)
			continue
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			continue
		}

		if resp.StatusCode != http.StatusOK {
			log.Printf("[Context] Summary API %s returned %d: %s", key.Name, resp.StatusCode, string(body))
			continue
		}

		var completionResp types.ChatCompletionResponse
		if err := json.Unmarshal(body, &completionResp); err != nil {
			continue
		}

		if len(completionResp.Choices) > 0 && completionResp.Choices[0].Message.Content != "" {
			log.Printf("[Context] Summary generated via %s (%s)", key.Name, key.Model)
			return completionResp.Choices[0].Message.Content
		}
	}

	log.Printf("[Context] All keys failed for summarization")
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
