package compressor

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"llm-router/internal/types"
)

const sidecarURL = "http://localhost:8081/compress"

// isCompactionRequest detects OpenClaw compaction requests by looking for explicit keywords
// in the conversation's messages.
func isCompactionRequest(messages []types.ChatMessage) bool {
    for _, m := range messages {
        content := fmt.Sprintf("%v", m.Content)
        if strings.Contains(content, "NO_REPLY") ||
           strings.Contains(content, "memory/") ||
           strings.Contains(content, "compaction") ||
           strings.Contains(content, "summarize") ||
           m.Role == "system" && strings.Contains(content, "nearing compaction") {
            return true
        }
    }
    return false
}

// compressRequest is the request body sent to the claw-compactor sidecar.
type compressRequest struct {
	Messages []types.ChatMessage `json:"messages"`
}

// compressResponse is the response body from the claw-compactor sidecar.
type compressResponse struct {
	Messages    []types.ChatMessage `json:"messages"`
	SavedTokens int                 `json:"saved_tokens"`
}

// Compressor calls the claw-compactor sidecar to compress messages before
// forwarding them to providers. If the sidecar is unreachable or returns
// an error, the original messages are returned unchanged.
type Compressor struct {
	enabled bool
	client  *http.Client
}

// NewCompressor creates a new Compressor. If enabled is false, Compress()
// is a no-op that returns messages unchanged.
func NewCompressor(enabled bool) *Compressor {
	return &Compressor{
		enabled: enabled,
		client:  &http.Client{Timeout: 200 * time.Millisecond},
	}
}

// Compress sends messages to the claw-compactor sidecar for token compression.
// Returns the original messages unchanged if:
//   - compression is disabled
//   - it is a compaction request (summarization)
//   - the sidecar is unreachable
//   - the sidecar returns an error
//
// This method never fails a request — compression is best-effort.
func (c *Compressor) Compress(messages []types.ChatMessage) []types.ChatMessage {
	if !c.enabled || isCompactionRequest(messages) {
		return messages
	}

	reqBody := compressRequest{Messages: messages}
	data, err := json.Marshal(reqBody)
	if err != nil {
		log.Printf("[Compressor] Failed to marshal request: %v", err)
		return messages
	}

	resp, err := c.client.Post(sidecarURL, "application/json", bytes.NewReader(data))
	if err != nil {
		log.Printf("[Compressor] Sidecar unreachable: %v", err)
		return messages
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("[Compressor] Sidecar returned status %d", resp.StatusCode)
		return messages
	}

	var compResp compressResponse
	if err := json.NewDecoder(resp.Body).Decode(&compResp); err != nil {
		log.Printf("[Compressor] Failed to decode response: %v", err)
		return messages
	}

	if len(compResp.Messages) == 0 {
		log.Printf("[Compressor] Sidecar returned empty messages, using originals")
		return messages
	}

	if compResp.SavedTokens > 0 {
		log.Printf("[Compressor] Saved ~%d tokens (%d messages)", compResp.SavedTokens, len(compResp.Messages))
	}

	return compResp.Messages
}
