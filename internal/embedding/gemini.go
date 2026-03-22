package embedding

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const geminiEmbeddingURL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"

// GeminiClient generates text embeddings via the Gemini Embedding API.
type GeminiClient struct {
	apiKey string
	client *http.Client
}

// NewGeminiClient creates a new GeminiClient with the given API key.
func NewGeminiClient(apiKey string) *GeminiClient {
	return &GeminiClient{
		apiKey: apiKey,
		client: &http.Client{Timeout: 10 * time.Second},
	}
}

// embeddingRequest is the request body for the Gemini embedding API.
type embeddingRequest struct {
	Content embeddingContent `json:"content"`
}

type embeddingContent struct {
	Parts []embeddingPart `json:"parts"`
}

type embeddingPart struct {
	Text string `json:"text"`
}

// embeddingResponse is the response body from the Gemini embedding API.
type embeddingResponse struct {
	Embedding struct {
		Values []float64 `json:"values"`
	} `json:"embedding"`
}

// Embed generates an embedding vector for the given text.
func (g *GeminiClient) Embed(text string) ([]float64, error) {
	if g.apiKey == "" {
		return nil, fmt.Errorf("gemini API key not configured")
	}

	reqBody := embeddingRequest{
		Content: embeddingContent{
			Parts: []embeddingPart{{Text: text}},
		},
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embedding request: %w", err)
	}

	url := fmt.Sprintf("%s?key=%s", geminiEmbeddingURL, g.apiKey)
	req, err := http.NewRequest("POST", url, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create embedding request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := g.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embedding API request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read embedding response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedding API returned %d: %s", resp.StatusCode, string(body))
	}

	var embResp embeddingResponse
	if err := json.Unmarshal(body, &embResp); err != nil {
		return nil, fmt.Errorf("failed to parse embedding response: %w", err)
	}

	if len(embResp.Embedding.Values) == 0 {
		return nil, fmt.Errorf("embedding API returned empty embedding")
	}

	return embResp.Embedding.Values, nil
}

// Enabled returns true if the client has a valid API key configured.
func (g *GeminiClient) Enabled() bool {
	return g.apiKey != "" && g.apiKey != "YOUR_GEMINI_API_KEY"
}
