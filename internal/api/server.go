package api

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"llm-router/internal/router"
	"llm-router/internal/types"
)

// Server wraps a Gin engine and the LLM router.
type Server struct {
	engine *gin.Engine
	router *router.Router
}

// NewServer creates a new API server with all routes configured.
func NewServer(r *router.Router) *Server {
	gin.SetMode(gin.ReleaseMode)
	engine := gin.New()
	engine.Use(gin.Logger(), gin.Recovery())

	s := &Server{
		engine: engine,
		router: r,
	}

	// OpenAI-compatible endpoint
	engine.POST("/v1/chat/completions", s.handleChatCompletion)

	// Health check
	engine.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	return s
}

// Run starts the HTTP server on the given address.
func (s *Server) Run(addr string) error {
	log.Printf("[Server] Starting on %s", addr)
	return s.engine.Run(addr)
}

// handleChatCompletion handles POST /v1/chat/completions.
func (s *Server) handleChatCompletion(c *gin.Context) {
	var req types.ChatCompletionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": gin.H{
				"message": fmt.Sprintf("Invalid request body: %v", err),
				"type":    "invalid_request_error",
			},
		})
		return
	}

	if len(req.Messages) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": gin.H{
				"message": "Messages array is required and must not be empty",
				"type":    "invalid_request_error",
			},
		})
		return
	}

	if req.Stream {
		s.handleStreaming(c, &req)
	} else {
		s.handleNonStreaming(c, &req)
	}
}

// handleNonStreaming handles non-streaming chat completion requests.
func (s *Server) handleNonStreaming(c *gin.Context, req *types.ChatCompletionRequest) {
	resp, err := s.router.HandleRequest(c.Request.Context(), req)
	if err != nil {
		log.Printf("[Server] Request failed: %v", err)
		c.JSON(http.StatusBadGateway, gin.H{
			"error": gin.H{
				"message": err.Error(),
				"type":    "server_error",
			},
		})
		return
	}

	// Ensure OpenClaw-required fields
	if resp.Created == 0 {
		resp.Created = time.Now().Unix()
	}
	if resp.Object == "" {
		resp.Object = "chat.completion"
	}

	c.JSON(http.StatusOK, resp)
}

// handleStreaming handles streaming (SSE) chat completion requests.
func (s *Server) handleStreaming(c *gin.Context, req *types.ChatCompletionRequest) {
	sresp, err := s.router.HandleStreamingRequest(c.Request.Context(), req)
	if err != nil {
		log.Printf("[Server] Streaming request failed: %v", err)
		c.JSON(http.StatusBadGateway, gin.H{
			"error": gin.H{
				"message": err.Error(),
				"type":    "server_error",
			},
		})
		return
	}
	defer sresp.Body.Close()

	// Set SSE headers
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Transfer-Encoding", "chunked")

	c.Stream(func(w io.Writer) bool {
		scanner := bufio.NewScanner(sresp.Body)
		// Increase scanner buffer for large chunks
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

		for scanner.Scan() {
			line := scanner.Text()

			// Forward the SSE line as-is
			fmt.Fprintf(w, "%s\n", line)

			// Flush after each line to ensure real-time delivery
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		}

		if err := scanner.Err(); err != nil {
			log.Printf("[Server] SSE scan error: %v", err)
		}

		return false // Stop streaming
	})
}
