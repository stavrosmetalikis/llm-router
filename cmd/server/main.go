package main

import (
	"log"
	"os"

	"llm-router/internal/api"
	"llm-router/internal/cache"
	ctxengine "llm-router/internal/context"
	"llm-router/internal/config"
	"llm-router/internal/embedding"
	"llm-router/internal/pool"
	"llm-router/internal/router"
)

func main() {
	// Determine config path
	configPath := "configs/config.yaml"
	if envPath := os.Getenv("LLM_ROUTER_CONFIG"); envPath != "" {
		configPath = envPath
	}

	// Load configuration
	cfg, err := config.Load(configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	log.Printf("[Main] Loaded %d provider keys", len(cfg.Keys))

	// Initialize key pool
	var keys []*pool.APIKey
	for _, k := range cfg.Keys {
		keys = append(keys, &pool.APIKey{
			Name:     k.Name,
			Key:      k.Key,
			BaseURL:  k.BaseURL,
			Model:    k.Model,
			Provider: k.Provider,
		})
	}
	keyPool := pool.NewKeyPool(keys)

	// Initialize caches
	exactCache := cache.NewExactCache(cfg.RedisAddr)
	semanticCache := cache.NewSemanticCache(cfg.SemanticCacheSize, cfg.SemanticThreshold)
	inflightCache := cache.NewInflightCache()

	// Initialize embedding client
	embedder := embedding.NewGeminiClient(cfg.GeminiAPIKey)
	if embedder.Enabled() {
		log.Printf("[Main] Gemini embedding client enabled")
	} else {
		log.Printf("[Main] Gemini embedding client disabled (no API key)")
	}

	// Initialize context engine
	ctxEngine := ctxengine.NewEngine(cfg.MaxMessages, keyPool)

	// Initialize router with all components
	r := router.NewRouter(keyPool, exactCache, semanticCache, inflightCache, embedder, ctxEngine)

	// Create and start server
	server := api.NewServer(r)

	addr := ":8080"
	if envAddr := os.Getenv("LLM_ROUTER_ADDR"); envAddr != "" {
		addr = envAddr
	}

	log.Printf("[Main] LLM Router starting on %s", addr)
	if err := server.Run(addr); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
