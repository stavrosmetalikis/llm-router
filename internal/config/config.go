package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// KeyConfig represents a single provider key entry in the config file.
type KeyConfig struct {
	Name     string `yaml:"name"`
	Provider string `yaml:"provider"`
	Key      string `yaml:"key"`
	BaseURL  string `yaml:"base_url"`
	Model    string `yaml:"model"`
}

// Config holds the full application configuration.
type Config struct {
	Keys              []KeyConfig `yaml:"keys"`
	GeminiAPIKey      string      `yaml:"gemini_api_key"`
	RedisAddr         string      `yaml:"redis_addr"`
	MaxMessages       int         `yaml:"max_messages"`
	SemanticCacheSize int         `yaml:"semantic_cache_size"`
	SemanticThreshold float64     `yaml:"semantic_threshold"`
}

// Load reads and parses a YAML config file at the given path.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", path, err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Apply defaults
	if cfg.MaxMessages <= 0 {
		cfg.MaxMessages = 12
	}
	if cfg.SemanticCacheSize <= 0 {
		cfg.SemanticCacheSize = 100
	}
	if cfg.SemanticThreshold <= 0 {
		cfg.SemanticThreshold = 0.92
	}
	if cfg.RedisAddr == "" {
		cfg.RedisAddr = "localhost:6379"
	}

	return &cfg, nil
}
