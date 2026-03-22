package pool

import (
	"math"
	"sync"
	"time"
)

// APIKey represents a single provider API key with cooldown tracking.
type APIKey struct {
	Name          string
	Key           string
	BaseURL       string
	Model         string
	Provider      string
	Failures      int
	CooldownUntil time.Time
}

// KeyPool manages an ordered list of API keys with exponential backoff cooldowns.
type KeyPool struct {
	mu   sync.Mutex
	keys []*APIKey
}

// NewKeyPool creates a new KeyPool from a slice of APIKey definitions.
func NewKeyPool(keys []*APIKey) *KeyPool {
	return &KeyPool{keys: keys}
}

// GetAvailableKeys returns all keys that are not currently in cooldown, in priority order.
func (p *KeyPool) GetAvailableKeys() []*APIKey {
	p.mu.Lock()
	defer p.mu.Unlock()

	now := time.Now()
	var available []*APIKey
	for _, k := range p.keys {
		if now.After(k.CooldownUntil) {
			available = append(available, k)
		}
	}
	return available
}

// MarkSuccess resets the failure count and cooldown for a key.
func (p *KeyPool) MarkSuccess(key *APIKey) {
	p.mu.Lock()
	defer p.mu.Unlock()

	key.Failures = 0
	key.CooldownUntil = time.Time{}
}

// MarkFailure increments the failure count and applies exponential backoff cooldown.
// Cooldown = 2^failures seconds, capped at 60 seconds.
func (p *KeyPool) MarkFailure(key *APIKey) {
	p.mu.Lock()
	defer p.mu.Unlock()

	key.Failures++
	cooldownSecs := math.Pow(2, float64(key.Failures))
	if cooldownSecs > 60 {
		cooldownSecs = 60
	}
	key.CooldownUntil = time.Now().Add(time.Duration(cooldownSecs) * time.Second)
}

// AllKeys returns all keys regardless of cooldown status.
func (p *KeyPool) AllKeys() []*APIKey {
	p.mu.Lock()
	defer p.mu.Unlock()

	result := make([]*APIKey, len(p.keys))
	copy(result, p.keys)
	return result
}
