package pool

import (
	"math"
	"sort"
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
	Priority      int
	Failures      int
	CooldownUntil time.Time
}

// tier groups keys at the same priority level with a round-robin counter.
type tier struct {
	priority int
	keys     []*APIKey
	next     uint64 // round-robin index
}

// KeyPool manages keys grouped by priority tiers with round-robin within each tier.
type KeyPool struct {
	mu    sync.Mutex
	tiers []*tier // sorted ascending by priority (1 = highest)
}

// NewKeyPool creates a new KeyPool, grouping keys by Priority and sorting tiers ascending.
func NewKeyPool(keys []*APIKey) *KeyPool {
	// Group keys by priority
	tierMap := make(map[int]*tier)
	for _, k := range keys {
		p := k.Priority
		if p <= 0 {
			p = 1 // default to highest priority
		}
		t, ok := tierMap[p]
		if !ok {
			t = &tier{priority: p}
			tierMap[p] = t
		}
		t.keys = append(t.keys, k)
	}

	// Sort tiers by priority ascending
	tiers := make([]*tier, 0, len(tierMap))
	for _, t := range tierMap {
		tiers = append(tiers, t)
	}
	sort.Slice(tiers, func(i, j int) bool {
		return tiers[i].priority < tiers[j].priority
	})

	return &KeyPool{tiers: tiers}
}

// GetAvailableKeys returns the available keys from the highest-priority tier that has
// at least one non-cooldown key, ordered by round-robin rotation within that tier.
// Falls to the next tier only when ALL keys in the current tier are in cooldown.
func (p *KeyPool) GetAvailableKeys() []*APIKey {
	p.mu.Lock()
	defer p.mu.Unlock()

	now := time.Now()

	for _, t := range p.tiers {
		// Collect available keys in this tier
		var available []*APIKey
		for _, k := range t.keys {
			if now.After(k.CooldownUntil) {
				available = append(available, k)
			}
		}

		if len(available) == 0 {
			continue // entire tier in cooldown — fall to next tier
		}

		// Round-robin: rotate the available list starting from t.next
		n := len(available)
		startIdx := int(t.next % uint64(n))
		t.next++

		rotated := make([]*APIKey, n)
		for i := 0; i < n; i++ {
			rotated[i] = available[(startIdx+i)%n]
		}

		return rotated
	}

	return nil // all tiers exhausted
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

// AllKeys returns all keys across all tiers regardless of cooldown status.
func (p *KeyPool) AllKeys() []*APIKey {
	p.mu.Lock()
	defer p.mu.Unlock()

	var result []*APIKey
	for _, t := range p.tiers {
		result = append(result, t.keys...)
	}
	return result
}
