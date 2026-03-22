package pool

import (
	"crypto/sha256"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"
)

const stickyTTL = 30 * time.Minute

// stickyEntry tracks the preferred provider for a session.
type stickyEntry struct {
	KeyName   string
	UpdatedAt time.Time
}

// StickyStore maps session IDs to preferred provider key names with TTL expiration.
type StickyStore struct {
	mu      sync.RWMutex
	entries map[string]*stickyEntry
}

// NewStickyStore creates a new StickyStore.
func NewStickyStore() *StickyStore {
	s := &StickyStore{
		entries: make(map[string]*stickyEntry),
	}
	// Background cleanup every 5 minutes
	go s.cleanup()
	return s
}

// SessionKey derives a session identifier from the messages in a request.
// Uses a hash of the system prompt + first user message to identify the conversation thread.
func SessionKey(messages []ChatMessage) string {
	var parts []string
	for _, m := range messages {
		if m.Role == "system" {
			parts = append(parts, "sys:"+contentToString(m.Content))
			break
		}
	}
	for _, m := range messages {
		if m.Role == "user" {
			parts = append(parts, "usr:"+contentToString(m.Content))
			break
		}
	}
	if len(parts) == 0 {
		return ""
	}
	hash := sha256.Sum256([]byte(strings.Join(parts, "|")))
	return fmt.Sprintf("%x", hash[:16]) // 32-char hex
}

// ChatMessage is a minimal struct to avoid circular imports.
// The router converts types.ChatMessage to this for session key derivation.
type ChatMessage struct {
	Role    string
	Content interface{}
}

// contentToString flattens content (string or array) to a plain string.
func contentToString(content interface{}) string {
	if content == nil {
		return ""
	}
	switch v := content.(type) {
	case string:
		return v
	default:
		return fmt.Sprintf("%v", v)
	}
}

// Get returns the preferred key name for a session, or "" if not set or expired.
func (s *StickyStore) Get(sessionID string) string {
	if sessionID == "" {
		return ""
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	entry, ok := s.entries[sessionID]
	if !ok {
		return ""
	}
	if time.Since(entry.UpdatedAt) > stickyTTL {
		return "" // expired, will be cleaned up later
	}
	return entry.KeyName
}

// Set stores the preferred key name for a session.
func (s *StickyStore) Set(sessionID, keyName string) {
	if sessionID == "" || keyName == "" {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.entries[sessionID] = &stickyEntry{
		KeyName:   keyName,
		UpdatedAt: time.Now(),
	}
	log.Printf("[Sticky] Session %s → preferred provider: %s", sessionID[:8], keyName)
}

// cleanup runs periodically to remove expired entries.
func (s *StickyStore) cleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	for range ticker.C {
		s.mu.Lock()
		now := time.Now()
		for id, entry := range s.entries {
			if now.Sub(entry.UpdatedAt) > stickyTTL {
				delete(s.entries, id)
			}
		}
		s.mu.Unlock()
	}
}
