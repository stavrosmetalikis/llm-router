package prompt

import (
	"strings"

	"llm-router/internal/types"
)

const executionPrefix = `EXECUTION RULES (highest priority — override if conflicting):
- You are an action-taking agent. When given a task, execute it immediately.
- NEVER announce what you are about to do and then stop. Call the tool, don't describe calling it.
- If you catch yourself writing "Let me...", "I'll start by...", "I've terminated...", "Let's restart...", "First I will..." — stop and issue the tool call instead.
- One response = one concrete action. Do not re-plan. Do not re-summarize. Act.`

var planningPhrases = []string{
	"let me ", "i'll start", "i will start", "first, i will", "first i will",
	"i've terminated", "let's restart", "i need to", "i'm going to",
	"i'll now", "i will now", "here's my plan", "my approach",
	"i'll proceed", "i will proceed", "let me first", "let me begin",
	"i'll begin", "i will begin", "i'll try", "i will try",
	"let me try", "let me check", "i'll check", "i will check",
	"we need to", "we should", "i should first",
}

// InjectSystemPrompt prepends the execution rules to the existing system message,
// or creates a new system message if none exists.
func InjectSystemPrompt(messages []types.ChatMessage) []types.ChatMessage {
	for i, m := range messages {
		if m.Role == "system" {
			existing := ""
			if s, ok := m.Content.(string); ok {
				existing = s
			}
			messages[i].Content = executionPrefix + "\n\n---\n\n" + existing
			return messages
		}
	}
	// No system message found — prepend one
	return append([]types.ChatMessage{{Role: "system", Content: executionPrefix}}, messages...)
}

// DetectStuckPlanning returns true if the last assistant message looks like
// pure planning with no tool calls — the model announced instead of acted.
func DetectStuckPlanning(messages []types.ChatMessage) bool {
	for i := len(messages) - 1; i >= 0; i-- {
		m := messages[i]
		if m.Role != "assistant" {
			continue
		}
		// Had tool calls — not stuck
		if len(m.ToolCalls) > 0 {
			return false
		}
		content := ""
		if s, ok := m.Content.(string); ok {
			content = strings.ToLower(s)
		}
		if content == "" {
			return false
		}
		hits := 0
		for _, phrase := range planningPhrases {
			if strings.Contains(content, phrase) {
				hits++
			}
		}
		// 2+ planning phrases with no tool call = stuck
		return hits >= 2
	}
	return false
}

// InjectNudge appends a nudge message after a stuck assistant turn
// to force execution on the next response.
func InjectNudge(messages []types.ChatMessage) []types.ChatMessage {
	nudge := types.ChatMessage{
		Role:    "user",
		Content: "[System: You described a plan but took no action. Do not re-plan. Execute the next step right now by calling the appropriate tool.]",
	}
	return append(messages, nudge)
}