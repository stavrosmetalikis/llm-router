# LLM Router

A self-hosted, OpenAI-compatible LLM gateway written in **Go** using the **Gin** framework. It sits between your AI agent platform (e.g. OpenClaw) and multiple LLM providers, handling failover, caching, context management, and streaming — all behind a single `/v1/chat/completions` endpoint.

```
Client → LLM Router (:8080) → Cerebras / Groq / Mistral / ...
```

---

## Features

| Feature | Description |
|---|---|
| **OpenAI-Compatible API** | Drop-in replacement — any client that speaks OpenAI works out of the box |
| **Multi-Provider Failover** | Routes through providers in priority order; automatically skips failures |
| **Streaming (SSE)** | Proxies Server-Sent Events directly from provider to client in real-time |
| **Key Pool & Cooldown** | Exponential backoff per key (2^failures sec, max 60s) — bad keys cool down automatically |
| **Exact Cache (Redis)** | SHA-256 hash of the full request → cached response with 5-minute TTL |
| **Semantic Cache (In-Memory)** | Cosine similarity on Gemini embeddings — returns cached responses for semantically similar questions |
| **In-Flight Deduplication** | Identical concurrent requests (e.g. double-tap) share a single provider call |
| **Context Compression** | Summarizes old messages into a `[MEMORY]` system message to prevent context window overflow |
| **Memory Injection** | When switching providers, injects conversation history so the new model has context |
| **Tool Call Support** | Passes through `tools`, `tool_choice`, and `tool_calls` for function-calling workflows |
| **Message Normalization** | Handles `content` as both `"string"` and `[{"type":"text","text":"..."}]` formats |

---

## Quick Start

### Prerequisites
- **Go 1.21+**
- **Redis** (optional — exact cache degrades gracefully if unavailable)

### Setup

```bash
# Clone and enter the project
cd llm-router

# Install dependencies
go mod tidy

# Edit config with your API keys
nano configs/config.yaml

# Run
go run cmd/server/main.go
```

The server starts on `:8080` by default. Override with the `LLM_ROUTER_ADDR` environment variable.

### Test It

```bash
# Non-streaming
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Streaming (SSE)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# Health check
curl http://localhost:8080/health
```

---

## Configuration

All settings live in `configs/config.yaml`. Override the path with the `LLM_ROUTER_CONFIG` environment variable.

```yaml
# Redis address for exact cache (optional)
redis_addr: "localhost:6379"

# Maximum messages before context compression kicks in
max_messages: 12

# Semantic cache settings
semantic_cache_size: 100     # Max LRU entries
semantic_threshold: 0.92     # Cosine similarity threshold (0.0 - 1.0)

# Gemini API key for generating embeddings (optional — semantic cache disabled without it)
gemini_api_key: "YOUR_GEMINI_API_KEY"

# Provider keys — tried in order, first success wins
keys:
  - name: cerebras-1
    provider: cerebras
    key: YOUR_CEREBRAS_KEY
    base_url: https://api.cerebras.ai/v1
    model: qwen-3-235b-a22b-instruct-2507

  - name: groq-1
    provider: groq
    key: YOUR_GROQ_KEY
    base_url: https://api.groq.com/openai/v1
    model: llama-3.1-8b-instant

  - name: mistral-1
    provider: mistral
    key: YOUR_MISTRAL_KEY
    base_url: https://api.mistral.ai/v1
    model: mistral-small
```

### Config Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `redis_addr` | string | `localhost:6379` | Redis server address. If unreachable, exact cache is silently disabled |
| `max_messages` | int | `12` | Message count threshold before context compression activates |
| `semantic_cache_size` | int | `100` | Maximum entries in the in-memory semantic LRU cache |
| `semantic_threshold` | float | `0.92` | Minimum cosine similarity to consider a semantic cache hit |
| `gemini_api_key` | string | — | API key for Gemini Embedding API. Semantic cache disabled if empty |
| `keys` | list | — | Ordered list of provider keys (see below) |

### Key Entry Fields

| Field | Description |
|---|---|
| `name` | Human-readable identifier (e.g. `cerebras-1`) |
| `provider` | Provider name (e.g. `cerebras`, `groq`, `mistral`) |
| `key` | API key / bearer token |
| `base_url` | Provider's OpenAI-compatible base URL (without `/chat/completions`) |
| `model` | Model name to use with this provider |

---

## Request Pipeline

Every incoming request flows through this pipeline in order:

```
1. Receive POST /v1/chat/completions
            │
2. In-flight dedup ──→ duplicate? wait & share result
            │
3. Exact cache (Redis) ──→ hit? return immediately
            │
4. Generate embedding (Gemini)
            │
5. Semantic cache ──→ similar enough? return cached
            │
6. Context compression (if messages > max_messages)
            │
7. Normalize messages (flatten content arrays)
            │
8. Try providers in order
            │   ├── Success → store in caches, reset key failures
            │   └── Failure (400/401/403/404/429/500) → cooldown, try next
            │
9. Return response (JSON or SSE stream)
```

---

## Project Structure

```
llm-router/
├── cmd/server/
│   └── main.go                 # Entrypoint — wires all components, starts server
├── configs/
│   └── config.yaml             # Provider keys, Redis, cache settings
├── internal/
│   ├── api/
│   │   └── server.go           # Gin HTTP server, streaming + non-streaming handlers
│   ├── cache/
│   │   ├── exact.go            # Redis exact cache (SHA-256 hash, 5min TTL)
│   │   ├── inflight.go         # In-flight request deduplication
│   │   └── semantic.go         # In-memory LRU cache with cosine similarity
│   ├── config/
│   │   └── config.go           # YAML config loader with defaults
│   ├── context/
│   │   └── engine.go           # Context compression + memory injection
│   ├── embedding/
│   │   └── gemini.go           # Gemini Embedding API client
│   ├── pool/
│   │   └── keypool.go          # Key pool with exponential backoff cooldown
│   ├── router/
│   │   └── router.go           # Core routing orchestrator, failover logic
│   └── types/
│       └── types.go            # Shared OpenAI-compatible structs
├── go.mod
└── go.sum
```

---

## How It Works

### Multi-Provider Failover

Providers are tried in the order defined in `config.yaml`. On failure (HTTP 400, 401, 403, 404, 429, or 500), the router skips to the next provider. Only returns an error if **all** providers fail.

### Key Cooldown

When a key fails, it enters exponential backoff cooldown:
- 1st failure → 2 seconds
- 2nd failure → 4 seconds
- 3rd failure → 8 seconds
- Capped at 60 seconds max

On success, failures reset to zero.

### Exact Cache (Redis)

The full request is hashed (SHA-256) and looked up in Redis. Identical requests within the 5-minute TTL window get instant responses without calling any provider. Redis being down doesn't crash the router — the cache is simply skipped.

### Semantic Cache

If a Gemini API key is configured, the router embeds the last user message and compares it against cached entries using cosine similarity. If similarity ≥ 0.92 (configurable), the cached response is returned. This catches rephrased questions that are semantically identical.

The cache uses LRU eviction and has a configurable max size.

### Context Compression

When conversations exceed `max_messages` (default 12), older messages are summarized by calling a cheap/fast provider. The summary is injected as a system message:

```
[MEMORY] Previous conversation summary: {summary}
```

This prevents context window overflow when history accumulates.

### Message Normalization

Some clients (especially during tool calls) send `content` as an array:
```json
{"content": [{"type": "text", "text": "Hello"}]}
```

The router normalizes this to a plain string before forwarding to providers:
```json
{"content": "Hello"}
```

### In-Flight Deduplication

If identical requests arrive simultaneously (e.g. a client double-sends), only one provider call is made. All concurrent waiters receive the same response.

---

## OpenClaw Integration

Point OpenClaw at the router as an OpenAI-compatible backend:

```yaml
# In your OpenClaw config
api: "openai-completions"
base_url: "http://localhost:8080/v1"
```

**Important notes for OpenClaw:**
- The response always includes the `created` (unix timestamp) field — OpenClaw silently drops responses without it
- Streaming is fully supported and works out of the box
- Tool calls are passed through when the provider supports them
- In `channels.telegram`, do **not** name the account `"default"` — use `"main"` instead (known OpenClaw bug #23123)

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_ROUTER_CONFIG` | `configs/config.yaml` | Path to the configuration file |
| `LLM_ROUTER_ADDR` | `:8080` | Address to listen on |

---

## Dependencies

| Package | Purpose |
|---|---|
| `github.com/gin-gonic/gin` | HTTP framework |
| `gopkg.in/yaml.v3` | YAML config parsing |
| `github.com/redis/go-redis/v9` | Redis client for exact cache |
| `golang.org/x/sync` | `singleflight` for in-flight dedup |

---

## License

Private / Internal Use
