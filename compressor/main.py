"""
Claw-compactor sidecar — Flask HTTP server for deterministic token compression.
Runs on port 8081 with a single POST /compress endpoint.
"""

from flask import Flask, request, jsonify
import logging
import sys

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="[Compressor] %(message)s")

# Try to import claw-compactor; if unavailable, compression is a no-op
try:
    from claw_compactor.fusion.engine import FusionEngine
    engine = FusionEngine()
    COMPACTOR_AVAILABLE = True
    logging.info("claw-compactor loaded successfully")
except Exception as e:
    import traceback
    engine = None
    COMPACTOR_AVAILABLE = False
    logging.warning(f"claw-compactor could not be imported. Exception: {e}")
    logging.warning(traceback.format_exc())


@app.route("/compress", methods=["POST"])
def compress():
    """
    Accepts: {"messages": [{"role": "...", "content": "..."}, ...]}
    Returns: {"messages": [...], "saved_tokens": N}
    """
    try:
        data = request.get_json(force=True)
        messages = data.get("messages", [])

        if not messages:
            return jsonify({"messages": [], "saved_tokens": 0})

        if not COMPACTOR_AVAILABLE:
            return jsonify({"messages": messages, "saved_tokens": 0})

        # Estimate input tokens (rough: chars / 4)
        input_chars = sum(len(m.get("content", "")) for m in messages)
        input_tokens = input_chars // 4

        # Compress via FusionEngine
        compressed = engine.compress_messages(messages)

        # Estimate output tokens
        output_chars = sum(len(m.get("content", "")) for m in compressed)
        output_tokens = output_chars // 4
        saved = max(0, input_tokens - output_tokens)

        logging.info(f"Compressed {len(messages)} messages: ~{input_tokens} → ~{output_tokens} tokens (saved ~{saved})")

        return jsonify({"messages": compressed, "saved_tokens": saved})

    except Exception as e:
        logging.error(f"Compression failed: {e}")
        # On any error, return original messages unchanged
        messages = request.get_json(force=True).get("messages", [])
        return jsonify({"messages": messages, "saved_tokens": 0})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "compactor_available": COMPACTOR_AVAILABLE
    })


if __name__ == "__main__":
    logging.info("Starting claw-compactor sidecar on :8081")
    app.run(host="0.0.0.0", port=8081)
