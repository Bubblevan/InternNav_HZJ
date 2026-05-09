# DualVLN Runtime

This package holds transport and sidecar runtime concerns that should stay
independent from both Habitat business logic and the vLLM fork itself.

Current modules:

- `protocol.py`: tensor/message/image codecs, including shared-memory image transport
- `http.py`: HTTP client and Flask app factory for the single-vLLM sidecar

The goal is that these pieces can be reused by other DualVLN entrypoints
without pulling in evaluator-specific code.
