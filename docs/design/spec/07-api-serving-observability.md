# 07 — API, serving surface and observability

Requirements: R-API-1..3. Crates: `api`, `obs`, `runtime` (sampler/grammar).

## 1. OpenAI compatibility contract (hard requirement)
Astral's public API **is** the OpenAI API: an existing app using the official `openai` Python/JS SDKs, LangChain, LiteLLM or similar must work by changing only `base_url` (and a dummy key). Compatibility is judged against OpenAI's published API reference and the official SDKs' behaviour, **not** against vLLM's or SGLang's extensions.
- **P1 must-haves:**
  - `POST /v1/chat/completions` and `POST /v1/completions`, non-streaming and SSE streaming (`data: {...}` chunks, final `data: [DONE]`, `stream_options.include_usage`).
  - Request fields: `model`, `messages` (system/user/assistant/tool roles), `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `stop`, `n`, `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `logprobs` / `top_logprobs`, `user`. Unsupported fields return an OpenAI-format 400 naming the field, never silent acceptance `[I]`.
  - Response shape: `id`, `object`, `created`, `model`, `choices[].{index,message|delta,finish_reason}`, `usage.{prompt_tokens,completion_tokens,total_tokens}`; correct `finish_reason` (`stop`, `length`, `tool_calls`).
  - `GET /v1/models`; error body `{"error":{"message","type","param","code"}}` with matching HTTP codes (400/401/404/429/500); `Authorization: Bearer` accepted.
  - Client disconnect cancels the request and frees its KV immediately.
- **P2 (R-API-2):** `tools` / `tool_choice` / `tool_calls` (including streaming deltas), `response_format` (`json_object`, `json_schema` strict), Responses API (`/v1/responses`), vision content parts (`image_url`), reasoning content field, `/v1/embeddings` later.
- **Extensions** (e.g. `reasoning_effort`, `top_k`, `min_p`) are accepted only as extra fields that OpenAI clients can ignore, and are documented as non-standard.
- **Conformance gate:** a suite that runs the official Python and JS SDKs (sync, async, streaming, error paths) against Astral and against recorded OpenAI response fixtures; schema-validated against OpenAI's OpenAPI spec `[I]`; run in CI on the mock backend and on the A100. Differences from OpenAI behaviour are listed in a public compatibility matrix.
- Other endpoints: `GET /health`, `GET /metrics`.

- Responses API is also required for gpt-oss-class tool use with Harmony parsing [R].
- Later: embeddings, batch API, LoRA multi-adapter — non-goals for v1.
Server: axum/hyper (HTTP). SSE throughput benchmarks for axum/hyper were **not found** [R]; measure in P1 before claiming a frontend advantage.

## 2. Request pipeline
HTTP → validate → apply chat template (Jinja-compatible renderer; Inkling ships `chat_template.jinja`, content not yet retrieved [M]) → tokenise (HF `tokenizers`, Rust; Inkling ships `tokenizer.json` + a `tiktoken/` dir [M]) → submit to engine channel → stream tokens back. Tokenisation/templating on the blocking pool; the engine thread never parses text (spec 02).

## 3. Parsers (per-model, repeatable cost [R])
Tool-call and reasoning parsers are a recurring correctness risk versus vLLM's behaviour. Spec: parsers are data-driven per model (`parsers/<model>.toml` + code for non-trivial formats), with a **conformance suite** of recorded chat-template outputs. **For Inkling 2 Mini the format is now known [T] (spec 06 §10):** the repo's `tokenizer_config.json` carries a `response_template` that defines `reasoning_content` (`<|content_thinking|>`…`<|end_message|>`), `content` (`<|content_text|>` / `<|content_tool_error|>` / `<|content_invoke_tool_text|>`…`<|end_message|>`) and `tool_calls` (`<|content_invoke_tool_json|>`…`<|end_message|>`, JSON `{"name","args"}`). The Inkling parser is generated from that schema and tested against llama.cpp's `common/parsers/inkling.cpp` behaviour (not yet read). Template specifics the server must honour: `reasoning_effort` request field (named levels or numeric 0–0.99, default 0.9), `developer` role rejected, audio rejected, tool arguments must be parsed objects, and **no generation prompt is emitted by the template** (my reading; engine supplies the assistant header — verify). Card caveat: model is less reliable at detailed tool instructions and may take destructive actions without confirmation [M] — document in the server's safety notes.

## 4. Structured output
llguidance linked in-process (Rust core; ~50 µs/token mask for a 128k vocab per its README; integrated in vLLM, SGLang, mistral.rs [R]). Mask computed on the output thread and applied in sampling (spec 05 §6). Third-party xgrammar-vs-llguidance comparisons are weak [R] → choose llguidance for language fit, re-benchmark with Inkling's 201k vocab.

## 5. Sampling defaults
Per model package: for Inkling, temperature 1.0, top-p 1.0, top-k off, min-p 0 [M/G]; request values override. Server-side `max_tokens` and per-turn limits enforced because the card warns of repetitive reasoning loops [M].

## 6. Metrics (R-API-3)
- **Prometheus:** emit `vllm:`-prefixed names matching vLLM's documented metrics (queue/running requests, TTFT/ITL/E2E histograms, token counters, KV usage) so existing dashboards/autoscalers work; add `astral_*` for Astral-specific signals (pool occupancy by kind, speculative acceptance, graph-bucket hits, plan/launch CPU time).
- **OpenTelemetry:** `gen_ai.server.*` GenAI metrics. These are **Development stability, not stable** [R]; isolate behind a feature/adapter so spec churn doesn't break the API.
- **TTFT definition must be documented:** vLLM measures from frontend arrival including tokenisation; the OTel definition is queue + prefill [R]. Astral exports both (`ttft_frontend`, `ttft_engine`).
- No packed-label anti-pattern for per-request/adapter labels [R].
- **Tracing:** `tracing` spans per request and per step; NVTX ranges per layer/op in profiling builds; tokio-console for the edge runtime; GPU telemetry via NVML (and DCGM where present). Nsight/CUPTI integration guidance was not researched [R] — treat as P2 tooling task.

## 7. Benchmark compatibility
Server must work with common load generators (vLLM bench, genai-perf/AIPerf; guidellm not verified [R]) and expose the usage fields they need.

## 8. Security and ops (OSS baseline)
Optional bearer-token auth, request size limits, rate limiting per key (basic). RBAC/SSO/audit logs are part of the Enterprise edition. Binds to localhost by default.

## 9. Open items
Generation-prompt behaviour for Inkling (verify against llama.cpp's parser and vLLM/SGLang recipes, not read); Responses API field mapping; vision request limits (40–4096 px per side [M]); the card gives no vLLM/SGLang parser flag names.
