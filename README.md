# openmoss-ggml

A standalone C++/GGML port of [MOSS-TTS-Delay](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) — a
Qwen3-8B language backbone + 32 RVQ audio codebooks + a 1.6 B pure-transformer audio codec, all
runnable from one self-contained binary. The Qwen3 backbone is hosted by **libllama** (so you get
all of llama.cpp's quantizations and CUDA / CPU / Vulkan backends for free); the embedding stack,
LM heads, and the codec encoder/decoder are GGML graphs we build ourselves.

Two entry points:

- **`moss-tts-cli`** — one-shot synthesis. Loads the model, synthesizes one utterance, writes a
  WAV, exits.
- **`moss-tts-server`** — keeps the model resident and exposes an HTTP API for repeated
  generations (TTS and voice cloning).

On a single 16 GB GPU (RTX 5060 Ti) the Q8\_0 backbone produces ~10 s of speech in ~4 s of
wall-clock; the codec runs at ~50× real-time. See `docs/STATUS.md` for a detailed feature matrix,
pipeline diagram, and benchmark numbers.

## Quick start

```bash
# 1. Build (assumes a built llama.cpp tree)
cmake -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp -DGGML_CUDA=ON
cmake --build build -j

# 2. Convert weights once
python scripts/convert_hf_to_gguf.py \
    --moss-tts OpenMOSS-Team/MOSS-TTS \
    --codec    OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --output   weights/moss-tts.gguf

# 3. (Optional) quantize the backbone — keep the embedding table as f16
/path/to/llama.cpp/build/bin/llama-quantize \
    --token-embedding-type f16 \
    weights/moss-tts.gguf weights/moss-tts-q8_0.gguf Q8_0

# 4. Synthesize
./build/moss-tts-cli \
    --model weights/moss-tts-q8_0.gguf \
    --text  "Hello, world!" \
    --output out.wav
```

## Build

Prerequisites:

- A built llama.cpp tree (`libllama.so`, `libggml*.so`, headers under `ggml/include/` and
  `include/`). Build it with the same backend you want here (`-DGGML_CUDA=ON` for NVIDIA, etc.).
- CMake ≥ 3.18, a C++17 compiler.
- `nlohmann/json.hpp` (only the header is needed; on Debian/Ubuntu: `apt install
  nlohmann-json3-dev`).
- The HTTP server is built against a vendored copy of `cpp-httplib` v0.18.7
  (`third_party/cpp-httplib/httplib.h`); no extra system dep.

```bash
cmake -B build \
    -DLLAMA_CPP_DIR=/devel/tools/llama.cpp \
    -DGGML_CUDA=ON
cmake --build build -j
```

Outputs:

- `build/moss-tts-cli`
- `build/moss-tts-server`
- `build/moss-tts-info`, `build/moss-tts-compute-test`, `build/moss-codec-roundtrip` (diagnostics)

## Convert weights

The converter produces **two** GGUF files alongside each other:

- `moss-tts.gguf` — the Qwen3-8B backbone, pure Qwen3 layout that libllama can load directly.
- `moss-tts.extras.gguf` — the sidecar: 32 audio embedding tables, 33 LM heads, codec encoder /
  RVQ codebooks / codec decoder, and the `moss.*` KV namespace (frame rate, special token ids,
  …). Loaded by us, not libllama.

```bash
pip install safetensors numpy huggingface_hub gguf

python scripts/convert_hf_to_gguf.py \
    --moss-tts OpenMOSS-Team/MOSS-TTS \
    --codec    OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --output   weights/moss-tts.gguf \
    --llama-cpp-dir /path/to/llama.cpp \
    --backbone-dtype f16
```

Useful flags: `--cache-dir` for the HF cache, `--scratch-dir`/`--keep-scratch` for inspecting the
intermediate extracted backbone, `--skip-extract` to reuse a previously-extracted backbone if you
already ran a conversion.

### Quantization

Quantize the backbone the same way you quantize any GGUF, but pass
`--token-embedding-type f16`: the embedding table is indexed via `ggml_get_rows`, which doesn't
support quantized `src0` on CUDA, so it must stay f16.

```bash
llama-quantize --token-embedding-type f16 \
    weights/moss-tts.gguf weights/moss-tts-q8_0.gguf Q8_0   # ~8.7 GB, validated end-to-end
llama-quantize --token-embedding-type f16 \
    weights/moss-tts.gguf weights/moss-tts-q4km.gguf Q4_K_M # ~4.7 GB
```

The sidecar is **not** quantized — keep `moss-tts.extras.gguf` next to whichever backbone you
chose. The CLI/server find the sidecar by replacing the backbone's `.gguf` suffix with
`.extras.gguf`.

## CLI usage

```bash
moss-tts-cli --model <gguf> --text "<text>" [options]
```

Common options:

| flag                  | meaning                                                            |
|-----------------------|--------------------------------------------------------------------|
| `--text "…"`          | utterance to synthesize (required)                                 |
| `--output PATH`       | output WAV (default `out.wav`, 16-bit mono @ 24 kHz)               |
| `--reference PATH`    | reference WAV for voice cloning — speaker is encoded and prepended |
| `--instruction "…"`   | voice / style description (voice-generation mode, no reference)    |
| `--language CODE`     | language hint (`en` / `zh` / …)                                     |
| `--tokens N`          | approximate audio length in tokens (1 s ≈ 12.5 tokens)             |
| `--max-new-tokens N`  | generation cap (default 4096)                                      |
| `--main-gpu N`        | pin model to GPU index N (default: auto-pick GPU with most free VRAM) |
| `--n-gpu-layers N`    | backbone GPU offload (default: all)                                |
| `--no-flash-attn`     | disable flash attention                                            |
| `--skip-codec`        | don't load codec tensors; emits codes only (debug, saves ~3.4 GB)  |

### Plain TTS

```bash
./build/moss-tts-cli \
    --model weights/moss-tts-q8_0.gguf \
    --text  "The quick brown fox jumps over the lazy dog." \
    --max-new-tokens 600 \
    --output out.wav
```

### Voice cloning

Pass a reference WAV with `--reference`. The codec encoder converts it to 32-codebook codes which
get spliced into the user-side audio block of the chat prompt, and the model continues in that
voice.

```bash
./build/moss-tts-cli \
    --model     weights/moss-tts-q8_0.gguf \
    --text      "A different sentence in the cloned voice." \
    --reference samples/alice.wav \
    --max-new-tokens 600 \
    --output    cloned.wav
```

Any sample rate works — the WAV is linearly resampled to 24 kHz internally. A few seconds of
clean speech is usually enough.

## Server usage

```bash
./build/moss-tts-server --model weights/moss-tts-q8_0.gguf --host 0.0.0.0 --port 8080
```

`moss-tts-server` shares the same `--main-gpu`, `--n-gpu-layers`, `--no-flash-attn`, and
`--skip-codec` flags as the CLI. Generations are serialized through a single mutex (libllama
state is not reentrant); concurrent requests queue cleanly.

### Endpoints

- `GET /health` — readiness probe, returns `ok`.
- `GET /info` — JSON with model dims, codec status, and the request counter.
- `POST /tts` — JSON in, `audio/wav` out (or `text/plain` with the error message on failure).

#### `POST /tts` request body

| field                 | type     | required | notes                                                    |
|-----------------------|----------|----------|----------------------------------------------------------|
| `text`                | string   | yes      | utterance to synthesize                                  |
| `instruction`         | string   | no       | voice / style description (voice-generation mode)        |
| `language`            | string   | no       | language hint                                            |
| `tokens`              | int      | no       | approximate length (1 s ≈ 12.5 tokens)                   |
| `max_new_tokens`      | int      | no       | default 4096                                             |
| `reference_wav_b64`   | string   | no       | base64-encoded WAV bytes for voice cloning               |
| `sampling`            | object   | no       | `text_temperature`, `text_top_p`, `text_top_k`, `audio_temperature`, `audio_top_p`, `audio_top_k`, `audio_repetition_penalty`, `seed` |

Response headers carry timing info: `X-MOSS-Audio-Frames`, `X-MOSS-Generate-Seconds`,
`X-MOSS-Decode-Seconds`.

The payload limit is 64 MB, which fits a ~60 s reference WAV after base64 encoding.

### Examples

```bash
# Plain TTS
curl -s -X POST http://localhost:8080/tts \
    -H 'Content-Type: application/json' \
    -d '{"text":"Hello from the persistent server.","max_new_tokens":300}' \
    -o out.wav

# Voice cloning
python3 -c "import base64,json,sys; \
    print(json.dumps({'text': sys.argv[1], 'max_new_tokens': 600, \
        'reference_wav_b64': base64.b64encode(open(sys.argv[2],'rb').read()).decode()}))" \
    "A different sentence in the cloned voice." samples/alice.wav > req.json
curl -s -X POST http://localhost:8080/tts \
    -H 'Content-Type: application/json' --data @req.json -o cloned.wav
```

## GPU placement

Both binaries pin the full model — libllama backbone **and** the aux GGML backend (embeddings,
LM heads, codec) — to a single device via `LLAMA_SPLIT_MODE_NONE`. With no flag, they auto-pick
the GPU that has the most free VRAM at load time:

```
Model::load: pinning to GPU 1 (CUDA1, 15703/15850 MiB free)
```

Override with `--main-gpu N` (index into the ggml GPU device list — same convention as
libllama's own `main_gpu`). Splitting the backbone across GPUs is *not* currently supported,
because the aux backend would still need to land on one device and would then race the backbone
fragment on that device for VRAM. At Q8\_0 (~9 GB backbone + ~5 GB aux ≈ 14 GB) anything ≥ 16 GB
fits.

## Architecture

```
                  ┌────────────────────────────────────────────┐
   text  ────►    │ BPE tokenizer (Qwen3, 155 648 vocab)       │
   ref.wav ──►    │ Codec encoder (audio → 32×T_a codes)       │ — voice cloning
                  └────────────────────────────────────────────┘
                            │
                  ┌─────────▼──────────────────────────────────┐
                  │ Embedding stack:                           │
                  │   text emb  +  Σᵢ audio_emb_i(code_i)      │  (33 tables, 4096 dim)
                  └─────────┬──────────────────────────────────┘
                            │ inputs_embeds (S, 4096) — fed via batch.embd
                  ┌─────────▼──────────────────────────────────┐
                  │ Qwen3-8B backbone (libllama)               │
                  │   36 layers, hidden=4096, GQA 32/8         │
                  └─────────┬──────────────────────────────────┘
                            │ hidden_state (4096)
                  ┌─────────▼──────────────────────────────────┐
                  │ 33 LM heads (1 text + 32 audio×1025)       │  (GGML matmul, GPU)
                  └─────────┬──────────────────────────────────┘
                            │ logits
                  ┌─────────▼──────────────────────────────────┐
                  │ Delay-pattern state machine + sampling     │  (CPU, deterministic)
                  └─────────┬──────────────────────────────────┘
                            │ codes (32 × T_a)
                  ┌─────────▼──────────────────────────────────┐
                  │ Codec decoder (4-stage pure transformer)   │  (GGML)
                  │   12.5 Hz → 24 000 Hz (×1920 upsample)     │
                  └─────────┬──────────────────────────────────┘
                            ▼
                       waveform (f32, 24 kHz)  →  16-bit WAV
```

For the full per-stage tensor shapes, list of bugs hit during bring-up, and benchmark numbers,
see [`docs/STATUS.md`](docs/STATUS.md).

## Repo layout

| path                                  | role                                                       |
|---------------------------------------|------------------------------------------------------------|
| `scripts/convert_hf_to_gguf.py`       | HF → backbone GGUF + sidecar GGUF                          |
| `src/model.cpp`, `src/aux_internal.h` | Two-file loader, aux backend, embed / LM-head graphs       |
| `src/codec.cpp`                       | RLFQ + 4-stage transformer encoder/decoder graphs          |
| `src/tokenizer.cpp`                   | libllama BPE wrapper                                       |
| `src/delay.cpp`                       | DelayState + sampling (top-k/p, repetition penalty)        |
| `src/pipeline.cpp`                    | Prompt builder + autoregressive loop + codec dispatch      |
| `src/wav.cpp`                         | RIFF/WAVE I/O (no libsndfile dep)                          |
| `src/cli/moss_tts_cli.cpp`            | CLI entry point                                            |
| `src/server/moss_tts_server.cpp`      | HTTP server                                                |
| `tests/`                              | Diagnostics: model info, compute-graph smoke, codec round-trip |
| `third_party/cpp-httplib/httplib.h`   | Vendored single-header HTTP library                        |

## License

Apache-2.0, matching upstream MOSS-TTS.
