# Benchmark methodology

All runs are logged in [results.csv](results.csv), one row per run. Runs worth explaining get their own file based on [TEMPLATE.md](TEMPLATE.md).

## Rules

- Same prompt set for all engines (`scripts/prompts.json`)
- One warm-up run discarded, median of 3 runs reported
- Decode speed measured on 256 generated tokens
- Engine version and container tag recorded for every run
- Failed runs are logged too, with the reason in `notes`

## Columns

| Column | Meaning |
|---|---|
| `date` | ISO date of the run |
| `chapter` | Chapter of the learning plan |
| `model` | Model name as used by the engine |
| `params_b` | Parameters in billions |
| `engine`, `engine_version` | e.g. `vllm`, `0.x` |
| `container` | NGC image and tag, or `host` |
| `precision` | BF16, FP8, NVFP4, q4_K_M, ... |
| `context_len` | Input tokens |
| `concurrency` | Parallel requests |
| `ttft_ms` | Time to first token |
| `prefill_tok_s` | Prompt processing speed (compute bound) |
| `decode_tok_s` | Generation speed per request (memory bound) |
| `peak_mem_gb` | Peak memory, same measurement method for every run |
| `notes` | Anything that matters for interpretation |

## Rule of thumb

Decode tokens/s per request ≈ memory bandwidth ÷ model size in bytes. With ~273 GB/s, an 8B model in BF16 (~16 GB) tops out around 17 tok/s, the same model in 4-bit (~4.5 GB) around 60 tok/s. Measured values below this ceiling show overhead, values above it point to speculative decoding or a measurement error.
