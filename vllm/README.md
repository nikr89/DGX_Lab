# vLLM start scripts

Every model in this lab is started with one of the scripts in this folder. They all run the official `vllm/vllm-openai` Docker image on a DGX Spark (ASUS GX10, GB10, 128 GB of shared CPU/GPU memory). Only one large model fits at a time, so most of these scripts assume they have the machine to themselves.

Want to know how the models compare? The field report is here: [../results/2026-11_model-comparison.md](../results/2026-11_model-comparison.md).

## production/ — what is running right now

I didn't trust my old scripts to match what was actually running. So I rebuilt these three from `docker inspect` of the live containers on November 1, 2026. What you see here is exactly what serves requests.

| Script | Model | Port | What's special |
|---|---|---|---|
| `qwen3.8-27b_v7.sh` | `unsloth/Qwen3.8-27B-NVFP4` | 8000 | The everyday model. MTP 3, KV cache fixed at 48 GiB, 256k context, fastsafetensors, prefix caching |
| `nemotron-3.5-super-vl.sh` | `unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` | 8000 (local only) | A deliberately careful profile: no CUDA graphs, FlashInfer FP4 MoE off, 64k context, MTP 1 |
| `qwen3-embed.sh` | `Qwen/Qwen3-Embedding-4B` | 8003 | Small embedding server that runs alongside the main model (10 % of memory) |

## scripts/ — the history

The launchers I wrote along the way. Some are templates, some have been superseded, and all of them taught me something.

| Script | Model | Where it stands |
|---|---|---|
| `start-qwen3.8-27b.sh` | Qwen3.8-27B | Flexible launcher driven by environment variables. Older defaults: MTP 1, 40 GiB KV |
| `vllm_relaunch.sh` | Qwen3.8-27B | Quick relaunch helper, MTP 1, 40 GiB KV |
| `recreate_containers.sh` | Qwen3.6-35B-A3B, Qwen3.8-27B, embeddings | Moves containers to a new vLLM image and keeps the old ones as `<name>-old`, in case I need to roll back |
| `serve_ornith15_122b_nvfp4.sh` | `ornith-ai/Ornith-1.5-122B-NVFP4` | My first Ornith attempt. Needs a few fixes for vLLM v0.31.0 (noted in the file); the settings that actually worked are in `benchmark/run_matrix3.sh` |
| `start_embedding_vllm.sh` | Qwen3-Embedding-4B | Earlier version of the embedding server |
| `restart-vllm-vision.sh` | Qwen2-VL-2B-Instruct | A tiny vision model from the early days on port 8001, since retired |

A note on the Ornith script: the original had a real token as its default value. In this copy it is replaced by the placeholder `hf_token`. Set `HF_TOKEN` to your own value before running it.

## llama-swap/

`config.yaml` lets [llama-swap](https://github.com/mostlygeek/llama-swap) start models on demand and swap between them: Gemma 4 26B, Qwen3.5 122B, Qwen3.6 35B and a few others, on `vllm/vllm-openai:nightly`.

## benchmark/ — the test runs

These scripts drove the model comparison. Each one stops the production model, works through a list of configurations, and starts the production model again at the end.

| Script | What it did |
|---|---|
| `run_matrix.sh`, `run_matrix2.sh` | Round 1 (October 28): Ornith, Nemotron and Qwen, each with MTP 0, 1 and 3 |
| `run_matrix3.sh` | Round 2 (November 1): Nemotron MTP 2/3 with CUDA graphs, FlashInfer and vision; Ornith MTP 1/2; Qwen MTP 1/2/3 |
| `video_phase.sh` | First attempts to start each model with video input |
| `oom_watchdog.sh` | Stops the benchmark container when free memory drops below 2 GiB, before the kernel starts killing other processes |

The test scripts they call (`model_bench.py`, `skill_bench.py`, `hermes_skill_bench.py`, `video_real_bench.py`, `long_bench.py`) aren't in this folder.

## Lessons learned on the GB10

- **`--gpu-memory-utilization` is not a hard limit.** Nemotron with vision and MTP 3 used up all 128 GB while preparing for the largest possible video input, even at 0.70, and took system services down with it. When you try something new, run a watchdog like `oom_watchdog.sh`.
- **Use `--load-format safetensors` for Ornith 122B with MTP.** `fastsafetensors` runs out of memory there.
- **Fix the KV cache size.** `--kv-cache-memory-bytes` gives much more predictable memory use than `--gpu-memory-utilization` alone. That's how the Qwen container runs.
- **Turn thinking off the right way.** Use `--default-chat-template-kwargs '{"enable_thinking": false}'` on the server, or `chat_template_kwargs` in the request. vLLM silently ignores a top-level `enable_thinking` field.
