# DGX_Lab

Experiments and results from my home lab with a DGX Spark class system (ASUS Ascent GX10, NVIDIA GB10 Grace Blackwell).

Hands-on learning log covering LLM inference, quantization, fine-tuning, agentic AI, guardrails and enterprise deployment patterns on the NVIDIA software stack. Every session leaves a trace here: notes, notebooks, scripts and reproducible benchmark numbers, including the things that did not work.

## Hardware

| Component | Spec |
|---|---|
| System | ASUS Ascent GX10 |
| Superchip | NVIDIA GB10 Grace Blackwell |
| CPU | 20-core Arm (aarch64) |
| Memory | 128 GB unified LPDDR5x, ~273 GB/s |
| AI compute | up to 1 PFLOP FP4 (sparse) |

## Environment

| Item | Version |
|---|---|
| OS | DGX OS (fill in) |
| GPU driver / CUDA | (fill in from `nvidia-smi`) |
| Containers | NGC, tag recorded per run in `benchmarks/results.csv` |

## Progress

| Module | Status | Highlights |
|---|---|---|
| 1 Setup | in progress | NGC containers on aarch64 |
| 2 GPU & CUDA basics | planned | |
| 3 LLM inference | planned | Ollama vs vLLM vs TensorRT-LLM vs NIM, NVFP4 |
| 4 Fine-tuning | planned | LoRA with Unsloth and NeMo |
| 5 RAG, agents, guardrails | planned | NVIDIA Blueprints, NeMo Agent Toolkit, NeMo Guardrails |
| 6 Multimodal & edge | planned | VLMs, video search, speech |
| 7 Infrastructure | planned | Kubernetes GPU Operator, TCO |
| 8 Capstone | planned | Sovereign enterprise assistant |

## Repository layout

```
benchmarks/   methodology, raw results (results.csv), run template
notebooks/    Jupyter notebooks per chapter
notes/        learning notes per chapter
scripts/      reusable scripts (benchmarks, setup)
```

## Benchmarks

See [benchmarks/](benchmarks/) for methodology and raw data. All numbers are measured on my own system and recorded with engine version, container tag, precision and concurrency so they can be reproduced.

## License

Code and docs: [Apache 2.0](LICENSE). Models used in experiments keep their own licenses.
