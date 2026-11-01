# Three big models, one small box: a DGX Spark field report

*October 28 – November 1, 2026*

## Why I did this

My DGX Spark (an ASUS GX10) does real work for me every day. It runs Hermes, a local agent, as well as Cline, which I use for development and knowledge work. Until now, everything ran on one model, Qwen3.8-27B. Two much larger models had appeared that also fit in 128 GB:
- Ornith 1.5 122B
- Nemotron 3.5 Super VL 120B

So the question was simple: is one of the big models better for *my* tasks, and how should it be configured?

Generic benchmarks don't answer that. So I built tests from the work Hermes actually does, and ran all three models through them over a few days.

## The setup

The machine is a DGX Spark design: a GB10 Grace Blackwell chip with 128 GB of memory shared between CPU and GPU (about 121 GB usable). The important thing to know about this box is that it is limited by memory bandwidth, not compute. Generating a token is mostly a question of how many bytes have to be read. That one fact explains most of the results below.

Every model ran in the official vLLM Docker image (v0.31.0), one at a time, because two big models don't fit side by side. Every model used:
- NVFP4 weights
- FP8 KV cache
- chunked prefill and async scheduling
- "thinking" switched off, since Hermes doesn't need it

| | Qwen3.8-27B | Ornith 1.5 122B | Nemotron 3.5 Super VL 120B |
|---|---|---|---|
| Checkpoint | `unsloth/Qwen3.8-27B-NVFP4` | `ornith-ai/Ornith-1.5-122B-NVFP4` | `unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` |
| What it is | A classic dense model, 27B | Mixture of experts: 256 experts, 8 active, mostly linear attention | Hybrid of Mamba-2, mixture of experts and attention, 12B active |
| Max context | 262k | 262k | 1M |
| Time to start | about 5 min | about 12 min | about 14 min |

The launch scripts are in [../vllm/](../vllm/).

## How the tests work

I wanted tests where a wrong answer means something. Grading had to be automatic, so I could run everything several times without reading thousands of answers by hand.

**The basics** came first, in a first round on October 28:
- a quality test with 12 mixed tasks, run three times
- a skill-routing test where the model picks one of 20 tools
- speed with 1, 4 and 8 parallel requests
- a needle-in-a-haystack test that hides a code in a long text and asks for it back

**The real-work tests** came in a second round on November 1. They are the ones I care about most.

- **Hermes skill selection.** The model sees all 120 of Hermes' skills and has to pick the right one for requests about YouTube and stocks, or pick none when no skill fits.
- **Hermes skill execution.** The model gets the real skill instructions and has to do the job:
  - Write the right shell command for the YouTube pipeline.
  - Write a wiki excerpt from a real trading video transcript.
  - Turn a transcript into investment claims without inventing numbers.
  - Size a stock position.
  - Write Python code against TradingView.
- **Real videos.** The model watches 6 real clips and answers numbered questions about what's on screen. It also describes 3 full-resolution frames with the same prompt the pipeline uses, and writes a summary of a real video that is checked for 12 facts.

Answers are graded with keyword groups that understand number formats ("1,500", "1.500" and "1.5k" all count). Wherever possible, answers are checked against the real reference code. For example, a position size has to match what the risk formula gives. There is also a check that flags numbers in generated claims that don't appear in the source. The quality test is coarse: one wrong answer costs 7–10 points, so I treat differences below about 5 points as noise.

## What happened

### Round 1: does speculative decoding help?

All three models ship with an MTP head, which lets vLLM guess several tokens ahead and verify them in one step. More guessed tokens should mean more speed, as long as the guesses are good. I tried 0, 1 and 3 guessed tokens.

| Configuration | Quality | First token | Tokens/s, 1 request | Tokens/s total, 4 / 8 requests | Guesses accepted | Skills |
|---|---|---|---|---|---|---|
| Ornith, MTP 0 | 96.0 % | 0.14 s | 28.1 | 67.7 / 97.0 | – | 18/20 |
| Ornith, MTP 1 | 91.7 % | 0.15 s | 35.2 | 72.6 / 101.3 | 67 % | 19/20 |
| Ornith, MTP 3 | 89.9 % | 0.17 s | 34.3 | 57.8 / 80.9 | 43 % | 19/20 |
| Nemotron, MTP 0 | 96.1 % | 0.93 s | 18.8 | 47.5 / 71.5 | – | 18/20 |
| Nemotron, MTP 1 | 93.5 % | 1.35 s | 28.4 | 62.0 / 88.2 | 88 % | 18/20 |
| Nemotron, MTP 3 | 95.3 % | 1.34 s | 35.7 | 61.3 / 88.8 | 72 % | 18/20 |
| Qwen, MTP 3 | 83.6 % | 0.21 s | 22.0 | 62.9 / 108.4 | 57 % | 20/20 |

Three things stood out:
- The big models are clearly smarter on the quality test.
- Qwen is the only model that never picked the wrong tool.
- Qwen, the smallest model, had the highest total throughput with 8 parallel requests.

The last point surprised me at first. It makes sense on a bandwidth-bound machine. A dense model reads the same weights for every request in a batch. A mixture-of-experts model has to fetch different experts for different requests, so a bigger batch costs it more.

Nemotron passed the needle test at 1.04 million tokens, which took about 25 minutes of reading. In this round it ran in a cautious profile (no CUDA graphs, text only), so its slow first token wasn't the final word.

### Round 2: better settings and the real-work tests

For the second round I gave each model a fairer shot:
- Nemotron got CUDA graphs, FlashInfer and vision.
- Ornith and Qwen were tried with the MTP values that round 1 suggested.

| Configuration | Quality | First token | Tokens/s, 1 request | Tokens/s total, 4 / 8 requests | Guesses accepted | Skills |
|---|---|---|---|---|---|---|
| Nemotron, MTP 2 | 95.3 % | 0.35 s | 35.2 | 65.4 / 95.7 | 83 % | 17/20 |
| Nemotron, MTP 3 | did not start (see below) | | | | | |
| Ornith, MTP 1 | 96.0 % | 0.15 s | 35.0 | 71.1 / 100.1 | 75 % | 18/20 |
| Ornith, MTP 2 | 91.7 % | 0.16 s | 35.0 | 62.6 / 89.8 | 64 % | 18/20 |
| Qwen, MTP 1 | 84.3 % | 0.18 s | 18.0 | 59.8 / 111.2 | 87 % | 20/20 |
| Qwen, MTP 2 | 84.4 % | 0.20 s | 20.8 | 64.0 / 111.9 | 78 % | 20/20 |
| Qwen, MTP 3 | 83.6 % | 0.21 s | 21.8 | 62.4 / 107.1 | 69 % | 20/20 |

The new settings cut Nemotron's first-token time from 1.35 s to 0.35 s. That was the single biggest improvement of the week.

And here are the tests that matter most to me:

| Configuration | Picks the right skill (YouTube / stocks / none) | Does the job (YouTube / stocks) | Real videos |
|---|---|---|---|
| Nemotron, MTP 2 | 100 / 75 / 100 % | 94 / 76 % | 80.6 % |
| Ornith, MTP 1 | 100 / 100 / 100 % | 94 / 90 % | 97.8 % |
| Ornith, MTP 2 | 80 / 100 / 100 % | 90 / 86 % | 96.8 % |
| Qwen, MTP 1 | 100 / 100 / 100 % | 94 / 100 % | 95.7 % |
| Qwen, MTP 2 | 100 / 100 / 100 % | 94 / 90 % | 95.7 % |
| Qwen, MTP 3 | 100 / 100 / 100 % | 94 / 100 % | 98.9 % |

This flips the picture from round 1 a little:
- **Qwen,** "the weak one" on the quality test, routes every request correctly and is excellent on video.
- **Ornith** is the most balanced of the big models. With MTP 1 it picked every skill correctly too, and scored 97.8 % on video.
- **Nemotron** is brilliant at reading huge texts, but with vision enabled it is the weakest at watching video and at picking skills.

All configurations found the needle at every length. On reading speed, Nemotron is in its own league:

| Model | Reading speed at about 256k tokens | Longest text passed |
|---|---|---|
| Nemotron | 1,474–1,704 tokens/s | 1.04M (text only), 515k (with vision) |
| Ornith | 774–831 tokens/s | 259k |
| Qwen | 626–634 tokens/s | 255k |

## Things I didn't expect

**The same mistakes, every time.** Some of Qwen's errors showed up in every single run, so they are real habits, not bad luck:
- It misreads one small number in a video clip (more on that below).
- It leaves one key number out of a video summary.

Neither is dramatic. Both would matter if nobody checked.

**The missing "2".** In one video clip, Qwen read a dashboard tile as "$43". In the full-resolution screenshot from the same video, it correctly read "$243". The model wasn't confused. vLLM shrinks every video to about 12,000 tokens no matter how long it is, and the small digit simply didn't survive. Lesson: when numbers matter, send frames, not video.

**The night the box ran out of memory.** Nemotron with MTP 3 and vision tried to reserve memory for the largest possible video input while starting up. It used up all 128 GB. The Linux OOM killer then started shooting at random processes: the display manager, dbus, printing services, and finally my benchmark script. That script was the one responsible for restarting my production model afterwards. Lowering vLLM's memory limit from 75 % to 70 % didn't help. That part of vLLM doesn't respect the limit.

The fix was a small watchdog that kills the benchmark container as soon as free memory drops below 2 GiB, so the kernel never has to step in. My first threshold of 6 GiB was too strict: Ornith normally runs with only 5–6 GiB free, and the watchdog killed it mid-test.

**Somebody else was using my server.** The first runs against the production Qwen were slow for no obvious reason. It turned out other clients on the network were keeping it busy with 10 running and 8–9 queued requests. Scores are still valid under load, timings aren't. All clean measurements now run on a separate, private model instance.

## What I'm doing with this

| Task | Choice | Why |
|---|---|---|
| Everyday agent work (Hermes, skills, tools, pipelines) | **Qwen3.8-27B**, MTP 2 or 3 | Never picks the wrong tool, best throughput with several requests, strong on video |
| Hard thinking, careful writing, video analysis | **Ornith 1.5 122B**, MTP 1 | Best quality scores (96 % with MTP 1), 97.8 % on video, fast for a single request |
| Very long documents (beyond 256k tokens) | **Nemotron 3.5 Super VL**, text only, MTP 2–3 | The only one tested up to 1M tokens, by far the fastest reader |
| Reading numbers from videos | Full-resolution frames (1280 px) instead of video input | Video sampling loses small digits |

On speculative decoding:
- **For one request at a time,** more guessed tokens help, up to about 35 tokens/s on the big models.
- **Under load, it backfires.** On the mixture-of-experts models, MTP 3 under load is slower than no MTP at all. One or two guessed tokens is the sweet spot.

Things I'm changing right away:
- **Video frames** are captured at 1280 px instead of 336 × 336, with one frame per two minutes (8–20 per video). The old 336 × 336 frames squeezed 16:9 video into a square and made on-screen text unreadable. And as the "$243" tile showed, small digits only survive at full resolution. A fixed eight frames per video also meant a 56-minute video got the same coverage as a 9-minute one.
- **Skills:** descriptions that models mix up get sharper boundaries (`loop` vs `schedule`, `youtube-ingestion` vs `youtube-content`). In a quick test this fixed every mix-up.
- **Benchmarks:** the watchdog is now part of every benchmark script, so a configuration that runs out of memory only kills itself and never the whole machine again.

## What's next

1. **Give Nemotron MTP 3 another chance** with limited vision (fewer frames per video, or text only), to see whether it beats MTP 2 once it fits.
2. **Let Nemotron see more of each video.** Its video input sends only about 4,700 tokens per clip, against about 12,000 for Qwen and Ornith. That is probably why it misses short or small details (job titles on a slide, single numbers), while it read every full-resolution frame perfectly. I want to rerun the clips with more frames per video and see how much of the gap closes, and how much memory it costs.
3. **A screenshot test with traps:** a slide plus questions whose honest answer is "that's not on the slide".
4. **Writing, code and long-form tests:**
   - summaries checked for missing and invented facts
   - about ten coding tasks with unit tests
   - a book chapter from an outline, checked for consistency
5. **Run every skill case five times**, to separate real mistakes from sampling noise.
6. **Find out why Nemotron sometimes skips the tool call entirely:**
   - log the raw output of failed calls
   - compare tool parsers
   - try forcing a tool call
   - run without async scheduling
7. **Long-context reasoning, not just retrieval:** questions that need facts from different parts of a 256k–1M document, plus several long requests in parallel.
8. **Housekeeping:**
   - find out who is keeping the production server busy
   - reboot after the out-of-memory night and check the services that failed
