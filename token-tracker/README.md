# vLLM token and energy tracker

*How many tokens does my DGX Spark actually produce, what would they have cost in the cloud, and what does the electricity cost?*

## The idea

Once a local model runs all day, you start wondering whether the box is paying for itself. Questions like:
- How many tokens did I generate this week?
- How much of my prompt traffic came from the prefix cache?
- What would the same traffic have cost through a public API, and what did the power cost?

vLLM already counts everything it does. Every server exposes its token counters on `/metrics`. So the tracker doesn't need to sit in the request path, proxy anything or touch the clients. It just looks over vLLM's shoulder.

`vllm_token_tracker.py` is a single Python file with no dependencies beyond the standard library. Its built-in help and output are in German.

## How it works

The tracker runs as a small background service (`collect`). Every 5 seconds it:
1. Finds every running vLLM server on the machine by itself, whether it runs natively or in Docker. It checks all listening ports and keeps the ones whose `/metrics` look like vLLM.
2. Reads the counters for prompt tokens, cached prompt tokens, generated tokens and successful requests.
3. Stores only the increase since the last reading in a SQLite database.
4. Samples GPU power draw through `nvidia-smi`.

A *session* is one server run: endpoint plus model plus start time. If a server restarts, its counters start again at zero, and the tracker notices and opens a new session. Restarting the tracker itself loses nothing, because the last raw counter values are kept in the database.

That means I can switch models as often as I like (and during the benchmarks I switched a lot) and every run ends up neatly in its own session.

If the tracker was down for a while (or a server was already running before it started), the increase it catches up on is spread evenly over the missed hours, so a daily report doesn't show the whole backlog as one huge spike on the day it was picked up. If the same server is reachable under two addresses, for example through `--endpoint` and the port scan, it is counted only once.

## What it tells me

```bash
./vllm_token_tracker.py report --since 7d --group-by day,model   # tokens per day and model
./vllm_token_tracker.py sessions                                 # every server run with totals and Wh per 1M output tokens
./vllm_token_tracker.py power --since 7d                         # energy in kWh and euros
./vllm_token_tracker.py roi --hours 40                           # break-even against a cloud API
```

`report` can filter by time, model, endpoint or session, group by hour, day, month, model, endpoint or session, and print a table, CSV or JSON.

**Cloud comparison.** You store a comparison price per million tokens for each model: input, cached input and output. The model name is matched by pattern, and `*` covers everything else. Prices are applied at report time, so changing a price also updates old data:

```bash
./vllm_token_tracker.py price set ornith --input 0.178 --output 0.712
./vllm_token_tracker.py price set '*' --input 0.107 --cache 0.107 --output 1.418
./vllm_token_tracker.py price set embed --input 0.02 --output 0   # embeddings are much cheaper in the cloud
./vllm_token_tracker.py price set bench --input 0 --output 0      # benchmark runs are not savings
```

**Energy.** `power` integrates the sampled watts over time into kWh and multiplies by your electricity price (default €0.34/kWh). Readings more than 60 seconds apart count as a gap (tracker or machine off) and are left out instead of being bridged, and `gemessen_h` shows how much of the period was actually measured. `nvidia-smi` sees the GPU only. If you know what the rest of the machine draws, for example from a smart plug, add it as a constant base load with `--overhead-watts` or `VLLM_OVERHEAD_WATTS`.

`sessions` also shows the energy during each server run and the Wh per million generated tokens, which is handy for comparing models. Sessions running in parallel (an embedding server next to the LLM, say) each get the full GPU draw of their time window.

**ROI.** `roi` puts it all together:
- what the traffic would have cost in the cloud (`--fx` converts the comparison prices to euros, e.g. `--fx 0.92` for USD prices)
- electricity
- hardware cost and depreciation
- the hours you put into the setup at your hourly rate

From that it estimates the day the machine breaks even, based on the daily savings minus the daily electricity cost over the same period.

## One week on my box

Here is the week of 25 to 31 October 2026, most of it Hermes running on Qwen3.8-27B:

| | |
|---|---|
| Requests | 12,669 |
| Prompt tokens | 534 million, of which 447 million (84 %) came from the prefix cache |
| Generated tokens | 23.6 million |
| Same traffic at the comparison prices above | about €91 |
| GPU energy | 3.7 kWh at an average of 38 W, about €1.25, measured over 95 of the 168 hours |

An earlier version of this table said 7.3 kWh. That number bridged a three-day outage of the tracker with the average of the readings on either side, which is the bug that made me add the gap handling. The 73 hours missing now are simply not counted. The tokens from those hours are not lost, because vLLM's counters kept running.

Two things jumped out at me:
- **The traffic is heavily prompt-side.** An agent re-sends its long context with every step, so prefix caching is not a nice-to-have. It is doing most of the work.
- **Electricity is almost nothing next to the cloud price**, at least for the GPU itself.

One caveat: `nvidia-smi` reports GPU power only. The whole machine at the wall draws more, so without `--overhead-watts` treat the energy numbers as a lower bound.

## Running it as a service

I first ran it as a systemd user service. Then loading big models filled the shared memory, and the OOM killer kept picking the tracker, once leaving it down for three days. It now runs as a system service for your user with a lowered OOM score:

```bash
mkdir -p ~/bin
cp vllm_token_tracker.py ~/bin/ && chmod +x ~/bin/vllm_token_tracker.py
sudo cp vllm-tokens@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-tokens@$USER
```

Your user needs to be in the `docker` group so the tracker can see the ports of your containers.

The database lives in `~/.local/share/vllm-tokens/tokens.db`, or wherever `VLLM_TOKEN_DB` points. For servers on other machines, add them with `collect --endpoint http://host:port`.
