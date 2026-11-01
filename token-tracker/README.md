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

## What it tells me

```bash
./vllm_token_tracker.py report --since 7d --group-by day,model   # tokens per day and model
./vllm_token_tracker.py sessions                                 # every server run with totals
./vllm_token_tracker.py power --since 7d                         # energy in kWh and euros
./vllm_token_tracker.py roi --hours 40                           # break-even against a cloud API
```

`report` can filter by time, model, endpoint or session, group by hour, day, month, model, endpoint or session, and print a table, CSV or JSON.

**Cloud comparison.** You store a comparison price per million tokens for each model: input, cached input and output. The model name is matched by pattern, and `*` covers everything else. Prices are applied at report time, so changing a price also updates old data:

```bash
./vllm_token_tracker.py price set ornith --input 0.178 --output 0.712
./vllm_token_tracker.py price set '*' --input 0.107 --cache 0.107 --output 1.418
```

**Energy.** `power` integrates the sampled watts over time into kWh and multiplies by your electricity price (default €0.34/kWh).

**ROI.** `roi` puts it all together:
- what the traffic would have cost in the cloud
- electricity
- hardware cost and depreciation
- the hours you put into the setup at your hourly rate

From that it estimates the day the machine breaks even.

## One week on my box

Here is the last week (late October 2026), most of it Hermes running on Qwen3.8-27B, plus the benchmark runs:

| | |
|---|---|
| Requests | 11,203 |
| Prompt tokens | 527 million, of which 428 million (81 %) came from the prefix cache |
| Generated tokens | 21.4 million |
| Same traffic at the comparison prices above | about €87 |
| GPU energy | 7.3 kWh at an average of 40 W, about €2.50 |

Two things jumped out at me:
- **The traffic is heavily prompt-side.** An agent re-sends its long context with every step, so prefix caching is not a nice-to-have. It is doing most of the work.
- **Electricity is almost nothing next to the cloud price**, at least for the GPU itself.

One caveat: `nvidia-smi` reports GPU power only. The whole machine at the wall draws more, so treat the energy numbers as a lower bound.

## Running it as a service

```bash
mkdir -p ~/bin ~/.config/systemd/user
cp vllm_token_tracker.py ~/bin/ && chmod +x ~/bin/vllm_token_tracker.py
cp vllm-tokens.service ~/.config/systemd/user/
systemctl --user enable --now vllm-tokens
sudo loginctl enable-linger $USER   # keep it running without a login session
```

The database lives in `~/.local/share/vllm-tokens/tokens.db`, or wherever `VLLM_TOKEN_DB` points. For servers on other machines, add them with `collect --endpoint http://host:port`.
