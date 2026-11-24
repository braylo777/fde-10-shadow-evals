# fde-10-shadow-evals

**Forward Deployed Engineering — Stage 10 of 12: Shadow Evals on Customer Data**

Decide with evidence whether a new agent version is better on this customer's real traffic: golden sets from logs, trajectory grading in shadow mode, a calibrated LLM judge, and business metrics with confidence intervals.

**Also built as Project 10 of 12, "Historical Shadow Replay Harness",** from the companion [12 FDE projects post](https://x.com/suraj_sharma14/status/2125197762752806998).

**Live demo:** https://fde-10-shadow-evals.braylo777.workers.dev — the Rust core runs in your browser as WebAssembly. No server, no tracking.

## Why this matters

Public benchmarks do not tell you whether the new prompt or model is better for this customer. Their logs do. The FDE turns past conversations into a golden set, runs the new agent in shadow mode next to production (it sees real inputs; its actions are never executed), grades the full tool trajectory, checks that the LLM judge actually agrees with humans, and ships only when the business metric improves and no guardrail gets worse.

## What it does

- Golden set from logs: PII scrub, near-duplicate removal (shingle Jaccard), outcome labels, good/bad balance per intent, stable hash split
- Trajectory grading: required tools in order, forbidden tools, key arguments, step budget, required answer facts; regression / fix diff vs production; promotion gate
- Judge calibration: Cohen's kappa threshold sweep, confusion matrix, precision / recall, length-bias and position-bias warnings
- Business metrics: rate definitions as filters, Wilson 95% intervals, two-proportion z-test, guardrails that block a ship
- Paired replay: set paired_by to the ticket id and each metric uses an exact McNemar test over the tickets where production and shadow disagree; grade_trajectories reports the same paired test on pass / fail

## What it does not do (yet)

- No model calls: the shadow agent's trajectories and the judge's scores are inputs
- Near-duplicate detection is lexical; semantic duplicates need embeddings
- The unpaired z-test is still the default; set paired_by when both sides saw the same tickets

## Operations

Every operation takes JSON in and gives JSON out. The CLI, the tests and the demo page all use the same entry point.

| Operation | What it does |
|---|---|
| `build_golden_set` | 36 past tickets: order numbers and emails are scrubbed, near-duplicates dropped, unclear outcomes (some edits) excluded, and each intent gets both good and bad examples. |
| `grade_trajectories` | Five replayed refund cases. The shadow agent fixes two production bugs but uses a forbidden tool once, so it is not promoted. |
| `calibrate_judge` | 40 items with human pass/fail and a 1-5 judge score. Finds the best threshold by kappa, then checks for length and position bias: this judge favors whichever answer is shown first. |
| `compare_metric` | First-contact resolution on 600 production vs 600 shadow tickets, with two guardrails. Ship only if the primary metric improves significantly and no guardrail is worse. |

## Run it

You need Rust (stable). No Node, no npm.

```bash
git clone https://github.com/braylo777/fde-10-shadow-evals
cd fde-10-shadow-evals
cargo test
cargo run -- ops
```

Try the first demo example from the command line:

```bash
python3 -c 'import json;print(json.dumps(json.load(open("web/examples.json"))["ops"][0]["input"]))' | cargo run -q -- build_golden_set
```

## Build and deploy the demo

```bash
./build.sh    # tests, then compiles the core to web/app.wasm
./deploy.sh   # serves web/ as a Cloudflare Workers static-assets site (needs a Cloudflare login)
```

`web/` is a plain static site (`index.html`, `app.js`, `app.wasm`). It has no build step, so any static host works.

## Layout

```
src/lib.rs        operation list and router
src/api.rs        JSON entry point + the 3 WASM exports (alloc, dealloc, run)
src/main.rs       CLI
src/*.rs          one module per capability, each with unit tests
tests/            integration tests (every demo example must pass)
web/              demo page; examples.json drives both the page and this README
```

## Design choices

- **Rust, few dependencies.** Only `serde`, `serde_json`, `sha2` and `hmac`, pinned to exact versions with a committed `Cargo.lock`.
- **Deterministic.** Time, randomness and keys are inputs, so every result can be replayed in a test or an audit.
- **Fail closed.** Unknown input, missing fields or an unmatched policy gives an error or a denial, never a silent pass.

## Part of a series

This is one of 12 repos that work through a public Forward Deployed Engineering roadmap ([12 stages](https://x.com/suraj_sharma14/status/2124827808535048566), [12 projects](https://x.com/suraj_sharma14/status/2125197762752806998)). All 12: [github.com/braylo777?q=fde-](https://github.com/braylo777?tab=repositories&q=fde-).

## License

MIT — see [LICENSE](LICENSE). Built by [Braylo SI](https://braylo.si).
