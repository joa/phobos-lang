# Agent Tasks

The tasks `scripts/agent_bench.py` gives the [pi](https://pi.dev) coding agent,
one folder each. The harness copies a folder without its `check.*` file, runs
pi on its `prompt.md` there, and records every request pi sends. A replay then
sends the same requests to each engine, with every answer capped at its
recorded length. See [BENCHMARKS.md](../BENCHMARKS.md) for the results.

| task      | work                                    | load                                  | check                      |
| --------- | --------------------------------------- | ------------------------------------- | -------------------------- |
| `fib`     | fix the bug in `fib.js`                 | several turns, prompt and decode      | `fib(1..30)`               |
| `pelican` | draw a pelican riding a bicycle as SVG  | one long answer, mostly decode        | `pelican.svg` parses       |
| `law`     | summarize a German law                  | one long document, mostly prompt      | none                       |

The law is the Tierschutz-Hundeverordnung from
[gesetze-im-internet.de](https://www.gesetze-im-internet.de/tierschhuv/BJNR083800001.html),
converted to Markdown. As an official work it is not under copyright
(Section 5 UrhG).

A task is completed when pi's last message contains DONE, and correct when its
check passes. Whether the pelican is a pelican takes a look at the SVG, which
stays in the results folder under `<engine>-rep<N>/pelican/`.
