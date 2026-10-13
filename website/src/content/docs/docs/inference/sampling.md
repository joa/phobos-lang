---
title: Sampling and chat
description: How phobos-cli picks each token, the flags that control it, and how prompts reach the model.
sidebar:
  order: 3
---

After each step the model gives a score, a logit, for every token in its
vocabulary. The sampler turns those scores into one token. `phobos-cli`'s
flags set how; when serving, they are the defaults a request falls back to
for any field it does not send.

## The flags

| Flag | Default | Effect |
| --- | --- | --- |
| `-t`, `--temp T` | `0` | Temperature. `0` is greedy: always the highest-scoring token. Higher values flatten the distribution. |
| `-k`, `--top-k K` | `0`, off | Sample only from the `K` highest-scoring tokens. |
| `-p`, `--top-p P` | `1.0`, off | Nucleus sampling: keep the most likely tokens until their probabilities add up to `P`. |
| `--min-p P` | `0.0`, off | Keep only tokens at least `P` times as likely as the most likely one. |
| `--presence-penalty P` | `0.0`, off | Subtract `P` from the logit of every token generated so far. The prompt does not count. |
| `--repetition-penalty P` | `1.0`, off | Push every token already in the sequence, prompt included, towards zero: a positive logit is divided by `P`, a negative one multiplied. |
| `--seed S` | `0` | Seed of the random generator, so a sampled run can be repeated. |
| `-n`, `--num N` | `200` | Generate at most `N` tokens. Generation stops earlier at an end-of-turn token or the model's context limit. |
| `--show K` | `5` | Show the `K` most likely candidates for the first token. |

The defaults are greedy and deterministic. A chat model usually comes with
recommended settings in its model card; for MiniCPM5-1B the README uses:

```sh
phobos-cli --gguf models/minicpm5-1b-Q8_0.gguf --temp 0.9 --top-p 0.95 "Write a haiku about the moon."
```

## In what order

Each step goes through the same stages:

1. **Penalties.** The repetition penalty applies to every distinct token in
   the prompt and the answer so far, the presence penalty to every distinct
   token of the answer. Each applies once per token, however often the token
   occurs.
2. **Greedy.** With a temperature of `0`, the highest logit wins and the
   remaining stages are skipped.
3. **Top-k.** Only the `K` highest logits stay.
4. **Temperature.** The survivors become probabilities through a softmax of
   the logits divided by the temperature.
5. **Top-p.** The shortest run of most likely tokens whose probabilities reach
   `P` stays.
6. **Min-p.** Tokens less likely than `P` times the most likely one are
   dropped. The most likely token always stays.
7. **Draw.** The remaining probabilities are renormalized and one token is
   drawn.

## How a prompt reaches the model

Most chat models carry a chat template in their GGUF file: the markers that
separate system, user and assistant turns. When the file has one,
`phobos-cli` sends your text as a user turn in it, the same way the server
formats a chat request, and the model answers as the assistant.

`--raw` skips that. The text goes to the model as typed, and the output is
the model's continuation of it. Use it for a base model, which has no chat
format, or for text that is already a complete prompt.

Some models write out their reasoning before the answer. `--prefill-thinking
TEXT` opens that reasoning on `TEXT` and lets the model continue from it.
When serving, it applies to every chat request, whatever the request asks
for. It is ignored with `--raw` or for a model without a chat template.

## Over the API

A request to the [server](/docs/inference/serving/) can set `temperature`,
`top_k`, `top_p`, `min_p`, `presence_penalty`, `repetition_penalty`, `seed`
and `max_tokens` for itself. Anything it leaves out takes the value of the
flag the server was started with.
