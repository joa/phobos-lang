"""Diff phobos's forward pass against a llama.cpp server, token by token.

    cargo run --release -p phobos-gguf --example oracle_check -- MODEL.gguf \
        -k 8 -n 8 "The capital of France is" > phobos.jsonl
    python scripts/oracle_diff.py phobos.jsonl [--port 8090]

The server must be serving the same file (`llama.exe serve -m MODEL.gguf
--port 8090`). Per prompt: whether the server tokenizes it the same, the
top-k log-probabilities of the next token on both sides as differences
against each side's own top token (which cancels the softmax normalization,
so what is left is the two implementations' disagreement), and the greedy
continuation. Near-tied logits legitimately diverge in the greedy text; the
log-prob spread is the measurement.
"""

import argparse
import json
import sys
import urllib.request


def post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    worst = 0.0
    for line in open(args.jsonl, encoding="utf-8"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        ours = json.loads(line)
        ids = ours["ids"]
        prompt = ours["prompt"]
        k = len(ours["top"])
        n = len(ours["greedy"])

        theirs_ids = post(args.port, "/tokenize", {"content": prompt})["tokens"]
        same_tokens = theirs_ids == ids

        res = post(
            args.port,
            "/completion",
            {
                "prompt": ids,
                "n_predict": n,
                "n_probs": k,
                "temperature": 0,
                "top_k": 1,
                "cache_prompt": False,
            },
        )
        first = res["completion_probabilities"][0]
        theirs = {int(t["id"]): float(t["logprob"]) for t in first["top_logprobs"]}
        mine = {int(i): float(lp) for i, lp in ours["top"]}
        their_top = max(theirs, key=theirs.get)
        my_top = ours["top"][0][0]

        # Log-prob differences against each side's own top token, over the
        # tokens both sides rank in their top k.
        shared = [t for t in mine if t in theirs]
        spread = 0.0
        rows = []
        for t in shared:
            d_mine = mine[t] - mine[my_top]
            d_theirs = theirs[t] - theirs[their_top]
            spread = max(spread, abs(d_mine - d_theirs))
            rows.append(f"    {t:>7}  phobos {d_mine:+8.4f}  llama.cpp {d_theirs:+8.4f}")
        worst = max(worst, spread)

        their_greedy = [int(tok["id"]) for tok in res["completion_probabilities"]]
        agree = 0
        for a, b in zip(ours["greedy"], their_greedy):
            if a != b:
                break
            agree += 1

        print(f"prompt: {prompt!r}")
        print(f"  tokens: {'same' if same_tokens else 'DIFFER ' + str(theirs_ids)}")
        print(f"  top token: phobos {my_top}, llama.cpp {their_top}, {len(shared)}/{k} of top-{k} shared, spread {spread:.4f}")
        print("\n".join(rows))
        print(f"  greedy: {agree}/{n} agree; phobos {ours['text']!r}, llama.cpp {res['content']!r}")
    print(f"worst spread {worst:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
