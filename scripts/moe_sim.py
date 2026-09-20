"""Replay a router trace against expert cache policies.

    python scripts/moe_sim.py trace.jsonl [--slots 800,1200,1600,2000,2400]

`trace.jsonl` is what `examples/moe_trace.rs` writes. Each block gets an
equal share of the slot budget (a slot holds one expert of one block), and
every position's chosen experts are looked up in that block's cache in
order. Reported per budget, for the generated positions only (the prompt
positions warm the cache, which is what a prompt pass does for the decode
that follows it):

- LRU: evict the least recently used.
- LFU: evict the least frequently used, counts halved every 256 positions
  so an old favourite can fall out.
- static: the experts most used over the whole trace, never evicted. An
  upper bound on any static placement, since it peeks at the whole trace.
- OPT: Belady's rule, evict the one next used furthest ahead. The upper
  bound on any policy at that budget, since it peeks at the future.
- LRU+la: LRU where block b's lookahead (its router on the residual leaving
  block b-1) is prefetched into the cache before the real choice is made;
  a prefetch that turns out wrong still costs a copy, and the column beside
  it says how many copies the prefetch issued per position on top of the
  misses.

The last line is the one-block lookahead's own accuracy: the share of chosen
experts its prediction named.
"""

import argparse
import json
import sys
from collections import Counter, OrderedDict, defaultdict


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


class Lru:
    def __init__(self, slots):
        self.slots, self.d = slots, OrderedDict()

    def access(self, e):
        hit = e in self.d
        if hit:
            self.d.move_to_end(e)
        else:
            self.d[e] = True
            if len(self.d) > self.slots:
                self.d.popitem(last=False)
        return hit


class Lfu:
    HALVE_EVERY = 256

    def __init__(self, slots):
        self.slots, self.count, self.held, self.seen = slots, Counter(), set(), 0

    def access(self, e):
        self.seen += 1
        if self.seen % self.HALVE_EVERY == 0:
            for k in list(self.count):
                self.count[k] //= 2
        self.count[e] += 1
        hit = e in self.held
        if not hit:
            if len(self.held) >= self.slots:
                victim = min(self.held, key=lambda k: self.count[k])
                self.held.discard(victim)
            self.held.add(e)
        return hit


class Static:
    def __init__(self, hot):
        self.hot = set(hot)

    def access(self, e):
        return e in self.hot


class Opt:
    def __init__(self, slots, sequence):
        self.slots, self.held = slots, set()
        self.nexts = defaultdict(list)
        for i, e in enumerate(sequence):
            self.nexts[e].append(i)
        self.i = 0

    def access(self, e):
        # Position self.i is the current access; drop it from e's future list.
        self.nexts[e].pop(0)
        hit = e in self.held
        if not hit:
            if len(self.held) >= self.slots:
                victim = max(self.held, key=lambda k: self.nexts[k][0] if self.nexts[k] else float("inf"))
                self.held.discard(victim)
            self.held.add(e)
        self.i += 1
        return hit


def simulate(rows, blocks, per_block, n_used):
    """Hit counts over generated positions, per policy, plus prefetch copies."""
    seq = {b: [e for r in rows for e in r["routes"][b]] for b in range(blocks)}
    hot = {b: [e for e, _ in Counter(seq[b]).most_common(per_block)] for b in range(blocks)}
    caches = {
        "LRU": {b: Lru(per_block) for b in range(blocks)},
        "LFU": {b: Lfu(per_block) for b in range(blocks)},
        "static": {b: Static(hot[b]) for b in range(blocks)},
        "OPT": {b: Opt(per_block, seq[b]) for b in range(blocks)},
        "LRU+la": {b: Lru(per_block) for b in range(blocks)},
    }
    hits = Counter()
    prefetches = 0
    accesses = 0
    for r in rows:
        counted = r["phase"] == "decode"
        for b in range(blocks):
            chosen = r["routes"][b]
            if b > 0:
                # Prefetch what block b-1's residual predicted for block b,
                # copying only what is not already resident.
                cache = caches["LRU+la"][b]
                for e in r["lookahead"][b - 1]:
                    if e not in cache.d:
                        if counted:
                            prefetches += 1
                        cache.access(e)
            for name, per in caches.items():
                for e in chosen:
                    hit = per[b].access(e)
                    if counted and hit:
                        hits[name] += 1
            if counted:
                accesses += len(chosen)
    return hits, accesses, prefetches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--slots", default="800,1200,1600,2000,2400")
    args = ap.parse_args()
    rows = load(args.trace)
    if not rows:
        print("empty trace")
        return 1
    blocks = len(rows[0]["routes"])
    n_used = len(rows[0]["routes"][0])
    decode = sum(1 for r in rows if r["phase"] == "decode")
    prompts = len({r["prompt"] for r in rows})
    print(f"{len(rows)} positions ({decode} generated) over {prompts} prompts, {blocks} blocks, top-{n_used}")

    experts = max(e for r in rows for b in r["routes"] for e in b) + 1
    print(f"{experts} experts a block; hit rates over generated positions:")
    print(f"{'slots':>6} {'/block':>6} {'LRU':>7} {'LFU':>7} {'static':>7} {'OPT':>7} {'LRU+la':>7} {'prefetch/pos':>13}")
    for total in (int(s) for s in args.slots.split(",")):
        per_block = max(1, total // blocks)
        hits, accesses, prefetches = simulate(rows, blocks, per_block, n_used)
        rate = lambda name: hits[name] / accesses if accesses else 0.0
        print(
            f"{total:>6} {per_block:>6} {rate('LRU'):>7.3f} {rate('LFU'):>7.3f} {rate('static'):>7.3f} "
            f"{rate('OPT'):>7.3f} {rate('LRU+la'):>7.3f} {prefetches / max(decode, 1):>13.1f}"
        )

    named = 0
    total = 0
    for r in rows:
        for b in range(1, blocks):
            predicted = set(r["lookahead"][b - 1])
            named += sum(1 for e in r["routes"][b] if e in predicted)
            total += len(r["routes"][b])
    print(f"one-block lookahead names {named / total:.3f} of the chosen experts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
