"""Export a Hugging Face GPT-2 as the two ONNX graphs the KV engine wants.

The GPT-2 exports in the ONNX model zoo take a single `input1` and no past
inputs, so every decode step has to recompute the whole sequence. This script
exports the same weights twice instead:

    decoder.onnx            input_ids -> logits, present.{l}.{key,value}
    decoder_with_past.onnx  input_ids + past.* -> logits, present.*

which is what `phobos_onnx::runtime::KvGraph` loads. It also drops the
tokenizer files next to them, since an ONNX file carries no vocabulary.

    pip install "torch==2.5.1" "transformers==4.47.1" onnx
    python models/export_kv.py --out models/gpt2-kv

Requires only a CPU. The two graphs are ~650 MB each.
"""

import argparse
import pathlib

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# The exporter that produced the graphs in use. opset 11 is the floor for the
# Range op the position arithmetic lowers to, and the ceiling that keeps the
# attention as plain MatMul/Softmax rather than a fused Attention node.
OPSET = 11


def flat_names(prefix, n_layer):
    """past.0.key, past.0.value, past.1.key, ... in cache order."""
    return [f"{prefix}.{l}.{kind}" for l in range(n_layer) for kind in ("key", "value")]


def to_cache(pairs):
    """Per-layer (key, value) tuples in whatever form this transformers wants."""
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return pairs
    return DynamicCache.from_legacy_cache(pairs)


def from_cache(cache):
    """The per-layer (key, value) tuples out of a forward pass, flattened."""
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return [t for layer in cache for t in layer]


class Decoder(torch.nn.Module):
    """The prompt pass: no cache in, a full cache out."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        out = self.model(input_ids=input_ids, use_cache=True)
        return (out.logits, *from_cache(out.past_key_values))


class DecoderWithPast(torch.nn.Module):
    """One step: a token plus the cache in, the grown cache out."""

    def __init__(self, model, n_layer):
        super().__init__()
        self.model = model
        self.n_layer = n_layer

    def forward(self, input_ids, *past):
        pairs = tuple((past[2 * l], past[2 * l + 1]) for l in range(self.n_layer))
        out = self.model(
            input_ids=input_ids, past_key_values=to_cache(pairs), use_cache=True
        )
        return (out.logits, *from_cache(out.past_key_values))


def export(module, args, path, input_names, output_names, dynamic_axes):
    print(f"exporting {path} ...")
    torch.onnx.export(
        module,
        args,
        str(path),
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2", help="HF model id or local path")
    ap.add_argument("--out", default="models/gpt2-kv", help="output directory")
    ap.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="skip writing vocab.json and merges.txt beside the graphs",
    )
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # sdpa exports as one opaque node the interpreter has no op for; eager
    # keeps the attention as MatMul, Softmax and Where.
    model = GPT2LMHeadModel.from_pretrained(args.model, attn_implementation="eager")
    model.eval()
    cfg = model.config
    n_layer, n_head = cfg.n_layer, cfg.n_head
    head_dim = cfg.n_embd // n_head

    past_names = flat_names("past", n_layer)
    present_names = flat_names("present", n_layer)

    # A 2-token prompt, then a 1-token step over a 2-long cache: the axes that
    # matter are all dynamic, so the example only has to be well formed.
    prompt_ids = torch.tensor([[15496, 995]], dtype=torch.long)
    step_ids = torch.tensor([[995]], dtype=torch.long)
    past = [torch.zeros(1, n_head, 2, head_dim) for _ in past_names]

    seq_axes = {0: "batch", 1: "seq"}
    with torch.no_grad():
        export(
            Decoder(model),
            (prompt_ids,),
            out / "decoder.onnx",
            ["input_ids"],
            ["logits", *present_names],
            {
                "input_ids": seq_axes,
                "logits": seq_axes,
                **{n: {0: "batch", 2: "seq"} for n in present_names},
            },
        )
        export(
            DecoderWithPast(model, n_layer),
            (step_ids, *past),
            out / "decoder_with_past.onnx",
            ["input_ids", *past_names],
            ["logits", *present_names],
            {
                "input_ids": seq_axes,
                "logits": seq_axes,
                **{n: {0: "batch", 2: "past"} for n in past_names},
                **{n: {0: "batch", 2: "total"} for n in present_names},
            },
        )

    if not args.no_tokenizer:
        GPT2Tokenizer.from_pretrained(args.model).save_vocabulary(str(out))
        print(f"wrote {out / 'vocab.json'} and {out / 'merges.txt'}")

    print(f"done. verify with: cargo run -p phobos-onnx --example kv_check -- {out}")


if __name__ == "__main__":
    main()
