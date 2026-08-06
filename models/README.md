# models/

Nothing in this directory is checked in apart from this README and
`export_kv.py`; `.gitignore` drops the rest. The commands below are how the
files get here.

Paths are relative to the repository root.

```
models/
  GPT2/                          zoo export, final hidden state + present KV
    model.onnx
    test_data_set_0/
  gpt2-lm-head-10/
    GPT-2-LM-HEAD/               zoo export, logits + present KV
      model.onnx
      test_data_set_0/
  gpt2-kv/                       the KV-cached pair, produced by export_kv.py
    decoder.onnx
    decoder_with_past.onnx
    vocab.json
    merges.txt
```

## The GPT-2 ONNX graphs

Both come from the ONNX model zoo, opset 10, fully dynamic. `GPT2` ends at the
final hidden state; `gpt2-lm-head-10` carries the LM head and so produces
logits, which is the one the full-recompute engine wants.

```sh
cd models
curl -LO https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/gpt-2/model/gpt2-10.tar.gz
curl -LO https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.tar.gz
tar -xzf gpt2-10.tar.gz            # unpacks ./GPT2
tar -xzf gpt2-lm-head-10.tar.gz    # unpacks ./gpt2-lm-head-10/GPT-2-LM-HEAD
```

Neither tarball is small (440 MB and 580 MB) and both unpack to a single
`model.onnx` of roughly the same size plus a `test_data_set_0` of reference
input/output tensors.

## The vocabulary and the merge list

An ONNX file carries no vocabulary, so the tokenizer is two files sitting
beside the model. `phobos-onnx`'s loader takes either naming, Hugging Face's
first and the original OpenAI one second:

- `vocab.json` beside `merges.txt`
- `encoder.json` beside `vocab.bpe`

Both hold the same two things, a JSON token-to-id map and a ranked merge list,
and both describe the same 50257-token byte-level BPE. Drop one pair into the
model directory, or point `--tokenizer DIR` at wherever they live.

```sh
# Hugging Face naming, into the LM-head export
cd models/gpt2-lm-head-10/GPT-2-LM-HEAD
curl -LO https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json
curl -LO https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt
```

```sh
# or the OpenAI naming, from the original GPT-2 release
cd models/GPT2
curl -LO https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json
curl -LO https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe
```

`phobos-onnx`'s tokenizer tests read the pair from `models/GPT2` and are the
only thing in the workspace that wants a model directory. They skip themselves
when it holds no tokenizer, so `cargo test -p phobos-onnx` passes on a fresh
checkout and gains four real checks, the reference encoding among them, once
the files are in place.

`export_kv.py` writes the Hugging Face pair itself, so `models/gpt2-kv` needs
nothing extra.

## The KV-cache export

The zoo exports take a single input and expose no `past` inputs, so a decode
step through them recomputes the whole sequence. `export_kv.py` re-exports the
same Hugging Face `gpt2` weights as the two graphs the KV engine loads:

- `decoder.onnx`: `input_ids [batch, seq]` -> `logits`, `present.{0..11}.{key,value}`
- `decoder_with_past.onnx`: `input_ids` plus `past.{0..11}.{key,value}` -> `logits`, `present.*`

The prompt runs through the first, each single token through the second with
the cache threaded `present` -> `past`, which makes a step O(1) in sequence
length.

```sh
pip install "torch==2.5.1" "transformers==4.47.1" onnx
python models/export_kv.py --out models/gpt2-kv
```

CPU only, a couple of minutes, and it writes two graphs of about 650 MB each
plus `vocab.json` and `merges.txt`. `--model` takes any GPT-2 shaped
checkpoint (id or local path); `--no-tokenizer` skips the two text files.

The attention has to be exported `eager`: `sdpa` folds it into one node the
interpreter has no op for. The opset has to be 11: 10 has no `Range` for the
position arithmetic, and later ones start emitting fused attention.
And the past/present names have to keep the `past.{layer}.{key|value}` form,
because that is what `KvGraph` looks up.

## Verifying

```sh
cargo run --release -p phobos-onnx --example kv_check -- models/gpt2-kv
```

A single with-past step has to reproduce the last-row logits of a full
recompute over the same sequence; anything above 1e-3 relative error fails the
example. The freshly exported pair matches bit for bit (max rel err 0e0).

End to end, on the host interpreter:

```sh
cargo run --release -p phobos-cli -- --onnx models/gpt2-kv "The color of the sky is"
```

The device paths need a GPU and the CUDA toolkit:

```sh
cargo run --release -p phobos-onnx --example run_gpt2_gpu --features cuda -- models/GPT2
cargo run --release -p phobos-onnx --example chain_gpt2 --features cuda -- models/gpt2-kv
```
