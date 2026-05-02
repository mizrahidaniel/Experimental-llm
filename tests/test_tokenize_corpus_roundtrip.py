"""scripts/tokenize_corpus.py round-trip via direct module import.

We import the script as a module to test the helpers without subprocess
overhead. Uses MiniBPE fallback (no_network=True) so CI passes offline.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "tokenize_corpus.py"
    spec = importlib.util.spec_from_file_location("tokenize_corpus", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_jsonl(path: Path, texts: list[str]):
    with path.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")


def test_iter_jsonl_basic(tmp_path):
    mod = _load_script()
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, ["hello", "world", "foo bar"])
    texts = list(mod.iter_jsonl(p, "text"))
    assert texts == ["hello", "world", "foo bar"]


def test_iter_jsonl_skip_docs(tmp_path):
    mod = _load_script()
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, ["a", "b", "c", "d"])
    texts = list(mod.iter_jsonl(p, "text", skip_docs=2))
    assert texts == ["c", "d"]


def test_iter_jsonl_handles_malformed(tmp_path):
    mod = _load_script()
    p = tmp_path / "corpus.jsonl"
    p.write_text('{"text": "good"}\nbad json line\n{"text": "also good"}\n')
    texts = list(mod.iter_jsonl(p, "text"))
    assert texts == ["good", "also good"]


def test_write_shard_round_trip(tmp_path):
    mod = _load_script()
    out = tmp_path / "out"
    out.mkdir()
    token_ids = [10, 20, 30, 40, 50]
    doc_offsets = [0, 2, 5]
    path = mod.write_shard(out, shard_id=0, token_ids=token_ids, doc_offsets=doc_offsets)
    assert path.exists()
    with np.load(path) as data:
        assert list(data["token_ids"]) == token_ids
        assert list(data["doc_offsets"]) == doc_offsets


def test_existing_shard_count(tmp_path):
    mod = _load_script()
    out = tmp_path / "out"
    out.mkdir()
    assert mod.existing_shard_count(out) == 0
    mod.write_shard(out, 0, [1, 2], [0, 2])
    mod.write_shard(out, 1, [3, 4], [0, 2])
    assert mod.existing_shard_count(out) == 2


def test_docs_per_existing_shard(tmp_path):
    mod = _load_script()
    out = tmp_path / "out"
    out.mkdir()
    mod.write_shard(out, 0, [1, 2, 3], [0, 1, 3])  # 2 docs
    mod.write_shard(out, 1, [4, 5], [0, 2])         # 1 doc
    assert mod.docs_per_existing_shard(out) == 3


def test_full_pipeline_writes_shards(tmp_path, monkeypatch, capsys):
    """Run main() end-to-end with the offline MiniBPE fallback."""
    mod = _load_script()
    src = tmp_path / "src.jsonl"
    out = tmp_path / "out"
    _write_jsonl(src, ["the quick brown fox", "jumps over the lazy dog", "abc"])

    monkeypatch.setattr(sys, "argv", [
        "tokenize_corpus.py",
        "--tokenizer", "tiny_bpe",
        "--input", str(src),
        "--output", str(out),
        "--field", "text",
        "--shard-size", "10",
        "--no-network",
    ])
    mod.main()

    shards = sorted(out.glob("shard_*.npz"))
    assert shards
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["tokenizer_source"] == "tiny_bpe"
    assert manifest["vocab_size"] == 32_000
    assert manifest["total_docs"] == 3
