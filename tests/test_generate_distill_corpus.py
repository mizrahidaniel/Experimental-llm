"""scripts/generate_distill_corpus.py — MarkovTeacher fallback path."""

import importlib.util
import json
import sys
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_distill_corpus.py"
    spec = importlib.util.spec_from_file_location("generate_distill_corpus", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_markov_teacher_deterministic():
    mod = _load_script()
    a = mod.MarkovTeacher(seed=42).generate("hello", max_new_chars=64)
    b = mod.MarkovTeacher(seed=42).generate("hello", max_new_chars=64)
    assert a == b
    assert len(a) == 64


def test_markov_teacher_variable_length():
    mod = _load_script()
    out = mod.MarkovTeacher(seed=0).generate("", max_new_chars=10)
    assert len(out) == 10


def test_iter_prompts_no_path():
    mod = _load_script()
    prompts = list(mod.iter_prompts(None, n_total=3))
    assert prompts == ["", "", ""]


def test_iter_prompts_jsonl(tmp_path):
    mod = _load_script()
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        '{"prompt": "first"}\n'
        '{"text": "second"}\n'
        '{"prompt": "third"}\n'
    )
    prompts = list(mod.iter_prompts(p, n_total=10))
    assert prompts == ["first", "second", "third"]


def test_iter_prompts_caps_at_n_total(tmp_path):
    mod = _load_script()
    p = tmp_path / "prompts.jsonl"
    p.write_text("\n".join(json.dumps({"prompt": f"p{i}"}) for i in range(20)))
    prompts = list(mod.iter_prompts(p, n_total=3))
    assert prompts == ["p0", "p1", "p2"]


def test_load_teacher_mock_explicit():
    mod = _load_script()
    backend, payload = mod.load_teacher("mock")
    assert backend == "mock"
    assert isinstance(payload, mod.MarkovTeacher)


def test_full_pipeline_mock(tmp_path, monkeypatch):
    mod = _load_script()
    out = tmp_path / "corpus.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "generate_distill_corpus.py",
        "--teacher", "mock",
        "--output", str(out),
        "--num-samples", "4",
        "--max-new-tokens", "16",
    ])
    mod.main()
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 4
    for line in lines:
        rec = json.loads(line)
        assert "prompt" in rec
        assert "completion" in rec
        assert isinstance(rec["completion"], str)
        assert len(rec["completion"]) > 0
