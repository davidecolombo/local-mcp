# /// script
# requires-python = ">=3.11"
# dependencies = ["mcp[cli]>=1.0.0", "httpx>=0.27.0", "pytest>=8.0", "tree-sitter>=0.21", "tree-sitter-java>=0.21"]
# ///
"""Minimal, model-free smoke test for the corruption-risk path (Task-007).

Exercises only pure functions: parsing, guards, size budget, path/IO, language
detection. No Ollama, no network, no mocking. Run either of:
    uv run tests/test_guards.py
    uv run --with pytest --with "mcp[cli]" --with httpx -m pytest tests/ -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import server as s

F = 'C:/proj/Foo.java'


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parse_single_block():
    raw = f'«file path="{F}"»\nbody\n«/file»'
    assert s._parse_file_blocks(raw) == {F: "body"}


def test_parse_multiple_blocks():
    raw = ('«file path="a.py"»\nA\n«/file»\n\n«file path="b.py"»\nB\n«/file»')
    assert s._parse_file_blocks(raw) == {"a.py": "A", "b.py": "B"}


def test_parse_missing_closing_tag():
    raw = f'«file path="{F}"»\nbody with no close'
    assert s._parse_file_blocks(raw) == {}


def test_fallback_markdown_single_file():
    text = "```java\npublic class Foo {}\n```"
    assert s._fallback_markdown_extract(text, [F]) == {F: "public class Foo {}\n"}


def test_fallback_markdown_multiple_files_returns_empty():
    text = "```\nx\n```"
    assert s._fallback_markdown_extract(text, ["a", "b"]) == {}


def test_fallback_markdown_no_fence():
    assert s._fallback_markdown_extract("no fence here", [F]) == {}


def test_extract_prefers_file_block_then_falls_back():
    assert s._extract_file_changes(f'«file path="{F}"»\nB\n«/file»', [F]) == {F: "B"}
    assert s._extract_file_changes("```\nB\n```", [F]) == {F: "B\n"}
    assert s._extract_file_changes("garbage", [F]) == {}


# --------------------------------------------------------------------------- #
# Guards: non-empty / truncation / shrink
# --------------------------------------------------------------------------- #
def test_check_non_empty():
    assert s._check_non_empty("x") is None
    assert s._check_non_empty("   \n ") is not None


def test_truncation_marker_flagged_when_new():
    new = "code\n// ... existing code ...\nmore"
    assert s._check_truncation_markers(new, "code\nmore") is not None


def test_truncation_marker_allowed_if_in_original():
    line = "// ... existing code ..."
    assert s._check_truncation_markers(f"a\n{line}\nb", f"a\n{line}\nb") is None


def test_truncation_marker_absent():
    assert s._check_truncation_markers("clean code", "clean") is None


def test_shrink_flagged_without_keyword():
    assert s._check_shrink("x", "x" * 100, "tweak it") is not None


def test_shrink_allowed_with_removal_keyword():
    assert s._check_shrink("x", "x" * 100, "remove the dead block") is None


def test_shrink_allowed_small_change():
    assert s._check_shrink("x" * 90, "x" * 100, "tweak") is None


def test_shrink_empty_original():
    assert s._check_shrink("x", "", "anything") is None


# --------------------------------------------------------------------------- #
# Guards: brackets
# --------------------------------------------------------------------------- #
def test_bracket_delta_tuple():
    assert s._bracket_delta("{([") == (1, 1, 1)
    assert s._bracket_delta("{}()[]") == (0, 0, 0)


def test_bracket_write_mode_absolute():
    # original None -> absolute balance required (used by local_write)
    assert s._check_bracket_delta("def f(): pass", None, ".py") is None
    assert s._check_bracket_delta("def f(: pass", None, ".py") is not None


def test_bracket_edit_mode_delta():
    # edit mode: new delta must match original's delta
    assert s._check_bracket_delta("a {}", "b {}", ".java") is None
    assert s._check_bracket_delta("a {", "b {}", ".java") is not None


def test_bracket_skips_non_code():
    assert s._check_bracket_delta("({[", None, ".txt") is None


# --------------------------------------------------------------------------- #
# Guards: semantic parse
# --------------------------------------------------------------------------- #
def test_parses_valid_python():
    assert s._check_parses("def f():\n    return 1\n", ".py") is None


def test_parses_invalid_python_reports_line():
    err = s._check_parses("def f(:\n    pass\n", ".py")
    assert err is not None and "line" in err


def test_parses_valid_json():
    assert s._check_parses('{"a": 1}', ".json") is None


def test_parses_invalid_json():
    assert s._check_parses('{"a": }', ".json") is not None


def test_parses_other_ext_skipped():
    assert s._check_parses("not code at all (", ".md") is None


# --------------------------------------------------------------------------- #
# Size guard
# --------------------------------------------------------------------------- #
def test_size_guard_under_and_over():
    limit = 1024  # available = 1024 - 512 = 512 tokens -> ~1536 chars at 3 cpt
    assert s._check_input_size("x" * 1500, limit, "t") is None
    assert s._check_input_size("x" * 1600, limit, "t") is not None


# --------------------------------------------------------------------------- #
# Path + I/O
# --------------------------------------------------------------------------- #
def test_norm_path_case_and_slash_agnostic():
    assert s._norm_path("C:/A/B/Foo.PY") == s._norm_path("c:\\a\\b\\foo.py")


def test_crlf_roundtrip_preserved(tmp_path):
    p = tmp_path / "crlf.txt"
    original = b"line1\r\nline2\r\n"
    p.write_bytes(original)
    lf, eol, raw = s._read_file(p)
    assert eol == b"\r\n" and lf == "line1\nline2\n"
    assert s._encode_with_eol(lf, eol) == original


def test_lf_roundtrip_preserved(tmp_path):
    p = tmp_path / "lf.txt"
    original = b"a\nb\n"
    p.write_bytes(original)
    lf, eol, raw = s._read_file(p)
    assert eol == b"\n"
    assert s._encode_with_eol(lf, eol) == original


def test_read_file_rejects_binary(tmp_path):
    p = tmp_path / "bin"
    p.write_bytes(b"abc\x00def")
    with pytest.raises(UnicodeDecodeError):
        s._read_file(p)


def test_atomic_write_creates_and_replaces(tmp_path):
    p = tmp_path / "sub" / "out.txt"
    s._atomic_write(p, b"hello")
    assert p.read_bytes() == b"hello"
    s._atomic_write(p, b"world")
    assert p.read_bytes() == b"world"


# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #
def test_english_passes():
    assert s._is_probably_english("add a method that returns the user") is True


def test_accented_fails():
    assert s._is_probably_english("ajoute une méthode") is False


def test_marker_word_fails():
    assert s._is_probably_english("aggiungi un metodo") is False


# --------------------------------------------------------------------------- #
# Java guards (pure, corruption-relevant)
# --------------------------------------------------------------------------- #
def test_java_omission_placeholder_rejected():
    root = r"C:\p\src\main\java\com\ex"
    new = "package com.ex;\npublic class Foo {\n// getters and setters\n}\n"
    fails = s._check_java(new, "package com.ex;\npublic class Foo {}\n", root + r"\Foo.java", "add x")
    assert any("omission placeholder" in f for f in fails)


def test_java_clean_edit_passes():
    root = r"C:\p\src\main\java\com\ex"
    new = "package com.ex;\npublic class Foo {\n    int x;\n}\n"
    assert s._check_java(new, "package com.ex;\npublic class Foo {}\n", root + r"\Foo.java", "add x") == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
