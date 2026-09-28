import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LOCAL_SERVICE = ROOT / "src/api/sentence_generation/sentence_service_local.py"


def _interpret_glosses_generate_call() -> ast.Call:
    tree = ast.parse(LOCAL_SERVICE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "LocalSentenceService":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "interpret_glosses":
                    for child in ast.walk(item):
                        if (
                            isinstance(child, ast.Call)
                            and isinstance(child.func, ast.Attribute)
                            and child.func.attr == "generate"
                        ):
                            return child
    raise AssertionError("LocalSentenceService.interpret_glosses model.generate call not found")


def test_interpret_glosses_uses_configured_generation_limit() -> None:
    generate_call = _interpret_glosses_generate_call()
    keyword = next(
        (kw for kw in generate_call.keywords if kw.arg == "max_new_tokens"),
        None,
    )

    assert keyword is not None
    assert isinstance(keyword.value, ast.Attribute)
    assert isinstance(keyword.value.value, ast.Name)
    assert keyword.value.value.id == "config"
    assert keyword.value.attr == "LLM_MAX_LENGTH"


def _load_config_with_limit(value: str | None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if value is None:
        env.pop("LLM_MAX_LENGTH", None)
    else:
        env["LLM_MAX_LENGTH"] = value
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.api.config import config; print(config.LLM_MAX_LENGTH)",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_default_llm_max_length_is_100() -> None:
    result = _load_config_with_limit(None)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "100"


def test_positive_llm_max_length_override_is_respected() -> None:
    result = _load_config_with_limit("64")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "64"


def test_non_positive_llm_max_length_is_rejected() -> None:
    result = _load_config_with_limit("0")

    assert result.returncode != 0
    assert "LLM_MAX_LENGTH must be greater than 0" in result.stderr
