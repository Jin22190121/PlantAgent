"""공통 설정 로더 (settings.yaml 단일 소스)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings() -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["project_root"] = str(PROJECT_ROOT)
    return cfg


def resolve_path(rel: str) -> Path:
    return (PROJECT_ROOT / rel).resolve()


def load_prompt(name: str) -> str:
    cfg = load_settings()
    path = PROJECT_ROOT / cfg["paths"]["prompts_dir"] / name
    return path.read_text(encoding="utf-8")


def google_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요."
        )
    return key


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")
