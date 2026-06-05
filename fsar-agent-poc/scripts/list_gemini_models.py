"""GOOGLE_API_KEY로 접근 가능한 Gemini 모델 목록을 출력.

용도: 404 'model not found' 오류 발생 시 어떤 모델 이름이 유효한지 확인.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import google.generativeai as genai

from src.config import google_api_key


def main() -> int:
    genai.configure(api_key=google_api_key())
    print(f"{'모델 ID':50s}  지원 메서드")
    print("-" * 100)
    for m in genai.list_models():
        methods = ",".join(m.supported_generation_methods)
        if "generateContent" in m.supported_generation_methods:
            print(f"{m.name:50s}  {methods}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
