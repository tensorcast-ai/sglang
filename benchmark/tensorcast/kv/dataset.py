from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tensorcast.kv.models import RequestPrompt


class _QuestionRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=1)


class _LongBenchRow(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    record_id: str | int = Field(alias="_id")
    context: str = Field(min_length=1)
    input: str = ""
    length: int = Field(ge=0)


def _detect_schema(row: dict[str, Any]) -> str:
    if "question" in row:
        return "question"
    if {"_id", "context", "input", "length"}.issubset(row):
        return "longbench"
    raise RuntimeError(f"Unsupported dataset schema with keys: {sorted(row.keys())}")


def _render_longbench_prompt(row: _LongBenchRow) -> str:
    instruction = row.input.strip() or "Respond to the context above."
    return "\n\n".join(
        [
            "Context:",
            row.context,
            "Instruction:",
            instruction,
            "Answer:",
        ]
    )


def load_prompts(
    path: str | Path,
    limit: int,
    *,
    min_prompt_chars: int = 0,
    max_prompt_chars: int = 0,
) -> list[RequestPrompt]:
    dataset_path = Path(path).expanduser().resolve()
    prompts: list[RequestPrompt] = []
    schema: str | None = None
    with dataset_path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if len(prompts) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            raw_row = json.loads(line)
            if schema is None:
                schema = _detect_schema(raw_row)

            if schema == "question":
                row = _QuestionRow.model_validate(raw_row)
                filter_length = len(row.question)
                if filter_length < min_prompt_chars:
                    continue
                if max_prompt_chars > 0 and filter_length > max_prompt_chars:
                    continue
                prompts.append(
                    RequestPrompt(
                        prompt_id=f"prompt-{index:04d}",
                        prompt_text=row.question,
                        prompt_filter_length=filter_length,
                    )
                )
                continue

            if schema == "longbench":
                row = _LongBenchRow.model_validate(raw_row)
                if row.length < min_prompt_chars:
                    continue
                if max_prompt_chars > 0 and row.length > max_prompt_chars:
                    continue
                prompts.append(
                    RequestPrompt(
                        prompt_id=str(row.record_id),
                        prompt_text=_render_longbench_prompt(row),
                        prompt_filter_length=row.length,
                    )
                )
                continue

            raise RuntimeError(f"Unhandled schema: {schema}")
    if len(prompts) < limit:
        raise RuntimeError(
            f"Requested {limit} prompts from {dataset_path}, found {len(prompts)} "
            f"after applying min_prompt_chars={min_prompt_chars}, "
            f"max_prompt_chars={max_prompt_chars}"
        )
    return prompts
