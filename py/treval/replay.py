"""
Interactive span replay.

Allows re-executing a saved trace by modifying parameters
(prompt, model, temperature) and comparing results.

Usage:
    treval replay <span_id>
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from treval.db import SpanStore


class ReplaySession:
    """Re-executes a saved span, with the option to modify parameters."""

    def __init__(self, span_id: int):
        store = SpanStore()
        self.original = store.get(span_id)
        if not self.original:
            raise ValueError(f"Span #{span_id} no encontrado")

        self.span_id = span_id
        self.children = store.get_children(span_id)
        self.modified_input = self.original.get("input", "")
        self.modified_model = None
        self.modified_temperature = None
        self.result = None
        self.duration_ms = 0.0

    @property
    def span_type(self) -> str:
        return self.original.get("type", "UNKNOWN")

    @property
    def span_name(self) -> str:
        return self.original.get("name", "unknown")

    @property
    def is_replayable(self) -> bool:
        """A span is replayable if it has input that we can modify."""
        return bool(self.original.get("input"))

    def set_input(self, new_input: str) -> None:
        """Modifies the input before re-executing."""
        self.modified_input = new_input

    def set_model(self, model: str) -> None:
        """Changes the model for re-execution."""
        self.modified_model = model

    def set_temperature(self, temp: float) -> None:
        """Changes the temperature."""
        self.modified_temperature = temp

    def replay(self, api_key: str | None = None) -> dict:
        """Re-executes the span.

        For LLM/OPERATION spans with text input, calls the LLM
        with the modified input.

        Returns:
            Dict with result, duration, comparison with original.
        """
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return {"error": "OPENROUTER_API_KEY not set"}

        input_text = self.modified_input
        model = self.modified_model or self._extract_model()
        temperature = self.modified_temperature or 0.1

        start = time.perf_counter()
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": "https://treval.dev",
                    "X-Title": "treval-replay",
                },
            )

            # Extraer mensajes del input original
            messages = self._parse_messages(input_text)
            if not messages:
                # Si no podemos parsear como mensajes, usarlo como user input
                messages = [{"role": "user", "content": input_text}]

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=500,
            )
            output = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
        except Exception as e:
            output = f"Error: {e}"
            usage = {}

        self.duration_ms = (time.perf_counter() - start) * 1000
        self.result = output

        return {
            "output": output,
            "model": model,
            "temperature": temperature,
            "duration_ms": self.duration_ms,
            "usage": usage,
            "original_output": self.original.get("output", ""),
            "original_duration_ms": self.original.get("duration_ms", 0),
        }

    def _extract_model(self) -> str:
        """Attempts to extract the model from the original span."""
        name = self.original.get("name", "")
        # LLM spans have format "llm.<model>"
        if name.startswith("llm."):
            return name[4:]
        return "deepseek/deepseek-v4-flash"

    def _parse_messages(self, input_text: str) -> list[dict] | None:
        """Attempts to parse the input as a list of messages."""
        try:
            # The input can be repr() of a list of dicts
            if input_text.startswith("[") and "role" in input_text:
                import ast
                parsed = ast.literal_eval(input_text)
                if isinstance(parsed, list) and all(isinstance(m, dict) for m in parsed):
                    return parsed
        except (ValueError, SyntaxError):
            pass
        return None

    def summary(self) -> str:
        """Returns a summary of the span and comparison."""
        lines = [
            f"Span #{self.span_id} — {self.span_type}: {self.span_name}",
            f"{'─' * 50}",
        ]

        if self.result:
            original = self.original.get("output", "")
            lines.extend([
                f"📊 Comparison:",
                f"   Original:  {original[:200]}",
                f"   Replay:    {self.result[:200]}",
                f"",
                f"⏱ Original: {self.original.get('duration_ms', 0):.1f}ms",
                f"⏱ Replay:   {self.duration_ms:.1f}ms",
            ])
        else:
            lines.append("(not yet re-executed)")

        return "\n".join(lines)


def interactive_replay(span_id: int, api_key: str | None = None) -> None:
    """Interactive mode: modify parameters and re-execute."""
    try:
        session = ReplaySession(span_id)
    except ValueError as e:
        print(f"❌ {e}")
        return

    print(f"\n{'='*60}")
    print(f"🔄  Interactive Replay — #{span_id}")
    print(f"{'='*60}")
    print(f"Tipo:   {session.span_type}")
    print(f"Name:   {session.span_name}")
    print(f"Input:  {session.original.get('input', '')[:150]}...")
    print(f"Output: {session.original.get('output', '')[:150]}...")
    print()

    if not session.is_replayable:
        print("❌ This span has no input to modify.")
        return

    # Batch mode: accept changes via args or use defaults
    # For now, execute with default values
    print("🔧 Re-executing with the same parameters...")
    result = session.replay(api_key)

    if "error" in result:
        print(f"❌ {result['error']}")
        return

    print(f"✅ Done in {result['duration_ms']:.1f}ms")
    print()
    print(session.summary())