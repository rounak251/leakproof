"""
Claude-written narrative for an audit report.

Takes the JSON report produced by engine.audit() (statistics only, never raw
rows beyond the short suspect lists) and asks Claude for a structured fix plan.
Works without an API key: returns None so the app degrades to the engine's own
headline and fix snippets.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

MODEL = os.environ.get("LEAKPROOF_CLAUDE_MODEL", "claude-opus-5-5")

SYSTEM = """You are Leakproof, a senior ML reviewer who has diagnosed data leakage, label noise and
broken validation on hundreds of real projects and competitions. You are given the JSON output of an
automated dataset audit. Write the narrative a strong colleague would leave in a code review:
direct, specific, no hedging, no filler, no praise. Refer to columns by their exact names.
Numbers must come from the JSON; never invent statistics. If the audit found nothing serious, say so
plainly and keep it short. Code snippets must be runnable pandas / scikit-learn and reference the
real column names. Order steps by impact. Keep the whole thing readable in two minutes."""

SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "description": "One sentence verdict, max 25 words."},
        "summary": {"type": "string", "description": "2-4 short paragraphs in plain English. What is wrong, why it matters, what the honest performance looks like."},
        "root_cause": {"type": "string", "description": "The single most likely story of how this dataset got this way (one paragraph), or 'none' if clean."},
        "steps": {
            "type": "array",
            "maxItems": 7,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "why": {"type": "string"},
                    "code": {"type": "string", "description": "Runnable pandas/sklearn snippet, or empty string."},
                    "expected_effect": {"type": "string", "description": "What the user should see after doing it."},
                },
                "required": ["title", "why", "code", "expected_effect"],
                "additionalProperties": False,
            },
        },
        "validation_recipe": {"type": "string", "description": "The exact cross-validation setup to use from now on, as code."},
        "what_to_tell_stakeholders": {"type": "string", "description": "Two sentences a manager can read."},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["headline", "summary", "root_cause", "steps", "validation_recipe", "what_to_tell_stakeholders", "confidence"],
    "additionalProperties": False,
}


def _trim(report: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the report small: drop long tables, keep every finding."""
    r = json.loads(json.dumps(report))
    sec = r.get("sections", {})
    if "schema" in sec:
        sec["schema"]["columns"] = [{k: v for k, v in c.items() if k != "example"} for c in sec["schema"]["columns"][:60]]
    if sec.get("label_noise") and sec["label_noise"].get("suspects"):
        sec["label_noise"]["suspects"] = [{k: v for k, v in x.items() if k != "id"} for x in sec["label_noise"]["suspects"]]
    if sec.get("target_leakage"):
        sec["target_leakage"]["features"] = sec["target_leakage"]["features"][:15]
    if sec.get("label_noise"):
        sec["label_noise"]["suspects"] = sec["label_noise"]["suspects"][:8]
    if sec.get("shift"):
        sec["shift"]["features"] = sec["shift"]["features"][:12]
    return r


def write_narrative(report: Dict[str, Any], api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    client = anthropic.Anthropic(api_key=key)
    payload = json.dumps(_trim(report), ensure_ascii=False)
    user = ("Here is the audit JSON. Write the narrative and fix plan.\n\n" + payload)
    kwargs: Dict[str, Any] = dict(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    try:
        try:
            # Opt into Anthropic's server-side refusal fallback so a classifier false positive
            # on, say, a medical dataset still gets a narrative.
            resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        except anthropic.BadRequestError:
            resp = client.messages.create(**kwargs)
    except anthropic.AuthenticationError:
        return {"error": "Invalid Anthropic API key."}
    except anthropic.RateLimitError:
        return {"error": "Narrative temporarily unavailable (rate limited). The engine findings below are complete."}
    except anthropic.APIStatusError as e:
        return {"error": f"Narrative unavailable (API error {e.status_code})."}
    except anthropic.APIConnectionError:
        return {"error": "Narrative unavailable (network error)."}

    if resp.stop_reason == "refusal":
        return {"error": "Narrative declined by the model's safety system; engine findings are complete."}
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        return {"error": "Empty narrative response."}
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Narrative could not be parsed."}
    out["model"] = resp.model
    out["usage"] = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return out
