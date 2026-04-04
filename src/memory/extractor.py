"""
Entity extractor — pulls structured facts from conversation turns.

Extracts: person, project, decision, goal, fact, topic

Called per turn-pair during chat (streaming, adds minimal latency).
Called over full session at end for deduplication pass.
"""

import logging

from src.llm import _call_ollama, _extract_json

logger = logging.getLogger(__name__)


def parse_entity_response(text: str) -> list[dict]:
    """
    Parse LLM entity extraction response into a clean list.
    Handles JSON wrapped in prose, missing fields, and invalid responses.
    """
    parsed = _extract_json(text)
    if not isinstance(parsed, list):
        return []

    entities = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if not item.get("name") or not item.get("type"):
            continue
        entities.append({
            "type": str(item["type"])[:50],
            "name": str(item["name"])[:200],
            "value": str(item.get("value", ""))[:500],
        })
    return entities


def extract_entities_from_turn(
    user_turn: str,
    assistant_turn: str,
    model: str,
    base_url: str,
) -> list[dict]:
    """
    Extract entities from one user+assistant exchange.
    Returns [] if extraction fails or nothing notable found.
    Truncates inputs to keep the prompt fast on local models.
    """
    prompt = f"""Extract named entities from this exchange. Return a JSON array only. Return [] if nothing notable.

Entity types: person | project | decision | goal | fact | topic

Format: [{{"type": "...", "name": "...", "value": "..."}}]

Rules:
- "decision" = something concrete the user decided ("decided to use Stripe")
- "goal" = something the user wants to achieve
- "fact" = a specific piece of stated information
- Only extract clear, specific items. Skip vague statements.

USER: {user_turn[:600]}
ASSISTANT: {assistant_turn[:300]}

JSON array:"""

    result = _call_ollama(prompt, model=model, base_url=base_url, timeout=25.0)
    if not result:
        return []
    return parse_entity_response(result)


def deduplicate_entities(entities: list[dict]) -> list[dict]:
    """
    Remove exact duplicates: same type + same name (case-insensitive).
    First occurrence wins.
    """
    seen: set[tuple] = set()
    result = []
    for entity in entities:
        key = (entity["type"], entity["name"].lower().strip())
        if key not in seen:
            seen.add(key)
            result.append(entity)
    return result
