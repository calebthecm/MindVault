"""
mindvault/council.py — Multi-agent reasoning council.

The council is five internal voices with distinct personalities and values.
Each voice reads the same retrieved memories but interprets them differently.

In most modes the council runs sequentially and then a Synthesizer
produces the final answer from all their inputs.

Council members:
  The Analyst      — evidence-first, skeptical, quantitative
  The Visionary    — big-picture, creative, optimistic
  The Pragmatist   — what's actually doable right now
  The Devil        — challenges assumptions, finds the flaw in every plan
  The Historian    — what do past memories say; patterns over time

Mode behavior:
  PLAN    — Analyst + Pragmatist draft the plan; Devil stress-tests it
  DECIDE  — All five vote (agree / disagree / abstain) and give a reason
  DEBATE  — Visionary argues FOR; Devil argues AGAINST; Analyst moderates
  REFLECT — Historian leads; others add nuance; Synthesizer unifies
  EXPLORE — Visionary leads with surprising connections; others react
  CHAT    — Synthesizer only (council is skipped for speed)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from mindvault.modes import Mode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CouncilMember:
    name: str
    role: str
    personality: str   # One sentence for the system prompt
    vote_symbol: str   # Used in DECIDE output


COUNCIL: list[CouncilMember] = [
    CouncilMember(
        name="The Analyst",
        role="analyst",
        personality=(
            "You reason from evidence. You cite specifics from memory, "
            "quantify uncertainty, and flag anything that looks like a gap or contradiction."
        ),
        vote_symbol="📊",
    ),
    CouncilMember(
        name="The Visionary",
        role="visionary",
        personality=(
            "You see the big picture and long-term potential. "
            "You connect distant ideas and propose bold interpretations. "
            "You are optimistic but grounded in what the memories actually show."
        ),
        vote_symbol="🚀",
    ),
    CouncilMember(
        name="The Pragmatist",
        role="pragmatist",
        personality=(
            "You care about what can be done right now with current resources. "
            "You ask: what is the simplest next action? You cut through abstraction."
        ),
        vote_symbol="🔧",
    ),
    CouncilMember(
        name="The Devil",
        role="devil",
        personality=(
            "You challenge every assumption. You find the flaw, the missing piece, "
            "the overlooked risk. You are not negative — you are rigorous. "
            "You make the final answer stronger."
        ),
        vote_symbol="😈",
    ),
    CouncilMember(
        name="The Historian",
        role="historian",
        personality=(
            "You look for patterns across time. You ask: have we been here before? "
            "What do past conversations and notes reveal about how this situation tends to evolve? "
            "You are the memory of the memory."
        ),
        vote_symbol="📜",
    ),
]


def _build_context(chunks: list[dict], max_chars: int = 3000) -> str:
    parts = []
    total = 0
    for chunk in chunks:
        title = chunk.get("title", "?")
        date = chunk.get("created_at", "")[:10]
        text = chunk.get("text", "").strip()
        layer = chunk.get("layer", "raw")
        entry = f"[{title} — {date}]\n{text}"
        if total + len(entry) > max_chars:
            parts.append("[...additional memories truncated...]")
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n---\n\n".join(parts)


def _call(
    member: CouncilMember,
    query: str,
    context: str,
    role_instruction: str,
    model: str,
    base_url: str,
    timeout: float = 180.0,
) -> Optional[str]:
    from src.llm import _call_ollama

    prompt = f"""You are {member.name} on a reasoning council.
{member.personality}

The user's brain memories relevant to this query:

{context}

Query: {query}

Your role right now: {role_instruction}

Keep your response under 120 words. Be direct and specific. No filler phrases."""

    return _call_ollama(prompt, model=model, base_url=base_url, timeout=timeout)


def _synthesize(
    query: str,
    voices: list[tuple[str, str]],   # [(member_name, response), ...]
    model: str,
    base_url: str,
    final_instruction: str = "Write a clear, concise final answer based on all the above.",
) -> Optional[str]:
    from src.llm import _call_ollama

    council_text = "\n\n".join(
        f"{name}:\n{text}" for name, text in voices if text
    )

    prompt = f"""You are the Synthesizer. You have just heard from the council on this query.

Query: {query}

Council voices:
{council_text}

{final_instruction}
Keep it under 200 words. Second person ("You..."). No bullet soup — use prose."""

    return _call_ollama(prompt, model=model, base_url=base_url, timeout=120.0)


# ─── Mode-specific reasoning loops ────────────────────────────────────────────

def run_chat(
    query: str,
    chunks: list[dict],
    model: str,
    base_url: str,
    history: Optional[list[dict]] = None,
    on_token: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """Standard CHAT — council skipped, direct synthesis."""
    from src.llm import chat_with_brain
    return chat_with_brain(
        query=query,
        context_chunks=chunks,
        model=model,
        base_url=base_url,
        conversation_history=history,
        on_token=on_token,
    )


def run_plan(
    query: str,
    chunks: list[dict],
    model: str,
    base_url: str,
) -> str:
    """PLAN mode — Analyst + Pragmatist draft; Devil stress-tests; Synthesizer finalises."""
    context = _build_context(chunks)
    voices: list[tuple[str, str]] = []

    analyst = COUNCIL[0]   # Analyst
    pragmatist = COUNCIL[2]  # Pragmatist
    devil = COUNCIL[3]     # Devil

    for member, instruction in [
        (analyst, "Break this task into logical phases. What are the key steps and decision points?"),
        (pragmatist, "What are the concrete, immediate next actions? What is the simplest path?"),
        (devil, "What could go wrong with this plan? What is missing or unrealistic?"),
    ]:
        result = _call(member, query, context, instruction, model, base_url)
        if result:
            voices.append((member.name, result))

    return _synthesize(
        query=query,
        voices=voices,
        model=model,
        base_url=base_url,
        final_instruction=(
            "Write a structured plan. Use numbered steps. Be specific. "
            "Address the main risks raised. Under 250 words."
        ),
    ) or "[Plan generation failed — check if Ollama is running]"


def run_decide(
    query: str,
    chunks: list[dict],
    model: str,
    base_url: str,
) -> str:
    """DECIDE mode — all five vote and give a one-sentence reason; tally shown."""
    context = _build_context(chunks)
    votes: list[tuple[str, str, str]] = []  # (symbol, name, vote+reason)

    for member in COUNCIL:
        result = _call(
            member, query, context,
            role_instruction=(
                "Vote: AGREE, DISAGREE, or ABSTAIN on whether this is a good idea/decision. "
                "Give one sentence explaining why."
            ),
            model=model,
            base_url=base_url,
            timeout=120.0,
        )
        if result:
            votes.append((member.vote_symbol, member.name, result.strip()))

    if not votes:
        return "[Council unreachable — check if Ollama is running]"

    # Tally
    agree = sum(1 for _, _, v in votes if "AGREE" in v.upper() and "DISAGREE" not in v.upper())
    disagree = sum(1 for _, _, v in votes if "DISAGREE" in v.upper())
    abstain = len(votes) - agree - disagree

    lines = ["Council vote:\n"]
    for symbol, name, text in votes:
        lines.append(f"  {symbol} {name}: {text}")

    verdict = "AGREED" if agree > disagree else ("REJECTED" if disagree > agree else "SPLIT")
    lines.append(f"\nVerdict: {verdict}  ({agree} agree · {disagree} disagree · {abstain} abstain)")
    return "\n".join(lines)


def run_debate(
    query: str,
    chunks: list[dict],
    model: str,
    base_url: str,
) -> str:
    """DEBATE mode — Visionary FOR, Devil AGAINST, Analyst moderates."""
    context = _build_context(chunks)
    visionary = COUNCIL[1]
    devil = COUNCIL[3]
    analyst = COUNCIL[0]

    pro = _call(
        visionary, query, context,
        role_instruction="Argue FOR this idea/decision. Make the strongest possible case using the memories.",
        model=model, base_url=base_url,
    )
    con = _call(
        devil, query, context,
        role_instruction="Argue AGAINST this idea/decision. Make the strongest possible case against it.",
        model=model, base_url=base_url,
    )
    moderate = _call(
        analyst, query, context,
        role_instruction=(
            "You have heard both sides. Moderate: what does the evidence actually support? "
            "Be impartial."
        ),
        model=model, base_url=base_url,
    )

    parts = []
    if pro:
        parts.append(f"FOR ({visionary.name}):\n{pro}")
    if con:
        parts.append(f"AGAINST ({devil.name}):\n{con}")
    if moderate:
        parts.append(f"VERDICT ({analyst.name}):\n{moderate}")

    return "\n\n".join(parts) if parts else "[Debate failed — check if Ollama is running]"


def run_reflect(
    query: str,
    chunks: list[dict],
    model: str,
    base_url: str,
) -> str:
    """REFLECT mode — Historian leads, others add, Synthesizer unifies."""
    context = _build_context(chunks)
    historian = COUNCIL[4]
    visionary = COUNCIL[1]
    analyst = COUNCIL[0]
    voices: list[tuple[str, str]] = []

    for member, instruction in [
        (historian, "What patterns across time do these memories reveal? What has stayed consistent?"),
        (visionary, "What does this collection of memories suggest about where things are heading?"),
        (analyst, "What is certain vs uncertain based on what is actually written in the memories?"),
    ]:
        result = _call(member, query, context, instruction, model, base_url)
        if result:
            voices.append((member.name, result))

    return _synthesize(
        query=query,
        voices=voices,
        model=model,
        base_url=base_url,
        final_instruction=(
            "Write a reflective synthesis of what your brain knows about this topic. "
            "Acknowledge gaps. Be honest about uncertainty. Under 250 words."
        ),
    ) or "[Reflection failed — check if Ollama is running]"


def run_explore(
    query: str,
    chunks: list[dict],
    model: str,
    base_url: str,
) -> str:
    """EXPLORE mode — Visionary leads on connections; others react to surprises."""
    context = _build_context(chunks)
    visionary = COUNCIL[1]
    historian = COUNCIL[4]
    pragmatist = COUNCIL[2]
    voices: list[tuple[str, str]] = []

    for member, instruction in [
        (visionary, "What surprising or unexpected connections do these memories reveal?"),
        (historian, "Have these topics intersected before in the memory history? When and how?"),
        (pragmatist, "What actionable insight emerges from these connections right now?"),
    ]:
        result = _call(member, query, context, instruction, model, base_url)
        if result:
            voices.append((member.name, result))

    return _synthesize(
        query=query,
        voices=voices,
        model=model,
        base_url=base_url,
        final_instruction=(
            "Write an exploratory answer that highlights surprising connections "
            "and what they reveal. Under 250 words."
        ),
    ) or "[Exploration failed — check if Ollama is running]"


# ─── Dispatch ─────────────────────────────────────────────────────────────────

def run_council(
    mode: Mode,
    query: str,
    chunks: list[dict],
    model: str,
    base_url: str,
    history: Optional[list[dict]] = None,
    on_token: Optional[Callable[[str], None]] = None,
) -> str:
    """Dispatch to the right reasoning loop for the current mode."""
    if mode == Mode.CHAT:
        return run_chat(query, chunks, model, base_url, history, on_token=on_token) or ""
    elif mode == Mode.PLAN:
        return run_plan(query, chunks, model, base_url)
    elif mode == Mode.DECIDE:
        return run_decide(query, chunks, model, base_url)
    elif mode == Mode.DEBATE:
        return run_debate(query, chunks, model, base_url)
    elif mode == Mode.REFLECT:
        return run_reflect(query, chunks, model, base_url)
    elif mode == Mode.EXPLORE:
        return run_explore(query, chunks, model, base_url)
    return ""
